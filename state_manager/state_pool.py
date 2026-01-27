### State Pool Manager for RWKV-7 Inference
### Manages three-level caching system for model states across GPU, CPU and disk
import torch
import sqlite3
import io
import pickle
import threading
import time
import queue
from collections import OrderedDict
from typing import List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

L1_CAPACITY = 16   # VRAM (Hot)
L2_CAPACITY = 32 # RAM (Warm)
DB_PATH = "rwkv_sessions.db" # infinite cold state pool HaHa!

class StateCacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StateCacheManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.l1_cache: OrderedDict[str, List[torch.Tensor]] = OrderedDict()
        
        self.l2_cache: OrderedDict[str, List[torch.Tensor]] = OrderedDict()
        
        self.cache_lock = threading.RLock()
        
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db_cursor = self.db_conn.cursor()
        self.db_lock = threading.Lock()
        
        self._init_db()
        
        self.io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_writer")
        
        self._initialized = True
        print(f"[StatePool] Initialized. L1: {L1_CAPACITY}, L2: {L2_CAPACITY}, DB: {DB_PATH}")

    def _init_db(self):
        """初始化数据库表"""
        with self.db_lock:
            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_blob BLOB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db_conn.commit()

    def _serialize(self, state: List[torch.Tensor]) -> bytes:
        """将 State (Tensor列表) 序列化为二进制"""
        buffer = io.BytesIO()
        torch.save(state, buffer)
        return buffer.getvalue()

    def _deserialize(self, blob: bytes) -> List[torch.Tensor]:
        """将二进制反序列化为 Tensor 列表"""
        buffer = io.BytesIO(blob)
        # map_location='cpu' 避免反序列化时直接占用显存，由上层逻辑决定何时搬运
        return torch.load(buffer, map_location='cpu', weights_only=True)

    def _persist_task(self, session_id: str, state_cpu: List[torch.Tensor]):
        """异步任务：序列化并写入数据库"""
        try:
            blob = self._serialize(state_cpu)
            with self.db_lock:
                self.db_cursor.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)",
                    (session_id, blob, time.time())
                )
                self.db_conn.commit()
            # print(f"[StatePool] Persisted session {session_id} to L3 (Disk).")
            
            # 显式删除引用协助 GC
            del state_cpu
            del blob
        except Exception as e:
            print(f"[StatePool] Error persisting session {session_id}: {e}")

    def put_state(self, session_id: str, state: List[torch.Tensor]):
        """
        存入状态。
        流程：
        1. 存入 L1 (GPU)。
        2. 如果 L1 满 -> 移出最久未使用的到 L2 (CPU)。
        3. 如果 L2 满 -> 移出最久未使用的到 L3 (Disk, Async)。
        """
        if session_id is None:
            return

        with self.cache_lock:
            if session_id in self.l1_cache:
                del self.l1_cache[session_id]
            if session_id in self.l2_cache:
                del self.l2_cache[session_id]
            
            self.l1_cache[session_id] = state
            
            if len(self.l1_cache) > L1_CAPACITY:
                # popitem(last=False) 弹出最早插入的元素 (FIFO/LRU Oldest)
                evicted_id, evicted_state_gpu = self.l1_cache.popitem(last=False)
                
                evicted_state_cpu = [t.to('cpu', non_blocking=True) for t in evicted_state_gpu]
                
                self.l2_cache[evicted_id] = evicted_state_cpu
                
                if len(self.l2_cache) > L2_CAPACITY:
                    l2_evicted_id, l2_evicted_state_cpu = self.l2_cache.popitem(last=False)
                    
                    self.io_executor.submit(self._persist_task, l2_evicted_id, l2_evicted_state_cpu)

    def get_state(self, session_id: str) -> Optional[List[torch.Tensor]]:

        if session_id is None:
            return None

        with self.cache_lock:
            # Case 1: L1 Hit (VRAM)
            if session_id in self.l1_cache:
                self.l1_cache.move_to_end(session_id) # 标记为最近使用
                return self.l1_cache[session_id]
            
            # Case 2: L2 Hit (RAM)
            if session_id in self.l2_cache:
                state_cpu = self.l2_cache.pop(session_id)
                state_gpu = [t.to('cuda', non_blocking=True) for t in state_cpu]
                
                self.put_state(session_id, state_gpu)
                return state_gpu

        blob = None
        with self.db_lock:
            self.db_cursor.execute("SELECT state_blob FROM sessions WHERE session_id = ?", (session_id,))
            row = self.db_cursor.fetchone()
            if row:
                blob = row[0]
        
        if blob:
            try:
                state_cpu = self._deserialize(blob)
                state_gpu = [t.to('cuda') for t in state_cpu]
                self.put_state(session_id, state_gpu)
                return state_gpu
            except Exception as e:
                print(f"[StatePool] Failed to deserialize session {session_id}: {e}")
                return None
        
        return None

    def close_session(self, session_id: str):

        state_to_save = None
        
        with self.cache_lock:
            if session_id in self.l1_cache:
                state_to_save = [t.to('cpu') for t in self.l1_cache.pop(session_id)]
            elif session_id in self.l2_cache:
                state_to_save = self.l2_cache.pop(session_id)
        
        if state_to_save:
            self._persist_task(session_id, state_to_save)
        
        print(f"[StatePool] Session {session_id} closed and persisted.")

    def flush_all(self):

        print("[StatePool] Flushing all states to disk...")
        
        self.io_executor.shutdown(wait=True)
        
        items_to_save = []
        with self.cache_lock:
            while self.l1_cache:
                sid, state = self.l1_cache.popitem()
                items_to_save.append((sid, [t.to('cpu') for t in state]))
            
            while self.l2_cache:
                sid, state = self.l2_cache.popitem()
                items_to_save.append((sid, state))
        
        with self.db_lock:
            try:
                self.db_conn.execute("BEGIN TRANSACTION")
                for sid, state in items_to_save:
                    blob = self._serialize(state)
                    self.db_conn.execute(
                        "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)",
                        (sid, blob, time.time())
                    )
                self.db_conn.commit()
                print(f"[StatePool] Successfully saved {len(items_to_save)} sessions.")
            except Exception as e:
                print(f"[StatePool] Error during flush: {e}")
                self.db_conn.rollback()
            finally:
                self.db_conn.close()

    def list_states_in_db(self) -> List[Tuple[str, float]]:

        with self.db_lock:
            self.db_cursor.execute("SELECT session_id, last_updated FROM sessions ORDER BY last_updated DESC")
            results = self.db_cursor.fetchall()
            return [(row[0], row[1]) for row in results]

    def list_all_states(self) -> dict:

        with self.cache_lock:
            l1_states = list(self.l1_cache.keys())
            l2_states = list(self.l2_cache.keys())

        db_states = self.list_states_in_db()
        db_states_keys = [state[0] for state in db_states]

        return {
            "l1_cache": l1_states,
            "l2_cache": l2_states,
            "database": db_states_keys,
            "total_count": len(l1_states) + len(l2_states) + len(db_states_keys)
        }

    def print_all_states_status(self):

        all_states = self.list_all_states()

        print(f"[StatePool] All States Status - Total {all_states['total_count']} sessions:")
        print("=" * 80)

        print(f"L1 Cache (VRAM) - Count: {len(all_states['l1_cache'])}")
        print("-" * 40)
        for session_id in all_states['l1_cache']:
            print(f"  {session_id}")

        print(f"\nL2 Cache (RAM) - Count: {len(all_states['l2_cache'])}")
        print("-" * 40)
        for session_id in all_states['l2_cache']:
            print(f"  {session_id}")

        print(f"\nDatabase (Disk) - Count: {len(all_states['database'])}")
        print("-" * 40)
        for session_id in all_states['database']:
            print(f"  {session_id}")

        if all_states['total_count'] == 0:
            print("No sessions found in any cache level.")
        print("=" * 80)

    def delete_state_from_any_level(self, session_id: str) -> bool:

        deleted_from_cache = False

        with self.cache_lock:
            # 从L1缓存删除
            if session_id in self.l1_cache:
                del self.l1_cache[session_id]
                deleted_from_cache = True
                print(f"[StatePool] Session {session_id} removed from L1 cache (VRAM).")

            # 从L2缓存删除
            if session_id in self.l2_cache:
                del self.l2_cache[session_id]
                deleted_from_cache = True
                print(f"[StatePool] Session {session_id} removed from L2 cache (RAM).")

        # 从数据库删除
        with self.db_lock:
            try:
                self.db_cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                self.db_conn.commit()

                affected_rows = self.db_cursor.rowcount
                if affected_rows > 0:
                    print(f"[StatePool] Session {session_id} removed from database (Disk).")
                    return True
                elif deleted_from_cache:
                    # 即使数据库中没有找到，但在缓存中已删除
                    return True
                else:
                    print(f"[StatePool] Session {session_id} not found in any cache level.")
                    return False
            except Exception as e:
                print(f"[StatePool] Error deleting session {session_id} from database: {e}")
                return False
            
def show_all_states_status():
    manager = get_state_manager()
    manager.print_all_states_status()

def remove_session_from_any_level(session_id: str) -> bool:
    manager = get_state_manager()
    return manager.delete_state_from_any_level(session_id)
def get_state_manager() -> StateCacheManager:
    return StateCacheManager()

def shutdown_state_manager():
    manager = get_state_manager()
    manager.flush_all()