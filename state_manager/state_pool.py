### State Pool Manager for RWKV-7 Inference
### Manages three-level session caching plus RAM+disk prefix-state cache
import hashlib
import io
import sqlite3
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

L1_CAPACITY = 16  # VRAM (Hot)
L2_CAPACITY = 64  # RAM (Warm)
DB_PATH = "rwkv_sessions.db"  # infinite cold state pool HaHa!

PREFIX_CACHE_BUCKETS = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
PREFIX_CACHE_BUCKET_CAPACITY = 16
PREFIX_HASH_COLUMNS = tuple(f"prefix_hash_{bucket}" for bucket in PREFIX_CACHE_BUCKETS)
SERDE_MAGIC = b"Z1"


def _serialize_token_ids(tokens: List[int] | Tuple[int, ...]) -> str:
    return " ".join(str(token) for token in tokens)


def _deserialize_token_ids(serialized: str) -> Tuple[int, ...]:
    if not serialized:
        return ()
    return tuple(int(token) for token in serialized.split(" "))


def _hash_token_ids(tokens: List[int] | Tuple[int, ...]) -> str:
    payload = _serialize_token_ids(tokens).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _build_prefix_hashes(tokens: List[int] | Tuple[int, ...]) -> Dict[int, Optional[str]]:
    prefix_hashes: Dict[int, Optional[str]] = {}
    token_count = len(tokens)
    for bucket in PREFIX_CACHE_BUCKETS:
        prefix_hashes[bucket] = _hash_token_ids(tokens[:bucket]) if token_count >= bucket else None
    return prefix_hashes


class _CompressedTrieNode:
    def __init__(self, label: Tuple[int, ...] = ()):
        self.label: Tuple[int, ...] = label
        self.children: Dict[int, "_CompressedTrieNode"] = {}
        self.terminal_key: Optional[str] = None


class _CompressedTrie:
    def __init__(self):
        self.root = _CompressedTrieNode()

    @staticmethod
    def _common_prefix_len(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return idx

    def clear(self):
        self.root = _CompressedTrieNode()

    def insert(self, tokens: Tuple[int, ...], terminal_key: str):
        self._insert(self.root, tokens, terminal_key)

    def _insert(self, node: _CompressedTrieNode, tokens: Tuple[int, ...], terminal_key: str):
        if not tokens:
            node.terminal_key = terminal_key
            return

        first = tokens[0]
        child = node.children.get(first)
        if child is None:
            new_child = _CompressedTrieNode(tokens)
            new_child.terminal_key = terminal_key
            node.children[first] = new_child
            return

        common = self._common_prefix_len(tokens, child.label)
        if common == len(child.label):
            self._insert(child, tokens[common:], terminal_key)
            return

        split_label = child.label[:common]
        split_node = _CompressedTrieNode(split_label)
        node.children[first] = split_node

        child.label = child.label[common:]
        split_node.children[child.label[0]] = child

        remaining = tokens[common:]
        if remaining:
            new_child = _CompressedTrieNode(remaining)
            new_child.terminal_key = terminal_key
            split_node.children[remaining[0]] = new_child
        else:
            split_node.terminal_key = terminal_key

    def longest_prefix(self, tokens: List[int] | Tuple[int, ...]) -> Tuple[Optional[str], int]:
        node = self.root
        idx = 0
        best_key = node.terminal_key
        best_len = 0 if best_key is not None else 0
        tokens_tuple = tuple(tokens)

        while idx < len(tokens_tuple):
            child = node.children.get(tokens_tuple[idx])
            if child is None:
                break

            label = child.label
            if tokens_tuple[idx : idx + len(label)] != label:
                break

            idx += len(label)
            node = child
            if node.terminal_key is not None:
                best_key = node.terminal_key
                best_len = idx

        return best_key, best_len


@dataclass
class PrefixCacheEntry:
    state_id: str
    bucket_len: int
    token_count: int
    prefix_tokens: Tuple[int, ...]
    prefix_hashes: Dict[int, Optional[str]]
    state_cpu: List[torch.Tensor]
    logits_cpu: Optional[torch.Tensor]
    last_updated: float


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
        
        self.prefix_l2_cache: Dict[int, OrderedDict[str, PrefixCacheEntry]] = {
            bucket: OrderedDict() for bucket in PREFIX_CACHE_BUCKETS
        }
        self.prefix_entry_index: Dict[str, PrefixCacheEntry] = {}
        self.prefix_trie = _CompressedTrie()
        self._prefix_trie_dirty = False
        
        self.cache_lock = threading.RLock()
        
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db_cursor = self.db_conn.cursor()
        self.db_lock = threading.Lock()
        
        self._init_db()
        
        self.io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_writer")
        
        self._initialized = True
        print(
            f"[StatePool] Initialized. L1: {L1_CAPACITY}, L2: {L2_CAPACITY}, "
            f"Prefix L2: {len(PREFIX_CACHE_BUCKETS)}x{PREFIX_CACHE_BUCKET_CAPACITY}, DB: {DB_PATH}"
        )

    def _init_db(self):
        """初始化数据库表"""
        with self.db_lock:
            self.db_cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    state_blob BLOB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            prefix_hash_sql = ", ".join(f"{column} TEXT" for column in PREFIX_HASH_COLUMNS)
            self.db_cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS prefix_cache (
                    state_id TEXT PRIMARY KEY,
                    bucket_len INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    {prefix_hash_sql},
                    state_blob BLOB NOT NULL,
                    logits_blob BLOB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            for bucket in PREFIX_CACHE_BUCKETS:
                self.db_cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_prefix_cache_{bucket}
                    ON prefix_cache (bucket_len, prefix_hash_{bucket}, last_updated)
                    """
                )
            self.db_conn.commit()

    def _serialize(self, state) -> bytes:
        buffer = io.BytesIO()
        torch.save(state, buffer)
        payload = buffer.getvalue()
        compressed = zlib.compress(payload, level=1)
        if len(compressed) < len(payload):
            return SERDE_MAGIC + compressed
        return payload

    def _deserialize(self, blob: bytes):
        if blob.startswith(SERDE_MAGIC):
            blob = zlib.decompress(blob[len(SERDE_MAGIC) :])
        buffer = io.BytesIO(blob)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    def _clone_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        """深拷贝状态，避免多线程共享导致污染"""
        return [t.clone() for t in state]

    def _clone_to_cpu_state(self, state: List[torch.Tensor]) -> List[torch.Tensor]:
        return [t.detach().to("cpu").clone() for t in state]

    def _clone_to_device_state(self, state: List[torch.Tensor], device: str) -> List[torch.Tensor]:
        return [t.detach().to(device, non_blocking=device == "cuda").clone() for t in state]

    def _clone_optional_tensor(self, tensor: Optional[torch.Tensor], device: str) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.detach().to(device, non_blocking=device == "cuda").clone()

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

    def _persist_prefix_task(self, entry: PrefixCacheEntry):
        try:
            state_blob = self._serialize(entry.state_cpu)
            logits_blob = self._serialize(entry.logits_cpu) if entry.logits_cpu is not None else None
            with self.db_lock:
                row = [
                    entry.state_id,
                    entry.bucket_len,
                    entry.token_count,
                ]
                row.extend(entry.prefix_hashes.get(bucket) for bucket in PREFIX_CACHE_BUCKETS)
                row.extend([state_blob, logits_blob, entry.last_updated])

                placeholders = ", ".join("?" for _ in row)
                columns = ", ".join(
                    ["state_id", "bucket_len", "token_count", *PREFIX_HASH_COLUMNS, "state_blob", "logits_blob", "last_updated"]
                )
                self.db_cursor.execute(
                    f"INSERT OR REPLACE INTO prefix_cache ({columns}) VALUES ({placeholders})",
                    row,
                )
                self.db_conn.commit()
        except Exception as e:
            print(f"[StatePool] Error persisting prefix cache {entry.state_id[:96]}...: {e}")

    def _rebuild_prefix_trie(self):
        trie = _CompressedTrie()
        for entry in self.prefix_entry_index.values():
            trie.insert(entry.prefix_tokens, entry.state_id)
        self.prefix_trie = trie
        self._prefix_trie_dirty = False

    def _ensure_prefix_trie_locked(self):
        if self._prefix_trie_dirty:
            self._rebuild_prefix_trie()

    def _store_prefix_entry_locked(self, entry: PrefixCacheEntry, persist: bool):
        bucket_cache = self.prefix_l2_cache[entry.bucket_len]
        existing_entry = bucket_cache.pop(entry.state_id, None)
        bucket_cache[entry.state_id] = entry
        self.prefix_entry_index[entry.state_id] = entry
        if existing_entry is None:
            self.prefix_trie.insert(entry.prefix_tokens, entry.state_id)

        evicted_entry = None
        if len(bucket_cache) > PREFIX_CACHE_BUCKET_CAPACITY:
            _, evicted_entry = bucket_cache.popitem(last=False)
            self.prefix_entry_index.pop(evicted_entry.state_id, None)
            self._prefix_trie_dirty = True

        if persist:
            self.io_executor.submit(self._persist_prefix_task, entry)
        if evicted_entry is not None:
            self.io_executor.submit(self._persist_prefix_task, evicted_entry)

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
                state = self.l1_cache[session_id]
                token_pos = state[2].item() if len(state) > 2 and hasattr(state[2], "item") else "unknown"
                print(
                    f"[StatePool][SESSION HIT][L1] session_id={session_id} "
                    f"token_pos={token_pos}"
                )
                return self._clone_state(self.l1_cache[session_id])
            
            # Case 2: L2 Hit (RAM)
            if session_id in self.l2_cache:
                state_cpu = self.l2_cache.pop(session_id)
                token_pos = state_cpu[2].item() if len(state_cpu) > 2 and hasattr(state_cpu[2], "item") else "unknown"
                print(
                    f"[StatePool][SESSION HIT][L2] session_id={session_id} "
                    f"token_pos={token_pos} -> promote_to_l1"
                )
                state_gpu = [t.to('cuda', non_blocking=True) for t in state_cpu]
                
                self.put_state(session_id, state_gpu)
                return self._clone_state(state_gpu)

        blob = None
        with self.db_lock:
            self.db_cursor.execute("SELECT state_blob FROM sessions WHERE session_id = ?", (session_id,))
            row = self.db_cursor.fetchone()
            if row:
                blob = row[0]
        
        if blob:
            try:
                state_cpu = self._deserialize(blob)
                token_pos = state_cpu[2].item() if len(state_cpu) > 2 and hasattr(state_cpu[2], "item") else "unknown"
                print(
                    f"[StatePool][SESSION HIT][DISK] session_id={session_id} "
                    f"token_pos={token_pos} -> load_to_l1"
                )
                state_gpu = [t.to('cuda') for t in state_cpu]
                self.put_state(session_id, state_gpu)
                return self._clone_state(state_gpu)
            except Exception as e:
                print(f"[StatePool] Failed to deserialize session {session_id}: {e}")
                return None

        return None

    def put_prefix_state(
        self,
        prefix_tokens: List[int] | Tuple[int, ...],
        state: List[torch.Tensor],
        logits: Optional[torch.Tensor] = None,
    ) -> bool:
        token_tuple = tuple(prefix_tokens)
        bucket_len = len(token_tuple)
        if bucket_len not in PREFIX_CACHE_BUCKETS:
            return False

        entry = PrefixCacheEntry(
            state_id=_serialize_token_ids(token_tuple),
            bucket_len=bucket_len,
            token_count=bucket_len,
            prefix_tokens=token_tuple,
            prefix_hashes=_build_prefix_hashes(token_tuple),
            state_cpu=self._clone_to_cpu_state(state),
            logits_cpu=self._clone_optional_tensor(logits, "cpu"),
            last_updated=time.time(),
        )
        with self.cache_lock:
            self._store_prefix_entry_locked(entry, persist=True)
        return True

    def _load_prefix_entry_from_db_locked(
        self,
        prefix_tokens: List[int] | Tuple[int, ...],
        bucket_len: int,
    ) -> Optional[PrefixCacheEntry]:
        state_id = _serialize_token_ids(prefix_tokens[:bucket_len])
        hash_column = f"prefix_hash_{bucket_len}"
        hash_value = _hash_token_ids(prefix_tokens[:bucket_len])

        with self.db_lock:
            self.db_cursor.execute(
                f"""
                SELECT state_blob, logits_blob, last_updated
                FROM prefix_cache
                WHERE state_id = ? AND bucket_len = ? AND {hash_column} = ?
                LIMIT 1
                """,
                (state_id, bucket_len, hash_value),
            )
            row = self.db_cursor.fetchone()

        if row is None:
            return None

        try:
            state_cpu = self._deserialize(row[0])
            logits_cpu = self._deserialize(row[1]) if row[1] is not None else None
        except Exception as e:
            print(f"[StatePool] Failed to deserialize prefix cache {state_id[:96]}...: {e}")
            return None

        entry = PrefixCacheEntry(
            state_id=state_id,
            bucket_len=bucket_len,
            token_count=bucket_len,
            prefix_tokens=tuple(prefix_tokens[:bucket_len]),
            prefix_hashes=_build_prefix_hashes(prefix_tokens[:bucket_len]),
            state_cpu=state_cpu,
            logits_cpu=logits_cpu,
            last_updated=float(row[2]) if row[2] is not None else time.time(),
        )
        self._store_prefix_entry_locked(entry, persist=False)
        return entry

    def match_prefix_state(
        self,
        prompt_tokens: List[int] | Tuple[int, ...],
        device: str = "cuda",
    ) -> Optional[dict]:
        token_tuple = tuple(prompt_tokens)
        if not token_tuple:
            return None

        with self.cache_lock:
            self._ensure_prefix_trie_locked()
            state_id, matched_len = self.prefix_trie.longest_prefix(token_tuple)
            if state_id is not None:
                entry = self.prefix_entry_index.get(state_id)
                if entry is not None:
                    bucket_cache = self.prefix_l2_cache[entry.bucket_len]
                    if state_id in bucket_cache:
                        bucket_cache.move_to_end(state_id)
                    prompt_prefix_hashes = _build_prefix_hashes(token_tuple)
                    print(
                        "[StatePool][PREFIX HIT][L2] "
                        f"matched_tokens={matched_len} "
                        f"bucket_len={entry.bucket_len} "
                        f"prompt_len={len(token_tuple)} "
                        f"state_id={entry.state_id[:160]} "
                        f"hash_{entry.bucket_len}={prompt_prefix_hashes.get(entry.bucket_len)}"
                    )
                    return {
                        "state_id": entry.state_id,
                        "matched_tokens": matched_len,
                        "bucket_len": entry.bucket_len,
                        "state": self._clone_to_device_state(entry.state_cpu, device),
                        "logits": self._clone_optional_tensor(entry.logits_cpu, device),
                        "cache_source": "l2_ram",
                    }

        for bucket in reversed(PREFIX_CACHE_BUCKETS):
            if len(token_tuple) < bucket:
                continue
            with self.cache_lock:
                entry = self._load_prefix_entry_from_db_locked(token_tuple, bucket)
                if entry is not None:
                    prompt_prefix_hashes = _build_prefix_hashes(token_tuple)
                    print(
                        "[StatePool][PREFIX HIT][DISK] "
                        f"matched_tokens={bucket} "
                        f"bucket_len={entry.bucket_len} "
                        f"prompt_len={len(token_tuple)} "
                        f"state_id={entry.state_id[:160]} "
                        f"hash_{entry.bucket_len}={prompt_prefix_hashes.get(entry.bucket_len)} "
                        "-> load_to_l2"
                    )
                    return {
                        "state_id": entry.state_id,
                        "matched_tokens": bucket,
                        "bucket_len": entry.bucket_len,
                        "state": self._clone_to_device_state(entry.state_cpu, device),
                        "logits": self._clone_optional_tensor(entry.logits_cpu, device),
                        "cache_source": "disk",
                    }

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
        prefix_entries_to_save: List[PrefixCacheEntry] = []
        with self.cache_lock:
            while self.l1_cache:
                sid, state = self.l1_cache.popitem()
                items_to_save.append((sid, [t.to('cpu') for t in state]))
            
            while self.l2_cache:
                sid, state = self.l2_cache.popitem()
                items_to_save.append((sid, state))

            for bucket_cache in self.prefix_l2_cache.values():
                while bucket_cache:
                    _, entry = bucket_cache.popitem()
                    prefix_entries_to_save.append(entry)
            self.prefix_entry_index.clear()
            self.prefix_trie.clear()
            self._prefix_trie_dirty = False

        with self.db_lock:
            try:
                self.db_conn.execute("BEGIN TRANSACTION")
                for sid, state in items_to_save:
                    blob = self._serialize(state)
                    self.db_conn.execute(
                        "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)",
                        (sid, blob, time.time()),
                    )

                for entry in prefix_entries_to_save:
                    state_blob = self._serialize(entry.state_cpu)
                    logits_blob = self._serialize(entry.logits_cpu) if entry.logits_cpu is not None else None
                    row = [
                        entry.state_id,
                        entry.bucket_len,
                        entry.token_count,
                    ]
                    row.extend(entry.prefix_hashes.get(bucket) for bucket in PREFIX_CACHE_BUCKETS)
                    row.extend([state_blob, logits_blob, entry.last_updated])
                    placeholders = ", ".join("?" for _ in row)
                    columns = ", ".join(
                        ["state_id", "bucket_len", "token_count", *PREFIX_HASH_COLUMNS, "state_blob", "logits_blob", "last_updated"]
                    )
                    self.db_conn.execute(
                        f"INSERT OR REPLACE INTO prefix_cache ({columns}) VALUES ({placeholders})",
                        row,
                    )

                self.db_conn.commit()
                print(
                    f"[StatePool] Successfully saved {len(items_to_save)} sessions "
                    f"and {len(prefix_entries_to_save)} prefix states."
                )
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

    def list_prefix_states_in_db(self) -> List[Tuple[str, int, float]]:
        with self.db_lock:
            self.db_cursor.execute(
                "SELECT state_id, bucket_len, last_updated FROM prefix_cache ORDER BY last_updated DESC"
            )
            results = self.db_cursor.fetchall()
            return [(row[0], int(row[1]), row[2]) for row in results]

    def list_all_states(self) -> dict:

        with self.cache_lock:
            l1_states = list(self.l1_cache.keys())
            l2_states = list(self.l2_cache.keys())
            prefix_l2_counts = {
                str(bucket): len(cache) for bucket, cache in self.prefix_l2_cache.items()
            }
            prefix_l2_keys = {
                str(bucket): list(cache.keys()) for bucket, cache in self.prefix_l2_cache.items()
            }

        db_states = self.list_states_in_db()
        db_states_keys = [state[0] for state in db_states]
        prefix_db_states = self.list_prefix_states_in_db()

        return {
            "l1_cache": l1_states,
            "l2_cache": l2_states,
            "database": db_states_keys,
            "total_count": len(l1_states) + len(l2_states) + len(db_states_keys),
            "prefix_l2_counts": prefix_l2_counts,
            "prefix_l2_cache": prefix_l2_keys,
            "prefix_database_count": len(prefix_db_states),
        }

    def print_all_states_status(self):

        all_states = self.list_all_states()

        print(f"[StatePool] All States Status - Total {all_states['total_count']} sessions:")
        print("=" * 80)

        print(f"L1 Cache (VRAM) - Count: {len(all_states['l1_cache'])}")
        print("-" * 40)
        for session_id in all_states["l1_cache"]:
            print(f"  {session_id}")

        print(f"\nL2 Cache (RAM) - Count: {len(all_states['l2_cache'])}")
        print("-" * 40)
        for session_id in all_states["l2_cache"]:
            print(f"  {session_id}")

        print(f"\nDatabase (Disk) - Count: {len(all_states['database'])}")
        print("-" * 40)
        for session_id in all_states["database"]:
            print(f"  {session_id}")

        print("\nPrefix Cache (RAM / L2)")
        print("-" * 40)
        for bucket in PREFIX_CACHE_BUCKETS:
            count = all_states["prefix_l2_counts"][str(bucket)]
            print(f"  bucket={bucket}: {count}/{PREFIX_CACHE_BUCKET_CAPACITY}")

        print(f"\nPrefix Cache (Disk) - Count: {all_states['prefix_database_count']}")

        if all_states["total_count"] == 0 and all_states["prefix_database_count"] == 0:
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
                if deleted_from_cache:
                    return True
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
