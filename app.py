import argparse
import atexit
import signal
import sys

import uvicorn

from API_servers.fastapi_service import create_app
from infer.high_throughput import (
    DEFAULT_CUDA_CACHE_BUDGET_GB,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_PREFILL_AREA,
    DEFAULT_PREFILL_CACHE_SHAPE_LIMIT,
    DEFAULT_PREFILL_TARGET_BATCH_SIZE,
    HighThroughputConfig,
)
from infer.inference import InferenceEngine
from model_load.model_loader import load_model_and_tokenizer
from state_manager.state_pool import shutdown_state_manager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--password", type=str, default=None, help="API password for authentication")
    parser.add_argument(
        "--enable-high-throughput",
        action="store_true",
        help="Enable the high_throughput endpoint and preallocate its resident decode pool",
    )
    parser.add_argument(
        "--high-throughput-max-active-states",
        "--high-throughput-max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        dest="high_throughput_max_batch_size",
        help="Resident pool max active states for the high_throughput endpoint",
    )
    parser.add_argument(
        "--high-throughput-prefill-target-batch-size",
        "--high-throughput-prefill-batch-size",
        type=int,
        dest="high_throughput_prefill_target_batch_size",
        default=DEFAULT_PREFILL_TARGET_BATCH_SIZE,
        help="Preferred power-of-two prefill batch size for the high_throughput endpoint",
    )
    parser.add_argument(
        "--high-throughput-prefill-target-area",
        "--high-throughput-prefill-area",
        type=int,
        dest="high_throughput_prefill_target_area",
        default=DEFAULT_PREFILL_AREA,
        help="Preferred prefill batch-size times chunk-size area for the high_throughput endpoint",
    )
    parser.add_argument(
        "--high-throughput-prefill-cache-shape-limit",
        type=int,
        default=DEFAULT_PREFILL_CACHE_SHAPE_LIMIT,
        help="Number of distinct high_throughput prefill shapes to keep cached before clearing CUDA cache; 0 disables the limit",
    )
    parser.add_argument(
        "--high-throughput-cuda-cache-budget-gb",
        type=float,
        default=DEFAULT_CUDA_CACHE_BUDGET_GB,
        help="Extra CUDA reserved-memory budget, in GB, above the resident high_throughput pool before clearing cache; 0 disables the budget",
    )
    parser.add_argument(
        "--high-throughput-clear-cuda-cache-each-request",
        action="store_true",
        help="Clear CUDA cache after every high_throughput request instead of reusing planned prefill shape cache",
    )
    return parser.parse_args()


def main():
    args_cli = parse_args()
    model, tokenizer, args, rocm_flag = load_model_and_tokenizer(args_cli.model_path)
    engine = InferenceEngine(model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm_flag)
    high_throughput_config = HighThroughputConfig(
        enabled=args_cli.enable_high_throughput,
        decode_max_batch_size=args_cli.high_throughput_max_batch_size,
        prefill_area=args_cli.high_throughput_prefill_target_area,
        prefill_target_batch_size=args_cli.high_throughput_prefill_target_batch_size,
        prefill_cache_shape_limit=args_cli.high_throughput_prefill_cache_shape_limit,
        cuda_cache_budget_gb=args_cli.high_throughput_cuda_cache_budget_gb,
        clear_cuda_cache_each_request=args_cli.high_throughput_clear_cuda_cache_each_request,
    )
    app = create_app(
        engine,
        password=args_cli.password,
        high_throughput_config=high_throughput_config,
    )

    def cleanup_handler(signum, frame):
        print("\nShutting down server...")
        sys.exit(0)

    def cleanup_at_exit():
        print("Persisting all states to database...")
        shutdown_state_manager()
        engine.shutdown()
        print("All states persisted to database.")

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_at_exit)

    uvicorn.run(app, host="0.0.0.0", port=args_cli.port)


if __name__ == "__main__":
    main()
