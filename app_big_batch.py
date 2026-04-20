import argparse
import atexit
import signal
import sys

from API_servers.api_service import create_app
from infer.inference import InferenceEngine
from model_load.model_loader import load_model_and_tokenizer
from state_manager.state_pool import shutdown_state_manager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
    parser.add_argument(
        "--runtime",
        type=str,
        default="fp16",
        choices=["fp16", "int8"],
        help="Inference runtime variant to load",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--password", type=str, default=None)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    print(
        "[INFO] app_big_batch.py is now a compatibility launcher for the unified API server. "
        "Use /v1/chat/completions with scheduler='throughput' or /big_batch/completions for the throughput path."
    )
    model, tokenizer, args, rocm_flag = load_model_and_tokenizer(
        cli_args.model_path, runtime=cli_args.runtime
    )
    engine = InferenceEngine(model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm_flag)
    app = create_app(engine, password=cli_args.password)

    def cleanup_handler(signum=None, frame=None):
        shutdown_state_manager()
        engine.shutdown()
        sys.exit(0)

    def cleanup_at_exit():
        shutdown_state_manager()
        engine.shutdown()

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_at_exit)
    app.start(host="0.0.0.0", port=cli_args.port)


if __name__ == "__main__":
    main()
