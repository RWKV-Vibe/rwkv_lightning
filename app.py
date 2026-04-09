import argparse
import atexit
import ast
import signal
import sys

from API_servers.api_service import create_app
from model_load.model_loader import load_model_and_tokenizer
from state_manager.state_pool import shutdown_state_manager


def parse_pp_devices(raw_value):
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--pp-devices must be a Python-style list, for example [0,1,2,3]"
        ) from exc

    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise argparse.ArgumentTypeError("--pp-devices must be a non-empty list")
    if not all(isinstance(device_id, int) for device_id in parsed):
        raise argparse.ArgumentTypeError("--pp-devices must contain integer CUDA device ids")
    return list(parsed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="RWKV model path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--password", type=str, default=None, help="API password for authentication")
    parser.add_argument(
        "--pp-devices",
        type=parse_pp_devices,
        default=None,
        help="Pipeline-parallel CUDA devices, for example [0,1,2,3]",
    )
    return parser.parse_args()


def main():
    args_cli = parse_args()
    if args_cli.pp_devices:
        from infer.inference_pp import InferenceEngine
    else:
        from infer.inference import InferenceEngine

    model, tokenizer, args, rocm_flag = load_model_and_tokenizer(
        args_cli.model_path,
        pp_devices=args_cli.pp_devices,
    )
    engine = InferenceEngine(model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm_flag)
    app = create_app(engine, password=args_cli.password)

    def cleanup_handler(signum, frame):
        print("\nShutting down server and persisting all states to database...")
        shutdown_state_manager()
        engine.shutdown()
        sys.exit(0)

    def cleanup_at_exit():
        shutdown_state_manager()
        engine.shutdown()
        print("All states persisted to database.")

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_at_exit)

    app.start(host="0.0.0.0", port=args_cli.port)


if __name__ == "__main__":
    main()
