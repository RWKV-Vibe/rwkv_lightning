from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from infer import inference_deps
from infer.batch_inference import BatchInferenceMixin
from infer.big_batch import BigBatchMixin
from infer.inference_utils import InferenceUtilsMixin


class InferenceEngine(
    InferenceUtilsMixin,
    BatchInferenceMixin,
    BigBatchMixin,
):
    def __init__(self, model, tokenizer, args, rocm_flag):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.rocm_flag = rocm_flag
        self.model_lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=128, thread_name_prefix="model_inference"
        )

    def shutdown(self):
        self.executor.shutdown(wait=False)
