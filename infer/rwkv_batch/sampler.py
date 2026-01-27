import torch, os
from torch.utils.cpp_extension import load
current_path = os.path.dirname(os.path.abspath(__file__))

ROCm_flag = torch.version.hip is not None
if ROCm_flag:
    sample = load(
        name="sample",
        sources = [f"{current_path}/hip/sampling_op.hip",f"{current_path}/hip/sampling.hip"],
        extra_cuda_cflags=['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics'],
        verbose=True,
    )
else:
    sample = load(
        name="sample",
        sources = [f"{current_path}/cuda/sampling.cpp",f"{current_path}/cuda/sampling.cu"],
        extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
        verbose=True,
    )

if __name__ == "__main__":
    batch_size = 128
    vocab_size = 131072
    temperature = 1.0
    top_p = 0.5
    top_k = -1
    presence_penalty = 1.0
    repetition_penalty = 0.1
    penalty_decay = 0.996
    states = sample.setup_rand(0, batch_size)
    logits = torch.rand(batch_size, vocab_size).to(0)
    penalties = torch.zeros(batch_size, vocab_size).to(0)
    print(logits)
    print(logits.shape)
    # samples = sample.batch_sampling_temperature_topk_topp(logits, states, temperature, top_k, top_p)
    samples = sample.batch_sampling_repetition_temperature_topk_topp(logits, penalties, states,
                                                                     presence_penalty, repetition_penalty, penalty_decay,
                                                                     temperature, top_k, top_p)
    print(samples)
