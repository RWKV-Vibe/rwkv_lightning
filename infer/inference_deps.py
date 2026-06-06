import torch

from infer.rwkv_batch.sampler import sample
from infer.rwkv_batch.utils import sampler_gumbel_batch


def get_torch():
    return torch


def get_sample():
    return sample


def get_sampler_gumbel_batch():
    return sampler_gumbel_batch
