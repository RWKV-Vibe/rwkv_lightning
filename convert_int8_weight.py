import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

DTYPE = torch.half
I8_SUFFIXES = ('.i8_w', '.i8_scale')


def quantize_w8a8_weight_cpu(w: torch.Tensor):
    # Input linear weight is [out_features, in_features]. Store as [N, M]
    # for row-major activation [B, N] @ weight [N, M].
    weight = w.float().t().contiguous()
    scale = torch.amax(weight.abs(), dim=0).div(127.0).clamp_min(1.0e-8).contiguous()
    q = weight.div(scale).round().clamp(-127, 127).to(dtype=torch.int8).contiguous()
    return q, scale


def store_i8_linear_weight(z: dict, key: str, linear_weight: torch.Tensor | None = None):
    if linear_weight is None:
        linear_weight = z[key]
    w, scale = quantize_w8a8_weight_cpu(linear_weight.contiguous())
    z[key + '.i8_w'] = w
    z[key + '.i8_scale'] = scale


def delete_fp16_weight(z: dict, key: str):
    z.pop(key, None)


def convert(input_path: Path, output_path: Path, keep_fp16_large: bool):
    z = torch.load(input_path, map_location='cpu')
    if '__int8_preprocessed' in z:
        raise RuntimeError(f'{input_path} already looks like an int8 preprocessed checkpoint')

    n_head, head_size = z['blocks.0.att.r_k'].shape
    n_embd = n_head * head_size

    keys = list(z.keys())
    max_layer = -1
    for k in keys:
        kk = k.split('.')
        if ('att.g1' in k or 'att.g2' in k or 'att.a1' in k or 'att.a2' in k or
                'att.w1' in k or 'att.w2' in k or 'att.v1' in k or 'att.v2' in k or
                'ffn.value.weight' in k):
            z[k] = z[k].t()
        z[k] = z[k].squeeze().to(dtype=DTYPE).contiguous()
        if k.endswith('att.r_k'):
            z[k] = z[k].flatten().contiguous()
        if kk[0] == 'blocks':
            max_layer = max(max_layer, int(kk[1]))
    n_layer = max_layer + 1

    z['emb.weight'] = F.layer_norm(
        z['emb.weight'].float(),
        (n_embd,),
        weight=z['blocks.0.ln0.weight'].float(),
        bias=z['blocks.0.ln0.bias'].float(),
    ).to(dtype=DTYPE).contiguous()
    z['blocks.0.att.v0'] = z['blocks.0.att.a0']
    z['blocks.0.att.v1'] = z['blocks.0.att.a1']
    z['blocks.0.att.v2'] = z['blocks.0.att.a2']

    large_weight_keys: list[str] = []
    for i in range(n_layer):
        att = f'blocks.{i}.att.'
        ffn = f'blocks.{i}.ffn.'
        for name in ('receptance.weight', 'key.weight', 'value.weight', 'output.weight'):
            key = att + name
            store_i8_linear_weight(z, key)
            large_weight_keys.append(key)
        key = ffn + 'key.weight'
        store_i8_linear_weight(z, key)
        large_weight_keys.append(key)
        key = ffn + 'value.weight'
        # ffn.value.weight is transposed above for the old k @ V path, so quantize
        # its linear form [out_features, in_features].
        store_i8_linear_weight(z, key, z[key].t().contiguous())
        large_weight_keys.append(key)

    store_i8_linear_weight(z, 'head.weight')
    large_weight_keys.append('head.weight')

    if not keep_fp16_large:
        for key in large_weight_keys:
            delete_fp16_weight(z, key)

    z['__int8_preprocessed'] = torch.tensor(1, dtype=torch.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, output_path)
    print(f'saved int8 checkpoint: {output_path}')
    print(f'layers={n_layer}, n_embd={n_embd}, keep_fp16_large={keep_fp16_large}')


def main():
    parser = argparse.ArgumentParser(description='Convert an RWKV7 checkpoint to a preprocessed W8A8 int8 checkpoint.')
    parser.add_argument('--input', type=Path, help='Input FP16/FP32 .pth checkpoint')
    parser.add_argument('--output', type=Path, help='Output int8 .pth checkpoint')
    parser.add_argument('--keep-fp16-large', action='store_true', help='Keep original large FP16 weights alongside int8 weights')
    args = parser.parse_args()
    convert(args.input, args.output, args.keep_fp16_large)


if __name__ == '__main__':
    main()
