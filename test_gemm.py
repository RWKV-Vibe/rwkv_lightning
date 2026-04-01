import torch
import torch.nn.functional as F

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

x = torch.randn(4, 1024, device="cuda", dtype=torch.float16)
w = torch.randn(2048, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(2048, device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
y = F.linear(x, w, b)
torch.cuda.synchronize()

print(y.shape)