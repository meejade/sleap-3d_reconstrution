import torch, time
import sleap
from sleap_nn.inference.predictors import load_legacy_model

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

a = torch.randn(10000, 10000, device='cuda')
b = torch.randn(10000, 10000, device='cuda')

torch.cuda.synchronize()
t0 = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
print("Matrix multiply OK, time:", time.time() - t0, "s")