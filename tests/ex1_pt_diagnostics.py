import torch
import torch._dynamo
import time
import logging

# Enable PyTorch 2.0 Diagnostics to see WHY it's failing

import logging
try:
    import torch._logging
    torch._logging.set_logs(dynamo=logging.INFO)
except ImportError:
    try:
        torch._dynamo.config.log_level = logging.INFO
    except AttributeError:
        logging.getLogger("torch._dynamo").setLevel(logging.INFO)

try:
    torch._dynamo.config.verbose = True
except AttributeError:
    pass

DEVICE = "cuda"
DTYPE = torch.float16
HIDDEN_DIM = 1024

class ManualNormBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias   = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean     = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var      = (centered * centered).mean(dim=-1, keepdim=True)
        normed   = centered * torch.rsqrt(var + 1e-5)
        scaled   = normed * self.weight + self.bias
        
        x3     = scaled * scaled * scaled
        inner  = 0.7978845608 * (scaled + 0.044715 * x3)
        gelu   = 0.5 * scaled * (1.0 + torch.tanh(inner))
        return x + gelu

print("=" * 60)
print("EXPERIMENT 3: Why is PT2 slow on ex1?")
print("=" * 60)

model = ManualNormBlock(HIDDEN_DIM).to(DEVICE, DTYPE).eval()

# Compile with PT2 and force dynamic shapes
pt2_model = torch.compile(model, dynamic=True)

shapes = [(1, HIDDEN_DIM), (8, HIDDEN_DIM), (32, HIDDEN_DIM), (128, HIDDEN_DIM)]

print("\n--- Running inferences ---")
for shape in shapes:
    print(f"\n>> Running shape {shape}")
    x = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
    
    # Warmup / execution
    with torch.no_grad():
        out = pt2_model(x)
        torch.cuda.synchronize()
