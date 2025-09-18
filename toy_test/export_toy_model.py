import torch
from torch import nn
from torch._inductor import aoti_compile_and_package
from torch.export import export

# --- 1. Self-contained Residual block ---
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# --- 2. Wrap in a submodule (nn.Linear) ---
fn = nn.Linear(10, 10)
residual_block = Residual(fn)

# --- 3. Dummy input ---
dummy_input = torch.randn(1, 10)

# --- 4. Export to ExportedProgram ---
ep = export(residual_block, (dummy_input,))

# --- 5. Compile and package into .pt2 ---
aoti_compile_and_package(
    ep,
    package_path="./residual_block.pt2"
)

print("Exported AOTInductor package to residual_block.pt2")

