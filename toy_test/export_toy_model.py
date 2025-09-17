import torch
from toy_model import ToyModel

model = ToyModel().eval()
example_input = torch.randn(1, 10)

# Export the model
exported = torch.export.export(model, (example_input,))

# Compile and package into model.pt2
torch._inductor.aoti_compile_and_package(
    exported,
    package_path="model.pt2"   # only this arg is required in your version
)

print("Exported AOTInductor package to model.pt2")

