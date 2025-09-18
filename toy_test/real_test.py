import torch
from compress_modules3d_mid_SR import CompressorMix

# 1. Create model
model = CompressorMix(
    dim=16,
    dim_mults=[1,2,3,4],
    reverse_dim_mults=[4,3,2],
    hyper_dims_mults=[4,4,4],
    channels=1,
    out_channels=1,
    d3=True,
    sr_dim=16
)

# 2. Load pretrained weights
state_dict = torch.load("/lustre/blue/ranka/eklasky/CAESAR_C/pretrained/caesar_v.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 3. Dummy input
dummy_input = torch.randn(1,1,8,256,256)

# 4. Export
torch._inductor.aoti_compile_and_package(
    model,
    (dummy_input,),
    "./model.pt2"
)

