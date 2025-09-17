import numpy as np

data = np.random.rand(1*1*8*256*256).astype(np.float32)
data.tofile("RHO_example.f32")

