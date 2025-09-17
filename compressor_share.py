#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader


# In[ ]:


# Export program


# In[10]:


from CAESAR.models.network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from CAESAR.models.utils import quantize, NormalDistribution
import time
import yaml
from CAESAR.models.BCRN.bcrn_model import BluePrintConvNeXt_SR
import torch.nn as nn
import torch.nn.init as init
from CAESAR.models.RangeEncoding import RangeCoder


#  compress_modules3d_mid_SR.py  -----------------------------------------------------------------------------------------------------------------------------------------
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def super_resolution_model(img_size = 64, in_chans=32, out_chans=1, sr_dim = "HAT", pretrain = False, sr_type = "BCRN"):

    if sr_type == "BCRN":
        sr_model = BluePrintConvNeXt_SR(in_chans, 1, 4, sr_dim)
        if pretrain:
            loaded_params, not_loaded_params = sr_model.load_part_model("./pretrain/BCRN_SRx4.pth")
        else:
            loaded_params, not_loaded_params = [], sr_model.parameters()

        return sr_model, loaded_params, not_loaded_params

def reshape_batch_2d_3d(batch_data, batch_size):
    BT,C,H,W = batch_data.shape
    T = BT//batch_size
    batch_data = batch_data.view([batch_size, T, C, H, W])
    batch_data = batch_data.permute([0,2,1,3,4])
    return batch_data

def reshape_batch_3d_2d(batch_data):
    B,C,T,H,W = batch_data.shape
    batch_data = batch_data.permute([0,2,1,3,4]).reshape([B*T,C,H,W])
    return batch_data

class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        d3 = False
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))

        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))

        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1], convert_module = True)

        self.range_coder = None

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, x):

        self.t_dim = x.shape[2]

        for i, (resnet, down) in enumerate(self.enc): # [b, 1, t, 256, 256]
            if i==0:
                x = x.permute(0,2,1,3,4)
                x = x.reshape(-1, *x.shape[2:]) # [b*t, 1, 256, 256]
            if i==2:
                x = x.reshape(-1, self.t_dim, *x.shape[1:])
                x = x.permute(0,2,1,3,4) # [b, c, t, h, w]

            x = resnet(x)
            x = down(x)


        x = x.permute(0,2,1,3,4)
        x = x.reshape(-1, *x.shape[2:])

        latent = x
        return latent



    def hyper_encode(self, x):




        for i, (conv, act) in enumerate(self.hyper_enc):
            x = conv(x)
            x = act(x)

        hyper_latent = x
        return hyper_latent


    def hyper_decode(self, x): 

        for i, (deconv, act) in enumerate(self.hyper_dec):
            x = deconv(x)
            x = act(x)

        mean, scale = x.chunk(2, 1)

        return mean, scale


    def decode(self, x): # [n*t, c, h,w ] [8, 256, 16, 16]
        # output = []

        for i, (resnet, up) in enumerate(self.dec):

            if i==0:
                x = x.reshape(-1, self.t_dim//4, *x.shape[1:])
                x = x.permute(0,2,1,3,4) # [b, c, t, h, w]

            if i==2:
                x = x.permute(0,2,1,3,4)
                x = x.reshape(-1, *x.shape[2:]) # [b*t, 1, 256, 256]

            x = resnet(x)
            x = up(x)

        return x


    def compress(self, x, return_latent = False, real = False, return_time = False):
        if self.range_coder is None:
            _quantized_cdf, _cdf_length, _offset = self.prior._update(30)
            self.range_coder = RangeCoder(_quantized_cdf = _quantized_cdf, _cdf_length= _cdf_length, _offset= _offset, medians = self.prior.medians.detach())

        B,C,T,H,W = x.shape
        original_shape = x.shape

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        latent = self.encode(x)
        hyper_latent = self.hyper_encode(latent)
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        mean, scale = self.hyper_decode(q_hyper_latent)
        '''
        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            elapsed_time = time.time() - start_time

        latent_string = self.range_coder.compress(latent, mean, scale)
        hyper_latent_string = self.range_coder.compress_hyperlatent(hyper_latent)

        bpf_real = torch.Tensor([(len(lc)+len(hc))*8 for lc, hc in zip(latent_string, hyper_latent_string)])

        compressed_data = (latent_string, hyper_latent_string, original_shape, hyper_latent.shape)

        state4bpp = {"latent": latent, "hyper_latent": hyper_latent, "mean":mean, "scale": scale}

        bpf_theory, bpp = self.bpp(original_shape, state4bpp)

        result = {}

        result["bpf_entropy"] = bpf_theory
        result["compressed"] = compressed_data
        result["bpf_real"] = bpf_real


        if return_latent:
            q_latent = quantize(latent, "dequantize", mean)
            result["q_latent"] = q_latent

        if return_time:
            result["elapsed_time"] = elapsed_time
        '''
        #return latent_string, hyper_latent_string
        return latent, hyper_latent

    def decompress(self, latent_string, hyper_latent_string, original_shape, hyper_shape, device = "cuda"):
        B, _, T, _, _ = original_shape
        q_hyper_latent = self.range_coder.decompress_hyperlatent(hyper_latent_string, hyper_shape)
        mean, scale = self.hyper_decode(q_hyper_latent.to(device))

        q_latent = self.range_coder.decompress(latent_string, mean.detach().cpu(), scale.detach().cpu())
        q_latent = q_latent.to(device)

        return self.decode(q_latent)


    def bpp(self, shape, state4bpp):
        B, H, W = shape[0], shape[-2], shape[-1]
        n_pixels = shape[-3] * shape[-2] * shape[-1]

        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = NormalDistribution(state4bpp['mean'], state4bpp['scale'].clamp(min=0.1))

        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)

        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()

        bpb = hyper_rate.reshape(B, -1).sum(dim=-1) + cond_rate.reshape(B, -1).sum(dim=-1) # bit per block
        bpp = (hyper_rate.reshape(B, -1).sum(dim=-1) + cond_rate.reshape(B, -1).sum(dim=-1)) / n_pixels

        return bpb, bpp

    def forward(self, x, return_time = False):

        result = {}

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        # q_latent, q_hyper_latent, state4bpp, mean = self.encode(x)


        latent = self.encode(x)
        hyper_latent = self.hyper_encode(latent) 
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        mean, scale = self.hyper_decode(q_hyper_latent)
        q_latent = quantize(latent, "dequantize", mean.detach())


        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            result["encoding_time"] = time.time() - start_time



        state4bpp = {"latent": latent, "hyper_latent":hyper_latent, "mean":mean, "scale":scale }    
        frame_bit, bpp = self.bpp(x.shape, state4bpp)



        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        output = self.decode(q_latent)

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            result["decoding_time"] = time.time() - start_time

        result.update({
            "output": output,
            "bpp": bpp,
            "frame_bit":frame_bit,
            "mean": mean,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        })

        return result


class ResnetCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        d3 = False
    ):
        super().__init__(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels,
            d3
        )
        self.d3 = d3
        self.conv_layer =  nn.Conv3d if d3 else nn.Conv2d
        self.deconv_layer = nn.ConvTranspose3d if d3 else nn.ConvTranspose2d

        self.build_network()


    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            d3 = self.d3 if ind>=2 else False
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False, d3 = d3),
                        Downsample(dim_out, d3 = d3),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            d3 = self.d3 if ind<2 else False

            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in, d3 = d3),
                        Upsample(dim_out if not is_last else dim_in, dim_out, d3 = d3) if d3 else nn.Identity()
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1) if ind == 0 else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1) if is_last else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )



class CompressorMix(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        d3=False,
        sr_dim = 16
    ):
        super().__init__()  # Initialize the nn.Module parent class

        self.entropy_model = ResnetCompressor(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels,
            d3
        )

        # Update channels for sr_model based on entropy_model's output
        channels = dim * reverse_dim_mults[-1]

        # Initialize super-resolution model
        self.sr_model, self.loaded_params, self.not_loaded_params = super_resolution_model(
            img_size=64, in_chans=channels, out_chans=out_channels, sr_type = "BCRN", sr_dim = sr_dim
        )
    '''
    def forward(self, x, return_time = False):
        B = x.shape[0]

        results = self.entropy_model(x, return_time)
        outputs = results["output"]

        # Apply super-resolution model
        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            start_time = time.time()

        outputs = self.sr_model(outputs)  # Use self.sr_model instead of sr_model

        if return_time:
            torch.cuda.synchronize()  # Wait for all GPU ops to finish
            results["decoding_time"] += time.time() - start_time


        # Reshape if needed
        outputs = reshape_batch_2d_3d(outputs, B)
        results["output"] = outputs

        return results
    '''
    def compress(self, x, return_latent = False,  real = False):
        return self.entropy_model.compress(x, return_latent, real)

    def forward(self, x):

        #dataset_org = dataloader.dataset
        #self.transform_shape = dataset_org.deblocking_hw

        #compressed_latent, latent_bytes = self.compress_caesar_v(x)
        latent, hyper_latent = self.compress(x)
        #latent_bytes = torch.sum(outputs["bpf_real"])                
        #compressed_latent = outputs["compressed"]

        #original_data = dataset_org.original_data()
        #print("original_data.shape after compress", original_data.shape, recons_data.shape)
        #original_data, org_padding = self.padding(original_data)
        #recons_data, rec_padding= self.padding(recons_data)

        #meta_data, compressed_gae = self.postprocessing_encoding(original_data, recons_data, eb)
        return latent, hyper_latent

    '''
    def compress_caesar_v(self, dataloader):

        total_bits = 0
        all_compressed_latent = []

        with torch.no_grad():
            for data in dataloader:
                outputs = self.compress(data[0].to('cuda'))
                total_bits += torch.sum(outputs["bpf_real"])

                compressed_latent = outputs["compressed"]
                all_compressed_latent.append(compressed_latent)

        return all_compressed_latent, total_bits/8
    '''

    def decompress(self, latent_string, hyper_latent_string, original_shape, hyper_shape, device = "cuda"):
        B = original_shape[0]

        outputs = self.entropy_model.decompress(latent_string, hyper_latent_string, original_shape, hyper_shape, device)
        outputs = self.sr_model(outputs)

        outputs = reshape_batch_2d_3d(outputs, B)
        return outputs

    def decode(self, x, batch_size):
        x = self.entropy_model.decode(x)
        x = self.sr_model(x)
        x = reshape_batch_2d_3d(x, batch_size)
        return x
# end of compress_modules3d_mid_SR.py -----------------------------------------------------------------------------------------------------------------------------------

# In[36]:


from collections import OrderedDict

def remove_module_prefix(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

model = CompressorMix(
            dim=16,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1,
            d3=True,
            sr_dim=16
        )

#  compressor --------------------------------------------------------------------------------------------------------------------------------------------------------------
# _load_caesar_v_compressor method under loaded model?????????????????
state_dict = remove_module_prefix(torch.load('./pretrained/caesar_v.pt', map_location='cuda'))
#state_dict = torch.load('./pretrained/caesar_v.pt', map_location='cuda')
model.load_state_dict(state_dict)
# done compressor --------------------------------------------------------------------------------------------------------------------------------------------------------------

# In[50]:


model.eval()
with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ', device)
    model = model.to(device)
    example_inputs=(torch.randn(8, 1, 8, 256, 256, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    # [Optional] Specify the first dimension of the input x as dynamic.
    exported = torch.export.export(model, example_inputs, dynamic_shapes={"x": {0: batch_dim}})
    # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
    # Depending on your use case, e.g. if your training platform and inference platform
    # are different, you may choose to save the exported model using torch.export.save and
    # then load it back using torch.export.load on your inference platform to run AOT compilation.
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        # [Optional] Specify the generated shared library path. If not specified,
        # the generated artifact is stored in your system temp directory.
        package_path=os.path.join(os.getcwd(), "model.pt2"),
    )


# In[ ]:




