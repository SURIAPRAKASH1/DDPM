import torch 
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import math
from inspect import isfunction 
from functools import partial
from typing import Optional


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):

    """
    Uses Residual path to jump inputs and gradients back and forth.

    Attributes:
        fn(function) : During class call args/kwargs feeded thorugh this function and then input added
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Downsample(dim:int, dim_out:int)-> nn.Module:

    " Reduces spatial resolution. Increase channels "

    return nn.Sequential(
        Rearrange("b c (h i) (w j)->b (i c j) h w", i = 2, j = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


def Upsample(dim: int, dim_out:int)-> nn.Module:

    " Increase spatial resolution. reduces channels "

    return nn.Sequential(
        nn.Upsample(scale_factor= 2, mode= 'nearest'),
        nn.Conv2d(dim, dim_out, 1)
    )


class PreNorm(nn.Module):

    " Normalizies the input before feeding to Attention "

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    

class SinusoidalPositionalEmbeddings(nn.Module):

    """ 
    Computes Positional embeddings to insert positional info to model 

    Attributes:
        dim (int): Dimentionaly of Input channels
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor)-> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class MultiHeadAttention(nn.Module):

    """
    Applies MultiHeadAttention to given Input tensor

    Attributes:

        dim (int): Dimentionality of Input channels
        scale (float): softmax scale for attention
        head_dim (int): Dimentionality of Each head
        heads (int): Number of heads in MultiHead-Attention
        hidden_dim (int): Total channel dimentionality in attention (heads * head_dim)
        to_qkv (nn.Module): Convolutional operation to get attention parameters
        to_out (nn.Module): Final Convolutional operation to get final output same as input size

    Returns:

        out (torch.Tensor): Same shape as input after MultiHead-Attention
    """

    def __init__(self, dim: int, head_dim: int = 32, heads: int = 4):
        super().__init__()

        self.dim = dim
        self.scale = head_dim ** - 0.5
        self.heads = heads
        self.head_dim = head_dim
        hidden_dim = heads * head_dim
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias= False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert self.dim == C

        # project x to get q,k,v
        qkv = self.to_qkv(x).chunk(3, dim = 1)     # (B, hs*hd, H, W)

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h = self.heads), qkv
        )

        q = q * self.scale
        attn = torch.einsum("b h d i,b h d j ->b h i j", q, k)         # as a Matirx (B, hs, H*W, hd) @ (B, hs, H*W, hd) -> (B, hs, H*W, H*W)
        attn = attn - attn.amax(dim = -1, keepdim= True).detach()
        attn = torch.softmax(attn, dim = -1)
        attn = torch.einsum("b h i j, b h d j-> b h i d", attn, v)     # as a Matrix (B, hs, H*W, H*W) @ (B, hs, H*W, hd) -> (B, hs, H*W, hd)

        out = rearrange(attn, "b h (x y) d -> b (h d) x y", x = H, y = W)       # (B, hs*hd, H, W)
        return self.to_out(out)                                           # (B, C, H, W)


class LinearAttention(nn.Module):

    """
    Applies LinearAttention to given Input:
        Compare to MultiHead Attention that takes quadratic time/memory, in other hand LinearAttention takes linear time/memory for sequence length

    Attributes:

        dim (int): Dimentionality of Input channels
        scale (float): softmax scale for attention
        head_dim (int): Dimentionality of Each head
        heads (int): Number of heads in MultiHead-Attention
        hidden_dim (int): Total channel dimentionality in attention (heads * head_dim)
        to_qkv (nn.Module): Convolutional operation to get attention parameters
        to_out (nn.Module): Final Convolutional operation & Norm to get final output that same as input size
    
    Returns:

        out (torch.Tensor): Same shape as input after Linear Attention
    """

    def __init__(self, dim: int, head_dim: int = 32, heads: int = 4):
        super().__init__()

        self.dim = dim
        self.scale = head_dim ** -0.5
        self.heads = heads
        self.head_dim = head_dim
        hidden_dim = heads * head_dim
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias= False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert self.dim == C

        # project x to get q,k,v
        qkv = self.to_qkv(x).chunk(3, dim = 1)     # (B, hs*hd, H, W)

        # (B, hs*hd, H, W) -> (B, hs, hd, H*W) colapsing spatial
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h = self.heads), qkv
        )

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale

        context = torch.einsum("b h d n, b h e n-> b h d e", k, v)        # as a Matrix (B, hs, hd, H*W) @ (B, hs, H*W, hd) -> (B, hs, hd, hd)
        out = torch.einsum("b h d e, b h d n-> b h e n", context, q)      # as a Matrix (B, hs, hd, hd) @ (B, hs, hd, H*W) -> (B, hs, hd, H*W)

        out =  rearrange(out, "b h d (x y) -> b (h d) x y", h = self.heads, x = H, y = W)      # (B, hs*hd, H, W)
        return self.to_out(out)                         # (B, C, H, W)



class WeightStandardizedConv2d(nn.Conv2d):

    """
    Standalizes (mean 0, variance 1) the weights of convolution layers
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):

    """
    Block: Sequence operations of Conv or WSconv -> Norm -> Act.
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 8, dropout = 0.):
        super().__init__()

        assert dim_out % groups == 0, "Number of channels {dim_out} must be divisable by Number of groups {groups}"

        self.conv = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scale_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)

        # add positional info
        if exists(scale_shift):
            scale, shift = scale_shift
            x =  x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(nn.Module):

    """
    ResnetBlock: Uses residual path to jump gradients one end to other end

    Attributes:

        mlp (nn.Module): Linear projection for time embeddings
        block1 (nn.Module): First block in Residual Block
        block2 (nn.Module): Second block in Residual Block
        block2 (nn.Conv2d): Final convolutional operation
    
    Returns:

        out (torch.Tensor): Output from Residual Block after all the operation being done

    """

    def __init__(self, dim:int, dim_out:int, groups:int, time_emd_dim:Optional[int] = None, dropout = 0.):
        super().__init__()

        # positional Embeddings projection
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emd_dim, dim_out * 2))
            if exists(time_emd_dim)
            else None
        )

        # blocks
        self.block1 = Block(dim, dim_out, groups, dropout = 0.)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1)   # conv1x1 preservers spatial, no need to use padding

    def forward(self, x:torch.Tensor, time_embd: Optional[torch.Tensor] = None)-> torch.Tensor:
        scale_shift = None

        if exists(time_embd) and exists(self.mlp):
            time_embd = self.mlp(time_embd)
            time_embd  = rearrange(time_embd, "b c -> b c 1 1")
            scale_shift = time_embd.chunk(2, dim= 1)       # (B, C, 1, 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
    
class Unet(nn.Module):

    """
    Unet: Model used to approximates 
          the posterior q(x_{t-1}|x_{t}, x0) by learning reverse diffusion process p(x_{t-1}|x_{t})

    Attributes:

        dim (int): Initial channel dimentionality to start the convolutional operations
        channels (int): Channels in a Image. (eg: RGB = 3, G = 1, RGBA = 4)
        out_dim (int): Optionaly we can specifie the final output dim of tensor (usefull when want to learn variance)
        dim_mults (tuple): Channel multiplier at step of spatial resolution reduction
        resnet_block_groups (int): How many groups in GroupNormalization (channels must be divisible groups)
        self_condition (bool): Optionaly we can train model with condition on previous generation
        dropout (float): Deactivating some of activation in resnet_block

    Returns:

        out (torch.Tensor): Generated model output
        
    """

    def __init__(self,
                dim: int,
                channels: int,
                init_dim: Optional[int] = None,
                out_dim: Optional[int] = None,
                dim_mults: tuple = (1, 2, 4, 8),
                resnet_block_groups = 4,
                self_condition: Optional[bool] = False,
                dropout = 0.
                ) -> None:
        super().__init__()

        # initiating dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self.self_condition else 1)

        init_dim = default(dim, init_dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1)  # init by applying conv1x1

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        assert all(map(lambda dim: dim % resnet_block_groups == 0, dims)), "every channels dim must be divisible by resnet_bloc_groups" 
        in_out = list(zip(dims[:-1], dims[1:]))                  # input and ouput dim of each resolution stage


        block_klass = partial(ResnetBlock, groups = resnet_block_groups, dropout = dropout)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)            # if it's a last resolution down then it's middle block

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emd_dim= time_dim),
                    block_klass(dim_in, dim_in, time_emd_dim= time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding= 1)
                ])
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emd_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, MultiHeadAttention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emd_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emd_dim = time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emd_dim = time_dim),
                    Residual(PreNorm(dim_out , LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in)
                    if not is_last
                    else nn.Conv2d(dim_out, dim_in, 3, padding = 1)
                    ])
            )

        self.out_dim = default(out_dim, channels)

        # final res block
        self.final_res_block = block_klass(dim * 2, dim, time_emd_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # report parameters count
        print("total parameters %.2f" % (self._parameters_count() / 1e+6), "M")

    def _parameters_count(self)-> int:
        "Returns total number of parameters in model"
        t = 0
        for p in self.parameters():
            t += p.nelement()
        return t

    def device(self) -> torch.DeviceObjType:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor,
                t: torch.Tensor,
                x_self_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        """
        Args:
            x (torch.Tensor): Gaussian noised image/images as Tensor
            t (torch.Tensor): timesteps for each noised image/images
            x_self_cond (torch.Tensor): Optionaly generate output condition on previous generation of the model
        
        Returns:

            out (torch.Tensor): Tensor output with shape based on what we wanna predict (eg: noise/x0/mean)
        
        """

        # simply just concating previous generation with current generation to give model more info about what it did last time.
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(t)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)