import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


# Meta-Former like Block
class gcc_mf_lg_Block(nn.Module):
    def __init__(self,
        dim,
        # gcc options
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True,
        mid_mix=True,
        bias=True,
        ffn_dim=2,
        ffn_dropout=0.0,
        dropout=0.1,
        local_kernel_size=7
        ):
        super(gcc_mf_lg_Block, self).__init__()
        # super().__init__()
        
        # record options
        self.global_dim, self.local_dim = dim//2, dim    # global_dim=dim/2=fs/4, local_dim=dim=fs/2, dim=fs/2
        self.instance_kernel_method = instance_kernel_method
        self.use_pe = use_pe
        self.mid_mix = mid_mix

        # spatial part
        self.dwconv = nn.Conv2d(self.local_dim, self.local_dim, kernel_size=local_kernel_size, padding=3, groups=self.local_dim) # depthwise conv
        self.local_norm = nn.BatchNorm2d(num_features=self.local_dim)

        self.pre_Norm_1 = nn.BatchNorm2d(num_features=self.global_dim)
        self.pre_Norm_2 = nn.BatchNorm2d(num_features=self.global_dim)

        self.meta_kernel_1_H = nn.Conv2d(self.global_dim, self.global_dim, (meta_kernel_size, 1), groups=self.global_dim).weight
        self.meta_kernel_1_W = nn.Conv2d(self.global_dim, self.global_dim, (1, meta_kernel_size), groups=self.global_dim).weight
        self.meta_kernel_2_H = nn.Conv2d(self.global_dim, self.global_dim, (meta_kernel_size, 1), groups=self.global_dim).weight
        self.meta_kernel_2_W = nn.Conv2d(self.global_dim, self.global_dim, (1, meta_kernel_size), groups=self.global_dim).weight

        self.meta_1_H_bias = nn.Parameter(torch.randn(self.global_dim)) if bias else None
        self.meta_1_W_bias = nn.Parameter(torch.randn(self.global_dim)) if bias else None
        self.meta_2_H_bias = nn.Parameter(torch.randn(self.global_dim)) if bias else None
        self.meta_2_W_bias = nn.Parameter(torch.randn(self.global_dim)) if bias else None

        self.meta_pe_1_H = nn.Parameter(torch.randn(1, self.global_dim, meta_kernel_size, 1)) if use_pe else None
        self.meta_pe_1_W = nn.Parameter(torch.randn(1, self.global_dim, 1, meta_kernel_size)) if use_pe else None
        self.meta_pe_2_H = nn.Parameter(torch.randn(1, self.global_dim, meta_kernel_size, 1)) if use_pe else None
        self.meta_pe_2_W = nn.Parameter(torch.randn(1, self.global_dim, 1, meta_kernel_size)) if use_pe else None

        self.mixer = nn.ChannelShuffle(groups=2) if mid_mix else None

        # channel part
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(num_features=2*dim),
            nn.Conv2d(2*dim, ffn_dim, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            nn.Dropout(p=ffn_dropout),  # using nn.Dropout
            nn.Conv2d(ffn_dim, 2*dim, kernel_size=(1, 1), bias=True),
            nn.Dropout(p=dropout)       # using nn.Dropout
        )
        self.ca = CA_layer(channel=2*dim)

    def get_instance_kernel(self, instance_kernel_size):
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.meta_kernel_1_H, self.meta_kernel_1_W, self.meta_kernel_2_H, self.meta_kernel_2_W
        elif self.instance_kernel_method == 'interpolation_bilinear':
            H_shape, W_shape = [instance_kernel_size, 1], [1, instance_kernel_size]
            return  F.interpolate(self.meta_kernel_1_H, H_shape, mode='bilinear', align_corners=True), \
                    F.interpolate(self.meta_kernel_1_W, W_shape, mode='bilinear', align_corners=True), \
                    F.interpolate(self.meta_kernel_2_H, H_shape, mode='bilinear', align_corners=True), \
                    F.interpolate(self.meta_kernel_2_W, W_shape, mode='bilinear', align_corners=True)

    def get_instance_pe(self, instance_kernel_size):
        if self.instance_kernel_method is None:
            return  self.meta_pe_1_H, self.meta_pe_1_W, self.meta_pe_2_H, self.meta_pe_2_W
        elif self.instance_kernel_method == 'interpolation_bilinear':
            H_shape, W_shape = [instance_kernel_size, 1], [1, instance_kernel_size]
            return  F.interpolate(self.meta_pe_1_H, H_shape, mode='bilinear', align_corners=True)\
                        .expand(1, self.global_dim, instance_kernel_size, instance_kernel_size), \
                    F.interpolate(self.meta_pe_1_W, W_shape, mode='bilinear', align_corners=True)\
                        .expand(1, self.global_dim, instance_kernel_size, instance_kernel_size), \
                    F.interpolate(self.meta_pe_2_H, H_shape, mode='bilinear', align_corners=True)\
                        .expand(1, self.global_dim, instance_kernel_size, instance_kernel_size), \
                    F.interpolate(self.meta_pe_2_W, W_shape, mode='bilinear', align_corners=True)\
                        .expand(1, self.global_dim, instance_kernel_size, instance_kernel_size)

    def forward(self, x):
        x_global, x_local = torch.chunk(x, 2, 1)
        # local
        x_local_res = x_local
        x_local = self.local_norm(x_local)
        x_local = self.dwconv(x_local)
        x_local = x_local + x_local_res
        # global
        x_1, x_2 = torch.chunk(x_global, 2, 1)
        x_1_res, x_2_res = x_1, x_2

        _, _, f_s, _ = x_1.shape
        K_1_H, K_1_W, K_2_H, K_2_W = self.get_instance_kernel(f_s)
        if self.use_pe:
            pe_1_H, pe_1_W, pe_2_H, pe_2_W = self.get_instance_pe(f_s)

        # =====================Spatial Part========================
        # pe
        if self.use_pe:
            x_1, x_2 = x_1 + pe_1_H, x_2 + pe_1_W
        # pre norm
        x_1, x_2 = self.pre_Norm_1(x_1), self.pre_Norm_2(x_2)
        # stage 1
        x_1_1 = F.conv2d(torch.cat((x_1, x_1[:, :, :-1, :]), dim=2),
            weight=K_1_H, bias=self.meta_1_H_bias, padding=0, groups=self.global_dim)
        x_2_1 = F.conv2d(torch.cat((x_2, x_2[:, :, :, :-1]), dim=3),
            weight=K_1_W, bias=self.meta_1_W_bias, padding=0, groups=self.global_dim)
        if self.mid_mix:
            mid_rep = torch.cat((x_1_1, x_2_1), dim=1)
            x_1_1, x_2_1 = torch.chunk(self.mixer(mid_rep), chunks=2, dim=1)

        # pe
        if self.use_pe:
            x_1_1, x_2_1 = x_1_1 + pe_2_W, x_2_1 + pe_2_H
        # stage 2
        x_1_2 = F.conv2d(torch.cat((x_1_1, x_1_1[:, :, :, :-1]), dim=3),
            weight=K_2_W, bias=self.meta_2_W_bias, padding=0, groups=self.global_dim)
        x_2_2 = F.conv2d(torch.cat((x_2_1, x_2_1[:, :, :-1, :]), dim=2), 
            weight=K_2_H, bias=self.meta_2_H_bias, padding=0, groups=self.global_dim)

        # residual
        x_1, x_2 = x_1_res + x_1_2, x_2_res + x_2_2

        # =====================Channel Part========================

        x_global = torch.cat((x_1, x_2), dim=1)
        x_ffn = torch.cat((x_global, x_local), dim=1)
        x_ffn = x_ffn + self.ca(self.ffn(x_ffn))

        return x_ffn

# channel wise attention
class CA_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_layer, self).__init__()
        # super().__init__()
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel//reduction),
            nn.Hardswish(),
            nn.Conv2d(channel//reduction, channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channel),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.fc(self.gap(x))
        return x*y.expand_as(x)

# Convnext Blocks (default weight init)
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        # super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x