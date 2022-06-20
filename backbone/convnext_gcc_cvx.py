import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from .modules.gcc_cvx_modules import gcc_cvx_Block, gcc_cvx_Block_2stage, Block, LayerNorm, gcc_Conv2d

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

@BACKBONES.register_module()
class ConvNeXt_cvx_gcc(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 gcc_stage=1, block_replace_mode="last_1/3"
                 ):
        super().__init__()
        # super(ConvNeXt_cvx_gcc, self).__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        stages_fs = [56, 28, 14, 7]
        for i in range(4):
            if i < 2:   # for stage 0 and 1, no gcc
                stage = nn.Sequential(*[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) \
                    for j in range(depths[i])
                ])
            else:       # for stage 2 and 3, gcc modules is used
                gcc_Block = gcc_cvx_Block_2stage if gcc_stage==2 else gcc_cvx_Block
                if block_replace_mode == "last_1/3": # here we use gcc in the last 1/3 blocks
                    lo = 2*depths[i]//3 # e.g. in stage3, j+1=7 > lo=2*9//3=6, so block 678 is gcc_block, while block 0-5 is normal
                elif block_replace_mode == "last_2/3": # here we use gcc in the last 2/3 blocks
                    lo = depths[i]//3 # e.g. in stage3, j+1=4 > lo=9//3=3, so block 4-8 is gcc_block, while block 0-3 is normal
                stage = nn.Sequential(*[    
                    gcc_Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value,
                        meta_kernel_size=stages_fs[i], instance_kernel_method=None, use_pe=True) \
                    if lo < j+1 else \
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) \
                    for j in range(depths[i])
                ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, gcc_Conv2d):
            m.gcc_init()    # convnext like initialization is used for gcc as well

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, gcc_Conv2d):
                m.gcc_init()    # convnext like initialization is used for gcc as well
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x