import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Any, Optional, List
from timm.models.layers import trunc_normal_

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class _gcc_conv(nn.Module):
    # only a bank for weight and bias
    def __init__(self, type: str, conv2d: nn.Conv2d):
        super(_gcc_conv, self).__init__()
        self.groups = 1
        self.type = type
        self.weight, self.bias = conv2d.weight, conv2d.bias

class GCC_conv(nn.Module):
    def __init__(self,
        dim,
        type,
        global_kernel_size,
        use_pe = True,
        instance_kernel_method = 'interpolation_bilinear'
    ):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.gcc_conv = _gcc_conv(type, nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=dim))
        self.instance_kernel_method = instance_kernel_method
        if use_pe:
            if self.type=='H':
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
                    # pe_H.expand(1, dim, self.global_kernel_size, self.global_kernel_size).clone()
            elif self.type=='W':
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            trunc_normal_(self.pe, std=.02)
        else:
            self.pe = None

    def get_instance_kernel(self, instance_kernel_size_2):
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.gcc_conv.weight
        elif self.instance_kernel_method == 'interpolation_bilinear':
            instance_kernel_size_2 =  (instance_kernel_size_2[0], 1) if self.type=='H' else (1, instance_kernel_size_2[1])
            return  F.interpolate(self.gcc_conv.weight, instance_kernel_size_2, mode='bilinear', align_corners=True)
    
    def get_instance_pe(self, instance_kernel_size_2):
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.pe
        elif self.instance_kernel_method == 'interpolation_bilinear':
            return  F.interpolate(self.pe, instance_kernel_size_2, mode='bilinear', align_corners=True)\
                        .expand(1, self.dim, *instance_kernel_size_2)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.pe is not None:
            x = x + self.get_instance_pe((H, W))
        weight = self.get_instance_kernel((H, W))
        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type=='H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = F.conv2d(x_cat, weight=weight, bias=self.gcc_conv.bias, padding=0, groups=self.dim)
        return x

class InvertedResidual_gcc(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        global_kernel_size=14,
        use_pe=True
    ) -> None:
        super(InvertedResidual_gcc, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        # pw
        self.pw_expand = ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)

        # gcc
        self.gcc_H = GCC_conv(hidden_dim//2, 'H', global_kernel_size, use_pe)
        self.gcc_W = GCC_conv(hidden_dim//2, 'W', global_kernel_size, use_pe)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU6()

        # pw-linear
        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        res = x

        x = self.pw_expand(x)

        x_H, x_W = torch.chunk(x, 2, dim=1)
        x_H, x_W = self.gcc_H(x_H), self.gcc_W(x_W)
        x = torch.cat((x_H, x_W), dim=1)
        x = self.act(self.bn(x))

        x = self.pw_linear(x)

        if self.use_res_connect:
            return x + res
        else:
            return x

@BACKBONES.register_module()
class _MobileNetV2_GCC(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        head_init_scale=1.,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        out_indices = (1, 2, 4, 7),
        frozen_stages = -1,
        norm_eval=False,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(_MobileNetV2_GCC, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, gcc, r
                [1, 16, 1, 1, 1, 112],
                [6, 24, 2, 2, 2, 56], # out0
                [6, 32, 3, 2, 3, 28], # out1
                [6, 64, 4, 2, 2, 14],
                [6, 96, 3, 1, 2, 14], # out2
                [6, 160, 3, 2, 2, 7],
                [6, 320, 1, 1, 1, 7], # out3
            ]

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 6:
            raise ValueError("inverted_residual_gcc_setting should be non-empty "
                             "or a 6-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.conv1 = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)

        # building inverted residual blocks
        self.layers = []
        for i, (t, c, n, s, gcc, r) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            stage = []
            for j in range(n):
                stride = s if j == 0 else 1
                if j < gcc:
                    stage.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                else:
                    stage.append(InvertedResidual_gcc(input_channel, output_channel, stride, expand_ratio=t,
                                                         norm_layer=norm_layer, global_kernel_size=r,
                                                         use_pe=True if j == gcc else False))
                input_channel = output_channel
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, nn.Sequential(*stage))
            self.layers.append(layer_name)

        # building last several layers
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.add_module('conv2', ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.layers.append('conv2')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(_MobileNetV2_GCC, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def get_model_size(self):
        return sum([p.numel() for p in self.parameters()])
