import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
from .miniViT import mViT

import numpy as np

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class DPTWithAdaBins(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        n_bins=100,
        min_val=0.1,
        max_val=10,
    ):

        super(DPTWithAdaBins, self).__init__()

        self.channels_last = channels_last
        self.min_val = min_val
        self.max_val = max_val

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        # 定义encoder和reassemble模块
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )
        # 定义fusion模块
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        '''如下三个层都是新加的'''
        # 临时增加一个conv1x1
        self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)

        # 此处会将head换成AdaBins
        # self.scratch.output_conv = head
        self.adaptive_bins_layer = head
        # self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
        #                               Interpolate(scale_factor=2, mode='bilinear', align_corners=True), # 尝试在这里加入一个双线性插值
        #                               nn.Softmax(dim=1))  # 举例：如果输入张量形状为[2, 100, 240, 320]，这里使得每一个批次每一个像素点的100个值之和等于1
        # 修改了256， n_bins=256
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      # Interpolate(scale_factor=2, mode='bilinear', align_corners=True),
                                      # 尝试在这里加入一个双线性插值,暂时去掉
                                      nn.Softmax(dim=1))  # 举例：如果输入张量形状为[2, 100, 240, 320]，这里使得每一个批次每一个像素点的100个值之和等于1


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        '''这一步应该是包括了encoder和reassemble'''
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        '''这部分应该依旧是reassemble部分'''
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # 上采样fusion block
        '''这部分是fusion block部分，开始上采样'''
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        '''这部分是上采样融合完成，开始将特征图转化成深度图'''
        '''此处head用的是AdaBins，所以要做如下修改'''
        # print(f"models_with_adabins.py DPT forward() tensor shape before AdaBins Block: {path_1.shape}")
        # out = self.scratch.output_conv(path_1)
        # 这里做了一个临时改动，因为path_1输出特征维度为256，后续一系列操作在AdaBins当中都是128，所以暂时用了一个1x1卷积将维度降到128
        path_1 = self.conv1x1(path_1)
        # bin_widths_normed, range_attention_maps = self.scratch.output_conv(path_1)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(path_1)
        # print(f"models_with_adabins.py DPT forward() range_attention_maps.shape: {range_attention_maps.shape}")
        out = self.conv_out(range_attention_maps)
        # print(f"models_with_adabins.py DPT forward() out.shape: {out.shape}")

        bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        # print(f"models_with_adabins.py DPT forward() pred.shape: {pred.shape}")

        return bin_edges, pred

class DPTDepthModelWithAdaBins(DPTWithAdaBins):
    """
        dpt_hybrid:
            non_negative = True
            enable_attention_hooks = False
            features = 256
    """
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, n_bins=100, norm='linear', min_val=0.1, max_val=10,**kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256 # dh: 256
        # print(f'models.py DPTDepthModel features: {features}')
        self.scale = scale # dh: 1.0;
        self.shift = shift # dh: 0.0;
        self.invert = invert # dh: False;
        self.min_val = min_val
        self.max_val = max_val
        # 形状从[N, features, H, W] 变成[N, 1, 2H, 2W]
        # nn.Identity()不对输入做任何操作，只是单纯的恒等映射
        # head = nn.Sequential(
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True) if non_negative else nn.Identity(),
        #     nn.Identity(),
        # )
        # AdaBins论文中decoder输出特征图特征维是128，所以in_channels=128, n_query_channels=128
        # 此处decoder输出特征维为256，所以改成256，embedding_dim也改成256，模型的regressor可能需要改
        # head = mViT(in_channels=128, n_query_channels=128, patch_size=16,
        #             dim_out=n_bins,
        #             embedding_dim=128, norm=norm)
        # 修改了dim_out
        head = mViT(in_channels=128, n_query_channels=128, patch_size=16,
                    dim_out=n_bins,
                    embedding_dim=128, norm=norm)

        super().__init__(head, n_bins=n_bins, min_val=min_val, max_val=max_val, **kwargs)

        if path is not None:
            self.load(path)
    def forward(self, x):
        # inv_depth = super().forward(x).squeeze(dim=1)
        bin_edges, pred = super().forward(x)

        # return centers, final
        return bin_edges, pred

if __name__ == '__main__':
    model_path = None
    model = DPTDepthModelWithAdaBins(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    # input = torch.ones(1, 3, 384, 512)
    # bin_edges, pred = model(input)
    # print(pred.shape)