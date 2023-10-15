import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

# 全局字典，用于存储从模型中提取的注意力权重。
attention = {}

# 定义一个名为get_attention的函数，它将返回一个hook函数。
def get_attention(name):
    # 内部hook函数，当与某个模块关联时，它会在每次该模块前向传播时被调用。
    def hook(module, input, output):
        # 从输入中提取信息的形状。
        x = input[0]
        B, N, C = x.shape
        # 使用模块的qkv函数计算查询、键和值。然后进行形状变换和置换以得到q、k和v。
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # 计算注意力得分，这是查询和键之间的点积乘以比例因子。
        attn = (q @ k.transpose(-2, -1)) * module.scale

        # 使用softmax函数归一化得分，以得到最终的注意力权重。
        attn = attn.softmax(dim=-1)  # [:,:,1,1:]

        # 将计算出的注意力权重保存到全局字典中。
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        # 定义一个顺序模型，首先是一个线性层，该层的输入特征的数量是原始特征数量的两倍，输出是原始特征数量。
        # 接着是一个GELU激活函数。
        # 提醒以下，全连接层处理的是最后一维
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        # 提取读出令牌（假定它是x的第一个元素），并将其维度扩展，使其与x中从start_index开始的部分具有相同的形状。
        # print(f'===vit.py ProjectReadout original readout.shape: {x[:, 0].shape}')
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        # print(f'===vit.py ProjectReadout readout.shape after expand: {readout.shape}')
        # print(f'===vit.py ProjectReadout x[:, self.start_index :].shape: {x[:, self.start_index :].shape}')
        features = torch.cat((x[:, self.start_index :], readout), -1)
        # print(f'===vit.py ProjectReadout features.shape: {features.shape}')
        # print(f"===vit.py ProjectReadout self.project(features).shape: {self.project(features).shape}")

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def forward_vit(pretrained, x):
    b, c, h, w = x.shape

    glob = pretrained.model.forward_flex(x)

    '''存疑，这个activation到底是个什么东西
        目前猜测是提取出来中间tokens
    '''
    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]
    '''存疑，这个act_postprocess是什么东西
        目前猜测是reassemble
    '''
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )
    '''这里应该是还原成图片的维度'''
    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)
    '''目前猜测这一部分仍然是reassemble'''
    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    # 分割token位置嵌入和grid位置嵌入
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )
    # 获取旧的grid大小
    gs_old = int(math.sqrt(len(posemb_grid)))
    # 重新形状grid位置嵌入以进行插值
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    # 使用双线性插值调整grid位置嵌入的大小
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    # 将调整大小后的grid位置嵌入重新形状为原始形式
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    # 合并token和调整大小后的grid位置嵌入
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    # 获取输入的形状
    b, c, h, w = x.shape
    # 调整位置嵌入的大小以匹配输入图像的大小
    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )
    # 获取批量大小
    B = x.shape[0]
    # 如果patch_embed有backbone属性，则对x进行处理
    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            # 如果backbone输出列表/元组的特征，则获取最后一个特征
            x = x[-1]  # last feature if backbone outputs list/tuple of features
    # 对x进行线性嵌入，并将其展平和转置
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    # 如果模型有一个'dist_token'，则将cls_token和dist_token与x拼接
    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        # 否则，只拼接cls_token
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
    # 添加位置嵌入
    x = x + pos_embed
    # 应用位置dropout
    x = self.pos_drop(x)
    # 将x传递给所有transformer块
    for blk in self.blocks:
        x = blk(x)
    # 应用层归一化
    x = self.norm(x)

    return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    # 如果指定使用"ignore"读出方式，创建一个包含多个'Slice'操作的列表。
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    # 如果指定使用"add"读出方式，创建一个包含多个'AddReadout'操作的列表。
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    # 如果指定使用"project"读出方式，为每一个特征创建一个'ProjectReadout'操作。
    # 这里假设ProjectReadout可能需要ViT的特征数(vit_features)和起始索引。
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

# 定义_make_vit_b16_backbone函数，它接受一系列参数，包括输入模型、期望的特征数量、图像大小、要挂钩的层、ViT的特征数量、如何使用readout等。
def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    # 创建一个新的nn.Module对象，名为pretrained。
    pretrained = nn.Module()
    # 将传入的模型赋值给pretrained的model属性。
    pretrained.model = model
    # 使用register_forward_hook收集给定模型中特定层的输出激活。
    # get_activation函数未在此代码段中给出，但显然它返回一个可用于注册钩子的函数。
    '''存疑，什么是register_forward_hook
        注册前向传播中的hook，必须在前向被调用之前注册好，比如可以用来获取某一个中间层的输出
    '''
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    # 将一个名为activations的变量（可能是一个字典或列表）赋值给pretrained。
    '''存疑，什么是nn.Module.activation'''
    pretrained.activations = activations

    # 如果启用了注意力钩子，就获取特定层的注意力权重：
    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    # 调用get_readout_oper函数来获取readout计算子，即定义readout计算子。
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    # 定义第一个后处理序列
    pretrained.act_postprocess1 = nn.Sequential(
        # 使用第一个读出操作
        readout_oper[0],
        # 交换第2维和第3维
        Transpose(1, 2),
        # 将第三维展开为[size[0] // 16, size[1] // 16]
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        # 使用1x1的卷积核，从`vit_features`降维到`features[0]`
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        # 使用转置卷积进行上采样，将特征映射到4倍的空间分辨率
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )
    # 定义第二个后处理序列，与第一个类似，但是参数和大小略有不同
    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        # 使用转置卷积进行上采样，将特征映射到2倍的空间分辨率
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )
    # 定义第三个后处理序列
    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        # 使用1x1的卷积核，从`vit_features`降维到`features[2]`
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )
    # 定义第四个后处理序列
    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        # 用于降低分辨率
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )
    # 设置预训练模型的起始索引和块大小
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    # 将函数注入到VisionTransformer实例中，从而可以与插值位置嵌入一起使用，而不修改库源代码。
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_vit_b_rn50_backbone(
    model,
    features=[256, 512, 768, 768],
    size=[384, 384],
    hooks=[0, 1, 8, 11],
    vit_features=768,
    use_vit_only=False,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    # 创建一个新的模块
    pretrained = nn.Module()
    # 将输入的模型赋给这个新模块的属性
    pretrained.model = model
    # 根据是否仅使用ViT来决定要注册哪些前向钩子
    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(
            get_activation("1")
        )
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(
            get_activation("2")
        )
    # 注册其他的前向钩子，以获取模型的激活
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
    # 如果启用注意力钩子，那么我们还要为模型的注意力层注册钩子
    if enable_attention_hooks:
        pretrained.model.blocks[2].attn.register_forward_hook(get_attention("attn_1"))
        pretrained.model.blocks[5].attn.register_forward_hook(get_attention("attn_2"))
        pretrained.model.blocks[8].attn.register_forward_hook(get_attention("attn_3"))
        pretrained.model.blocks[11].attn.register_forward_hook(get_attention("attn_4"))
        pretrained.attention = attention
    # 存储激活
    pretrained.activations = activations
    # 根据提供的参数获取适当的readout操作
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    # 如果设置仅使用Vision Transformer (ViT)的部分
    if use_vit_only == True:
        # 定义处理激活的第一部分，其中包括一个readout操作，一个转置操作，
        # 一个Unflatten操作，一个1x1卷积，以及一个转置卷积（上采样）
        pretrained.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # 翻四倍
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        # 处理激活的第二部分，与上面类似，但使用不同的参数
        pretrained.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # 翻两倍
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
    else:
        # 如果不仅使用ViT，将处理激活的第一部分和第二部分设置为简单的身份操作。
        pretrained.act_postprocess1 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
        pretrained.act_postprocess2 = nn.Sequential(
            nn.Identity(), nn.Identity(), nn.Identity()
        )
    # 定义激活的第三部分处理，包含readout操作、转置操作、Unflatten操作和一个1x1卷积。
    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )
    # 定义激活的第四部分处理，与第三部分类似，但添加了另一个3x3卷积层。
    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )
    # 为模型设置起始索引
    pretrained.model.start_index = start_index
    # 设置模型的patch大小为[16, 16]
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    # 往VisionTransformer模型中注入forward_flex函数。这样我们就可以在不修改库源代码的情况下使用具有插值位置嵌入的模型。
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    # 往VisionTransformer模型中注入_resize_pos_embed函数。这样我们可以在不修改库源代码的情况下使用带有插值位置嵌入的模型。
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitb_rn50_384(
    pretrained,
    use_readout="ignore",
    hooks=None,
    use_vit_only=False,
    enable_attention_hooks=False,
):
    # 使用timm库来创建一个预训练模型：ViT结合ResNet50的hybrid模型
    # 'pretrained'参数决定是否加载预训练的权重
    model = timm.create_model("vit_base_resnet50_384", pretrained=pretrained)
    # 如果没有提供特定的钩子，则默认为[0, 1, 8, 11]。这些钩子的索引决定了从模型中提取特征的层。
    hooks = [0, 1, 8, 11] if hooks == None else hooks
    # 调用另一个函数来构建真正的backbone，使用指定的特征、图像大小、钩子等参数。
    return _make_vit_b_rn50_backbone(
        model,
        features=[256, 512, 768, 768],
        size=[384, 384],
        hooks=hooks,
        use_vit_only=use_vit_only,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )

# 定义_make_pretrained_vitl16_384函数，该函数具有以下参数：
# pretrained: 是否加载预训练权重。
# use_readout: 如何使用ViT模型的readout部分（最后的分类头部分）。
# hooks: 指定在哪些ViT层后添加钩子（通常用于中间层特征提取）。
# enable_attention_hooks: 是否在注意力机制部分启用钩子。
def _make_pretrained_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    # 使用timm库创建一个预训练的vit_large_patch16_384模型。
    # timm库提供了许多现代深度学习模型的实现，并且可以方便地加载预训练权重。
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    # 如果没有提供钩子，则为特定层设置默认钩子。
    hooks = [5, 11, 17, 23] if hooks == None else hooks

    # 使用_make_vit_b16_backbone函数进一步处理模型，
    # 调整模型以使其作为一个backbone，提取特定的中间层特征。
    # 函数返回处理后的模型。
    return _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_vitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    # 使用timm库创建vit_base_patch16_384模型
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained)
    # 如果未提供钩子，使用默认钩子
    hooks = [2, 5, 8, 11] if hooks == None else hooks
    # 调用另一个函数来进一步配置模型，并返回配置后的模型
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model("vit_deit_base_patch16_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )


def _make_pretrained_deitb16_distil_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    model = timm.create_model(
        "vit_deit_base_distilled_patch16_384", pretrained=pretrained
    )

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    return _make_vit_b16_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
        use_readout=use_readout,
        start_index=2,
        enable_attention_hooks=enable_attention_hooks,
    )
