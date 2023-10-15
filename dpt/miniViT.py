import torch
import torch.nn as nn

from .layers import PatchTransformerEncoder, PixelWiseDotProduct


class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()
        # 不改变尺寸
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, dim_out))

    def forward(self, x):
        # print(f"miniViT.py forward() input tensor x.shape: {x.shape}")
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E
        # print(f"miniViT.py forward() tgt.shape: {tgt.shape}")
        # 尺寸不变，纯embedding用途
        x = self.conv3x3(x)
        # print(f"miniViT.py forward() x.shape after con3x3: {x.shape}")

        regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]
        # print(f"miniViT.py forward() regression_head.shape: {regression_head.shape}")
        # print(f"miniViT.py forward() queries.shape: {queries.shape}")

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        # print(f"miniViT.py forward() return range_attention_maps.shape: {range_attention_maps.shape}")
        return y, range_attention_maps
