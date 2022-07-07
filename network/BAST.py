import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import numpy as np


"""
Reference: https://github.com/lucidrains/vit-pytorch
"""


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """

        :param dim:
        :param depth:
        :param heads:
        :param dim_head:
        :param mlp_dim:
        :param dropout:
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class BAST(nn.Module):
    def __init__(self, *, image_size, patch_size, patch_overlap, num_classes, dim, depth, heads, mlp_dim, pool='mean', channels=2,
                 dim_head=64, dropout=0.2, emb_dropout=0., binaural_integration='SUB', polar_output=False, share_params=False):
        """
        :param image_size: the size of spectrogram, default 129*61.
        :param patch_size: default 16.
        :param patch_overlap: the overlap pixels between patches, default 10.
        :param num_classes: the output dimension, default 2 in 2D plane.
        :param dim: the embedding dimension of patches, default 1024.
        :param depth: the depth of each Transformer encoder, default 3.
        :param heads: the number of attention head, default 16.
        :param mlp_dim: the dimension of MLP block, default 1024.
        :param pool: pooling method in the last layer of the network, default 'mean', options: 'mean', 'linear', 'conv'.
        :param channels: the number of input channel on each side, default 1.
        :param dim_head: the dimension of attention layers, default 64.
        :param dropout: the dropout rate in each Transformer encoder, default 0.2.
        :param emb_dropout: the dropout rate in both embedding layers, default 0.
        :param binaural_integration: binaural integration methods, default 'SUB', options: 'SUB', 'ADD', 'CONCAT'.
        :param polar_output: abandoned
        :param share_params: whether the T1 (left) and T2 (right) share parameters, default False.
        """
        super().__init__()
        image_height = image_size[0]
        image_width = image_size[1]
        patch_height = patch_width = patch_size

        if patch_overlap != 0:
            num_patches_height = int(np.ceil((image_height - patch_height) / (patch_height - patch_overlap))) + 1
            num_patches_width = int(np.ceil((image_width - patch_width) / (patch_width - patch_overlap))) + 1

            padding_height = (num_patches_height - 1) * (patch_height - patch_overlap) + patch_height - image_height
            padding_width = (num_patches_width - 1) * (patch_height - patch_overlap) + patch_width - image_width
        else:
            num_patches_height = int(np.ceil(image_height / patch_height))
            num_patches_width = int(np.ceil(image_width / patch_width))

            padding_height = num_patches_height * patch_height - image_height
            padding_width = num_patches_width * patch_width - image_width
        num_patches = num_patches_height * num_patches_width
        patch_dim = channels * patch_height * patch_width
        assert pool in {'mean', 'conv', 'linear'}, 'Illegal pooling type !'

        self.binaural_integration = binaural_integration
        self.polar_output = polar_output
        self.share_params = share_params
        self.to_patch_embedding = nn.Sequential(
            nn.ReflectionPad2d((0, padding_width, padding_height, 0)),  # (left, right, top, bottom)
            nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_height - patch_overlap),  # b, (c k1 k2), (n_ph n_pw)
            Rearrange('b (c k1 k2) (n_ph n_pw) -> b (n_ph n_pw) (k1 k2 c)', k1=patch_height, k2=patch_width, n_ph=num_patches_height),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        if not self.share_params:
            self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        if self.binaural_integration == 'CONCAT':
            self.transformer3 = Transformer(dim * 2, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer3 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        if self.pool == 'conv':
            self.patch_pooling = nn.Sequential(
                nn.Conv1d(num_patches, 1, 1),
                nn.GELU()
            )
        elif self.pool == 'linear':
            self.patch_pooling = nn.Sequential(
                nn.Linear(num_patches, 1),
                nn.GELU()
            )
        self.to_latent = nn.Identity()

        if self.binaural_integration == 'CONCAT' and not self.polar_output:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, num_classes),
            )
        elif self.binaural_integration != 'CONCAT' and not self.polar_output:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes),
            )
        elif self.binaural_integration == 'CONCAT' and self.polar_output:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, num_classes),
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes),
            )

    def forward(self, img):
        img_l = img[:, 0:1, :, :]
        img_r = img[:, 1:, :, :]
        x_l = self.to_patch_embedding(img_l)
        b_l, n_l, _ = x_l.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b_l)
        x_l = torch.cat((cls_tokens, x_l), dim=1)
        x_l += self.pos_embedding[:, :(n_l + 1)]
        x_l = self.dropout(x_l)
        x_l = self.transformer1(x_l)

        x_r = self.to_patch_embedding(img_r)
        b_r, n_r, _ = x_r.shape
        x_r = torch.cat((cls_tokens, x_r), dim=1)
        x_r += self.pos_embedding[:, :(n_r + 1)]
        x_r = self.dropout(x_r)
        if self.share_params:
            x_r = self.transformer1(x_r)
        else:
            x_r = self.transformer2(x_r)
        if self.binaural_integration == 'ADD':
            x = x_l + x_r
        elif self.binaural_integration == 'SUB':
            x = x_l - x_r
        elif self.binaural_integration == 'CONCAT':
            x = torch.cat((x_l, x_r), dim=2)

        x = self.transformer3(x)

        if self.pool == 'mean':
            x = x.mean(dim=1)
            x = self.mlp_head(x)
        elif self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'conv':
            x = self.patch_pooling(x[:, 1:])[:, 0, :]
            x = self.mlp_head(x)
        elif self.pool == 'linear':
            x = self.patch_pooling(x[:, 1:].transpose(1, 2))
            x = self.mlp_head(x[:, :, 0])
        return x


class AngularLossWithCartesianCoordinate(nn.Module):
    def __init__(self):
        super(AngularLossWithCartesianCoordinate, self).__init__()

    def forward(self, x, y):
        x = x / torch.linalg.norm(x, dim=1)[:, None]
        y = y / torch.linalg.norm(y, dim=1)[:, None]
        dot = torch.clamp(torch.sum(x * y, dim=1), min=-0.999, max=0.999)
        loss = torch.mean(torch.acos(dot))
        return loss


class MixWithCartesianCoordinate(nn.Module):
    def __init__(self):
        super(MixWithCartesianCoordinate, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        loss1 = self.mse(x, y)
        x = x / torch.linalg.norm(x, dim=1)[:, None]
        y = y / torch.linalg.norm(y, dim=1)[:, None]
        dot = torch.clamp(torch.sum(x * y, dim=1), min=-0.999, max=0.999)
        loss2 = torch.mean(torch.acos(dot))
        return loss1 + loss2


class AngularLossWithPolarCoordinate(nn.Module):
    def __init__(self):
        super(AngularLossWithPolarCoordinate, self).__init__()

    def forward(self, x, y):
        x1 = x[:, 1]
        y_r = torch.atan2(y[:, 1], y[:, 0])
        diff = torch.abs(x1 - y_r)
        loss = torch.mean(torch.pow(diff, 2))
        return loss


class MSELossWithPolarCoordinate(nn.Module):
    def __init__(self):
        super(MSELossWithPolarCoordinate, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        x_coord_x = (x[:, 0] * torch.cos(x[:, 1])).unsqueeze(1)
        x_coord_y = (x[:, 0] * torch.sin(x[:, 1])).unsqueeze(1)
        x_coord = torch.cat((x_coord_x, x_coord_y), dim=1)
        loss = self.mse(x_coord, y)

        return loss


if __name__ == '__main__':
    v = BAST(
        image_size=[129, 61],
        patch_size=16,
        patch_overlap=10,
        num_classes=2,
        dim=256,
        depth=3,
        heads=4,
        mlp_dim=256,
        pool='mean',
        channels=1,
        dim_head=16,
        dropout=0.2,
        emb_dropout=0.2,
        binaural_integration='SUB',
        polar_output=False,
        share_params=False
    )
    v.cuda()
    from torchsummary import summary
    summary(v, input_size=(2, 129, 61), batch_size=-1)

