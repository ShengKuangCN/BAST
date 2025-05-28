import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import numpy as np
from timm.models.swin_transformer import SwinTransformerBlock
# from vision_mamba.model import VisionEncoderMambaBlock
from mambavision.models.mamba_vision import MambaVisionLayer


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


# --- Variant-switchable BAST model ---
class BAST_Variant(nn.Module):
    def __init__(self, *,
                 image_size,  # e.g., (129, 61)
                 patch_size,  # e.g., 16
                 patch_overlap,  # e.g., 10
                 num_classes,  # e.g., 2
                 dim,  # embedding dimension, e.g., 1024 or lower
                 depth,  # transformer depth, e.g., 3
                 heads,  # number of attention heads, e.g., 16
                 mlp_dim,  # MLP dimension, e.g., 1024
                 pool='mean',  # pooling method: 'mean', 'conv', 'linear'
                 channels=2,  # input channels (stereo: left/right)
                 dim_head=64,
                 dropout=0.2,
                 emb_dropout=0.,
                 binaural_integration='SUB',  # 'SUB', 'ADD', or 'CONCAT'
                 share_params=False,
                 transformer_variant='vanilla'  # choose between 'vanilla', 'swin' and 'mamba'
                 ):
        super().__init__()
        self.pool = pool
        self.binaural_integration = binaural_integration
        self.share_params = share_params
        self.transformer_variant = transformer_variant

        # --- Compute patch grid dimensions and padding ---
        image_height, image_width = image_size
        patch_height = patch_width = patch_size
        if patch_overlap != 0:
            num_patches_height = int(np.ceil((image_height - patch_height) / (patch_height - patch_overlap))) + 1
            num_patches_width = int(np.ceil((image_width - patch_width) / (patch_width - patch_overlap))) + 1
            padding_height = (num_patches_height - 1) * (patch_height - patch_overlap) + patch_height - image_height
            padding_width = (num_patches_width - 1) * (patch_width - patch_overlap) + patch_width - image_width
        else:
            num_patches_height = int(np.ceil(image_height / patch_height))
            num_patches_width = int(np.ceil(image_width / patch_width))
            padding_height = num_patches_height * patch_height - image_height
            padding_width = num_patches_width * patch_width - image_width

        self.num_patches_height = num_patches_height
        self.num_patches_width = num_patches_width
        self.num_patches = num_patches_height * num_patches_width
        patch_dim = 1 * patch_height * patch_width  # each branch has 1 channel

        # --- Shared Patch Embedding (for both variants) ---
        self.to_patch_embedding = nn.Sequential(
            nn.ReflectionPad2d((0, padding_width, padding_height, 0)),  # (left, right, top, bottom)
            nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_height - patch_overlap),
            Rearrange('b (c k1 k2) n -> b n (k1 k2 c)', k1=patch_height, k2=patch_width, n=self.num_patches),
            nn.Linear(patch_dim, dim)
        )

        if transformer_variant == 'vanilla':
            # --- Vanilla branch: add CLS token and positional embeddings ---
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
            self.dropout = nn.Dropout(emb_dropout)
            self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            if not share_params:
                self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            integration_dim = dim if binaural_integration != 'CONCAT' else dim * 2
            self.transformer3 = Transformer(integration_dim, depth, heads, dim_head, mlp_dim, dropout)
        elif transformer_variant == 'swin':
            swin_patch_size = 7
            # --- Swin branch: reshape tokens into grid for Swin blocks ---
            input_resolution = (self.num_patches_height, self.num_patches_width)
            self.transformer1 = nn.Sequential(*[
                SwinTransformerBlock(dim=dim,
                                     input_resolution=input_resolution,
                                     num_heads=heads,
                                     window_size=swin_patch_size,  # you can experiment with this
                                     shift_size=0,
                                     mlp_ratio=mlp_dim / dim,
                                     proj_drop=dropout,
                                     attn_drop=dropout,
                                     drop_path=0.0,
                                     norm_layer=nn.LayerNorm)
                for _ in range(depth)
            ])
            if not share_params:
                self.transformer2 = nn.Sequential(*[
                    SwinTransformerBlock(dim=dim,
                                         input_resolution=input_resolution,
                                         num_heads=heads,
                                         window_size=swin_patch_size,
                                         shift_size=0,
                                         mlp_ratio=mlp_dim / dim,
                                         proj_drop=dropout,
                                         attn_drop=dropout,
                                         drop_path=0.0,
                                         norm_layer=nn.LayerNorm)
                    for _ in range(depth)
                ])
            integration_dim = dim if binaural_integration != 'CONCAT' else dim * 2
            self.transformer3 = nn.Sequential(*[
                SwinTransformerBlock(dim=integration_dim,
                                     input_resolution=input_resolution,
                                     num_heads=heads,
                                     window_size=swin_patch_size,
                                     shift_size=0,
                                     mlp_ratio=mlp_dim / integration_dim,
                                     proj_drop=dropout,
                                     attn_drop=dropout,
                                     drop_path=0.0,
                                     norm_layer=nn.LayerNorm)
                for _ in range(depth)
            ])
        elif transformer_variant == 'vim':
            # For fair comparison, we use the patch embedding from BAST and then feed the resulting tokens
            # to a stack of VisionEncoderMambaBlock blocks.
            # Note: Here we fix dt_rank to 32, use mlp_dim as dim_inner, and set d_state equal to dim.
            self.transformer1 = nn.Sequential(*[
                VisionEncoderMambaBlock(dim=dim, dt_rank=16, dim_inner=mlp_dim, d_state=int(dim/4))
                for _ in range(depth)
            ])
            if not share_params:
                self.transformer2 = nn.Sequential(*[
                    VisionEncoderMambaBlock(dim=dim, dt_rank=16, dim_inner=mlp_dim, d_state=int(dim/4))
                    for _ in range(depth)
                ])
            integration_dim = dim if binaural_integration != 'CONCAT' else dim * 2
            self.transformer3 = nn.Sequential(*[
                VisionEncoderMambaBlock(dim=integration_dim, dt_rank=16, dim_inner=mlp_dim, d_state=int(dim/4))
                for _ in range(depth)
            ])
        elif transformer_variant == 'mamba':
            # Here, we directly use a MambaVisionLayer (or a stack of them)
            # Note: MambaVisionLayer expects a 2D feature map, so we will reshape the tokens accordingly.
            # We set conv=False and downsample=False to keep the spatial resolution.
            self.transformer1 = MambaVisionLayer(dim=dim, depth=3, num_heads=heads, window_size=8, conv=False,
                                                 downsample=False, mlp_ratio=int(mlp_dim/dim))

            if not share_params:
                self.transformer2 = MambaVisionLayer(dim=dim, depth=3, num_heads=heads, window_size=8, conv=False,
                                                     downsample=False, mlp_ratio=int(mlp_dim/dim))

            integration_dim = dim if binaural_integration != 'CONCAT' else dim * 2
            self.transformer3 = MambaVisionLayer(dim=integration_dim, depth=3, num_heads=heads, window_size=8,
                                                 conv=False, downsample=False, mlp_ratio=int(mlp_dim/dim))

        else:
            raise ValueError("Unknown transformer_variant. Choose 'vanilla' 'swin' or 'mamba'.")

        # --- Optional Pooling ---
        if pool == 'conv':
            self.patch_pooling = nn.Sequential(
                nn.Conv1d(self.num_patches if transformer_variant == 'vanilla' else (
                            self.num_patches_height * self.num_patches_width), 1, 1),
                nn.GELU()
            )
        elif pool == 'linear':
            self.patch_pooling = nn.Sequential(
                nn.Linear(self.num_patches if transformer_variant == 'vanilla' else (
                            self.num_patches_height * self.num_patches_width), 1),
                nn.GELU()
            )

        # --- Final Classification Head ---
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(integration_dim),
            nn.Linear(integration_dim, num_classes),
        )

    def process_branch(self, img_branch, transformer_module):
        # Shared patch embedding: output shape [B, N, dim]
        x = self.to_patch_embedding(img_branch)
        if self.transformer_variant == 'vanilla':
            b, n, d = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embedding[:, :x.shape[1]]
            x = self.dropout(x)
        elif self.transformer_variant == 'swin':
            x = rearrange(x, 'b (h w) d -> b h w d', h=self.num_patches_height, w=self.num_patches_width)
        elif self.transformer_variant == 'mamba':
            # Reshape sequence tokens to 2D feature map: [B, N, dim] -> [B, dim, H, W]
            x = rearrange(x, 'b (h w) d -> b d h w', h=self.num_patches_height, w=self.num_patches_width)
        else:
            raise ValueError("Unknown transformer variant in process_branch.")

        # Apply the corresponding transformer module
        x = transformer_module(x)
        # For mamba, if the module returns a 2D map, flatten back to sequence
        # if self.transformer_variant == 'mamba':
        #     x = rearrange(x, 'b d h w -> b (h w) d')
        return x

    def forward(self, img):
        # Assume input img shape: [B, channels, H, W] with channels = 2 (stereo)
        img_l = img[:, 0:1, :, :]
        img_r = img[:, 1:2, :, :]
        x_l = self.process_branch(img_l, self.transformer1)
        if self.share_params:
            x_r = self.process_branch(img_r, self.transformer1)
        else:
            x_r = self.process_branch(img_r, self.transformer2)
        # Binaural integration
        if self.binaural_integration == 'ADD':
            x = x_l + x_r
        elif self.binaural_integration == 'SUB':
            x = x_l - x_r
        elif self.binaural_integration == 'CONCAT' and self.transformer_variant != 'mamba':
            x = torch.cat((x_l, x_r), dim=-1)
        elif self.binaural_integration == 'CONCAT' and self.transformer_variant == 'mamba':
            x = torch.cat((x_l, x_r), dim=1)
        else:
            raise ValueError("Unsupported binaural_integration option.")

        # Integration transformer stage
        x = self.transformer3(x)

        # Pooling and classification head
        if self.pool == 'mean':
            if self.transformer_variant == 'vanilla':
                x = x.mean(dim=1)
            elif self.transformer_variant == 'swin':
                x = x.mean(dim=(1, 2))  # average over spatial dimensions for swin
            elif self.transformer_variant == 'mamba':
                x = x.mean(dim=(2, 3))
            x = self.mlp_head(x)
        elif self.pool == 'cls':
            if self.transformer_variant == 'vanilla':
                x = x[:, 0]
                x = self.mlp_head(x)
            else:
                raise ValueError("CLS pooling is not supported for swin transformer variant.")
        elif self.pool == 'conv':
            if self.transformer_variant == 'vanilla':
                x = self.patch_pooling(x[:, 1:])[:, 0, :]
            else:
                x_seq = rearrange(x, 'b h w d -> b (h w) d')
                x = self.patch_pooling(x_seq)[:, 0, :]
            x = self.mlp_head(x)
        elif self.pool == 'linear':
            if self.transformer_variant == 'vanilla':
                x = self.patch_pooling(x[:, 1:].transpose(1, 2)).squeeze(-1)
            else:
                x_seq = rearrange(x, 'b h w d -> b (h w) d')
                x = self.patch_pooling(x_seq.transpose(1, 2)).squeeze(-1)
            x = self.mlp_head(x)
        else:
            raise ValueError("Unsupported pooling type.")

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
    from conf import *
    net = BAST_Variant(
        image_size=SPECTROGRAM_SIZE,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        num_classes=NUM_OUTPUT,
        dim=EMBEDDING_DIM,
        depth=TRANSFORMER_DEPTH,
        heads=TRANSFORMER_HEADS,
        mlp_dim=TRANSFORMER_MLP_DIM,
        pool=TRANSFORMER_POOL,
        channels=INPUT_CHANNEL,
        dim_head=TRANSFORMER_DIM_HEAD,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
        binaural_integration=BINAURAL_INTEGRATION,
        share_params=SHARE_PARAMS,
        transformer_variant='mamba',
    )
    net.cuda()
    from torchsummary import summary
    summary(net, input_size=(2, 129, 61), batch_size=-1)

