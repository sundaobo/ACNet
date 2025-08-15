import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from einops import rearrange
from .WF import WF
from .dat import DAT


class Classifier(nn.Module):
    def __init__(self, in_chan=128, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
    

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        #self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        #x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class LGFE0(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        assert dim % heads == 0, f"维度 {dim} 必须能被头数 {heads} 整除"
        self.head_dim = dim // heads
        
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv(x)                             # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.heads, C // self.heads, H, W)
        qkv = qkv.permute(0, 2, 1, 3, 4, 5)            # [B, heads, 3, C/heads, H, W]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # 每个 [B, heads, C/heads, H, W]

        q_flat = q.flatten(3)                          # [B, heads, C/heads, H*W]
        k_flat = k.flatten(3)
        v_flat = v.flatten(3)

        attn = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)                 # [B, heads, H*W, H*W]

        global_out = torch.matmul(attn, v_flat)        # [B, heads, C/heads, H*W]
        global_out = global_out.reshape(B, C, H, W)    # [B, C, H, W]


        return global_out

class EncoderBlock0(nn.Module):
    """
    dim: number of channels of input features
    """
    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.attn = LGFE0(dim, heads=heads)
        
    def forward(self, x):
        # B, C, H, W = x.shape
        inp_copy = x
              
        x = self.layer_norm1(inp_copy)
        x = self.attn(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        out = out + x
        return out

class LGFE(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        assert dim % heads == 0, f"维度 {dim} 必须能被头数 {heads} 整除"
        self.head_dim = dim // heads
        
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1)
        self.qkv1 = nn.Conv2d(dim, 3 * dim, kernel_size=1)

        self.fusion = nn.Conv2d(2*dim, dim, kernel_size=1)

    def forward(self, x, x1):
        B, C, H, W = x.shape

        qkv = self.qkv(x)                             # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.heads, C // self.heads, H, W)
        qkv = qkv.permute(0, 2, 1, 3, 4, 5)            # [B, heads, 3, C/heads, H, W]
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # 每个 [B, heads, C/heads, H, W]

        
        q_flat = q.flatten(3)                          # [B, heads, C/heads, H*W]
        k_flat = k.flatten(3)
        v_flat = v.flatten(3)

        qkv1 = self.qkv1(x1)                             # [B, 3*C, H, W]
        qkv1 = qkv1.reshape(B, 3, self.heads, C // self.heads, H, W)
        qkv1 = qkv1.permute(0, 2, 1, 3, 4, 5)            # [B, heads, 3, C/heads, H, W]
        q1, k1, v1 = qkv1[:, :, 0], qkv1[:, :, 1], qkv1[:, :, 2]  # 每个 [B, heads, C/heads, H, W]

        
        q_flat1 = q1.flatten(3)                          # [B, heads, C/heads, H*W]
        k_flat1 = k1.flatten(3)
        v_flat1 = v1.flatten(3)

        attn1 = torch.matmul(q_flat1, k_flat.transpose(-2, -1)) * self.scale
        attn1 = F.softmax(attn1, dim=-1)                 # [B, heads, H*W, H*W]

        global_out = torch.matmul(attn1, v_flat)        # [B, heads, C/heads, H*W]
        global_out = global_out.reshape(B, C, H, W)    # [B, C, H, W]

        attn = torch.matmul(q_flat, k_flat1.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)                 # [B, heads, H*W, H*W]
        
        global_out1 = torch.matmul(attn, v_flat1)        # [B, heads, C/heads, H*W]
        global_out1 = global_out1.reshape(B, C, H, W)    # [B, C, H, W]

        out = self.fusion(torch.cat([global_out, global_out1], dim=1))  # [B, C, H, W]
        
        return out

class EncoderBlock(nn.Module):
    """
    dim: number of channels of input features
    """
    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm3 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.attn = LGFE(dim, heads=heads)
        
    def forward(self, x, x1):
        # B, C, H, W = x.shape
        inp_copy = x
        inp_copy2 = x1

        x = self.layer_norm1(inp_copy)
        x1 = self.layer_norm2(inp_copy2)
        x = self.attn(x, x1)
        x = self.layer_norm3(x)
        x = self.mlp(x)

        return x

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
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

class ACNet(nn.Module):
    def __init__(self,  backbone='resnet18', output_stride=16, img_size = 512, img_chan=3, chan_num = 32, n_class =2, f_c = 64):
        super(ACNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan)
        
        self.backboneT = DAT()  # [96, 192, 384, 768]
        path = '../pretrain/dat_tiny_in1k_224.pth'
        save_model = torch.load(path)
        model_dict = self.backboneT.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backboneT.load_state_dict(model_dict)

        self.LGFE1 = EncoderBlock(dim=32, mlp_ratio=4)
        self.LGFE2 = EncoderBlock(dim=32, mlp_ratio=4)
        self.LGFE3 = EncoderBlock(dim=32, mlp_ratio=4)
        self.LGFE4 = EncoderBlock0(dim=32, mlp_ratio=4)


        self.down1 = ConvBnLeakyRelu2d(768, 96)
        self.down2 = ConvBnLeakyRelu2d(384, 96)
        self.down3 = ConvBnLeakyRelu2d(192, 96)


        self.dat4 = ConvBnLeakyRelu2d(128, 32)
        self.dat3 = ConvBnLeakyRelu2d(128, 32)
        self.dat2 = ConvBnLeakyRelu2d(128, 32)
        self.dat1 = ConvBnLeakyRelu2d(128, 32)

        self.datdown4 = ConvBnLeakyRelu2d(96, 32)
        self.datdown3 = ConvBnLeakyRelu2d(96, 32)
        self.datdown2 = ConvBnLeakyRelu2d(96, 32)
        self.datdown1 = ConvBnLeakyRelu2d(96, 32)

        self.mlp32 = MLP(dim=32, mlp_ratio=4)
        self.mlp16 = MLP(dim=32, mlp_ratio=4)
        self.mlp8 = MLP(dim=32, mlp_ratio=4)
        self.mlp4 = MLP(dim=32, mlp_ratio=4)
        self.act = nn.GELU()
        self.Fout1_s32 = WF(in_ch=32, out_ch=32)
        self.Fout1_s16 = WF(in_ch=32, out_ch=32)
        self.Fout1_s8 = WF(in_ch=32, out_ch=32)
        self.Fout1_s4 = WF(in_ch=32, out_ch=32)

        self.decode32 = ConvBnLeakyRelu2d(32, 256)
        self.classifier0 = Classifier(n_class = n_class)

    def forward(self, img1, img2):
        out1_s32, out1_s16, out1_s8, out1_s4 = self.backbone(img1) 

        out2_s32, out2_s16, out2_s8, out2_s4 = self.backbone(img2)

        out1_s32T, out1_s16T, out1_s8T, out1_s4T = self.backboneT(img1) 
        out2_s32T, out2_s16T, out2_s8T, out2_s4T = self.backboneT(img2)

        out1_s32T = self.down1(out1_s32T)
        out2_s32T = self.down1(out2_s32T)
        out1_s16T = self.down2(out1_s16T)
        out2_s16T = self.down2(out2_s16T)
        out1_s8T = self.down3(out1_s8T)
        out2_s8T = self.down3(out2_s8T)

        out1_s32 = torch.cat([out1_s32, out1_s32T], dim=1)
        out2_s32 = torch.cat([out2_s32, out2_s32T], dim=1)
        out1_s32 = self.dat4(out1_s32)
        out2_s32 = self.dat4(out2_s32)

        out1_s16 = torch.cat([out1_s16, out1_s16T], dim=1)
        out2_s16 = torch.cat([out2_s16, out2_s16T], dim=1)
        out1_s16 = self.dat3(out1_s16)
        out2_s16 = self.dat3(out2_s16)

        out1_s8 = torch.cat([out1_s8, out1_s8T], dim=1)
        out2_s8 = torch.cat([out2_s8, out2_s8T], dim=1)
        out1_s8 = self.dat2(out1_s8)
        out2_s8 = self.dat2(out2_s8)

        out1_s4 = torch.cat([out1_s4, out1_s4T], dim=1)
        out2_s4 = torch.cat([out2_s4, out2_s4T], dim=1)
        out1_s4 = self.dat1(out1_s4)
        out2_s4 = self.dat1(out2_s4)

        F1s32, loss321 = self.Fout1_s32(out1_s32) #torch.Size([4, 32, 16, 16])
        F2s32, loss322 = self.Fout1_s32(out2_s32)
        loss32 = loss321 + loss322

        F1s32, loss321 = self.Fout1_s32(out1_s32) #torch.Size([4, 32, 16, 16])
        F2s32, loss322 = self.Fout1_s32(out2_s32)
        loss32 = loss321 + loss322

        diff_dat32 = torch.abs(F2s32 - F1s32)
        F1s32 = F1s32 + out1_s32
        F2s32 = F2s32 + out2_s32
        cats32 = torch.cat([F1s32, F2s32, diff_dat32], dim=1)
        cats32 = self.datdown4(cats32)
        cats32 = self.LGFE4(cats32)
        cats32 = F.interpolate(cats32, scale_factor=2, mode='bicubic', align_corners=True)
        cats32 = self.mlp32(cats32)

        F1s16, loss161 = self.Fout1_s16(out1_s16) #torch.Size([4, 32, 32, 32])
        F2s16, loss162 = self.Fout1_s16(out2_s16)
        loss16 = loss161 + loss162

        diff_dat16 = torch.abs(F2s16 - F1s16)
        F1s16 = F1s16 + out1_s16
        F2s16 = F2s16 + out2_s16
        cats16 = torch.cat([F1s16, F2s16, diff_dat16], dim=1)
        cats16 = self.datdown3(cats16)
        cats16 = self.LGFE3(cats16, cats32)

        cats16 = F.interpolate(cats16, scale_factor=2, mode='bicubic', align_corners=True)
        cats16 = self.mlp16(cats16)

        F1s8, loss81 = self.Fout1_s8(out1_s8) 
        F2s8, loss82 = self.Fout1_s8(out2_s8)
        loss8 = loss81 + loss82
        
        diff_dat8 = torch.abs(F2s8 - F1s8)
        F1s8 = F1s8 + out1_s8
        F2s8 = F2s8 + out2_s8
        cats8 = torch.cat([F1s8, F2s8, diff_dat8], dim=1)
        cats8 = self.datdown2(cats8)
        cats8 = self.LGFE2(cats8, cats16)
        
        cats8 = F.interpolate(cats8, scale_factor=2, mode='bicubic', align_corners=True)
        cats8 = self.mlp8(cats8)
    
        F1s4, loss41 = self.Fout1_s4(out1_s4) 
        F2s4, loss42 = self.Fout1_s4(out2_s4)
        loss4 = loss41 + loss42

        diff_dat4 = torch.abs(F2s4 - F1s4)
        F1s4 = F1s4 + out1_s4
        F2s4 = F2s4 + out2_s4
        cats4 = torch.cat([F1s4, F2s4, diff_dat4], dim=1)
        cats4 = self.datdown1(cats4)
        cats4 = self.LGFE1(cats4, cats8)

        cats4 = self.mlp4(cats4)

        x = F.interpolate(cats4, size=img1.shape[2:], mode='bicubic', align_corners=True)
        x = self.decode32(x)

        x = self.classifier0(x)                                                     

        loss = loss32 + loss16 + loss8 + loss4
        return x, loss / 4


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def advanced_visualization(self, tensor, name, iteration, save_dir='./viz/'):
        import os
        import cv2
        import numpy as np
        from torchvision.utils import make_grid
        
        os.makedirs(save_dir, exist_ok=True)
        
        feature = tensor[0].detach().cpu()  # [C, H, W]
        
        
        num_channels = feature.shape[0]
        feature_sum = torch.sum(feature, dim=0)  # [H, W]
        feature_mean = feature_sum / num_channels
        
        
        feature_norm = (feature_mean - feature_mean.min()) / (feature_mean.max() - feature_mean.min() + 1e-8)
        heatmap = (feature_norm.numpy() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        
        cv2.imwrite(os.path.join(save_dir, f'{name}_iter{iteration}_heatmap.jpg'), heatmap)
        grid = make_grid(feature.unsqueeze(1), nrow=8, normalize=True, padding=2)
        grid_np = grid.numpy().transpose((1, 2, 0)) * 255
        cv2.imwrite(os.path.join(save_dir, f'{name}_iter{iteration}_grid.jpg'), grid_np)
