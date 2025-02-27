import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
    def forward(self, x):
        """
            torch.Size([1, 40, 224, 224])
            torch.Size([1, 8, 224, 224])
            torch.Size([1, 8, 224, 224])
            torch.Size([1, 8, 224, 224])
        """
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class ModifyPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=[3, 6, 9, 12]):
        super(ModifyPPM, self).__init__()
        self.features = []
        
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim),
            nn.GELU(),
        )
        
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, padding=1, bias=False, groups=reduction_dim),
                nn.GELU()
            ))
        
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        
        return torch.cat(out, 1)

class MultiScaleChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(MultiScaleChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.Conv2d(in_channels // ratio, in_channels // ratio, kernel_size=3, padding=1, groups=in_channels // ratio, bias=False)
        )
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.gelu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.gelu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * torch.sigmoid(out)

class MultiScaleSpatialAttention(nn.Module):
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, groups=1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * torch.sigmoid(out)

class MultiScalePAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(MultiScalePAM, self).__init__()
        self.scale_channels = in_channels // 8
        self.reduction_ratio = reduction_ratio  # 空间降采样比例
        self.ppm = ModifyPPM(in_channels, self.scale_channels)
        
        total_channels = in_channels + in_channels + 4 * self.scale_channels
        self.query_conv = nn.Conv2d(total_channels, self.scale_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(total_channels, self.scale_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(total_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.up = nn.Upsample(scale_factor=reduction_ratio, mode='bilinear', align_corners=True)

    def forward(self, x):
        ms_features = self.ppm(x)
        enhanced_features = torch.cat([x, ms_features], dim=1)
        batch_size, _, height, width = enhanced_features.size()
        h, w = height // self.reduction_ratio, width // self.reduction_ratio
        enhanced_features_down = F.interpolate(enhanced_features, size=(h, w), 
                                             mode='bilinear', align_corners=True)
        
        query = self.query_conv(enhanced_features_down).view(batch_size, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(enhanced_features_down).view(batch_size, -1, h * w)
        value = self.value_conv(enhanced_features_down).view(batch_size, -1, h * w)
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, h, w)
        out = F.interpolate(out, size=(height, width), mode='bilinear', align_corners=True)
        
        return self.gamma * out + x

class META(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(MPCBAM, self).__init__()
        self.ms_channel_attention = MultiScaleChannelAttention(in_channels, ratio)
        self.ms_spatial_attention = MultiScaleSpatialAttention()
        self.ms_position_attention = MultiScalePAM(in_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x_c = self.ms_channel_attention(x)
        x_s = self.ms_spatial_attention(x_c)      
        x_p = self.ms_position_attention(x)
        out = self.fusion(x_s + x_p)
        return out + x
    
class CBG(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size, padding=padding, dilation=dilation, 
                     groups=in_c, bias=False, stride=stride),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.gelu(x)
        return x

class MS_CDFA(nn.Module):
    def __init__(self, in_c, out_c=128, num_heads=4, kernel_size=3, padding=1, stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = out_c
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.attn_fg = nn.Conv2d(dim, kernel_size ** 4 * num_heads, kernel_size=1, bias=True)
        self.attn_bg = nn.Conv2d(dim, kernel_size ** 4 * num_heads, kernel_size=1, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(stride, stride, ceil_mode=True)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.edge_pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.input_cbr = nn.Sequential(
            CBG(in_c, dim, kernel_size=3, padding=1),
            CBG(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBG(dim, dim, kernel_size=3, padding=1),
            CBG(dim, dim, kernel_size=3, padding=1),
        )

    def enhance_edges(self, x):
        edge = self.edge_pool(x)  
        edge = x - edge  
        edge = self.edge_conv(edge)  
        return x + edge  

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)
        
        x = self.enhance_edges(x)

        B, C, H, W = x.shape
        
        v = self.v(x)
        
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                          self.kernel_size * self.kernel_size,
                                          -1).permute(0, 1, 4, 3, 2)
        
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg).reshape(
            B, self.num_heads, self.head_dim,
            self.kernel_size * self.kernel_size,
            -1).permute(0, 1, 4, 3, 2)
        
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        out = self.output_cbr(x_weighted_bg)
        out = self.enhance_edges(out)
        
        return out


    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map)
        attn = attn_layer(feature_map_pooled)
        
        attn = attn.reshape(B, h * w, self.num_heads,
                           self.kernel_size * self.kernel_size,
                           self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        x_weighted = self.proj(x_weighted)
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted

class CDI_UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_c=64):
        super().__init__()
        
        # Residual adapters for encoders
        self.res_adapter_enc1 = nn.Conv2d(in_channels, base_c, kernel_size=1)
        self.res_adapter_enc2 = nn.Conv2d(base_c, base_c*2, kernel_size=1)
        self.res_adapter_enc3 = nn.Conv2d(base_c*2, base_c*4, kernel_size=1)
        self.res_adapter_enc4 = nn.Conv2d(base_c*4, base_c*8, kernel_size=1)
        self.res_adapter_enc5 = nn.Conv2d(base_c*8, base_c*16, kernel_size=1)
        
        # ====== Encoder Blocks with Residual Connections ======
        self.enc1_main = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c),
            META(base_c),
            nn.GELU()
        )
        
        self.enc2_main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_c, base_c*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*2),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*2),
            META(base_c*2),
            nn.GELU()
        )
        
        self.enc3_main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_c*2, base_c*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*4),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*4),
            META(base_c*4),
            nn.GELU()
        )
        
        self.enc4_main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_c*4, base_c*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*8),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*8),
            META(base_c*8),
            nn.GELU()
        )
        
        self.enc5_main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_c*8, base_c*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*16),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*16),
            META(base_c*16),
            nn.GELU()
        )
        
        # Bridge (Encoder 6)
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_c*16, base_c*32, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*32),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*32),
            META(base_c*32),
            nn.GELU()
        )
        
        # ====== Auxiliary Segmentation Heads ======
        self.aux_seg_1 = nn.Conv2d(base_c, num_classes, kernel_size=1)
        self.aux_seg_2 = nn.Conv2d(base_c*2, num_classes, kernel_size=1)
        self.aux_seg_3 = nn.Conv2d(base_c*4, num_classes, kernel_size=1)
        self.aux_seg_4 = nn.Conv2d(base_c*8, num_classes, kernel_size=1)
        self.aux_seg_5 = nn.Conv2d(base_c*16, num_classes, kernel_size=1)
        
        # ====== Decoder Blocks ======
        self.up5 = nn.ConvTranspose2d(base_c*32, base_c*16, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(base_c*32, base_c*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*16),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*16),
            nn.GELU()
        )
        self.cdfa5 = MS_CDFA(base_c*16, out_c=base_c*16)
        
        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_c*16, base_c*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*8),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*8),
            nn.GELU()
        )
        self.cdfa4 = MS_CDFA(base_c*8, out_c=base_c*8)
        
        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_c*8, base_c*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*4),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*4),
            nn.GELU()
        )
        self.cdfa3 = MS_CDFA(base_c*4, out_c=base_c*4)
        
        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_c*4, base_c*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c*2),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c*2),
            nn.GELU()
        )
        self.cdfa2 = MS_CDFA(base_c*2, out_c=base_c*2)
        
        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_c*2, base_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_c),
            nn.Dropout2d(0.15),
            InceptionDWConv2d(base_c),
            nn.GELU()
        )
        self.cdfa1 = MS_CDFA(base_c, out_c=base_c)
        
        # Final Refinement
        self.final = nn.Sequential(
            META(base_c),
            nn.Conv2d(base_c, num_classes, kernel_size=1)
        )
        
        # 添加特征转换层
        self.fg_transform5 = nn.Conv2d(1, base_c*16, kernel_size=1)
        self.bg_transform5 = nn.Conv2d(1, base_c*16, kernel_size=1)
        
        self.fg_transform4 = nn.Conv2d(1, base_c*8, kernel_size=1)
        self.bg_transform4 = nn.Conv2d(1, base_c*8, kernel_size=1)
        
        self.fg_transform3 = nn.Conv2d(1, base_c*4, kernel_size=1)
        self.bg_transform3 = nn.Conv2d(1, base_c*4, kernel_size=1)
        
        self.fg_transform2 = nn.Conv2d(1, base_c*2, kernel_size=1)
        self.bg_transform2 = nn.Conv2d(1, base_c*2, kernel_size=1)
        
        self.fg_transform1 = nn.Conv2d(1, base_c, kernel_size=1)
        self.bg_transform1 = nn.Conv2d(1, base_c, kernel_size=1)

        # Residual adapters for decoders
        self.res_adapter_dec5 = nn.Conv2d(base_c*32, base_c*16, kernel_size=1)  # 32 = 16(up) + 16(skip)
        self.res_adapter_dec4 = nn.Conv2d(base_c*16, base_c*8, kernel_size=1)   # 16 = 8(up) + 8(skip)
        self.res_adapter_dec3 = nn.Conv2d(base_c*8, base_c*4, kernel_size=1)    # 8 = 4(up) + 4(skip)
        self.res_adapter_dec2 = nn.Conv2d(base_c*4, base_c*2, kernel_size=1)    # 4 = 2(up) + 2(skip)
        self.res_adapter_dec1 = nn.Conv2d(base_c*2, base_c, kernel_size=1)      # 2 = 1(up) + 1(skip)

    def forward(self, x):
        # Encoder Path with Residual Connections
        e1 = self.enc1_main(x) + self.res_adapter_enc1(x)
        aux1 = self.aux_seg_1(e1)
        
        e2_input = e1
        e2 = self.enc2_main(e1) + F.max_pool2d(self.res_adapter_enc2(e2_input), 2)
        aux2 = self.aux_seg_2(e2)
        
        e3_input = e2
        e3 = self.enc3_main(e2) + F.max_pool2d(self.res_adapter_enc3(e3_input), 2)
        aux3 = self.aux_seg_3(e3)
        
        e4_input = e3
        e4 = self.enc4_main(e3) + F.max_pool2d(self.res_adapter_enc4(e4_input), 2)
        aux4 = self.aux_seg_4(e4)
        
        e5_input = e4
        e5 = self.enc5_main(e4) + F.max_pool2d(self.res_adapter_enc5(e5_input), 2)
        aux5 = self.aux_seg_5(e5)
        
        # Bridge
        bridge = self.bridge(e5)
        
        # Decoder Path with Residual Connections
        d5 = self.up5(bridge)
        d5_cat = torch.cat([d5, e5], dim=1)  
        d5 = self.dec5(d5_cat)  
        d5 = d5 + self.res_adapter_dec5(d5_cat)  
        aux5_up = F.interpolate(aux5, size=d5.shape[2:], mode='bilinear', align_corners=True)
        fg5 = self.fg_transform5(aux5_up)
        bg5 = self.bg_transform5(aux5_up)
        d5 = self.cdfa5(d5, fg5, bg5)
        
        d4 = self.up4(d5)
        d4_cat = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4_cat)
        d4 = d4 + self.res_adapter_dec4(d4_cat)
        aux4_up = F.interpolate(aux4, size=d4.shape[2:], mode='bilinear', align_corners=True)
        fg4 = self.fg_transform4(aux4_up)
        bg4 = self.bg_transform4(aux4_up)
        d4 = self.cdfa4(d4, fg4, bg4)
        
        d3 = self.up3(d4)
        d3_cat = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3_cat)
        d3 = d3 + self.res_adapter_dec3(d3_cat)
        aux3_up = F.interpolate(aux3, size=d3.shape[2:], mode='bilinear', align_corners=True)
        fg3 = self.fg_transform3(aux3_up)
        bg3 = self.bg_transform3(aux3_up)
        d3 = self.cdfa3(d3, fg3, bg3)
        
        d2 = self.up2(d3)
        d2_cat = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2_cat)
        d2 = d2 + self.res_adapter_dec2(d2_cat)
        aux2_up = F.interpolate(aux2, size=d2.shape[2:], mode='bilinear', align_corners=True)
        fg2 = self.fg_transform2(aux2_up)
        bg2 = self.bg_transform2(aux2_up)
        d2 = self.cdfa2(d2, fg2, bg2)
        
        d1 = self.up1(d2)
        d1_cat = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1_cat)
        d1 = d1 + self.res_adapter_dec1(d1_cat)
        aux1_up = F.interpolate(aux1, size=d1.shape[2:], mode='bilinear', align_corners=True)
        fg1 = self.fg_transform1(aux1_up)
        bg1 = self.bg_transform1(aux1_up)
        d1 = self.cdfa1(d1, fg1, bg1)
        
        # Final Output
        output = self.final(d1)
        
        if self.training:
            # Upsample auxiliary outputs
            aux1_up = F.interpolate(aux1, size=x.shape[2:], mode='bilinear', align_corners=True)
            aux2_up = F.interpolate(aux2, size=x.shape[2:], mode='bilinear', align_corners=True)
            aux3_up = F.interpolate(aux3, size=x.shape[2:], mode='bilinear', align_corners=True)
            aux4_up = F.interpolate(aux4, size=x.shape[2:], mode='bilinear', align_corners=True)
            aux5_up = F.interpolate(aux5, size=x.shape[2:], mode='bilinear', align_corners=True)
            return output, aux1_up, aux2_up, aux3_up, aux4_up, aux5_up
        
        return output
