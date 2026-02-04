"""
PMDNet: Progressive modulation network with global-local representations for single image deraining
"""

import torch
import torch.nn as nn
from Sandwich_Transformer_Block import Sandwich_Transformer_Block
import utils

##########################################################################
# Overlapped image patch embedding with 3×3 Conv
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=48, bias=False):
        super(PatchEmbedding, self).__init__()

        self.project = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.shallow_fea_img = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.shallow_fea_rain = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x_fea = self.project(x)
        img_fea = self.shallow_fea_img(x_fea)
        rain_fea = self.shallow_fea_rain(x_fea)

        return [img_fea, rain_fea]


##########################################################################
##########
# ECAU
class Enhanced_Channel_Attention_Unit(nn.Module):
    def __init__(self, n_feat, ratio=16):
        super(Enhanced_Channel_Attention_Unit, self).__init__()

        self.inconv1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.inconv2 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.inconv3 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(n_feat // ratio, n_feat, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_add_dilated = self.inconv1(x) + self.inconv2(x) + self.inconv3(x)
        avg_out = self.shared_mlp(self.avg_pool(x_add_dilated))
        max_out = self.shared_mlp(self.max_pool(x_add_dilated))
        out = avg_out + max_out
        return self.sigmoid(out)


##########
# ESAU
class Enhanced_Spatial_Attention_Unit(nn.Module):
    def __init__(self, kernel_size=3):
        super(Enhanced_Spatial_Attention_Unit, self).__init__()

        self.inconv1 = nn.Conv2d(2, 1, kernel_size, stride=1, padding=1, dilation=1, bias=False)
        self.inconv2 = nn.Conv2d(2, 1, kernel_size, stride=1, padding=3, dilation=3, bias=False)
        self.inconv3 = nn.Conv2d(2, 1, kernel_size, stride=1, padding=5, dilation=5, bias=False)

        self.outconv = nn.Conv2d(3, 1, kernel_size, padding=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x_cat_dilated = torch.cat([self.inconv1(x), self.inconv2(x), self.inconv3(x)], dim=1)
        out = self.outconv(x_cat_dilated)
        return self.sigmoid(out)


##########
# Feature Modulation Block (FMB)
class FMB(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(FMB, self).__init__()

        self.rain1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_feat*2, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        self.rain2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        self.img1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        self.img2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        self.ESAU = Enhanced_Spatial_Attention_Unit()
        self.ECAU = Enhanced_Channel_Attention_Unit(n_feat)

    def forward(self, x_img, x_rain):
        shortcut_img = x_img
        shortcut_rain = x_rain

        x_rain_0 = torch.cat([x_rain, x_img], dim=1)
        x_rain_1 = self.rain1(x_rain_0)
        x_img_1 = self.img1(x_img) * self.ESAU(x_rain_1)

        x_rain_2 = self.rain2(x_rain_1)
        x_img_2 = self.img2(x_img_1) * self.ECAU(x_rain_2)

        x_rain_final = x_rain_2 + shortcut_rain
        x_img_final = x_img_2 + shortcut_img

        return [x_img_final, x_rain_final]


##########################################################################
# Cross-Scale Reconstruction Module (CSRM)
class RM(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(RM, self).__init__()

        self.recon_rain = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1, bias=bias)
        self.recon_img = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1, bias=bias)

        self.stconv_rain = nn.Conv2d(3, n_feat, kernel_size=3, padding=1, stride=2, bias=bias)
        self.stconv_img = nn.Conv2d(3, n_feat, kernel_size=3, padding=1, stride=2, bias=bias)
        self.stconv_hr_re_rain = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1, stride=2, bias=bias)

        self.trconv_rain = nn.ConvTranspose2d(3, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias)
        self.trconv_img = nn.ConvTranspose2d(3, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias)
        self.trconv_lr_re_rain = nn.ConvTranspose2d(n_feat, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=bias)

    def forward(self, x):
        y_img = x[0]
        y_rain = x[1]

        y1_img = self.recon_img(y_img)
        y1_rain = self.recon_rain(y_rain)

        hr_re_rain = self.trconv_img(y1_img) + self.trconv_rain(y1_rain)
        lr_re_rain = self.stconv_img(y1_img) + self.stconv_rain(y1_rain)

        final_re_rain = self.stconv_hr_re_rain(hr_re_rain) + self.trconv_lr_re_rain(lr_re_rain)

        return [y1_img, final_re_rain, y1_rain]


##########################################################################
##########
class Spatial_DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(Spatial_DownSample, self).__init__()

        self.down = nn.Sequential(
            nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.down(x)


##########
class Spatial_UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(Spatial_UpSample, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.up(x)


##########
# Image-Rain Joint Representation Module (IR-JRM)
# Dual-Branch Joint Representation Module (DBJRM)
class DBJRM(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(DBJRM, self).__init__()

        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  # Other option 'BiasFree'

        self.encoder_dbjrm_0 = Sandwich_Transformer_Block(dim=n_feat, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.down0_1 = Spatial_DownSample(n_feat, n_feat*2, 0.5)
        self.encoder_dbjrm_1 = Sandwich_Transformer_Block(dim=n_feat, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.down1_2 = Spatial_DownSample(n_feat, n_feat*4, 0.5)
        self.latent_dbjrm_2 = Sandwich_Transformer_Block(dim=n_feat, num_heads=4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.up2_1 = Spatial_UpSample(n_feat, n_feat*4, 2)
        self.reduce_channels_2 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.decoder_dbjrm_1 = Sandwich_Transformer_Block(dim=n_feat, num_heads=2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.up1_0 = Spatial_UpSample(n_feat, n_feat*2, 2)
        self.reduce_channels_1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)
        self.decoder_dbjrm_0 = Sandwich_Transformer_Block(dim=n_feat, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.output_latent_dbjrm_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.output_decoder_dbjrm_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce_channels = nn.Conv2d(n_feat*3, n_feat, kernel_size=1, bias=bias)

        self.fmb = FMB(n_feat, bias)

    def forward(self, x):
        f_img = x[0]
        f_rain = x[1]

        rain_encoder_0 = self.encoder_dbjrm_0(f_rain)
        rain_down0_1 = self.down0_1(rain_encoder_0)
        rain_encoder_1 = self.encoder_dbjrm_1(rain_down0_1)
        rain_down1_2 = self.down1_2(rain_encoder_1)

        rain_latent = self.latent_dbjrm_2(rain_down1_2)

        rain_up2_1 = self.up2_1(rain_latent)
        rain_en1_up21_concat = torch.cat([rain_encoder_1, rain_up2_1], dim=1)
        rain_decoder_1 = self.decoder_dbjrm_1(self.reduce_channels_2(rain_en1_up21_concat))

        rain_up1_0 = self.up1_0(rain_decoder_1)
        rain_en0_up10_concat = torch.cat([rain_encoder_0, rain_up1_0], dim=1)
        rain_decoder_0 = self.decoder_dbjrm_0(self.reduce_channels_1(rain_en0_up10_concat))

        rain_output_1 = self.output_latent_dbjrm_2(rain_latent)
        rain_output_2 = self.output_decoder_dbjrm_1(rain_decoder_1)
        rain_output = torch.cat([rain_decoder_0, rain_output_2, rain_output_1], dim=1)
        rain = self.reduce_channels(rain_output)

        f_rain_res = rain + f_rain
        f_img_im = f_img

        # FMB
        x = self.fmb(f_img_im, f_rain_res)

        return x


##########################################################################
class Model(nn.Module):
    def __init__(self, in_channels, n_feat, bias):
        super(Model, self).__init__()

        # embedding
        self.patch_embedding = PatchEmbedding(in_channels, n_feat, bias)
        self.dbjrm_3 = nn.Sequential(*[DBJRM(n_feat, bias) for _ in range(3)])
        self.rm = RM(n_feat, bias)

    def forward(self, x):
        [img_fea, rain_fea] = self.patch_embedding(x)
        [out_img_dbjrm_3, out_rain_dbjrm_3] = self.dbjrm_3([img_fea, rain_fea])
        [img, rainy_input, rain] = self.rm([out_img_dbjrm_3, out_rain_dbjrm_3])

        return img, rainy_input, rain


##########################################################################
# PMDNet
class PMDNet(nn.Module):
    def __init__(self, in_channels=3, n_feat=48, bias=False):
        super(PMDNet, self).__init__()

        self.model = Model(in_channels, n_feat, bias)

    def forward(self, rainy):
        pre_img, re_rainy_input, pre_rain = self.model(rainy)
        return [pre_img, re_rainy_input, pre_rain]