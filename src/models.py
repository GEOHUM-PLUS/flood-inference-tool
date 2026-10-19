import torch
from torch import nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val=0.2):
        super().__init__()
        self.conv1 = self.GetConvBlock(in_channels=in_channels, out_channels=out_channels, dropout_val=dropout_val)
        self.conv2 = self.GetConvBlock(in_channels=out_channels, out_channels=out_channels, dropout_val=dropout_val)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def GetConvBlock(self, in_channels, out_channels, dropout_val=0.2, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2), padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val),
            nn.ReLU()
        )

class Encoder(nn.Module):
    def __init__(self, in_channels:int=4, base:int=32, dropout_val=0.2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv1 = DoubleConvBlock(in_channels, base, dropout_val=dropout_val)
        self.double_conv2 = DoubleConvBlock(base, base*2, dropout_val=dropout_val)
        self.double_conv3 = DoubleConvBlock(base*2, base*4, dropout_val=dropout_val)
        self.double_conv4 = DoubleConvBlock(base*4, base*8, dropout_val=dropout_val)

    def forward(self, x):
        skip1 = self.double_conv1(x)
        skip2 = self.double_conv2(self.pool(skip1))
        skip3 = self.double_conv3(self.pool(skip2))
        skip4 = self.double_conv4(self.pool(skip3))

        return skip4, skip3, skip2, skip1

class Decoder(nn.Module):
    def __init__(self, base=32, dropout_val=0.2):
        super().__init__()
        self.double_conv1 = DoubleConvBlock(base*2, base, dropout_val=dropout_val)
        self.double_conv2 = DoubleConvBlock(base*4, base*2, dropout_val=dropout_val)
        self.double_conv3 = DoubleConvBlock(base*8, base*4, dropout_val=dropout_val)

        self.transp_conv1 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.transp_conv2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv3 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)

    def forward(self, skip4, skip3, skip2, skip1):
        x = self.transp_conv1(skip4)
        x = self.double_conv3(torch.cat([skip3, x], axis=1))

        x = self.transp_conv2(x)
        x = self.double_conv2(torch.cat([skip2, x], axis=1))

        x = self.transp_conv3(x)
        x = self.double_conv1(torch.cat([skip1, x], axis=1))

        return x

class UNetBackbone(nn.Module):
    def __init__(self, in_channels, base, dropout_val=0.2):
        super().__init__()

        self.encoder = Encoder(in_channels, base, dropout_val)
        self.decoder = Decoder(base, dropout_val)
    
    def forward(self, x):
        skip4, skip3, skip2, skip1 = self.encoder(x)
        out = self.decoder(skip4, skip3, skip2, skip1)
        return out

class UNetSemanticSegmentation(nn.Module):
    def __init__(self, in_channels, out_channels, base, dropout_val=0.2):
        super().__init__()
        self.backbone = UNetBackbone(in_channels, base, dropout_val)
        self.classification_head = nn.Sequential(
            nn.Conv2d(base, out_channels, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classification_head(x)
        return x

class EncoderS2(nn.Module):
    def __init__(self, in_channels:int=6, base:int=64, dropout_val=0.2):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv1 = DoubleConvBlock(in_channels, base, dropout_val=dropout_val)
        self.double_conv2 = DoubleConvBlock(base, base*2, dropout_val=dropout_val)
        self.double_conv3 = DoubleConvBlock(base*2, base*4, dropout_val=dropout_val)
        self.double_conv4 = DoubleConvBlock(base*4, base*8, dropout_val=dropout_val)

    def forward(self, x):
        x = self.bn(x)
        skip1 = self.double_conv1(x)
        skip2 = self.double_conv2(self.pool(skip1))
        skip3 = self.double_conv3(self.pool(skip2))
        skip4 = self.double_conv4(self.pool(skip3))

        return skip4, skip3, skip2, skip1

class DecoderS2(nn.Module):
    def __init__(self, out_channels:int=3, base:int=64, dropout_val=0.2):
        super().__init__()
        self.double_conv1 = DoubleConvBlock(base*2, base, dropout_val=dropout_val)
        self.double_conv2 = DoubleConvBlock(base*4, base*2, dropout_val=dropout_val)
        self.double_conv3 = DoubleConvBlock(base*8, base*4, dropout_val=dropout_val)

        self.transp_conv1 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.transp_conv2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv3 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)

        self.out = nn.Sequential(
            nn.Conv2d(base, out_channels, kernel_size=1, padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, skip4, skip3, skip2, skip1):
        x = self.transp_conv1(skip4)
        x = self.double_conv3(torch.cat([skip3, x], axis=1))

        x = self.transp_conv2(x)
        x = self.double_conv2(torch.cat([skip2, x], axis=1))

        x = self.transp_conv3(x)
        x = self.double_conv1(torch.cat([skip1, x], axis=1))

        return self.out(x)

class UNetSemanticSegmentationS2(nn.Module):
    '''
    Matches the checkpoint layout of models/UNet-S2.pt: a flat encoder/decoder
    (no separate classification_head) with a learnable input BatchNorm2d,
    trained on Sentinel-2 blue/green/red/nir/swir16/swir22 bands (in this order).
    '''
    def __init__(self, in_channels:int=6, out_channels:int=3, base:int=64, dropout_val=0.2):
        super().__init__()
        self.encoder = EncoderS2(in_channels, base, dropout_val)
        self.decoder = DecoderS2(out_channels, base, dropout_val)

    def forward(self, x):
        skip4, skip3, skip2, skip1 = self.encoder(x)
        return self.decoder(skip4, skip3, skip2, skip1)

def _wf2_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNetML4FloodsWF2(nn.Module):
    '''
    Matches the checkpoint layout of ml4floods' WF2_unetv2_bgriswirs model
    (https://huggingface.co/isp-uv-es/ml4floods): 4 encoder levels, bilinear
    upsampling, no batchnorm/dropout. Trained on Sentinel-2
    blue/green/red/nir/swir16/swir22 bands (in this order), predicting 2
    independent sigmoid outputs: cloud probability and water probability.
    '''
    def __init__(self, n_channels:int=6, n_class:int=2):
        super().__init__()
        self.dconv_down1 = _wf2_double_conv(n_channels, 64)
        self.dconv_down2 = _wf2_double_conv(64, 128)
        self.dconv_down3 = _wf2_double_conv(128, 256)
        self.dconv_down4 = _wf2_double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.dconv_up3 = _wf2_double_conv(256 + 512, 256)
        self.dconv_up2 = _wf2_double_conv(128 + 256, 128)
        self.dconv_up1 = _wf2_double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        up = lambda t: nn.functional.interpolate(t, scale_factor=2, mode='bilinear', align_corners=True)
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        x = self.dconv_down4(self.maxpool(conv3))
        x = self.dconv_up3(torch.cat([up(x), conv3], dim=1))
        x = self.dconv_up2(torch.cat([up(x), conv2], dim=1))
        x = self.dconv_up1(torch.cat([up(x), conv1], dim=1))
        return self.conv_last(x)

# --- DOFA v2 + UPerNet --------------------------------------------------------
# Standalone port of the model trained with DOFA-transferability (DOFA v2 Base
# encoder + PANGAEA's Feature2Pyramid neck and UPerNet decoder), so it runs
# without the PANGAEA/timm/hydra stack. Module and parameter names mirror the
# originals so the training checkpoint loads strictly. Sources:
#   pangaea/encoders/dofa_encoder.py (Dynamic_MLP_OFA and helpers),
#   dofa_transferability/dofav2_encoder.py, timm's VisionTransformer,
#   pangaea/decoders/upernet.py (SegUPerNet, PPM, Feature2Pyramid).

def _sincos_1d(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / 10000**(omega / (embed_dim / 2.0))
    out = torch.einsum('m,d->md', pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, activation='gelu', norm_first=False, batch_first=False, dropout=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.zeros([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.zeros([1, input_dim]))

    def forward(self, x):
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num:-1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias

class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super().__init__()
        self.w1 = nn.Linear(linear_size, linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)

    def forward(self, x):
        return x + torch.relu(self.w2(torch.relu(self.w1(x))))

class DynamicMLPOFA(nn.Module):
    '''DOFA's wavelength-conditioned patch embedding: conv weights are generated from the band wavelengths.'''
    def __init__(self, wv_planes=128, kernel_size=14, embed_dim=768):
        super().__init__()
        self.wv_planes = wv_planes
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.scaler = 0.01
        self.weight_generator = TransformerWeightGenerator(wv_planes, kernel_size*kernel_size*embed_dim, embed_dim)
        self.fclayer = FCResLayer(wv_planes)

    def forward(self, x, wavelengths):
        waves = self.fclayer(_sincos_1d(self.wv_planes, wavelengths*1000).float())
        weight, bias = self.weight_generator(waves)
        weight = weight.view(len(wavelengths), self.kernel_size, self.kernel_size, self.embed_dim).permute([3, 0, 1, 2])
        return nn.functional.conv2d(x, weight*self.scaler, bias=bias.view([self.embed_dim])*self.scaler, stride=self.kernel_size, padding=1)

class ViTAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values*torch.ones(dim))

    def forward(self, x):
        return x*self.gamma

class ViTMlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = ViTAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = ViTMlp(dim, int(dim*mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))

class ViTBackbone(nn.Module):
    '''The parts of timm's VisionTransformer that DOFA v2 uses (its own patch embedding is replaced).'''
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, pretrain_grid=16):
        super().__init__()
        self.pretrain_grid = pretrain_grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, pretrain_grid*pretrain_grid+1, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)  # present in the checkpoint; DOFA v2 features skip it

    def pos_embed_for(self, h, w):
        if (h, w)==(self.pretrain_grid, self.pretrain_grid):
            return self.pos_embed
        cls_pos, grid_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        grid_pos = grid_pos.reshape(1, self.pretrain_grid, self.pretrain_grid, -1).permute(0, 3, 1, 2)
        grid_pos = nn.functional.interpolate(grid_pos, size=(h, w), mode='bicubic', antialias=True)
        return torch.cat([cls_pos, grid_pos.permute(0, 2, 3, 1).reshape(1, h*w, -1)], dim=1)

class DOFAv2Encoder(nn.Module):
    def __init__(self, wavelengths, embed_dim=768, depth=12, num_heads=12, patch_size=14, output_layers=(3, 5, 7, 11)):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.output_layers = output_layers
        self.register_buffer('wavelengths', torch.tensor(wavelengths, dtype=torch.float32), persistent=False)
        self.patch_embed = DynamicMLPOFA(kernel_size=patch_size, embed_dim=embed_dim)
        self.model = ViTBackbone(embed_dim, depth, num_heads)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x, self.wavelengths)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.model.cls_token.expand(B, -1, -1), x], dim=1) + self.model.pos_embed_for(h, w)

        output = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.output_layers:
                output.append(x[:, 1:].transpose(1, 2).reshape(B, self.embed_dim, h, w).contiguous())
        return output

def _conv_bn_relu(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Feature2Pyramid(nn.Module):
    '''Turns the 4 same-resolution ViT feature maps into a 4x/2x/1x/0.5x pyramid.'''
    def __init__(self, embed_dim):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)),
            nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)),
            nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

    def forward(self, inputs):
        return [op(x) for op, x in zip(self.ops, inputs)]

class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels):
        super().__init__([
            nn.Sequential(nn.AdaptiveAvgPool2d(scale), *_conv_bn_relu(in_channels, channels, 1)) for scale in pool_scales])

    def forward(self, x):
        outs = []
        for ppm in self:
            pool, rest = ppm[0], ppm[1:]
            # MPS's adaptive_avg_pool2d only supports input sizes evenly divisible
            # by the output size (pytorch/pytorch#96056); x's spatial size here
            # isn't guaranteed to divide evenly into every pool_scale, so run just
            # this op on CPU when needed rather than requiring it of every input.
            pooled = pool(x.cpu()).to(x.device) if x.device.type=='mps' else pool(x)
            outs.append(nn.functional.interpolate(rest(pooled), size=x.shape[2:], mode='bilinear', align_corners=False))
        return outs

class DOFAv2UPerNet(nn.Module):
    '''
    Matches the checkpoint layout of DOFA-transferability's
    dofa_v2_base_seg_upernet runs: DOFA v2 Base encoder -> Feature2Pyramid
    -> UPerNet (PSP + FPN) -> 2 classes (not water, water). Outputs softmax
    probabilities like this tool's other models. Input bands are described by
    their centre wavelengths (micrometres), so the same weights accept any band set.
    '''
    def __init__(self, wavelengths, num_classes:int=2, embed_dim:int=768, channels:int=512, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.encoder = DOFAv2Encoder(wavelengths, embed_dim=embed_dim)
        self.neck = Feature2Pyramid(embed_dim)
        self.psp_modules = PPM(pool_scales, embed_dim, channels)
        self.bottleneck = _conv_bn_relu(embed_dim + len(pool_scales)*channels, channels, 3)
        self.lateral_convs = nn.ModuleList([_conv_bn_relu(embed_dim, channels, 1) for _ in range(3)])
        self.fpn_convs = nn.ModuleList([_conv_bn_relu(channels, channels, 3) for _ in range(3)])
        self.fpn_bottleneck = _conv_bn_relu(4*channels, channels, 3)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        output_shape = x.shape[-2:]
        feats = self.neck(self.encoder(x))

        laterals = [conv(feats[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self.bottleneck(torch.cat([feats[-1], *self.psp_modules(feats[-1])], dim=1)))
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] = laterals[i-1] + nn.functional.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='bilinear', align_corners=False)

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals)-1)] + [laterals[-1]]
        outs = [outs[0]] + [nn.functional.interpolate(o, size=outs[0].shape[2:], mode='bilinear', align_corners=False) for o in outs[1:]]
        x = self.conv_seg(self.dropout(self.fpn_bottleneck(torch.cat(outs, dim=1))))
        x = nn.functional.interpolate(x, size=output_shape, mode='bilinear')
        return torch.softmax(x, dim=1)

if __name__=='__main__':
    # show model
    model = UNetSemanticSegmentation(in_channels=3, out_channels=2, base=16)
    pred = model(torch.zeros([32,3,128,128]))
    print(pred.shape)