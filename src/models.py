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

if __name__=='__main__':
    # show model
    model = UNetSemanticSegmentation(in_channels=3, out_channels=2, base=16)
    pred = model(torch.zeros([32,3,128,128]))
    print(pred.shape)