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

if __name__=='__main__':
    # show model
    model = UNetSemanticSegmentation(in_channels=3, out_channels=2, base=16)
    pred = model(torch.zeros([32,3,128,128]))
    print(pred.shape)