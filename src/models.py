import torch
from torch import nn

class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, out_channels):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, s):
        g1 = self.Wg(g)          # Decoder features
        s1 = self.Ws(s)          # Skip connection features
        out = self.relu(g1 + s1) # Merge signals
        psi = self.psi(out)      # Attention map (0 to 1)
        return s * psi           # Filtered skip

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1 = self.get_conv_block(in_channels, out_channels, dropout_val)
        self.conv2 = self.get_conv_block(out_channels, out_channels, dropout_val)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y+self.x_proj(x)

    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

class ResAttUNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transp_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.transp_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv1 = ConvBlock(16, 64, dropout)
        self.conv2 = ConvBlock(64, 128, dropout)
        self.conv3 = ConvBlock(128, 256, dropout)
        self.conv4 = ConvBlock(256, 128, dropout)
        self.conv5 = ConvBlock(128, 64, dropout)

        self.att1 = AttentionGate(128,128,128)
        self.att2 = AttentionGate(64,64,64)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )

    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.in_proj(x)
        end1 = self.conv1(x)
        down1 = self.pool(end1)

        end2 = self.conv2(down1)
        down2 = self.pool(end2)

        end3 = self.conv3(down2)
        up1 = self.transp_conv1(end3)

        a1 = self.att1(up1, end2)
        end4 = self.conv4(torch.cat([a1, up1], axis=1))
        up2 = self.transp_conv2(end4)

        a2 = self.att2(up2, end1)
        end5 = self.conv5(torch.cat([a2, up2], axis=1))

        return self.out(end5)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, dropout=0.2, base=32):
        super().__init__()

        self.name = 'SimpleUNet'

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transp_conv_1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv_2 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            self.get_conv_block(4, base, dropout),
            self.get_conv_block(base, base, dropout),
        )
        self.conv2 = nn.Sequential(
            self.get_conv_block(base, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout),
        )
        self.conv3 = nn.Sequential(
            self.get_conv_block(base*2, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout),
        )
        self.conv4 = nn.Sequential(
            self.get_conv_block(base*4, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout),
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(base*2, base, dropout),
            self.get_conv_block(base, base, dropout),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        end1 = self.conv1(x)
        down1 = self.pool(end1)

        end2 = self.conv2(down1)
        down2 = self.pool(end2)

        end3 = self.conv3(down2)
        up1 = self.transp_conv_1(end3)

        end4 = self.conv4(torch.cat([up1, end2], axis=1))
        up2 = self.transp_conv_2(end4)

        end5 = self.conv5(torch.cat([up2, end1], axis=1))

        return self.out(end5)
    
    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

class SimpleUNetEmb(nn.Module):
    def __init__(self, in_channels, out_channels, chip_size, dropout=0.2, base=32):
        super().__init__()

        self.name = 'SimpleUNetEmb'

        self.chip_size = chip_size

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transp_conv_1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.transp_conv_2 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            self.get_conv_block(in_channels+1, base, dropout),
            self.get_conv_block(base, base, dropout),
        )
        self.conv2 = nn.Sequential(
            self.get_conv_block(base, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout),
        )
        self.conv3 = nn.Sequential(
            self.get_conv_block(base*2, base*4, dropout),
            self.get_conv_block(base*4, base*4, dropout),
        )
        self.conv4 = nn.Sequential(
            self.get_conv_block(base*4, base*2, dropout),
            self.get_conv_block(base*2, base*2, dropout),
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(base*2, base, dropout),
            self.get_conv_block(base, base, dropout),
        )

        self.in_emb = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

        self.transpose_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=int(chip_size/16), stride=int(chip_size/16))

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, emb):
        emb_x = self.in_emb(emb).view(emb.shape[0], 1, 16, 16)
        emb_x = self.transpose_conv(emb_x)

        end1 = self.conv1(torch.cat([x, emb_x], axis=1))
        down1 = self.pool(end1)

        end2 = self.conv2(down1)
        down2 = self.pool(end2)

        end3 = self.conv3(down2)
        up1 = self.transp_conv_1(end3)

        end4 = self.conv4(torch.cat([up1, end2], axis=1))
        up2 = self.transp_conv_2(end4)

        end5 = self.conv5(torch.cat([up2, end1], axis=1))

        return self.out(end5)
    
    def get_conv_block(self, in_channels, out_channels, dropout_val):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_val, inplace=True),
            nn.LeakyReLU(inplace=True)
        )