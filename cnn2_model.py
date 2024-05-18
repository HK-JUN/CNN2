import torch
import torch.nn as nn
import torch.nn.functional as F

class CM_Pooling(nn.Module):
    def __init__(self, channels, num_scales):
        super(CM_Pooling, self).__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.avgpools = nn.ModuleList([nn.AvgPool2d(kernel_size=(2**i, 2**i), stride=(2**i, 2**i))
                                        for i in range(num_scales)])

    def forward(self, x):
        multi_scale = [F.interpolate(pool(x), scale_factor=(2**i, 2**i), mode='nearest')
                       for i, pool in enumerate(self.avgpools)]
        out = torch.cat(multi_scale + [x], 1)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        if use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x.clone()
        out = self.conv(x.clone())
        out = self.bn(out)
        result_out = self.activation(out)
        if self.use_residual:
            residual = self.shortcut(residual)
            result_out = result_out + residual
        return result_out

class DualParallaxAugmentation(nn.Module):
    def __init__(self, channels):
        super(DualParallaxAugmentation, self).__init__()
        # 1x1 convolution for channel reduction after augmentation
        self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, bias=False)
    
    def forward(self, left, right):
        # Calculate the difference between the left and right feature maps
        diff_L = left - right
        diff_R = right - left

        # Concatenate the original and the difference feature maps
        augmented_L = torch.cat((left, diff_L), dim=1)
        augmented_R = torch.cat((right, diff_R), dim=1)

        # Reduce the channel dimensions back to original
        conv_augmented_L = self.conv1x1(augmented_L)
        conv_augmented_R = self.conv1x1(augmented_R)

        return conv_augmented_L, conv_augmented_R

class CNN2(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN2, self).__init__()
        self.in_channels = 1

        self.cm_pool1 = CM_Pooling(channels=3, num_scales=3)  # 첫 번째 ConvBlock의 CM Pooling
        self.layer1 = ConvBlock(1 * (1 + 3), 64)  # 1 original + 3 scales, channel=3이라면 수정해줘야함
        self.cm_pool2 = CM_Pooling(channels=64, num_scales=3)  # 두 번째 ConvBlock의 CM Pooling
        self.layer2 = ConvBlock(64 * (1 + 3), 128)  # CM_Pooling 후 증가된 채널 수 반영
        self.cm_pool3 = CM_Pooling(channels=128, num_scales=3)  # 세 번째 ConvBlock의 CM Pooling
        self.layer3 = ConvBlock(128 * (1 + 3), 256)  # CM_Pooling 후 증가된 채널 수 반영

        self.classifier = nn.Linear(256, num_classes)
        self.dpa1 = DualParallaxAugmentation(1)  # Assuming 1 is the number of channels after the first conv block
        self.dpa2 = DualParallaxAugmentation(64)  
        self.dpa3 = DualParallaxAugmentation(128)  
    def forward(self, xL, xR):
       # First Dual Parallax Augmentation
        xL, xR = self.dpa1(xL, xR)
        xL = self.cm_pool1(xL)
        xR = self.cm_pool1(xR)
        xL = self.layer1(xL)
        xR = self.layer1(xR)

        # Second Dual Parallax Augmentation
        xL, xR = self.dpa2(xL, xR)
        xL = self.cm_pool2(xL)
        xR = self.cm_pool2(xR)
        xL = self.layer2(xL)
        xR = self.layer2(xR)

        # Third Dual Parallax Augmentation
        xL, xR = self.dpa3(xL, xR)
        xL = self.cm_pool3(xL)
        xR = self.cm_pool3(xR)
        xL = self.layer3(xL)
        xR = self.layer3(xR)

        # Combine the features from the left and right pathways
        out = xL + xR
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
