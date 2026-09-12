import torch
import torch.nn as nn
import os

from .masks import CentralMaskedConv2d, RowMaskedConv2d, ColMaskedConv2d, \
    SzMaskedConv2d, fSzMaskedConv2d, angle45MaskedConv2d, \
    angle135MaskedConv2d, chaMaskedConv2d, fchaMaskedConv2d, huiMaskedConv2d
from . import regist_model
# from .MSFBlock import MSFblock
@regist_model
class MMBSN(nn.Module):
    '''
    Multi-Masked Blind-Spot Network (Customized Light Version)

    Self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    with several modifications included. The network uses multiple masked convolutions to capture
    different directional features while maintaining the blind-spot property.
    
    Key features:
    1. Multiple masked convolution branches for different spatial patterns
    2. Dilated convolution layers for expanded receptive fields
    3. Feature fusion from multiple branches
    4. Configurable mask types for different denoising scenarios
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, DCL1_num=2, DCL2_num=7, mask_type='o_a45'):
        '''
        Initialize Multi-Masked Blind-Spot Network
        
        Args:
            in_ch      : Number of input channels (default: 3 for RGB)
            out_ch     : Number of output channels (default: 3 for RGB)
            base_ch    : Number of base channels (default: 128)
            DCL1_num   : Number of dilated convolution layers in first stage (default: 2)
            DCL2_num   : Number of dilated convolution layers in second stage (default: 7)
            mask_type  : Mask types separated by underscore (default: 'o_a45')
                        o: central mask, c: column mask, r: row mask
                        sz: sz mask, fsz: fsz mask, a45: 45-degree angle mask
                        a135: 135-degree angle mask, cha: cha mask, fcha: fcha mask, hui: hui mask
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divisible by 2"

        # Head layer: 1x1 convolution to adjust channel dimension
        ly0 = []
        ly0 += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]
        ly0 += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly0)

        # Parse mask types and initialize parameters
        self.mask_types = mask_type.split('_')
        mask_number = len(self.mask_types)
        DCL_number1 = DCL1_num
        DCL_number2 = DCL2_num

        # Initialize mask-specific branches based on mask types
        if 'o' in self.mask_types:  # Central mask (omit center pixel)
            self.branch1_1 = DC_branchl(2, base_ch, 'central', DCL_number1)
            self.branch1_2 = DC_branchl(3, base_ch, 'central', DCL_number1)
        if 'c' in self.mask_types:  # Column mask
            self.branch2_1 = DC_branchl(2, base_ch, 'col', DCL_number1)
            self.branch2_2 = DC_branchl(3, base_ch, 'col', DCL_number1)
        if 'r' in self.mask_types:  # Row mask
            self.branch3_1 = DC_branchl(2, base_ch, 'row', DCL_number1)
            self.branch3_2 = DC_branchl(3, base_ch, 'row', DCL_number1)
        if 'sz' in self.mask_types:  # SZ mask
            self.branch4_1 = DC_branchl(2, base_ch, 'sz', DCL_number1)
            self.branch4_2 = DC_branchl(3, base_ch, 'sz', DCL_number1)
        if 'fsz' in self.mask_types:  # FSZ mask
            self.branch5_1 = DC_branchl(2, base_ch, 'fsz', DCL_number1)
            self.branch5_2 = DC_branchl(3, base_ch, 'fsz', DCL_number1)
        if 'a45' in self.mask_types:  # 45-degree angle mask
            self.branch6_1 = DC_branchl(2, base_ch, 'a45', DCL_number1)
            self.branch6_2 = DC_branchl(3, base_ch, 'a45', DCL_number1)
        if 'a135' in self.mask_types:  # 135-degree angle mask
            self.branch7_1 = DC_branchl(2, base_ch, 'a135', DCL_number1)
            self.branch7_2 = DC_branchl(3, base_ch, 'a135', DCL_number1)
        if 'cha' in self.mask_types:  # CHA mask
            self.branch9_1 = DC_branchl(2, base_ch, 'cha', DCL_number1)
            self.branch9_2 = DC_branchl(3, base_ch, 'cha', DCL_number1)
        if 'fcha' in self.mask_types:  # FCHA mask
            self.branch10_1 = DC_branchl(2, base_ch, 'fcha', DCL_number1)
            self.branch10_2 = DC_branchl(3, base_ch, 'fcha', DCL_number1)
        if 'hui' in self.mask_types:  # HUI mask
            self.branch8_1 = DC_branchl(2, base_ch, 'hui', DCL_number1)
            self.branch8_2 = DC_branchl(3, base_ch, 'hui', DCL_number1)

        # Channel fusion layers
        ly_c = []
        ly_c += [nn.Conv2d(base_ch*mask_number, base_ch, kernel_size=1)]
        ly_c += [nn.ReLU(inplace=True)]
        self.conv2_1 = nn.Sequential(*ly_c)
        self.conv2_2 = nn.Sequential(*ly_c)

        # Second stage dilated convolution branches
        self.dc_branchl2_mask3 = DC_branchl2(2, base_ch, DCL_number2)  # Stride 2 branch
        self.dc_branchl2_mask5 = DC_branchl2(3, base_ch, DCL_number2)  # Stride 3 branch
        # self.MSF = MSFblock(in_channels=base_ch)


        # Tail layer: progressive channel reduction and final output
        ly = []
        ly += [ nn.Conv2d(base_ch*(2+2*mask_number),  base_ch,    kernel_size=1) ]  # Merge all features
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]  # Reduce channels
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]  # Further reduction
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]  # Final output
        self.tail = nn.Sequential(*ly)
    def forward(self, x, masked=True):
        """
        Forward pass through Multi-Masked Blind-Spot Network
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            masked: Whether to apply masking (for training/inference)
            
        Returns:
            Denoised output tensor of shape [B, C, H, W]
        """
        mask_types = self.mask_types

        # Head processing
        x = self.head(x)
        y1 = []  # Features from stride-2 branches
        y2 = []  # Features from stride-3 branches
        e = []   # All extracted features
        # Process features through different masked branches
        if 'o' in mask_types:  # Central mask
            e1_1, br1_1 = self.branch1_1(x)
            e1_2, br1_2 = self.branch1_2(x)
            y1.append(br1_1)
            y2.append(br1_2)
            e.append(e1_1)
            e.append(e1_2)
        if 'c' in mask_types:
            e2_1, br2_1 = self.branch2_1(x)
            e2_2, br2_2 = self.branch2_2(x)
            y1.append(br2_1)
            y2.append(br2_2)
            e.append(e2_1)
            e.append(e2_2)
        if 'r' in mask_types:
            e3_1, br3_1 = self.branch3_1(x)
            e3_2, br3_2 = self.branch3_2(x)
            y1.append(br3_1)
            y2.append(br3_2)
            e.append(e3_1)
            e.append(e3_2)
        if 'sz' in mask_types:
            e4_1, br4_1 = self.branch4_1(x)
            e4_2, br4_2 = self.branch4_2(x)
            y1.append(br4_1)
            y2.append(br4_2)
            e.append(e4_1)
            e.append(e4_2)
        if 'fsz' in mask_types:
            e5_1, br5_1 = self.branch5_1(x)
            e5_2, br5_2 = self.branch5_2(x)
            y1.append(br5_1)
            y2.append(br5_2)
            e.append(e5_1)
            e.append(e5_2)
        if 'a45' in mask_types:
            e6_1, br6_1 = self.branch6_1(x)
            e6_2, br6_2 = self.branch6_2(x)
            y1.append(br6_1)
            y2.append(br6_2)
            e.append(e6_1)
            e.append(e6_2)
        if 'a135' in mask_types:
            e7_1, br7_1 = self.branch7_1(x)
            e7_2, br7_2 = self.branch7_2(x)
            y1.append(br7_1)
            y2.append(br7_2)
            e.append(e7_1)
            e.append(e7_2)
        if 'cha' in mask_types:
            e9_1, br9_1 = self.branch9_1(x)
            e9_2, br9_2 = self.branch9_2(x)
            y1.append(br9_1)
            y2.append(br9_2)
            e.append(e9_1)
            e.append(e9_2)
        if 'fcha' in mask_types:
            e10_1, br10_1 = self.branch10_1(x)
            e10_2, br10_2 = self.branch10_2(x)
            y1.append(br10_1)
            y2.append(br10_2)
            e.append(e10_1)
            e.append(e10_2)
        if 'hui' in mask_types:
            e8_1, br8_1 = self.branch8_1(x)
            e8_2, br8_2 = self.branch8_2(x)
            y1.append(br8_1)
            y2.append(br8_2)
            e.append(e8_1)
            e.append(e8_2)

        # Feature fusion and second stage processing
        cat1 = torch.cat(y1, dim=1)  # Concatenate stride-2 features
        conv2_1 = self.conv2_1(cat1)
        dc_branchl2_m3 = self.dc_branchl2_mask3(conv2_1)  # Second stage stride-2

        cat2 = torch.cat(y2, dim=1)  # Concatenate stride-3 features
        conv2_2 = self.conv2_2(cat2)
        dc_branchl2_m5 = self.dc_branchl2_mask5(conv2_2)  # Second stage stride-3

        # Final feature aggregation
        e.append(dc_branchl2_m3)
        e.append(dc_branchl2_m5)
        cat3 = torch.cat(e, dim=1)  # Concatenate all features
        # cat3 = self.MSF(e)  # Alternative: Multi-Scale Fusion (commented out)

        return self.tail(cat3)  # Generate final output

    def _initialize_weights(self):
        """
        Initialize network weights using normal distribution
        Based on Liyong's initialization scheme
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize conv layers with normal distribution
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    """
    Dilated Convolution Branch with Masked Convolutions
    
    This module implements a branch with masked convolutions followed by
    dilated convolution layers for feature extraction.
    
    Args:
        stride: Dilation stride for convolutions
        in_ch: Number of input channels
        mask_type: Type of mask to apply ('central', 'row', 'col', etc.)
        num_module: Number of dilated convolution modules
    """
    def __init__(self, stride, in_ch, mask_type, num_module):
        super().__init__()

        ly0 = []  # Head layers with masked convolution
        ly1_1 = [] # First processing branch
        ly1_2 = [] # Second processing branch  
        ly2 = []  # Body layers with dilated convolutions

        # Select appropriate masked convolution based on mask type
        if mask_type == 'r':
            ly0 += [ RowMaskedConv2d(in_ch, in_ch, kernel_size=2*stride-1, stride=1, padding=stride-1) ]
        elif mask_type == 'c':
            ly0 += [ ColMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'sz':
            ly0 += [ SzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fsz':
            ly0 += [ fSzMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a45':
            ly0 += [ angle45MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'a135':
            ly0 += [ angle135MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'hui':
            ly0 += [huiMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'cha':
            ly0 += [chaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif mask_type == 'fcha':
            ly0 += [fchaMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        else:  # Default to central mask
            ly0 += [ CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]

        # Complete head layers with feature processing
        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]  # 1x1 conv for feature adjustment
        ly0 += [ nn.ReLU(inplace=True) ]
        ly0 += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]  # Another 1x1 conv
        ly0 += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly0)

        # Feature processing branches
        ly1_1 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly1_1 += [nn.ReLU(inplace=True)]
        self.conv1_1 = nn.Sequential(*ly1_1)  # First branch processing
        self.conv1_2 = nn.Sequential(*ly1_1)  # Second branch processing (shared weights)

        # Body with dilated convolution layers
        ly2 += [ DCl(stride, in_ch) for _ in range(num_module) ]  # Multiple DCL modules
        ly2 += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]  # Final feature adjustment
        ly2 += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly2)

        # Feature fusion layer
        ly1_2 += [nn.Conv2d(in_ch*2, in_ch, kernel_size=1)]  # Combine two feature streams
        ly1_2 += [nn.ReLU(inplace=True)]
        self.conv1_3 = nn.Sequential(*ly1_2)
        # self.conv1_4 = nn.Sequential(*ly1_2)

    def forward(self, x):
        """
        Forward pass through dilated convolution branch
        
        Args:
            x: Input feature tensor
            
        Returns:
            conv1_2: Direct processed features
            conv1_3: Features processed through dilated convolution layers
        """
        # Process through masked convolution and feature layers
        y0 = self.head(x)
        
        # Branch 1: Direct feature processing
        conv1_1 = self.conv1_1(y0)
        
        # Branch 2: Process through dilated convolution layers
        y1 = self.body(conv1_1)
        
        # Feature fusion: concatenate direct and processed features
        cat0 = torch.cat([conv1_1, y1], dim=1)
        
        # Final feature processing
        conv1_3 = self.conv1_3(cat0)  # Combined DCL result
        
        # Direct feature branch
        conv1_2 = self.conv1_2(y0)
        
        return conv1_2, conv1_3


class DC_branchl2(nn.Module):
    """
    Second stage dilated convolution branch
    
    This module implements the second stage of dilated convolution processing
    with multiple DCL modules for deeper feature extraction.
    
    Args:
        stride: Dilation stride for convolutions
        in_ch: Number of input channels
        num_module: Number of dilated convolution modules
    """
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        ly = []
        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]  # Multiple DCL layers
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]  # Final feature adjustment
        ly += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        """
        Forward pass through second stage DCL layers
        
        Args:
            x: Input feature tensor
            
        Returns:
            Processed feature tensor
        """
        return self.body(x)


class DCl(nn.Module):
    """
    Dilated Convolution Layer with Residual Connection
    
    This module implements a dilated convolution block with residual connection
    for expanded receptive field while maintaining spatial resolution.
    
    Args:
        stride: Dilation rate for the convolution
        in_ch: Number of input/output channels
    """
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        # Dilated convolution for expanded receptive field
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        # 1x1 convolution for feature adjustment
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input feature tensor
            
        Returns:
            Output with residual connection: x + processed_features
        """
        return x + self.body(x)  # Residual connection
if __name__ == '__main__':
    model = MMBSN()
    x = torch.randn(1, 3, 192, 192)
    print(model(x).shape)