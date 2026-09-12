import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .HILO import HiLo, analyze_frequency
from .masks import angle45MaskedConv2d
from . import regist_model
from .masks import MultiScaleMask

@regist_model
class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network Light Version (DBSNl)
    
    Key Features:
    1. Uses dilated convolutions to expand receptive field with fewer parameters
    2. Dual-branch structure enhances feature extraction capability
    3. Central masking ensures current pixel is not used for prediction
    4. Residual connections improve gradient flow
    
    Architecture Overview:
    - Head: Initial feature extraction with 1x1 convolution
    - Masked Convolutions: Two branches with different dilation rates
    - Dual Branches: Parallel processing with stride 2 and 3
    - Tail: Progressive channel reduction for final reconstruction
    
    Parameters:
    - in_ch: Number of input channels, default 3 (RGB images)
    - out_ch: Number of output channels, default 3
    - base_ch: Base number of channels, determines network width
    - num_module: Network depth, number of module repetitions
    - new_ch: Number of channels in branch processing
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9, new_ch=128):
        '''
        Initialize Dilated Blind-Spot Network Light Version
        
        Args:
            in_ch      : Number of input channels (default: 3 for RGB)
            out_ch     : Number of output channels (default: 3 for RGB)
            base_ch    : Number of base channels (default: 128)
            num_module : Number of dilated convolution modules (default: 9)
            new_ch     : Number of channels in branch processing (default: 128)
        '''
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divisible by 2"

        # Head feature extraction
        ly = []
        ly += [ nn.Conv2d(in_ch, base_ch, kernel_size=1) ]  # 1x1 conv to adjust channel dimension
        ly += [ nn.ReLU(inplace=True) ]
        self.head = nn.Sequential(*ly)
        
        # Multi-scale mask generator for feature block processing
        self.mask = MultiScaleMask(scale_num=6)
        
        # Create masked convolution layers
        # self.mask_conv1 = MMaskedConv2d_block(base_ch, base_ch, stride=2)
        # self.mask_conv2 = MMaskedConv2d_block(base_ch, base_ch, stride=3)
        self.mask_conv1 = CentralMaskedConv2d(base_ch, base_ch, kernel_size=2*2-1, stride=1, padding=2-1)
        self.mask_conv2 = CentralMaskedConv2d(base_ch, base_ch, kernel_size=2*3-1, stride=1, padding=3-1)
        # Normal convolution layers share parameters with masked convolutions
        self.normal_conv1 = self.mask_conv1
        self.normal_conv2 = self.mask_conv2
        
        # Dual-branch structure for parallel feature processing
        self.branch1 = DC_branchl(2, base_ch, num_module, new_ch=new_ch)  # Branch with stride=2
        self.branch2 = DC_branchl(3, base_ch, num_module, new_ch=new_ch)  # Branch with stride=3

        # Tail reconstruction network
        ly = []
        ly += [ nn.Conv2d(new_ch*2,  new_ch,    kernel_size=1) ]    # Merge dual-branch features
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(new_ch,    new_ch//2, kernel_size=1) ]    # Progressive channel reduction
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(new_ch//2, new_ch//2, kernel_size=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(new_ch//2, out_ch,     kernel_size=1) ]    # Output reconstruction result
        self.tail = nn.Sequential(*ly)
        # ly = []
        # ly += [ nn.Conv2d(base_ch*2,  base_ch,    kernel_size=1) ]    # 合并双分支特征
        # ly += [ nn.ReLU(inplace=True) ]
        # ly += [ nn.Conv2d(base_ch,    base_ch//2, kernel_size=1) ]    # 逐步降低通道数
        # ly += [ nn.ReLU(inplace=True) ]
        # ly += [ nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1) ]
        # ly += [ nn.ReLU(inplace=True) ]
        # ly += [ nn.Conv2d(base_ch//2, out_ch,     kernel_size=1) ]    # 输出重建结果
        # self.tail = nn.Sequential(*ly)

    def forward(self, x, masked=True):
        """
        Forward propagation process
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            masked: Whether to apply masking (True for training/inference, False for analysis)
            
        Returns:
            If masked=True: Denoised output tensor of shape [B, C, H, W]
            If masked=False: Tuple of (denoised output, masks) for analysis
        """
        if masked:
            x = self.head(x)
            # Use masked convolutions for blind-spot property
            x1 = self.mask_conv1(x)
            x2 = self.mask_conv2(x)
        else:
            # Generate masked images and masks [mask_num,b,c,h,w]
            x, masks = self.mask(x)
            x = self.head(x)
            # Use shared parameter convolutions but temporarily remove masks
            with torch.no_grad():
                self.mask_conv1.central_masked.mask.fill_(1)
                self.mask_conv1.angle45_masked.mask.fill_(1)
                self.mask_conv2.central_masked.mask.fill_(1)
                self.mask_conv2.angle45_masked.mask.fill_(1)
            x1 = self.mask_conv1(x)
            x2 = self.mask_conv2(x)
            # Restore masks
            with torch.no_grad():
                self.mask_conv1.central_masked._reset_mask()
                self.mask_conv1.angle45_masked._reset_mask()
                self.mask_conv2.central_masked._reset_mask()
                self.mask_conv2.angle45_masked._reset_mask()
        
        # print(x1.shape)
        # Dual-branch feature extraction
        br1 = self.branch1(x1)  # Process through stride-2 branch
        br2 = self.branch2(x2)  # Process through stride-3 branch
        
        # Feature fusion and final reconstruction
        x = self.tail(torch.cat([br1, br2], dim=1))
        
        return x if masked else (x, masks)

    def _initialize_weights(self):
        """
        Weight initialization method
        Uses normal distribution to initialize convolution layer weights
        Based on He initialization with custom scaling factor
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)

class MMaskedConv2d_block(nn.Module):
    """
    Multi-Masked Convolution Block
    
    Features:
    1. Uses multiple masked convolution layers to process features from different directions
    2. Combines central and angle-based masked convolutions
    3. Maintains blind-spot property while capturing directional information
    
    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels  
        stride: Stride for the masked convolutions
        normal: Whether to use normal convolution (without masking)
    """ 
    def __init__(self, in_ch, out_ch, stride, normal=False):
        super().__init__()

        ly = []
        # Central masked convolution layer
        self.central_masked = CentralMaskedConv2d(in_ch, out_ch, kernel_size=2*stride-1, stride=1, padding=stride-1)
        # 45-degree angle masked convolution layer
        self.angle45_masked = angle45MaskedConv2d(in_ch, out_ch, kernel_size=2*stride-1, stride=1, padding=stride-1)
        
        # Option to disable masking for normal convolution
        if normal:
            self.central_masked.mask.fill_(1)
            self.angle45_masked.mask.fill_(1)
            
        # Feature fusion and processing
        ly += [ nn.Conv2d(2*in_ch, in_ch, kernel_size=1) ]  # Combine features from both masked convs
        # ly += [ DilatedMDTA(dim=2*in_ch, num_heads=4, bias=True, reduction_ratio=2, dilation=stride) ]  # Optional attention
        ly += [ nn.ReLU(inplace=True) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        """
        Forward pass through multi-masked convolution block
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor with combined masked features
        """
        # Concatenate features from central and angle masked convolutions
        return self.body(torch.cat([self.central_masked(x), self.angle45_masked(x)], dim=1))

class DC_branchl(nn.Module):
    """
    Dilated Convolution Branch Network
    
    Features:
    1. Uses central masked convolution to implement blind-spot property
    2. Multi-layer convolutions with residual connections for deep feature construction
    3. Processes features through dilated convolution modules
    
    This branch processes input features through multiple dilated convolution layers
    to expand the receptive field while maintaining spatial resolution.
    
    Args:
        stride: Dilation stride for convolutions
        in_ch: Number of input channels
        num_module: Number of module repetitions
        new_ch: Number of output channels after processing
    """
    def __init__(self, stride, in_ch, num_module, new_ch=128):
        super().__init__()

        ly = []
        
        # Feature adjustment layers
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]  # Initial feature processing
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]  # Further feature adjustment
        ly += [ nn.ReLU(inplace=True) ]
        
        # Add multiple dilated convolution modules
        ly += [ DCl(stride, in_ch) for _ in range(num_module) ]
        
        # ly += [ TransformerBlock(mask_stride=stride,inputdim=128,dim=new_ch,window_size=stride*4) ]  # Optional transformer
        
        # Final feature integration
        ly += [ nn.Conv2d(new_ch, new_ch, kernel_size=1) ]  # Channel adjustment to new_ch
        ly += [ nn.ReLU(inplace=True) ]
        
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        """
        Forward pass through dilated convolution branch
        
        Args:
            x: Input feature tensor
            
        Returns:
            Processed feature tensor with expanded receptive field
        """
        # Process through dilated convolution layers
        out = self.body(x)
        return out

class DCl(nn.Module):
    """
    Dilated Convolution Layer with Residual Connection
    
    Features:
    1. Uses dilated convolution to expand receptive field
    2. Includes residual connection for better gradient flow
    3. Maintains feature map size unchanged
    4. Processes features with expanded context awareness
    
    This module implements a basic building block for dilated convolution networks,
    combining dilated convolution with residual learning for effective feature extraction.
    
    Args:
        stride: Dilation rate for the convolution
        in_ch: Number of input/output channels
    """
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        # Dilated convolution layer for expanded receptive field
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
            Output with residual connection: input + processed_features
        """
        return x + self.body(x)  # Residual connection

class CentralMaskedConv2d(nn.Conv2d):
    """
    Central Masked Convolution Layer
    
    Features:
    1. Adds mask at the center position of convolution kernel
    2. Ensures network doesn't use current pixel for prediction
    3. Implements blind-spot property for self-supervised learning
    
    This layer extends standard Conv2d by applying a mask to the center of the
    convolution kernel, preventing the network from directly copying the center pixel
    and forcing it to learn denoising from surrounding context.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create mask with same shape as weights
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)  # Initialize all positions to 1
        self.mask[:, :, kH//2, kW//2] = 0  # Set center position to 0 (masked)

    def _reset_mask(self):
        """
        Reset mask to initial state
        
        This method restores the mask to its original configuration,
        useful when switching between masked and unmasked modes.
        """
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)  # Reset all positions to 1
        self.mask[:, :, kH//2, kW//2] = 0  # Re-mask center position

    def forward(self, x):
        """
        Forward pass with masked weights
        
        Args:
            x: Input tensor
            
        Returns:
            Convolution output with masked center weights
        """
        # Apply mask to weights before convolution
        self.weight.data *= self.mask
        return super().forward(x)
