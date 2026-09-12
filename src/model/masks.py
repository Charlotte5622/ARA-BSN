import torch.nn as nn
import torch
import torch.nn.functional as F

# Central masked convolution layer
# Mask shape: All-ones matrix with only center point set to 0
# Function: Masks the center position weights of the convolution kernel
class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH//2] = 0
        # if kH == 5:
        #     self.mask[:, :, 1:-1, 1:-1] = 0
        # else:
        #     pass

    def _reset_mask(self):
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Column masked convolution layer
# Mask shape: All-ones matrix with middle row set to 0
# Function: Masks the middle row weights of the convolution kernel
class ColMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Row masked convolution layer
# Mask shape: All-ones matrix with middle column set to 0
# Function: Masks the middle column weights of the convolution kernel
class RowMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Cross-shaped inverse mask convolution layer
# Mask shape: All-ones matrix with middle row and column set to 0
# Function: Masks the cross-shaped region weights of the convolution kernel
class fSzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, kW // 2, :] = 0

        
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Cross-shaped mask convolution layer
# Mask shape: All-zeros matrix with only middle row and column set to 1 (center point is 0)
# Function: Only preserves cross-shaped region weights
class SzMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0


    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# 135-degree diagonal mask convolution layer
# Mask shape: All-ones matrix with main diagonal (top-left to bottom-right) elements set to 0
# Function: Masks diagonal information in 135-degree direction
class angle135MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# 45-degree diagonal mask convolution layer
# Mask shape: All-ones matrix with anti-diagonal (top-right to bottom-left) elements set to 0
# Function: Masks diagonal information in 45-degree direction
class angle45MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, kW -1-i, i] = 0

    def _reset_mask(self):
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Cross-channel mask convolution layer
# Mask shape: Basically zeros, both diagonals set to 1, middle row set to 0
# Function: Only preserves information from both diagonals, but middle row is masked
class chaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
            self.mask[:, :, kW - 1 - i, i] = 1
            self.mask[:, :, kH // 2, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Inverse cross-channel mask convolution layer
# Mask shape: All-ones matrix with both diagonals set to 0
# Function: Masks information from both diagonals
class fchaMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
            self.mask[:, :, kW -1-i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Ring-shaped mask convolution layer
# Mask shape: Edges set to 1, inner region (excluding border) set to 0
# Function: Only preserves edge weights of the convolution kernel
class huiMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 1:-1, 1:-1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# Special pattern mask convolution layer
# Mask shape: Four corners, xy-axis endpoints and center set to 0, other positions set to 1
# Example(5x5): 0 1 0 1 0
#              1 1 1 1 1
#              0 1 0 1 0
#              1 1 1 1 1
#              0 1 0 1 0
class SpecialPatternMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        
        # Set four corners to 0
        self.mask[:, :, 0, 0] = 0
        self.mask[:, :, 0, -1] = 0
        self.mask[:, :, -1, 0] = 0
        self.mask[:, :, -1, -1] = 0
        
        # Set x and y axis endpoints to 0
        self.mask[:, :, kH//2, 0] = 0
        self.mask[:, :, kH//2, -1] = 0
        self.mask[:, :, 0, kW//2] = 0
        self.mask[:, :, -1, kW//2] = 0
        
        # Set center point to 0
        self.mask[:, :, kH//2, kW//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
    

class MultiScaleMask(nn.Module):
    """Multi-scale mask generator
    Generates multiple different masks, each mask covers different regions of the image,
    uses interpolation to fill masked regions
    """
    def __init__(self, scale_num: int = 2):
        """
        Args:
            scale_num: Number of masks to generate, default is 2
        """
        super().__init__()
        self.scale_num = scale_num
        
        # Define 3x3 interpolation convolution kernel
        kernel = torch.tensor([
            [0.5, 1.0, 0.5],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 0.5]
        ], dtype=torch.float32)
        
        # Convert kernel to 4D tensor [1, 1, 3, 3] and register as buffer
        kernel = kernel.view(1, 1, 3, 3)
        # Normalize kernel weights
        self.register_buffer('kernel', kernel / kernel.sum())

    def interpolate_masked_regions(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Use convolution interpolation to fill regions masked by the mask
        
        Args:
            x: Input image [b,c,h,w]
            mask: Mask [b,c,h,w], True indicates positions that need interpolation
        Returns:
            Interpolated image [b,c,h,w]
        """
        b, c, h, w = x.shape
        
        # Prepare masks
        mask = mask.float()  # Convert boolean mask to float
        mask_inv = 1 - mask  # Create inverse mask
        
        # Use convolution for interpolation
        # 1. Reshape tensor to [b*c, 1, h, w] for channel-wise convolution
        # 2. Use defined convolution kernel for convolution operation
        filtered = F.conv2d(
            x.view(b*c, 1, h, w), 
            self.kernel.to(x.device), 
            padding=1
        ).view_as(x)
        
        # Combine results:
        # 1. filtered * mask: Retain interpolated values at masked positions
        # 2. x * mask_inv: Retain original image values at unmasked positions
        return filtered * mask + x * mask_inv

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate multi-scale masks and use interpolation to fill masked regions
        
        Args:
            x: Input tensor, shape [b,c,h,w] or [c,h,w]
        Returns:
            res: Interpolated image, shape [b,c,h,w]
            masks: Mask tensor, shape [b,c,h,w]
        """
        # Handle 3D input
        if (len(x.shape) == 3):
            x = x.unsqueeze(0)  # [c,h,w] -> [1,c,h,w]
        b, c, h, w = x.shape
        
        # Generate random mask indices [b,1,h,w] -> [b,c,h,w]
        mask_index = torch.randint(
            0, self.scale_num, (b, 1, h, w)).expand(-1, c, -1, -1).to(x.device)
        
        # masks = torch.BoolTensor(x.shape).to(x.device)
            
        # Generate boolean mask [b,c,h,w]
        masks = mask_index != 0
        
        # Use interpolation to fill masked regions
        res = self.interpolate_masked_regions(x, masks)
        
        return res, masks
