from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import random


def np2tensor(n:np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2,0,1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2,0,1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s'%(n.shape,))

def tensor2np(t:torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    t = t.cpu().detach()

    # gray
    if len(t.shape) == 2:
        return t.permute(1,2,0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.permute(1,2,0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))

def imwrite_tensor(t, name='test.png'):
    cv2.imwrite('./%s'%name, tensor2np(t.cpu()))

def imread_tensor(name='test'):
    return np2tensor(cv2.imread('./%s'%name))

def rot_hflip_img(img:torch.Tensor, rot_times:int=0, hflip:int=0):
    '''
    rotate '90 x times degree' & horizontal flip image 
    (shape of img: b,c,h,w or c,h,w)
    '''
    b=0 if len(img.shape)==3 else 1
    # no flip
    if hflip % 2 == 0:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(b+1).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(b+2).flip(b+1)
        # 270 degrees
        else:               
            return img.flip(b+2).transpose(b+1,b+2)
    # horizontal flip
    else:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img.flip(b+2)
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(b+1).flip(b+2).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(b+1)
        # 270 degrees
        else:               
            return img.transpose(b+1,b+2)

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def randomTransform(x: torch.Tensor, pd_factor: int):
    """
    Apply random transformations (rotation, flipping) to each sub-image after PD
    Args:
        x: Feature map after PD, shape [b,c,h,w]
        pd_factor: Number of small image blocks per row
    Returns:
        transformed_x: Transformed feature map
        transform_ops: List of transformation operations, each element is (rot_times, flip_type)
            rot_times: Number of rotations (each 90 degrees)
            flip_type: Flip type
                0: No flip
                1: Horizontal flip
                2: Vertical flip
                3: Main diagonal flip (top-left to bottom-right)
                4: Anti-diagonal flip (bottom-left to top-right)
    """
    b, c, h, w = x.shape
    assert h % pd_factor == 0, f"dim[-2] of input x {h} cannot be divided by {pd_factor}"
    assert w % pd_factor == 0, f"dim[-1] of input x {w} cannot be divided by {pd_factor}"
    
    # Calculate the size of each sub-image
    sub_h = h // pd_factor
    sub_w = w // pd_factor
    
    # Initialize output tensor
    transformed_x = x.clone()
    transform_ops = []
    
    # Apply random transformations to each sub-image
    for i in range(pd_factor):
        for j in range(pd_factor):
            # Randomly generate transformation parameters
            rot_times = np.random.randint(0, 4)    # 0,1,2,3 correspond to 0,90,180,270 degrees
            flip_type = np.random.randint(0, 5)    # 0,1,2,3,4 correspond to no flip, horizontal, vertical, main diagonal, anti-diagonal
            transform_ops.append((rot_times, flip_type))
            
            # Extract current sub-image
            start_h = i * sub_h
            start_w = j * sub_w
            sub_img = transformed_x[..., start_h:start_h+sub_h, start_w:start_w+sub_w]
            
            # Apply transformations (note order: flip first, then rotate)
            if flip_type == 1:  # Horizontal flip
                sub_img = sub_img.flip(-1)
            elif flip_type == 2:  # Vertical flip
                sub_img = sub_img.flip(-2)
            elif flip_type == 3:  # Main diagonal flip
                sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-1)
            elif flip_type == 4:  # Anti-diagonal flip
                sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-2)
            
            if rot_times > 0:
                sub_img = torch.rot90(sub_img, rot_times, [-2, -1])
            
            # Put back the transformed sub-image
            transformed_x[..., start_h:start_h+sub_h, start_w:start_w+sub_w] = sub_img
            
    return transformed_x, transform_ops

def inverseTransform(x: torch.Tensor, transform_ops: list, pd_factor: int):
    """
    Restore the transformations from randomTransform
    Args:
        x: Transformed feature map
        transform_ops: List of transformation operations
        pd_factor: Number of small image blocks per row
    Returns:
        restored_x: Restored feature map
    """
    b, c, h, w = x.shape
    assert h % pd_factor == 0, f"dim[-2] of input x {h} cannot be divided by {pd_factor}"
    assert w % pd_factor == 0, f"dim[-1] of input x {w} cannot be divided by {pd_factor}"
    
    # Calculate the size of each sub-image
    sub_h = h // pd_factor
    sub_w = w // pd_factor
    
    # Initialize output tensor
    restored_x = x.clone()
    
    # Apply inverse transformations to each sub-image
    for idx, (rot_times, flip_type) in enumerate(transform_ops):
        i = idx // pd_factor
        j = idx % pd_factor
        
        # Extract current sub-image
        start_h = i * sub_h
        start_w = j * sub_w
        sub_img = restored_x[..., start_h:start_h+sub_h, start_w:start_w+sub_w]
        
        # Apply inverse transformations (note order is reverse of forward transform: inverse rotation first, then inverse flip)
        if rot_times > 0:
            sub_img = torch.rot90(sub_img, -rot_times, [-2, -1])
            
        if flip_type == 1:  # Horizontal flip
            sub_img = sub_img.flip(-1)
        elif flip_type == 2:  # Vertical flip
            sub_img = sub_img.flip(-2)
        elif flip_type == 3:  # Main diagonal flip
            sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-1)
        elif flip_type == 4:  # Anti-diagonal flip
            sub_img = sub_img.transpose(-1, -2).flip(-1).flip(-2)
        
        # Put back the restored sub-image
        restored_x[..., start_h:start_h+sub_h, start_w:start_w+sub_w] = sub_img
            
    return restored_x


def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def psnr(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping & chnage type to uint8
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)

    return peak_signal_noise_ratio(img1, img2, data_range=255)

def ssim(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping
    img2 = np.clip(img2, 0, 255)
    img1 = np.clip(img1, 0, 255)

    # Calculate SSIM with correct channel axis (2 for HWC format)
    return structural_similarity(img1, img2, data_range=255, channel_axis=2)

def ssim_new(img1, img2):
    '''
    image value range : [0 - 1]
    clipping for model output
    
    Note: This function calculates SSIM based on [0-1] range images, typically produces higher values
    Closer to SSIM values calculated in papers using [0-1] range (generally in 0.9+ range)
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # Convert [0-255] range to [0-1] range
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    
    # numpy value cliping to [0-1]
    img2 = np.clip(img2, 0, 1)
    img1 = np.clip(img1, 0, 1)

    # Calculate SSIM with correct channel axis (2 for HWC format)
    # Calculate SSIM for [0-1] range using data_range=1
    return structural_similarity(img1, img2, data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

def ssim_pytorch(img1, img2, window_size=11, sigma=1.5):
    '''
    PyTorch version of SSIM implementation, more consistent with mainstream paper calculations
    Parameters:
        img1, img2: Input images, tensor format [N, C, H, W]
        window_size: Gaussian window size
        sigma: Standard deviation of Gaussian window
    Returns:
        SSIM value, range [0,1], 1 means completely identical
    '''
    # Ensure input is 4D tensor [N, C, H, W]
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # Ensure they are PyTorch tensors
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
    
    # Move to same device as img1
    device = img1.device
    
    # Limit image value range to [0,1]
    if img1.max() > 1.0 or img2.max() > 1.0:
        img1 = img1 / 255.0
        img2 = img2 / 255.0
    
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Get number of channels
    _, channels, _, _ = img1.shape
    
    # Create Gaussian window
    window = get_gaussian_2d_filter(window_size, sigma, channels, device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variance and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # SSIM stability factors
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return mean value
    return ssim_map.mean()



def lpips2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # print(img1.is_cuda, img2.is_cuda)

    device = torch.device("cuda:0")

    lpips_model = lpips.LPIPS(net="vgg").to(device)#alex
    distance = lpips_model(img1, img2)
    return distance.item()

def dists2(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''


    device = torch.device("cuda:0")

    D = DISTS().to(device)

    dists_value = D(img1, img2)

    return dists_value.item()

def get_gaussian_2d_filter(window_size, sigma, channel=1, device=torch.device('cpu')):
    '''
    return 2d gaussian filter window as tensor form
    Arg:
        window_size : filter window size
        sigma : standard deviation
    '''
    gauss = torch.ones(window_size, device=device)
    for x in range(window_size): gauss[x] = exp(-(x - window_size//2)**2/float(2*sigma**2))
    gauss = gauss.unsqueeze(1)
    #gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=device).unsqueeze(1)
    filter2d = gauss.mm(gauss.t()).float()
    filter2d = (filter2d/filter2d.sum()).unsqueeze(0).unsqueeze(0)
    return filter2d.expand(channel, 1, window_size, window_size)

def get_mean_2d_filter(window_size, channel=1, device=torch.device('cpu')):
    '''
    return 2d mean filter as tensor form
    Args:
        window_size : filter window size
    '''
    window = torch.ones((window_size, window_size), device=device)
    window = (window/window.sum()).unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size)

def mean_conv2d(x, window_size=None, window=None, filter_type='gau', sigma=None, keep_sigma=False, padd=True):
    '''
    color channel-wise 2d mean or gaussian convolution
    Args:
        x : input image
        window_size : filter window size
        filter_type(opt) : 'gau' or 'mean'
        sigma : standard deviation of gaussian filter
    '''
    b_x = x.unsqueeze(0) if len(x.shape) == 3 else x

    if window is None:
        if sigma is None: sigma = (window_size-1)/6
        if filter_type == 'gau':
            window = get_gaussian_2d_filter(window_size, sigma=sigma, channel=b_x.shape[1], device=x.device)
        else:
            window = get_mean_2d_filter(window_size, channel=b_x.shape[1], device=x.device)
    else:
        window_size = window.shape[-1]

    if padd:
        pl = (window_size-1)//2
        b_x = F.pad(b_x, (pl,pl,pl,pl), 'reflect')

    m_b_x = F.conv2d(b_x, window, groups=b_x.shape[1])

    if keep_sigma:
        m_b_x /= (window**2).sum().sqrt()

    if len(x.shape) == 4:
        return m_b_x
    elif len(x.shape) == 3:
        return m_b_x.squeeze(0)
    else:
        raise ValueError('input image shape is not correct')

def pd_down(x: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    """Pixel downsampling
    Shape transformation process:
    1. Input x: [b,c,h,w]
    2. pixel_unshuffle: [b,c,h,w] -> [b,c*pd_factor*pd_factor,h/pd_factor,w/pd_factor]
    3. view: [b,c*pd_factor*pd_factor,h/pd_factor,w/pd_factor] -> [b,c,pd_factor,pd_factor,h/pd_factor,w/pd_factor]
    4. permute: [b,c,pd_factor,pd_factor,h/pd_factor,w/pd_factor] -> [b,pd_factor,pd_factor,c,h/pd_factor,w/pd_factor]
    5. reshape: [b,pd_factor,pd_factor,c,h/pd_factor,w/pd_factor] -> [b*pd_factor*pd_factor,c,h/pd_factor,w/pd_factor]
    
    Args:
        x: Input tensor [b,c,h,w]
        pd_factor: Downsampling factor
        pad: Padding size
    Returns:
        Downsampled tensor [b*pd_factor*pd_factor,c,h/pd_factor,w/pd_factor]
    """
    b, c, h, w = x.shape
    # Step 1-2: pixel_unshuffle
    x_down = F.pixel_unshuffle(x, pd_factor)  # [b,c*pd_factor*pd_factor,h/pd_factor,w/pd_factor]
    # Step 3-5: Rearrange tensor dimensions
    out = x_down.view(b, c, pd_factor, pd_factor, h // pd_factor, w // pd_factor)  # [b,c,pd_factor,pd_factor,h/pd_factor,w/pd_factor]
    out = out.permute(0, 2, 3, 1, 4, 5)  # [b,pd_factor,pd_factor,c,h/pd_factor,w/pd_factor]
    out = out.reshape(b * pd_factor * pd_factor, c, h // pd_factor, w // pd_factor)  # [b*pd_factor*pd_factor,c,h/pd_factor,w/pd_factor]
    return out


def pd_up(out: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    """Pixel upsampling
    Shape transformation process:
    1. Input out: [b*pd_factor*pd_factor,c,h/pd_factor,w/pd_factor]
    2. view: -> [b,pd_factor,pd_factor,c,h/pd_factor,w/pd_factor]
    3. permute: -> [b,c,pd_factor,pd_factor,h/pd_factor,w/pd_factor]
    4. reshape: -> [b,c*pd_factor*pd_factor,h/pd_factor,w/pd_factor]
    5. pixel_shuffle: -> [b,c,h,w]
    
    Args:
        out: Input tensor [b*pd_factor*pd_factor,c,h/pd_factor,w/pd_factor]
        pd_factor: Upsampling factor
        pad: Padding size
    Returns:
        Upsampled tensor [b,c,h,w]
    """
    b, c, h, w = out.shape
    # Step 1-2: Rearrange tensor dimensions
    x_down = out.view(b // (pd_factor ** 2), pd_factor, pd_factor, c, h, w)  # [b,pd_factor,pd_factor,c,h/pd_factor,w/pd_factor]
    # Step 3-4: Adjust dimension order and reshape
    x_down = x_down.permute(0, 3, 1, 2, 4, 5)  # [b,c,pd_factor,pd_factor,h/pd_factor,w/pd_factor]
    x_down = x_down.reshape(b // (pd_factor ** 2), c * pd_factor * pd_factor, h, w)  # [b,c*pd_factor*pd_factor,h/pd_factor,w/pd_factor]
    # Step 5: pixel_shuffle upsampling
    x_up = F.pixel_shuffle(x_down, pd_factor)  # [b,c,h,w]
    return x_up

# Space-to-depth transformation function, divide image into blocks by block_size and reorganize
def space_to_depth(x, block_size):
    # Space-to-depth transformation function, divide image into blocks by block_size and reorganize
    # For example: when block_size=5, input x shape is (16, 3, 160, 160)
    # Output shape is (16, 3*5*5, 160/5, 160/5) = (16, 75, 32, 32)
    n, c, h, w = x.size()  # Get input dimensions
    # Use unfold operation to divide image into blocks
    # unfold operation divides image into block_size×block_size blocks, each block as a separate feature
    # This step is the foundation of RSG strategy, allowing us to reorganize spatial structure of images
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    # Reshape tensor, converting spatial dimensions to channel dimensions
    # From (N, C×block_size², H×W/block_size²) to (N, C×block_size², H/block_size, W/block_size)
    # This transformation distributes originally spatially adjacent pixels across different channels
    # Facilitating subsequent random sampling operations, key step for implementing RSG strategy
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


# Generate mask pairs for random sub-image generation (RSG)
def generate_mask_pair(img, p):
    # p=5, divide image into 5×5=25 sub-blocks
    # Input image shape is (N, C, H, W), e.g., (16, 3, 160, 160)
    # Output sub-image shape is (N, C, H/5, W/5), e.g., (16, 3, 32, 32)
    n, c, h, w = img.shape  # Get image dimensions
    mask = []
    # Create p*p masks
    # Each mask will be used to select specific pixels from original image to build sub-images
    for i in range(p*p):
        mask.append(torch.zeros(size=(n * h // p * w // p * p*p, ),
                        dtype=torch.bool,
                        device=img.device))

    # Create random index pairs, this is the core part of RSG strategy
    # Through random shuffling of indices, ensure pixels in each sub-image are randomly selected
    rd_pair_idx = np.zeros([n * h // p * w // p, p*p], dtype=np.int64)
    for i in range(n * h // p * w // p):
        array = np.arange(p*p)  # Create array from 0 to p*p-1
        np.random.shuffle(array)  # Randomly shuffle array to achieve random sampling
        rd_pair_idx[i, :] = array  # Store shuffled array into index pairs
    rd_pair_idx = torch.from_numpy(rd_pair_idx).cuda()  # Convert to CUDA tensor

    # Adjust index range to ensure indices can correctly map to flattened image data
    # This step converts local indices to global indices for correct pixel positioning in flattened tensor
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // p * w // p * p*p,
                                step=p*p,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # Set mask values, set randomly selected index positions to 1
    # These positions with value 1 will be used to construct sub-images
    for i in range(p*p):
        mask[i][rd_pair_idx[:, i]] = 1
    return mask


# Generate sub-images based on masks
def generate_subimages(img, mask, p):
    # p=5, divide 160×160 image into 25 sub-images of 32×32
    # Input image shape is (16, 3, 160, 160)
    # Output 25 sub-images, each with shape (16, 3, 32, 32)
    n, c, h, w = img.shape  # Get image dimensions
    # Create sub-image tensor with size 1/p of original image
    # These sub-images will be used for loss calculation in self-supervised learning
    subimage = torch.zeros(n,
                           c,
                           h // p,
                           w // p,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # Process each channel
    for i in range(c):
        # Use space_to_depth to divide image into blocks, converting spatial dimensions to channel dimensions
        # This step reorganizes image into smaller blocks for subsequent random sampling
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=p)
        # Adjust tensor dimension order and flatten for mask indexing
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        # Select pixels based on mask and reshape into sub-images
        # This step implements random sampling, selecting specific pixels from original image to build sub-images
        # Since masks are randomly generated, pixels in each sub-image are also random
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // p, w // p, 1).permute(0, 3, 1, 2)
    return subimage

def random_denoise(img: torch.Tensor, methods: list = None) -> torch.Tensor:
    """
    Randomly select traditional denoising methods or no denoising
    Args:
        img: Input image with shape [b, c, h, w]
        methods: Optional list of denoising methods, default ['gaussian', 'median', 'bilateral', 'none']
    Returns:
        Processed image
    """
    if methods is None:
        methods = ['gaussian', 'median', 'bilateral', 'none']
    
    # Randomly select a denoising method
    method = random.choice(methods)
    
    # Convert image from Tensor to numpy
    img_np = tensor2np(img)
    
    # Apply denoising method
    if method == 'gaussian':
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
    elif method == 'median':
        img_np = cv2.medianBlur(img_np, 5)
    elif method == 'bilateral':
        img_np = cv2.bilateralFilter(img_np, 9, 75, 75)
    # 'none' method does no processing
    
    # Convert image back from numpy to Tensor
    img_tensor = np2tensor(img_np)
    
    return img_tensor

def random_shuffle_2x2(x: torch.Tensor):
    """Random shuffle image using 2x2 window with stride 2
    Efficient 2x2 window shuffling using unfold and fold operations
    
    Args:
        x: Input tensor [B,C,H,W]
    Returns:
        shuffled: Shuffled tensor [B,C,H,W]
    """
    B, C, H, W = x.shape
    window_size = 2
    
    # Ensure image dimensions are multiples of 2
    assert H % window_size == 0 and W % window_size == 0, \
        f"Image size must be multiple of {window_size}, current size is {H}x{W}"
    
    # Reshape tensor for batch operations [B*C,1,H,W]
    x_reshaped = x.view(B*C, 1, H, W)
    
    # Use unfold to split image into 2x2 blocks
    # Output shape: [B*C, 1, H/2*W/2, 4]
    patches = F.unfold(x_reshaped, kernel_size=window_size, stride=window_size)
    
    # Reshape for convenient shuffling [B*C*H/2*W/2, 4]
    patches = patches.transpose(1, 2).reshape(-1, window_size*window_size)
    
    # Generate random indices for shuffling
    # Generate independent random permutations for each 2x2 block
    idx = torch.rand(patches.shape[0], window_size*window_size, device=x.device).argsort(dim=1)
    
    # Apply random shuffling
    shuffled_patches = torch.gather(patches, 1, idx)
    
    # Reshape back to original unfold format [B*C, 4, H/2*W/2]
    shuffled_patches = shuffled_patches.view(B*C, -1, window_size*window_size).transpose(1, 2)
    
    # Use fold operation to reconstruct image
    shuffled = F.fold(shuffled_patches, 
                     output_size=(H, W),
                     kernel_size=window_size,
                     stride=window_size)
    
    # Restore original shape [B,C,H,W]
    shuffled = shuffled.view(B, C, H, W)
    
    return shuffled


def adaptive_shuffle_by_std(x: torch.Tensor, std_threshold=5, denoised=None, visualize=False):
    """Adaptively shuffle image based on local standard deviation
    Using non-overlapping 3x3 windows for adaptive shuffling
    
    Args:
        x: Input tensor [32,C,H,W] original downsampled image
        std_threshold: Standard deviation threshold
        denoised: Denoised image for std calculation [32,C,H,W]
        visualize: Whether to visualize shuffling process
    Returns:
        output: Shuffled tensor [32,C,H,W]
        vis_dict: Visualization dictionary (if visualize=True)
    """
    b, c, h, w = x.shape
    window_size = 3
    stride = 3
    
    # Save original dimensions
    orig_h, orig_w = h, w
    
    # Calculate pixels to be truncated
    h_remainder = h % window_size
    w_remainder = w % window_size
    
    # Save parts to be truncated
    h_trunc = None
    w_trunc = None
    
    # If height is not a multiple of window_size, truncate excess pixels from bottom
    if h_remainder > 0:
        h_trunc = x[:, :, -h_remainder:, :]
        x = x[:, :, :-h_remainder, :]
        if denoised is not None:
            denoised = denoised[:, :, :-h_remainder, :]
    
    # If width is not a multiple of window_size, truncate excess pixels from right
    if w_remainder > 0:
        w_trunc = x[:, :, :, -w_remainder:]
        x = x[:, :, :, :-w_remainder]
        if denoised is not None:
            denoised = denoised[:, :, :, :-w_remainder]
    
    # Now process the main part that can be divided by window_size
    b, c, h, w = x.shape
    
    # Extract patches from denoised image for standard deviation calculation
    denoised_patches = F.unfold(denoised, kernel_size=window_size, stride=stride)  # [32, C*9, N_patches]
    n_patches = denoised_patches.size(-1)
    denoised_patches = denoised_patches.transpose(1, 2)  # [32, N_patches, C*9]
    denoised_patches = denoised_patches.reshape(b, n_patches, c, window_size*window_size)  # [32, N_patches, C, 9]
    
    # Calculate standard deviation for each patch
    std_values = torch.std(denoised_patches, dim=-1).mean(dim=-1)  # [32, N_patches]
    
    # Extract patches from original image for shuffling
    patches = F.unfold(x, kernel_size=window_size, stride=stride)  # [32, C*9, N_patches]
    patches = patches.transpose(1, 2)  # [32, N_patches, C*9]
    patches = patches.reshape(b, n_patches, c, window_size*window_size)  # [32, N_patches, C, 9]
    
    # Create shuffle mask
    shuffle_mask = (std_values <= std_threshold)  # [32, N_patches]
    
    # Generate random permutations for all patches that need shuffling
    rand_indices = torch.rand(b, n_patches, window_size*window_size, device=x.device)
    rand_indices = rand_indices.argsort(dim=-1)  # [32, N_patches, 9]
    
    # Create base indices for gather operation
    base_indices = torch.arange(window_size*window_size, device=x.device)
    base_indices = base_indices.view(1, 1, -1).expand(b, n_patches, -1)  # [32, N_patches, 9]
    
    # Select random indices or original indices based on shuffle_mask
    final_indices = torch.where(shuffle_mask.unsqueeze(-1), rand_indices, base_indices)  # [32, N_patches, 9]
    
    # Apply shuffling
    shuffled_patches = torch.zeros_like(patches)  # [32, N_patches, C, 9]
    for i in range(b):
        for j in range(n_patches):
            idx = final_indices[i, j]  # [9]
            shuffled_patches[i, j] = patches[i, j, :, idx]  # [C, 9]
    
    # Reconstruct image
    shuffled_patches = shuffled_patches.reshape(b, n_patches, c * window_size * window_size)  # [32, N_patches, C*9]
    shuffled_patches = shuffled_patches.transpose(1, 2)  # [32, C*9, N_patches]
    output = F.fold(shuffled_patches,
                    output_size=(h, w),
                    kernel_size=window_size,
                    stride=stride)
    
    # Merge truncated parts back to restore original dimensions
    if h_remainder > 0 or w_remainder > 0:
        # Initialize final output tensor
        final_output = torch.zeros((b, c, orig_h, orig_w), device=x.device)
        
        # Place main part
        final_output[:, :, :h, :w] = output
        
        # Process and add truncated parts
        if h_trunc is not None:
            final_output[:, :, -h_remainder:, :w] = h_trunc[:, :, :, :w]
        
        if w_trunc is not None:
            if h_trunc is not None:
                # Handle bottom-right corner intersection
                corner = x[:, :, -h_remainder:, -w_remainder:]
                final_output[:, :, -h_remainder:, -w_remainder:] = corner
            
            final_output[:, :, :h, -w_remainder:] = w_trunc
        
        output = final_output
    
    if not visualize:
        return output
        
    # Create visualization mask image
    mask_img = torch.zeros((b, 1, orig_h, orig_w), device=x.device)
    for i in range(h // window_size):
        for j in range(w // window_size):
            patch_idx = i * (w // window_size) + j
            if shuffle_mask[0, patch_idx]:  # Only take mask from first batch
                mask_img[0, :, i*window_size:(i+1)*window_size, 
                        j*window_size:(j+1)*window_size] = 1.0
    # Generate unconstrained shuffling (all patches shuffled)
    all_shuffle_patches = torch.zeros_like(patches)
    for i in range(b):
        for j in range(n_patches):
            idx = rand_indices[i, j]
            all_shuffle_patches[i, j] = patches[i, j, :, idx]
    
    # Reconstruct unconstrained shuffled image
    all_shuffle_patches = all_shuffle_patches.reshape(b, n_patches, c * window_size * window_size)
    all_shuffle_patches = all_shuffle_patches.transpose(1, 2)
    all_shuffle_output = F.fold(all_shuffle_patches,
                                output_size=(h, w),
                                kernel_size=window_size,
                                stride=stride)
    
    # Return information needed for visualization
    vis_dict = {
        'original': x[0].cpu(),  # Only take first image
        'shuffled': output[0].cpu(),
        'all_shuffled': all_shuffle_output[0].cpu(),
        'mask': mask_img[0].cpu(),
        'std_values': std_values[0].cpu(),  # [N_patches]
        'window_size': window_size
    }
    
    
    # Generate visualization results immediately
    visualize_shuffle_results(vis_dict)
    
    return output

def visualize_shuffle_results(vis_dict):
    """Visualize shuffling results
    Args:
        vis_dict: Dictionary containing information needed for visualization
    """
    import matplotlib.pyplot as plt
    
    # Prepare four subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Display original image
    orig_img = tensor2np(vis_dict['original'])
    orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)  # Ensure data range is 0-255
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display adaptive shuffled image
    shuffled_img = tensor2np(vis_dict['shuffled'])
    shuffled_img = np.clip(shuffled_img, 0, 255).astype(np.uint8)  # Ensure data range is 0-255
    axes[1].imshow(shuffled_img)
    axes[1].set_title('Adaptive 3x3 Shuffled Image')
    axes[1].axis('off')
    
    # Display shuffle mask (red indicates shuffled regions)
    mask = vis_dict['mask'].squeeze().numpy()
    orig_img_with_mask = orig_img.copy()
    window_size = vis_dict['window_size']
    
    # Mark shuffled regions with red boxes on original image
    h, w = mask.shape
    for i in range(h // window_size):
        for j in range(w // window_size):
            if mask[i*window_size, j*window_size] > 0.5:
                # Draw red border
                y, x = i*window_size, j*window_size
                orig_img_with_mask[y:y+window_size, x, :] = [255, 0, 0]
                orig_img_with_mask[y:y+window_size, x+window_size-1, :] = [255, 0, 0]
                orig_img_with_mask[y, x:x+window_size, :] = [255, 0, 0]
                orig_img_with_mask[y+window_size-1, x:x+window_size, :] = [255, 0, 0]
    
    axes[2].imshow(orig_img_with_mask)
    axes[2].set_title('Shuffled Region Annotation (Red Box)')
    axes[2].axis('off')
    
    # Display unconstrained shuffled image
    all_shuffled_img = tensor2np(vis_dict['all_shuffled'])
    all_shuffled_img = np.clip(all_shuffled_img, 0, 255).astype(np.uint8)
    axes[3].imshow(all_shuffled_img)
    axes[3].set_title('Unconstrained 3x3 Shuffled Image')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()

def compute_local_std(img, window_size=3):
    """Compute local standard deviation"""
    
    pad = window_size // 2
    # Add channel dimension if single-channel
    if len(img.shape) == 3:  # [B,H,W]
        img = img.unsqueeze(1)  # [B,1,H,W]
    
    N, C, H, W = img.shape
    img_padded = F.pad(img, [pad] * 4, mode='reflect')
    
    # Use unfold to get pixels of each window
    patches = F.unfold(img_padded, kernel_size=window_size).view(N, C, window_size*window_size, H, W)
    
    # Compute standard deviation of each window
    std = torch.std(patches, dim=2, keepdim=True)
    return std.squeeze(2)  # [N,C,H,W]

def depth_to_space(x, block_size):
    """
    Restore channel dimension information to spatial dimension
    
    Shape transformation process:
    Input:  [n, c * block_size * block_size, h, w]
    Output: [n, c, h * block_size, w * block_size]
    
    Example when block_size=2:
    [n, c*4, 4, 4] -> [n, c, 8, 8]
    
    Uses pixel_shuffle operation, inverse operation of space_to_depth
    """
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img, width=4, mask_type='random'):
    """
    Generate mask tensor to specify positions that need to be predicted
    
    Shape transformation process:
    1. Input: img [n, c, h, w]
    2. Initial mask: [(n * h/width * w/width * width^2)] one-dimensional tensor
    3. Final output: [n, h, w] binary mask same size as original image
    
    Args:
        img: Input image
        width: Basic unit size of mask (default 4, i.e., 4x4)
        mask_type: Mask generation method ('random','batch','all','fix_*')
    """
    n, c, h, w = img.shape
    # Create one-dimensional mask tensor, length is n*(h/w)*(w/w)*w^2
    # Here w^2 is because each width*width block needs w^2 mask versions
    mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                      dtype=torch.int64,
                      device=img.device)
    
    # Create width^2 indices, corresponding to positions within each block
    idx_list = torch.arange(
        0, width**2, 1, dtype=torch.int64, device=img.device)
    
    # Create random index tensor
    rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                        dtype=torch.int64,
                        device=img.device)

    # Generate random indices based on different mask_type
    if mask_type == 'random':  # Random selection for each position
        torch.randint(low=0,
                     high=len(idx_list),
                     size=(n * h // width * w // width, ),
                     device=img.device,
                     generator=get_generator(device=img.device),
                     out=rd_idx)
    elif mask_type == 'batch':  # Each batch uses same random pattern
        rd_idx = torch.randint(low=0,
                             high=len(idx_list),
                             size=(n, ),
                             device=img.device,
                             generator=get_generator(device=img.device)).repeat(h // width * w // width)
    elif mask_type == 'all':    # All positions use same random pattern
        rd_idx = torch.randint(low=0,
                             high=len(idx_list),
                             size=(1, ),
                             device=img.device,
                             generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:    # Use fixed position mask
        index = mask_type.split('_')[-1]
        index = torch.from_numpy(np.array(index).astype(
            np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    # Generate final mask indices
    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                               end=n * h // width * w // width * width**2,
                               step=width**2,
                               dtype=torch.int64,
                               device=img.device)

    # Set selected positions to 1
    mask[rd_pair_idx] = 1

    # Reshape mask dimensions and convert back to spatial domain
    # 1. view: [n, h/width, w/width, width^2]
    # 2. permute: [n, width^2, h/width, w/width]
    # 3. depth_to_space: [n, 1, h, w]
    mask = depth_to_space(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

    return mask

class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        
        # Return processed image and corresponding mask
        return net_input, mask

    def train(self, img):
        """
        Generate multiple mask versions of training images
        
        Args:
            img: Input image tensor, shape [n, c, h, w]
            
        Returns:
            tensors: Mask-processed image tensor collection, shape [n*width^2, c, h, w]
            masks: Corresponding mask tensor collection, shape [n*width^2, 1, h, w]
        """
        n, c, h, w = img.shape
        # Create tensor to store all mask version images, width^2 represents total number of masks
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        # Create tensor to store all masks
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        
        # Process each mask position
        for i in range(self.width**2):
            # Process image using fixed position mask
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            # Save mask-processed image
            tensors[:, i, ...] = x
            # Save corresponding mask
            masks[:, i, ...] = mask
        
        # Reshape tensor dimensions, merge batch and mask count dimensions
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks



def interpolate_mask(tensor, mask, mask_inv):
    """
    Process masked regions using interpolation method
    
    Args:
        tensor: Input image tensor [n, c, h, w]
        mask: Binary mask tensor, 1 indicates positions to keep
        mask_inv: Inverse mask of mask, 1 indicates positions to interpolate
    """
    # Get shape and device information of input tensor
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    
    # Define 3x3 interpolation convolution kernel
    # [[0.5, 1.0, 0.5],
    #  [1.0, 0.0, 1.0],
    #  [0.5, 1.0, 0.5]]
    # Center point weight is 0, surrounding 8 points weights set by distance
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    # Convert convolution kernel to 4D tensor [1, 1, 3, 3]
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    # Normalize convolution kernel weights
    kernel = kernel / kernel.sum()

    # Use convolution for interpolation
    # 1. Reshape tensor to [n*c, 1, h, w] for channel-wise convolution
    # 2. Use defined convolution kernel for convolution operation, padding=1 keeps size unchanged
    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    # Combine results:
    # 1. filtered_tensor * mask: Keep interpolated values at masked positions
    # 2. tensor * mask_inv: Keep original image values at unmasked positions
    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv

def ms_ssim_pytorch(img1, img2, window_size=11, sigma=1.5, weights=None, levels=5):
    '''
    PyTorch version of MS-SSIM (Multi-Scale Structural Similarity) implementation
    MS-SSIM calculates SSIM at multiple scales, typically better reflects human visual system perception of image quality compared to single-scale SSIM
    
    Parameters:
        img1, img2: Input images, tensor format [N, C, H, W]
        window_size: Gaussian window size
        sigma: Standard deviation of Gaussian window
        weights: Weights for different scales, default None, will use weights suggested in papers
        levels: Number of scales to compute
    
    Returns:
        MS-SSIM value, range [0,1], 1 means completely identical
    '''
    # Ensure input is 4D tensor [N, C, H, W]
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # Ensure they are PyTorch tensors
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
    
    # Move to same device as img1
    device = img1.device
    
    # Limit image value range to [0,1]
    if img1.max() > 1.0 or img2.max() > 1.0:
        img1 = img1 / 255.0
        img2 = img2 / 255.0
    
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Check if image size meets requirements
    _, _, h, w = img1.shape
    min_size = 2 ** (levels - 1) + 1  # Ensure minimum size meets multi-scale requirements
    if h < min_size or w < min_size:
        levels = int(np.log2(min(h, w))) + 1
    
    # If no weights provided, use default weights suggested in papers
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
        weights = weights[:levels]  # Only take the first few weights needed
        weights = weights / weights.sum()  # Ensure weights sum to 1
    
    # Get number of channels
    _, channels, _, _ = img1.shape
    
    # Create Gaussian window
    window = get_gaussian_2d_filter(window_size, sigma, channels, device)
    
    # Used to store SSIM and contrast sensitivity (CS) for each scale
    mssim = []
    mcs = []
    
    # SSIM stability factors
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Calculate at different scales
    for i in range(levels):
        # Calculate SSIM and CS at current scale
        ssim_map, cs_map = _ssim_tensor(img1, img2, window, window_size, C1, C2, channels)
        
        # Add to list
        mssim.append(ssim_map.mean() if i == levels-1 else None)  # Only save SSIM at last scale
        mcs.append(cs_map.mean())
        
        # If not the last scale, downsample the images
        if i < levels - 1:
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
    
    # Convert to tensor
    mcs = torch.stack(mcs)
    
    # Calculate MS-SSIM
    # For the last scale, we consider both luminance (SSIM) and contrast sensitivity (CS)
    # For other scales, we only consider contrast sensitivity (CS)
    msssim = torch.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
    
    return msssim

def _ssim_tensor(img1, img2, window, window_size, C1, C2, channels):
    '''
    Calculate SSIM and contrast sensitivity (CS) for a single scale
    '''
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variance and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    # For some extreme cases, ensure variance is non-negative
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)
    
    # Calculate luminance similarity (l) and contrast-structure similarity (cs)
    l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    # Combine to calculate SSIM
    ssim_map = l * cs
    
    return ssim_map, cs



def block_pd_down_sampling(x, scale, block_size=1, pad=0):
    """
    Block-based Pixel Shuffle downsampling - directly use block_size×block_size blocks as basic units (GPU efficient version)
    
    Args:
        x: Input tensor, shape [B, C, H, W]
        scale: Downsampling scale, e.g., scale=5 means dividing image into 5×5 sub-images
        block_size: Block size, default 1 (traditional pixel shuffle)
        pad: Number of pixels to pad for each sub-image
        
    Returns:
        Downsampled tensor
    """
    B, C, H, W = x.shape
    
    # Ensure image size can be divided by block_size*scale
    assert H % (block_size * scale) == 0 and W % (block_size * scale) == 0, \
        f"Image size must be divisible by block_size*scale={block_size*scale}"
    
    # If block_size=1, use standard pixel shuffle
    if block_size == 1:
        return pixel_shuffle_down_sampling(x, scale, pad)
    
    # Calculate output size
    h_blocks = H // block_size  # Number of blocks in each dimension
    w_blocks = W // block_size
    out_h = h_blocks // scale  # Number of blocks per sub-image
    out_w = w_blocks // scale
    
    # Create output tensor
    sub_h = out_h * block_size  # Height of each sub-image (pixel count)
    sub_w = out_w * block_size  # 每个子图的宽度（像素数）
    result_h = (sub_h + 2*pad) * scale  # 输出高度（包含所有子图）
    result_w = (sub_w + 2*pad) * scale  # 输出宽度（包含所有子图）
    
    # Create output tensor并填充 - 使用torch.full替代zeros+fill_
    result = torch.full((B, C, result_h, result_w), fill_value=pad, 
                       device=x.device, dtype=x.dtype)
    
    # 使用unfold操作提取块
    # 首先按block_size分块
    x_unfolded = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    # 现在x_unfolded的形状是[B, C, h_blocks, w_blocks, block_size, block_size]
    
    # 预先计算所有子图的位置索引以减少循环中的计算
    sub_positions = []
    for i in range(scale):
        for j in range(scale):
            start_h = i * (sub_h + 2*pad) + pad
            start_w = j * (sub_w + 2*pad) + pad
            sub_positions.append((i, j, start_h, start_w))
    
    # 使用单个批处理操作处理所有块
    for i, j, start_h, start_w in sub_positions:
        # 创建网格索引
        h_indices = torch.arange(out_h, device=x.device)
        w_indices = torch.arange(out_w, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_indices, w_indices, indexing='ij')
        
        # 计算源索引和目标索引
        src_h_indices = grid_h * scale + i
        src_w_indices = grid_w * scale + j
        
        dst_h_base = start_h + grid_h * block_size
        dst_w_base = start_w + grid_w * block_size
        
        # 对所有块进行批处理
        for h_offset in range(block_size):
            for w_offset in range(block_size):
                # 从unfolded张量中提取值
                values = x_unfolded[:, :, 
                                   src_h_indices.flatten(), 
                                   src_w_indices.flatten(), 
                                   h_offset, w_offset]
                
                # 重塑为正确的形状
                values = values.reshape(B, C, out_h, out_w)
                
                # 计算目标位置
                dst_h = dst_h_base + h_offset
                dst_w = dst_w_base + w_offset
                
                # 将值放入结果张量
                result[:, :, dst_h, dst_w] = values
    
    return result

def block_pd_up_sampling(x, scale, block_size=1, pad=0):
    """
    基于Block的Pixel Shuffle上采样 - 直接将block_size×block_size的块作为基本单位 (GPU高效版)
    
    Args:
        x: 下采样后的张量，形状为[B, C, H+2*pad, W+2*pad]
        scale: 上采样尺度，与下采样的scale一致
        block_size: 块大小，与下采样的block_size一致
        pad: 填充像素数，与下采样的pad一致
        
    Returns:
        上采样后的张量，形状为[B, C, H*scale, W*scale]
    """
    # If block_size=1, use standard pixel shuffle
    if block_size == 1:
        return pixel_shuffle_up_sampling(x, scale, pad)
    
    B, C, H_padded, W_padded = x.shape
    
    # 去除padding区域
    H = H_padded - scale * 2*pad
    W = W_padded - scale * 2*pad
    # 计算每个子图的大小
    sub_h = H // scale
    sub_w = W // scale
    
    # Calculate output size
    out_h = sub_h * scale
    out_w = sub_w * scale
    
    # Create output tensor - 直接使用zeros而不是fill_
    result = torch.zeros(B, C, out_h, out_w, device=x.device, dtype=x.dtype)
    
    # 预先计算所有子图的位置索引
    sub_positions = []
    for i in range(scale):
        for j in range(scale):
            src_h = i * (sub_h + 2*pad) + pad
            src_w = j * (sub_w + 2*pad) + pad
            sub_positions.append((i, j, src_h, src_w))
    
    # 批处理所有子图
    for i, j, src_h, src_w in sub_positions:
        # 提取子图
        sub_img = x[:, :, src_h:src_h+sub_h, src_w:src_w+sub_w]
        
        # 将子图转换为块
        blocks = sub_img.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        # 形状为 [B, C, sub_h//block_size, sub_w//block_size, block_size, block_size]
        
        # 计算块的数量
        h_blocks = sub_h // block_size
        w_blocks = sub_w // block_size
        
        # 创建网格索引
        h_indices = torch.arange(h_blocks, device=x.device)
        w_indices = torch.arange(w_blocks, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_indices, w_indices, indexing='ij')
        
        # 计算目标位置
        dst_h_base = (grid_h * scale + i) * block_size
        dst_w_base = (grid_w * scale + j) * block_size
        
        # 对所有块进行批处理
        for h_offset in range(block_size):
            for w_offset in range(block_size):
                # 从blocks张量中提取值
                values = blocks[:, :, 
                              grid_h.flatten(), 
                              grid_w.flatten(), 
                              h_offset, w_offset]
                
                # 重塑为正确的形状
                values = values.reshape(B, C, h_blocks, w_blocks)
                
                # 计算目标位置
                dst_h = dst_h_base + h_offset
                dst_w = dst_w_base + w_offset
                
                # 将值放入结果张量
                result[:, :, dst_h, dst_w] = values
    
    return result[:, :, :H, :W]





def Random_down_sampling(x:torch.Tensor,f:int=2,pad:int=0):
    random_idx = torch.randint(0, 2, size=(1,))
    if random_idx == 0:
        return block_pd_down_sampling(x,scale=f,pad=pad,block_size=1),random_idx
    else:
        return block_pd_down_sampling(x,scale=f,pad=pad,block_size=2),random_idx
    
def Random_up_sampling(x:torch.Tensor,f:int=2,pad:int=0,random_idx:int=0):
    if random_idx == 0:
        return block_pd_up_sampling(x,scale=f,pad=pad,block_size=1)
    else:
        return block_pd_up_sampling(x,scale=f,pad=pad,block_size=2)