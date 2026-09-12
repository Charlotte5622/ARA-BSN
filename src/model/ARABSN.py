import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..util.util import (pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, 
                        randomTransform, inverseTransform,pd_down,pd_up,psnr,generate_subimages,generate_mask_pair,Masker,adaptive_shuffle_by_std,random_shuffle_2x2)
from . import regist_model
from .MMBSN import MMBSN
from .DBSNl import DBSNl

@regist_model
class ARABSN(nn.Module):
    '''
    Adaptive Region-aware Blind-spot Network for Self-supervised Real Image Denoising(ARA-BSN)
    
    Key Features:
    1. Uses asymmetric pixel downsampling for feature extraction
    2. Different downsampling factors for training and inference
    3. Integrates blind-spot networks for denoising
    4. Supports Random Replacement Refinement (R3) for performance enhancement
    5. Supports random transformation of sub-images to enhance feature extraction
    '''
    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, scale1=1, scale2=3,
                    bsn='MMBSN', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        '''
        Parameters:
        pd_a: Pixel downsampling factor for training
        pd_b: Pixel downsampling factor for inference
        pd_pad: Padding size between sub-images
        R3: Whether to use Random Replacement Refinement
        R3_T: Number of masks for R3
        R3_p: Replacement probability for R3
        scale1: Fusion weight for the low-frequency region (per dataset: SIDD 1, DND 3)
        scale2: Fusion weight for the high-frequency region (per dataset: SIDD 3, DND 3)
        bsn: Type of blind-spot network
        in_ch: Number of input image channels
        bsn_base_ch: Base number of channels for BSN
        bsn_num_module: Number of BSN modules
        '''
        super().__init__()

        # Network hyperparameters
        self.pd_a    = pd_a
        self.pd_b    = pd_b
        self.pd_pad  = pd_pad
        self.R3      = R3
        self.R3_T    = R3_T
        self.R3_p    = R3_p
        self.scale1  = scale1
        self.scale2  = scale2
        self.masker = Masker(width=4, mode='interpolate', mask_type='all')
        
        # Define network architecture
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        elif bsn == 'MMBSN':
            self.bsn = MMBSN(in_ch, in_ch, bsn_base_ch)
        else:
            raise NotImplementedError('bsn %s is not implemented'%bsn)
    def one_forward(self, img, pd=None, update_params=True):
        # pd 缺省时使用训练重排因子 pd_a
        if pd is None:
            pd = self.pd_a

        if not update_params:
            with torch.no_grad():
                # 执行像素重排
                if pd > 1:
                    pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
                    
                    # 应用子图随机变换
                    transformed_img, transform_ops = randomTransform(pd_img, pd)
                    
                    # BSN网络前向传播
                    transformed_img_denoised = self.bsn(transformed_img)
                    
                    # 还原子图变换
                    pd_img_denoised = inverseTransform(transformed_img_denoised, transform_ops, pd)
                    
                    # 执行逆重排
                    img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
                else:
                    p = self.pd_pad
                    pd_img = F.pad(img, (p,p,p,p))
                    pd_img_denoised = self.bsn(pd_img)
                    img_pd_bsn = pd_img_denoised[:,:,p:-p,p:-p]
        else:
            # 执行像素重排
            if pd > 1:
                pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
                # 应用子图随机变换
                transformed_img, transform_ops = randomTransform(pd_img, pd)
                
                # # 去除padding=2的边缘
                # if pd == 5 and transformed_img.shape[2] % 8 != 0:
                #     p = 2
                #     transformed_img = transformed_img[:,:,p:-p,p:-p]
                
               
                # BSN网络前向传播
                transformed_img_denoised = self.bsn(transformed_img)
                
                # # 补充padding=2的边缘
                # if pd == 5 :
                #     p = 2
                #     transformed_img_denoised = F.pad(transformed_img_denoised, (p,p,p,p), mode='reflect')
                
                # 还原子图变换
                pd_img_denoised = inverseTransform(transformed_img_denoised, transform_ops, pd)
                
                # 执行逆重排
                img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
            else:
                p = self.pd_pad
                pd_img = F.pad(img, (p,p,p,p))
                pd_img_denoised = self.bsn(pd_img)
                img_pd_bsn = pd_img_denoised[:,:,p:-p,p:-p]

        return img_pd_bsn
    def two_forward(self, img, pd=5):
        """Three-stage forward pass with adaptive shuffling
        
        Args:
            img: Input image tensor [B,C,H,W]
            pd: Pixel shuffle factor for final stage
            update_params: Whether to update model parameters
            
        Returns:
            noisy_sub: Sub-images for final stage
            sub_denoised: Denoised sub-images
        """

        # Training mode - similar to above but without torch.no_grad()
        shuffled_2x2 = random_shuffle_2x2(img)
        downsampled = pd_down(shuffled_2x2, pd_factor=2)
        
        
        n, c, h, w = downsampled.shape
        denoised_sub = torch.zeros_like(downsampled)
        with torch.no_grad():
            for i in range(4):
                denoised_sub[i::4] = self.bsn(downsampled[i::4])
        
        # Final stage - using pd_down directly
        # Step 5: Adaptive 3x3 shuffle based on std
        shuffled_3x3= adaptive_shuffle_by_std(downsampled, std_threshold=1, denoised=denoised_sub)
        # Step 6: Pixel shuffle upsampling back to original size
        upsampled = pd_up(shuffled_3x3, pd_factor=2)
        
        # Step 7: Final stage - using pd_down directly
        p = pd

        pd_img = pixel_shuffle_down_sampling(upsampled, f=pd, pad=self.pd_pad)
        # Apply random transformation to sub-images
        transformed_img, transform_ops = randomTransform(pd_img, pd)
    
        # BSN network forward pass
        transformed_img_denoised = self.bsn(transformed_img)
        
        # Restore sub-image transformations
        pd_img_denoised = inverseTransform(transformed_img_denoised, transform_ops, pd)
        # pd_img = inverseTransform(transformed_img, transform_ops, pd)
        
        # Execute inverse pixel shuffle
        img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
    
        return img_pd_bsn
    def forward(self, img, pd=None, update_params=True):
        '''
        Forward propagation function including pixel downsampling (PD), sub-image transformation, BSN processing and inverse transformation
        
        Feature map size changes:
        1. Input: [B, C, H, W] 
        2. After pixel downsampling: [B, C×pd×pd, H/pd, W/pd]
        3. Sub-image transformation: maintains size
        4. BSN processing: maintains size
        5. Sub-image inverse transformation: maintains size
        6. After inverse downsampling: [B, C, H, W]
        
        Parameters:
        img: Input image
        pd: Pixel downsampling factor, default uses training factor pd_a
        update_params: Whether to update parameters
        '''
        # pd 缺省时使用训练重排因子 pd_a
        if pd is None:
            pd = self.pd_a

        if update_params:
            return self.one_forward(img, pd), self.two_forward(img, pd)
        return self.one_forward(img, pd, update_params)


    def denoise(self, x):

        return self.denoise_mpd(x, window_size=3, scale1=self.scale1, scale2=self.scale2)

        '''
        elif self.R3 == 'PD-refinement':
            s = 2
            denoised = torch.empty(*(x.shape), s**2, device=x.device)
            for i in range(s):
                for j in range(s):
                    tmp_input = torch.clone(x_mean).detach()
                    tmp_input[:,:,i::s,j::s] = x[:,:,i::s,j::s]
                    p = self.pd_pad
                    tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                    if self.pd_pad == 0:
                        denoised[..., i*s+j] = self.bsn(tmp_input)
                    else:
                        denoised[..., i*s+j] = self.bsn(tmp_input)[:,:,p:-p,p:-p]
            return_denoised = torch.mean(denoised, dim=-1)
        else:
            raise RuntimeError('post-processing type not supported')
        '''
    def denoise_mpd(self, x, low=1, high=5, window_size=3, scale1 = 1, scale2 = 3, lpd=4): # This scale1 parameter should be designed with reference to different datasets. 
        '''
        Multi-scale pixel downsampling denoising for inference stage
        
        Steps:
        1. Apply necessary padding to input
        2. Use inference downsampling factors for PD-BSN processing
        3. Decide whether to use R3 optimization based on settings
        
        Args:
            x: Input image
            low: Low frequency threshold
            high: High frequency threshold
            test: Whether to display visualization results
            window_size: Window size for local standard deviation calculation
            scale1: Scale factor for low frequency regions
            scale2: Scale factor for high frequency regions
            lpd: Low pixel downsampling factor
        '''
        b,c,h,w = x.shape

        # ============== PD = 1 ====================
        img_pd1_bsn = self.forward(x, pd=1,update_params=False)
        img_pd1_bsn = img_pd1_bsn[:, :, :h, :w]
        # ============== PD = 2 ====================
        img_pd2_bsn = x
        if h % self.pd_b != 0:
            img_pd2_bsn = F.pad(img_pd2_bsn, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            img_pd2_bsn = F.pad(img_pd2_bsn, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
        img_pd2_bsn = self.forward(img_pd2_bsn, pd=self.pd_b,update_params=False)
        img_pd2_bsn = img_pd2_bsn[:, :, :h, :w]  
        # # ============== PD = 4 ====================

        img_pd4_bsn = x
        if h % lpd != 0:
            img_pd4_bsn = F.pad(img_pd4_bsn, (0, 0, 0, lpd - h % lpd), mode='constant', value=0)
        if w % lpd != 0:
            img_pd4_bsn = F.pad(img_pd4_bsn, (0, lpd - w % lpd, 0, 0), mode='constant', value=0)
        img_pd4_bsn = self.forward(img_pd4_bsn, pd=lpd,update_params=False)
        img_pd4_bsn = img_pd4_bsn[:, :, :h, :w]
        

        # Calculate local standard deviation
        def std(img, window_size = 3):
            pad = window_size // 2
            img = torch.mean(img, dim=1, keepdim=True)  # [N, 1, H, W]
            N, C, H, W = img.shape
            img = F.pad(img, [pad] * 4, mode='reflect')
            img = F.unfold(img, kernel_size=window_size)
            img = img.view(N, C, window_size * window_size, H, W)
            img = img - torch.mean(img, dim=2, keepdim=True)
            img = img * img
            img = torch.mean(img, dim=2, keepdim=True)
            img = torch.sqrt(img)
            return img.squeeze(2)  # [N, C, H, W]
            
        # Calculate adaptive replacement probability using R3 optimization
        seita = std(img_pd2_bsn,window_size)  # [N, 1, H, W]
        
        # Create masks, divide into three regions based on seita values
        mask_low = seita <= low  # Low frequency region (seita <= 1)
        mask_mid = (seita > low) & (seita <= high)  # Mid frequency region (1 < seita <= 5)
        mask_high = seita > high  # High frequency region (seita > 5)
        
        # Expand masks to same number of channels as x
        mask_low = mask_low.expand_as(x)
        mask_mid = mask_mid.expand_as(x)
        mask_high = mask_high.expand_as(x)
        
        # Select different images based on regions
        img_pd_bsn = torch.zeros_like(x)
        img_pd_bsn = torch.where(mask_low, ((scale1-1)*img_pd2_bsn+img_pd4_bsn)/scale1, img_pd_bsn)  # Low freq region uses img_pd4_bsn
        img_pd_bsn = torch.where(mask_mid, img_pd2_bsn, img_pd_bsn)  # Mid freq region uses img_pd2_bsn
        img_pd_bsn = torch.where(mask_high, (img_pd1_bsn+(scale2-1)*img_pd2_bsn)/scale2, img_pd_bsn)  # High freq region uses img_pd1_bsn
        

        # Random Replacement Refinement (R3) processing
        if not self.R3:
            # Directly return result (without using R3)
            return img_pd_bsn
        else:
            # Use R3 for optimization
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            denoised.fill_(0)
            for t in range(self.R3_T):
                # Generate random mask (using original size)
                indice = torch.rand_like(x)
                mask_r3 = indice < self.R3_p

                # Random replacement and processing
                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask_r3] = x[mask_r3]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                
                # BSN processing
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:,:,p:-p,p:-p]

            # Return average result of multiple processing
            # If in test mode, display visualization results
            return torch.mean(denoised, dim=-1)
            
        '''
        elif self.R3 == 'PD-refinement':
            s = 2
            denoised = torch.empty(*(x.shape), s**2, device=x.device)
            for i in range(s):
                for j in range(s):
                    tmp_input = torch.clone(x_mean).detach()
                    tmp_input[:,:,i::s,j::s] = x[:,:,i::s,j::s]
                    p = self.pd_pad
                    tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                    if self.pd_pad == 0:
                        denoised[..., i*s+j] = self.bsn(tmp_input)
                    else:
                        denoised[..., i*s+j] = self.bsn(tmp_input)[:,:,p:-p,p:-p]
            return_denoised = torch.mean(denoised, dim=-1)
        else:
            raise RuntimeError('post-processing type not supported')
        '''


