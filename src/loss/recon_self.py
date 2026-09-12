import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss


eps = 1e-6

# ============================ #
#  Self-reconstruction loss    #
# ============================ #
@regist_loss
class self_Pd_L1():
    def __call__(self, input_data, model_output, data, module):
        output = output = model_output['recon'][0]
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.l1_loss(output, target_noisy)
@regist_loss
class self_Adapt_L1():
    def __call__(self, input_data, model_output, data, module):
        with torch.no_grad():
            output = model_output['recon'][1] 
    

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        return F.l1_loss(output,target_noisy)
    
