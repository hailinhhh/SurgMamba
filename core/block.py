import math
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from CNN import ConvMBlock
from HVMamba import VSSBlock
from LGFCU import BiCrossCNN_Mamba, BiCrossMamba_CNN 
from timm.models.layers import DropPath, trunc_normal_
from math import log
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class ConvMambaBlock(nn.Module):
    def __init__(
                self, 
                inplanes, 
                outplanes,
                res_conv,  
                stride, 
                inchannel,
                outchannel,
                embed_dim, 
                mlp_ratio=4.,
                dims=128,
                ssm_d_state=16,
                ssm_ratio=2.0,
                ssm_dt_rank="auto",
                ssm_act_layer="silu",        
                ssm_conv=3,
                ssm_conv_bias=True,
                ssm_drop_rate=0.0, 
                drop_path_rate=0.1,
                mlp_act_layer="gelu",
                mlp_drop_rate=0.0,
                gmlp=False,
                drop_path=0,
                last_fusion=False, 
                num_med_block=0, 
                groups=1,
                dc_inner=8):
        
        super(ConvMambaBlock, self).__init__()
        self.stride = stride
        self.embed_dim = embed_dim
        self.inchannel = int(inchannel)
        self.outchannel = int(outchannel)
        self.num_med_block = num_med_block
        self.cnn_block = ConvMBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups,expansion=2)
       
        if last_fusion:
            self.fusion_block = ConvMBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups,expansion=1)
        else:
            self.fusion_block = ConvMBlock(inplanes=outplanes, outplanes=outplanes, groups=groups,expansion=1)
       
        self.squeeze_block = BiCrossCNN_Mamba(inchannel=self.inchannel, outchannel=self.outchannel)
        self.expand_block = BiCrossMamba_CNN(inplanes=embed_dim, outplanes=outplanes)

        self.mamba_block = VSSBlock(
                    hidden_dim=dims, 
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                    channel_first=False,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=False,
                    dc_inner=dc_inner,
                    )
    
    def forward(self, x, x_m):
        x, x2 = self.cnn_block(x)
        _, _, H, W = x2.shape

        x_sm = self.squeeze_block(x2, x_m) 
        x_m = self.mamba_block(x_sm)

        x_m_r = self.expand_block(x, x_m) 
        x = self.fusion_block(x, x_m_r, return_x_2=False)

        return x, x_m
