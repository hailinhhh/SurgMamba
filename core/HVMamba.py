import math
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from math import log
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2) 
        return x
      
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):   
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
      
# https://arxiv.org/abs/2105.08050  gMLP
class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

# https://github.com/MzeroMiko/VMamba
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,**factory_kwargs):   
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True,**factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant": 
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random": 
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
          
        dt = torch.exp(
            torch.rand(d_inner,**factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj
    
    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous() 
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous() 
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True 
        return A_log
    
    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous() 
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=2):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) 
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) 
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) 
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

def get_indices(H, W, direction,device='cpu'):
    indices = []
    if direction == 'left_to_right':
        for d in range(H + W - 1):
            indices.extend([(i, d - i) for i in range(max(0, d - W + 1), min(d + 1, H))])
    elif direction == 'right_to_left':
        for d in range(H + W - 1):
            indices.extend([(H - 1 - i, W - 1 - (d - i)) for i in range(max(0, d - W + 1), min(d + 1, H))])
    elif direction == 'top_right_to_bottom_left':
        for d in range(H + W - 1):
            indices.extend([(i, W - 1 - (d - i)) for i in range(max(0, d - W + 1), min(d + 1, H))])
    elif direction == 'bottom_left_to_top_right':
        for d in range(H + W - 1):
            indices.extend([(H - 1 - i, d - i) for i in range(max(0, d - W + 1), min(d + 1, H))])
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return torch.tensor([(i * W + j) for i, j in indices], device=device).long()

def restore_from_scan(flatten_scan, H, W, indices):
    device = flatten_scan.device
    B, C, L = flatten_scan.shape
    assert L == H * W, "Flattened scan length must match H * W"
    restored = torch.zeros(B, C, H * W, device=flatten_scan.device)
    flatten_scan = flatten_scan.to(torch.float32)
    indices = indices.to(device)  
    for b in range(B):
        restored[b].scatter_(dim=1, index=indices.unsqueeze(0).expand(C, -1), src=flatten_scan[b])
    return restored.view(B, C, H, W)

# https://github.com/houqb/CoordAttention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n,c,h,w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        d_conv=3,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        
        channel_first=False,
        dc_inner=8,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype} 
        super().__init__()

        self.k_group = 2
        self.kc_group = 2
        self.dc_inner = dc_inner
       
        self.d_model = int(d_model) 
        self.d_state = int(d_state) 
        self.d_inner = int(ssm_ratio * d_model)
        
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1

        self.forward_core = self.forward_corev0
        self.forward_corec = self.forward_corevc
       
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer() 

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner, 
            groups=self.d_inner, 
            bias=conv_bias, 
            kernel_size=d_conv, 
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False,**factory_kwargs) 
            for _ in range(self.k_group) 
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) 
        del self.x_proj
        
        self.xc_proj = [
                    nn.Linear(dc_inner, (self.dt_rank + self.d_state * 2), bias=False,**factory_kwargs) 
                    for _ in range(self.kc_group) 
                ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0)) 
        del self.xc_proj

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=2,
        )
        self.Ac_logs, self.Dsc, self.dtc_projs_weight, self.dtc_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, dc_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=2,
        )
        
        self.SpatialAttention1 = CoordAtt(inp=self.d_inner,oup=self.d_inner,reduction=32)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, 1, bias=bias, **factory_kwargs)

        self.dropout = nn.Dropout(dropout) 
        self.channel_branch = nn.Sequential(
            nn.Conv2d(self.d_inner,self.d_inner,3,2,1),
            nn.Conv2d(self.d_inner,self.d_inner,3,2,1)
        )

        self.fuse_gate = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner//8, 1),
            nn.ReLU(),
            nn.Conv2d(self.d_inner//8, 1, 1),
            nn.Sigmoid()
        )
       
    def forward_corev0(self, x: torch.Tensor,force_fp32):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 2
        
        left_to_right_indices = get_indices(H, W, 'left_to_right',device=x.device)
        right_to_left_indices = get_indices(H, W, 'right_to_left',device=x.device)
        top_right_to_bottom_left_indices = get_indices(H, W, 'top_right_to_bottom_left',device=x.device)
        bottom_left_to_top_right_indices = get_indices(H, W, 'bottom_left_to_top_right',device=x.device)
        flattened = x.view(B, C, -1)
        scan1 = flattened[:, :, left_to_right_indices].view(B, 1, -1, L)
        scan2 = flattened[:, :, right_to_left_indices].view(B, 1, -1, L)
        scan3 = flattened[:, :, top_right_to_bottom_left_indices].view(B, 1, -1, L)
        scan4 = flattened[:, :, bottom_left_to_top_right_indices].view(B, 1, -1, L)
        
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) 
       
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1]), scan1, scan2, scan3, scan4], dim=1)  
        xs_1 = self.SpatialAttention1(xs[:,0,:,:].view(B,C,H,W))
        xs_2 = self.SpatialAttention1(xs[:,1,:,:].view(B,C,H,W))
        xs_3 = self.SpatialAttention1(xs[:,2,:,:].view(B,C,H,W))
        xs_4 = self.SpatialAttention1(xs[:,3,:,:].view(B,C,H,W))
        xs_5 = self.SpatialAttention1(xs[:,4,:,:].view(B,C,H,W))
        xs_6 = self.SpatialAttention1(xs[:,5,:,:].view(B,C,H,W))
        xs_7 = self.SpatialAttention1(xs[:,6,:,:].view(B,C,H,W))
        xs_8 = self.SpatialAttention1(xs[:,7,:,:].view(B,C,H,W))
        xs_0 = torch.cat([xs_1,xs_2,xs_3,xs_4,xs_5,xs_6,xs_7,xs_8], dim=1)
        xs_ = xs_0.view(B,8,C,L)
       
        inv_y = torch.flip(xs_[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(xs_[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        diag_y_1 = xs_[:, 4:5].view(B, 1, -1, L).squeeze(1)
        diag_y_2 = xs_[:, 5:6].view(B, 1, -1, L).squeeze(1)
        diag_y_3 = xs_[:, 6:7].view(B, 1, -1, L).squeeze(1)
        diag_y_4 = xs_[:, 7:8].view(B, 1, -1, L).squeeze(1)       

        diag_y_1 = restore_from_scan(diag_y_1, H, W, left_to_right_indices).contiguous().view(B, -1, L)
        diag_y_2 = restore_from_scan(diag_y_2, H, W, right_to_left_indices).contiguous().view(B, -1, L) 
        diag_y_3 = restore_from_scan(diag_y_3, H, W, top_right_to_bottom_left_indices).contiguous().view(B, -1, L)
        diag_y_4 = restore_from_scan(diag_y_4, H, W, bottom_left_to_top_right_indices).contiguous().view(B, -1, L)
        
        xs_ = xs[:, 0] + xs_[:, 0]+inv_y[:, 0]+wh_y+invwh_y+diag_y_1+diag_y_2+diag_y_3+diag_y_4 
        
        xs_weight = torch.cat([xs_.view(B, 1, -1, L),torch.flip(xs_.view(B, 1, -1, L), dims=[-1])], dim=1)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_weight.view(B, K, -1, L), self.x_proj_weight)
       
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        xs_weight = xs_weight.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) 
        Bs = Bs.float().view(B, K, -1, L) 
        Cs = Cs.float().view(B, K, -1, L) 
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) 
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        if force_fp32:
            xs_weight, dts, Bs, Cs = to_fp32(xs_weight, dts, Bs, Cs)
            
        out_y = self.selective_scan(
            xs_weight, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 1], dims=[-1]).contiguous().view(B, -1, L)
        return out_y[:, 0].contiguous().view(B, -1, L),inv_y
    
    def forward_corevc(self, x: torch.Tensor,force_fp32):
        _, _, C = x.shape
        K = 2
        y_c = x
        self.selective_scan_c = selective_scan_fn
        B, D, L = y_c.shape 

        xc_ = torch.cat([y_c.view(B, D, L), torch.flip(y_c, dims=[-1])], dim=1).view(B, 2, D, L) 
        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xc_.view(B, K, -1, L), self.xc_proj_weight)

        if hasattr(self, "x_proj_bias"):
            xc_dbl = xc_dbl + self.xc_proj_bias.view(1, K, -1, 1)
        dtsc, Bsc, Csc = torch.split(xc_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dtsc = torch.einsum("b k r l, k d r -> b k d l", dtsc.view(B, K, -1, L).to(y_c.device), self.dtc_projs_weight.to(y_c.device))        

        xc_weight = xc_ .float().view(B, -1, L)
        dtsc = dtsc.contiguous().float().view(B, -1, L).to(y_c.device) 
        Bsc = Bsc.float().view(B, K, -1, L).to(y_c.device) 
        Csc = Csc.float().view(B, K, -1, L).to(y_c.device) 
        Dsc = self.Dsc.float().view(-1).to(y_c.device) 
        Asc = -torch.exp(self.Ac_logs.float()).view(-1, self.d_state).to(y_c.device)  
        dtc_projs_bias = self.dtc_projs_bias.float().view(-1).to(y_c.device) 

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        if force_fp32:
            xc_weight, dtsc, Bsc, Csc = to_fp32(xc_weight, dtsc, Bsc, Csc)
            
        out_yc = self.selective_scan_c(
            xc_weight, dtsc, 
            Asc, Bsc, Csc, Dsc, z=None,
            delta_bias=dtc_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_yc.dtype == torch.float

        inv_yc = torch.flip(out_yc[:, 1], dims=[-1]).contiguous().view(B, -1, L)
        return out_yc[:, 0].contiguous().view(B, -1, L),inv_yc

    def forward(self, x: torch.Tensor, force_fp32=True, relative_pos=None, **kwargs):
        x = self.in_proj(x) 
        x, z = x.chunk(2, dim=-1)
        residual_c = x 
        z = self.act(z)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        B, D, H, W = x.shape
        y1, y2 = self.forward_core(x,force_fp32)
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = y.transpose(dim0=1, dim1=2).contiguous() 
        y = self.out_norm(y).view(B, H, W, -1)
        
        y = y * z
        
        b_c,h_c,w_c,d_c = residual_c.shape

        x_c = self.channel_branch(residual_c.permute(0,3,1,2))
        _,x_c_d,x_c_h,x_c_w = x_c.shape
        x_c = x_c.permute(0,2,3,1).view(b_c,-1,d_c)

        yc1, yc2 = self.forward_corec(x_c,force_fp32)
        yc = yc1 + yc2 
        yc = yc.permute(0,2,1).view(b_c,x_c_d,x_c_h,x_c_w)
        yc = F.interpolate(yc,size=(h_c,w_c),mode='bilinear',align_corners=False).permute(0,2,3,1)
        yc = yc * z 

        y_yc = y.permute(0,3,1,2) + yc.permute(0,3,1,2)

        out = self.dropout(self.out_proj(y_yc).permute(0,2,3,1))# BHWD
        return out

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        dc_inner=8,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,   
                channel_first=channel_first,
                dc_inner=dc_inner
            )
        self.drop_path = DropPath(drop_path)     
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))          
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) 
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
