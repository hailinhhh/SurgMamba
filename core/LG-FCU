import torch
import torch.nn as nn
import torch.nn.functional as F

class BiCrossCNN_Mamba(nn.Module):
    def __init__(self, inchannel, outchannel,num_heads=1,qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(BiCrossCNN_Mamba, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.num_heads = num_heads
        
        self.head_dim = self.outchannel // self.num_heads
        
        self.scale = self.outchannel ** -0.5

        self.key = nn.Linear(self.outchannel,self.outchannel)
        
        self.query = nn.Linear(self.inchannel, self.outchannel)
        self.value = nn.Linear(self.inchannel, self.outchannel)
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.out = nn.Linear(self.outchannel, self.outchannel)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x_c, x_m):  

        resudial_x_m = x_m
        b_cnn,d_cnn,h_cnn,w_cnn = x_c.shape
        b_mamba,h_mamba,w_mamba,d_mamba, = x_m.shape
        
        # Step 1: Transform features to sequence format
        x_c = x_c.permute(0,2,3,1).reshape(-1,h_cnn*w_cnn,d_cnn) 
        x_m = x_m.reshape(-1,h_mamba*w_mamba,d_mamba) 
      
        # Step 2: Generate Query, Key, Value
        Q_local = self.query(x_c).reshape(b_cnn,h_cnn*w_cnn,self.num_heads, d_mamba // self.num_heads).permute(0,2,1,3)  
        K_global = self.key(x_m).reshape(b_mamba,h_mamba*w_mamba,self.num_heads, d_mamba // self.num_heads).permute(0,2,1,3) 
        V_local = self.value(x_c).reshape(b_cnn,h_cnn*w_cnn,self.num_heads, d_mamba // self.num_heads).permute(0,2,1,3) 

        # Step 3: Apply scaled dot-product attention
        attention_scores = torch.matmul(Q_local, K_global.transpose(-2, -1)) / (self.outchannel ** 0.5)  
        attention_weights = F.softmax(attention_scores, dim=-1) 
        attention_weights = self.attn_drop(attention_weights)

        # Step 4: Attention output
        attention_output = torch.matmul(attention_weights, V_local).transpose(1, 2).reshape(b_mamba,h_mamba*w_mamba, d_mamba) 
        
        # Step 5: Reshape back to spatial format and apply final projection
        fused_feat = attention_output.reshape(b_mamba, h_mamba, w_mamba, d_mamba)
        x_m = fused_feat + resudial_x_m

        return x_m
    
class BiCrossMamba_CNN(nn.Module):
    def __init__(self, inplanes, outplanes,num_heads=1,qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(BiCrossMamba_CNN, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.num_heads = num_heads
        
        self.head_dim = self.outplanes // self.num_heads

        self.scale = self.outplanes ** -0.5

        self.key = nn.Linear(self.outplanes, self.outplanes)
        self.query = nn.Linear(self.inplanes, self.outplanes)
        self.value = nn.Linear(self.inplanes, self.outplanes)     
        self.attn_drop = nn.Dropout(attn_drop)

        self.out = nn.Linear(self.outplanes, self.outplanes)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x_c, x_m):

        b_cnn,d_cnn,h_cnn,w_cnn = x_c.shape
        b_mamba,h_mamba,w_mamba,d_mamba, = x_m.shape

        x_c = x_c.permute(0,2,3,1).reshape(-1,h_cnn*w_cnn,d_cnn) 

        x_m = x_m.reshape(-1,h_mamba*w_mamba,d_mamba) 

        Q_global = self.query(x_m).reshape(b_mamba,h_mamba*w_mamba,self.num_heads, d_cnn // self.num_heads).permute(0,2,1,3)   
        K_local = self.key(x_c).reshape(b_cnn,h_cnn*w_cnn,self.num_heads, d_cnn // self.num_heads).permute(0,2,1,3)  
        
        V_global = self.value(x_m).reshape(b_mamba,h_mamba*w_mamba,self.num_heads, d_cnn // self.num_heads).permute(0,2,1,3)  


        attention_scores = torch.matmul(Q_global, K_local.transpose(-2, -1)) / (self.outplanes ** 0.5)  
        attention_weights = F.softmax(attention_scores, dim=-1)  
        attention_weights = self.attn_drop(attention_weights)

        attention_output = torch.matmul(attention_weights, V_global).reshape(b_cnn,d_cnn,h_cnn*w_cnn) 

        fused_feat = attention_output.reshape(b_cnn, d_cnn,h_cnn, w_cnn)

        return fused_feat

