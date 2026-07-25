import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFusion(nn.Module):
    def __init__(self,in_planes):
        # Split-Fuse-Select
        self.init__ = super(SegFusion, self).__init__()
        self.gate_3x3 = nn.Conv2d(in_planes*2, in_planes, kernel_size=3, padding=1)
        self.gate_5x5 = nn.Conv2d(in_planes*2, in_planes, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//2, 2, kernel_size=1),  
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, feature1, feature2):
        feature_cat = torch.cat([feature1,feature2], dim=1)
        gate_w3 = self.gate_3x3(feature_cat)
        gate_w5 = self.gate_5x5(feature_cat)
        fused = gate_w3+gate_w5
        pooled = self.gap(fused)
        attention_weights = self.fc(pooled)
        attention_weights = self.softmax(attention_weights)
        
        attention_1 = attention_weights[:, 0:1, :, :]  
        attention_2 = attention_weights[:, 1:2, :, :]
        attention_1 = attention_1.expand_as(feature1)  
        attention_2 = attention_2.expand_as(feature2)
        
        output = attention_1 * feature1 + attention_2 * feature2
        
        return output

