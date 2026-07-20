import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import torch.nn.functional as F

class CausalInferenceResNetDeep(nn.Module):
    def __init__(self, input_dim=512, width_multiplier=1.5):
        super(CausalInferenceResNetDeep, self).__init__()

        hidden_dim1 = int(input_dim * width_multiplier)  # e.g. 768
        hidden_dim2 = int(hidden_dim1 * 0.6)             # e.g. 460
        hidden_dim3 = int(hidden_dim1 * 0.4)             # e.g. 307
        hidden_dim4 = int(hidden_dim1 * 0.3)             # e.g. 230

        self.projection_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
        )

        self.projection_y = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
        )

        self.F1 = nn.Sequential(
            nn.Linear(hidden_dim2 // 2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.BatchNorm1d(hidden_dim4),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim4, hidden_dim4 // 2),
            nn.BatchNorm1d(hidden_dim4 // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.attention_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        f2_input_dim = 4 * (hidden_dim4 // 2)
        f2_hidden1_dim = f2_input_dim // 2
        f2_hidden2_dim = f2_hidden1_dim // 2

        self.F2 = nn.Sequential(
            nn.Linear(f2_input_dim, f2_hidden1_dim),
            nn.BatchNorm1d(f2_hidden1_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(f2_hidden1_dim, f2_hidden2_dim),
            nn.BatchNorm1d(f2_hidden2_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(f2_hidden2_dim, 1)
        )

        self.residual_scale = nn.Parameter(torch.ones(1))
        self.tanh = nn.Tanh()

        self.input_dim = input_dim
        self.width_multiplier = width_multiplier

    def forward(self, x, y):
        batch_size = x.shape[0]
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        if y.dim() > 2:
            y = y.view(batch_size, -1)

        x_proj = self.projection_x(x)
        y_proj = self.projection_y(y)

        F1_x = self.F1(x_proj)
        F1_y = self.F1(y_proj)

        diff = F1_x - F1_y
        prod = F1_x * F1_y

        distance = torch.norm(diff, p=2, dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(F1_x, F1_y, dim=1, eps=1e-8).unsqueeze(1)

        gate_input = torch.cat([distance, cos_sim], dim=1)
        gate_weight = self.attention_gate(gate_input) * 2.0
        d_expanded = gate_weight.expand_as(F1_x)

        F1_x_modulated = F1_x * d_expanded
        F1_y_modulated = F1_y * d_expanded
        diff_modulated = diff * d_expanded
        prod_modulated = prod * d_expanded

        combined = torch.cat([
            F1_x_modulated,
            F1_y_modulated,
            diff_modulated,
            prod_modulated
        ], dim=1)

        identity = combined.mean(dim=1, keepdim=True) * self.residual_scale
        output = self.F2(combined) + identity
        output = self.tanh(output)

        return output.squeeze()

    def get_hidden(self, x, y):
        batch_size = x.shape[0]
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        if y.dim() > 2:
            y = y.view(batch_size, -1)

        x_proj = self.projection_x(x)
        y_proj = self.projection_y(y)

        F1_x = self.F1(x_proj)
        F1_y = self.F1(y_proj)

        diff = F1_x - F1_y
        prod = F1_x * F1_y
        
        distance = torch.norm(diff, p=2, dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(F1_x, F1_y, dim=1, eps=1e-8).unsqueeze(1)

        gate_input = torch.cat([distance, cos_sim], dim=1)
        gate_weight = self.attention_gate(gate_input) * 2.0
        d_expanded = gate_weight.expand_as(F1_x)

        F1_x_modulated = F1_x * d_expanded
        F1_y_modulated = F1_y * d_expanded
        diff_modulated = diff * d_expanded
        prod_modulated = prod * d_expanded

        combined = torch.cat([
            F1_x_modulated,
            F1_y_modulated,
            diff_modulated,
            prod_modulated
        ], dim=1)

        return combined
        
    def get_hidden(self, x, y):
        batch_size = x.shape[0]
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        if y.dim() > 2:
            y = y.view(batch_size, -1)

        x_proj = self.projection_x(x)
        y_proj = self.projection_y(y)

        F1_x = self.F1(x_proj)
        F1_y = self.F1(y_proj)

        diff = F1_x - F1_y
        prod = F1_x * F1_y
        
        distance = torch.norm(diff, p=2, dim=1, keepdim=True)
        cos_sim = F.cosine_similarity(F1_x, F1_y, dim=1, eps=1e-8).unsqueeze(1)

        gate_input = torch.cat([distance, cos_sim], dim=1)
        gate_weight = self.attention_gate(gate_input) * 2.0
        d_expanded = gate_weight.expand_as(F1_x)

        F1_x_modulated = F1_x * d_expanded
        F1_y_modulated = F1_y * d_expanded
        diff_modulated = diff * d_expanded
        prod_modulated = prod * d_expanded

        combined = torch.cat([
            F1_x_modulated,
            F1_y_modulated,
            diff_modulated,
            prod_modulated
        ], dim=1)

        return combined


class CausalInferenceResNetDeepAblation(nn.Module):
    """
    支持特征消融的VNCRN模型
    
    可消融的5个特征（对应论文公式8）:
    - 'F1_x': F₁(x) 原始visual token特征
    - 'F1_y': F₁(y) 原始visual token特征  
    - 'diff': F₁(x) - F₁(y) 差分特征
    - 'prod': F₁(x) ⊙ F₁(y) 元素积特征
    - 'distance': dis(F₁(x), F₁(y)) 距离度量
    - 'cos_sim': cos(F₁(x), F₁(y)) 余弦相似度
    """
    
    def __init__(self, input_dim=512, width_multiplier=1.5, 
                 ablate_features=None):
        """
        Args:
            ablate_features: 要移除的特征列表，如 ['diff', 'cos_sim']
        """
        super(CausalInferenceResNetDeepAblation, self).__init__()
        
        self.ablate_features = ablate_features or []
        
        # 计算各层维度
        hidden_dim1 = int(input_dim * width_multiplier)
        hidden_dim2 = int(hidden_dim1 * 0.6)
        hidden_dim3 = int(hidden_dim1 * 0.4)
        hidden_dim4 = int(hidden_dim1 * 0.3)
        self.feature_dim = hidden_dim4 // 2
        
        # 投影层
        self.projection_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, hidden_dim2//2),
        )
        
        self.projection_y = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, hidden_dim2//2),
        )
        
        # F1网络
        self.F1 = nn.Sequential(
            nn.Linear(hidden_dim2//2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim3, hidden_dim4),
            nn.BatchNorm1d(hidden_dim4),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim4, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 门控attention
        self.attention_gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 动态计算F2输入维度
        f2_input_dim = self._calculate_f2_input_dim()
        f2_hidden1_dim = max(f2_input_dim // 2, 32)
        f2_hidden2_dim = max(f2_hidden1_dim // 2, 16)
        
        self.F2 = nn.Sequential(
            nn.Linear(f2_input_dim, f2_hidden1_dim),
            nn.BatchNorm1d(f2_hidden1_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(f2_hidden1_dim, f2_hidden2_dim),
            nn.BatchNorm1d(f2_hidden2_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(f2_hidden2_dim, 1)
        )
        
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.tanh = nn.Tanh()
    
    def _calculate_f2_input_dim(self):
        """根据消融配置计算F2输入维度"""
        dim = 0
        if 'F1_x' not in self.ablate_features:
            dim += self.feature_dim
        if 'F1_y' not in self.ablate_features:
            dim += self.feature_dim
        if 'diff' not in self.ablate_features:
            dim += self.feature_dim
        if 'prod' not in self.ablate_features:
            dim += self.feature_dim
        return max(dim, self.feature_dim)  # 至少保留一个特征维度
    
    def forward(self, x, y):
        batch_size = x.shape[0]
        
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        if y.dim() > 2:
            y = y.view(batch_size, -1)
        
        x_proj = self.projection_x(x)
        y_proj = self.projection_y(y)
        
        F1_x = self.F1(x_proj)
        F1_y = self.F1(y_proj)
        
        # 计算所有特征
        diff = F1_x - F1_y
        prod = F1_x * F1_y
        
        # 计算门控（根据消融配置）
        if 'distance' not in self.ablate_features:
            distance = torch.norm(diff, p=2, dim=1, keepdim=True)
        else:
            distance = torch.ones(batch_size, 1, device=x.device)
            
        if 'cos_sim' not in self.ablate_features:
            cos_sim = F.cosine_similarity(F1_x, F1_y, dim=1, eps=1e-8).unsqueeze(1)
        else:
            cos_sim = torch.ones(batch_size, 1, device=x.device)
        
        gate_input = torch.cat([distance, cos_sim], dim=1)
        gate_weight = self.attention_gate(gate_input) * 2.0
        d_expanded = gate_weight.expand_as(F1_x)
        
        # 根据消融配置构建特征向量
        features = []
        if 'F1_x' not in self.ablate_features:
            features.append(F1_x * d_expanded)
        if 'F1_y' not in self.ablate_features:
            features.append(F1_y * d_expanded)
        if 'diff' not in self.ablate_features:
            features.append(diff * d_expanded)
        if 'prod' not in self.ablate_features:
            features.append(prod * d_expanded)
        
        # 确保至少有一个特征
        if not features:
            features.append(F1_x * d_expanded)
        
        combined = torch.cat(features, dim=1)
        
        identity = combined.mean(dim=1, keepdim=True) * self.residual_scale
        output = self.F2(combined) + identity
        output = self.tanh(output)
        
        return output.squeeze()
    
    def get_ablation_config(self):
        """返回当前消融配置"""
        return {
            'ablated': self.ablate_features,
            'active': [f for f in ['F1_x', 'F1_y', 'diff', 'prod', 'distance', 'cos_sim'] 
                      if f not in self.ablate_features]
        }


