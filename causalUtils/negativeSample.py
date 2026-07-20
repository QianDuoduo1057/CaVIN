import torch

def generate_intervention_negatives(model, V_i, V_i_next, alpha=0.1, n_steps=3):
    """
    干预式负样本生成器
    
    参数:
        CaVIN: CaVIN模型（公式8-10的实现）
        V_i: 原因token, shape [batch_size, dim]
        V_i_next: 结果token, shape [batch_size, dim]
        alpha: 每步扰动步长
        n_steps: 迭代次数（PGD风格的多步攻击）
    
    返回:
        V_tilde: 干预后的token，因果关系被破坏但语义相近
    """
    
    # ================== Step 1: 准备阶段 ==================
    model.eval()  # 切换到评估模式，冻结Dropout/BatchNorm
    
    # 冻结VNCRN的所有参数（我们不更新模型，只更新输入）
    for param in model.parameters():
        param.requires_grad = False
    
    # 复制V_i作为可优化变量（关键：需要追踪梯度）
    V_tilde = V_i.clone().detach().requires_grad_(True)
    
    # ================== Step 2: 迭代扰动 ==================
    for step in range(n_steps):
        # -------- 2.1 前向传播：计算当前因果分数 --------
        causal_score = model(V_tilde, V_i_next)  
        # 输出形状: [batch_size, 1]，值域约为[-1, +1]
        # +1表示强因果，0表示无关，-1表示反向因果
        
        # -------- 2.2 反向传播：计算V_tilde的梯度 --------
        causal_score.sum().backward()
        # 为什么sum()? 因为batch中每个样本都要算梯度，sum聚合后再backward
        # 此时 V_tilde.grad 就是 ∇_{V_tilde} P_hat(V_tilde -> V_i_next)
        
        # -------- 2.3 梯度下降式扰动 --------
        with torch.no_grad():  # 这里不需要再构建计算图
            grad = V_tilde.grad  # 形状: [batch_size, dim]
            
            # 归一化梯度（核心公式）
            grad_norm = grad.norm(dim=-1, keepdim=True) + 1e-8  # 防除零
            perturbation = alpha * grad / grad_norm
            # perturbation的L2范数 = alpha（每步固定扰动强度）
            
            # 沿负梯度方向更新（降低因果分数）
            V_tilde = V_tilde - perturbation
            
            # -------- 2.4 投影约束（可选但推荐）--------
            # 防止扰动累积后偏离原token太远
            delta = V_tilde - V_i  # 累积扰动量
            max_norm = alpha * n_steps * 1.5  # 允许的最大总偏移
            
            # 裁剪到L∞球内（也可以用L2球投影）
            delta = torch.clamp(delta, -max_norm, max_norm)
            V_tilde = V_i + delta
        
        # -------- 2.5 准备下一轮迭代 --------
        V_tilde = V_tilde.detach().requires_grad_(True)
        # detach切断之前的计算图，requires_grad重新开启梯度追踪
    
    # ================== Step 3: 恢复模型状态 ==================
    model.train()  # 恢复训练模式
    for param in model.parameters():
        param.requires_grad = True
    
    return V_tilde.detach()  # 返回时detach，不需要梯度了