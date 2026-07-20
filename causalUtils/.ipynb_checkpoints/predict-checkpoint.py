"""
CaVIN 因果推理检测模块
用于外部调用，对未知的 visual token 对进行因果方向预测。

输出含义（对应论文公式）：
  output ∈ [ 0.333,  1.0]  →  x → y (x 是 y 的因)
  output ∈ [-1.0,  -0.333] →  y → x (y 是 x 的因)
  output ∈ (-0.333, 0.333) →  x ↛ y (无因果关系)
"""

import torch
import torch.nn as nn
import numpy as np
from .causalNet import CausalInferenceResNetDeep

# ══════════════════════════════════════════════════════════════
#  模型加载（单例缓存）
# ══════════════════════════════════════════════════════════════

_cached_model = None
_cached_model_path = None

def load_model(model_path, model_width=2, input_dim=512, device='cuda'):
    global _cached_model, _cached_model_path

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model, device

    checkpoint = torch.load(model_path, map_location=device)
    model = CausalInferenceResNetDeep(input_dim=input_dim, width_multiplier=model_width)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    _cached_model = model
    _cached_model_path = model_path

    return model, device

# ══════════════════════════════════════════════════════════════
#  自动填充 / 截断工具
# ══════════════════════════════════════════════════════════════

def _pad_or_truncate(tensor, target_length, pad_value=0, dim=-1):
    """
    将 tensor 在指定维度上截断或填充到 target_length。

    Args:
        tensor:        输入张量，任意维度
        target_length: 目标长度
        pad_value:     填充值
        dim:           操作维度（支持负索引）

    Returns:
        处理后的张量，指定维度长度 == target_length
    """
    # 统一为正索引
    ndim = tensor.dim()
    dim = dim % ndim

    cur_length = tensor.shape[dim]

    if cur_length == target_length:
        return tensor

    if cur_length > target_length:
        # 截断
        slice_obj = [slice(None)] * ndim
        slice_obj[dim] = slice(0, target_length)
        return tensor[tuple(slice_obj)]

    # 填充：F.pad 的 pad 参数从最后一个维度开始，每个维度 (before, after)
    # 对于第 dim 维，它在 pad 列表中的位置是倒数第 (ndim - 1 - dim) 对
    pad = [0] * (2 * ndim)
    pad_pair_idx = ndim - 1 - dim          # 第几对（从0开始）
    pad[2 * pad_pair_idx + 1] = target_length - cur_length  # after
    return torch.nn.functional.pad(tensor, pad, mode='constant', value=pad_value)

# ══════════════════════════════════════════════════════════════
#  张量转换（集成自动填充）
# ══════════════════════════════════════════════════════════════

def _to_tensor(data, device, target_length=None, pad_value=0, dim=-1):
    """
    将 numpy / list / tensor 统一转为 (B, D) 的 float tensor。
    若指定 target_length，自动在 dim 维截断或填充。

    Args:
        data:          输入数据
        device:        目标设备
        target_length: 目标长度，None 表示不做处理
        pad_value:     填充值
        dim:           操作维度
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif isinstance(data, list):
        data = torch.tensor(data, dtype=torch.float32)
    elif isinstance(data, torch.Tensor):
        data = data.float()
    else:
        raise TypeError(f"不支持的输入类型: {type(data)}")

    if data.dim() == 1:
        data = data.unsqueeze(0)  # (D,) → (1, D)

    # 自动填充 / 截断
    if target_length is not None:
        data = _pad_or_truncate(data, target_length, pad_value=pad_value, dim=dim)

    return data.to(device)

# ══════════════════════════════════════════════════════════════
#  标签映射
# ══════════════════════════════════════════════════════════════

def _output_to_label(output_val, threshold=0.333):
    if output_val >= threshold:
        return 1    # x → y
    elif output_val <= -threshold:
        return -1   # y → x
    else:
        return 0    # x ↛ y

LABEL_DESC = {
    1:  "x → y (x causes y)",
    -1: "y → x (y causes x)",
    0:  "x ↛ y (no causation)",
}

# ══════════════════════════════════════════════════════════════
#  核心预测函数
# ══════════════════════════════════════════════════════════════

def CausalInferencePredictor(x, y, model_path, model_width=2, input_dim=512,
                             device='cuda', threshold=0.333,
                             auto_pad=True, pad_value=0):
    """
    对单对或批量 visual token 进行因果方向预测。
    当 auto_pad=True 时，自动将 x/y 在最后一维填充或截断到 input_dim，
    无需调用方手动对齐维度。

    Args:
        x, y:        cause/effect 候选，shape (D,) 或 (B, D)
        model_path:  checkpoint 路径
        model_width: 模型宽度乘子
        input_dim:   输入维度（同时作为自动填充的目标长度）
        device:      推理设备
        threshold:   分类阈值
        auto_pad:    是否自动填充/截断到 input_dim
        pad_value:   填充值

    Returns:
        dict with keys: raw_output, labels, descriptions, confidence
    """
    model, device = load_model(model_path, model_width, input_dim, device)

    target_len = input_dim if auto_pad else None
    x_tensor = _to_tensor(x, device, target_length=target_len, pad_value=pad_value, dim=-1)
    y_tensor = _to_tensor(y, device, target_length=target_len, pad_value=pad_value, dim=-1)

    assert x_tensor.shape == y_tensor.shape, \
        f"x 和 y 形状不匹配: {x_tensor.shape} vs {y_tensor.shape}"

    # import pdb; pdb.set_trace()
    with torch.no_grad():
        outputs = model(x_tensor, y_tensor)

    if outputs.dim() == 0:
        outputs = outputs.unsqueeze(0)

    raw_np = outputs.cpu().numpy()
    labels = [_output_to_label(v, threshold) for v in raw_np]
    descriptions = [LABEL_DESC[l] for l in labels]
    confidence = np.abs(raw_np)

    return {
        'raw_output':   raw_np,
        'labels':       labels,
        'descriptions': descriptions,
        'confidence':   confidence,
    }

def detect_batch_from_pairs(pairs, model_path, model_width=0.5,
                            input_dim=512, device='cuda', threshold=0.333,
                            auto_pad=True, pad_value=0):
    """
    批量预测，输入为 (x, y) 对的列表。
    支持变长输入，自动填充到 input_dim。
    """
    if not pairs:
        return []

    target_len = input_dim if auto_pad else None
    xs = torch.stack([
        _to_tensor(p[0], 'cpu', target_length=target_len,
                   pad_value=pad_value, dim=-1).squeeze(0)
        for p in pairs
    ])
    ys = torch.stack([
        _to_tensor(p[1], 'cpu', target_length=target_len,
                   pad_value=pad_value, dim=-1).squeeze(0)
        for p in pairs
    ])

    result = CausalInferencePredictor(
        xs, ys, model_path, model_width=model_width,
        input_dim=input_dim, device=device, threshold=threshold,
        auto_pad=auto_pad, pad_value=pad_value,
    )

    return [
        {
            'raw_output':  float(result['raw_output'][i]),
            'label':       result['labels'][i],
            'description': result['descriptions'][i],
            'confidence':  float(result['confidence'][i]),
        }
        for i in range(len(pairs))
    ]

# ══════════════════════════════════════════════════════════════
#  DataLoader collate（改进版，自动推断目标长度）
# ══════════════════════════════════════════════════════════════

def collate_fn_auto_pad(batch, fixed_length=None, pad_value=0, dim=-1):
    """
    改进版 collate_fn：
    - fixed_length=None 时，自动取 batch 内最大长度对齐（动态填充）
    - fixed_length=int  时，退化为固定长度填充/截断

    这样训练时无需预设长度，推理时也能灵活适配。
    """
    x_batch, y_batch, labels = [], [], []

    for x, y, label in batch:
        x_batch.append(x)
        y_batch.append(y)
        labels.append(label)

    # 自动推断目标长度
    if fixed_length is None:
        ndim = dim % x_batch[0].dim()
        max_x = max(t.shape[ndim] for t in x_batch)
        max_y = max(t.shape[ndim] for t in y_batch)
        target_x, target_y = max_x, max_y
    else:
        target_x = target_y = fixed_length

    x_batch = [_pad_or_truncate(t, target_x, pad_value, dim) for t in x_batch]
    y_batch = [_pad_or_truncate(t, target_y, pad_value, dim) for t in y_batch]

    return torch.stack(x_batch), torch.stack(y_batch), torch.stack(labels)

    