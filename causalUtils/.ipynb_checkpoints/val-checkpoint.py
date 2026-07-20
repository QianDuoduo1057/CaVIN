import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from causalNet import CausalInferenceResNetDeep, CausalInferenceResNetDeepAblation
from data import CausalDataset
from negativeSample import generate_intervention_negatives
import os
import csv

# ── FLOPs 依赖 ──────────────────────────────────────────────
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  未安装 thop，请运行: pip install thop")

try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("⚠️  未安装 torchinfo（备用方案），请运行: pip install torchinfo")
# ────────────────────────────────────────────────────────────

def collate_fn_fixed_length(batch, fixed_length=1024, pad_value=0, dim=-1):
    """将批次中的所有序列统一到固定长度，用指定值填充"""
    x_batch = []
    y_batch = []
    labels = []
    
    for item in batch:
        x, y, label = item
        x = x[:256]
        y = y[:256]
        x_len = x.shape[dim]
        y_len = y.shape[dim]

        if x_len > fixed_length:
            slice_obj = [slice(None)] * x.dim()
            slice_obj[dim] = slice(0, fixed_length)
            x = x[slice_obj]
        else:
            padding_size = fixed_length - x_len
            if padding_size > 0:
                pad = [0] * (2 * x.dim())
                pad[-2*(dim+1) + 1] = padding_size
                x = torch.nn.functional.pad(x, pad, mode='constant', value=pad_value)
        
        if y_len > fixed_length:
            slice_obj = [slice(None)] * y.dim()
            slice_obj[dim] = slice(0, fixed_length)
            y = y[slice_obj]
        else:
            padding_size = fixed_length - y_len
            if padding_size > 0:
                pad = [0] * (2 * y.dim())
                pad[-2*(dim+1) + 1] = padding_size
                y = torch.nn.functional.pad(y, pad, mode='constant', value=pad_value)
        
        x_batch.append(x)
        y_batch.append(y)
        labels.append(label)
    
    return torch.stack(x_batch), torch.stack(y_batch), torch.stack(labels)

# ══════════════════════════════════════════════════════════════
#  FLOPs 计算（自动形状探测 + torchinfo 兜底）
# ══════════════════════════════════════════════════════════════

def _try_thop(model, shape, device):
    """用指定形状尝试一次 thop.profile，返回 (flops, params) 或抛出异常。"""
    dummy_x = torch.randn(*shape).to(device)
    dummy_y = torch.randn(*shape).to(device)
    flops, params = profile(model, inputs=(dummy_x, dummy_y), verbose=False)
    return flops, params, shape

def _try_torchinfo(model, shape, device):
    """用 torchinfo 计算，返回结果字典或 None。"""
    if not TORCHINFO_AVAILABLE:
        return None
    try:
        dummy_x = torch.randn(*shape).to(device)
        dummy_y = torch.randn(*shape).to(device)
        info = torchinfo_summary(
            model,
            input_data=[dummy_x, dummy_y],
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
        )
        return {
            'flops': info.total_mult_adds,
            'flops_readable': f"{info.total_mult_adds / 1e9:.3f}G",
            'params_thop': info.total_params,
            'params_thop_readable': f"{info.total_params / 1e6:.3f}M",
            'input_shape': shape,
            'source': 'torchinfo',
        }
    except Exception as e:
        print(f"    torchinfo 也失败: {e}")
        return None

def calculate_flops(model, input_dim=512, fixed_length=512, device='cpu', batch_size=1):
    """
    自动探测正确输入形状并计算 FLOPs。

    探测顺序（thop）：
      1. (B, input_dim)              ← 2-D，最常见
      2. (B, 1, input_dim)           ← 3-D，seq_len=1
      3. (B, input_dim, 1)           ← 3-D，channel-last=1
      4. (B, input_dim, fixed_length) ← 3-D，完整序列

    若 thop 全部失败，尝试 torchinfo（从形状 1 开始）。
    """
    if not THOP_AVAILABLE and not TORCHINFO_AVAILABLE:
        print("⚠️  thop 和 torchinfo 均未安装，跳过 FLOPs 计算。")
        return None

    model.eval()

    # ── 候选形状，优先级从高到低 ─────────────────────────────
    candidate_shapes = [
        (batch_size, input_dim),                      # ① 2-D
        (batch_size, 1, input_dim),                   # ② 3-D seq=1 (B,1,C)
        (batch_size, input_dim, 1),                   # ③ 3-D seq=1 (B,C,1)
        (batch_size, input_dim, fixed_length),        # ④ 3-D 全长
    ]

    last_error = None

    # ── 先用 thop 逐一尝试 ──────────────────────────────────
    if THOP_AVAILABLE:
        for shape in candidate_shapes:
            try:
                flops, params, used_shape = _try_thop(model, shape, device)
                flops_r, params_r = clever_format([flops, params], "%.3f")

                _print_flops_banner(used_shape, flops, flops_r, params, params_r, source='thop')

                return {
                    'flops': flops,
                    'flops_readable': flops_r,
                    'params_thop': params,
                    'params_thop_readable': params_r,
                    'input_shape': used_shape,
                    'source': 'thop',
                }
            except Exception as e:
                print(f"  ✗ thop 尝试形状 {shape} 失败: {e}")
                last_error = e

    # ── thop 全失败，尝试 torchinfo ─────────────────────────
    print(f"\n  thop 全部形状均失败，切换至 torchinfo …")
    for shape in candidate_shapes:
        result = _try_torchinfo(model, shape, device)
        if result is not None:
            _print_flops_banner(
                result['input_shape'],
                result['flops'], result['flops_readable'],
                result['params_thop'], result['params_thop_readable'],
                source='torchinfo',
            )
            return result

    print(f"❌ FLOPs 计算完全失败，最后一个错误: {last_error}")
    return None

def _print_flops_banner(shape, flops, flops_r, params, params_r, source='thop'):
    """统一的 FLOPs 打印格式。"""
    print("\n" + "=" * 60)
    print(f"📊 模型复杂度分析 (FLOPs)  [{source}]")
    print("=" * 60)
    print(f"  输入形状 x / y    : {shape}")
    print(f"  FLOPs (单次前向)  : {flops_r}  ({flops:,.0f})")
    print(f"  参数量            : {params_r}  ({params:,.0f} 个参数)")
    print("=" * 60)

# ══════════════════════════════════════════════════════════════
#  其余函数保持不变
# ══════════════════════════════════════════════════════════════
def calculate_interval_accuracy(outputs, labels, size):
    """
    根据标签值计算准确率：
    - label=1: 预测值在[0.333, 1]区间算正确
    - label=0: 预测值在(-0.333, 0.333)区间算正确  
    - label=-1: 预测值在[-1, -0.333]区间算正确
    返回整体准确率和每个类别的准确率
    """
    outputs = outputs.squeeze()
    labels = labels.squeeze()
    
    # 处理单个样本的情况
    if outputs.dim() == 0:
        outputs = outputs.unsqueeze(0)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    
    total_correct = 0
    total_samples = len(outputs)
    
    # 初始化每个类别的统计
    class_stats = {
        -1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
        0: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
        1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []}
    }
    
    for output, label in zip(outputs, labels):
        label_int = round(label.item())
        output_val = output.item()
        
        # 记录预测类别
        if output_val > size:
            pred_class = 1
        elif output_val < -size:
            pred_class = -1
        else:
            pred_class = 0
        
        # 统计每个类别的样本数
        if label_int in class_stats:
            class_stats[label_int]['total'] += 1
            class_stats[label_int]['outputs'].append(output_val)
            class_stats[label_int]['predictions'].append(pred_class)
        
        # 检查是否正确
        correct = False
        if label_int == 1:
            if size < output_val <= 1:
                correct = True
                total_correct += 1
        elif label_int == -1:
            if -1 <= output_val < -size:
                correct = True
                total_correct += 1
        elif label_int == 0:
            if -size <= output_val <= size:
                correct = True
                total_correct += 1

        
        # 记录每个类别的正确数
        if correct and label_int in class_stats:
            class_stats[label_int]['correct'] += 1
    
    # 计算每个类别的准确率
    class_accuracies = {}
    for class_label in [-1, 0, 1]:
        total = class_stats[class_label]['total']
        correct = class_stats[class_label]['correct']
        accuracy = correct / total if total > 0 else 0
        class_accuracies[class_label] = accuracy
        
        # 计算每个类别的平均预测值（可选）
        if len(class_stats[class_label]['outputs']) > 0:
            avg_output = np.mean(class_stats[class_label]['outputs'])
            class_stats[class_label]['avg_output'] = avg_output
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return total_correct, total_samples, overall_accuracy, class_accuracies, class_stats

def print_class_statistics(class_stats, prefix=""):
    for class_label in [-1, 0, 1]:
        stats = class_stats[class_label]
        if stats['total'] > 0:
            accuracy  = stats['correct'] / stats['total']
            avg_output = np.mean(stats['outputs']) if stats['outputs'] else 0
            pred_dist = {
                pc: stats['predictions'].count(pc) / stats['total']
                for pc in [-1, 0, 1]
            }
            print(f"{prefix}类别 {class_label:2d}: 准确率={accuracy:.4f}, 样本数={stats['total']}, "
                  f"平均输出={avg_output:.4f}, 预测分布: "
                  f"-1={pred_dist[-1]:.3f}, 0={pred_dist[0]:.3f}, 1={pred_dist[1]:.3f}")

def validate(model_path='models/use/final_model.pth',
             val_data_file='datasets/val.jsonl',
             batch_size=32, model_width=1, device='cuda', size=0,
             use_intervention=True,
             intervention_alpha=0.1, intervention_steps=3,
             save_results=True, result_save_path=None,
             compute_flops=True,
             flops_input_dim=512, flops_fixed_length=512):

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── 加载模型 ─────────────────────────────────────────────
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = CausalInferenceResNetDeep(input_dim=512, width_multiplier=model_width)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if 'epoch'   in checkpoint: print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'val_acc' in checkpoint: print(f"Checkpoint val accuracy: {checkpoint['val_acc']*100:.2f}%")
    if 'val_loss'in checkpoint: print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")

    # ── FLOPs 计算（验证前执行）────────────────────────────
    flops_result = None
    if compute_flops:
        flops_result = calculate_flops(
            model,
            input_dim=flops_input_dim,
            fixed_length=flops_fixed_length,
            device=device,
            batch_size=1,
        )

    # ── 加载数据 ─────────────────────────────────────────────
    print(f"\nLoading validation dataset from: {val_data_file}")
    val_dataset = CausalDataset(val_data_file)
    val_loader  = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0,
        collate_fn=lambda b: collate_fn_fixed_length(b, fixed_length=512)
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    criterion = nn.MSELoss()
    print(f"\n验证配置:")
    print(f"  use_intervention={use_intervention}")
    if use_intervention:
        print(f"  intervention_alpha={intervention_alpha}, intervention_steps={intervention_steps}")

# ══════════════════ 验证循环 ══════════════════════════════
    print("\n开始验证...")

    val_loss    = 0.0
    val_correct = 0
    val_total   = 0
    val_class_stats = {
        -1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
         0: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
         1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []}
    }

    original_stats = {
        'loss': 0.0, 'correct': 0, 'total': 0,
        'class_stats': {
            -1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
             0: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
             1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []}
        }
    }
    intervention_stats = {
        'loss': 0.0, 'correct': 0, 'total': 0,
        'outputs': [], 'predictions': []
    }

    val_progress = tqdm(val_loader, desc='Validating')
    for batch_idx, (x_batch, y_batch, labels) in enumerate(val_progress):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        labels  = labels.to(device).squeeze()

        # ── 原始样本评估 ─────────────────────────────────────
        with torch.no_grad():
            outputs = model(x_batch, y_batch)
            loss    = criterion(outputs, labels)

        all_outputs = [outputs]
        all_labels  = [labels]
        all_loss    = loss.item() * x_batch.size(0)

        original_stats['loss'] += loss.item() * x_batch.size(0)
        batch_correct, batch_size_cur, _, _, batch_class_stats = \
            calculate_interval_accuracy(outputs, labels, size)
        original_stats['correct'] += batch_correct
        original_stats['total']   += batch_size_cur
        for cl in [-1, 0, 1]:
            original_stats['class_stats'][cl]['correct']    += batch_class_stats[cl]['correct']
            original_stats['class_stats'][cl]['total']      += batch_class_stats[cl]['total']
            original_stats['class_stats'][cl]['outputs'].extend(batch_class_stats[cl]['outputs'])
            original_stats['class_stats'][cl]['predictions'].extend(batch_class_stats[cl]['predictions'])

        # ── 干预负样本评估 ───────────────────────────────────
        if use_intervention:
            labels_int = torch.round(labels).long()
            pos_mask   = (labels_int == 1)

            if pos_mask.sum() > 0:
                x_pos = x_batch[pos_mask]
                y_pos = y_batch[pos_mask]

                x_tilde = generate_intervention_negatives(
                    model, x_pos, y_pos,
                    alpha=intervention_alpha,
                    n_steps=intervention_steps
                )

                with torch.no_grad():
                    intervene_outputs = model(x_tilde, y_pos)
                    intervene_labels  = torch.zeros(x_tilde.size(0), device=device)
                    intervene_loss    = criterion(intervene_outputs, intervene_labels)

                all_outputs.append(intervene_outputs)
                all_labels.append(intervene_labels)
                all_loss += intervene_loss.item() * x_tilde.size(0)

                intervention_stats['loss'] += intervene_loss.item() * x_tilde.size(0)
                int_correct, int_size, _, _, int_class_stats = \
                    calculate_interval_accuracy(intervene_outputs, intervene_labels, size)
                intervention_stats['correct'] += int_correct
                intervention_stats['total']   += int_size
                intervention_stats['outputs'].extend(int_class_stats[0]['outputs'])
                intervention_stats['predictions'].extend(int_class_stats[0]['predictions'])

        # ── 合并统计 ─────────────────────────────────────────
        all_outputs_cat = torch.cat(all_outputs, dim=0)
        all_labels_cat  = torch.cat(all_labels,  dim=0)

        val_loss += all_loss

        batch_correct, batch_size_cur, _, _, batch_class_stats = \
            calculate_interval_accuracy(all_outputs_cat, all_labels_cat, size)
        val_correct += batch_correct
        val_total   += batch_size_cur

        for cl in [-1, 0, 1]:
            val_class_stats[cl]['correct']    += batch_class_stats[cl]['correct']
            val_class_stats[cl]['total']      += batch_class_stats[cl]['total']
            val_class_stats[cl]['outputs'].extend(batch_class_stats[cl]['outputs'])
            val_class_stats[cl]['predictions'].extend(batch_class_stats[cl]['predictions'])

        val_progress.set_postfix({
            'loss': f"{all_loss / max(batch_size_cur, 1):.4f}",
            'acc' : f"{val_correct / max(val_total, 1):.4f}"
        })

    # ══════════════════ 结果汇总 ══════════════════════════════
    avg_val_loss = val_loss    / max(val_total, 1)
    avg_val_acc  = val_correct / max(val_total, 1)

    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    title = "【总体结果】（含干预负样本）" if use_intervention else "【总体结果】"
    print(f"\n{title}")
    print(f"  总样本数  : {val_total}")
    print(f"  平均损失  : {avg_val_loss:.4f}")
    print(f"  总体准确率: {avg_val_acc * 100:.2f}%")

    print("\n【每个类别详细统计】")
    print_class_statistics(val_class_stats, prefix="  ")

    print("\n【每个类别准确率】")
    val_class_acc = {}
    for cl in [-1, 0, 1]:
        if val_class_stats[cl]['total'] > 0:
            val_class_acc[cl] = val_class_stats[cl]['correct'] / val_class_stats[cl]['total']
            print(f"  类别 {cl:2d}: {val_class_acc[cl] * 100:.2f}%")
        else:
            val_class_acc[cl] = 0
            print(f"  类别 {cl:2d}: 无样本")

    print("\n【原始样本统计】")
    if original_stats['total'] > 0:
        orig_acc  = original_stats['correct'] / original_stats['total']
        orig_loss = original_stats['loss']    / original_stats['total']
        print(f"  样本数  : {original_stats['total']}")
        print(f"  平均损失: {orig_loss:.4f}")
        print(f"  准确率  : {orig_acc * 100:.2f}%")
        print("\n  原始样本每个类别统计:")
        print_class_statistics(original_stats['class_stats'], prefix="    ")

    # 用于后续 CSV 写入，先在此处计算好
    int_acc  = 0.0
    int_loss = 0.0
    if use_intervention and intervention_stats['total'] > 0:
        int_acc  = intervention_stats['correct'] / intervention_stats['total']
        int_loss = intervention_stats['loss']    / intervention_stats['total']
        print("\n【干预负样本统计】")
        print(f"  样本数    : {intervention_stats['total']}")
        print(f"  平均损失  : {int_loss:.4f}")
        print(f"  准确率    : {int_acc * 100:.2f}%")
        if intervention_stats['outputs']:
            avg_int_out = np.mean(intervention_stats['outputs'])
            print(f"  平均输出值: {avg_int_out:.4f}")

    # ── FLOPs 汇总展示 ───────────────────────────────────────
    if flops_result:
        print("\n【模型复杂度（FLOPs）】")
        print(f"  计算工具           : {flops_result['source']}")
        print(f"  输入形状 x / y     : {flops_result['input_shape']}")
        print(f"  FLOPs (单次前向)   : {flops_result['flops_readable']}"
              f"  ({flops_result['flops']:,.0f})")
        print(f"  参数量             : {flops_result['params_thop_readable']}"
              f"  ({flops_result['params_thop']:,.0f} 个参数)")

    # ══════════════════ 保存 CSV ══════════════════════════════
    if save_results:
        if result_save_path is None:
            model_dir = os.path.dirname(model_path)
            result_save_path = (
                os.path.join(model_dir, 'val_results.csv') if model_dir
                else 'val_results.csv'
            )

        result_dir = os.path.dirname(result_save_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        with open(result_save_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)

            # 基础信息
            w.writerow(['Metric', 'Value'])
            w.writerow(['Model Path',       model_path])
            w.writerow(['Validation Data',  val_data_file])
            w.writerow(['Use Intervention', use_intervention])
            w.writerow(['Total Samples',    val_total])
            w.writerow(['Average Loss',     f'{avg_val_loss:.4f}'])
            w.writerow(['Overall Accuracy', f'{avg_val_acc * 100:.2f}%'])

            # FLOPs
            if flops_result:
                w.writerow([])
                w.writerow(['--- FLOPs ---', ''])
                w.writerow(['FLOPs Source',   flops_result['source']])
                w.writerow(['Input Shape',    str(flops_result['input_shape'])])
                w.writerow(['FLOPs (readable)', flops_result['flops_readable']])
                w.writerow(['FLOPs (raw)',    f"{flops_result['flops']:,.0f}"])
                w.writerow(['Params (readable)', flops_result['params_thop_readable']])
                w.writerow(['Params (raw)',   f"{flops_result['params_thop']:,.0f}"])

            # 类别准确率
            w.writerow([])
            w.writerow(['--- Per-Class Results ---', ''])
            w.writerow(['Class', 'Accuracy', 'Samples', 'Correct'])
            for cl in [-1, 0, 1]:
                stats = val_class_stats[cl]
                acc   = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                w.writerow([cl, f'{acc * 100:.2f}%', stats['total'], stats['correct']])

            # 干预样本
            if use_intervention and intervention_stats['total'] > 0:
                w.writerow([])
                w.writerow(['--- Intervention Stats ---', ''])
                w.writerow(['Intervention Samples',  intervention_stats['total']])
                w.writerow(['Intervention Accuracy', f'{int_acc * 100:.2f}%'])
                w.writerow(['Intervention Loss',     f'{int_loss:.4f}'])

        print(f"\n结果已保存到: {result_save_path}")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)

    return {
        'loss':               avg_val_loss,
        'accuracy':           avg_val_acc,
        'class_accuracies':   val_class_acc,
        'class_stats':        val_class_stats,
        'original_stats':     original_stats,
        'intervention_stats': intervention_stats if use_intervention else None,
        'flops_result':       flops_result,
    }

# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== 模型验证 ===")
    results = validate(
        model_path='models/ablation/params3.22M_width2_data150k_step2_alpha0.3_noProd/best_model.pth',
        val_data_file='datasets/val.jsonl',
        batch_size=32,
        model_width=2,
        size=0.15,    #Acc
        device='cuda',
        use_intervention=True,
        intervention_alpha=0.3,
        intervention_steps=3,
        save_results=True,
        result_save_path=None,
        compute_flops=True,
        flops_input_dim=512,
        flops_fixed_length=512,
    )