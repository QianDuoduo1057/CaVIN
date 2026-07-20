import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from causalNet import CausalInferenceResNetDeep, CausalInferenceResNetDeepAblation
from data import read_jsonl_file, get_jsonl_length_basic, CausalDataset
from negativeSample import generate_intervention_negatives
import os
import csv
from plot import plot_training_history
from torch.profiler import profile, ProfilerActivity

def collate_fn_fixed_length(batch, fixed_length=1024, pad_value=0, dim=-1):
    x_batch, y_batch, labels = [], [], []
    for item in batch:
        x, y, label = item
        x = x[:256]
        y = y[:256]
        x_len = x.shape[dim]
        y_len = y.shape[dim]

        if x_len > fixed_length:
            slice_obj = [slice(None)] * x.dim()
            slice_obj[dim] = slice(0, fixed_length)
            x = x[tuple(slice_obj)]
        else:
            padding_size = fixed_length - x_len
            if padding_size > 0:
                pad = [0] * (2 * x.dim())
                pad[-2*(dim+1) + 1] = padding_size
                x = torch.nn.functional.pad(x, pad, mode='constant', value=pad_value)

        if y_len > fixed_length:
            slice_obj = [slice(None)] * y.dim()
            slice_obj[dim] = slice(0, fixed_length)
            y = y[tuple(slice_obj)]
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

def calculate_interval_accuracy(outputs, labels, size):
    outputs = outputs.squeeze()
    labels = labels.squeeze()
    if outputs.dim() == 0:
        outputs = outputs.unsqueeze(0)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    total_correct = 0
    total_samples = len(outputs)
    class_stats = {
        -1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
         0: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
         1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []}
    }

    for output, label in zip(outputs, labels):
        label_int = round(label.item())
        output_val = output.item()

        if output_val > size:
            pred_class = 1
        elif output_val < -size:
            pred_class = -1
        else:
            pred_class = 0

        if label_int in class_stats:
            class_stats[label_int]['total'] += 1
            class_stats[label_int]['outputs'].append(output_val)
            class_stats[label_int]['predictions'].append(pred_class)

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

        if correct and label_int in class_stats:
            class_stats[label_int]['correct'] += 1

    class_accuracies = {}
    for class_label in [-1, 0, 1]:
        total = class_stats[class_label]['total']
        correct = class_stats[class_label]['correct']
        class_accuracies[class_label] = correct / total if total > 0 else 0
        if len(class_stats[class_label]['outputs']) > 0:
            class_stats[class_label]['avg_output'] = np.mean(class_stats[class_label]['outputs'])

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    return total_correct, total_samples, overall_accuracy, class_accuracies, class_stats

def print_class_statistics(class_stats, prefix=""):
    for class_label in [-1, 0, 1]:
        stats = class_stats[class_label]
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            avg_output = np.mean(stats['outputs']) if stats['outputs'] else 0
            pred_dist = {pc: stats['predictions'].count(pc) / stats['total'] for pc in [-1, 0, 1]}
            print(f"{prefix}类别 {class_label:2d}: 准确率={accuracy:.4f}, 样本数={stats['total']}, "
                  f"平均输出={avg_output:.4f}, 预测分布: -1={pred_dist[-1]:.3f}, 0={pred_dist[0]:.3f}, 1={pred_dist[1]:.3f}")

def train(train_data_file=None, val_data_file=None, width=None, model_save_path='exp0', resume=None, size=0,
          target_dim=256, hidden_dim=128, batch_size=32,
          learning_rate=0.0001, num_epochs=30, device='cuda', early_stop_patience=None,
          intervention_alpha=0.1, intervention_steps=3, intervention_lambda=0.5,
          intervention_warmup=5):

    save_dir = os.path.join('models', model_save_path)
    os.makedirs(save_dir, exist_ok=True)

    final_model_path = os.path.join(save_dir, 'final_model.pth')
    best_model_path  = os.path.join(save_dir, 'best_model.pth')
    plot_save_path   = os.path.join(save_dir, 'training_history.png')
    csv_save_path    = os.path.join(save_dir, 'training_history.csv')

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if resume is not None:
        checkpoint = torch.load(resume)
        model = CausalInferenceResNetDeep(input_dim=512, width_multiplier=width)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = CausalInferenceResNetDeep(input_dim=512, width_multiplier=width)

    print(f"Model architecture:\n{model}")
    with torch.no_grad():
        dummy_x = torch.randn(4, 512)
        dummy_y = torch.randn(4, 512)
        model(dummy_x, dummy_y)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ===== FLOPs 测试 =====
    print("\n=== Measuring FLOPs ===")
    flops_x = torch.randn(4, 512).to(device)
    flops_y = torch.randn(4, 512).to(device)
    model.to(device)
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
            with torch.no_grad():
                model(flops_x, flops_y)
        total_flops = sum([item.flops for item in prof.key_averages()])
        print(f"FLOPs (batch=4): {total_flops/1e9:.2f} GFLOPs")
        print(f"FLOPs per sample: {total_flops/4/1e9:.4f} GFLOPs")
    except Exception as e:
        print(f"Profiler failed: {e}")
        estimated_flops = 2 * total_params
        print(f"Estimated FLOPs per sample: {estimated_flops/1e9:.4f} GFLOPs")
    print("=" * 50)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_dataset = CausalDataset(train_data_file)
    val_dataset   = CausalDataset(val_data_file)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
        collate_fn=lambda batch: collate_fn_fixed_length(batch, fixed_length=512)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0,
        collate_fn=lambda batch: collate_fn_fixed_length(batch, fixed_length=512)
    )

    print(f"\n干预负样本配置:")
    print(f"  alpha={intervention_alpha}, steps={intervention_steps}, "
          f"lambda={intervention_lambda}, warmup={intervention_warmup} epochs")
    print(f"\n开始训练，训练历史将保存到: {csv_save_path}")
    if early_stop_patience is not None:
        print(f"启用早停机制，耐心值为: {early_stop_patience} epochs")

    history, trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        best_model_path, plot_save_path, csv_save_path, size, num_epochs, device,
        early_stop_patience,
        intervention_alpha=intervention_alpha,
        intervention_steps=intervention_steps,
        intervention_lambda=intervention_lambda,
        intervention_warmup=intervention_warmup
    )

    print("\n在验证集上测试模型（包含干预负样本）...")
    trained_model.eval()
    test_correct = 0
    test_total   = 0
    test_loss    = 0.0
    test_class_stats = {
        -1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
         0: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
         1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []}
    }

    for x_batch, y_batch, labels in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        labels  = labels.to(device).squeeze()

        with torch.no_grad():
            outputs = trained_model(x_batch, y_batch)
            loss    = criterion(outputs, labels)
            test_loss += loss.item() * x_batch.size(0)

        batch_correct, batch_sz, _, _, batch_class_stats = calculate_interval_accuracy(outputs, labels, size)
        test_correct += batch_correct
        test_total   += batch_sz
        for cl in [-1, 0, 1]:
            test_class_stats[cl]['correct']    += batch_class_stats[cl]['correct']
            test_class_stats[cl]['total']      += batch_class_stats[cl]['total']
            test_class_stats[cl]['outputs'].extend(batch_class_stats[cl]['outputs'])
            test_class_stats[cl]['predictions'].extend(batch_class_stats[cl]['predictions'])

        labels_int = torch.round(labels).long()
        pos_mask   = (labels_int == 1)
        if pos_mask.sum() > 0:
            x_pos = x_batch[pos_mask]
            y_pos = y_batch[pos_mask]
            x_tilde = generate_intervention_negatives(
                trained_model, x_pos, y_pos,
                alpha=intervention_alpha, n_steps=intervention_steps
            )
            with torch.no_grad():
                intervene_outputs = trained_model(x_tilde, y_pos)
                intervene_labels  = torch.zeros(x_tilde.size(0), device=device)
                intervene_loss    = criterion(intervene_outputs, intervene_labels)
                test_loss += intervene_loss.item() * x_tilde.size(0)

            batch_correct, batch_sz, _, _, batch_class_stats = calculate_interval_accuracy(
                intervene_outputs, intervene_labels, size
            )
            test_correct += batch_correct
            test_total   += batch_sz
            for cl in [-1, 0, 1]:
                test_class_stats[cl]['correct']    += batch_class_stats[cl]['correct']
                test_class_stats[cl]['total']      += batch_class_stats[cl]['total']
                test_class_stats[cl]['outputs'].extend(batch_class_stats[cl]['outputs'])
                test_class_stats[cl]['predictions'].extend(batch_class_stats[cl]['predictions'])

    avg_test_loss = test_loss / max(test_total, 1)
    test_accuracy = test_correct / max(test_total, 1)

    test_class_accuracies = {}
    for cl in [-1, 0, 1]:
        t = test_class_stats[cl]['total']
        c = test_class_stats[cl]['correct']
        test_class_accuracies[cl] = c / t if t > 0 else 0

    print(f"最终验证损失（含干预负样本）: {avg_test_loss:.4f}")
    print(f"最终验证精度（含干预负样本）: {test_accuracy:.4f}")
    print("\n最终测试每个类别的详细统计:")
    print_class_statistics(test_class_stats, prefix="  ")
    for cl in [-1, 0, 1]:
        if test_class_stats[cl]['total'] > 0:
            print(f"  类别 {cl:2d} 准确率: {test_class_accuracies[cl]:.4f}")

    torch.save({'model_state_dict': trained_model.state_dict(), 'test_accuracy': test_accuracy}, final_model_path)

    with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv.writer(csvfile).writerow([
            'Final Test', '', '', '', '',
            avg_test_loss, test_accuracy, '', '', '',
            test_class_accuracies.get(-1, 0),
            test_class_accuracies.get(0, 0),
            test_class_accuracies.get(1, 0)
        ])

    print(f"\n训练完成!")
    print(f"模型保存为: {final_model_path}")
    print(f"最佳模型保存为: {best_model_path}")
    print(f"训练历史已保存为: {csv_save_path}")
    return trained_model, history

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                best_model_path, plot_save_path, csv_save_path,
                size=0, num_epochs=50, device='cuda', early_stop_patience=None,
                intervention_alpha=0.1, intervention_steps=3, intervention_lambda=0.5,
                intervention_warmup=5):

    model = model.to(device)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'train_class_acc': {-1: [], 0: [], 1: []},
        'val_class_acc':   {-1: [], 0: [], 1: []}
    }

    with open(csv_save_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            'Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy',
            'Train Class -1 Acc', 'Train Class 0 Acc', 'Train Class 1 Acc',
            'Val Class -1 Acc',   'Val Class 0 Acc',   'Val Class 1 Acc'
        ])

    best_val_loss      = float('inf')
    best_val_acc       = 0.0
    best_epoch         = 0
    early_stop_counter = 0
    min_val_loss       = float('inf')
    stop_training      = False

    for epoch in range(num_epochs):
        if stop_training:
            print(f"训练在第{epoch+1}轮提前结束，验证损失连续{early_stop_patience}个epoch没有降低")
            break

        use_intervention = (epoch >= intervention_warmup)
        if epoch == intervention_warmup:
            print(f"  [Intervention] Warmup结束，从Epoch {epoch+1}开始启用干预负样本生成")

        # ══════════════ 训练阶段 ══════════════
        model.train()
        train_loss_sum = 0.0
        train_correct  = 0
        train_total    = 0
        train_class_stats = {cl: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []} for cl in [-1, 0, 1]}

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for x_batch, y_batch, labels in train_progress:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            labels  = labels.to(device).squeeze()

            outputs = model(x_batch, y_batch)
            loss    = criterion(outputs, labels)

            all_outputs       = [outputs]
            all_labels        = [labels]
            weighted_loss_sum = loss.item() * x_batch.size(0)

            intervene_loss = torch.tensor(0.0, device=device)
            if use_intervention:
                pos_mask = (torch.round(labels).long() == 1)
                if pos_mask.sum() > 0:
                    x_pos   = x_batch[pos_mask]
                    y_pos   = y_batch[pos_mask]
                    x_tilde = generate_intervention_negatives(
                        model, x_pos, y_pos,
                        alpha=intervention_alpha, n_steps=intervention_steps
                    )
                    model.train()
                    intervene_outputs = model(x_tilde, y_pos)
                    intervene_labels  = torch.zeros(x_tilde.size(0), device=device)
                    intervene_loss    = intervention_lambda * criterion(intervene_outputs, intervene_labels)

                    all_outputs.append(intervene_outputs)
                    all_labels.append(intervene_labels)
                    weighted_loss_sum += intervene_loss.item() * x_tilde.size(0)

            total_loss = loss + intervene_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            all_outputs_cat = torch.cat(all_outputs, dim=0)
            all_labels_cat  = torch.cat(all_labels,  dim=0)

            train_loss_sum += weighted_loss_sum

            batch_correct, batch_sz, _, _, batch_class_stats = calculate_interval_accuracy(all_outputs_cat, all_labels_cat, size)
            train_correct += batch_correct
            train_total   += batch_sz

            for cl in [-1, 0, 1]:
                train_class_stats[cl]['correct']    += batch_class_stats[cl]['correct']
                train_class_stats[cl]['total']      += batch_class_stats[cl]['total']
                train_class_stats[cl]['outputs'].extend(batch_class_stats[cl]['outputs'])
                train_class_stats[cl]['predictions'].extend(batch_class_stats[cl]['predictions'])

            train_progress.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc':  f"{train_correct / max(train_total, 1):.4f}"
            })

        avg_train_loss = train_loss_sum / max(train_total, 1)
        avg_train_acc  = train_correct  / max(train_total, 1)

        # ══════════════ 验证阶段 ══════════════
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0
        val_class_stats = {cl: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []} for cl in [-1, 0, 1]}

        test_num = 0
        for x_batch, y_batch, labels in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            labels  = labels.to(device).squeeze()

            with torch.no_grad():
                outputs = model(x_batch, y_batch)
                loss    = criterion(outputs, labels)

            all_outputs    = [outputs]
            all_labels     = [labels]
            batch_loss_sum = loss.item() * x_batch.size(0)

            pos_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=device)
            if use_intervention:
                pos_mask = (torch.round(labels).long() == 1)
                if pos_mask.sum() > 0:
                    x_pos   = x_batch[pos_mask]
                    y_pos   = y_batch[pos_mask]
                    x_tilde = generate_intervention_negatives(
                        model, x_pos, y_pos,
                        alpha=intervention_alpha, n_steps=intervention_steps
                    )
                    with torch.no_grad():
                        intervene_outputs = model(x_tilde, y_pos)
                        intervene_labels  = torch.zeros(x_tilde.size(0), device=device)
                        intervene_loss    = criterion(intervene_outputs, intervene_labels)

                    all_outputs.append(intervene_outputs)
                    all_labels.append(intervene_labels)
                    batch_loss_sum += intervene_loss.item() * x_tilde.size(0)

            all_outputs_cat = torch.cat(all_outputs, dim=0)
            all_labels_cat  = torch.cat(all_labels,  dim=0)

            val_loss_sum += batch_loss_sum

            batch_correct, batch_sz, _, _, batch_class_stats = calculate_interval_accuracy(all_outputs_cat, all_labels_cat, size)
            val_correct += batch_correct
            val_total   += batch_sz

            for cl in [-1, 0, 1]:
                val_class_stats[cl]['correct']    += batch_class_stats[cl]['correct']
                val_class_stats[cl]['total']      += batch_class_stats[cl]['total']
                val_class_stats[cl]['outputs'].extend(batch_class_stats[cl]['outputs'])
                val_class_stats[cl]['predictions'].extend(batch_class_stats[cl]['predictions'])

            if test_num == 0:
                print(f"\n原始输出: {outputs[:5]}")
                if use_intervention and pos_mask.sum() > 0:
                    print(f"干预输出: {intervene_outputs[:5]}")
            test_num += 1

        avg_val_loss = val_loss_sum / max(val_total, 1)
        avg_val_acc  = val_correct  / max(val_total, 1)

        scheduler.step(avg_val_loss)

        train_class_acc = {}
        val_class_acc   = {}
        for cl in [-1, 0, 1]:
            t = train_class_stats[cl]['total']
            train_class_acc[cl] = train_class_stats[cl]['correct'] / t if t > 0 else 0
            t = val_class_stats[cl]['total']
            val_class_acc[cl]   = val_class_stats[cl]['correct']   / t if t > 0 else 0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        for cl in [-1, 0, 1]:
            history['train_class_acc'][cl].append(train_class_acc[cl])
            history['val_class_acc'][cl].append(val_class_acc[cl])

        with open(csv_save_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch + 1,
                avg_train_loss, avg_train_acc,
                avg_val_loss,   avg_val_acc,
                train_class_acc.get(-1, 0), train_class_acc.get(0, 0), train_class_acc.get(1, 0),
                val_class_acc.get(-1, 0),   val_class_acc.get(0, 0),   val_class_acc.get(1, 0)
            ])

        plot_training_history(history, plot_save_path)

        intervention_status = "ON" if use_intervention else "OFF (warmup)"
        print(f'\nEpoch {epoch+1}/{num_epochs} [Intervention: {intervention_status}]:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% (total samples: {train_total})')
        print(f'  Val Loss:   {avg_val_loss:.4f}, Val Acc:   {avg_val_acc*100:.2f}% (total samples: {val_total})')
        print("\n  训练集每个类别统计:")
        print_class_statistics(train_class_stats, prefix="    ")
        print("\n  验证集每个类别统计:")
        print_class_statistics(val_class_stats, prefix="    ")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc  = avg_val_acc
            best_epoch    = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss, 'train_acc': avg_train_acc,
                'val_loss':   avg_val_loss,   'val_acc':   avg_val_acc,
                'train_class_acc': train_class_acc,
                'val_class_acc':   val_class_acc
            }, best_model_path)
            print(f'  Best model saved: val_loss={best_val_loss:.4f}, val_acc={best_val_acc*100:.2f}%')

        if early_stop_patience is not None:
            if avg_val_loss < min_val_loss:
                min_val_loss       = avg_val_loss
                early_stop_counter = 0
                print(f'  Validation loss improved to {avg_val_loss:.4f}')
            else:
                early_stop_counter += 1
                print(f'  Validation loss did not improve for {early_stop_counter} epochs')
                if early_stop_counter >= early_stop_patience:
                    stop_training = True

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch}, accuracy: {best_val_acc*100:.2f}%)")
    return history, model

if __name__ == "__main__":
    print("=== 示例训练 ===")
    model, history = train(
        train_data_file='datasets/train150k.jsonl',
        val_data_file='datasets/val.jsonl',
        width=2,
        size=0.15,
        model_save_path='ablation/params3.22M_width2_data150k_step2_alpha0.3_noProd',
        resume=None,
        target_dim=256,
        hidden_dim=128,
        batch_size=32,
        learning_rate=0.0001,
        num_epochs=1000,
        device='cuda',
        early_stop_patience=30,
        intervention_alpha=0.3,
        intervention_steps=3,
        intervention_lambda=1,
        intervention_warmup=0
    )
