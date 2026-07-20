import os
import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from causalNet import CausalInferenceResNetDeep
# 假设你的数据和绘图工具模块
from data import CausalDataset
from plot import plot_training_history

def collate_fn_fixed_length(batch, fixed_length=512, pad_value=0, dim=-1):
    x_batch = []
    y_batch = []
    labels = []

    for item in batch:
        x, y, label = item
        #import pdb; pdb.set_trace()
        x = x[:261]
        y = y[:261]
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
                pad[-2 * (dim + 1) + 1] = padding_size
                x = torch.nn.functional.pad(x, pad, mode='constant', value=pad_value)

        if y_len > fixed_length:
            slice_obj = [slice(None)] * y.dim()
            slice_obj[dim] = slice(0, fixed_length)
            y = y[slice_obj]
        else:
            padding_size = fixed_length - y_len
            if padding_size > 0:
                pad = [0] * (2 * y.dim())
                pad[-2 * (dim + 1) + 1] = padding_size
                y = torch.nn.functional.pad(y, pad, mode='constant', value=pad_value)

        x_batch.append(x)
        y_batch.append(y)
        labels.append(label)

    return torch.stack(x_batch), torch.stack(y_batch), torch.stack(labels)

def calculate_interval_accuracy(outputs, labels):
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

        if output_val >= 0.333:
            pred_class = 1
        elif output_val <= -0.333:
            pred_class = -1
        else:
            pred_class = 0

        if label_int in class_stats:
            class_stats[label_int]['total'] += 1
            class_stats[label_int]['outputs'].append(output_val)
            class_stats[label_int]['predictions'].append(pred_class)

        correct = False
        if label_int == 1 and 0.333 <= output_val <= 1:
            correct = True
        elif label_int == -1 and -1 <= output_val <= -0.333:
            correct = True
        elif label_int == 0 and -0.333 < output_val < 0.333:
            correct = True

        if correct and label_int in class_stats:
            class_stats[label_int]['correct'] += 1
            total_correct += 1

    class_accuracies = {}
    for class_label in [-1, 0, 1]:
        total = class_stats[class_label]['total']
        correct = class_stats[class_label]['correct']
        class_accuracies[class_label] = correct / total if total > 0 else 0

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    return total_correct, total_samples, overall_accuracy, class_accuracies, class_stats

def train_teacher_student(teacher_model, student_model, train_loader, val_loader,
                          criterion, optimizer, best_model_path, plot_save_path, csv_save_path,
                          num_epochs=100, device='cuda', early_stop_patience=30,
                          lambda_kd=0.3, lambda_align=0.1):

    teacher_model.eval()
    teacher_model.to(device)

    student_model.train()
    student_model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_class_acc': {-1: [], 0: [], 1: []},
        'val_class_acc': {-1: [], 0: [], 1: []}
    }

    with open(csv_save_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy',
            'Train Class -1 Acc', 'Train Class 0 Acc', 'Train Class 1 Acc',
            'Val Class -1 Acc', 'Val Class 0 Acc', 'Val Class 1 Acc'
        ])

    best_val_loss = float('inf')
    early_stop_counter = 0
    min_val_loss = float('inf')
    stop_training = False

    for epoch in range(num_epochs):
        if stop_training:
            print(f"早停触发，训练终止于第{epoch}轮")
            break

        student_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_class_stats = {
            -1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
            0: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
            1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []}
        }

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for x_batch, y_batch, labels in train_progress:
            #import pdb; pdb.set_trace()
            x_batch, y_batch, labels = x_batch.to(device), y_batch.to(device), labels.to(device)
            labels = labels.squeeze()

            with torch.no_grad():
                teacher_outputs = teacher_model(x_batch, y_batch)
                teacher_hidden = teacher_model.get_hidden(x_batch, y_batch)

            student_outputs = student_model(x_batch, y_batch)
            student_hidden = student_model.get_hidden(x_batch, y_batch)

            loss_supervised = criterion(student_outputs, labels)
            loss_kd = criterion(student_outputs, teacher_outputs)
            loss_align = F.mse_loss(student_hidden, teacher_hidden)

            total_loss = loss_supervised + lambda_kd * loss_kd + lambda_align * loss_align

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * x_batch.size(0)

            batch_correct, batch_size, batch_acc, batch_class_acc, batch_class_stats = calculate_interval_accuracy(student_outputs, labels)
            train_correct += batch_correct
            train_total += batch_size

            for c in [-1, 0, 1]:
                train_class_stats[c]['correct'] += batch_class_stats[c]['correct']
                train_class_stats[c]['total'] += batch_class_stats[c]['total']
                train_class_stats[c]['outputs'].extend(batch_class_stats[c]['outputs'])
                train_class_stats[c]['predictions'].extend(batch_class_stats[c]['predictions'])

            train_progress.set_postfix(loss=total_loss.item(), acc=train_correct / max(train_total, 1))

        avg_train_loss = train_loss / max(len(train_loader) * train_loader.batch_size, 1)
        avg_train_acc = train_correct / max(train_total, 1)

        student_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_class_stats = {
            -1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
            0: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []},
            1: {'correct': 0, 'total': 0, 'outputs': [], 'predictions': []}
        }

        with torch.no_grad():
            for x_batch, y_batch, labels in val_loader:
                x_batch, y_batch, labels = x_batch.to(device), y_batch.to(device), labels.to(device)
                labels = labels.squeeze()

                teacher_outputs = teacher_model(x_batch, y_batch)
                teacher_hidden = teacher_model.get_hidden(x_batch, y_batch)

                student_outputs = student_model(x_batch, y_batch)
                student_hidden = student_model.get_hidden(x_batch, y_batch)

                loss_supervised = criterion(student_outputs, labels)
                loss_kd = criterion(student_outputs, teacher_outputs)
                loss_align = F.mse_loss(student_hidden, teacher_hidden)
                total_loss = loss_supervised + lambda_kd * loss_kd + lambda_align * loss_align

                val_loss += total_loss.item() * x_batch.size(0)

                batch_correct, batch_size, batch_acc, batch_class_acc, batch_class_stats = calculate_interval_accuracy(student_outputs, labels)
                val_correct += batch_correct
                val_total += batch_size

                for c in [-1, 0, 1]:
                    val_class_stats[c]['correct'] += batch_class_stats[c]['correct']
                    val_class_stats[c]['total'] += batch_class_stats[c]['total']
                    val_class_stats[c]['outputs'].extend(batch_class_stats[c]['outputs'])
                    val_class_stats[c]['predictions'].extend(batch_class_stats[c]['predictions'])

        avg_val_loss = val_loss / max(len(val_loader) * val_loader.batch_size, 1)
        avg_val_acc = val_correct / max(val_total, 1)
        train_class_acc = {c: (train_class_stats[c]['correct'] / train_class_stats[c]['total'] if train_class_stats[c]['total'] > 0 else 0) for c in [-1, 0, 1]}
        val_class_acc = {c: (val_class_stats[c]['correct'] / val_class_stats[c]['total'] if val_class_stats[c]['total'] > 0 else 0) for c in [-1, 0, 1]}

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        for c in [-1, 0, 1]:
            history['train_class_acc'][c].append(train_class_acc[c])
            history['val_class_acc'][c].append(val_class_acc[c])

        with open(csv_save_path, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc,
                train_class_acc.get(-1, 0), train_class_acc.get(0, 0), train_class_acc.get(1, 0),
                val_class_acc.get(-1, 0), val_class_acc.get(0, 0), val_class_acc.get(1, 0)
            ])

        plot_training_history(history, plot_save_path)

        print(f"\nEpoch {epoch + 1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc*100:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc*100:.2f}%")
        print("训练集类别准确率:", {k: f"{v:.4f}" for k, v in train_class_acc.items()})
        print("验证集类别准确率:", {k: f"{v:.4f}" for k, v in val_class_acc.items()})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
                'train_class_acc': train_class_acc,
                'val_class_acc': val_class_acc
            }, best_model_path)
            print(f"保存最佳模型 (验证损失 {best_val_loss:.4f})")

        if early_stop_patience is not None:
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    stop_training = True
                    print("达到早停条件，提前终止训练")

    print("训练结束！")
    return history, student_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载并冻结多模态教师模型（带文本token）
    teacher_model = CausalInferenceResNetDeep(input_dim=512, width_multiplier=2)
    teacher_checkpoint = torch.load('models/best_model.pth', map_location=device)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.to(device)

    # 学生模型初始化，从头训练（不加载预训练权重）
    student_model = CausalInferenceResNetDeep(input_dim=512, width_multiplier=2)
    student_checkpoint = torch.load('models/noTextToken/best_model.pth', map_location=device)
    student_model.load_state_dict(student_checkpoint['model_state_dict'])
    student_model.train()
    student_model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    train_dataset = CausalDataset('datasets/150k/train150k.jsonl')
    val_dataset = CausalDataset('datasets/val.jsonl')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True,
                              collate_fn=lambda b: collate_fn_fixed_length(b, fixed_length=512))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True,
                            collate_fn=lambda b: collate_fn_fixed_length(b, fixed_length=512))

    save_dir = 'models/finally'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_student_model.pth')
    plot_save_path = os.path.join(save_dir, 'train_history.png')
    csv_save_path = os.path.join(save_dir, 'train_history.csv')

    history, trained_student = train_teacher_student(
        teacher_model, student_model,
        train_loader, val_loader,
        criterion, optimizer,
        best_model_path,
        plot_save_path,
        csv_save_path,
        num_epochs=1000,
        device=device,
        early_stop_patience=30,
        lambda_kd=0.3,
        lambda_align=0.1
    )

    torch.save(trained_student.state_dict(), os.path.join(save_dir, 'final_student_model.pth'))
    print("第三阶段训练完成，模型已保存。")

if __name__ == "__main__":
    main()