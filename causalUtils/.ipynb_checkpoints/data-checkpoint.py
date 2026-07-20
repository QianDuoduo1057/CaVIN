import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split

# 1. 数据集类定义
# class CausalDataset(Dataset):
#     def __init__(self, data_pairs, labels):
#         """
#         数据集类
        
#         参数:
#         data_pairs: 包含(x, y)对的列表，每个元素是(x, y)元组
#         labels: 对应的标签，可以是因果关系方向（0/1）或相似度分数
#         """
#         self.data_pairs = data_pairs
#         self.labels = labels
        
#     def __len__(self):
#         return len(self.data_pairs)
    
#     def __getitem__(self, idx):
#         data = read_jsonl_file('datasets/causalNetDataset6.jsonl', idx+1, idx+1)
#         x, y = data[0]['reason'], data[0]['result']
#         label = data[0]['label']
#         x = [i if i >= 2 else 0.1 for i in x] 
#         y = [i if i >= 2 else 0.1 for i in y]
#         # import pdb; pdb.set_trace()
#         # 转换为tensor
#         x_tensor = torch.FloatTensor(x)
#         y_tensor = torch.FloatTensor(y)
#         label_tensor = torch.FloatTensor([label])

#         return x_tensor, y_tensor, label_tensor
from concurrent.futures import ThreadPoolExecutor

class CausalDataset(Dataset):
    """优化版数据集类，一次性加载所有数据到内存"""
    def __init__(self, file_path, sample_indices=None):
        """
        参数:
        file_path: JSONL文件路径
        sample_indices: 使用的样本索引列表，None表示使用全部
        """
        self.data = []
        
        # 一次性读取所有数据到内存
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 并行解析JSON（对于6万条数据很有效）
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(json.loads, lines))
        
        # 过滤和预处理数据
        for i, item in enumerate(results):
            if sample_indices is None or i in sample_indices:
                x = torch.FloatTensor([val if val >= 2 else 0.1 for val in item['reason']])
                y = torch.FloatTensor([val if val >= 2 else 0.1 for val in item['result']])
                label = torch.FloatTensor([item['label']])
                
                self.data.append({
                    'x': x,
                    'y': y, 
                    'label': label,
                    'id': item.get('id', i)
                })
        
        print(f"数据集加载完成，共 {len(self.data)} 条样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['x'], item['y'], item['label']

# 2. JSONL文件读取函数
def read_jsonl_file(file_path, start_line=None, end_line=None):
    """
    读取JSONL文件并返回数据列表
    
    Args:
        file_path (str): JSONL文件路径
        start_line (int, optional): 起始行号（从1开始）。如果为None，从第一行开始
        end_line (int, optional): 结束行号（包含）。如果为None，读取到最后一行
        
    Returns:
        list: 包含指定范围内所有JSON对象的列表
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                # 跳过起始行之前的行
                if start_line and line_num < start_line:
                    continue
                    
                # 如果超过结束行，停止读取
                if end_line and line_num > end_line:
                    break
                
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []
    
    return data

# 3. 文件长度获取函数
def get_jsonl_length_basic(file_path):
    """
    获取JSONL文件的总行数（包括空行）
    
    Args:
        file_path (str): JSONL文件路径
        
    Returns:
        int: 文件总行数，如果文件不存在返回-1
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return -1
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return -1