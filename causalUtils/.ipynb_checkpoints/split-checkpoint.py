import json

def extract_first_n_lines(input_file, output_file, n):
    """
    从JSONL文件中提取前n行并保存到新文件
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出文件路径
        n: 要提取的行数
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            count = 0
            for line in f_in:
                if count >= n:
                    break
                    
                # 写入当前行
                f_out.write(line)
                count += 1
            
        print(f"已成功提取前 {min(count, n)} 行到 {output_file}")
        print(f"原文件 {input_file} 保持不变")
        
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
    except Exception as e:
        print(f"处理文件时出错：{e}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_filename = "datasets/allTrue_id0-193659&253660-341077.jsonl"    # 原始文件
    output_filename = "datasets/train10k.jsonl"  # 输出文件
    n_lines = 10000                    # 提取前100行
    
    # 执行提取
    extract_first_n_lines(input_filename, output_filename, n_lines)