import numpy as np
from math import sqrt

def generate_custom_anchors(values, anchors):
    """
    在平方矩阵上滑动多个anchor窗口，返回每个anchor在每个位置的矩阵
    
    参数:
    values: 长度为k的列表，k必须是平方数
    anchors: anchor列表，每个anchor为(height, width)元组
    
    返回:
    字典，键为anchor字符串，值为对应的矩阵列表
    """
    # 检查输入列表长度是否为平方数
    k = len(values)
    n = int(sqrt(k))
    if n * n != k:
        raise ValueError(f"列表长度{k}不是平方数")
    
    # 将列表转换为n*n矩阵
    matrix = np.array(values).reshape(n, n)
    
    # results = {}
    anchor_results = []
    for anchor_idx, (h, w) in enumerate(anchors):
        # 检查anchor尺寸是否合法
        if h > n or w > n:
            raise ValueError(f"anchor尺寸({h},{w})超出矩阵尺寸{n}x{n}")
        
        # 在矩阵上滑动anchor窗口
        for i in range(n - h + 1):
            for j in range(n - w + 1):
                # 创建与原始矩阵相同大小的全0.1矩阵
                result_matrix = np.full((n, n), 0.1)
                
                # 提取anchor区域的值
                anchor_region = matrix[i:i+h, j:j+w]
                
                # 将anchor区域的值放入结果矩阵
                result_matrix[i:i+h, j:j+w] = anchor_region

                anchor_results.append(result_matrix.flatten().tolist())
        
        # results[f"anchor_{anchor_idx+1}_{h}x{w}"] = anchor_results
    
    return anchor_results 

# 返回需要恢复的token蒙版
def anchors_to_token_mask(values, anchor_sizes, binary_mask):
    """
    根据 anchor 级别的 binary_mask，生成 token 级别的恢复蒙版。

    Args:
        values:       原始 token 列表，长度 K（K 必须是平方数）
        anchor_sizes: anchor 尺寸列表，如 [(8,8), (5,9), (9,5)]
        binary_mask:  shape (B,) 的 np 数组，1=该 anchor 需要恢复，0=不需要
                      B = 所有 anchor 在所有滑动位置的总数

    Returns:
        token_mask:   shape (K,) 的 np 数组，1=该 token 需要恢复，0=不需要
    """
    k = len(values)
    n = int(k ** 0.5)
    assert n * n == k

    token_mask = np.zeros(k, dtype=np.int32)

    idx = 0  # 遍历 binary_mask 的指针，和 generate_custom_anchors 的生成顺序严格对齐
    for (h, w) in anchor_sizes:
        for i in range(n - h + 1):
            for j in range(n - w + 1):
                if binary_mask[idx] == 1:
                    # 该 anchor 需要恢复 → 把它覆盖的所有 token 位置标记为 1
                    for di in range(h):
                        for dj in range(w):
                            token_mask[(i + di) * n + (j + dj)] = 1
                idx += 1

    return token_mask
if __name__ == "__main__":
    # 示例：16个元素的列表（4x4矩阵）
    test_values = list(range(1, 257))
    
    # 设置anchor尺寸
    test_anchors = [(5, 5), (5, 6), (6, 5)]
    import pdb; pdb.set_trace()
    # 调用函数
    result = generate_custom_anchors(test_values, test_anchors)
    
    # 打印结果
    for anchor_name, matrices in result.items():
        print(f"\n{anchor_name} 的结果 ({len(matrices)} 个位置):")
        for idx, mat in enumerate(matrices[:3]):  # 只显示前3个位置
            print(f"位置 {idx + 1}:")
            print(np.round(mat, 2))
            print()



# def generate_custom_anchors(input_list: List, anchor_sizes: List[Tuple[int, int]] = None):
#     """
#     为输入列表生成自定义大小的锚框
    
#     参数:
#         input_list: 长度为 n 的列表，n 必须是完全平方数
#         anchor_sizes: 锚框大小列表，每个元素为(高度, 宽度)的元组
#                      默认值为[(5,5), (3,7), (7,3)]
        
#     返回:
#         包含所有锚框列表的列表，每个锚框列表长度与输入相同
#     """
#     n = len(input_list)
#     grid_size = int(math.isqrt(n))
    
#     # 验证输入是否是有效的完全平方数
#     if grid_size * grid_size != n:
#         raise ValueError(f"输入列表长度 {n} 不是完全平方数")
    
#     # 设置默认锚框大小
#     if anchor_sizes is None:
#         anchor_sizes = [(5, 5), (3, 7), (7, 3)]
    
#     # 验证锚框大小
#     for h, w in anchor_sizes:
#         if h <= 0 or w <= 0:
#             raise ValueError(f"锚框大小必须为正数: ({h}, {w})")
    
#     anchors = []

#     # 将一维列表转换为二维网格
#     grid_2d = [[input_list[i*grid_size + j] for j in range(grid_size)] for i in range(grid_size)]

#     # 为每个位置生成指定数量的锚框
#     for i in range(grid_size):
#         for j in range(grid_size):
#             # 为当前patch生成多个不同大小的锚框
#             for anchor_h, anchor_w in anchor_sizes:
#                 # 创建新的锚框列表，初始化为-99999
#                 anchor_list = [0.1] * n
                
#                 # 计算锚框的高度和宽度的一半
#                 half_h = anchor_h // 2
#                 half_w = anchor_w // 2
                
#                 # 计算锚框的边界（考虑网格边缘）
#                 start_row = max(0, i - half_h)
#                 end_row = min(grid_size, i + half_h + 1) if anchor_h % 2 == 1 else min(grid_size, i + half_h)
#                 start_col = max(0, j - half_w)
#                 end_col = min(grid_size, j + half_w + 1) if anchor_w % 2 == 1 else min(grid_size, j + half_w)
                
#                 # 将锚框内的patch复制到新列表中
#                 for r in range(start_row, end_row):
#                     for c in range(start_col, end_col):
#                         # 计算一维索引
#                         index = r * grid_size + c
#                         anchor_list[index] = grid_2d[r][c]
#                 anchors.append(anchor_list)
                
#     return anchors

# # 测试代码
# if __name__ == "__main__":
#     # 示例1：创建一个16×16（256个元素）的测试列表
#     n = 256
#     test_list = [i for i in range(n)]
    
#     # 自定义锚框大小
#     custom_sizes = [(5, 5), (3, 7), (7, 3)]
#     print("示例1: 使用默认锚框大小 (5,5), (3,7), (7,3)")
    
#     # 生成锚框
#     anchor_boxes = generate_custom_anchors(test_list, custom_sizes)
    
#     print(f"输入列表长度: {len(test_list)}")
#     print(f"生成的锚框数量: {len(anchor_boxes)} (期望: {16 * 16 * 3} = 768)")
#     print(f"每个锚框长度: {len(anchor_boxes[0])}")
    


    
    
    # 获取位置(2,2)的第一个锚框(5×5)信息
    # pos_idx = 2 * grid_size + 2
    # anchor_positions = get_anchor_info(anchor_boxes, grid_size, pos_idx, 0, custom_sizes)
    # if anchor_positions:
    #     print(f"位置(2,2)的5×5锚框包含{len(anchor_positions)}个patch:")
    #     print(f"位置范围: 行{min(p[0] for p in anchor_positions)}-{max(p[0] for p in anchor_positions)}")
    #     print(f"位置范围: 列{min(p[1] for p in anchor_positions)}-{max(p[1] for p in anchor_positions)}")
# if __name__ == "__main__":
#     # 示例：16个元素的列表（4x4矩阵）
#     test_values = list(range(1, 17))
    
#     # 设置anchor尺寸
#     test_anchors = [(2, 2), (2, 3), (3, 2)]
    
#     # 调用函数
#     result = anchor_sliding_window(test_values, test_anchors)
    
#     # 打印结果
#     for anchor_name, matrices in result.items():
#         print(f"\n{anchor_name} 的结果 ({len(matrices)} 个位置):")
#         for idx, mat in enumerate(matrices[:3]):  # 只显示前3个位置
#             print(f"位置 {idx + 1}:")
#             print(np.round(mat, 2))
#             print()
