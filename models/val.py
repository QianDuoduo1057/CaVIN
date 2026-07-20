import torch, os, json
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, schedule

#============================================================
# 模型加载
# ============================================================
model_name = "InternVL2-26B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ============================================================
# 工具函数：参数量统计
# ============================================================
def count_parameters(model):
    """分模块统计参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "modules": module_params
    }

# ============================================================
# 手动FLOPs计算：逐层精确分析
# ============================================================
def compute_linear_flops(in_features, out_features, batch_size=1, seq_len=1):
    """
    Linear层FLOPs: 每个输出元素需要 in_features 次乘法 + in_features-1 次加法
    近似: 2* batch * seq * in * out
    """
    return 2 * batch_size * seq_len * in_features * out_features

def compute_attention_flops(seq_len, head_dim, num_heads, batch_size=1):
    """
    Self-Attention FLOPs分解:
    - QKV projection: 3 * 2 * B * S * d_model * d_model
    - QK^T: 2 * B * H * S * S * head_dim
    - Softmax: ~5 * B * H * S * S (近似)
    - AV: 2 * B * H * S * S * head_dim
    - Output proj: 2 * B * S * d_model * d_model
    """
    d_model = num_heads * head_dim
    # QKV projections
    qkv_flops = 3 * compute_linear_flops(d_model, d_model, batch_size, seq_len)
    
    # Attention scores: Q @ K^T
    attn_score_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Softmax (近似)
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    
    # Attention output: A @ V
    attn_out_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Output projection
    out_proj_flops = compute_linear_flops(d_model, d_model, batch_size, seq_len)
    
    return {
        "qkv": qkv_flops,
        "attn_score": attn_score_flops,
        "softmax": softmax_flops,
        "attn_out": attn_out_flops,
        "out_proj": out_proj_flops,
        "total": qkv_flops + attn_score_flops + softmax_flops + attn_out_flops + out_proj_flops
    }

def compute_mlp_flops(d_model, intermediate_size, batch_size=1, seq_len=1):
    """
    MLP/FFN FLOPs (SwiGLU: gate_proj, up_proj, down_proj)
    - gate_proj: 2 * B * S * d * intermediate
    - up_proj:2 * B * S * d * intermediate
    - activation: B * S * intermediate
    - element-wise multiply: B * S * intermediate
    - down_proj: 2 * B * S * intermediate * d
    """
    gate_flops = compute_linear_flops(d_model, intermediate_size, batch_size, seq_len)
    up_flops   = compute_linear_flops(d_model, intermediate_size, batch_size, seq_len)
    act_flops  = batch_size * seq_len * intermediate_size
    mul_flops  = batch_size * seq_len * intermediate_size
    down_flops = compute_linear_flops(intermediate_size, d_model, batch_size, seq_len)
    
    return {
        "gate": gate_flops,
        "up": up_flops,
        "activation": act_flops,
        "multiply": mul_flops,
        "down": down_flops,
        "total": gate_flops + up_flops + act_flops + mul_flops + down_flops
    }

def compute_vision_encoder_flops(model, image_size=448, patch_size=14, batch_size=1):
    """
    计算Vision Encoder (InternViT) FLOPs
    InternViT-300M结构:
    - Patch Embedding: Conv2d
    - N x Transformer Block (Attention + MLP)
    """
    num_patches = (image_size // patch_size) ** 2  # 448/14 = 32 → 32*32=1024
    seq_len = num_patches + 1  # +1 for CLS token
    
    flops_detail = {
        "patch_embedding": 0,
        "transformer_blocks": [],
        "total_attention": 0,
        "total_mlp": 0,}
    
    # 尝试获取Vision Encoder配置
    try:
        vision_model = model.vision_model
        vision_config = vision_model.config
        
        hidden_size      = vision_config.hidden_size        # e.g., 1024
        num_heads        = vision_config.num_attention_heads # e.g., 16
        intermediate_size = vision_config.intermediate_size  # e.g., 4096
        num_layers       = vision_config.num_hidden_layers   # e.g., 24
        head_dim         = hidden_size // num_heads
        
        print(f"\n[Vision Encoder Config]")
        print(f"  hidden_size={hidden_size}, num_heads={num_heads}")
        print(f"  intermediate_size={intermediate_size}, num_layers={num_layers}")
        print(f"  seq_len (patches+CLS)={seq_len}")
        # Patch Embedding: Conv2d(3, hidden_size, patch_size, patch_size)
        # FLOPs = 2 * out_h * out_w * C_in * C_out * k_h * k_w
        patch_embed_flops = (
            2 * num_patches * 3 * hidden_size * patch_size * patch_size)
        flops_detail["patch_embedding"] = patch_embed_flops
        
        # Transformer Blocks
        total_attn_flops = 0
        total_mlp_flops  = 0
        
        for layer_idx in range(num_layers):
            attn_flops = compute_attention_flops(seq_len, head_dim, num_heads, batch_size)
            mlp_flops= compute_mlp_flops(hidden_size, intermediate_size, batch_size, seq_len)
            
            layer_total = attn_flops["total"] + mlp_flops["total"]
            total_attn_flops += attn_flops["total"]
            total_mlp_flops  += mlp_flops["total"]
            
            flops_detail["transformer_blocks"].append({
                "layer": layer_idx,
                "attention": attn_flops["total"],
                "mlp": mlp_flops["total"],
                "total": layer_total
            })
        
        flops_detail["total_attention"] = total_attn_flops
        flops_detail["total_mlp"]= total_mlp_flops
        
        total_vision_flops = patch_embed_flops + total_attn_flops + total_mlp_flops
        flops_detail["grand_total"] = total_vision_flops
        
        return total_vision_flops, flops_detail
        
    except AttributeError as e:
        print(f"  [WARN] Cannot access vision_model directly: {e}")
        print("  Using parameter-based estimation...")
        
        # 回退: 基于参数量估算
        try:
            vision_params = sum(p.numel() for p in model.vision_model.parameters())
        except:
            vision_params = 300_000_000  # InternViT-300M
        
        estimated_flops = 2 * vision_params * seq_len
        flops_detail["grand_total"] = estimated_flops
        return estimated_flops, flops_detail

def compute_llm_flops(model, text_seq_len=10, image_token_len=256, batch_size=1):
    """
    计算LLM (InternLM2) FLOPs
    
    总seq_len = text_seq_len + image_token_len (拼接后送入LLM)
    """
    total_seq_len = text_seq_len + image_token_len
    
    flops_detail = {
        "embedding": 0,
        "transformer_blocks": [],
        "lm_head": 0,
    }
    
    try:
        # 获取LLM配置
        llm = model.language_model
        llm_config = llm.config
        
        hidden_size       = llm_config.hidden_size         # e.g., 2048
        num_heads         = llm_config.num_attention_heads  # e.g., 16
        num_kv_heads      = getattr(llm_config, 'num_key_value_heads', num_heads)  # GQA
        intermediate_size = llm_config.intermediate_size   # e.g., 8192
        num_layers        = llm_config.num_hidden_layers   # e.g., 24
        vocab_size        = llm_config.vocab_size           # e.g., 92544
        head_dim          = hidden_size // num_heads
        
        print(f"\n[LLM Config]")
        print(f"  hidden_size={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        print(f"  intermediate_size={intermediate_size}, num_layers={num_layers}")
        print(f"  vocab_size={vocab_size}")
        print(f"  total_seq_len (text+image_tokens)={total_seq_len}")
        
        # Embedding lookup: 无乘法，FLOPs≈0
        flops_detail["embedding"] = 0
        
        # Transformer Blocks (考虑GQA)
        total_block_flops = 0
        
        for layer_idx in range(num_layers):
            # Q projection: hidden → hidden
            q_flops = compute_linear_flops(hidden_size, hidden_size, batch_size, total_seq_len)
            # K, V projection (GQA): hidden → kv_heads * head_dim
            kv_dim = num_kv_heads * head_dim
            k_flops = compute_linear_flops(hidden_size, kv_dim, batch_size, total_seq_len)
            v_flops = compute_linear_flops(hidden_size, kv_dim, batch_size, total_seq_len)
            
            # Attention scores: Q(S,H,d) @ K^T(S,KVH,d) with broadcasting
            #等效: num_heads组，每组 2*B*S*S*head_dim
            attn_score_flops = 2 * batch_size * num_heads * total_seq_len * total_seq_len * head_dim
            softmax_flops    = 5 * batch_size * num_heads * total_seq_len * total_seq_len
            attn_out_flops   = 2 * batch_size * num_heads * total_seq_len * total_seq_len * head_dim
            
            # Output projection
            out_proj_flops = compute_linear_flops(hidden_size, hidden_size, batch_size, total_seq_len)
            
            # MLP (SwiGLU)
            mlp = compute_mlp_flops(hidden_size, intermediate_size, batch_size, total_seq_len)
            
            layer_attn_flops = (q_flops + k_flops + v_flops +
                                attn_score_flops + softmax_flops +
                                attn_out_flops + out_proj_flops)
            layer_total = layer_attn_flops + mlp["total"]
            total_block_flops += layer_total
            
            flops_detail["transformer_blocks"].append({
                "layer": layer_idx,
                "attention": layer_attn_flops,
                "mlp": mlp["total"],
                "total": layer_total
            })
        
        # LM Head: hidden → vocab_size
        lm_head_flops = compute_linear_flops(hidden_size, vocab_size, batch_size, seq_len=1)
        flops_detail["lm_head"] = lm_head_flops
        
        total_llm_flops = total_block_flops + lm_head_flops
        flops_detail["grand_total"] = total_llm_flops
        
        return total_llm_flops, flops_detail
        
    except AttributeError as e:
        print(f"  [WARN] Cannot access language_model config: {e}")
        
        try:
            llm_params = sum(p.numel() for p in model.language_model.parameters())
        except:
            llm_params = 1_800_000_000  # ~1.8B
        
        estimated_flops = 2 * llm_params * total_seq_len
        flops_detail["grand_total"] = estimated_flops
        return estimated_flops, flops_detail

def compute_mlp_projector_flops(model, image_token_len=256, batch_size=1):
    """
    计算MLP Projector FLOPs (Vision → LLM 维度对齐)
    通常是2层线性+激活
    """
    try:
        projector = model.mlp1# InternVL中常见命名
        
        total_flops = 0
        current_input = None
        
        for name, module in projector.named_modules():
            if isinstance(module, nn.Linear):
                in_f= module.in_features
                out_f = module.out_features
                flops = compute_linear_flops(in_f, out_f, batch_size, image_token_len)
                total_flops += flops
                print(f"  Projector Linear({in_f}→{out_f}): {flops/1e9:.4f} GFLOPs")
        
        return total_flops
        
    except AttributeError:
        print("  [WARN] mlp1 not found, using estimation")
        # InternViT hidden(1024) → LLM hidden(2048), 两层
        estimated = 2 * (2* batch_size * image_token_len * 1024 * 2048)
        return estimated

# ============================================================
# Profiler方式计算FLOPs（作为交叉验证）
# ============================================================
def measure_flops_with_profiler(model, tokenizer, device, text="What is in this image?"):
    """使用PyTorch Profiler测量实际运行FLOPs"""
    
    print("\n[Profiler] Warming up...")
    dummy_input = tokenizer(text, return_tensors="pt").input_ids.to(device)
    dummy_image = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16).to(device)
    
    # 预热
    with torch.no_grad():
        try:
            _ = model(input_ids=dummy_input)
        except:
            pass
    
    results = {}
    
    #── Text-only ──────────────────────────────────────────
    print("[Profiler] Measuring text-only FLOPs...")
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                _ = model(input_ids=dummy_input)
        
        text_flops = sum(item.flops for item in prof.key_averages())
        results["text_only"] = text_flops
        print(f"  Text-only: {text_flops/1e9:.2f} GFLOPs")
        # 打印Top-10耗时算子
        print("\n  Top-10 operators (text-only):")
        print(prof.key_averages().table(
            sort_by="flops", row_limit=10
        ))
    except Exception as e:
        print(f"  [WARN] Text-only profiling failed: {e}")
        results["text_only"] = 0
    
    # ── Text + Image ────────────────────────────────────────
    print("[Profiler] Measuring text+image FLOPs...")
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                _ = model(input_ids=dummy_input, pixel_values=dummy_image)
        
        total_flops = sum(item.flops for item in prof.key_averages())
        results["text_image"] = total_flops
        print(f"  Text+Image: {total_flops/1e9:.2f} GFLOPs")
        
        print("\n  Top-10 operators (text+image):")
        print(prof.key_averages().table(
            sort_by="flops", row_limit=10
        ))
        
    except Exception as e:
        print(f"  [WARN] Text+image profiling failed: {e}")
        results["text_image"] = 0
    
    if results.get("text_only") and results.get("text_image"):
        results["image_only"] = results["text_image"] - results["text_only"]
    
    return results

# ============================================================
# 主FLOPs计算入口
# ============================================================
def compute_total_flops(model, tokenizer, device,
                        image_size=448, patch_size=14, batch_size=1):
    """
    综合计算 InternVL2-2B 的FLOPs
    
    InternVL2-2B 架构:InternViT-300M (视觉编码器)
      + MLP Projector
      + InternLM2-1.8B (语言模型)
    """
    print("\n" + "="*60)
    print("  InternVL2-2B FLOPs Analysis")
    print("="*60)
    
    #── 参数量统计 ──────────────────────────────────────────
    param_info = count_parameters(model)
    print(f"\n[Parameters]")
    print(f"  Total:{param_info['total']/1e9:.3f} B")
    print(f"  Trainable: {param_info['trainable']/1e9:.3f} B")
    print("By module:")
    for name, cnt in param_info["modules"].items():
        print(f"    {name:<25}: {cnt/1e6:>8.1f} M  ({cnt/param_info['total']*100:.1f}%)")
    
    # ── 手动FLOPs计算 ────────────────────────────────────────
    text_seq_len    = 10     # 典型提示词长度
    # InternVL2: num_image_tokens = (image_size/patch_size)^2 / pixel_shuffle_factor
    # pixel_shuffle factor=2→ 1024/4=256
    image_token_len = (image_size // patch_size) ** 2 // 4  # = 256
    
    print(f"\n[Computation Setup]")
    print(f"  Image size:       {image_size}x{image_size}")
    print(f"  Patch size:       {patch_size}x{patch_size}")
    print(f"  Image tokens:     {image_token_len}")
    print(f"  Text seq length:  {text_seq_len}")
    print(f"  Batch size:       {batch_size}")
    
    # Vision Encoder
    print("\n" + "-"*40)
    print("[1] Vision Encoder (InternViT)")
    vision_flops, vision_detail = compute_vision_encoder_flops(
        model, image_size, patch_size, batch_size
    )
    print(f"  Patch Embedding: {vision_detail.get('patch_embedding',0)/1e9:.4f} GFLOPs")
    print(f"  Total Attention: {vision_detail.get('total_attention',0)/1e9:.3f} GFLOPs")
    print(f"  Total MLP:       {vision_detail.get('total_mlp',0)/1e9:.3f} GFLOPs")
    print(f"  ► Vision Total:  {vision_flops/1e9:.3f} GFLOPs")
    
    # MLP Projector
    print("\n" + "-"*40)
    print("[2] MLP Projector")
    projector_flops = compute_mlp_projector_flops(model, image_token_len, batch_size)
    print(f"  ► Projector Total: {projector_flops/1e9:.4f} GFLOPs")
    
    # LLM
    print("\n" + "-"*40)
    print("[3] Language Model (InternLM2)")
    llm_flops, llm_detail = compute_llm_flops(
        model, text_seq_len, image_token_len, batch_size
    )
    print(f"  LM Head:         {llm_detail.get('lm_head',0)/1e9:.4f} GFLOPs")
    print(f"► LLM Total:     {llm_flops/1e9:.3f} GFLOPs")
    
    # 汇总
    total_flops = vision_flops + projector_flops + llm_flops
    
    print("\n" + "="*60)
    print("[Summary] Total FLOPs Breakdown")
    print("="*60)
    
    components = {
        "Vision Encoder": vision_flops,
        "MLP Projector":  projector_flops,
        "Language Model": llm_flops
    }
    
    for comp_name, comp_flops in components.items():
        pct = comp_flops / total_flops * 100
        bar = "█" * int(pct / 3)
        print(f"  {comp_name:<18}: {comp_flops/1e9:>8.2f} GFLOPs({pct:5.1f}%)  {bar}")
    
    print(f"\n  {'TOTAL':<18}: {total_flops/1e9:>8.2f} GFLOPs")
    print(f"  {'TOTAL':<18}: {total_flops/1e12:>8.4f} TFLOPs")
    print("="*60)
    
    # ── Profiler交叉验证 ─────────────────────────────────────
    print("\n[Cross-Validation via PyTorch Profiler]")
    profiler_results = measure_flops_with_profiler(model, tokenizer, device)
    
    if profiler_results.get("text_image"):
        prof_total = profiler_results["text_image"]
        ratio = total_flops / prof_total if prof_total > 0 else float('inf')
        print(f"\n  Manual estimate:  {total_flops/1e9:.2f} GFLOPs")
        print(f"  Profiler measure: {prof_total/1e9:.2f} GFLOPs")
        print(f"  Ratio (manual/profiler): {ratio:.3f}")
        print("  Note: Profiler may undercount due to custom CUDA kernels")
    
    return {
        "vision_encoder":vision_flops,
        "mlp_projector":   projector_flops,
        "language_model":  llm_flops,
        "total_gflops":    total_flops / 1e9,
        "total_tflops":    total_flops / 1e12,
        "profiler":        profiler_results
    }

# ============================================================
# 图像处理与推理
# ============================================================
def build_transform(input_size=448):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

def load_image(image_path, input_size=448):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0).to(torch.bfloat16).to(device)

def predict_single_image(image_path, question):
    try:
        clean_question = question.replace("<image>\n", "").replace("<image>", "").strip()
        pixel_values = load_image(image_path)
        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=clean_question,
                generation_config={"max_new_tokens": 50, "do_sample": False}
            )
        return response.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def load_json_or_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)
        try:
            json.loads(first_line)
            second_line = f.readline().strip()
            f.seek(0)
            if second_line and second_line.startswith('{'):
                print("Detected JSONL format")
                return [json.loads(line) for line in f if line.strip()]
            else:
                print("Detected JSON format")
                data = json.loads(f.read())
                return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            data = json.loads(f.read())
            return data if isinstance(data, list) else [data]

def evaluate_dataset(json_file_path, image_base_path, max_samples=None):
    print("Loading dataset...")
    dataset = load_json_or_jsonl(json_file_path)
    if max_samples:
        dataset = dataset[:max_samples]
    print(f"Loaded {len(dataset)} samples")

    results, error_count = [], 0

    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            question_id= item['question_id']
            question= item['question']
            image_relative_path = item['image']
            image_filename     = os.path.basename(image_relative_path)

            # 路径查找优先级
            for candidate in [
                os.path.join(image_base_path, image_filename),
                image_relative_path,
                os.path.join("..", image_relative_path)
            ]:
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            else:
                results.append({"question_id": question_id, "answer": ""})
                error_count += 1
                continue

            prediction = predict_single_image(image_path, question)
            if prediction.startswith("ERROR"):
                results.append({"question_id": question_id, "answer": ""})
                error_count += 1
            else:
                results.append({"question_id": question_id, "answer": prediction})

        except Exception:
            qid = item.get('question_id', f'unknown_{i}')
            results.append({"question_id": qid, "answer": ""})
            error_count += 1

    total = len(results)
    print(f"\nEvaluation: Total={total}, Valid={total-error_count}, Errors={error_count}")

    suffix = '_predictions.json'
    output_file = json_file_path.replace('.jsonl', suffix).replace('.json', suffix)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {output_file}")
    return results

# ============================================================
# 主入口
# ============================================================
def main():
    # Step 1: FLOPs分析
    flops_report = compute_total_flops(
        model, tokenizer, device,
        image_size=448, patch_size=14, batch_size=1
    )
    
    #保存FLOPs报告
    report_path = "flops_report.json"
    with open(report_path, 'w') as f:
        # profiler结果不可JSON序列化，单独处理
        save_report = {k: v for k, v in flops_report.items() if k != "profiler"}
        json.dump(save_report, f, indent=2)
    print(f"\nFLOPs report saved → {report_path}")
    
    # Step 2: 数据集评测
    json_file_path = "../data/coco-caption/val.jsonl"
    image_base_path = "../data/VQAv2/image/val2014"
    evaluate_dataset(json_file_path, image_base_path, max_samples=1000000)

if __name__ == "__main__":
    main()