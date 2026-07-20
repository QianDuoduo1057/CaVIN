import torch, os, json
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity

# 加载模型和tokenizer
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

# ===== FLOPs 测试 =====
print("\n=== Measuring FLOPs ===")
dummy_text = "What is in this image?"
dummy_input = tokenizer(dummy_text, return_tensors="pt").input_ids.to(device)
dummy_image = torch.randn(1, 3, 448, 448, dtype=torch.bfloat16).to(device)

try:
    # 1. 纯文本
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            _ = model(input_ids=dummy_input)
    text_flops = sum([item.flops for item in prof.key_averages()])
    print(f"Text-only FLOPs: {text_flops/1e9:.2f} GFLOPs")

    # 2. 文本+图像
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            _ = model(input_ids=dummy_input, pixel_values=dummy_image)
    total_flops = sum([item.flops for item in prof.key_averages()])
    print(f"Text+Image FLOPs: {total_flops/1e9:.2f} GFLOPs")
    print(f"Image encoding FLOPs: {(total_flops - text_flops)/1e9:.2f} GFLOPs")
    print(f"Total FLOPs: {total_flops/1e12:.4f} TFLOPs")

except Exception as e:
    print(f"Profiler failed: {e}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params/1e9:.2f} B")
    seq_len = dummy_input.shape[1]
    estimated_flops = 2 * total_params * seq_len
    print(f"Estimated FLOPs (seq_len={seq_len}): {estimated_flops/1e12:.2f} TFLOPs")

print("=" * 50)

# 图像预处理
def build_transform(input_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform

def load_image(image_path, input_size=448):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(device)
    return pixel_values

def predict_single_image(image_path, question, choices):
    try:
        choices_str = str(choices)
        formatted_question = f"{question}\nChoices: {choices_str}\nSelect ONE answer from the choices above. Reply with ONLY the exact option string, no explanation, no punctuation, nothing else."
        
        pixel_values = load_image(image_path)
        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=formatted_question,
                generation_config={"max_new_tokens": 50, "do_sample": False}
            )
        return response.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def load_json_or_jsonl(file_path):
    """
    通用加载函数：自动检测并加载 JSON 或 JSONL 格式
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)  # 重置文件指针
        
        # 尝试判断格式
        try:
            # 如果第一行就是完整的 JSON 对象，可能是 JSONL
            json.loads(first_line)
            # 检查第二行
            second_line = f.readline().strip()
            f.seek(0)
            
            if second_line and second_line.startswith('{'):
                # 多行且每行都是 JSON 对象 -> JSONL
                print("Detected JSONL format")
                return [json.loads(line) for line in f if line.strip()]
            else:
                # 单行 JSON 对象 -> 可能是格式化的 JSON 数组
                print("Detected JSON format")
                content = f.read()
                data = json.loads(content)
                return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # 第一行不是完整 JSON -> 标准 JSON 格式
            print("Detected JSON format")
            content = f.read()
            data = json.loads(content)
            return data if isinstance(data, list) else [data]

def evaluate_dataset(json_file_path, image_folder_path, max_samples=None):
    print("Loading dataset...")
    dataset = load_json_or_jsonl(json_file_path)
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    print(f"Loaded {len(dataset)} samples")
    
    predictions = {}
    correct_count = 0
    
    print("Starting evaluation...")
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            image_id = item['image_id']
            question_id = item['question_id']
            question = item['question']
            correct_choice_idx = item['correct_choice_idx']
            choices = item['choices']
            
            image_path = os.path.join(image_folder_path, f"{image_id:012d}.jpg")
            
            if not os.path.exists(image_path):
                predictions[question_id] = {"direct_answer": "IMAGE_NOT_FOUND"}
                continue
            
            prediction = predict_single_image(image_path, question, choices)
            predictions[question_id] = {"direct_answer": prediction}
            
            correct_answer = choices[correct_choice_idx].lower().strip()
            pred_lower = prediction.lower().strip()
            if pred_lower == correct_answer or correct_answer in pred_lower:
                correct_count += 1
            
        except Exception as e:
            predictions[item['question_id']] = {"direct_answer": f"ERROR: {str(e)}"}
    
    total = len(predictions)
    accuracy = correct_count / total if total > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"Total: {total}, Correct: {correct_count}, Accuracy: {accuracy:.4f}")
    
    # 智能输出文件名
    if json_file_path.endswith('.jsonl'):
        output_file = json_file_path.replace('.jsonl', '_predictions.json')
    else:
        output_file = json_file_path.replace('.json', '_predictions.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    
    print(f"Predictions saved to: {output_file}")
    return predictions, accuracy

def main():
    json_file_path = "../data/coco-caption/val.jsonl"
    image_folder_path = "../data/VQAv2/image/val2014"
    
    results, accuracy = evaluate_dataset(json_file_path, image_folder_path, max_samples=1000000)

if __name__ == "__main__":
    main()