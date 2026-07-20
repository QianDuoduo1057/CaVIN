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
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
        with torch.no_grad():
            _ = model(input_ids=dummy_input)
    text_flops = sum([item.flops for item in prof.key_averages()])
    print(f"Text-only FLOPs: {text_flops/1e9:.2f} GFLOPs")

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
                content = f.read()
                data = json.loads(content)
                return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            print("Detected JSON format")
            content = f.read()
            data = json.loads(content)
            return data if isinstance(data, list) else [data]

def evaluate_dataset(json_file_path, image_base_path, max_samples=None):
    print("Loading dataset...")
    dataset = load_json_or_jsonl(json_file_path)

    if max_samples:
        dataset = dataset[:max_samples]

    print(f"Loaded {len(dataset)} samples")

    results = []
    error_count = 0

    print("Starting evaluation...")
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            question = item['question']
            image_relative_path = item['image']
            image_filename = os.path.basename(image_relative_path)
            image_path = os.path.join(image_base_path, image_filename)

            if not os.path.exists(image_path):
                image_path = image_relative_path
            if not os.path.exists(image_path):
                image_path = os.path.join("..", image_relative_path)

            if not os.path.exists(image_path):
                results.append({"image": image_filename, "answer": ""})
                error_count += 1
                continue

            prediction = predict_single_image(image_path, question)

            if prediction.startswith("ERROR"):
                results.append({"image": image_filename, "answer": ""})
                error_count += 1
            else:
                results.append({"image": image_filename, "answer": prediction})

        except Exception as e:
            image_filename = os.path.basename(item.get('image', f'unknown_{i}'))
            results.append({"image": image_filename, "answer": ""})
            error_count += 1

    total = len(results)
    valid_count = total - error_count

    print(f"\nEvaluation Results:")
    print(f"Total: {total}")
    print(f"Valid: {valid_count}")
    print(f"Errors: {error_count}")

    if json_file_path.endswith('.jsonl'):
        output_file = json_file_path.replace('.jsonl', '_predictions.json')
    else:
        output_file = json_file_path.replace('.json', '_predictions.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to: {output_file}")
    return results

def main():
    json_file_path = "../data/coco-caption/val.jsonl"
    image_base_path = "../data/VQAv2/image/val2014"

    results = evaluate_dataset(json_file_path, image_base_path, max_samples=1000000)

if __name__ == "__main__":
    main()