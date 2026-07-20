import json

# 读取原始 JSON 文件
with open("predictions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换格式
result = {}
for item in data:
    qid = str(item["question_id"])
    result[qid] = {
        "direct_answer": item["answer"]
    }

# 写入新的 JSON 文件
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print("转换完成！")