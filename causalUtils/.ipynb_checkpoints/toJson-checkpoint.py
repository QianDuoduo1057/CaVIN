import json



def toOkVQA(fin_list: list, split: str = '2B_initial0.08_keep0_all0.08'):
    predictions = {}
    
    for data in fin_list:
        question_id = data['question_id']
        
        # 如果是 list，转为字符串或取第一个元素
        if isinstance(question_id, list):
            question_id = str(question_id[0])  # 取第一个元素
        else:
            question_id = str(question_id)
        
        predictions[question_id] = {
            'direct_answer': data['direct_answer']
        }

    with open(f'predictions_{split}.json', 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"已写入 predictions_{split}.json，共 {len(predictions)} 条")