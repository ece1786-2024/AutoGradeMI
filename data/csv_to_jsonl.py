import os
import csv
import json

csv_file_path = os.path.join("data", "dataset", "processed", "clean_data.csv")
jsonl_file_path = os.path.join("data", "dataset", "processed", "clean_data.jsonl")
mini_data_path = os.path.join("data", "dataset", "processed", "mini_data.jsonl")


# Open the CSV file and JSONL file
conversations = []
sys_mess = {"role": "system", "content": "You are a IELTS writing part examiner who is responsible to provide a band score given an essay"}
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)  # Automatically use CSV headers as keys
    for row in reader:
        curr_mess = {"messages": []}
        user_mess = {"role": "user", "content": "Question: " + row["prompt"] + "\n" + "Essay: " + row["essay"]}
        res_mess = {"role": "assistant", "content": row["band"]}
        curr_mess["messages"].append(sys_mess)
        curr_mess["messages"].append(user_mess)
        curr_mess["messages"].append(res_mess)

        conversations.append(curr_mess)

with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
    for conversation in conversations:
        jsonl_file.write(json.dumps(conversation) + '\n')
    
with open(mini_data_path, mode='w', encoding='utf-8') as mini_file:
    count = 0
    for conversation in conversations:
        mini_file.write(json.dumps(conversation) + '\n')
        count += 1
        if count == 20:
            break

