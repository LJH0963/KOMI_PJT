import json

file_path = 'C:\WANTED\LLM\KOMI_PJT\pose_data.json'

# JSON 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

print(len(data))
