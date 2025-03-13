import json

# JSON 파일 불러오기
file_path = "tests/LJH/json_output/pose_data_copy.json"

with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# image_id 추가
for i, item in enumerate(data, start=1):
    item["image_id"] = f"frame_{i:05d}.jpg"

# 수정된 JSON 저장
output_path = "tests/LJH/json_output/json_modified.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"수정된 JSON이 {output_path} 에 저장되었습니다.")
