import requests

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
filename = "sam_vit_h_4b8939.pth"

response = requests.get(url, stream=True)
with open(filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("파일 다운로드 완료:", filename)
