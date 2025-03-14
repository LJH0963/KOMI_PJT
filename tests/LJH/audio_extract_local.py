import subprocess

def extract_audio(video_path, output_audio_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        output_audio_path
    ]
    subprocess.run(command, check=True)

# 사용 예시
extract_audio("tests/KHS/video_data/mediun_video_640_480.mp4", "tests/LJH/audio_output/output_audio.mp3")
