from fastapi import HTTPException
import yt_dlp  # (pip install yt-dlp)
import os

OUTPUT_FOLDER = "./tests/LJH/audio_output"

# 1. 오디오 추출 함수
def download_audio(video_url: str):
    """유튜브 비디오에서 오디오 추출하여 output_folder에 저장"""

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)       # output_folder 생성

    # 타임스탬프 추가 (파일명 중복 방지)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = "extracted_audio"
    audio_path = os.path.join(OUTPUT_FOLDER, audio_filename)

    # 유튜브 오디오 다운로드 옵션 설정
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_path,                      # 저장될 경로 지정
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])               # 유튜브에서 오디오 다운로드
    except Exception as e:
        # print(e)
        raise HTTPException(status_code=500, detail=f"오디오 추출 실패: {str(e)}")

    return audio_path                               # 다운로드된 파일 경로 반환

download_audio('https://youtu.be/9FRVCN_5ATc?si=BnN4qeiJ1SIWpMrT')