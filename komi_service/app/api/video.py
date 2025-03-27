"""
비디오 스트리밍 및 관리 API 라우터
"""

from fastapi import APIRouter, Header, Response
from fastapi.responses import FileResponse, StreamingResponse
import os

from app.config import DATA_DIR
from app.utils.file_utils import (
    check_file_exists, get_file_size, get_file_mime_type,
    parse_range_header, stream_file_generator
)

router = APIRouter(tags=["video"])

@router.get("/video_info/{video_path:path}")
async def get_video_info(video_path: str):
    """비디오 파일 정보 확인"""
    file_path = os.path.join(DATA_DIR, video_path)
    
    if not check_file_exists(file_path):
        return {
            "exists": False,
            "message": f"파일이 존재하지 않습니다: {file_path}"
        }
    
    try:
        file_size = get_file_size(file_path)
        
        # 파일이 너무 작으면 손상되었거나 비어있을 가능성이 높음
        if file_size < 1000:  # 1KB 미만
            return {
                "exists": True,
                "valid": False,
                "file_size": file_size,
                "message": f"파일이 너무 작습니다 ({file_size} bytes). 손상되었거나 비어있을 수 있습니다."
            }
        
        return {
            "exists": True,
            "valid": True,
            "file_size": file_size,
            "file_path": file_path,
            "content_type": get_file_mime_type(file_path)
        }
    except Exception as e:
        return {
            "exists": True,
            "valid": False,
            "message": f"파일 정보 확인 중 오류 발생: {str(e)}"
        }

@router.get("/video/{video_path:path}")
async def get_video(video_path: str, range: str = Header(None)):
    """비디오 파일 직접 서빙 (스트리밍 지원)"""
    file_path = os.path.join(DATA_DIR, video_path)
    
    if not check_file_exists(file_path):
        return Response(
            content=f"파일을 찾을 수 없습니다: {video_path}",
            status_code=404
        )
    
    # 파일 크기 확인
    file_size = get_file_size(file_path)
    
    # Range 요청이 없으면 전체 파일 반환
    if not range:
        return FileResponse(
            path=file_path,
            media_type=get_file_mime_type(file_path),
            filename=os.path.basename(file_path)
        )
    
    # Range 요청 처리
    range_header = parse_range_header(range, file_size)
    if range_header:
        start_byte, end_byte = range_header
        content_length = end_byte - start_byte + 1
        
        # StreamingResponse로 반환
        response = StreamingResponse(
            stream_file_generator(file_path, start_byte, 1024 * 64, content_length),
            media_type=get_file_mime_type(file_path),
            status_code=206  # Partial Content
        )
        
        # 필요한 헤더 설정
        response.headers["Content-Range"] = f"bytes {start_byte}-{end_byte}/{file_size}"
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Content-Length"] = str(content_length)
        response.headers["Content-Disposition"] = f"inline; filename={os.path.basename(file_path)}"
        
        # CORS 헤더 추가
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = '*'
        
        return response
    
    # Range 형식이 올바르지 않은 경우
    return FileResponse(
        path=file_path,
        media_type=get_file_mime_type(file_path),
        filename=os.path.basename(file_path)
    ) 