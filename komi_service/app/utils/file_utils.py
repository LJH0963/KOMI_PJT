"""
파일 및 비디오 처리 유틸리티 함수
"""

import os
import re
import mimetypes
from typing import Tuple, Optional, BinaryIO, AsyncGenerator
from pathlib import Path

# MIME 타입 등록
mimetypes.add_type("video/mp4", ".mp4")

def check_file_exists(file_path: str) -> bool:
    """파일 존재 여부 확인
    
    Args:
        file_path: 파일 경로
        
    Returns:
        파일 존재 여부
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)

def get_file_size(file_path: str) -> int:
    """파일 크기 반환
    
    Args:
        file_path: 파일 경로
        
    Returns:
        파일 크기 (바이트)
    """
    return os.path.getsize(file_path)

def get_file_mime_type(file_path: str) -> str:
    """파일 MIME 타입 반환
    
    Args:
        file_path: 파일 경로
        
    Returns:
        MIME 타입
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def parse_range_header(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    """Range 헤더 파싱
    
    Args:
        range_header: Range 헤더 값
        file_size: 파일 크기
        
    Returns:
        시작 바이트와 끝 바이트 튜플, 또는 None
    """
    if not range_header:
        return None
    
    range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if not range_match:
        return None
    
    start_byte = int(range_match.group(1))
    end_byte = int(range_match.group(2)) if range_match.group(2) else file_size - 1
    end_byte = min(end_byte, file_size - 1)
    
    # 유효한 범위인지 확인
    if start_byte > end_byte or start_byte >= file_size:
        return None
    
    return start_byte, end_byte

async def stream_file_generator(file_path: str, start_byte: int, chunk_size: int, length: int) -> AsyncGenerator[bytes, None]:
    """파일 스트리밍 생성기
    
    Args:
        file_path: 파일 경로
        start_byte: 시작 바이트
        chunk_size: 청크 크기
        length: 전송할 총 바이트 수
        
    Yields:
        파일 청크
    """
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        bytes_sent = 0
        
        while bytes_sent < length:
            bytes_to_read = min(chunk_size, length - bytes_sent)
            data = f.read(bytes_to_read)
            
            if not data:
                break
                
            bytes_sent += len(data)
            yield data 