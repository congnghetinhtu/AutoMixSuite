"""
File handling utilities
"""

import hashlib
from pathlib import Path
from typing import List, Set


def get_file_hash(file_path: Path) -> str:
    """
    Generate a hash of the file to detect changes
    Uses file size, modification time, and first 8KB for speed
    """
    stat = file_path.stat()
    hash_input = f"{stat.st_size}_{stat.st_mtime_ns}".encode()
    
    # Add first 8KB of file content for better detection
    try:
        with open(file_path, 'rb') as f:
            hash_input += f.read(8192)
    except:
        pass
    
    return hashlib.md5(hash_input).hexdigest()


def get_audio_files(folder: Path, supported_formats: Set[str]) -> List[Path]:
    """Get all supported audio files from a folder"""
    audio_files = []
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in supported_formats:
            audio_files.append(file_path)
    
    # Sort files alphabetically
    audio_files.sort(key=lambda x: x.name.lower())
    return audio_files
