import os
import logging
import streamlit as st
import yt_dlp
import moviepy.editor as mp
import whisper
import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import imageio_ffmpeg
from typing import Optional, List, Dict, Any, Tuple

# Ensure Whisper can find ffmpeg
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

logger = logging.getLogger(__name__)

@st.cache_resource
def load_whisper_model():
    logger.info("Loading Whisper model...")
    return whisper.load_model("base")

class VideoPipeline:
    def __init__(self):
        pass

    def download_youtube_video(self, url: str, output_path: str = "downloads") -> str:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'ffmpeg_location': imageio_ffmpeg.get_ffmpeg_exe()
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading video from {url}")
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            logger.info(f"Downloaded video to {filename}")
            return filename

    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extracts audio from video and saves as temporary wav file."""
        try:
            logger.info(f"Extracting audio from {video_path}")
            video = VideoFileClip(video_path)
            audio_path = video_path.replace(".mp4", ".wav").replace(".mov", ".wav").replace(".avi", ".wav")
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        model = load_whisper_model()
        # Load with librosa to bypass whisper's ffmpeg dependency
        # Whisper expects 16kHz audio
        try:
            logger.info(f"Loading audio with librosa: {audio_path}")
            audio_data, _ = librosa.load(audio_path, sr=16000)
            logger.info("Starting transcription...")
            result = model.transcribe(audio_data, word_timestamps=True)
            logger.info("Transcription complete.")
            return result
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise e

    def get_audio_features(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates RMS energy to find loud/emphasized parts."""
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms)
        return times, rms

    def process_video_upload(self, uploaded_file) -> str:
        """Saves uploaded file to temp disk."""
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file to {file_path}")
        return file_path

class ClipGenerator:
    def __init__(self):
        pass

    def generate_clip(self, video_path: str, start_time: float, end_time: float, transcript_words: List[Dict[str, Any]], output_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generates a 9:16 clip with captions.
        """
        try:
            logger.info(f"Generating clip from {start_time} to {end_time}")
            # Load video
            clip = VideoFileClip(video_path).subclip(start_time, end_time)
            
            # Smart Crop 9:16
            w, h = clip.size
            target_ratio = 9/16
            
            # Simple Center Crop to 9:16
            new_w = h * target_ratio
            if new_w > w:
                new_w = w
                new_h = w / target_ratio
            else:
                new_h = h
                
            clip_cropped = clip.crop(x1=w/2 - new_w/2, y1=h/2 - new_h/2, width=new_w, height=new_h)
            clip_resized = clip_cropped.resize(height=1920) # Width will match ~1080
            
            # Add Captions
            txt_clips = []
            for word in transcript_words:
                w_start = word['start']
                w_end = word['end']
                
                # Check if word is in this clip
                if w_end < start_time or w_start > end_time:
                    continue
                    
                # Adjust to clip time
                rel_start = max(0, w_start - start_time)
                rel_end = min(end_time - start_time, w_end - start_time)
                duration = rel_end - rel_start
                
                if duration <= 0:
                    continue

                # Create TextClip
                try:
                    txt = (mp.TextClip(word['word'], fontsize=70, color='yellow', font='Arial-Bold', stroke_color='black', stroke_width=2)
                           .set_position(('center', 'bottom'))
                           .set_start(rel_start)
                           .set_duration(duration))
                    txt_clips.append(txt)
                except Exception as e:
                    logger.warning(f"TextClip error (ImageMagick missing?): {e}")
                    pass

            # Export
            logger.info(f"Exporting clip to {output_path}...")
            final_clip = mp.CompositeVideoClip([clip_resized] + txt_clips)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
            
            # Generate thumbnail
            thumbnail_path = output_path.replace(".mp4", ".png")
            final_clip.save_frame(thumbnail_path, t=0)
            
            return output_path, thumbnail_path
            
        except Exception as e:
            logger.error(f"Error generating clip: {e}")
            return None, None
