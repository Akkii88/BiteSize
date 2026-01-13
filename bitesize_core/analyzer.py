import cv2
import numpy as np
import librosa
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import math
import logging
from typing import List, Tuple, Dict, Any, Set

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def analyze(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a time-series of normalized energy scores.
        """
        logger.info(f"Analyzing audio: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
        # RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        
        # Normalize 0-1
        if len(rms) > 0:
            rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
        else:
            rms_norm = rms
            
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        return times, rms_norm

class TextAnalyzer:
    def __init__(self):
        # In a real scenario, we might download NLTK data here or assume it's present
        pass

    def analyze_segment(self, text_segment: str) -> float:
        """
        Returns a 'wisdom' score based on keyword density and sentiment.
        For now, we use a heuristic based on 'insightful' keywords.
        """
        keywords = {
            "life", "secret", "truth", "money", "success", "fail", "learn", 
            "world", "mind", "power", "change", "growth", "believe", "start", 
            "stop", "why", "how", "important", "remember", "key", "future"
        }
        
        tokens = word_tokenize(text_segment.lower())
        word_count = len(tokens)
        if word_count == 0:
            return 0.0
            
        keyword_hits = sum(1 for w in tokens if w in keywords)
        density = keyword_hits / word_count
        
        # Normalize: expect at best 1 keyword every 5 words -> density 0.2 is score 1.0
        score = min(density * 5.0, 1.0)
        return score

    def extract_keywords(self, full_text: str, top_n: int = 20) -> List[Tuple[Any, int]]:
        tokens = word_tokenize(full_text.lower())
        # Filter stopwords would precise this, simple filter for now
        filtered = [w for w in tokens if len(w) > 4] 
        return Counter(filtered).most_common(top_n)

class VisionAnalyzer:
    def __init__(self):
        # Load Haar Cascade
        haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        logger.info(f"Loading face cascade from {haarcascade_path}")
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

    def detect_faces(self, frame) -> Any:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def analyze_video(self, video_path: str, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyzes video for face visibility.
        sample_rate: Analyze 1 frame every 'sample_rate' seconds.
        """
        logger.info(f"Analyzing video vision: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_step = int(fps * sample_rate)
        vis_scores = []
        timestamps = []
        
        for i in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            faces = self.detect_faces(frame)
            # Score: 1.0 if at least one face, 0.0 otherwise. 
            # Could be improved to measure face size relative to frame.
            score = 1.0 if len(faces) > 0 else 0.0
            vis_scores.append(score)
            timestamps.append(i / fps)
            
        cap.release()
        return np.array(timestamps), np.array(vis_scores)

class ViralScorer:
    def __init__(self, weights: Dict[str, float]):
        self.w_audio = weights.get('audio', 0.25)
        self.w_wisdom = weights.get('wisdom', 0.35)
        self.w_face = weights.get('face', 0.20)
        self.w_emotion = weights.get('emotion', 0.20)

    def score_segments(self, transcript_segments: List[Dict[str, Any]], audio_times: np.ndarray, audio_energy: np.ndarray, vis_times: np.ndarray, vis_scores: np.ndarray) -> List[Dict[str, Any]]:
        """
        Scores each transcript segment.
        """
        logger.info("Scoring segments...")
        scored_segments = []
        text_analyzer = TextAnalyzer()
        
        for seg in transcript_segments:
            start = seg['start']
            end = seg['end']
            text = seg['text']
            
            # Text Score
            wisdom_score = text_analyzer.analyze_segment(text)
            
            # Audio Score (Avg energy in this window)
            # Find indices in audio_times within [start, end]
            a_indices = np.where((audio_times >= start) & (audio_times <= end))
            if len(a_indices[0]) > 0:
                audio_score = np.mean(audio_energy[a_indices])
            else:
                audio_score = 0.0
                
            # Vision Score (Avg face visibility in this window)
            v_indices = np.where((vis_times >= start) & (vis_times <= end))
            if len(v_indices[0]) > 0:
                face_score = np.mean(vis_scores[v_indices])
            else:
                face_score = 0.0
                
            # Emotion Score (Simplified: use energy variance or peaks as proxy for 'intensity')
            if len(a_indices[0]) > 0:
                emotion_score = np.max(audio_energy[a_indices]) # Peak energy = high emotion
            else:
                emotion_score = 0.0
                
            final_score = (
                self.w_audio * audio_score +
                self.w_wisdom * wisdom_score +
                self.w_face * face_score +
                self.w_emotion * emotion_score
            )
            
            scored_segments.append({
                "start": start,
                "end": end,
                "text": text,
                "score": final_score,
                "components": {
                    "audio": audio_score,
                    "wisdom": wisdom_score, 
                    "face": face_score,
                    "emotion": emotion_score
                }
            })
            
        return scored_segments
