from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from bitesize_core.pipeline import VideoPipeline
from bitesize_core.analyzer import TextAnalyzer, AudioAnalyzer, VisionAnalyzer

@patch('bitesize_core.pipeline.yt_dlp.YoutubeDL')
def test_download_youtube_video(mock_ytdl):
    # Setup mock
    instance = mock_ytdl.return_value
    instance.__enter__.return_value = instance
    inst

    pipeline = VideoPipeline()
    filename = pipeline.download_youtube_video("http://youtube.com/watch?v=123")
    
    assert filename == 'downloads/test.mp4'
    instance.extract_info.assert_called_once()

def test_text_analyzer_sentiment():
    analyzer = TextAnalyzer()
    # "life" is a keyword in our simple heuristic
    score = analyzer.analyze_segment("This is the secret of life")
    assert score > 0.0

    score_empty = analyzer.analyze_segment("")
    assert score_empty == 0.0

@patch('bitesize_core.analyzer.librosa.load')
@patch('bitesize_core.analyzer.librosa.feature.rms')
def test_audio_analyzer(mock_rms, mock_load):
    # Mock audio data
    mock_load.return_value = (np.zeros(100), 22050)
    mock_rms.return_value = np.array([[0.1, 0.5, 0.1]]) # reduced shape
    
    analyzer = AudioAnalyzer()
    times, energy = analyzer.analyze("dummy.wav")
    
    assert len(energy) > 0
    assert len(times) == len(energy)

def test_vision_analyzer_init():
    # Just test that it initializes without error (requires opencv)
    try:
        analyzer = VisionAnalyzer()
        assert analyzer.face_cascade is not None
    except Exception as e:
        pytest.fail(f"VisionAnalyzer failed to init: {e}")
