from setuptools import setup, find_packages

setup(
    name="bitesize-sage",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "moviepy",
        "openai-whisper",
        "librosa",
        "opencv-python",
        "numpy",
        "scipy",
        "nltk",
        "yt-dlp",
        "plotly",
        "wordcloud",
        "Pillow",
        "torch",
        "torchaudio",
        "imageio_ffmpeg"
    ],
)
