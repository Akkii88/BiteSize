# ByteSize Sage 🎬

ByteSize Sage is an AI-powered viral clip extractor that processes long-form content to find "wisdom bytes". It uses multimodal analysis (Audio, Text, Vision) to identify high-potential segments and automatically generates vertical, captioned clips for social media.

## Features

- **Multimodal Intelligence**: Analyzes videos using:
  - **Audio**: RMS energy and emphasis detection.
  - **Text**: Keyword density and sentiment (Wisdom Score).
  - **Vision**: Face detection and visibility scoring.
  - **Emotion**: Intensity tracking.
- **Viral Scoring**: Custom algorithm to rank segments.
- **Auto-Clipping**: Generates 9:16 vertical clips with yellow/black captioned overlays.
- **Local Processing**: Uses `openai-whisper` locally. No paid APIs required.

## Installation

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You may need `ffmpeg` and `imagemagick` installed on your system for `moviepy` to work fully.*

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

### Docker (Recommended)

1. **Build the image**:
    ```bash
    docker build -t bitesize-sage .
    ```

2. **Run the container**:
    ```bash
    docker run -p 8501:8501 bitesize-sage
    ```
2. **Upload a video** or enter a YouTube URL.
3. **Adjust weights** in the sidebar if needed.
4. **Click "Analyze Video"**: processing may take a few minutes depending on video length and hardware.
5. **View Results**: Explore the dashboard and insights.
6. **Generate Clips**: Go to the "Viral Clips" tab to generating and download individual clips.

## Troubleshooting

- **ImageMagick Error**: If captions fail, ensure ImageMagick is installed and configured in `moviepy`.
- **Performance**: First run will download the Whisper 'base' model (~140MB).
