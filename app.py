import streamlit as st
import os
import nltk
import logging
import sys
from bitesize_core.pipeline import VideoPipeline, ClipGenerator
from bitesize_core.analyzer import AudioAnalyzer, TextAnalyzer, VisionAnalyzer, ViralScorer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Set page config
# Set page config
st.set_page_config(page_title="ByteSize Sage", layout="wide", page_icon="🎬")

def main():
    # Custom CSS - Premium UI/UX
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Gradient Background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* Sidebar Glassmorphism */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(0,0,0,0.05);
        }

        /* Main Header "Hero" */
        .main-header {
            background: linear-gradient(90deg, #F63366 0%, #FF8E53 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        
        .sub-header {
            font-size: 1.2rem;
            color: #555;
            text-align: center;
            margin-bottom: 3rem;
            font-weight: 400;
        }

        /* Card Styling */
        .card {
            background: rgba(255, 255, 255, 0.85);
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1rem;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }

        /* Custom Button */
        .stButton>button {
            background: linear-gradient(90deg, #F63366 0%, #FF8E53 100%);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            border-radius: 50px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(246, 51, 102, 0.3);
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(246, 51, 102, 0.5);
            color: white !important;
        }

        /* Metrics Card */
        .metric-card {
            text-align: center;
            padding: 1rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1E1E1E;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown('<div class="main-header">ByteSize Sage</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Transform Long Videos into Viral Wisdom Bytes 🚀</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/clouds/200/000000/video-editor.png", width=120)
    st.sidebar.markdown("### ⚙️ Control Panel")
    
    st.sidebar.header("Input Source")
    upload_option = st.sidebar.radio("Choose Source", ["Video File", "YouTube URL"])
    
    uploaded_file = None
    youtube_url = None
    
    if upload_option == "Video File":
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    else:
        youtube_url = st.sidebar.text_input("YouTube URL")

    st.sidebar.header("Processing Weights")
    w_audio = st.sidebar.slider("Audio Emphasis", 0.0, 1.0, 0.25)
    w_wisdom = st.sidebar.slider("Wisdom Density", 0.0, 1.0, 0.35)
    w_face = st.sidebar.slider("Face Visibility", 0.0, 1.0, 0.20)
    w_emotion = st.sidebar.slider("Emotional Intensity", 0.0, 1.0, 0.20)
    
    if st.sidebar.button("Analyze Video"):
        st.session_state.processing_started = True

    # Main Area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload & Preview", 
        "📊 Analysis Dashboard", 
        "🧠 Multimodal Insights", 
        "📱 Viral Clips", 
        "💾 Export & Results"
    ])

    with tab1:
        st.header("Upload & Preview")
        if uploaded_file is not None:
            st.video(uploaded_file)
            st.info(f"Filename: {uploaded_file.name} | Size: {uploaded_file.size / 1024 / 1024:.2f} MB")
        elif youtube_url:
            st.info(f"Ready to process URL: {youtube_url}")
        else:
            st.write("Please upload a video or enter a YouTube URL in the sidebar.")

    with tab2:
        st.header("Analysis Dashboard")
        
        if st.session_state.get("processing_started") and (uploaded_file or youtube_url):
            # Processing Logic
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. Setup & Load
                status_text.text("Initializing pipeline...")
                pipeline = VideoPipeline()
                
                # Handle File
                if uploaded_file:
                    video_path = pipeline.process_video_upload(uploaded_file)
                else:
                    status_text.text("Downloading from YouTube...")
                    video_path = pipeline.download_youtube_video(youtube_url)
                
                progress_bar.progress(10)
                
                # 2. Audio Extraction
                status_text.text("Extracting audio...")
                audio_path = pipeline.extract_audio(video_path)
                progress_bar.progress(20)
                
                # 3. Transcription
                status_text.text("Transcribing audio (this may take a while)...")
                transcription_result = pipeline.transcribe_audio(audio_path)
                st.session_state.transcript = transcription_result
                progress_bar.progress(50)
                
                # 4. Multimodal Analysis
                status_text.text("Running multimodal analysis...")
                audio_analyzer = AudioAnalyzer()
                vision_analyzer = VisionAnalyzer()
                text_analyzer = TextAnalyzer()
                
                # Audio
                status_text.text("Analyzing audio dynamics...")
                audio_times, audio_energy = audio_analyzer.analyze(audio_path)
                progress_bar.progress(60)

                # Vision
                status_text.text("Analyzing visual content...")
                # Sample every 1 second for performance
                vis_times, vis_scores = vision_analyzer.analyze_video(video_path, sample_rate=1.0)
                progress_bar.progress(80)
                
                # 5. Viral Scoring
                status_text.text("Calculating viral scores...")
                
                weights = {
                    'audio': w_audio,
                    'wisdom': w_wisdom,
                    'face': w_face,
                    'emotion': w_emotion
                }
                scorer = ViralScorer(weights)
                scored_segments = scorer.score_segments(
                    transcription_result['segments'],
                    audio_times, audio_energy,
                    vis_times, vis_scores
                )
                
                # Store results
                st.session_state.results = {
                    "video_path": video_path,
                    "audio_times": audio_times,
                    "audio_energy": audio_energy,
                    "vis_times": vis_times,
                    "vis_scores": vis_scores,
                    "transcript": transcription_result,
                    "scored_segments": scored_segments
                }
                
                status_text.text("Analysis complete!")
                progress_bar.progress(100)
                st.success("Video processed successfully!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

        # Metrics Display
        if "results" in st.session_state:
            res = st.session_state.results
            scores = [s['score'] for s in res['scored_segments']]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            duration = res["audio_times"][-1] if len(res["audio_times"]) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{duration/60:.1f}m</div>
                    <div class="metric-label">Duration</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(res["transcript"]["segments"])}</div>
                    <div class="metric-label">Segments</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_score:.2f}</div>
                    <div class="metric-label">Avg Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#2ecc71;">Ready</div>
                    <div class="metric-label">Status</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive Timeline Visualization
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Viral Score over time (mapped from segments)
            seg_starts = [s['start'] for s in res['scored_segments']]
            seg_scores = [s['score'] for s in res['scored_segments']]
            
            fig.add_trace(go.Scatter(x=seg_starts, y=seg_scores, name="Viral Score", line=dict(color='#ff4b4b', width=4)), secondary_y=False)
            
            # Metadata overlay (Audio Energy)
            ds_factor = max(1, len(res["audio_times"]) // 1000)
            fig.add_trace(go.Scatter(x=res["audio_times"][::ds_factor], y=res["audio_energy"][::ds_factor], name="Audio Energy", opacity=0.3, line=dict(color='#667eea')), secondary_y=True)
            
            fig.update_layout(title="Viral Score Timeline", xaxis_title="Time (s)", height=400)
            fig.update_yaxes(title_text="Viral Score", secondary_y=False)
            fig.update_yaxes(title_text="Audio Energy", showgrid=False, secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Multimodal Insights")
        if "results" in st.session_state:
            res = st.session_state.results
            scored_segments = res['scored_segments']
            
            # Sort by score
            sorted_segments = sorted(scored_segments, key=lambda x: x['score'], reverse=True)
            
            st.subheader("Top Segments Breakdown")
            for i, seg in enumerate(sorted_segments[:5]):
                with st.expander(f"#{i+1}: {seg['start']:.1f}s - {seg['end']:.1f}s (Score: {seg['score']:.2f})"):
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.write(f"**Text:** \"{seg['text']}\"")
                    with c2:
                        comps = seg['components']
                        # Mini radar chart for components
                        radar_fig = go.Figure(data=go.Scatterpolar(
                            r=[comps['audio'], comps['wisdom'], comps['face'], comps['emotion']],
                            theta=['Audio', 'Wisdom', 'Face', 'Emotion'],
                            fill='toself'
                        ))
                        radar_fig.update_layout(margin=dict(t=0, b=0, l=30, r=30), height=150)
                        st.plotly_chart(radar_fig, use_container_width=True)

    with tab4:
        st.header("Viral Clips")
        if "results" in st.session_state:
            res = st.session_state.results
            score_segs = sorted(res['scored_segments'], key=lambda x: x['score'], reverse=True)[:3]
            
            gen = ClipGenerator()
            
            cols = st.columns(3)
            for i, seg in enumerate(score_segs):
                with cols[i]:
                    st.subheader(f"Clip #{i+1}")
                    st.metric("Viral Score", f"{seg['score']:.2f}")
                    st.caption(f"{seg['start']:.1f}s - {seg['end']:.1f}s")
                    
                    clip_key = f"clip_{i}"
                    
                    if st.button(f"Generate Clip {i+1}", key=f"btn_{i}"):
                        with st.spinner("Generating clip (cropping, captioning)..."):
                            # Get word-level timestamps for this segment
                            # We need to filter the original transcript's segments/words
                            # Whisper 'word_timestamps=True' structure: segments -> words
                            
                            all_words = []
                            for t_seg in res['transcript']['segments']:
                                if 'words' in t_seg:
                                    all_words.extend(t_seg['words'])
                            
                            # Filter words in range
                            clip_words = [w for w in all_words if w['start'] >= seg['start'] and w['end'] <= seg['end']]
                            
                            out_path = f"generated_clip_{i}.mp4"
                            vid_path, thumb_path = gen.generate_clip(
                                res['video_path'], 
                                seg['start'], 
                                seg['end'], 
                                clip_words, 
                                out_path
                            )
                            
                            if vid_path:
                                st.session_state[clip_key] = vid_path
                                st.success("Generated!")
                            else:
                                st.error("Failed to generate clip.")

                    if clip_key in st.session_state:
                        st.video(st.session_state[clip_key])
                        with open(st.session_state[clip_key], "rb") as f:
                            st.download_button("Download", f, file_name=f"viral_clip_{i+1}.mp4")

    with tab5:
        st.header("Export & Results")
        if "results" in st.session_state:
            res = st.session_state.results
            st.success("Processing complete. You can download individual clips in the 'Viral Clips' tab.")
            
            st.subheader("Session Data")
            st.json({
                "video_path": res["video_path"],
                "duration_seconds": res["audio_times"][-1],
                "total_segments": len(res["scored_segments"]),
                "top_score": max([s['score'] for s in res['scored_segments']]) if res['scored_segments'] else 0
            })
            
            # Placeholder for bulk download (zipping clips)
            st.info("Batch download functionality coming in v2.0")

if __name__ == "__main__":
    main()
