import io
import math
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ClassPulse | Attendance & Engagement", page_icon="🎓", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#101b3f,#254e8a); color:white; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.1rem;}
.hero p {margin:.4rem 0 0 0; opacity:.88;}
.metric-card {border:1px solid rgba(120,130,150,.25); border-radius:14px; padding:14px 16px;}
.small-note {color:#737b8c;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 ClassPulse</h1>
  <p>Multimodal Attendance & Engagement Agent · Privacy-first classroom analytics</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Session setup")
    session_name = st.text_input("Session name", "Introduction to Data Science")
    course_name = st.text_input("Course / class", "CSE — Section A")
    session_minutes = st.number_input("Session duration (minutes)", min_value=1, max_value=300, value=50)
    st.divider()
    page = st.radio("Navigate", ["Overview", "Video analysis", "Audio analysis", "Session insights", "Documentation"])
    st.divider()
    st.caption("Runs locally. Uploaded media is processed in this app session and is not intentionally sent to an external AI API.")

# State
if "video_result" not in st.session_state:
    st.session_state.video_result = None
if "audio_result" not in st.session_state:
    st.session_state.audio_result = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False

def make_demo_data():
    rng = np.random.default_rng(17)
    mins = np.arange(0, int(session_minutes) + 1, 5)
    if len(mins) < 2:
        mins = np.array([0, int(session_minutes)])
    engagement = np.clip(78 - mins * (20 / max(session_minutes, 1)) + rng.normal(0, 5, len(mins)), 35, 95)
    audio = np.clip(rng.normal(0.35, 0.10, len(mins)), 0.05, 0.8)
    return pd.DataFrame({"Minute": mins, "Engagement indicator (%)": engagement.round(1), "Audio activity (relative)": audio.round(2)})

def recommendation(score, audio_activity=None):
    suggestions = []
    if score < 45:
        suggestions.append("Pause for a quick question or a 2-minute pair discussion; the current visual activity indicator is low.")
    elif score < 65:
        suggestions.append("Try a short poll, worked example, or check-for-understanding question to re-engage the class.")
    else:
        suggestions.append("The current indicators look steady. Maintain the pace and continue checking understanding.")
    if audio_activity is not None:
        if audio_activity < 0.12:
            suggestions.append("Audio activity is very low. If this is a discussion session, invite questions; if it is a lecture, quiet audio may be expected.")
        elif audio_activity > 0.65:
            suggestions.append("Audio activity is high. Check whether it reflects productive discussion or background noise before changing the lesson.")
    suggestions.append("Treat these as prompts for the instructor, not definitive measurements of attention or learning.")
    return suggestions

def process_video(uploaded, sample_every=12, max_frames=240):
    import cv2
    data = uploaded.getvalue()
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, "Could not open this video. Try MP4 or AVI."
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_duration = frame_count / fps if fps > 0 else 0
    rows = []
    sampled = 0
    frame_idx = 0
    max_faces = 0
    face_sum = 0
    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
            count = int(len(faces))
            sec = frame_idx / fps if fps else frame_idx
            rows.append({"Time (seconds)": round(sec, 1), "Detected faces": count})
            max_faces = max(max_faces, count)
            face_sum += count
            sampled += 1
        frame_idx += 1
    cap.release()
    try:
        os.unlink(path)
    except OSError:
        pass
    if not rows:
        return None, "No readable frames were found in the video."
    df = pd.DataFrame(rows)
    avg_faces = face_sum / len(rows)
    summary = {
        "sampled_frames": len(rows),
        "duration_seconds": round(total_duration, 1),
        "average_faces": round(avg_faces, 2),
        "peak_faces": int(max_faces),
        "presence_ratio": round(sum(1 for x in df["Detected faces"] if x > 0) / len(df), 3),
        "timeline": df,
    }
    return summary, None

def analyze_audio(uploaded):
    import soundfile as sf
    raw = uploaded.getvalue()
    try:
        audio, sr = sf.read(io.BytesIO(raw), always_2d=True)
    except Exception as e:
        return None, f"Could not read audio: {e}. Try WAV, FLAC, or OGG."
    if audio.size == 0:
        return None, "The audio file appears to be empty."
    mono = np.mean(audio, axis=1)
    duration = len(mono) / sr
    frame_seconds = 1.0
    chunk = max(1, int(sr * frame_seconds))
    values = []
    for start in range(0, len(mono), chunk):
        segment = mono[start:start + chunk]
        if len(segment) == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(segment))))
        peak = float(np.max(np.abs(segment)))
        values.append({"Time (seconds)": start / sr, "RMS loudness": rms, "Peak amplitude": peak})
    df = pd.DataFrame(values)
    # Relative threshold for activity, intentionally not speech recognition.
    threshold = max(float(df["RMS loudness"].quantile(0.35)) * 1.6, 0.008)
    active = df["RMS loudness"] > threshold
    activity_ratio = float(active.mean()) if len(active) else 0.0
    noise_floor = float(df["RMS loudness"].quantile(0.2)) if len(df) else 0.0
    return {
        "duration_seconds": round(duration, 1),
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "activity_ratio": round(activity_ratio, 3),
        "mean_rms": round(float(df["RMS loudness"].mean()), 4),
        "noise_floor_estimate": round(noise_floor, 4),
        "timeline": df,
    }, None

# ---------- Pages ----------
if page == "Overview":
    st.subheader("Session overview")
    st.write(f"**{session_name}** · {course_name} · Planned duration: {session_minutes} minutes")
    st.info("Start with demo data to explore the dashboard, or upload a video/audio file from the sidebar pages. Face detection counts visible faces; it does not identify students by name.")
    c1, c2, c3, c4 = st.columns(4)
    video = st.session_state.video_result
    audio = st.session_state.audio_result
    c1.metric("Vision status", "Analyzed" if video else "Not run")
    c2.metric("Peak faces detected", video["peak_faces"] if video else "—")
    c3.metric("Audio activity", f"{audio['activity_ratio']*100:.0f}%" if audio else "—")
    c4.metric("Privacy mode", "Local processing")
    if st.button("Load sample session", type="primary"):
        st.session_state.demo_mode = True
        st.session_state.demo_data = make_demo_data()
        st.success("Sample session loaded. Open Session insights.")
    if st.session_state.get("demo_mode"):
        st.subheader("Sample engagement timeline")
        st.plotly_chart(px.line(st.session_state.demo_data, x="Minute", y="Engagement indicator (%)", markers=True), use_container_width=True)
        st.caption("Illustrative synthetic data only — not measurements from real students.")
    st.subheader("What this prototype does")
    cols = st.columns(3)
    cols[0].markdown("**👁️ Vision**\n\nSamples video frames and counts detected faces to estimate sustained visible presence.")
    cols[1].markdown("**🔊 Audio**\n\nMeasures loudness/activity patterns without recording transcripts or recognizing speech.")
    cols[2].markdown("**🧭 Agent**\n\nUses transparent rules to suggest instructor actions and session reflection prompts.")

elif page == "Video analysis":
    st.subheader("Vision module — attendance presence")
    st.write("Upload a classroom video. The app samples frames and detects faces using OpenCV's Haar cascade.")
    st.warning("This is anonymous face detection, not face recognition. It cannot reliably tell which named student is present, and overlapping faces or poor lighting can affect counts.")
    upload = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    sample_rate = st.select_slider("Sample one frame every", options=[3, 6, 12, 24, 48], value=12, format_func=lambda x: f"{x} frames")
    if upload and st.button("Analyze video", type="primary"):
        with st.spinner("Sampling frames and detecting faces locally..."):
            result, error = process_video(upload, sample_every=sample_rate)
        if error:
            st.error(error)
        else:
            st.session_state.video_result = result
            st.success("Video analysis complete.")
    result = st.session_state.video_result
    if result:
        a,b,c,d = st.columns(4)
        a.metric("Video duration", f"{result['duration_seconds']} s")
        b.metric("Frames sampled", result["sampled_frames"])
        c.metric("Average faces / sampled frame", result["average_faces"])
        d.metric("Frames with ≥1 face", f"{result['presence_ratio']*100:.0f}%")
        st.plotly_chart(px.line(result["timeline"], x="Time (seconds)", y="Detected faces", markers=True, title="Visible face count over time"), use_container_width=True)
        st.download_button("Download vision timeline CSV", result["timeline"].to_csv(index=False), "vision_timeline.csv", "text/csv")
        st.caption("Attendance limitation: use this as a room-level presence signal. Do not treat a face count as verified individual attendance.")

elif page == "Audio analysis":
    st.subheader("Audio module — activity and noise indicators")
    st.write("Upload a WAV, FLAC, or OGG recording. The app estimates loudness by one-second windows; it does not transcribe speech.")
    audio_upload = st.file_uploader("Choose audio", type=["wav", "flac", "ogg"], key="audio_uploader")
    if audio_upload and st.button("Analyze audio", type="primary"):
        with st.spinner("Analyzing audio locally..."):
            result, error = analyze_audio(audio_upload)
        if error:
            st.error(error)
        else:
            st.session_state.audio_result = result
            st.success("Audio analysis complete.")
    result = st.session_state.audio_result
    if result:
        a,b,c,d = st.columns(4)
        a.metric("Duration", f"{result['duration_seconds']} s")
        b.metric("Sample rate", f"{result['sample_rate']} Hz")
        c.metric("Active windows", f"{result['activity_ratio']*100:.0f}%")
        d.metric("Mean RMS", result["mean_rms"])
        st.plotly_chart(px.line(result["timeline"], x="Time (seconds)", y="RMS loudness", title="Audio loudness by second"), use_container_width=True)
        st.download_button("Download audio timeline CSV", result["timeline"].to_csv(index=False), "audio_timeline.csv", "text/csv")
        st.caption("RMS activity can reflect speech, noise, or other sounds. It is not a speech detector and does not indicate learning by itself.")

elif page == "Session insights":
    st.subheader("Analytics & recommendation engine")
    video = st.session_state.video_result
    audio = st.session_state.audio_result
    if not video and not audio and not st.session_state.get("demo_mode"):
        st.info("Run video/audio analysis or load a sample session from Overview to populate insights.")
    else:
        if st.session_state.get("demo_mode"):
            df = st.session_state.demo_data
            score = float(df["Engagement indicator (%)"].mean())
        else:
            # A deliberately explainable proxy score, not a psychological measurement.
            visual = (video["presence_ratio"] * 100) if video else 60.0
            audio_score = (audio["activity_ratio"] * 100) if audio else 50.0
            score = 0.7 * visual + 0.3 * audio_score
        audio_activity = audio["activity_ratio"] if audio else None
        a,b = st.columns(2)
        a.metric("Engagement proxy", f"{score:.0f}/100")
        b.metric("Signals available", sum([bool(video), bool(audio), bool(st.session_state.get("demo_mode"))]))
        st.caption("The proxy combines visible-face presence and audio activity only when real media was analyzed. It is an operational indicator, not a validated measure of student attention.")
        st.subheader("Instructor suggestions")
        for i, tip in enumerate(recommendation(score, audio_activity), 1):
            st.markdown(f"{i}. {tip}")
        if video:
            st.subheader("Vision summary")
            st.json({k:v for k,v in video.items() if k != "timeline"})
        if audio:
            st.subheader("Audio summary")
            st.json({k:v for k,v in audio.items() if k != "timeline"})
        report = {
            "session_name": session_name,
            "course": course_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "engagement_proxy_score": round(score, 1),
            "vision": {k:v for k,v in video.items() if k != "timeline"} if video else None,
            "audio": {k:v for k,v in audio.items() if k != "timeline"} if audio else None,
            "recommendations": recommendation(score, audio_activity),
            "limitations": "Anonymous face counts and audio loudness are weak proxies, not verified attendance or attention."
        }
        import json
        st.download_button("Download session report (JSON)", json.dumps(report, indent=2), "session_report.json", "application/json")

elif page == "Documentation":
    st.subheader("Project documentation")
    st.markdown("""
### Problem statement
Traditional attendance systems focus on presence and miss session-level context. ClassPulse demonstrates a privacy-first prototype that combines visual presence indicators and audio activity with explainable rules.

### Functional requirements
- Upload and sample a video file for face detection.
- Estimate visible presence from the proportion of sampled frames containing at least one face.
- Upload audio and compute one-second RMS loudness, active-window ratio, and a noise-floor estimate.
- Combine available indicators into an explicitly labelled engagement proxy.
- Generate rule-based instructor suggestions.
- Export timelines as CSV and a session report as JSON.

### Non-functional requirements
- Local processing; no external model API is called.
- Clear limitations and human oversight.
- Reproducible Python environment.
- Downloadable outputs for inspection.

### Architecture
`Video → OpenCV face detector → presence timeline`
`Audio → RMS signal analysis → activity timeline`
`Presence + activity → explainable rule engine → dashboard + report`

### Ethical and technical limitations
- The system does not identify a person by name.
- Face detection can fail with occlusion, camera angles, lighting, or low resolution.
- Audio activity is not equivalent to speech, participation, or learning.
- The engagement proxy is not validated and must not be used for grading or disciplinary decisions.
- Obtain consent and follow institutional privacy policies before processing real classroom media.
""")

st.divider()
st.caption("ClassPulse prototype · Python · Streamlit · OpenCV · NumPy · SoundFile · Plotly · Pandas")
