# ClassPulse — Multimodal Attendance & Engagement Agent

A local-first demonstration prototype for classroom video presence indicators, audio activity analysis, explainable recommendations, and session reporting.

## Features
- Streamlit dashboard with Overview, Video Analysis, Audio Analysis, Session Insights, and Documentation.
- OpenCV Haar-cascade face detection on sampled video frames.
- One-second RMS audio activity analysis using SoundFile and NumPy.
- Transparent rule-based instructor suggestions.
- CSV and JSON exports.
- Synthetic sample session to demonstrate the interface without classroom media.

## Important limitations
This is a prototype, not a production attendance system. Face detection does not identify named students, so it cannot independently verify per-student attendance. Audio loudness is not speech recognition and is not a reliable measure of participation. The engagement proxy is illustrative and not scientifically validated. Do not use it for grading or disciplinary decisions.

## Setup (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/.
2. Extract the ZIP into a folder.
3. Open that folder in VS Code.
4. Open Terminal → New Terminal.
5. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
6. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Run:
   ```powershell
   streamlit run app.py
   ```
8. Open the local URL shown in the terminal (usually http://localhost:8501).

## Demo flow
1. Open Overview and click **Load sample session**.
2. Open Session insights to show the recommendation engine.
3. For a real media demo, use a short video under Video analysis and a WAV/FLAC/OGG file under Audio analysis.
4. Download CSV/JSON reports.

## Suggested GitHub repository description
Privacy-first classroom analytics prototype combining OpenCV face detection, audio activity features, and explainable rule-based recommendations.

## Suggested project demo description
ClassPulse is a local-first prototype for classroom session analytics. It samples classroom video to count visible faces, measures audio activity without speech transcription, and combines available signals into an explainable proxy score with instructor suggestions. Raw media is processed locally by the application. Results are indicative only and require human interpretation.

## Project structure
- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `requirements.md` — functional/non-functional requirements and acceptance criteria.
- `workflow.md` — development workflow and architecture.
