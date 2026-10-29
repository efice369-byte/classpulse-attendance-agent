# ClassPulse — Multimodal Attendance & Engagement Agent

ClassPulse is a privacy-conscious classroom analytics **prototype** combining sampled-video face detection, approximate anonymous tracking, audio signal features, explainable multimodal scoring, instructor suggestions, and exportable reports.

## What works

- **Overview:** session setup, synthetic demo data, timeline chart.
- **Vision:** OpenCV face detection on sampled frames; approximate centroid matching between adjacent sampled frames; room-level presence ratio; anonymous track summary; CSV export.
- **Audio:** one-second RMS loudness, peak amplitude, zero-crossing rate, spectral centroid, relative active-window ratio, noise-floor estimate, and an experimental speech-like heuristic; CSV export.
- **Insights:** clearly labelled proxy score, transparent recommendations, modality summaries, JSON report.
- **Documentation:** functional requirements, architecture, privacy and known limitations.

## Important truth-in-advertising note

This is a working prototype, not a fully validated attendance system. It does **not** identify students by name, and anonymous face tracks do not verify individual attendance. Audio features cannot reliably distinguish speech from noise; the speech-like heuristic is experimental. The engagement proxy is not scientifically validated and must not be used for grading, discipline, or other high-stakes decisions.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Test

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py
.\.venv\Scripts\python.exe smoke_test.py
```

See `TESTING.md` for manual acceptance checks and `requirements.md` for a task-by-task feature matrix.

## Streamlit Community Cloud deployment

The configured app uses `app.py` and `requirements.txt`. Commit/push updated files to the same GitHub repository and `main` branch to trigger a rebuild. A cloud-hosted instance processes uploaded files on the app host, even though the app does not call external AI APIs.

## Privacy

Only process classroom media with appropriate consent and institutional authorization. Do not use the proxy for grading or disciplinary decisions. Review `workflow.md` and `requirements.md` for technical and ethical limitations.
