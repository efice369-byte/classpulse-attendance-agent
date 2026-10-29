# Requirements Specification

## Functional requirements
1. Process an uploaded video locally and sample frames.
2. Detect visible faces using OpenCV.
3. Summarize face-count trends and visible-presence ratio.
4. Analyze uploaded audio by measuring RMS loudness in one-second windows.
5. Estimate relative audio activity without speech recognition.
6. Combine available indicators into an explainable proxy score.
7. Generate human-readable instructor suggestions.
8. Export vision/audio timelines as CSV and session summary as JSON.

## Non-functional requirements
- Python 3.10+ recommended.
- Runs locally in a browser-based Streamlit interface.
- No external AI API required.
- Explain limitations and require human judgment.
- Avoid retaining raw media beyond the app's processing session.

## Acceptance criteria
- App starts with `streamlit run app.py`.
- Overview and documentation pages render.
- Video upload can produce a face-count timeline.
- WAV/FLAC/OGG upload can produce an audio activity timeline.
- Insights page produces recommendations and report download.
