# Project workflow

1. Requirements specification and local environment setup.
2. Vision module: sample video frames, detect faces, visualize face-count timeline.
3. Audio module: compute RMS loudness and activity windows without transcription.
4. Agent implementation: combine signals using explainable rules.
5. Analytics: produce teaching prompts and downloadable session report.
6. Interface/testing: Streamlit UI, smoke tests, documentation, demo preparation.

## Architecture
- Video input → OpenCV Haar cascade → face-count timeline → visible-presence ratio.
- Audio input → SoundFile/NumPy → RMS timeline → activity ratio.
- Available indicators → rule-based recommendation engine → UI and JSON report.
