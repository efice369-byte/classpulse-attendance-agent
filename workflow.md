# ClassPulse workflow and architecture

## Processing pipeline

1. **Session configuration** — instructor enters session name, class, and duration.
2. **Vision path** — user uploads a video; OpenCV samples frames, detects faces, and matches detections between adjacent sampled frames using nearest-centroid matching. It emits frame-level counts, a room-level presence ratio, and temporary anonymous track summaries.
3. **Audio path** — user uploads WAV/FLAC/OGG; SoundFile reads samples and NumPy computes one-second RMS, peak amplitude, zero-crossing rate, and spectral centroid. A relative threshold estimates active windows; an energy/ZCR heuristic estimates speech-like windows but is explicitly unvalidated.
4. **Signal fusion** — available visual presence and audio activity are combined with an explainable weighted proxy (70% visual and 30% audio when both are present). When one modality is missing, the proxy uses the available signal with a disclosed default for the absent modality only in the current implementation; interpret cautiously.
5. **Recommendation engine** — rule-based suggestions offer possible instructor interventions and warn against interpreting signals as definitive attention.
6. **Reporting** — dashboard, CSV timelines, anonymous track CSV, and JSON session report are generated.

## Privacy and responsible use

- No face embeddings or identity recognition.
- Anonymous track IDs are ephemeral and approximate.
- Raw video/audio is not sent to a third-party AI API by the app.
- Process only media with appropriate consent and institutional authorization.
- Do not use scores for grading, discipline, or consequential student decisions.

## Deployment

Streamlit Community Cloud runs `app.py` using dependencies in `requirements.txt`. Push updates to the configured GitHub branch to trigger a redeploy. The deployed app processes uploads on the app host; “local-first” here means no external AI service is called, not that a cloud-hosted deployment keeps files on the user's own device.
