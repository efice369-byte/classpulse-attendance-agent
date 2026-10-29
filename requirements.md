# ClassPulse — Requirements and Acceptance Criteria

## Functional requirements

| ID | Requirement | Acceptance check | Implementation status |
|---|---|---|---|
| FR-01 | Configure session/class and duration | Sidebar accepts session metadata | Implemented |
| FR-02 | Load an illustrative demo session | Overview creates a labelled synthetic timeline | Implemented; clearly marked synthetic |
| FR-03 | Detect visible faces in sampled video frames | Upload supported video; see face-count timeline and CSV | Implemented; depends on lighting/quality |
| FR-04 | Estimate sustained room-level presence | Show share of sampled frames with ≥1 face | Implemented as room-level proxy only |
| FR-05 | Estimate anonymous track continuity | Download approximate temporary track summaries | Prototype; IDs can split/merge and are not identities |
| FR-06 | Extract audio signal features | RMS, peak, zero-crossing rate, spectral centroid per second | Implemented |
| FR-07 | Estimate audio activity/noise floor | Display activity ratio and low-quantile RMS | Implemented as signal features, not semantic labels |
| FR-08 | Provide speech activity cue | Energy/ZCR speech-like heuristic | Experimental; not validated VAD or speech recognition |
| FR-09 | Fuse modalities transparently | Use available visual presence and audio activity with disclosed weights | Implemented as explainable proxy |
| FR-10 | Recommend teaching interventions | Rule-based suggestions based on proxy and audio activity | Implemented; instructor decides |
| FR-11 | Generate session analytics | Show metrics and modality summaries | Implemented |
| FR-12 | Export data | Download vision/audio timelines, anonymous track summary, JSON session report | Implemented |
| FR-13 | Provide documentation and ethical limits | Documentation page and README explain limitations | Implemented |

## Non-functional requirements

- Local processing in the running app; no external AI API is called by this prototype.
- Uploaded media is held in application memory/temp storage during processing and temporary video files are deleted after processing.
- Reproducible dependencies are declared in `requirements.txt`.
- Results are explainable and exportable.
- No face recognition or named-student identity matching is performed.
- Never use the proxy for grading, discipline, or high-stakes decisions.

## Environment

- Python 3.10+
- Streamlit, OpenCV, NumPy, Pandas, Plotly, SoundFile

## Known limitations / remaining production work

- Face tracking is centroid-based and approximate; it is not DeepSORT or a robust multi-object tracker.
- Anonymous tracks cannot establish verified per-student attendance.
- Audio features and the speech-like heuristic are not validated speech/noise classifiers.
- The engagement score is not a validated measure of attention, comprehension, or learning.
- The app is a functional prototype, not a production-grade institutional attendance system. Production deployment would require consent, retention/access controls, systematic test data, accessibility review, and security review.
