"""Fast source/dependency smoke test; does not claim model accuracy."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)
required = [
    "def process_video(", "def analyze_audio(", "def recommendation(",
    "Detected faces", "Zero-crossing rate", "Spectral centroid (Hz)",
    "Download session report (JSON)", "Load sample session", "Documentation",
]
missing = [token for token in required if token not in source]
if missing:
    raise SystemExit("Missing expected app feature(s): " + ", ".join(missing))
reqs = (root / "requirements.txt").read_text(encoding="utf-8").lower()
for package in ("streamlit", "opencv-python-headless", "numpy", "pandas", "plotly", "soundfile"):
    if package not in reqs:
        raise SystemExit(f"Missing dependency declaration: {package}")
print("PASS: app.py parses and expected feature/dependency declarations are present.")
print("NOTE: This smoke test does not validate real-media accuracy or production readiness.")
