# Test and validation checklist

## Automated smoke checks

Run from the project folder:

```bash
python -m py_compile app.py
python smoke_test.py
```

## Manual acceptance checks

- [ ] App opens to Overview without errors.
- [ ] Load sample session shows a clearly labelled synthetic timeline.
- [ ] Video upload accepts MP4/AVI/MOV/MKV and produces a face-count timeline where frames can be decoded.
- [ ] Video report includes anonymous tracks and CSV downloads.
- [ ] Audio upload accepts WAV/FLAC/OGG and produces RMS/ZCR/spectral-centroid features.
- [ ] Session insights explains whether one or both modalities were used.
- [ ] JSON report downloads and opens as valid JSON.
- [ ] Documentation page is accessible.
- [ ] Test with short, consented sample media; compare output with manual observations.

## Limitations of validation

A dashboard opening successfully is only a smoke test. It does not validate detection accuracy, attendance correctness, audio classification accuracy, fairness, or suitability for high-stakes use. Those require a consented, representative evaluation dataset and defined metrics.
