# AI Hitting Analyzer

> **Status: Experimental work in progress**

A Flask and OpenCV prototype that estimates baseball exit velocity, launch angle, and projected air distance from a recorded swing.

## How it works

1. Upload a video of a batted ball.
2. Enter the contact time in seconds.
3. Mark two points across home plate to establish scale.
4. Run the analysis.

The backend tracks bright moving objects in the frames immediately after contact and uses the 17-inch width of home plate as a visual reference.

## Tech stack

- Python
- Flask
- OpenCV
- NumPy
- HTML, CSS, and JavaScript

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

## Limitations

This is not a calibrated measurement system. Results depend on camera angle, frame rate, lighting, motion blur, and accurate reference points. Uploaded videos are stored in the local `uploads/` directory; production deployment would require cleanup, upload limits, rate limiting, and stronger validation.
