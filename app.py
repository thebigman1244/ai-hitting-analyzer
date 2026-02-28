import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- ALWAYS RETURN JSON ----------
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


# ---------- SERVE FRONTEND ----------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


# ---------- TRACKING ----------
def track_ball(video_path, contact_time):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("OpenCV failed to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 5:
        cap.release()
        raise ValueError("Invalid FPS.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    contact_frame = int(contact_time * fps)

    if contact_frame >= total_frames - 3:
        cap.release()
        raise ValueError("Contact time too late in clip.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, contact_frame)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read contact frame.")

    scale = 0.75
    prev = cv2.resize(prev, None, fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3), np.uint8)
    positions = []
    prev_pt = None

    for _ in range(20):
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1e9

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (8 < area < 400):
                continue

            x,y,w,h = cv2.boundingRect(cnt)
            if h == 0:
                continue

            ar = w/float(h)
            if not (0.5 < ar < 1.6):
                continue

            roi = gray[y:y+h, x:x+w]
            brightness = np.mean(roi)
            if brightness < 150:
                continue

            cx = x + w*0.5
            cy = y + h*0.5

            if prev_pt is not None:
                dx = cx - prev_pt[0]
                dy = cy - prev_pt[1]
                jump = np.sqrt(dx*dx + dy*dy)
                if jump < 2 or jump > 80:
                    continue
                score = brightness*2 - jump*6
            else:
                score = brightness*2

            if score > best_score:
                best_score = score
                best = (cx, cy)

        if best:
            positions.append(best)
            prev_pt = best

        prev_gray = gray

    cap.release()

    if len(positions) < 6:
        raise ValueError("Tracking failed. Adjust contact time ±0.10s.")

    return fps, np.array(positions)/scale


def compute_stats(fps, positions, ft_per_px):
    x = positions[:,0]
    y = positions[:,1]

    dt = 1.0/fps
    speeds = []

    for i in range(1, min(5, len(x))):
        dx_ft = (x[i]-x[i-1])*ft_per_px
        dy_ft = (y[i]-y[i-1])*ft_per_px
        vx = dx_ft/dt
        vy = -dy_ft/dt
        speeds.append(np.sqrt(vx*vx + vy*vy))

    speeds.sort()
    if len(speeds) > 2:
        speeds = speeds[:-1]

    v_fps = max(speeds)
    v_mph = v_fps * 0.681818

    dx0 = (x[1]-x[0])*ft_per_px
    dy0 = (y[1]-y[0])*ft_per_px
    angle = np.degrees(np.arctan2(-dy0, abs(dx0)))

    g = 32.174
    if angle > 0:
        air = (v_fps**2)*np.sin(2*np.radians(angle))/g
    else:
        air = 0

    return {
        "exit_velocity_mph": round(float(v_mph),2),
        "launch_angle_deg": round(float(angle),2),
        "air_distance_ft": round(float(air),1)
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        video = request.files["video"]
        contact_time = float(request.form["contact"])
        plate_pts = json.loads(request.form["plate"])

        dx = plate_pts[0][0] - plate_pts[1][0]
        dy = plate_pts[0][1] - plate_pts[1][1]
        plate_px = np.sqrt(dx*dx + dy*dy)
        ft_per_px = (17.0/12.0)/plate_px

        filename = secure_filename(video.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(path)

        fps, positions = track_ball(path, contact_time)
        stats = compute_stats(fps, positions, ft_per_px)

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
