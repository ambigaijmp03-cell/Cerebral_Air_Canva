import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np

# --- MediaPipe face mesh connection map (468 landmarks) ---
# This gives us real topology instead of random i+1 links
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

# 1. Setup Detectors
base_options_f = python.BaseOptions(model_asset_path='face_landmarker.task')
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=base_options_f,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
)
base_options_h = python.BaseOptions(model_asset_path='hand_landmarker.task')
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=base_options_h,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
)

# 2. State
anchor_x, anchor_y = 200.0, 200.0      # Smoothed anchor position
target_x, target_y = 200.0, 200.0      # Raw hand target
vel_x, vel_y       = 0.0, 0.0          # Anchor velocity for inertia
is_locked          = False

# Per-landmark particle state: velocity for disintegration
num_lm = 478
particle_vx = np.zeros(num_lm)
particle_vy = np.zeros(num_lm)
particle_ox = np.zeros(num_lm)         # Offset accumulated from physics
particle_oy = np.zeros(num_lm)

disintegration = 0.0
LERP_SPEED     = 0.14   # How fast anchor follows hand
DRAG           = 0.88   # Velocity drag per frame
DRIFT_SCALE    = 1.8    # How strongly particles drift on disintegrate
GRAVITY        = 0.12   # Gentle downward pull on particles

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    timestamp = int(time.time() * 1000)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    f_res = face_detector.detect_for_video(mp_image, timestamp)
    h_res = hand_detector.detect_for_video(mp_image, timestamp)

    # 3. Hand / Anchor logic
    if h_res.hand_landmarks:
        lms = h_res.hand_landmarks[0]
        dist = np.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
        ix, iy = lms[8].x * w, lms[8].y * h

        if dist > 0.07:
            is_locked = False
            target_x, target_y = ix, iy

            # Kick each particle's velocity outward from anchor on disintegration start
            if disintegration < 1.0:
                particle_vx[:] = np.random.uniform(-DRIFT_SCALE, DRIFT_SCALE, num_lm)
                particle_vy[:] = np.random.uniform(-DRIFT_SCALE, DRIFT_SCALE, num_lm)

            disintegration = min(disintegration + 2.5, 40.0)
            cv2.circle(frame, (int(ix), int(iy)), 10, (0, 220, 220), 2)
        else:
            is_locked = True
            disintegration = max(disintegration - 5.0, 0.0)

            # Re-attract: pull particle offsets back toward zero
            particle_ox *= 0.82
            particle_oy *= 0.82
            particle_vx *= 0.5
            particle_vy *= 0.5

            cv2.drawMarker(frame, (int(anchor_x), int(anchor_y)),
                           (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

    # --- Exponential lerp for smooth anchor movement ---
    dx = target_x - anchor_x
    dy = target_y - anchor_y
    vel_x = vel_x * DRAG + dx * LERP_SPEED
    vel_y = vel_y * DRAG + dy * LERP_SPEED
    anchor_x += vel_x
    anchor_y += vel_y

    # --- Particle physics update ---
    if disintegration > 0:
        particle_vx *= DRAG
        particle_vy *= DRAG
        particle_vy += GRAVITY                # gravity
        particle_ox += particle_vx * (disintegration / 20.0)
        particle_oy += particle_vy * (disintegration / 20.0)

    # 4. Volumetric rendering
    if f_res.face_landmarks:
        face_lms = f_res.face_landmarks[0]
        nose = face_lms[1]
        nw, nh = nose.x * w, nose.y * h

        # Collect all 3D points relative to nose, then depth-sort (far → near)
        pts = []
        for i, lm in enumerate(face_lms):
            rel_x = (lm.x * w) - nw
            rel_y = (lm.y * h) - nh
            z = lm.z  # negative = closer to camera in MediaPipe coords

            # Full 3D projection: z affects both X and Y (head tilt parallax)
            proj_scale = 1.0 + lm.z * 0.35
            cx = anchor_x + rel_x * proj_scale + particle_ox[i]
            cy = anchor_y + rel_y * proj_scale + particle_oy[i]

            # Depth-graded color: warm (close) → cool (far)
            # z range is roughly -0.1 (close) to 0.1 (far)
            t = np.clip((lm.z + 0.08) / 0.16, 0, 1)  # 0=close, 1=far
            b = int(80  + t * 175)   # blue  channel
            g = int(220 - t * 120)   # green channel
            r = int(255 - t * 200)   # red   channel  (warm close, cool far)

            # Alpha / radius from depth (close = larger, brighter)
            radius = max(1, int(2.5 - lm.z * 18))

            pts.append((z, int(cx), int(cy), (b, g, r), radius, i))

        # Depth sort: draw far points first so near points render on top
        pts.sort(key=lambda p: p[0], reverse=True)

        # Draw points
        for z, cx, cy, color, radius, i in pts:
            if 0 < cx < w and 0 < cy < h:
                alpha = np.clip(1.0 - abs(z) * 6, 0.3, 1.0)
                blended = tuple(int(c * alpha) for c in color)
                cv2.circle(frame, (cx, cy), radius, blended, -1)

        # Draw real topology edges (only when solid / low disintegration)
        if disintegration < 5:
            edge_alpha = max(0.0, 1.0 - disintegration / 5.0)
            pt_map = {p[5]: p for p in pts}  # index → point data
            for (a_idx, b_idx) in FACEMESH_TESSELATION:
                if a_idx in pt_map and b_idx in pt_map:
                    _, ax, ay, ac, _, _ = pt_map[a_idx]
                    _, bx, by, bc, _, _ = pt_map[b_idx]
                    if (0 < ax < w and 0 < ay < h and 0 < bx < w and 0 < by < h):
                        edge_color = tuple(int((ac[i] + bc[i]) / 2 * edge_alpha * 0.6)
                                           for i in range(3))
                        cv2.line(frame, (ax, ay), (bx, by), edge_color, 1)

    # 5. UI
    d_pct = int(disintegration / 40.0 * 100)
    status = "SOLID" if is_locked else f"DISSOLVING {d_pct}%"
    cv2.putText(frame, status, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(frame, "PINCH=lock  OPEN=float", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Cerebral Anchor — Improved", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_detector.close()
hand_detector.close()