# Cerebral Air Canvas — 4D Volumetric AI Telekinesis

An interactive, real-time computer vision interface that combines 478-point 3D facial volumetric tracking with gesture-based telekinetic control. Using MediaPipe and OpenCV, the interface translates facial contours into a high-density digital particle essence that can be anchored in 3D space, dragged across the screen, or dissolved into a 4D particle cloud via hand gestures.

---

## Features

* **360° Volumetric Facial Mesh:** Leverages 478 MediaPipe landmarks with Z-axis depth extrusion to maintain accurate depth for jawline, nose, and ear contours from any viewing angle.
* **Telekinetic Hand-Off Switch:** Smoothly transitions control between head-tracking and hand-tracking using linear interpolation (Lerp).
* **Gesture-Driven Disintegration:** 
  * **Pinch Gesture:** Solidifies the particle cloud into a structured 3D face mesh and anchors it in place.
  * **Open Hand / Release:** Disintegrates the face mesh into dynamic "4D dust" that follows hand movement.
* **Recursive Decay Trail:** Custom accumulation buffers create an organic motion trail / digital ghosting effect.
* **Decoupled Architecture:** Modular split between inference (`engine.py`) and rendering/physics (`main.py`) to maximize frame rates and stability.

---

## File Structure

```text
├── engine.py                 # MediaPipe inference engine (Face + Hand detection)
├── main.py                   # Render loop, gesture logic, particle physics, Lerp
├── face_landmarker.task      # MediaPipe Face Landmarker model asset
├── hand_landmarker.task      # MediaPipe Hand Landmarker model asset
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
