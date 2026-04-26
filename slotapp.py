import cv2
import numpy as np
import gxipy as gx
from ctypes import c_ubyte
from _ctypes import addressof
from gxipy import GxPixelFormatEntry, DxValidBit
from scipy.interpolate import splprep, splev

import timeit
import serial
import time

import pyvisual as pv
from PySide6.QtCore import QTimer
from ui.ui import create_ui

# ─── Config ───────────────────────────────────────────────────────────────────

MARKER_REFERENCE = 10
MARKER_CAR       = 1
MARKER_SIZE      = 0.05

TRACK_POINT_MIN_DIST = 0.02
TRACK_MIN_POINTS     = 50
LOOP_CLOSE_DIST      = 0.08

SPEED_MIN = 135
SPEED_MAX = 225
LOOKAHEAD = 0.05

MAPPING_LAPS  = 3
MAPPING_SPEED = "130"

EXPOSURE = 2000.0
GAIN     = 20.0

# ─── Camera helpers ───────────────────────────────────────────────────────────

def convert_to_rgb(image_convert, raw_image):
    image_convert.set_dest_format(GxPixelFormatEntry.RGB8)
    image_convert.set_valid_bits(DxValidBit.BIT0_7)
    buffer_size = image_convert.get_buffer_size_for_conversion(raw_image)
    buffer = (c_ubyte * buffer_size)()
    image_convert.convert(raw_image, addressof(buffer), buffer_size, False)
    return np.frombuffer(buffer, dtype=np.uint8).reshape(
        raw_image.frame_data.height, raw_image.frame_data.width, 3
    )

def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def matrix_to_pose(T):
    R = T[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec, T[:3, 3].reshape(1, 3)

# ─── Spline helpers ───────────────────────────────────────────────────────────

def filter_outliers(points, z_thresh=2.0):
    pts = np.array(points)
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mean, std = dists.mean(), dists.std()
    keep = np.ones(len(pts), dtype=bool)
    for i, d in enumerate(dists):
        if d > mean + z_thresh * std:
            keep[i + 1] = False
    return pts[keep].tolist()

def fit_spline(points):
    clean = filter_outliers(points)
    pts = np.array(clean)
    x, y = pts[:, 0], pts[:, 1]
    tck, _ = splprep([x, y], s=0.05, per=True, k=3)
    return tck

def sample_spline(tck, n=500):
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)

def project_onto_spline(tck, point, n=500):
    samples = sample_spline(tck, n)
    dists = np.linalg.norm(samples - np.array(point[:2]), axis=1)
    idx = np.argmin(dists)
    progress = idx / n
    return progress, dists[idx]

def compute_curvature_map(tck, n=500):
    u = np.linspace(0, 1, n)
    dx, dy   = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

def speed_for_position(tck, progress, curvature_map, n=500):
    idx      = int(progress * n)
    ahead    = int(LOOKAHEAD * n)
    indices  = [(idx + i) % n for i in range(ahead)]
    max_curv = curvature_map[indices].max()
    curv_min, curv_max = 0.0, 20.0
    t = np.clip((max_curv - curv_min) / (curv_max - curv_min), 0.0, 1.0)
    speed = int(SPEED_MAX - t * (SPEED_MAX - SPEED_MIN))
    return speed, max_curv

# ─── Debug view ───────────────────────────────────────────────────────────────

def draw_debug(track, car_xy, frame=None, speed=None, progress=None,
               lap_count=0, lap_time=None, best_lap=None,
               canvas_size=600, scale=200):
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    cx, cy = canvas_size // 2, canvas_size // 2

    def to_px(x, y):
        return int(cx + x * scale), int(cy - y * scale)

    for pt in track.points:
        px, py = to_px(*pt)
        cv2.circle(img, (px, py), 2, (60, 60, 60), -1)

    if track.spline is not None:
        pts = sample_spline(track.spline, n=500)
        curv = track.curvature if track.curvature is not None else np.zeros(500)
        curv_norm = np.clip(curv / 20.0, 0, 1)
        for i in range(len(pts) - 1):
            t = curv_norm[i]
            color = (0, int(255 * (1 - t)), int(255 * t))
            p1 = to_px(*pts[i])
            p2 = to_px(*pts[i + 1])
            cv2.line(img, p1, p2, color, 2)

    if car_xy is not None:
        px, py = to_px(*car_xy)
        cv2.circle(img, (px, py), 8, (255, 255, 0), -1)

    mode       = "RACE" if track.locked else f"MAPPING {track.lap_count+1}/{MAPPING_LAPS}"
    mode_color = (0, 255, 150) if track.locked else (0, 200, 255)

    cv2.putText(img, f"MODE:  {mode}",                                     (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color,      1)
    cv2.putText(img, f"speed: {speed if speed else '--'}",                 (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150),   1)
    cv2.putText(img, f"time:  {f'{lap_time:.2f}s' if lap_time else '--'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(img, f"best:  {f'{best_lap:.2f}s' if best_lap else '--'}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255),   1)

    if frame is not None:
        cam_panel = cv2.resize(frame, (canvas_size, canvas_size))
    else:
        cam_panel = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        cv2.putText(cam_panel, "no frame", (canvas_size//2 - 50, canvas_size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    return np.hstack([img, cam_panel])

# ─── Track builder ────────────────────────────────────────────────────────────

class TrackBuilder:
    def __init__(self):
        self.points      = []
        self.spline      = None
        self.curvature   = None
        self.lap_count   = 0
        self.locked      = False
        self._near_start = False

    def add_point(self, x, y):
        if self.locked:
            return

        pt = (x, y)

        if self.points:
            last = self.points[-1]
            if np.hypot(pt[0]-last[0], pt[1]-last[1]) < TRACK_POINT_MIN_DIST:
                return

        self.points.append(pt)
        n = len(self.points)

        if n < TRACK_MIN_POINTS:
            return

        dist_to_start = np.hypot(pt[0]-self.points[0][0], pt[1]-self.points[0][1])
        if dist_to_start < LOOP_CLOSE_DIST and not self._near_start:
            self._near_start = True
            self.lap_count += 1
            print(f"[lap] Lap {self.lap_count} complete | {n} points total")

            try:
                self.spline    = fit_spline(self.points)
                self.curvature = compute_curvature_map(self.spline)
                print(f"[spline] Refit with {n} points (outliers filtered)")
            except Exception as e:
                print(f"[spline] fit failed: {e}")

            if self.lap_count >= MAPPING_LAPS:
                self.locked = True
                print(f"[track] {MAPPING_LAPS} mapping laps done — switching to RACE mode!")

        elif dist_to_start > LOOP_CLOSE_DIST * 2:
            self._near_start = False

# ─── Shared state (init once at startup) ──────────────────────────────────────

class AppState:
    """Holds all long-lived objects so process_frame can reuse them."""
    def __init__(self):
        # Serial
        self.arduino = serial.Serial(port='COM6', baudrate=115200, timeout=.1)
        self._write_serial("130")

        # Calibration
        calib = np.load("camera_calibration.npz")
        self.camera_matrix = calib["camera_matrix"]
        self.dist_coeffs   = calib["dist_coeffs"]

        # Camera
        manager = gx.DeviceManager()
        dev_num, _ = manager.update_device_list()
        if dev_num == 0:
            raise RuntimeError("No camera found")

        self.cam            = manager.open_device_by_index(1)
        self.image_convert  = manager.create_image_format_convert()
        remote              = self.cam.get_remote_device_feature_control()

        remote.get_enum_feature("TriggerMode").set("Off")
        remote.get_enum_feature("ExposureAuto").set("Off")
        remote.get_enum_feature("GainAuto").set("Off")
        remote.get_float_feature("ExposureTime").set(EXPOSURE)
        remote.get_float_feature("Gain").set(GAIN)

        self.cam.stream_on()

        # ArUco
        aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params        = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        # solvePnP marker corner geometry (once)
        half = MARKER_SIZE / 2
        self.obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        # Runtime state
        self.track           = TrackBuilder()
        self.lap_count       = 0
        self.lap_start_time  = time.time()
        self.last_lap_time   = None
        self.best_lap_time   = None
        self._was_near_start = False

        print("=" * 50)
        print(f"MODE: Mapping — {MAPPING_LAPS} slow laps to build track")
        print("=" * 50)

    def _write_serial(self, x):
        self.arduino.write(bytes(x, 'utf-8'))
        time.sleep(0.002)

    def write_serial(self, x):
        self._write_serial(x)


# Populated in main()
state: AppState | None = None


# ==============================================================================
# 1. LOGIC CODE
# ==============================================================================

def process_frame(_ignored=None):
    """Grab from GigE cam, run pipeline, return combined debug view (BGR)."""
    global state
    if state is None:
        return None

    raw = state.cam.data_stream[0].get_image()
    if raw is None:
        return draw_debug(state.track, None, frame=None,
                          lap_count=state.lap_count,
                          lap_time=state.last_lap_time,
                          best_lap=state.best_lap_time)

    cam_frame = convert_to_rgb(state.image_convert, raw)
    cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)
    cam_frame = cv2.rotate(cam_frame, cv2.ROTATE_180)
    gray      = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = state.detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(cam_frame, corners, ids)

    if ids is None:
        return draw_debug(state.track, None, frame=cam_frame,
                          lap_count=state.lap_count,
                          lap_time=state.last_lap_time,
                          best_lap=state.best_lap_time)

    # solvePnP per marker
    rvecs, tvecs = [], []
    for c in corners:
        _, rvec, tvec = cv2.solvePnP(
            state.obj_points, c[0],
            state.camera_matrix, state.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        rvecs.append(rvec)
        tvecs.append(tvec)

    transforms = {}
    for i, mid in enumerate(ids.flatten()):
        transforms[mid] = pose_to_matrix(rvecs[i], tvecs[i])

    if MARKER_REFERENCE not in transforms or MARKER_CAR not in transforms:
        return draw_debug(state.track, None, frame=cam_frame,
                          lap_count=state.lap_count,
                          lap_time=state.last_lap_time,
                          best_lap=state.best_lap_time)

    T_ref_inv = np.linalg.inv(transforms[MARKER_REFERENCE])
    T_rel     = T_ref_inv @ transforms[MARKER_CAR]
    _, t_rel  = matrix_to_pose(T_rel)
    x, y, z   = t_rel.flatten()

    # Lap timing
    if state.track.points:
        dist_to_start = np.hypot(x - state.track.points[0][0],
                                 y - state.track.points[0][1])
    else:
        dist_to_start = 999

    if dist_to_start < LOOP_CLOSE_DIST and not state._was_near_start and state.lap_count > 0:
        state._was_near_start = True
        state.last_lap_time   = time.time() - state.lap_start_time
        state.lap_start_time  = time.time()
        state.lap_count      += 1
        if state.best_lap_time is None or (state.last_lap_time < state.best_lap_time and state.last_lap_time > 0.5):
            state.best_lap_time = state.last_lap_time
        print(f"[lap] #{state.lap_count} | time={state.last_lap_time:.2f}s | best={state.best_lap_time:.2f}s")
    elif dist_to_start > LOOP_CLOSE_DIST * 2:
        state._was_near_start = False

    prev_lap_count = state.track.lap_count
    state.track.add_point(x, y)

    if state.track.lap_count == 1 and prev_lap_count == 0:
        state.lap_start_time = time.time()
        state.lap_count = 1

    # Mapping phase
    if not state.track.locked:
        state.write_serial(MAPPING_SPEED)
        return draw_debug(state.track, (x, y), frame=cam_frame,
                          lap_count=state.lap_count,
                          lap_time=state.last_lap_time,
                          best_lap=state.best_lap_time)

    # Race phase
    if state.track.spline is not None:
        progress, _ = project_onto_spline(state.track.spline, (x, y))
        speed, _    = speed_for_position(state.track.spline, progress, state.track.curvature)
        state.write_serial(str(speed))
        return draw_debug(state.track, (x, y), frame=cam_frame,
                          speed=speed, progress=progress,
                          lap_count=state.lap_count,
                          lap_time=state.last_lap_time,
                          best_lap=state.best_lap_time)

    return draw_debug(state.track, (x, y), frame=cam_frame,
                      lap_count=state.lap_count,
                      lap_time=state.last_lap_time,
                      best_lap=state.best_lap_time)


# ==============================================================================
# 2. EVENT BINDINGS / FRAME LOOP
# ==============================================================================

def attach_events(ui):
    """Drive the frame loop with a QTimer at ~30 fps."""
    def tick():
        img = process_frame()
        if img is not None:
            ui['page_0']['image'].update_image(img)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(33)  # ~30 fps
    ui['_timer'] = timer  # keep reference alive


# ==============================================================================
# 3. MAIN FUNCTION
# ==============================================================================

def main():
    global state
    app = pv.PvApp()
    ui  = create_ui()

    state = AppState()

    attach_events(ui)
    ui['window'].show()
    try:
        app.run()
    finally:
        if state is not None:
            state.cam.stream_off()
            state.cam.close_device()


if __name__ == '__main__':
    main()