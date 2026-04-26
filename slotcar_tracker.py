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

# ─── Config ───────────────────────────────────────────────────────────────────

arduino = serial.Serial(port='COM6', baudrate=115200, timeout=.1)

MARKER_REFERENCE = 10       # fixed reference marker on track
MARKER_CAR       = 1        # car marker
MARKER_SIZE      = 0.05     # meters

# Minimum distance (m) between recorded track points
TRACK_POINT_MIN_DIST = 0.02

# How many points before we try fitting the spline
TRACK_MIN_POINTS = 50

# How close the car must return to start (m) to "close" the loop
LOOP_CLOSE_DIST = 0.08

SPEED_MIN = 135   # slowest (tight corner)
SPEED_MAX = 225   # fastest (straight)

# How far ahead on the spline to look for corners (0.0–1.0 of lap)
LOOKAHEAD = 0.05

# Number of slow mapping laps before switching to race mode
MAPPING_LAPS = 3
MAPPING_SPEED = "130"

# ─── Serial ───────────────────────────────────────────────────────────────────

def write_serial(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.002)

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
    """Remove points that are too far from their local neighbours."""
    pts = np.array(points)
    # Distance from each point to the next
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    mean, std = dists.mean(), dists.std()
    # Keep a point if the distance to its neighbour is within z_thresh std devs
    keep = np.ones(len(pts), dtype=bool)
    for i, d in enumerate(dists):
        if d > mean + z_thresh * std:
            keep[i + 1] = False  # mark the outlier point
    return pts[keep].tolist()

def fit_spline(points):
    """Filter outliers then fit a smooth closed 2D spline through XY points."""
    clean = filter_outliers(points)
    pts = np.array(clean)
    x, y = pts[:, 0], pts[:, 1]
    # Higher s = smoother spline (less sensitive to noise)
    tck, _ = splprep([x, y], s=0.05, per=True, k=3)
    return tck

def sample_spline(tck, n=500):
    """Return evenly sampled points along the spline."""
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)

def project_onto_spline(tck, point, n=500):
    """
    Find where `point` (x,y) is on the spline.
    Returns progress 0.0 → 1.0 and the distance to the spline.
    """
    samples = sample_spline(tck, n)
    dists = np.linalg.norm(samples - np.array(point[:2]), axis=1)
    idx = np.argmin(dists)
    progress = idx / n
    return progress, dists[idx]

def compute_curvature_map(tck, n=500):
    """
    Compute curvature at each of n points along the spline.
    Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^1.5
    Returns array of n curvature values.
    """
    u = np.linspace(0, 1, n)
    dx, dy   = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

def speed_for_position(tck, progress, curvature_map, n=500):
    """
    Look ahead from current progress by LOOKAHEAD fraction of the lap.
    Take the max curvature in that window → map to speed 111–200.
    High curvature = corner = slow. Low curvature = straight = fast.
    """
    idx      = int(progress * n)
    ahead    = int(LOOKAHEAD * n)
    indices  = [(idx + i) % n for i in range(ahead)]
    max_curv = curvature_map[indices].max()

    # Clamp curvature to a sane range (tune if needed)
    curv_min, curv_max = 0.0, 20.0
    t = np.clip((max_curv - curv_min) / (curv_max - curv_min), 0.0, 1.0)

    # Invert: high curvature → low speed
    speed = int(SPEED_MAX - t * (SPEED_MAX - SPEED_MIN))
    return speed, max_curv

# ─── Debug view ───────────────────────────────────────────────────────────────

def draw_debug(track, car_xy, frame=None, speed=None, progress=None,
               lap_count=0, lap_time=None, best_lap=None,
               canvas_size=600, scale=200):
    """
    Left panel:  top-down spline view
    Right panel: camera frame with aruco detection
    """
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    cx, cy = canvas_size // 2, canvas_size // 2

    def to_px(x, y):
        return int(cx + x * scale), int(cy - y * scale)

    # Draw raw recorded points (dim white)
    for pt in track.points:
        px, py = to_px(*pt)
        cv2.circle(img, (px, py), 2, (60, 60, 60), -1)

    # Draw spline, colored by curvature (green=fast, red=corner)
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

    # Draw car position
    if car_xy is not None:
        px, py = to_px(*car_xy)
        cv2.circle(img, (px, py), 8, (255, 255, 0), -1)

    # HUD text — hardcoded positions, nothing shifts
    mode       = "RACE" if track.locked else f"MAPPING {track.lap_count+1}/{MAPPING_LAPS}"
    mode_color = (0, 255, 150) if track.locked else (0, 200, 255)

    cv2.putText(img, f"MODE:  {mode}",                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color,      1)
    cv2.putText(img, f"speed: {speed if speed else '--'}",    (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150),   1)
    cv2.putText(img, f"time:  {f'{lap_time:.2f}s' if lap_time else '--'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(img, f"best:  {f'{best_lap:.2f}s' if best_lap else '--'}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255),   1)

    # Right panel: camera frame
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
        self.points       = []       # all recorded (x, y) positions
        self.spline       = None     # fitted spline tck
        self.curvature    = None     # curvature map
        self.lap_count    = 0        # completed laps
        self.locked       = False    # True after MAPPING_LAPS completed
        self._near_start  = False    # debounce for start crossing

    def add_point(self, x, y):
        if self.locked:
            return

        pt = (x, y)

        # Enforce minimum distance between points
        if self.points:
            last = self.points[-1]
            if np.hypot(pt[0]-last[0], pt[1]-last[1]) < TRACK_POINT_MIN_DIST:
                return

        self.points.append(pt)
        n = len(self.points)

        if n < TRACK_MIN_POINTS:
            return

        # ── Lap detection ──
        # Debounce: only trigger once per crossing
        dist_to_start = np.hypot(pt[0]-self.points[0][0], pt[1]-self.points[0][1])
        if dist_to_start < LOOP_CLOSE_DIST and not self._near_start:
            self._near_start = True
            self.lap_count += 1
            lap_pts = n
            print(f"[lap] Lap {self.lap_count} complete | {lap_pts} points total")

            # Refit spline with all data so far
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
            self._near_start = False  # reset once car moves away from start

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    EXPOSURE = 2000.0
    GAIN     = 20.0

    calib         = np.load("camera_calibration.npz")
    camera_matrix = calib["camera_matrix"]
    dist_coeffs   = calib["dist_coeffs"]

    manager = gx.DeviceManager()
    dev_num, _ = manager.update_device_list()
    if dev_num == 0:
        print("No camera found")
        return

    write_serial("130")

    cam          = manager.open_device_by_index(1)
    image_convert = manager.create_image_format_convert()
    remote        = cam.get_remote_device_feature_control()

    remote.get_enum_feature("TriggerMode").set("Off")
    remote.get_enum_feature("ExposureAuto").set("Off")
    remote.get_enum_feature("GainAuto").set("Off")
    remote.get_float_feature("ExposureTime").set(EXPOSURE)
    remote.get_float_feature("Gain").set(GAIN)

    cam.stream_on()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params     = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

    track = TrackBuilder()

    lap_count      = 0
    lap_start_time = time.time()
    last_lap_time  = None
    best_lap_time  = None
    _was_near_start = False

    print("=" * 50)
    print(f"MODE: Mapping — {MAPPING_LAPS} slow laps to build track")
    print("=" * 50)

    while True:
        start_timer = timeit.default_timer()

        raw = cam.data_stream[0].get_image()
        if raw is None:
            continue

        frame = convert_to_rgb(image_convert, raw)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Always show debug with latest frame
        debug = draw_debug(track, None, frame=frame,
                           lap_count=lap_count, lap_time=last_lap_time, best_lap=best_lap_time)
        cv2.imshow("debug", debug)
        cv2.waitKey(1)

        if ids is None:
            continue

        half = MARKER_SIZE / 2
        obj_points = np.array([
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        rvecs, tvecs = [], []
        for c in corners:
            _, rvec, tvec = cv2.solvePnP(
                obj_points, c[0],
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            rvecs.append(rvec)
            tvecs.append(tvec)

        # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        #     corners, MARKER_SIZE, camera_matrix, dist_coeffs
        # )

        transforms = {}
        for i, mid in enumerate(ids.flatten()):
            transforms[mid] = pose_to_matrix(rvecs[i], tvecs[i])

        if MARKER_REFERENCE not in transforms or MARKER_CAR not in transforms:
            continue

        T_ref_inv = np.linalg.inv(transforms[MARKER_REFERENCE])
        T_rel     = T_ref_inv @ transforms[MARKER_CAR]
        _, t_rel  = matrix_to_pose(T_rel)
        x, y, z   = t_rel.flatten()

        # ── Lap timing (works in both mapping and race mode) ──
        dist_to_start = np.hypot(x - track.points[0][0], y - track.points[0][1]) if track.points else 999
        if dist_to_start < LOOP_CLOSE_DIST and not _was_near_start and lap_count > 0:
            _was_near_start = True
            last_lap_time = time.time() - lap_start_time
            lap_start_time = time.time()
            lap_count += 1
            if best_lap_time is None or (last_lap_time < best_lap_time and last_lap_time > 0.5):
                best_lap_time = last_lap_time
            print(f"[lap] #{lap_count} | time={last_lap_time:.2f}s | best={best_lap_time:.2f}s")
        elif dist_to_start > LOOP_CLOSE_DIST * 2:
            _was_near_start = False

        # Record always
        prev_lap_count = track.lap_count
        track.add_point(x, y)

        # Start lap timer after first lap closes
        if track.lap_count == 1 and prev_lap_count == 0:
            lap_start_time = time.time()
            lap_count = 1

        # ── Mapping phase ──
        if not track.locked:
            write_serial(MAPPING_SPEED)
            debug = draw_debug(track, (x, y), frame=frame,
                               lap_count=lap_count, lap_time=last_lap_time, best_lap=best_lap_time)

        # ── Race phase ──
        else:
            if track.spline is not None:
                progress, dist_to_spline = project_onto_spline(track.spline, (x, y))
                speed, max_curv = speed_for_position(track.spline, progress, track.curvature)

                write_serial(str(speed))

                print(
                    f"[race] progress={progress:.2f} | "
                    f"curvature={max_curv:.2f} | "
                    f"speed={speed}"
                )
                debug = draw_debug(track, (x, y), frame=frame, speed=speed, progress=progress,
                                   lap_count=lap_count, lap_time=last_lap_time, best_lap=best_lap_time)

        cv2.imshow("debug", debug)
        cv2.waitKey(1)

        cycle_ms = (timeit.default_timer() - start_timer) * 1e3
        # print(f"[cycle] {cycle_ms:.1f} ms")

    cam.stream_off()
    cam.close_device()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()