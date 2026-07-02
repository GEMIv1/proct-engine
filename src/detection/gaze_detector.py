import cv2
import numpy as np
from collections import deque
from mediapipe.python.solutions import face_mesh

from models import GazeDirection, GazeState, HeadPose, PupilPosition, AppConfig, EyeDetectionConfig
from .base import BaseDetector

class _LandmarkIndex:
    LEFT_EYE_IDX = [33, 133]
    RIGHT_EYE_IDX = [362, 263]
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    RIGHT_TOP = 159
    RIGHT_BOTTOM = 145
    LEFT_TOP = 386
    LEFT_BOTTOM = 374

class GazeDetector(BaseDetector):
    def __init__(self, config: AppConfig = None, smoothing: int = 5):
        """Initialize MediaPipe FaceMesh with performance optimizations.

        Parameters
        ----------
        config : AppConfig, optional
            The configuration object.
        smoothing : int
            Window size of the moving average for iris positions to reduce jitter.
        """
        self.face_mesh = face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.buffer_x = deque(maxlen=smoothing)
        self.buffer_y = deque(maxlen=smoothing)
        self.calibrated = False
        self.baseline_x = 0.5
        self.baseline_y = 0.5
        self.last_good_state = GazeState()
        self.frames_since_detection = 0

        # Store configuration or fallback to defaults
        self.eyes_config = config.detection.eyes if config else EyeDetectionConfig()
        
        # Load from config or EyeDetectionConfig defaults
        self.iris_up_thresh = self.eyes_config.iris_up_thresh
        self.iris_down_thresh = self.eyes_config.iris_down_thresh
        self.gaze_left_thresh = self.eyes_config.gaze_left_thresh
        self.gaze_right_thresh = self.eyes_config.gaze_right_thresh
        self.iris_pupil_blend = self.eyes_config.iris_pupil_blend
        self.head_pose_scale = self.eyes_config.head_pose_scale
        self.max_frames_without_detection = self.eyes_config.max_frames_without_detection
        self.max_frame_dim = self.eyes_config.max_frame_dim

    def close(self):
        """Close MediaPipe FaceMesh resources."""
        self.face_mesh.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, frame: np.ndarray) -> bool:
        """Calibrate baseline using current frame (user looks at the camera/screen).

        Returns True if a non-uncertain gaze was obtained and baseline stored.
        """
        state = self.process(frame)
        if state.gaze != GazeDirection.UNCERTAIN:
            self.baseline_x = state.pupil_rel.x
            self.baseline_y = state.pupil_rel.y
            self.calibrated = True
            return True
        return False

    def process(self, frame: np.ndarray, **kwargs) -> GazeState:
        """Process a BGR frame and return a GazeState."""
        h, w = frame.shape[:2]
        frame_small, scale_back = self._downscale_frame(frame)
        small_h, small_w = frame_small.shape[:2]

        results = self.face_mesh.process(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_frames_without_detection:
                return GazeState()
            return self.last_good_state

        self.frames_since_detection = 0
        lm = results.multi_face_landmarks[0].landmark

        l_min, l_max, l_center, r_min, r_max, r_center = self._extract_eye_coords(
            lm, small_w, small_h, scale_back
        )

        lx, ly = self._normalize(l_center, l_min, l_max)
        rx, ry = self._normalize(r_center, r_min, r_max)
        nx = (lx + rx) / 2
        ny = (ly + ry) / 2

        # Blend iris-relative position with absolute pupil position
        pupil_x = (l_center[0] + r_center[0]) / 2 / max(w, 1)
        pupil_y = (l_center[1] + r_center[1]) / 2 / max(h, 1)
        nx = self.iris_pupil_blend * nx + (1 - self.iris_pupil_blend) * pupil_x
        ny = self.iris_pupil_blend * ny + (1 - self.iris_pupil_blend) * pupil_y

        nx, ny = self._smooth_and_calibrate(nx, ny)
        iris_vert_ratio = self._calc_iris_vertical_ratio(lm, small_h, scale_back)
        gaze, gaze_conf = self._classify_gaze(nx, ny, iris_vert_ratio)

        head_pose = HeadPose(
            pitch=float((0.5 - ny) * self.head_pose_scale),
            yaw=float((nx - 0.5) * self.head_pose_scale),
            roll=0.0,
        )

        result = GazeState(
            gaze=gaze,
            gaze_conf=float(np.clip(gaze_conf, 0.0, 1.0)),
            head_pose=head_pose,
            pupil_rel=PupilPosition(x=float(nx), y=float(ny)),
        )
        self.last_good_state = result
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _downscale_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Downscale frame so its longest edge is at most max_frame_dim.

        Returns the (possibly downscaled) frame and the inverse scale factor.
        """
        h, w = frame.shape[:2]
        if max(h, w) > self.max_frame_dim:
            scale = self.max_frame_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(frame, (new_w, new_h)), max(h, w) / self.max_frame_dim
        return frame, 1.0

    def _lm_to_pixel(
        self, lm_point, small_w: int, small_h: int, scale_back: float
    ) -> tuple[int, int]:
        """Convert a normalized MediaPipe landmark to original-frame pixel coordinates."""
        return (
            int(lm_point.x * small_w * scale_back),
            int(lm_point.y * small_h * scale_back),
        )

    def _extract_eye_coords(
        self, lm, small_w: int, small_h: int, scale_back: float
    ) -> tuple:
        """Extract pixel coordinates for both eye corners and iris centers."""
        l_min    = self._lm_to_pixel(lm[_LandmarkIndex.LEFT_EYE_IDX[0]],    small_w, small_h, scale_back)
        l_max    = self._lm_to_pixel(lm[_LandmarkIndex.LEFT_EYE_IDX[1]],    small_w, small_h, scale_back)
        l_center = self._lm_to_pixel(lm[_LandmarkIndex.LEFT_IRIS_CENTER],    small_w, small_h, scale_back)
        r_min    = self._lm_to_pixel(lm[_LandmarkIndex.RIGHT_EYE_IDX[0]],   small_w, small_h, scale_back)
        r_max    = self._lm_to_pixel(lm[_LandmarkIndex.RIGHT_EYE_IDX[1]],   small_w, small_h, scale_back)
        r_center = self._lm_to_pixel(lm[_LandmarkIndex.RIGHT_IRIS_CENTER],   small_w, small_h, scale_back)
        return l_min, l_max, l_center, r_min, r_max, r_center

    def _normalize(self, c, mn, mx) -> tuple[float, float]:
        """Normalize point c into [0, 1] using min (mn) and max (mx) corners."""
        return (
            (c[0] - mn[0]) / (mx[0] - mn[0] + 1e-6),
            (c[1] - mn[1]) / (mx[1] - mn[1] + 1e-6),
        )

    def _smooth_and_calibrate(self, nx: float, ny: float) -> tuple[float, float]:
        """Apply moving-average smoothing and subtract calibration baseline."""
        self.buffer_x.append(nx)
        self.buffer_y.append(ny)
        nx = float(np.mean(self.buffer_x))
        ny = float(np.mean(self.buffer_y))
        if self.calibrated:
            nx -= self.baseline_x - 0.5
            ny -= self.baseline_y - 0.5
        return nx, ny

    def _calc_iris_vertical_ratio(self, lm, small_h: int, scale_back: float) -> float:
        """Calculate vertical iris position ratio within the eyelid opening for both eyes."""
        r_iris_y   = lm[_LandmarkIndex.RIGHT_IRIS_CENTER].y * small_h * scale_back
        r_top_y    = lm[_LandmarkIndex.RIGHT_TOP].y          * small_h * scale_back
        r_bottom_y = lm[_LandmarkIndex.RIGHT_BOTTOM].y       * small_h * scale_back

        l_iris_y   = lm[_LandmarkIndex.LEFT_IRIS_CENTER].y  * small_h * scale_back
        l_top_y    = lm[_LandmarkIndex.LEFT_TOP].y           * small_h * scale_back
        l_bottom_y = lm[_LandmarkIndex.LEFT_BOTTOM].y        * small_h * scale_back

        r_ratio = (r_iris_y - r_top_y) / (r_bottom_y - r_top_y + 1e-6)
        l_ratio = (l_iris_y - l_top_y) / (l_bottom_y - l_top_y + 1e-6)
        return (r_ratio + l_ratio) / 2.0

    def _classify_gaze(
        self, nx: float, ny: float, iris_vert_ratio: float
    ) -> tuple[GazeDirection, float]:
        """Map smoothed gaze coordinates to a GazeDirection and confidence score."""
        v_gaze = GazeDirection.ON_SCREEN
        v_conf = 0.6

        if iris_vert_ratio > self.iris_up_thresh:
            v_gaze = GazeDirection.UP
            v_conf = min(1.0, (iris_vert_ratio - self.iris_up_thresh) * 2 + 0.6)
        elif iris_vert_ratio < self.iris_down_thresh:
            v_gaze = GazeDirection.DOWN
            v_conf = min(1.0, (self.iris_down_thresh - iris_vert_ratio) * 2 + 0.6)

        h_gaze = GazeDirection.ON_SCREEN
        h_conf = 0.6

        if nx < self.gaze_left_thresh:
            h_gaze = GazeDirection.OFF_LEFT
            h_conf = min(1.0, (self.gaze_left_thresh - nx) * 5 + 0.6)
        elif nx > self.gaze_right_thresh:
            h_gaze = GazeDirection.OFF_RIGHT
            h_conf = min(1.0, (nx - self.gaze_right_thresh) * 5 + 0.6)

        # Pick the direction with the higher confidence if looking away, or default to ON_SCREEN
        if v_gaze != GazeDirection.ON_SCREEN and h_gaze != GazeDirection.ON_SCREEN:
            if h_conf >= v_conf:
                return h_gaze, h_conf
            else:
                return v_gaze, v_conf
        elif h_gaze != GazeDirection.ON_SCREEN:
            return h_gaze, h_conf
        else:
            return v_gaze, v_conf