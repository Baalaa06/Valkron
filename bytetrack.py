import cv2
import numpy as np
import torch
import time
import json
import pandas as pd
import os
import pickle
import shutil
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.linalg

from ultralytics import YOLO


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space (x, y, w, h, vx, vy, vw, vh) contains
    the bounding box center position (x, y), width w, height h,
    and their respective velocities.
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        projected_mean, projected_cov = self.project(mean, covariance)

        try:
            chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        except scipy.linalg.LinAlgError:
            # Fallback to pseudo-inverse if Cholesky decomposition fails
            kalman_gain = np.dot(covariance, self._update_mat.T).dot(np.linalg.pinv(projected_cov))

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class STrack:
    """Single target track with Kalman filter state."""

    # Shared class variables
    _count = 0

    def __init__(self, tlwh, score, cls_id=None, detection_data=None):
        # Wait for activation
        self.kalman_filter = KalmanFilter()
        self.track_id = 0
        self.is_activated = False
        self.state = 'Tracked'  # 'Tracked', 'Lost', or 'Removed'
        self.history = []

        self.tlwh = np.asarray(tlwh, dtype=np.float32).copy()
        self.score = score
        self.cls_id = cls_id if cls_id is not None else 0
        self.detection_data = detection_data or {}

        self.frame_id = 0
        self.tracklet_len = 0
        self.time_since_update = 0

        # Kalman filter state
        self.mean, self.covariance = None, None

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self.tlwh))

        self.tracklet_len = 0
        self.state = 'Tracked'
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.tlwh = new_track.tlwh
        self.score = new_track.score
        self.cls_id = new_track.cls_id
        self.detection_data = new_track.detection_data

        self.tracklet_len = 0
        self.state = 'Tracked'
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def predict(self):
        """Predict next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != 'Tracked':
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def update(self, new_track, frame_id):
        """Update track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = 'Tracked'
        self.is_activated = True

        self.score = new_track.score
        self.cls_id = new_track.cls_id
        self.detection_data = new_track.detection_data

    def mark_lost(self):
        """Mark track as lost."""
        self.state = 'Lost'

    def mark_removed(self):
        """Mark track as removed."""
        self.state = 'Removed'

    @staticmethod
    def next_id():
        """Generate next track ID."""
        STrack._count += 1
        return STrack._count

    def end_frame(self):
        """End current frame."""
        self.frame_id += 1

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to (center_x, center_y, aspect_ratio, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]  # aspect ratio
        return ret

    def to_xyah(self):
        """Get current position in (center_x, center_y, aspect_ratio, height) format."""
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def xyah_to_tlwh(xyah):
        """Convert (center_x, center_y, aspect_ratio, height) to (top_left_x, top_left_y, width, height)."""
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]  # width = aspect_ratio * height
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlwh(self):
        """Get current position in (top_left_x, top_left_y, width, height) format."""
        if self.mean is None:
            return self.tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_ltrb(self):
        """Get current position in (left, top, right, bottom) format."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def is_confirmed(self):
        """Check if track is confirmed (activated)."""
        return self.is_activated

    def __repr__(self):
        return f'OT_{self.track_id}_({self.state})'


def matching_distance(tracks, detections, track_indices=None, detection_indices=None):
    """
    Compute distance matrix between tracks and detections using IoU.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return cost_matrix
        
    for row, track_idx in enumerate(track_indices):
        try:
            if tracks[track_idx].time_since_update > 1:
                cost_matrix[row, :] = 1e+5
                continue

            bbox = tracks[track_idx].to_tlwh()
            candidates = np.asarray([detections[i].tlwh for i in detection_indices])
            
            if candidates.size == 0:
                cost_matrix[row, :] = 1e+5
                continue
                
            ious = bbox_ious(bbox[None, :], candidates)
            if ious.size > 0 and ious.shape[1] > 0:
                cost_matrix[row, :] = 1.0 - ious[0]
            else:
                cost_matrix[row, :] = 1e+5
        except Exception:
            cost_matrix[row, :] = 1e+5

    return cost_matrix


def bbox_ious(atlbrs, btlbrs):
    """
    Compute IoU between two sets of bounding boxes.
    """
    try:
        atlbrs = np.asarray(atlbrs)
        btlbrs = np.asarray(btlbrs)

        if atlbrs.size == 0 or btlbrs.size == 0:
            return np.array([[0.0]])

        if atlbrs.ndim == 1:
            atlbrs = atlbrs[None, :]
        if btlbrs.ndim == 1:
            btlbrs = btlbrs[None, :]
            
        if atlbrs.shape[1] < 4 or btlbrs.shape[1] < 4:
            return np.zeros((atlbrs.shape[0], btlbrs.shape[0]))

        al, at, aw, ah = atlbrs[:, 0], atlbrs[:, 1], atlbrs[:, 2], atlbrs[:, 3]
        ar, ab = al + aw, at + ah

        bl, bt, bw, bh = btlbrs[:, 0], btlbrs[:, 1], btlbrs[:, 2], btlbrs[:, 3]
        br, bb = bl + bw, bt + bh

        left = np.maximum(al[:, None], bl)
        top = np.maximum(at[:, None], bt)
        right = np.minimum(ar[:, None], br)
        bottom = np.minimum(ab[:, None], bb)

        intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        area_a = (ar - al) * (ab - at)
        area_b = (br - bl) * (bb - bt)
        union = area_a[:, None] + area_b - intersection

        iou = intersection / np.maximum(union, 1e-6)
        return iou
        
    except Exception:
        return np.array([[0.0]])


def linear_assignment(cost_matrix, thresh):
    """
    Perform linear assignment using Hungarian algorithm.
    """
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = []
    unmatched_b = []

    for i in range(cost_matrix.shape[0]):
        if i not in row_indices:
            unmatched_a.append(i)

    for j in range(cost_matrix.shape[1]):
        if j not in col_indices:
            unmatched_b.append(j)

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] > thresh:
            unmatched_a.append(row)
            unmatched_b.append(col)
        else:
            matches.append([row, col])

    return np.array(matches), tuple(unmatched_a), tuple(unmatched_b)


class ByteTracker:
    """
    ByteTrack multi-object tracker implementation.
    """

    def __init__(self, max_age=30, frame_rate=30, track_thresh=0.4, match_thresh=0.8, 
                 high_thresh=0.5, low_thresh=0.01):
        """
        Initialize ByteTrack with parameters from the paper.
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            frame_rate: Video frame rate for time-based calculations
            track_thresh: Detection confidence for initializing tracks (lowered to 0.4)
            match_thresh: IoU threshold for matching (paper: 0.8)
            high_thresh: High detection confidence (lowered to 0.5)
            low_thresh: Low detection confidence (paper: 0.1)

        Note: Thresholds adjusted to be more permissive for better detection of 
        hard-to-detect objects while still maintaining tracking quality.
        """
        self.max_age = int(frame_rate / 30.0 * max_age)
        self.kalman_filter = KalmanFilter()

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh  # IoU matching threshold
        self.high_thresh = high_thresh    # High detection confidence
        self.low_thresh = low_thresh      # Low detection confidence

        self.frame_id = 0
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

    def update(self, detections):
        """
        Update tracker with new detections using ByteTrack algorithm.

        Args:
            detections: List of detections, each detection should be 
                      [x1, y1, x2, y2, score, class_id] or similar format.
                      This follows the ByteTrack paper's input format.
        """
        self.frame_id += 1  # Increment internal frame counter

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(detections) == 0:
            # Handle case with no detections
            for track in self.tracked_stracks:
                track.predict()
                track.time_since_update += 1
                if track.time_since_update > self.max_age:
                    track.mark_removed()
                    removed_stracks.append(track)
                else:
                    track.mark_lost()
                    lost_stracks.append(track)

            self.tracked_stracks = []
            self.lost_stracks.extend(lost_stracks)
            self.removed_stracks.extend(removed_stracks)

            return self.tracked_stracks

        # Convert detections to STrack format
        detection_stracks = []
        for det in detections:
            if len(det) >= 6:  # [x1, y1, x2, y2, score, class_id, ...]
                x1, y1, x2, y2, score, cls_id = det[:6]
                detection_data = det[6] if len(det) > 6 else {}
            elif len(det) >= 5:  # [x1, y1, x2, y2, score]
                x1, y1, x2, y2, score = det[:5]
                cls_id = 0
                detection_data = {}
            else:
                continue

            tlwh = [x1, y1, x2 - x1, y2 - y1]  # Convert to [top_left_x, top_left_y, width, height]
            detection_stracks.append(STrack(tlwh, score, cls_id, detection_data))

        # Separate high and low confidence detections
        high_dets = [track for track in detection_stracks if track.score >= self.high_thresh]
        low_dets = [track for track in detection_stracks if track.score >= self.low_thresh and track.score < self.high_thresh]

        # Step 1: Prediction
        strack_pool = self.tracked_stracks + self.lost_stracks
        for track in strack_pool:
            track.predict()

        # Step 2: First association with high confidence detections
        dists = matching_distance(strack_pool, high_dets)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = high_dets[idet]

            if track.state == 'Tracked':
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:  # re-activate lost track
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association with remaining tracks and low confidence detections  
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 'Tracked']

        if len(low_dets) > 0 and len(r_tracked_stracks) > 0:
            dists = matching_distance(r_tracked_stracks, low_dets)
            matches, u_track_remain, u_detection_second = linear_assignment(dists, thresh=0.5)  # Lower threshold for low confidence

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = low_dets[idet]
                if track.state == 'Tracked':
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track_remain:
                track = r_tracked_stracks[it]
                if track not in activated_stracks and track not in refind_stracks:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            for track in r_tracked_stracks:
                if track not in activated_stracks and track not in refind_stracks:
                    track.mark_lost()  
                    lost_stracks.append(track)

        # Step 4: Init new stracks with unmatched high confidence detections
        unmatched_dets = [high_dets[i] for i in u_detection]
        for inew in unmatched_dets:
            if inew.score >= self.track_thresh:
                inew.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(inew)

        # Step 5: Update time since update and remove dead tracks
        for track in self.lost_stracks:
            track.time_since_update += 1
            if track.time_since_update > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        # Update track lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'Tracked']
        self.tracked_stracks = activated_stracks + refind_stracks
        self.lost_stracks = [t for t in self.lost_stracks if t.state != 'Removed']
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Reset time_since_update for active tracks
        for track in self.tracked_stracks:
            track.time_since_update = 0

        return self.tracked_stracks


class VehicleTracker:
    """
    Enhanced Vehicle Tracker using YOLOv11 + ByteTrack with Local Storage

    Features:
    - Vehicle detection and tracking with custom model
    - Frame-by-frame data storage to local machine
    - Comprehensive analytics and reporting
    - Real-time visualization with performance metrics
    - Accuracy metrics: FP, FN, IDS, MOTA
    """

    def __init__(self, yolo_model, confidence=0.05, local_folder=None):
        print("Initializing Enhanced Vehicle Tracker with ByteTrack...")

        if isinstance(yolo_model, str):
            self.model = YOLO(yolo_model)
        else:
            self.model = yolo_model

        # Lower YOLO confidence threshold to get more detections for ByteTrack
        self.confidence = confidence  # Set to 0.1 to match ByteTrack paper
        self.class_names = self.model.names

        # Initialize ByteTracker with optimized parameters for maximum detection
        self.tracker = ByteTracker(
            max_age=50,  # Keep tracks longer
            frame_rate=30,
            track_thresh=0.1,  # Very low threshold for track initialization
            match_thresh=0.7,  # Slightly more permissive matching
            high_thresh=0.2,   # Lower high confidence threshold
            low_thresh=0.01    # Minimum low confidence threshold
        )

        self.vehicle_config = {
            'SMV': {'color': (255, 0, 0), 'priority': 8, 'type': 'SMALL_MILITARY_VEHICLE', 'threat_level': 'HIGH'},
            'LMV': {'color': (0, 128, 255), 'priority': 9, 'type': 'LARGE_MILITARY_VEHICLE', 'threat_level': 'VERY_HIGH'},
            'AFV': {'color': (0, 100, 0), 'priority': 10, 'type': 'ARMORED_FIGHTING_VEHICLE', 'threat_level': 'VERY_HIGH'},
            'CV': {'color': (0, 255, 255), 'priority': 6, 'type': 'CIVILIAN_VEHICLE', 'threat_level': 'LOW'},
            'MCV': {'color': (128, 128, 128), 'priority': 7, 'type': 'MILITARY_CARGO_VEHICLE', 'threat_level': 'MEDIUM'}
        }

        self.local_folder = local_folder or f"./VehicleTracking_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.setup_local_storage()

        self.stats = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'frames_processed': 0,
            'total_detections': 0,
            'active_tracks': 0,
            'max_concurrent_tracks': 0,
            'class_counts': defaultdict(int),
            'threat_level_counts': defaultdict(int),
            'processing_times': deque(maxlen=100),
            'fps_history': deque(maxlen=50),
            'start_time': time.time(),
            'track_history': defaultdict(list),
            'frame_data': [],
            'track_data': defaultdict(dict),
            'false_positives': 0,
            'false_negatives': 0,
            'identity_switches': 0,
            'total_objects': 0,

            'previous_track_ids': set(),
            'current_track_ids': set(),
        }

        self.frame_detections = []
        self.frame_tracks = []
        self.frame_analytics = []

        print(f"Vehicle Tracker Initialized with ByteTrack:")
        print(f"   • Custom Vehicle Classes: {len(self.class_names)}")
        print(f"   • Classes: {list(self.class_names.values())}")
        print(f"   • Confidence: {confidence}")
        print(f"   • Local Storage: {self.local_folder}")
        print(f"   • GPU Acceleration: {torch.cuda.is_available()}")

    def setup_local_storage(self):
        """Setup folder structure locally for data storage"""
        os.makedirs(self.local_folder, exist_ok=True)

        self.subfolders = {
            'videos': os.path.join(self.local_folder, 'processed_videos'),
            'data': os.path.join(self.local_folder, 'frame_data'),
            'analytics': os.path.join(self.local_folder, 'analytics'),
            'reports': os.path.join(self.local_folder, 'reports'),
            'visualizations': os.path.join(self.local_folder, 'visualizations'),
            'raw_data': os.path.join(self.local_folder, 'raw_tracking_data')
        }

        for folder in self.subfolders.values():
            os.makedirs(folder, exist_ok=True)

        print(f"Local storage structure created at: {self.local_folder}")

    def update_accuracy_metrics(self, current_detections, current_tracks):
        """Update accuracy metrics based on current detections and tracks"""
        for track in current_tracks:
            if hasattr(track, 'detection_data'):
                if track.detection_data.get('confidence', 0) < self.confidence * 0.7:
                    self.stats['false_positives'] += 1

        if len(current_tracks) < len(self.stats['previous_track_ids']) * 0.7:
            self.stats['false_negatives'] += (len(self.stats['previous_track_ids']) - len(current_tracks))

        current_ids = {t.track_id for t in current_tracks if hasattr(t, 'track_id')}
        previous_ids = self.stats['previous_track_ids']

        if previous_ids:
            disappeared_ids = previous_ids - current_ids
            new_ids = current_ids - previous_ids
            if disappeared_ids and new_ids:
                self.stats['identity_switches'] += min(len(disappeared_ids), len(new_ids))

        self.stats['previous_track_ids'] = current_ids

    def process_frame(self, frame, frame_number):
        """Enhanced frame processing with detailed data storage"""
        frame_start = time.time()

        detections = []
        frame_detections_data = {
            'frame_number': frame_number,
            'timestamp': time.time(),
            'detections': []
        }

        try:
            # Advanced multi-scale and multi-angle detection
            all_detections = []
            
            # 1. Multi-angle detection (more comprehensive)
            angles = [0, -5, 5, -10, 10, -15, 15, -20, 20, -25, 25]
            
            # 2. Multi-scale detection
            scales = [0.8, 1.0, 1.2]
            
            # 3. Brightness/contrast adjustments for different lighting
            brightness_factors = [0.8, 1.0, 1.2]
            contrast_factors = [0.9, 1.0, 1.1]
            
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            
            for scale in scales:
                for angle in angles:
                    for brightness in brightness_factors:
                        for contrast in contrast_factors:
                            # Skip some combinations to reduce computation
                            if scale != 1.0 and angle != 0 and brightness != 1.0 and contrast != 1.0:
                                continue
                                
                            test_frame = frame.copy()
                            
                            # Apply brightness/contrast adjustment
                            if brightness != 1.0 or contrast != 1.0:
                                test_frame = cv2.convertScaleAbs(test_frame, alpha=contrast, beta=(brightness-1)*50)
                            
                            # Apply scaling
                            if scale != 1.0:
                                new_w, new_h = int(w * scale), int(h * scale)
                                test_frame = cv2.resize(test_frame, (new_w, new_h))
                                # Pad or crop to original size
                                if scale < 1.0:
                                    pad_w, pad_h = (w - new_w) // 2, (h - new_h) // 2
                                    test_frame = cv2.copyMakeBorder(test_frame, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT)
                                else:
                                    crop_w, crop_h = (new_w - w) // 2, (new_h - h) // 2
                                    test_frame = test_frame[crop_h:crop_h+h, crop_w:crop_w+w]
                            
                            # Apply rotation
                            if angle != 0:
                                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                                test_frame = cv2.warpAffine(test_frame, M, (w, h))
                            
                            # Run detection
                            results = self.model(test_frame, conf=self.confidence, verbose=False)
                
                if (results and len(results) > 0 and 
                    hasattr(results[0], 'boxes') and results[0].boxes is not None and 
                    len(results[0].boxes) > 0):
                    
                    boxes = results[0].boxes
                    
                    for i, box in enumerate(boxes):
                        try:
                            if not hasattr(box, 'xyxy') or box.xyxy is None:
                                continue
                                
                            xyxy = box.xyxy.cpu().numpy()
                            if xyxy.size == 0 or len(xyxy.shape) < 2 or xyxy.shape[1] < 4:
                                continue
                            
                            x1, y1, x2, y2 = xyxy.flatten()[:4]
                            
                            # Transform coordinates back to original frame
                            orig_x1, orig_y1, orig_x2, orig_y2 = x1, y1, x2, y2
                            
                            # Reverse scaling transformation
                            if scale != 1.0:
                                if scale < 1.0:
                                    pad_w, pad_h = (w - int(w * scale)) // 2, (h - int(h * scale)) // 2
                                    orig_x1 = (orig_x1 - pad_w) / scale
                                    orig_y1 = (orig_y1 - pad_h) / scale
                                    orig_x2 = (orig_x2 - pad_w) / scale
                                    orig_y2 = (orig_y2 - pad_h) / scale
                                else:
                                    crop_w, crop_h = (int(w * scale) - w) // 2, (int(h * scale) - h) // 2
                                    orig_x1 = (orig_x1 + crop_w) / scale
                                    orig_y1 = (orig_y1 + crop_h) / scale
                                    orig_x2 = (orig_x2 + crop_w) / scale
                                    orig_y2 = (orig_y2 + crop_h) / scale
                            
                            # Reverse rotation transformation
                            if angle != 0:
                                M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                                corners = np.array([[orig_x1, orig_y1, 1], [orig_x2, orig_y2, 1]]).T
                                transformed = M_inv @ corners
                                orig_x1, orig_y1 = transformed[0, 0], transformed[1, 0]
                                orig_x2, orig_y2 = transformed[0, 1], transformed[1, 1]
                            
                            # Store detection with transformation info
                            all_detections.append((box, orig_x1, orig_y1, orig_x2, orig_y2, angle, scale))
                        except Exception:
                            continue
            
            # Apply Non-Maximum Suppression to remove duplicate detections
            if all_detections:
                # Group detections by coordinates for NMS
                detection_boxes = []
                detection_scores = []
                detection_data = []
                
                for box, x1, y1, x2, y2, angle, scale in all_detections:
                    try:
                        conf_array = box.conf.cpu().numpy()
                        if conf_array.size > 0:
                            conf = float(conf_array.flatten()[0])
                            detection_boxes.append([x1, y1, x2, y2])
                            detection_scores.append(conf)
                            detection_data.append((box, x1, y1, x2, y2, angle, scale))
                    except:
                        continue
                
                # Apply NMS
                if detection_boxes:
                    indices = cv2.dnn.NMSBoxes(detection_boxes, detection_scores, self.confidence, 0.4)
                    if len(indices) > 0:
                        indices = indices.flatten()
                        all_detections = [detection_data[i] for i in indices]
            
            # Process filtered detections
            for box, x1, y1, x2, y2, angle, scale in all_detections:

                try:
                        
                        # Safely extract confidence and class with checks
                        if not hasattr(box, 'conf') or box.conf is None:
                            continue
                        if not hasattr(box, 'cls') or box.cls is None:
                            continue
                            
                        conf_array = box.conf.cpu().numpy()
                        cls_array = box.cls.cpu().numpy()
                        
                        if conf_array.size == 0 or cls_array.size == 0:
                            continue
                            
                        conf = float(conf_array.flatten()[0])
                        cls_id = int(cls_array.flatten()[0])
                        cls_name = self.class_names.get(cls_id, 'unknown')

                        if conf >= self.confidence:
                            w, h = x2 - x1, y2 - y1

                            if w > 1 and h > 1:
                                vehicle_config = self.vehicle_config.get(cls_name, {
                                    'color': (128, 128, 128),
                                    'priority': 1,
                                    'type': 'UNKNOWN',
                                    'threat_level': 'UNKNOWN'
                                })

                                self.stats['total_detections'] += 1
                                self.stats['class_counts'][cls_name] += 1
                                self.stats['threat_level_counts'][vehicle_config['threat_level']] += 1

                                detection_data = {
                                    'class_id': cls_id,
                                    'class_name': cls_name,
                                    'confidence': conf,
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'center': [float((x1+x2)/2), float((y1+y2)/2)],
                                    'area': float(w * h),
                                    'color': vehicle_config['color'],
                                    'priority': vehicle_config['priority'],
                                    'vehicle_type': vehicle_config['type'],
                                    'threat_level': vehicle_config['threat_level'],
                                    'detection_time': time.time()
                                }

                                frame_detections_data['detections'].append(detection_data)
                                # Format for ByteTracker: [x1, y1, x2, y2, score, cls_id, detection_data]
                                detections.append([x1, y1, x2, y2, conf, cls_id, detection_data])
                                
                except Exception as box_error:
                    # Skip this detection if there's any error processing it
                    print(f"Warning: Skipping detection {i} in frame {frame_number}: {box_error}")
                    continue
                        
        except Exception as detection_error:
            # Handle any YOLO detection errors
            print(f"Warning: Detection error in frame {frame_number}: {detection_error}")
            # Continue with empty detections

        # Update tracks using ByteTracker with error handling
        try:
            if len(detections) == 0:
                # Pass empty list to tracker
                tracks = self.tracker.update([])
            else:
                tracks = self.tracker.update(detections)
        except Exception:
            tracks = []  # Use empty tracks if tracking fails

        self.update_accuracy_metrics(detections, tracks)

        frame_tracks_data = {
            'frame_number': frame_number,
            'timestamp': time.time(),
            'tracks': []
        }

        # Process tracks
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_ltrb()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                self.stats['track_history'][track_id].append(center)
                if len(self.stats['track_history'][track_id]) > 30:
                    self.stats['track_history'][track_id].pop(0)

                track_data = {
                    'track_id': track_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center[0]), float(center[1])],
                    'class_name': getattr(track, 'detection_data', {}).get('class_name', 'unknown'),
                    'confidence': getattr(track, 'detection_data', {}).get('confidence', track.score),
                    'vehicle_type': getattr(track, 'detection_data', {}).get('vehicle_type', 'UNKNOWN'),
                    'threat_level': getattr(track, 'detection_data', {}).get('threat_level', 'UNKNOWN'),
                    'track_age': track.time_since_update if hasattr(track, 'time_since_update') else 0,
                    'is_confirmed': track.is_confirmed()
                }

                frame_tracks_data['tracks'].append(track_data)

                if track_id not in self.stats['track_data']:
                    self.stats['track_data'][track_id] = {
                        'first_seen': frame_number,
                        'last_seen': frame_number,
                        'total_frames': 1,
                        'positions': [],
                        'class_name': track_data['class_name'],
                        'vehicle_type': track_data['vehicle_type'],
                        'threat_level': track_data['threat_level'],
                        'max_confidence': track_data['confidence']
                    }
                else:
                    self.stats['track_data'][track_id]['last_seen'] = frame_number
                    self.stats['track_data'][track_id]['total_frames'] += 1
                    self.stats['track_data'][track_id]['max_confidence'] = max(
                        self.stats['track_data'][track_id]['max_confidence'],
                        track_data['confidence']
                    )

                self.stats['track_data'][track_id]['positions'].append({
                    'frame': frame_number,
                    'center': center,
                    'bbox': [x1, y1, x2, y2]
                })

        processing_time = time.time() - frame_start
        self.stats['processing_times'].append(processing_time)
        self.stats['frames_processed'] += 1
        self.stats['active_tracks'] = len([t for t in tracks if t.is_confirmed()])
        self.stats['max_concurrent_tracks'] = max(
            self.stats['max_concurrent_tracks'],
            self.stats['active_tracks']
        )

        self.frame_detections.append(frame_detections_data)
        self.frame_tracks.append(frame_tracks_data)

        frame_analytics = {
            'frame_number': frame_number,
            'processing_time_ms': processing_time * 1000,
            'detection_count': len(frame_detections_data['detections']),
            'track_count': len([t for t in tracks if t.is_confirmed()]),
            'threat_detections': {
                'HIGH': sum(1 for d in frame_detections_data['detections'] if d['threat_level'] == 'HIGH'),
                'VERY_HIGH': sum(1 for d in frame_detections_data['detections'] if d['threat_level'] == 'VERY_HIGH'),
                'MEDIUM': sum(1 for d in frame_detections_data['detections'] if d['threat_level'] == 'MEDIUM'),
                'LOW': sum(1 for d in frame_detections_data['detections'] if d['threat_level'] == 'LOW')
            }
        }
        self.frame_analytics.append(frame_analytics)

        return tracks, detections

    def draw_tracking_results(self, frame, tracks):
        """Enhanced visualization with threat level indicators"""
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]

        for track in confirmed_tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            data = getattr(track, 'detection_data', {})
            class_name = data.get('class_name', 'unknown')
            color = data.get('color', (128, 128, 128))
            conf = data.get('confidence', track.score if hasattr(track, 'score') else 0.0)
            priority = data.get('priority', 1)
            vehicle_type = data.get('vehicle_type', 'UNKNOWN')
            threat_level = data.get('threat_level', 'UNKNOWN')

            threat_colors = {
                'VERY_HIGH': (0, 0, 255),
                'HIGH': (0, 100, 255),
                'MEDIUM': (0, 255, 255),
                'LOW': (0, 255, 0),
                'UNKNOWN': (128, 128, 128)
            }

            threat_thickness = {
                'VERY_HIGH': 4,
                'HIGH': 3,
                'MEDIUM': 2,
                'LOW': 2,
                'UNKNOWN': 1
            }

            threat_color = threat_colors.get(threat_level, (128, 128, 128))
            thickness = threat_thickness.get(threat_level, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), threat_color, thickness)

            if conf > 0:
                label = f"ID-{track_id}: {class_name} ({conf:.2f})"
            else:
                label = f"ID-{track_id}: {class_name}"

            label = f"{label} [{threat_level}]"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 15), (x1 + label_w + 10, y1 - 2), threat_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, center, 4, threat_color, -1)
            cv2.circle(frame, center, 4, (255, 255, 255), 1)

            if priority >= 8 and track_id in self.stats['track_history']:
                trail = self.stats['track_history'][track_id]
                if len(trail) > 1:
                    for i in range(1, len(trail)):
                        alpha = i / len(trail)
                        trail_color = tuple(int(c * alpha) for c in threat_color)
                        cv2.line(frame, tuple(map(int, trail[i-1])), tuple(map(int, trail[i])), trail_color, 2)

        return frame

    def add_performance_overlay(self, frame, fps=0):
        """Enhanced overlay with threat level summary"""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        overlay_height = min(300, h // 3)
        cv2.rectangle(overlay, (10, 10), (min(1200, w-10), overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        elapsed = time.time() - self.stats['start_time']
        avg_fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        avg_processing = np.mean(self.stats['processing_times']) * 1000 if self.stats['processing_times'] else 0

        mota = self.calculate_mota()

        title = "Enhanced Vehicle Tracking System - ByteTrack Algorithm"
        cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        info_lines = [
            f"Custom Model | Confidence: {self.confidence} | GPU: {torch.cuda.is_available()} | Session: {self.stats['session_id']}",
            f"Frame: {self.stats['frames_processed']} | FPS: {fps:.1f} | Avg FPS: {avg_fps:.1f} | Processing: {avg_processing:.1f}ms",
            f"Active Tracks: {self.stats['active_tracks']} | Max: {self.stats['max_concurrent_tracks']} | Total Detections: {self.stats['total_detections']}",
            f"Local Storage: {len(self.frame_detections)} frames saved | Runtime: {elapsed:.0f}s",
            f"MOTA: {mota:.2f}% | FP: {self.stats['false_positives']} | FN: {self.stats['false_negatives']} | IDS: {self.stats['identity_switches']}"
        ]

        for i, line in enumerate(info_lines):
            y = 65 + i * 25
            color = (255, 255, 255) if i > 0 else (255, 255, 0)
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        if w > 1000:
            threat_x = w - 400
            cv2.putText(frame, "Threat Analysis:", (threat_x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            threat_colors = {
                'VERY_HIGH': (0, 0, 255),
                'HIGH': (0, 100, 255),
                'MEDIUM': (0, 255, 255),
                'LOW': (0, 255, 0)
            }

            y_offset = 70
            for threat, count in self.stats['threat_level_counts'].items():
                if count > 0:
                    color = threat_colors.get(threat, (200, 200, 200))
                    cv2.putText(frame, f"{threat}: {count}", (threat_x, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                    y_offset += 25

            cv2.putText(frame, "Vehicle Classes:", (threat_x, y_offset + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 40
            for cls_name, count in list(self.stats['class_counts'].items())[:6]:
                vehicle_config = self.vehicle_config.get(cls_name, {})
                color = vehicle_config.get('color', (200, 200, 200))
                cv2.putText(frame, f"{cls_name}: {count}", (threat_x, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20

        return frame

    def calculate_mota(self):
        """Calculate Multiple Object Tracking Accuracy (MOTA)"""
        total_objects = max(1, self.stats['total_detections'])
        mota = 1 - (self.stats['false_negatives'] + self.stats['false_positives'] + self.stats['identity_switches']) / total_objects
        return max(0, mota) * 100

    def save_frame_data(self, frame_number, save_interval=50):
        """Save accumulated data locally at intervals"""
        if frame_number % save_interval == 0 and self.frame_detections:
            try:
                detections_file = os.path.join(self.subfolders['data'], f'detections_frames_{frame_number-save_interval+1}_to_{frame_number}.json')
                with open(detections_file, 'w') as f:
                    json.dump(self.frame_detections[-save_interval:], f, indent=2)

                tracks_file = os.path.join(self.subfolders['data'], f'tracks_frames_{frame_number-save_interval+1}_to_{frame_number}.json')
                with open(tracks_file, 'w') as f:
                    json.dump(self.frame_tracks[-save_interval:], f, indent=2)

                analytics_file = os.path.join(self.subfolders['analytics'], f'analytics_frames_{frame_number-save_interval+1}_to_{frame_number}.json')
                with open(analytics_file, 'w') as f:
                    json.dump(self.frame_analytics[-save_interval:], f, indent=2)

                print(f"   • Data saved locally for frames {frame_number-save_interval+1} to {frame_number}")

            except Exception as e:
                print(f"   • Warning: Could not save data locally: {e}")

    def save_final_results(self, output_video_path):
        """Save complete results and generate comprehensive report"""
        print("Saving complete results locally...")

        try:
            # Copy processed video to local storage
            local_video_path = os.path.join(self.subfolders['videos'], os.path.basename(output_video_path))
            if output_video_path != local_video_path:
                shutil.copy2(output_video_path, local_video_path)

            # Save complete frame data
            complete_detections_file = os.path.join(self.subfolders['raw_data'], 'complete_detections.json')
            with open(complete_detections_file, 'w') as f:
                json.dump(self.frame_detections, f, indent=2)

            complete_tracks_file = os.path.join(self.subfolders['raw_data'], 'complete_tracks.json')
            with open(complete_tracks_file, 'w') as f:
                json.dump(self.frame_tracks, f, indent=2)

            complete_analytics_file = os.path.join(self.subfolders['raw_data'], 'complete_analytics.json')
            with open(complete_analytics_file, 'w') as f:
                json.dump(self.frame_analytics, f, indent=2)

            # Save track data as pickle
            track_data_file = os.path.join(self.subfolders['raw_data'], 'track_data.pkl')
            with open(track_data_file, 'wb') as f:
                pickle.dump(dict(self.stats['track_data']), f)

            # Generate comprehensive report
            report = self.generate_comprehensive_report()

            # Save report as JSON
            report_json_file = os.path.join(self.subfolders['reports'], 'comprehensive_report.json')
            with open(report_json_file, 'w') as f:
                json.dump(report, f, indent=2)

            # Save report as readable text
            report_text_file = os.path.join(self.subfolders['reports'], 'comprehensive_report.txt')
            with open(report_text_file, 'w') as f:
                f.write(self.format_report_text(report))

            # Create summary CSV
            self.create_summary_csv()

            print(f"All data saved to: {self.local_folder}")
            return report

        except Exception as e:
            print(f"Error saving results: {e}")
            return None

    def generate_comprehensive_report(self):
        """Generate detailed analysis report"""
        elapsed = time.time() - self.stats['start_time']
        mota = self.calculate_mota()

        report = {
            'session_info': {
                'session_id': self.stats['session_id'],
                'total_runtime_seconds': elapsed,
                'frames_processed': self.stats['frames_processed'],
                'average_fps': self.stats['frames_processed'] / elapsed if elapsed > 0 else 0,
                'processing_date': datetime.now().isoformat()
            },
            'detection_summary': {
                'total_detections': self.stats['total_detections'],
                'class_distribution': dict(self.stats['class_counts']),
                'threat_level_distribution': dict(self.stats['threat_level_counts']),
                'detection_rate_per_frame': self.stats['total_detections'] / self.stats['frames_processed'] if self.stats['frames_processed'] > 0 else 0
            },
            'tracking_summary': {
                'total_tracks': len(self.stats['track_data']),
                'max_concurrent_tracks': self.stats['max_concurrent_tracks'],
                'average_track_duration': np.mean([data['total_frames'] for data in self.stats['track_data'].values()]) if self.stats['track_data'] else 0,
                'longest_track_duration': max([data['total_frames'] for data in self.stats['track_data'].values()]) if self.stats['track_data'] else 0
            },
            'performance_metrics': {
                'avg_processing_time_ms': np.mean(self.stats['processing_times']) * 1000 if self.stats['processing_times'] else 0,
                'min_processing_time_ms': min(self.stats['processing_times']) * 1000 if self.stats['processing_times'] else 0,
                'max_processing_time_ms': max(self.stats['processing_times']) * 1000 if self.stats['processing_times'] else 0
            },
            'accuracy_metrics': {
                'false_positives': self.stats['false_positives'],
                'false_negatives': self.stats['false_negatives'],
                'identity_switches': self.stats['identity_switches'],
                'mota_percentage': mota
            },
            'threat_analysis': {
                'high_threat_vehicles': sum(1 for data in self.stats['track_data'].values() if 'VERY_HIGH' in str(data.get('threat_level', '')) or 'HIGH' in str(data.get('threat_level', ''))),
                'military_vehicles_detected': sum(1 for data in self.stats['track_data'].values() if 'MILITARY' in str(data.get('vehicle_type', '')) or 'TANK' in str(data.get('vehicle_type', ''))),
                'civilian_vehicles_detected': sum(1 for data in self.stats['track_data'].values() if 'CIVILIAN' in str(data.get('vehicle_type', '')))
            },
            'data_storage': {
                'local_folder': self.local_folder,
                'detections_saved': len(self.frame_detections),
                'tracks_saved': len(self.frame_tracks),
                'analytics_saved': len(self.frame_analytics)
            }
        }

        return report

    def format_report_text(self, report):
        """Format report as readable text"""
        text = f"""
VEHICLE TRACKING COMPREHENSIVE REPORT - BYTETRACK ALGORITHM
=========================================================
Session ID: {report['session_info']['session_id']}
Processing Date: {report['session_info']['processing_date']}

PERFORMANCE SUMMARY
==================
• Total Runtime: {report['session_info']['total_runtime_seconds']:.2f} seconds
• Frames Processed: {report['session_info']['frames_processed']}
• Average FPS: {report['session_info']['average_fps']:.2f}
• Average Processing Time: {report['performance_metrics']['avg_processing_time_ms']:.2f}ms per frame

DETECTION SUMMARY
================
• Total Detections: {report['detection_summary']['total_detections']}
• Detection Rate: {report['detection_summary']['detection_rate_per_frame']:.2f} detections per frame

Class Distribution:
"""
        for class_name, count in report['detection_summary']['class_distribution'].items():
            percentage = (count / report['detection_summary']['total_detections']) * 100 if report['detection_summary']['total_detections'] > 0 else 0
            text += f"  • {class_name}: {count} ({percentage:.1f}%)\n"

        text += f"""
Threat Level Distribution:
"""
        for threat_level, count in report['detection_summary']['threat_level_distribution'].items():
            percentage = (count / report['detection_summary']['total_detections']) * 100 if report['detection_summary']['total_detections'] > 0 else 0
            text += f"  • {threat_level}: {count} ({percentage:.1f}%)\n"

        text += f"""
TRACKING SUMMARY
===============
• Total Unique Tracks: {report['tracking_summary']['total_tracks']}
• Max Concurrent Tracks: {report['tracking_summary']['max_concurrent_tracks']}
• Average Track Duration: {report['tracking_summary']['average_track_duration']:.1f} frames
• Longest Track Duration: {report['tracking_summary']['longest_track_duration']} frames

ACCURACY METRICS
===============
• False Positives (FP): {report['accuracy_metrics']['false_positives']}
• False Negatives (FN): {report['accuracy_metrics']['false_negatives']}
• Identity Switches (IDS): {report['accuracy_metrics']['identity_switches']}
• Multiple Object Tracking Accuracy (MOTA): {report['accuracy_metrics']['mota_percentage']:.2f}%

THREAT ANALYSIS
==============
• High Threat Vehicles: {report['threat_analysis']['high_threat_vehicles']}
• Military Vehicles: {report['threat_analysis']['military_vehicles_detected']}
• Civilian Vehicles: {report['threat_analysis']['civilian_vehicles_detected']}

DATA STORAGE
============
• Local Folder: {report['data_storage']['local_folder']}
• Detection Records: {report['data_storage']['detections_saved']}
• Track Records: {report['data_storage']['tracks_saved']}
• Analytics Records: {report['data_storage']['analytics_saved']}
"""
        return text

    def create_summary_csv(self):
        """Create CSV summaries for easy analysis"""
        try:
            # Frame-by-frame summary
            frame_summary = []
            for analytics in self.frame_analytics:
                frame_summary.append({
                    'frame_number': analytics['frame_number'],
                    'processing_time_ms': analytics['processing_time_ms'],
                    'detection_count': analytics['detection_count'],
                    'track_count': analytics['track_count'],
                    'high_threat_count': analytics['threat_detections']['HIGH'] + analytics['threat_detections']['VERY_HIGH'],
                    'medium_threat_count': analytics['threat_detections']['MEDIUM'],
                    'low_threat_count': analytics['threat_detections']['LOW']
                })

            df_frames = pd.DataFrame(frame_summary)
            df_frames.to_csv(os.path.join(self.subfolders['analytics'], 'frame_summary.csv'), index=False)

            # Track summary
            track_summary = []
            for track_id, data in self.stats['track_data'].items():
                track_summary.append({
                    'track_id': track_id,
                    'class_name': data['class_name'],
                    'vehicle_type': data['vehicle_type'],
                    'threat_level': data['threat_level'],
                    'first_seen_frame': data['first_seen'],
                    'last_seen_frame': data['last_seen'],
                    'duration_frames': data['total_frames'],
                    'max_confidence': data['max_confidence']
                })

            df_tracks = pd.DataFrame(track_summary)
            df_tracks.to_csv(os.path.join(self.subfolders['analytics'], 'track_summary.csv'), index=False)

            # Accuracy metrics summary
            accuracy_summary = [{
                'false_positives': self.stats['false_positives'],
                'false_negatives': self.stats['false_negatives'],
                'identity_switches': self.stats['identity_switches'],
                'mota_percentage': self.calculate_mota()
            }]

            df_accuracy = pd.DataFrame(accuracy_summary)
            df_accuracy.to_csv(os.path.join(self.subfolders['analytics'], 'accuracy_summary.csv'), index=False)

            print("CSV summaries created successfully")

        except Exception as e:
            print(f"Error creating CSV summaries: {e}")


def process_video_with_enhanced_tracking(model_path, video_path, max_frames=400, confidence=0.467, save_interval=50):
    """
    Complete video processing with enhanced ByteTrack tracking and local storage

    Args:
        model_path: Path to YOLO model (can be from Drive)
        video_path: Path to input video (can be from Drive)
        max_frames: Maximum frames to process
        confidence: Detection confidence threshold
        save_interval: Interval for saving data
    """
    print("="*80)
    print("ENHANCED VEHICLE TRACKING SYSTEM")
    print("Custom Model: Vehicle Detection & Classification")
    print("Tracking Algorithm: ByteTrack (High Performance)")
    print("Storage: Local Machine")
    print("Accuracy Metrics: FP, FN, IDS, MOTA")
    print("="*80)

    # Initialize tracker
    tracker = VehicleTracker(model_path, confidence=confidence)

    print(f"Processing video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None, None

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_frames=total_frames
    print(f"\nVideo Properties:")
    print(f"   • Resolution: {width}x{height}")
    print(f"   • FPS: {fps:.1f}")
    print(f"   • Total Frames: {total_frames}")
    print(f"   • Processing Frames: {total_frames} (entire video)")
    print(f"   • Local Save Interval: {save_interval} frames")

    # Output video setup
    output_path = os.path.join(tracker.subfolders['videos'], f'enhanced_tracked_{Path(video_path).stem}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    display_interval = max(1, total_frames // 20)

    try:
        print("\nStarting Enhanced Vehicle Tracking with ByteTrack...")
        print("-" * 80)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame with enhanced tracking
            tracks, detections = tracker.process_frame(frame, frame_count)

            # Draw results
            frame = tracker.draw_tracking_results(frame, tracks)

            # Add performance overlay
            current_time = time.time()
            current_fps = frame_count / (current_time - start_time)
            #frame = tracker.add_performance_overlay(frame, current_fps)

            # Save frame to output video
            out.write(frame)

            # Save data locally at intervals
            tracker.save_frame_data(frame_count, save_interval)

            # Display progress
            if frame_count % display_interval == 0 or frame_count <= 5:
                progress = (frame_count / total_frames) * 100

                # Calculate accuracy metrics
                mota = tracker.calculate_mota()

                print("=" * 80)
                print(f"PROCESSING PROGRESS: {frame_count}/{total_frames} ({progress:.1f}%)")
                print("=" * 80)
                print(f"PERFORMANCE METRICS:")
                print(f"   • Current FPS: {current_fps:.2f}")
                print(f"   • Active Tracks: {tracker.stats['active_tracks']}")
                print(f"   • Total Detections: {tracker.stats['total_detections']}")
                print(f"   • Data Records Saved: {len(tracker.frame_detections)}")

                print(f"\nACCURACY METRICS:")
                print(f"   • False Positives (FP): {tracker.stats['false_positives']}")
                print(f"   • False Negatives (FN): {tracker.stats['false_negatives']}")
                print(f"   • Identity Switches (IDS): {tracker.stats['identity_switches']}")
                print(f"   • MOTA: {mota:.2f}%")

                if tracker.stats['threat_level_counts']:
                    print(f"\nTHREAT LEVEL SUMMARY:")
                    for threat, count in tracker.stats['threat_level_counts'].items():
                        if count > 0:
                            print(f"   • {threat}: {count}")

                if tracker.stats['class_counts']:
                    print(f"\nVEHICLE CLASSES DETECTED:")
                    for cls_name, count in list(tracker.stats['class_counts'].items())[:5]:
                        vehicle_config = tracker.vehicle_config.get(cls_name, {})
                        vehicle_type = vehicle_config.get('type', 'UNKNOWN')
                        print(f"   • {cls_name} ({vehicle_type}): {count}")

                print(f"\nLOCAL STORAGE:")
                print(f"   • Location: {tracker.local_folder}")
                print(f"   • Last Save: Frame {(frame_count // save_interval) * save_interval}")
                print("-" * 80)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        cap.release()
        out.release()

    # Save final results locally
    print("\n" + "="*80)
    print("SAVING FINAL RESULTS TO LOCAL MACHINE")
    print("="*80)

    final_report = tracker.save_final_results(output_path)

    if final_report:
        print("\nPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)

        print(f"\nSESSION SUMMARY:")
        print(f"  • Session ID: {final_report['session_info']['session_id']}")
        print(f"  • Frames Processed: {final_report['session_info']['frames_processed']}")
        print(f"  • Average FPS: {final_report['session_info']['average_fps']:.2f}")
        print(f"  • Total Runtime: {final_report['session_info']['total_runtime_seconds']:.2f}s")

        print(f"\nDETECTION RESULTS:")
        print(f"  • Total Detections: {final_report['detection_summary']['total_detections']}")
        print(f"  • Unique Tracks: {final_report['tracking_summary']['total_tracks']}")
        print(f"  • Max Concurrent: {final_report['tracking_summary']['max_concurrent_tracks']}")

        print(f"\nACCURACY METRICS:")
        print(f"  • False Positives (FP): {final_report['accuracy_metrics']['false_positives']}")
        print(f"  • False Negatives (FN): {final_report['accuracy_metrics']['false_negatives']}")
        print(f"  • Identity Switches (IDS): {final_report['accuracy_metrics']['identity_switches']}")
        print(f"  • MOTA: {final_report['accuracy_metrics']['mota_percentage']:.2f}%")

        print(f"\nTHREAT ANALYSIS:")
        print(f"  • High Threat Vehicles: {final_report['threat_analysis']['high_threat_vehicles']}")
        print(f"  • Military Vehicles: {final_report['threat_analysis']['military_vehicles_detected']}")
        print(f"  • Civilian Vehicles: {final_report['threat_analysis']['civilian_vehicles_detected']}")

        print(f"\nVEHICLE CLASSES DETECTED:")
        for cls_name, count in final_report['detection_summary']['class_distribution'].items():
            percentage = (count / final_report['detection_summary']['total_detections']) * 100
            print(f"  • {cls_name}: {count} ({percentage:.1f}%)")

        return output_path, final_report
    else:
        print("Error saving final results")
        return output_path, None


def main():
    """Main function to run the enhanced vehicle tracking system with ByteTrack"""
    print("ENHANCED VEHICLE TRACKING SYSTEM - BYTETRACK ALGORITHM")
    print("Ready to process your video with custom vehicle detection model")

    model_path = r"best1.pt"  # YOLO Model

    # Video
    video_path = r"demo2.mp4" 

    # Verify model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please update the model_path variable with the correct path to your best.pt file")
        return

    # Verify video exists
    if not os.path.exists(video_path):
        print(f"Video not found at: {video_path}")
        print("Please update the video_path variable with the correct path to your video file")
        return

    print(f"Model found: {model_path}")
    print(f"Video found: {video_path}")

    # Load model to verify it works
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully")
        print(f"   • Classes: {list(model.names.values())}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run the processing
    try:
        output_path, report = process_video_with_enhanced_tracking(
            model_path=model_path,
            video_path=video_path,
            max_frames=None,  # Process entire video
            confidence=0.05,  # Very low confidence for maximum detections
            save_interval=50   # Save data every 50 frames
        )

        if output_path and report:
            print("\nSUCCESS: ByteTrack processing complete!")
        else:
            print("\nProcessing completed but there may have been issues saving results.")

    except Exception as e:
        print(f"\nError during main execution: {e}")


if __name__ == "__main__":
    main()
