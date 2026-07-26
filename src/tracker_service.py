"""
Tracker Service Module
Handles person tracking using Kalman filter, motion prediction, and identity matching
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass
class KalmanState:
    """Kalman filter state for a tracked person."""
    position: np.ndarray  # [x, y] center position
    velocity: np.ndarray  # [vx, vy] velocity
    bbox: Tuple[int, int, int, int]  # [x1, y1, x2, y2] bounding box
    age: int = 0  # Frames since track started
    hits: int = 0  # Successful detections
    hit_streak: int = 0  # Consecutive successful detections


class KalmanFilter:
    """
    Kalman filter for 2D motion tracking.
    Models position and velocity in 2D space.
    """
    
    def __init__(self, dt: float = 1.0):
        """
        Initialize Kalman filter.
        
        Args:
            dt: Time step between frames
        """
        self.dt = dt
        # State: [x, y, vx, vy]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Process noise covariance
        self.Q = np.eye(4) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(2) * 1.0
        
        # State covariance
        self.P = np.eye(4) * 1.0
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict next state.
        
        Args:
            x: Current state [x, y, vx, vy]
            
        Returns:
            Predicted state
        """
        x_pred = self.F @ x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_pred
    
    def update(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.
        
        Args:
            x: Predicted state
            z: Measurement [x, y]
            
        Returns:
            Updated state
        """
        y = z - (self.H @ x[:2])  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        x[:2] += (K @ y).flatten()
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return x


class MotionPredictor:
    """
    Predicts future motion of tracked objects.
    Uses Kalman filter state to extrapolate trajectories.
    """
    
    def __init__(self, kalman_filter: KalmanFilter):
        """
        Initialize motion predictor.
        
        Args:
            kalman_filter: Kalman filter instance
        """
        self.kf = kalman_filter
    
    def predict_next_position(self, state: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict future position.
        
        Args:
            state: Current state [x, y, vx, vy]
            steps: Number of steps to predict ahead
            
        Returns:
            Predicted position [x, y]
        """
        pred_state = state.copy()
        for _ in range(steps):
            pred_state = self.kf.F @ pred_state
        return pred_state[:2]
    
    def predict_trajectory(self, state: np.ndarray, steps: int = 5) -> np.ndarray:
        """
        Predict future trajectory.
        
        Args:
            state: Current state
            steps: Number of steps
            
        Returns:
            Array of predicted positions
        """
        trajectory = []
        pred_state = state.copy()
        
        for _ in range(steps):
            pred_state = self.kf.F @ pred_state
            trajectory.append(pred_state[:2].copy())
        
        return np.array(trajectory)


class IdentityMatcher:
    """
    Matches detected persons to existing tracks using appearance and motion features.
    """
    
    def __init__(self, max_distance: float = 50.0, max_iou_distance: float = 0.5):
        """
        Initialize identity matcher.
        
        Args:
            max_distance: Max Euclidean distance for matching
            max_iou_distance: Max IoU distance threshold
        """
        self.max_distance = max_distance
        self.max_iou_distance = max_iou_distance
    
    def compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Compute Intersection over Union between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value [0, 1]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def euclidean_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Compute Euclidean distance between positions.
        
        Args:
            pos1: Position [x, y]
            pos2: Position [x, y]
            
        Returns:
            Distance
        """
        return np.linalg.norm(pos1 - pos2)
    
    def match_detections(self, tracks: Dict[int, KalmanState], 
                        detections: List[Tuple]) -> Tuple[Dict[int, Tuple], List[Tuple]]:
        """
        Match detections to existing tracks.
        
        Args:
            tracks: Dictionary of active tracks {track_id: KalmanState}
            detections: List of detections (bboxes)
            
        Returns:
            (matched_pairs, unmatched_detections)
        """
        matched_pairs = {}
        matched_detections = set()
        
        # Compute distance matrix
        cost_matrix = np.full((len(tracks), len(detections)), np.inf)
        
        for track_idx, (track_id, track) in enumerate(tracks.items()):
            for det_idx, detection in enumerate(detections):
                iou = self.compute_iou(track.bbox, detection)
                if iou > self.max_iou_distance:
                    cost_matrix[track_idx, det_idx] = 1.0 - iou
        
        # Simple greedy matching
        for track_idx, (track_id, track) in enumerate(tracks.items()):
            if np.all(np.isinf(cost_matrix[track_idx])):
                continue
            
            best_det_idx = np.argmin(cost_matrix[track_idx])
            if cost_matrix[track_idx, best_det_idx] < np.inf:
                matched_pairs[track_id] = detections[best_det_idx]
                matched_detections.add(best_det_idx)
        
        unmatched_detections = [det for idx, det in enumerate(detections) 
                               if idx not in matched_detections]
        
        return matched_pairs, unmatched_detections


class OcclusionDetector:
    """
    Detects and handles occlusions in tracking.
    """
    
    def __init__(self, max_age: int = 30):
        """
        Initialize occlusion detector.
        
        Args:
            max_age: Maximum frames to keep occluded track
        """
        self.max_age = max_age
    
    def detect_occlusion(self, track: KalmanState, frame_idx: int) -> bool:
        """
        Detect if track is occluded.
        
        Args:
            track: Track to check
            frame_idx: Current frame index
            
        Returns:
            True if track is likely occluded
        """
        # Track is considered occluded if no hits for several frames
        return (frame_idx - track.hits) > self.max_age // 2
    
    def handle_occlusion(self, track: KalmanState, frame_idx: int) -> bool:
        """
        Handle occluded track.
        
        Args:
            track: Track to handle
            frame_idx: Current frame index
            
        Returns:
            True if track should be removed
        """
        age_since_hit = frame_idx - track.hits
        return age_since_hit > self.max_age


class TrackerService:
    """
    Main tracker service managing all tracking operations.
    """
    
    def __init__(self, max_age: int = 30):
        """
        Initialize tracker service.
        
        Args:
            max_age: Maximum age for tracks
        """
        self.kalman_filter = KalmanFilter()
        self.motion_predictor = MotionPredictor(self.kalman_filter)
        self.identity_matcher = IdentityMatcher()
        self.occlusion_detector = OcclusionDetector(max_age)
        
        self.tracks: Dict[int, KalmanState] = {}
        self.next_track_id = 0
        self.frame_count = 0
    
    def update(self, detections: List[Tuple]) -> Dict[int, KalmanState]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection bboxes
            
        Returns:
            Dictionary of active tracks
        """
        self.frame_count += 1
        
        # Match detections to tracks
        matched_pairs, unmatched_detections = self.identity_matcher.match_detections(
            self.tracks, detections
        )
        
        # Update matched tracks
        for track_id, bbox in matched_pairs.items():
            track = self.tracks[track_id]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            track.bbox = bbox
            track.hits = self.frame_count
            track.hit_streak += 1
        
        # Create new tracks for unmatched detections
        for bbox in unmatched_detections:
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            state = np.array([center[0], center[1], 0, 0], dtype=float)
            
            new_track = KalmanState(
                position=center,
                velocity=np.zeros(2),
                bbox=bbox,
                hits=self.frame_count,
                hit_streak=1
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1
        
        # Remove dead tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if self.occlusion_detector.handle_occlusion(track, self.frame_count):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return self.tracks.copy()
