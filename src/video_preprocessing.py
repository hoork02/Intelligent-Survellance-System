"""
Video Input & Pre-processing Module
Handles video frame extraction, sliding window, and optical flow computation
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, List, Optional


class VideoPreprocessor:
    """
    Handles video input and preprocessing including frame extraction,
    sliding window management, and optical flow computation.
    """
    
    def __init__(self, window_size: int = 8, stride: int = 1):
        """
        Initialize video preprocessor.
        
        Args:
            window_size: Number of frames in sliding window
            stride: Stride for frame sampling
        """
        self.window_size = window_size
        self.stride = stride
        self.frame_buffer = deque(maxlen=window_size)
        self.flow_buffer = deque(maxlen=window_size - 1)
        
    def read_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """
        Open video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoCapture object or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        return cap
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames (numpy arrays)
        """
        cap = self.read_video(video_path)
        if cap is None:
            return []
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def sliding_window(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Create sliding windows from frame sequence.
        
        Args:
            frames: List of frames
            
        Returns:
            List of sliding windows (each window contains window_size frames)
        """
        windows = []
        for i in range(0, len(frames) - self.window_size + 1, self.stride):
            window = frames[i:i + self.window_size]
            windows.append(window)
        return windows
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between consecutive frames using Farneback method.
        
        Args:
            frame1: Previous frame
            frame2: Current frame
            
        Returns:
            Optical flow field (2-channel array: dx, dy)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            n8=5,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        return flow
    
    def compute_flow_sequence(self, frame_window: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute optical flow for entire frame window.
        
        Args:
            frame_window: List of consecutive frames
            
        Returns:
            List of optical flow fields
        """
        flows = []
        for i in range(len(frame_window) - 1):
            flow = self.compute_optical_flow(frame_window[i], frame_window[i + 1])
            flows.append(flow)
        return flows
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Normalize and resize frame.
        
        Args:
            frame: Input frame
            target_size: Target resolution
            
        Returns:
            Preprocessed frame
        """
        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def get_motion_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute motion magnitude from optical flow.
        
        Args:
            flow: Optical flow field
            
        Returns:
            Motion magnitude field
        """
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return magnitude
    
    def visualize_optical_flow(self, frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
        """
        Visualize optical flow on frame.
        
        Args:
            frame: Input frame
            flow: Optical flow field
            step: Step size for flow visualization
            
        Returns:
            Frame with optical flow visualization
        """
        h, w = flow.shape[:2]
        y, x = np.mgrid[0:h:step, 0:w:step]
        fx, fy = flow[::step, ::step].T
        
        vis = frame.copy()
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        
        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        
        return vis
