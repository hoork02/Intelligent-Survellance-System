"""
Main orchestration module
Coordinates all system components
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from video_preprocessing import VideoPreprocessor
from tracker_service import TrackerService, KalmanState
from suspicious_service import SuspiciousService, PersonInstance, AgeGroup
from crowd_analysis import CrowdAnalyzer
from event_aggregation import EventReporter, EventLog


class SurveillanceSystem:
    """
    Main surveillance system orchestrator.
    Integrates all modules for harassment detection.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize surveillance system.
        
        Args:
            video_path: Path to input video
        """
        self.video_path = video_path
        
        # Initialize modules
        self.video_preprocessor = VideoPreprocessor(window_size=8)
        self.tracker_service = TrackerService(max_age=30)
        self.suspicious_service = SuspiciousService()
        self.crowd_analyzer = CrowdAnalyzer()
        self.event_reporter = EventReporter()
        
        self.frame_count = 0
        self.current_events = []
    
    def process_video(self, output_video: Optional[str] = None) -> List[Dict]:
        """
        Process entire video and detect suspicious activities.
        
        Args:
            output_video: Optional output video path for visualization
            
        Returns:
            List of detected incidents
        """
        # Extract frames
        print("Extracting frames...")
        frames = self.video_preprocessor.extract_frames(self.video_path)
        
        if not frames:
            print("Failed to extract frames")
            return []
        
        print(f"Processing {len(frames)} frames...")
        
        # Set up video writer if output requested
        out_writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            frame_size = (frames[0].shape[1], frames[0].shape[0])
            out_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
        
        # Process each frame
        all_incidents = []
        
        for frame_idx, frame in enumerate(frames):
            self.frame_count = frame_idx
            
            # In real scenario, would use actual person detection (YOLO, etc.)
            persons = self._detect_persons(frame)
            
            # Update tracker
            detections = [p['bbox'] for p in persons]
            self.tracker_service.update(detections)
            
            # Update person IDs from tracking
            for i, person in enumerate(persons):
                if i < len(self.tracker_service.tracks):
                    person['track_id'] = list(self.tracker_service.tracks.keys())[i]
            
            # Detect suspicious events
            person_instances = [
                PersonInstance(
                    track_id=p.get('track_id', i),
                    bbox=p['bbox'],
                    keypoints=p.get('keypoints', np.zeros((17, 3))),
                    age_group=AgeGroup.UNKNOWN,
                    confidence=p.get('confidence', 0.8)
                )
                for i, p in enumerate(persons)
            ]
            
            events = self.suspicious_service.detect_suspicious_events(person_instances, frame_idx)
            
            # Analyze crowd
            crowd_metrics = self.crowd_analyzer.analyze(frame, persons)
            
            # Log events
            for event in events:
                event_log = EventLog(
                    timestamp=datetime.now(),
                    frame_id=frame_idx,
                    event_type=event.event_type,
                    person_ids=[event.person1_id, event.person2_id],
                    confidence=event.confidence,
                    location=(0, 0),
                    details=event.details
                )
                self.current_events.append(event_log)
            
            # Generate reports from accumulated events
            if frame_idx % 30 == 0 and self.current_events:
                incidents = self.event_reporter.process_events(self.current_events)
                all_incidents.extend(incidents)
                self.current_events = []
            
            # Visualization
            vis_frame = self._visualize_frame(frame, persons, events)
            
            if out_writer:
                out_writer.write(vis_frame)
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{len(frames)} frames")
        
        # Process remaining events
        if self.current_events:
            incidents = self.event_reporter.process_events(self.current_events)
            all_incidents.extend(incidents)
        
        if out_writer:
            out_writer.release()
        
        print(f"Processing complete. Detected {len(all_incidents)} incidents.")
        return all_incidents
    
    def _detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame.
        In real implementation, would use YOLO or similar.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected persons with bboxes
        """
        # Placeholder - in real scenario would use object detection
        persons = []
        
        # Simple placeholder: detect some random persons
        h, w = frame.shape[:2]
        num_persons = np.random.randint(1, 5)
        
        for _ in range(num_persons):
            x1 = np.random.randint(0, w - 50)
            y1 = np.random.randint(0, h - 100)
            x2 = x1 + np.random.randint(40, 100)
            y2 = y1 + np.random.randint(80, 200)
            
            persons.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': np.random.random(),
                'keypoints': np.random.random((17, 3))
            })
        
        return persons
    
    def _visualize_frame(self, frame: np.ndarray, persons: List[Dict], 
                        events: List) -> np.ndarray:
        """
        Visualize frame with detections and events.
        
        Args:
            frame: Input frame
            persons: Detected persons
            events: Detected events
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw person bboxes
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw event indicators
        for event in events:
            cv2.putText(vis_frame, f"Event: {event.event_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame
    
    def generate_report(self) -> str:
        """
        Generate final system report.
        
        Returns:
            Report string
        """
        report = "\n" + "="*60 + "\n"
        report += "SURVEILLANCE SYSTEM REPORT\n"
        report += "="*60 + "\n"
        report += f"Total Frames Processed: {self.frame_count}\n"
        report += f"Total Incidents Detected: {len(self.event_reporter.incident_history)}\n"
        
        if self.event_reporter.incident_history:
            report += "\nIncident Summary:\n"
            for incident in self.event_reporter.incident_history:
                report += f"  - Incident #{incident.incident_id}: {incident.severity.value} severity\n"
        
        report += "="*60 + "\n"
        return report


if __name__ == "__main__":
    # Example usage
    video_path = "input_video.mp4"
    output_path = "output_video.mp4"
    
    system = SurveillanceSystem(video_path)
    incidents = system.process_video(output_path)
    
    # Print report
    print(system.generate_report())
    
    # Export detailed reports
    for incident in system.event_reporter.incident_history:
        print(system.event_reporter.export_report(incident, format='text'))
