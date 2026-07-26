# Intelligent Surveillance System - Harassment Detection

## System Overview

This intelligent video surveillance system detects suspicious activities and potential harassment incidents in real-time. It combines computer vision, tracking, and machine learning to identify concerning behavioral patterns and anomalies in crowded environments.

## Architecture

The system is designed with a modular architecture consisting of the following components:

### 1. **Video Input & Pre-processing** (`src/video_preprocessing.py`)
- **Frame Extraction**: Extracts frames from video input
- **Sliding Window**: Creates temporal windows of frames for analysis
- **Optical Flow**: Computes motion vectors between consecutive frames
- **Frame Normalization**: Resizes and normalizes frames for processing

**Key Classes:**
- `VideoPreprocessor`: Main preprocessing handler
  - `read_video()`: Open and read video files
  - `extract_frames()`: Extract all frames from video
  - `sliding_window()`: Create temporal frame windows
  - `compute_optical_flow()`: Calculate motion between frames
  - `visualize_optical_flow()`: Visualize motion vectors

### 2. **Tracker Service** (`src/tracker_service.py`)
Maintains consistent tracking of individuals across frames using Kalman filtering and motion prediction.

**Key Components:**
- **Kalman Filter**: Predicts person trajectories
  - State representation: `[x, y, vx, vy]` (position and velocity)
  - Handles smooth motion prediction
  
- **Motion Predictor**: Extrapolates future positions
  - `predict_next_position()`: Single-step prediction
  - `predict_trajectory()`: Multi-step trajectory prediction
  
- **Identity Matcher**: Associates detections with existing tracks
  - IoU-based distance calculation
  - Greedy matching algorithm
  
- **Occlusion Detector**: Handles temporarily occluded persons
  - Maximum age threshold for track retention
  - Re-identification when person reappears

**Key Classes:**
- `TrackerService`: Main tracking orchestrator
  - `update()`: Process new detections and update tracks
  - Returns: Dictionary of active tracks with IDs

### 3. **Suspicious Service** (`src/suspicious_service.py`)
Detects suspicious behaviors and patterns indicating potential harassment.

**Key Components:**
- **Age Classifier**: Determines if person is child or adult
  - Height-based heuristic (configurable)
  - Confidence-based classification
  
- **Adult-Child Pair Identifier**: Detects suspicious pairings
  - Proximity-based matching (default: 100 pixels)
  - Flags adults inappropriately close to children
  
- **Hand Contact Detector**: Identifies hand-to-person contact
  - Keypoint-based analysis
  - Detects contact with vulnerable areas
  
- **Witness Analyzer**: Identifies nearby observers
  - Reaction analysis
  - Witness engagement assessment

**Key Classes:**
- `SuspiciousService`: Main suspicious event detector
  - `detect_suspicious_events()`: Multi-stage detection pipeline
  - Returns: List of suspicious events with confidence scores

### 4. **Crowd Analysis** (`src/crowd_analysis.py`)
Analyzes crowd dynamics and identifies regions of interest.

**Key Components:**
- **Density Estimator**: Maps crowd concentration
  - Grid-based density calculation
  - High-density region identification
  - Global density metrics
  
- **Person Selector**: Identifies persons of interest
  - Behavioral anomaly detection
  - Isolation scoring
  - High-priority person ranking
  
- **SCIP Processor**: Spatio-temporal context analysis
  - Edge detection and feature extraction
  - Frame segmentation
  - Context-based anomaly detection

**Key Classes:**
- `CrowdAnalyzer`: Main crowd analysis service
  - `analyze()`: Complete crowd assessment
  - Returns: `CrowdMetrics` with density and priority persons

### 5. **Event Aggregation & Reporting** (`src/event_aggregation.py`)
Aggregates individual events into coherent incidents and generates reports.

**Key Components:**
- **Event Aggregator**: Clusters related events
  - Temporal proximity clustering (default: 30 frames)
  - Spatial clustering (default: 100 pixel radius)
  - Event merging and deduplication
  
- **Severity Classifier**: Assesses incident severity
  - Multi-factor scoring
  - Severity levels: LOW, MEDIUM, HIGH, CRITICAL
  - Confidence and event count weighting
  
- **Report Generator**: Creates comprehensive reports
  - Incident description generation
  - Action recommendation system
  - Evidence tracking

**Key Classes:**
- `EventReporter`: Main reporting orchestrator
  - `process_events()`: Convert events to incidents
  - `export_report()`: Generate text/JSON reports

### 6. **Main Orchestrator** (`src/main.py`)
Integrates all modules into a cohesive system pipeline.

**Key Class:**
- `SurveillanceSystem`: System coordinator
  - `process_video()`: End-to-end video processing
  - `generate_report()`: Final system report generation

## Data Flow

```
Video Input
    ↓
Video Preprocessing (optical flow, frame extraction)
    ↓
Person Detection (YOLO/SSD in real implementation)
    ↓
Tracker Service (Kalman filter, ID matching)
    ↓
Parallel Processing:
    ├→ Suspicious Service (behavioral analysis)
    ├→ Crowd Analysis (density, context)
    └→ Additional Detectors
    ↓
Event Aggregation (temporal/spatial clustering)
    ↓
Severity Classification
    ↓
Incident Reporting
    ↓
Output & Alerts
```

## Key Algorithms

### Kalman Filter (Tracking)
- **State Model**: Constant velocity motion model
- **Update Rate**: Every frame
- **Process Noise**: 0.01 (motion uncertainty)
- **Measurement Noise**: 1.0 (detection uncertainty)

### Optical Flow (Motion Detection)
- **Method**: Farneback dense flow
- **Window Size**: 15×15
- **Pyramid Levels**: 3
- **Iterations**: 3

### IoU Matching (Identity Association)
- **Distance Metric**: 1 - IoU (Intersection over Union)
- **Threshold**: 0.5 IoU minimum
- **Strategy**: Greedy nearest-neighbor

### Event Clustering
- **Temporal Window**: 30 frames
- **Spatial Threshold**: 100 pixels
- **Clustering Method**: Single-linkage

### Severity Scoring
- **Confidence**: 60% weight
- **Event Count**: 30% weight  
- **Person Involvement**: 10% weight

## Module Interfaces

### Input/Output Types

**PersonInstance**
```python
@dataclass
class PersonInstance:
    track_id: int
    bbox: Tuple[int, int, int, int]  # [x1, y1, x2, y2]
    keypoints: np.ndarray  # [17, 3] for pose
    age_group: AgeGroup  # CHILD/ADULT/UNKNOWN
    confidence: float
```

**SuspiciousEvent**
```python
@dataclass
class SuspiciousEvent:
    event_id: int
    event_type: str
    person1_id: int
    person2_id: int
    frame_range: Tuple[int, int]
    confidence: float
    details: Dict
```

**IncidentReport**
```python
@dataclass
class IncidentReport:
    incident_id: int
    timestamp: datetime
    severity: IncidentSeverity  # LOW/MEDIUM/HIGH/CRITICAL
    event_logs: List[EventLog]
    involved_persons: List[int]
    location: Tuple[int, int]
    description: str
    evidence_frames: List[int]
    recommended_actions: List[str]
```

## Configuration Parameters

### Video Processing
- `window_size`: 8 frames per window
- `stride`: 1 frame step

### Tracking
- `max_age`: 30 frames
- `kalman_dt`: 1.0 (time step)

### Detection
- `child_height_threshold`: 100 pixels
- `proximity_threshold`: 100 pixels (adult-child)
- `contact_threshold`: 30 pixels (hand contact)
- `witness_radius`: 200 pixels

### Aggregation
- `temporal_window`: 30 frames
- `spatial_threshold`: 100 pixels

## Usage Example

```python
from src.main import SurveillanceSystem

# Initialize system
system = SurveillanceSystem('input_video.mp4')

# Process video
incidents = system.process_video('output_video.mp4')

# Generate reports
print(system.generate_report())

# Export detailed incident reports
for incident in system.event_reporter.incident_history:
    report_text = system.event_reporter.export_report(incident, format='text')
    print(report_text)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- OpenCV 4.8.0+
- NumPy 1.24+
- PyTorch 2.0+ (for deep learning components)
- scikit-learn 1.3+ (for ML utilities)

## Performance Considerations

### Computational Complexity
- **Video Preprocessing**: O(frames × frame_size)
- **Tracking**: O(tracks × detections)
- **Suspicious Detection**: O(persons²)
- **Event Aggregation**: O(events²)

### Memory Requirements
- **Frame Buffer**: 8 × frame_resolution
- **Track History**: 30 frames per track
- **Event History**: Linear with number of events

### Real-time Performance
- Target: 30 FPS on 640×480 video
- Bottleneck: Person detection (if using heavy models)
- Optimization: Frame batching, GPU acceleration

## Extensibility

The modular design allows easy integration of:

1. **Better Person Detectors**
   - Replace with YOLO v8, Faster R-CNN, etc.
   - Update `_detect_persons()` in main.py

2. **Pose Estimation**
   - Integrate OpenPose, MediaPipe
   - Enhanced keypoint-based detection

3. **Action Recognition**
   - Add temporal CNN models
   - Detect specific actions (pushing, grabbing, etc.)

4. **Gaze Detection**
   - Enhance witness analysis
   - Determine attention direction

5. **Re-identification (Re-ID)**
   - Cross-camera tracking
   - Long-term person association

## Future Enhancements

- [ ] Integration with deep learning pose estimation
- [ ] Multi-camera tracking
- [ ] Real-time alerting system
- [ ] Web-based visualization dashboard
- [ ] Mobile app for incident review
- [ ] Advanced action recognition
- [ ] Anomaly detection using autoencoders
- [ ] Integration with law enforcement systems

## License

ProprietarySystemDescription

## Authors

Surveillance Team

## Support

For issues or questions, please contact the development team.
