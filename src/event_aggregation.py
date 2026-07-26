"""
Event Aggregation & Reporting Module
Aggregates suspicious events and generates incident reports
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EventLog:
    """Log entry for a detected event."""
    timestamp: datetime
    frame_id: int
    event_type: str
    person_ids: List[int]
    confidence: float
    location: Tuple[int, int]  # [x, y]
    details: Dict


@dataclass
class IncidentReport:
    """Comprehensive incident report."""
    incident_id: int
    timestamp: datetime
    severity: IncidentSeverity
    event_logs: List[EventLog] = field(default_factory=list)
    involved_persons: List[int] = field(default_factory=list)
    location: Tuple[int, int] = (0, 0)
    description: str = ""
    evidence_frames: List[int] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


class EventAggregator:
    """
    Aggregates individual events into coherent incidents.
    """
    
    def __init__(self, temporal_window: int = 30, spatial_threshold: float = 100.0):
        """
        Initialize event aggregator.
        
        Args:
            temporal_window: Frames to consider for aggregation
            spatial_threshold: Distance threshold for spatial clustering
        """
        self.temporal_window = temporal_window
        self.spatial_threshold = spatial_threshold
        self.event_history: List[EventLog] = []
    
    def add_event(self, event: EventLog) -> None:
        """
        Add event to history.
        
        Args:
            event: Event log entry
        """
        self.event_history.append(event)
    
    def cluster_events(self) -> List[List[EventLog]]:
        """
        Cluster events by temporal and spatial proximity.
        
        Returns:
            List of event clusters
        """
        if not self.event_history:
            return []
        
        clusters = []
        used_indices = set()
        
        for i, event1 in enumerate(self.event_history):
            if i in used_indices:
                continue
            
            cluster = [event1]
            used_indices.add(i)
            
            for j in range(i + 1, len(self.event_history)):
                if j in used_indices:
                    continue
                
                event2 = self.event_history[j]
                
                # Check temporal proximity
                time_diff = (event2.timestamp - event1.timestamp).total_seconds()
                if abs(time_diff) > self.temporal_window:
                    continue
                
                # Check spatial proximity
                spatial_dist = np.linalg.norm(
                    np.array(event1.location) - np.array(event2.location)
                )
                if spatial_dist < self.spatial_threshold:
                    cluster.append(event2)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def merge_similar_events(self, events: List[EventLog]) -> EventLog:
        """
        Merge similar events into single representative event.
        
        Args:
            events: List of events to merge
            
        Returns:
            Merged event
        """
        if not events:
            return None
        
        avg_confidence = np.mean([e.confidence for e in events])
        avg_location = np.mean([e.location for e in events], axis=0)
        all_persons = list(set([p for e in events for p in e.person_ids]))
        
        merged = EventLog(
            timestamp=events[0].timestamp,
            frame_id=events[0].frame_id,
            event_type=events[0].event_type,
            person_ids=all_persons,
            confidence=avg_confidence,
            location=tuple(avg_location.astype(int)),
            details={'merged_count': len(events), 'original_events': events}
        )
        
        return merged


class SeverityClassifier:
    """
    Classifies incidents by severity level.
    """
    
    def __init__(self):
        """
        Initialize severity classifier.
        """
        self.confidence_thresholds = {
            IncidentSeverity.CRITICAL: 0.9,
            IncidentSeverity.HIGH: 0.75,
            IncidentSeverity.MEDIUM: 0.6,
            IncidentSeverity.LOW: 0.0
        }
    
    def classify(self, events: List[EventLog]) -> IncidentSeverity:
        """
        Classify incident severity based on events.
        
        Args:
            events: List of events
            
        Returns:
            Severity classification
        """
        if not events:
            return IncidentSeverity.LOW
        
        max_confidence = max([e.confidence for e in events])
        event_count = len(events)
        unique_persons = len(set([p for e in events for p in e.person_ids]))
        
        # Multi-factor severity assessment
        severity_score = max_confidence * 0.6 + (event_count / 10.0) * 0.3 + (unique_persons / 5.0) * 0.1
        
        if severity_score >= 0.9:
            return IncidentSeverity.CRITICAL
        elif severity_score >= 0.75:
            return IncidentSeverity.HIGH
        elif severity_score >= 0.6:
            return IncidentSeverity.MEDIUM
        else:
            return IncidentSeverity.LOW


class ReportGenerator:
    """
    Generates comprehensive incident reports.
    """
    
    def __init__(self):
        """
        Initialize report generator.
        """
        self.incident_counter = 0
    
    def generate_report(self, events: List[EventLog], 
                       severity: IncidentSeverity) -> IncidentReport:
        """
        Generate incident report from events.
        
        Args:
            events: List of events
            severity: Incident severity
            
        Returns:
            Incident report
        """
        involved_persons = list(set([p for e in events for p in e.person_ids]))
        evidence_frames = [e.frame_id for e in events]
        
        report = IncidentReport(
            incident_id=self.incident_counter,
            timestamp=datetime.now(),
            severity=severity,
            event_logs=events,
            involved_persons=involved_persons,
            location=self._compute_incident_location(events),
            description=self._generate_description(events, severity),
            evidence_frames=evidence_frames,
            recommended_actions=self._generate_recommendations(severity)
        )
        
        self.incident_counter += 1
        return report
    
    def _compute_incident_location(self, events: List[EventLog]) -> Tuple[int, int]:
        """
        Compute incident location from events.
        
        Args:
            events: List of events
            
        Returns:
            Incident location [x, y]
        """
        if not events:
            return (0, 0)
        
        locations = np.array([e.location for e in events])
        return tuple(np.mean(locations, axis=0).astype(int))
    
    def _generate_description(self, events: List[EventLog], 
                            severity: IncidentSeverity) -> str:
        """
        Generate incident description.
        
        Args:
            events: List of events
            severity: Incident severity
            
        Returns:
            Description string
        """
        event_types = set([e.event_type for e in events])
        person_count = len(set([p for e in events for p in e.person_ids]))
        
        description = f"Incident [{severity.value.upper()}]: "
        description += f"Detected {len(events)} suspicious events involving {person_count} persons. "
        description += f"Event types: {', '.join(event_types)}. "
        description += f"Average confidence: {np.mean([e.confidence for e in events]):.2f}"
        
        return description
    
    def _generate_recommendations(self, severity: IncidentSeverity) -> List[str]:
        """
        Generate recommended actions based on severity.
        
        Args:
            severity: Incident severity
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if severity == IncidentSeverity.CRITICAL:
            recommendations.extend([
                "Immediately alert security personnel",
                "Increase monitoring in affected area",
                "Record evidence video",
                "Prepare for intervention"
            ])
        elif severity == IncidentSeverity.HIGH:
            recommendations.extend([
                "Alert security personnel",
                "Monitor situation closely",
                "Be ready for escalation",
                "Record incident details"
            ])
        elif severity == IncidentSeverity.MEDIUM:
            recommendations.extend([
                "Log incident for review",
                "Monitor involved persons",
                "Increase attention to area"
            ])
        else:
            recommendations.extend([
                "Log incident",
                "Standard monitoring"
            ])
        
        return recommendations


class EventReporter:
    """
    Main event reporting service.
    """
    
    def __init__(self):
        """
        Initialize event reporter.
        """
        self.aggregator = EventAggregator()
        self.severity_classifier = SeverityClassifier()
        self.report_generator = ReportGenerator()
        self.incident_history: List[IncidentReport] = []
    
    def process_events(self, events: List[EventLog]) -> List[IncidentReport]:
        """
        Process events and generate reports.
        
        Args:
            events: List of detected events
            
        Returns:
            List of incident reports
        """
        # Add events to history
        for event in events:
            self.aggregator.add_event(event)
        
        # Cluster similar events
        clusters = self.aggregator.cluster_events()
        
        reports = []
        for cluster in clusters:
            # Classify severity
            severity = self.severity_classifier.classify(cluster)
            
            # Generate report
            report = self.report_generator.generate_report(cluster, severity)
            reports.append(report)
            self.incident_history.append(report)
        
        return reports
    
    def export_report(self, report: IncidentReport, format: str = 'json') -> str:
        """
        Export report in specified format.
        
        Args:
            report: Incident report
            format: Export format ('json', 'csv', 'text')
            
        Returns:
            Formatted report string
        """
        if format == 'json':
            return self._export_json(report)
        elif format == 'text':
            return self._export_text(report)
        else:
            return self._export_text(report)
    
    def _export_text(self, report: IncidentReport) -> str:
        """
        Export report as text.
        
        Args:
            report: Incident report
            
        Returns:
            Text report
        """
        text = f"\n{'='*60}\n"
        text += f"INCIDENT REPORT #{report.incident_id}\n"
        text += f"{'='*60}\n"
        text += f"Timestamp: {report.timestamp}\n"
        text += f"Severity: {report.severity.value.upper()}\n"
        text += f"Location: {report.location}\n"
        text += f"Involved Persons: {len(report.involved_persons)}\n"
        text += f"\nDescription:\n{report.description}\n"
        text += f"\nRecommended Actions:\n"
        for action in report.recommended_actions:
            text += f"  - {action}\n"
        text += f"\nEvidence Frames: {report.evidence_frames[:5]}...\n"
        text += f"{'='*60}\n"
        
        return text
    
    def _export_json(self, report: IncidentReport) -> str:
        """
        Export report as JSON.
        
        Args:
            report: Incident report
            
        Returns:
            JSON report string
        """
        import json
        
        report_dict = {
            'incident_id': report.incident_id,
            'timestamp': report.timestamp.isoformat(),
            'severity': report.severity.value,
            'location': report.location,
            'involved_persons': report.involved_persons,
            'description': report.description,
            'recommendations': report.recommended_actions,
            'event_count': len(report.event_logs),
            'evidence_frames': report.evidence_frames
        }
        
        return json.dumps(report_dict, indent=2)
