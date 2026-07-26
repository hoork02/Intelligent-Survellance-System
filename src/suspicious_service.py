"""
Suspicious Service Module
Detects suspicious behaviors: adult-child pairs, hand contact, and witness analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class AgeGroup(Enum):
    """Age group categories."""
    CHILD = "child"
    ADULT = "adult"
    UNKNOWN = "unknown"


class ContactType(Enum):
    """Types of physical contact."""
    HAND_TO_CHILD = "hand_to_child"
    HAND_TO_ADULT = "hand_to_adult"
    BODY_CONTACT = "body_contact"
    NONE = "none"


@dataclass
class PersonInstance:
    """Instance of a person in frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # [x1, y1, x2, y2]
    keypoints: np.ndarray  # Detected keypoints
    age_group: AgeGroup
    confidence: float


@dataclass
class SuspiciousEvent:
    """Represents a suspicious event."""
    event_id: int
    event_type: str  # 'adult-child-pair', 'hand-contact', etc.
    person1_id: int
    person2_id: int
    frame_range: Tuple[int, int]  # [start_frame, end_frame]
    confidence: float
    details: Dict


class AgeClassifier:
    """
    Classifies persons into age groups (child/adult).
    Based on bounding box height and other features.
    """
    
    def __init__(self, child_height_threshold: float = 100):
        """
        Initialize age classifier.
        
        Args:
            child_height_threshold: Height threshold for child detection
        """
        self.child_height_threshold = child_height_threshold
    
    def classify(self, bbox: Tuple[int, int, int, int]) -> Tuple[AgeGroup, float]:
        """
        Classify person's age group.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            (AgeGroup, confidence)
        """
        height = bbox[3] - bbox[1]
        
        # Simple heuristic: shorter people are likely children
        if height < self.child_height_threshold:
            confidence = min(1.0, 1.0 - (height / self.child_height_threshold))
            return AgeGroup.CHILD, confidence
        else:
            confidence = min(1.0, height / 200.0)
            return AgeGroup.ADULT, confidence
    
    def classify_persons(self, persons: List[PersonInstance]) -> List[PersonInstance]:
        """
        Classify age group for all persons.
        
        Args:
            persons: List of persons
            
        Returns:
            Updated persons list with age classification
        """
        for person in persons:
            age_group, confidence = self.classify(person.bbox)
            person.age_group = age_group
            person.confidence = confidence
        return persons


class AdultChildPairIdentifier:
    """
    Identifies suspicious adult-child pairs.
    """
    
    def __init__(self, proximity_threshold: float = 100.0):
        """
        Initialize adult-child pair identifier.
        
        Args:
            proximity_threshold: Distance threshold for pair detection
        """
        self.proximity_threshold = proximity_threshold
    
    def compute_distance(self, bbox1: Tuple[int, int, int, int], 
                        bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute distance between two persons.
        
        Args:
            bbox1: First bbox
            bbox2: Second bbox
            
        Returns:
            Euclidean distance between centers
        """
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        return np.linalg.norm(center1 - center2)
    
    def identify_pairs(self, persons: List[PersonInstance]) -> List[Tuple[PersonInstance, PersonInstance]]:
        """
        Identify suspicious adult-child pairs.
        
        Args:
            persons: List of classified persons
            
        Returns:
            List of (adult, child) pairs
        """
        pairs = []
        
        for i, person1 in enumerate(persons):
            for j, person2 in enumerate(persons[i+1:], i+1):
                # Check if one is child and other is adult
                if ((person1.age_group == AgeGroup.CHILD and person2.age_group == AgeGroup.ADULT) or
                    (person1.age_group == AgeGroup.ADULT and person2.age_group == AgeGroup.CHILD)):
                    
                    distance = self.compute_distance(person1.bbox, person2.bbox)
                    if distance < self.proximity_threshold:
                        pairs.append((person1, person2))
        
        return pairs


class HandContactDetector:
    """
    Detects hand-to-person contact events.
    Analyzes keypoint positions to detect potential harmful contact.
    """
    
    def __init__(self, contact_threshold: float = 30.0):
        """
        Initialize hand contact detector.
        
        Args:
            contact_threshold: Distance threshold for contact detection
        """
        self.contact_threshold = contact_threshold
        # Hand keypoint indices (assuming OpenPose format)
        self.hand_keypoints = [4, 7]  # Left and right hand
        # Body part keypoints that indicate vulnerable areas
        self.vulnerable_keypoints = [0, 1, 2, 3]  # Head, neck, shoulders
    
    def extract_hand_positions(self, keypoints: np.ndarray) -> List[np.ndarray]:
        """
        Extract hand positions from keypoints.
        
        Args:
            keypoints: Array of keypoints [x, y, confidence] for each joint
            
        Returns:
            List of hand positions
        """
        hands = []
        for idx in self.hand_keypoints:
            if idx < len(keypoints):
                hands.append(keypoints[idx][:2])
        return hands
    
    def detect_contact(self, person1_keypoints: np.ndarray, 
                      person2_keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        Detect hand contact between two persons.
        
        Args:
            person1_keypoints: Keypoints of person 1
            person2_keypoints: Keypoints of person 2
            
        Returns:
            (contact_detected, confidence)
        """
        person1_hands = self.extract_hand_positions(person1_keypoints)
        person2_vulnerable = [person2_keypoints[idx][:2] for idx in self.vulnerable_keypoints 
                            if idx < len(person2_keypoints)]
        
        min_distance = float('inf')
        
        for hand in person1_hands:
            if len(hand) > 0 and hand[2] > 0.3:  # Confidence check
                for vulnerable in person2_vulnerable:
                    if len(vulnerable) > 0 and vulnerable[2] > 0.3:
                        dist = np.linalg.norm(hand[:2] - vulnerable[:2])
                        min_distance = min(min_distance, dist)
        
        contact_detected = min_distance < self.contact_threshold
        confidence = max(0, 1.0 - (min_distance / self.contact_threshold)) if min_distance < float('inf') else 0.0
        
        return contact_detected, confidence


class WitnessAnalyzer:
    """
    Analyzes nearby persons and their reactions as potential witnesses.
    """
    
    def __init__(self, witness_radius: float = 200.0):
        """
        Initialize witness analyzer.
        
        Args:
            witness_radius: Radius within which to consider witnesses
        """
        self.witness_radius = witness_radius
    
    def find_nearby_persons(self, center_bbox: Tuple[int, int, int, int], 
                           all_persons: List[PersonInstance]) -> List[PersonInstance]:
        """
        Find persons within witness radius.
        
        Args:
            center_bbox: Reference bounding box
            all_persons: List of all persons
            
        Returns:
            List of nearby persons (potential witnesses)
        """
        center = np.array([(center_bbox[0] + center_bbox[2]) / 2, 
                          (center_bbox[1] + center_bbox[3]) / 2])
        
        nearby = []
        for person in all_persons:
            person_center = np.array([(person.bbox[0] + person.bbox[2]) / 2,
                                     (person.bbox[1] + person.bbox[3]) / 2])
            distance = np.linalg.norm(center - person_center)
            
            if distance < self.witness_radius and distance > 0:
                nearby.append(person)
        
        return nearby
    
    def analyze_witness_reaction(self, witness: PersonInstance, 
                                event_persons: List[PersonInstance]) -> Dict:
        """
        Analyze witness's reaction to event.
        
        Args:
            witness: Witness person
            event_persons: Persons involved in event
            
        Returns:
            Dictionary with witness analysis
        """
        return {
            'witness_id': witness.track_id,
            'is_looking_at_event': True,  # Would use gaze detection in real implementation
            'reaction_type': 'passive',  # Would classify actual reaction
            'confidence': 0.7
        }


class SuspiciousService:
    """
    Main service for suspicious behavior detection.
    """
    
    def __init__(self):
        """
        Initialize suspicious service.
        """
        self.age_classifier = AgeClassifier()
        self.pair_identifier = AdultChildPairIdentifier()
        self.contact_detector = HandContactDetector()
        self.witness_analyzer = WitnessAnalyzer()
        
        self.event_counter = 0
        self.active_events: Dict[int, SuspiciousEvent] = {}
    
    def detect_suspicious_events(self, persons: List[PersonInstance], 
                                frame_idx: int) -> List[SuspiciousEvent]:
        """
        Detect all suspicious events in frame.
        
        Args:
            persons: List of detected persons
            frame_idx: Current frame index
            
        Returns:
            List of detected suspicious events
        """
        events = []
        
        # Classify ages
        self.age_classifier.classify_persons(persons)
        
        # Identify adult-child pairs
        pairs = self.pair_identifier.identify_pairs(persons)
        
        for person1, person2 in pairs:
            adult = person1 if person1.age_group == AgeGroup.ADULT else person2
            child = person2 if person1.age_group == AgeGroup.ADULT else person1
            
            # Check for hand contact
            if person1.keypoints is not None and person2.keypoints is not None:
                contact_detected, contact_conf = self.contact_detector.detect_contact(
                    person1.keypoints, person2.keypoints
                )
                
                if contact_detected:
                    event = SuspiciousEvent(
                        event_id=self.event_counter,
                        event_type='hand-contact',
                        person1_id=adult.track_id,
                        person2_id=child.track_id,
                        frame_range=(frame_idx, frame_idx),
                        confidence=contact_conf,
                        details={'contact_type': 'hand-to-child'}
                    )
                    events.append(event)
                    self.event_counter += 1
            
            # Analyze witnesses
            witnesses = self.witness_analyzer.find_nearby_persons(
                adult.bbox, [p for p in persons if p.track_id not in [adult.track_id, child.track_id]]
            )
        
        return events
