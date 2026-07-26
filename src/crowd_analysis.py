"""
Crowd Analysis Module
Analyzes crowd density, person selection, and SCIP processing
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CrowdMetrics:
    """Crowd analysis metrics."""
    density: float  # Persons per unit area
    total_persons: int
    high_density_regions: List[Tuple[int, int, int, int]]
    person_ids: List[int]


class CrowdDensityEstimator:
    """
    Estimates crowd density and identifies dense regions.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (640, 480), 
                 cell_size: Tuple[int, int] = (64, 48)):
        """
        Initialize crowd density estimator.
        
        Args:
            grid_size: Frame dimensions
            cell_size: Size of grid cells for density estimation
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_h = grid_size[1] // cell_size[1]
        self.grid_w = grid_size[0] // cell_size[0]
    
    def create_density_map(self, persons: List[Dict]) -> np.ndarray:
        """
        Create crowd density map.
        
        Args:
            persons: List of persons with bboxes
            
        Returns:
            Density map (grid)
        """
        density_map = np.zeros((self.grid_h, self.grid_w))
        
        for person in persons:
            bbox = person['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            cell_x = int(center_x / self.cell_size[0])
            cell_y = int(center_y / self.cell_size[1])
            
            if 0 <= cell_x < self.grid_w and 0 <= cell_y < self.grid_h:
                density_map[cell_y, cell_x] += 1
        
        return density_map
    
    def identify_high_density_regions(self, density_map: np.ndarray, 
                                     threshold: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        Identify high density regions.
        
        Args:
            density_map: Crowd density map
            threshold: Minimum persons per cell
            
        Returns:
            List of high density regions as bboxes
        """
        regions = []
        
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if density_map[y, x] >= threshold:
                    x1 = x * self.cell_size[0]
                    y1 = y * self.cell_size[1]
                    x2 = (x + 1) * self.cell_size[0]
                    y2 = (y + 1) * self.cell_size[1]
                    regions.append((x1, y1, x2, y2))
        
        return regions
    
    def estimate_global_density(self, persons: List[Dict]) -> float:
        """
        Estimate overall crowd density.
        
        Args:
            persons: List of persons
            
        Returns:
            Density value (persons per unit area)
        """
        total_area = self.grid_size[0] * self.grid_size[1]
        return len(persons) / total_area


class PersonSelector:
    """
    Selects persons of interest based on various criteria.
    Used for anomaly detection in crowd scenarios.
    """
    
    def __init__(self):
        """Initialize person selector."""
        pass
    
    def select_by_behavior(self, persons: List[Dict]) -> List[int]:
        """
        Select persons with unusual behavior.
        
        Args:
            persons: List of persons
            
        Returns:
            List of selected person IDs
        """
        selected = []
        
        for person in persons:
            # Check for unusual motion, stopped behavior, etc.
            if 'velocity' in person and np.linalg.norm(person['velocity']) < 5:
                selected.append(person['id'])
        
        return selected
    
    def select_by_isolation(self, persons: List[Dict], 
                           isolation_distance: float = 150.0) -> List[int]:
        """
        Select persons isolated from crowd.
        
        Args:
            persons: List of persons
            isolation_distance: Distance threshold
            
        Returns:
            List of selected person IDs
        """
        selected = []
        
        for person in persons:
            center = np.array([(person['bbox'][0] + person['bbox'][2]) / 2,
                            (person['bbox'][1] + person['bbox'][3]) / 2])
            
            # Count neighbors
            neighbors = 0
            for other in persons:
                if other['id'] != person['id']:
                    other_center = np.array([(other['bbox'][0] + other['bbox'][2]) / 2,
                                           (other['bbox'][1] + other['bbox'][3]) / 2])
                    distance = np.linalg.norm(center - other_center)
                    if distance < isolation_distance:
                        neighbors += 1
            
            # If person is isolated
            if neighbors < 2:
                selected.append(person['id'])
        
        return selected
    
    def select_high_priority(self, persons: List[Dict]) -> List[int]:
        """
        Select high priority persons for analysis.
        
        Args:
            persons: List of persons
            
        Returns:
            List of selected person IDs
        """
        selected = set()
        
        # Add behaviorally unusual persons
        selected.update(self.select_by_behavior(persons))
        
        # Add isolated persons
        selected.update(self.select_by_isolation(persons))
        
        return list(selected)


class SCIPProcessor:
    """
    Processes SCIP (Spatio-temporal Context Image Processing) frames.
    Analyzes frames using spatial context information.
    """
    
    def __init__(self, frame_size: Tuple[int, int] = (640, 480)):
        """
        Initialize SCIP processor.
        
        Args:
            frame_size: Frame dimensions
        """
        self.frame_size = frame_size
    
    def compute_context_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute spatial context features from frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Context feature map
        """
        # Compute edge information
        gray = frame.mean(axis=2) if len(frame.shape) == 3 else frame
        
        # Simple edge detection using gradients
        edges_x = np.gradient(gray, axis=1)
        edges_y = np.gradient(gray, axis=0)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        
        return edge_magnitude
    
    def segment_frame(self, frame: np.ndarray, num_segments: int = 4) -> List[np.ndarray]:
        """
        Segment frame into regions.
        
        Args:
            frame: Input frame
            num_segments: Number of segments per dimension
            
        Returns:
            List of frame segments
        """
        h, w = frame.shape[:2]
        seg_h = h // num_segments
        seg_w = w // num_segments
        
        segments = []
        for i in range(num_segments):
            for j in range(num_segments):
                y1 = i * seg_h
                x1 = j * seg_w
                y2 = (i + 1) * seg_h if i < num_segments - 1 else h
                x2 = (j + 1) * seg_w if j < num_segments - 1 else w
                
                segment = frame[y1:y2, x1:x2]
                segments.append(segment)
        
        return segments
    
    def extract_spatial_features(self, frame: np.ndarray) -> Dict:
        """
        Extract comprehensive spatial features.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary of spatial features
        """
        context = self.compute_context_features(frame)
        segments = self.segment_frame(frame)
        
        features = {
            'context': context,
            'segments': segments,
            'edge_density': np.mean(context > np.mean(context)),
            'num_segments': len(segments)
        }
        
        return features


class CrowdAnalyzer:
    """
    Main crowd analysis service.
    """
    
    def __init__(self):
        """
        Initialize crowd analyzer.
        """
        self.density_estimator = CrowdDensityEstimator()
        self.person_selector = PersonSelector()
        self.scip_processor = SCIPProcessor()
    
    def analyze(self, frame: np.ndarray, persons: List[Dict]) -> CrowdMetrics:
        """
        Analyze crowd in frame.
        
        Args:
            frame: Input frame
            persons: List of detected persons
            
        Returns:
            Crowd metrics
        """
        # Create density map
        density_map = self.density_estimator.create_density_map(persons)
        
        # Identify high density regions
        high_density = self.density_estimator.identify_high_density_regions(density_map)
        
        # Estimate global density
        global_density = self.density_estimator.estimate_global_density(persons)
        
        # Select persons of interest
        selected_ids = self.person_selector.select_high_priority(persons)
        
        metrics = CrowdMetrics(
            density=global_density,
            total_persons=len(persons),
            high_density_regions=high_density,
            person_ids=selected_ids
        )
        
        return metrics
