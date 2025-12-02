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

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    """
    Enhanced Vehicle Tracker using YOLOv11 + DeepSORT - Military Vehicles Only
    
    Features:
    - Military vehicle detection and tracking only (priority > 8)
    - Frame-by-frame data storage to local machine
    - Comprehensive analytics and reporting
    - Real-time visualization with performance metrics
    - Accuracy metrics: FP, FN, IDS, MOTA, MOTP
    """

    def __init__(self, yolo_model, confidence=0.467, local_folder=None):
        print("Initializing Enhanced Military Vehicle Tracker...")

        if isinstance(yolo_model, str):
            self.model = YOLO(yolo_model)
        else:
            self.model = yolo_model

        self.confidence = confidence
        self.class_names = self.model.names

        self.tracker = DeepSort(
            max_age=50,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            embedder_gpu=torch.cuda.is_available(),
            bgr=True,
            half=True if torch.cuda.is_available() else False
        )

        self.vehicle_config = {
            'Car': {'color': (255, 0, 0), 'priority': 8, 'type': 'CIVILIAN_CAR', 'threat_level': 'LOW'},
            'Truck': {'color': (0, 128, 255), 'priority': 7, 'type': 'TRANSPORT', 'threat_level': 'MEDIUM'},
            'Bus': {'color': (0, 255, 255), 'priority': 6, 'type': 'PUBLIC_TRANSPORT', 'threat_level': 'LOW'},
            'Military Truck': {'color': (0, 100, 0), 'priority': 10, 'type': 'MILITARY_VEHICLE', 'threat_level': 'HIGH'},
            'Tank': {'color': (128, 128, 128), 'priority': 10, 'type': 'MAIN_BATTLE_TANK', 'threat_level': 'VERY_HIGH'},
            'Armored Tank': {'color': (64, 64, 64), 'priority': 10, 'type': 'ARMORED_VEHICLE', 'threat_level': 'VERY_HIGH'}
        }
        
        # Military vehicle filter
        self.military_classes = {'Military Truck', 'Tank', 'Armored Tank'}

        self.local_folder = local_folder or f"./MilitaryVehicleTracking_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            'total_iou': 0.0,
            'matched_objects': 0,
            'previous_track_ids': set(),
            'current_track_ids': set(),
        }

        self.frame_detections = []
        self.frame_tracks = []
        self.frame_analytics = []

        print(f"Military Vehicle Tracker Initialized:")
        print(f"   • Custom Vehicle Classes: {len(self.class_names)}")
        print(f"   • Military Classes Only: {self.military_classes}")
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

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

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
        """Enhanced frame processing - Military vehicles only with priority > 8"""
        frame_start = time.time()
        
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        detections = []
        frame_detections_data = {
            'frame_number': frame_number,
            'timestamp': time.time(),
            'detections': []
        }
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
                conf = float(box.conf.cpu().numpy()[0])
                cls_id = int(box.cls.cpu().numpy()[0])
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
                        
                        # Only process military vehicles with priority > 8
                        if cls_name in self.military_classes and vehicle_config['priority'] > 8:
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
                            detections.append(([x1, y1, w, h], conf, detection_data))
        
        raw_detections = [(det[0], det[1], 0) for det in detections]
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        self.update_accuracy_metrics(detections, tracks)
        
        frame_tracks_data = {
            'frame_number': frame_number,
            'timestamp': time.time(),
            'tracks': []
        }
        
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]
        
        for i, track in enumerate(tracks):
            if i < len(detections):
                track.detection_data = detections[i][2]
            elif not hasattr(track, 'detection_data'):
                track.detection_data = {
                    'class_name': 'unknown',
                    'confidence': 0.0,
                    'color': (128, 128, 128),
                    'priority': 1,
                    'vehicle_type': 'UNKNOWN',
                    'threat_level': 'UNKNOWN',
                    'bbox': [0, 0, 0, 0]
                }
        
        for track in confirmed_tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # Only process military vehicles with priority > 8
            class_name = getattr(track, 'detection_data', {}).get('class_name', 'unknown')
            priority = getattr(track, 'detection_data', {}).get('priority', 0)
            
            if class_name in self.military_classes and priority > 8:
                self.stats['track_history'][track_id].append(center)
                if len(self.stats['track_history'][track_id]) > 30:
                    self.stats['track_history'][track_id].pop(0)
                
                track_data = {
                    'track_id': track_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center[0]), float(center[1])],
                    'class_name': class_name,
                    'confidence': getattr(track, 'detection_data', {}).get('confidence', 0.0),
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
        self.stats['active_tracks'] = len(frame_tracks_data['tracks'])
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
            'track_count': len(frame_tracks_data['tracks']),
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
        """Enhanced visualization - Military vehicles only"""
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]
        
        # Filter only military vehicles with priority > 8
        military_tracks = []
        for track in confirmed_tracks:
            class_name = getattr(track, 'detection_data', {}).get('class_name', 'unknown')
            priority = getattr(track, 'detection_data', {}).get('priority', 0)
            if class_name in self.military_classes and priority > 8:
                military_tracks.append(track)
        
        for track in military_tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            
            data = getattr(track, 'detection_data', {})
            class_name = data.get('class_name', 'unknown')
            color = data.get('color', (128, 128, 128))
            conf = data.get('confidence', 0.0)
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
            
            label = f"{label} [{threat_level}] P:{priority}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 15), (x1 + label_w + 10, y1 - 2), threat_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, center, 4, threat_color, -1)
            cv2.circle(frame, center, 4, (255, 255, 255), 1)
            
            if track_id in self.stats['track_history']:
                trail = self.stats['track_history'][track_id]
                if len(trail) > 1:
                    for i in range(1, len(trail)):
                        alpha = i / len(trail)
                        trail_color = tuple(int(c * alpha) for c in threat_color)
                        cv2.line(frame, tuple(map(int, trail[i-1])), tuple(map(int, trail[i])), trail_color, 2)
        
        return frame

    def add_performance_overlay(self, frame, fps=0):
        """Enhanced overlay - Military vehicles only"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        overlay_height = min(300, h // 3)
        cv2.rectangle(overlay, (10, 10), (min(1200, w-10), overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        elapsed = time.time() - self.stats['start_time']
        avg_fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        avg_processing = np.mean(self.stats['processing_times']) * 1000 if self.stats['processing_times'] else 0
        
        mota = self.calculate_mota()
        motp = self.calculate_motp()
        
        title = "Military Vehicle Tracking System - Priority > 8 Only"
        cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        info_lines = [
            f"Military Only | Confidence: {self.confidence} | GPU: {torch.cuda.is_available()} | Session: {self.stats['session_id']}",
            f"Frame: {self.stats['frames_processed']} | FPS: {fps:.1f} | Avg FPS: {avg_fps:.1f} | Processing: {avg_processing:.1f}ms",
            f"Active Military Tracks: {self.stats['active_tracks']} | Max: {self.stats['max_concurrent_tracks']} | Total Detections: {self.stats['total_detections']}",
            f"Military Vehicles Only | Local Storage: {len(self.frame_detections)} frames saved | Runtime: {elapsed:.0f}s",
            f"MOTA: {mota:.2f}% | MOTP: {motp:.2f}% | FP: {self.stats['false_positives']} | FN: {self.stats['false_negatives']} | IDS: {self.stats['identity_switches']}"
        ]
        
        for i, line in enumerate(info_lines):
            y = 65 + i * 25
            color = (255, 255, 255) if i > 0 else (255, 255, 0)
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        if w > 1000:
            threat_x = w - 400
            cv2.putText(frame, "Military Threat Analysis:", (threat_x, 40),
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
            
            cv2.putText(frame, "Military Vehicles:", (threat_x, y_offset + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 40
            for cls_name, count in list(self.stats['class_counts'].items())[:6]:
                if cls_name in self.military_classes:
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

    def calculate_motp(self):
        """Calculate Multiple Object Tracking Precision (MOTP)"""
        if self.stats['matched_objects'] > 0:
            motp = self.stats['total_iou'] / self.stats['matched_objects']
            return motp * 100
        return 0

    def save_frame_data(self, frame_number, save_interval=50):
        """Save accumulated data locally at intervals"""
        if frame_number % save_interval == 0 and self.frame_detections:
            try:
                detections_file = os.path.join(self.subfolders['data'], f'military_detections_frames_{frame_number-save_interval+1}_to_{frame_number}.json')
                with open(detections_file, 'w') as f:
                    json.dump(self.frame_detections[-save_interval:], f, indent=2)
                
                tracks_file = os.path.join(self.subfolders['data'], f'military_tracks_frames_{frame_number-save_interval+1}_to_{frame_number}.json')
                with open(tracks_file, 'w') as f:
                    json.dump(self.frame_tracks[-save_interval:], f, indent=2)
                
                analytics_file = os.path.join(self.subfolders['analytics'], f'military_analytics_frames_{frame_number-save_interval+1}_to_{frame_number}.json')
                with open(analytics_file, 'w') as f:
                    json.dump(self.frame_analytics[-save_interval:], f, indent=2)
                
                print(f"   • Military vehicle data saved locally for frames {frame_number-save_interval+1} to {frame_number}")
            
            except Exception as e:
                print(f"   • Warning: Could not save data locally: {e}")

    def save_final_results(self, output_video_path):
        """Save complete results and generate comprehensive report"""
        print("Saving complete military vehicle results locally...")
        
        try:
            local_video_path = os.path.join(self.subfolders['videos'], os.path.basename(output_video_path))
            if output_video_path != local_video_path:
                shutil.copy2(output_video_path, local_video_path)
            
            complete_detections_file = os.path.join(self.subfolders['raw_data'], 'military_complete_detections.json')
            with open(complete_detections_file, 'w') as f:
                json.dump(self.frame_detections, f, indent=2)
            
            complete_tracks_file = os.path.join(self.subfolders['raw_data'], 'military_complete_tracks.json')
            with open(complete_tracks_file, 'w') as f:
                json.dump(self.frame_tracks, f, indent=2)
            
            complete_analytics_file = os.path.join(self.subfolders['raw_data'], 'military_complete_analytics.json')
            with open(complete_analytics_file, 'w') as f:
                json.dump(self.frame_analytics, f, indent=2)
            
            track_data_file = os.path.join(self.subfolders['raw_data'], 'military_track_data.pkl')
            with open(track_data_file, 'wb') as f:
                pickle.dump(dict(self.stats['track_data']), f)
            
            report = self.generate_comprehensive_report()
            
            report_json_file = os.path.join(self.subfolders['reports'], 'military_comprehensive_report.json')
            with open(report_json_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            report_text_file = os.path.join(self.subfolders['reports'], 'military_comprehensive_report.txt')
            with open(report_text_file, 'w') as f:
                f.write(self.format_report_text(report))
            
            self.create_summary_csv()
            
            print(f"All military vehicle data saved to: {self.local_folder}")
            return report
        
        except Exception as e:
            print(f"Error saving results: {e}")
            return None

    def generate_comprehensive_report(self):
        """Generate detailed analysis report"""
        elapsed = time.time() - self.stats['start_time']
        mota = self.calculate_mota()
        motp = self.calculate_motp()
        
        report = {
            'session_info': {
                'session_id': self.stats['session_id'],
                'total_runtime_seconds': elapsed,
                'frames_processed': self.stats['frames_processed'],
                'average_fps': self.stats['frames_processed'] / elapsed if elapsed > 0 else 0,
                'processing_date': datetime.now().isoformat()
            },
            'detection_summary': {
                'total_military_detections': self.stats['total_detections'],
                'military_class_distribution': dict(self.stats['class_counts']),
                'threat_level_distribution': dict(self.stats['threat_level_counts']),
                'detection_rate_per_frame': self.stats['total_detections'] / self.stats['frames_processed'] if self.stats['frames_processed'] > 0 else 0
            },
            'tracking_summary': {
                'total_military_tracks': len(self.stats['track_data']),
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
                'mota_percentage': mota,
                'motp_percentage': motp
            },
            'military_threat_analysis': {
                'high_threat_military_vehicles': sum(1 for data in self.stats['track_data'].values() if 'VERY_HIGH' in str(data.get('threat_level', '')) or 'HIGH' in str(data.get('threat_level', ''))),
                'total_military_vehicles_detected': len(self.stats['track_data']),
                'tanks_detected': sum(1 for data in self.stats['track_data'].values() if 'TANK' in str(data.get('vehicle_type', ''))),
                'military_trucks_detected': sum(1 for data in self.stats['track_data'].values() if 'MILITARY' in str(data.get('vehicle_type', '')))
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
MILITARY VEHICLE TRACKING COMPREHENSIVE REPORT
==============================================
Session ID: {report['session_info']['session_id']}
Processing Date: {report['session_info']['processing_date']}

PERFORMANCE SUMMARY
==================
• Total Runtime: {report['session_info']['total_runtime_seconds']:.2f} seconds
• Frames Processed: {report['session_info']['frames_processed']}
• Average FPS: {report['session_info']['average_fps']:.2f}
• Average Processing Time: {report['performance_metrics']['avg_processing_time_ms']:.2f}ms per frame

MILITARY VEHICLE DETECTION SUMMARY (Priority > 8)
=================================================
• Total Military Detections: {report['detection_summary']['total_military_detections']}
• Detection Rate: {report['detection_summary']['detection_rate_per_frame']:.2f} detections per frame

Military Vehicle Classes:
"""
        for class_name, count in report['detection_summary']['military_class_distribution'].items():
            percentage = (count / report['detection_summary']['total_military_detections']) * 100 if report['detection_summary']['total_military_detections'] > 0 else 0
            text += f"  • {class_name}: {count} ({percentage:.1f}%)\n"
        
        text += f"""
Threat Level Distribution:
"""
        for threat_level, count in report['detection_summary']['threat_level_distribution'].items():
            percentage = (count / report['detection_summary']['total_military_detections']) * 100 if report['detection_summary']['total_military_detections'] > 0 else 0
            text += f"  • {threat_level}: {count} ({percentage:.1f}%)\n"
        
        text += f"""
MILITARY TRACKING SUMMARY
=========================
• Total Unique Military Tracks: {report['tracking_summary']['total_military_tracks']}
• Max Concurrent Tracks: {report['tracking_summary']['max_concurrent_tracks']}
• Average Track Duration: {report['tracking_summary']['average_track_duration']:.1f} frames
• Longest Track Duration: {report['tracking_summary']['longest_track_duration']} frames

ACCURACY METRICS
===============
• False Positives (FP): {report['accuracy_metrics']['false_positives']}
• False Negatives (FN): {report['accuracy_metrics']['false_negatives']}
• Identity Switches (IDS): {report['accuracy_metrics']['identity_switches']}
• Multiple Object Tracking Accuracy (MOTA): {report['accuracy_metrics']['mota_percentage']:.2f}%
• Multiple Object Tracking Precision (MOTP): {report['accuracy_metrics']['motp_percentage']:.2f}%

MILITARY THREAT ANALYSIS
========================
• High Threat Military Vehicles: {report['military_threat_analysis']['high_threat_military_vehicles']}
• Total Military Vehicles: {report['military_threat_analysis']['total_military_vehicles_detected']}
• Tanks Detected: {report['military_threat_analysis']['tanks_detected']}
• Military Trucks Detected: {report['military_threat_analysis']['military_trucks_detected']}

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
            frame_summary = []
            for analytics in self.frame_analytics:
                frame_summary.append({
                    'frame_number': analytics['frame_number'],
                    'processing_time_ms': analytics['processing_time_ms'],
                    'military_detection_count': analytics['detection_count'],
                    'military_track_count': analytics['track_count'],
                    'high_threat_count': analytics['threat_detections']['HIGH'] + analytics['threat_detections']['VERY_HIGH'],
                    'medium_threat_count': analytics['threat_detections']['MEDIUM'],
                    'low_threat_count': analytics['threat_detections']['LOW']
                })
            
            df_frames = pd.DataFrame(frame_summary)
            df_frames.to_csv(os.path.join(self.subfolders['analytics'], 'military_frame_summary.csv'), index=False)
            
            track_summary = []
            for track_id, data in self.stats['track_data'].items():
                track_summary.append({
                    'track_id': track_id,
                    'military_class_name': data['class_name'],
                    'vehicle_type': data['vehicle_type'],
                    'threat_level': data['threat_level'],
                    'first_seen_frame': data['first_seen'],
                    'last_seen_frame': data['last_seen'],
                    'duration_frames': data['total_frames'],
                    'max_confidence': data['max_confidence']
                })
            
            df_tracks = pd.DataFrame(track_summary)
            df_tracks.to_csv(os.path.join(self.subfolders['analytics'], 'military_track_summary.csv'), index=False)
            
            accuracy_summary = [{
                'false_positives': self.stats['false_positives'],
                'false_negatives': self.stats['false_negatives'],
                'identity_switches': self.stats['identity_switches'],
                'mota_percentage': self.calculate_mota(),
                'motp_percentage': self.calculate_motp()
            }]
            
            df_accuracy = pd.DataFrame(accuracy_summary)
            df_accuracy.to_csv(os.path.join(self.subfolders['analytics'], 'military_accuracy_summary.csv'), index=False)
            
            print("Military vehicle CSV summaries created successfully")
        
        except Exception as e:
            print(f"Error creating CSV summaries: {e}")


def process_video_with_enhanced_tracking(model_path, video_path, max_frames=400, confidence=0.467, save_interval=50):
    """
    Complete video processing with enhanced DeepSORT tracking - Military vehicles only
    """
    print("="*80)
    print("MILITARY VEHICLE TRACKING SYSTEM")
    print("Custom Model: Military Vehicle Detection Only (Priority > 8)")
    print("Storage: Local Machine")
    print("Accuracy Metrics: FP, FN, IDS, MOTA, MOTP")
    print("="*80)
    
    tracker = VehicleTracker(model_path, confidence=confidence)
    
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Properties:")
    print(f"   • Resolution: {width}x{height}")
    print(f"   • FPS: {fps:.1f}")
    print(f"   • Total Frames: {total_frames}")
    print(f"   • Processing Frames: {min(max_frames, total_frames)}")
    print(f"   • Local Save Interval: {save_interval} frames")
    
    output_path = os.path.join(tracker.subfolders['videos'], f'military_tracked_{Path(video_path).stem}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    display_interval = max(1, max_frames // 20)
    
    try:
        print("\nStarting Military Vehicle Tracking with Local Storage...")
        print("-" * 80)
        
        while frame_count < min(max_frames, total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            tracks, detections = tracker.process_frame(frame, frame_count)
            frame = tracker.draw_tracking_results(frame, tracks)
            
            current_time = time.time()
            current_fps = frame_count / (current_time - start_time)
            frame = tracker.add_performance_overlay(frame, current_fps)
            
            out.write(frame)
            tracker.save_frame_data(frame_count, save_interval)
            
            if frame_count % display_interval == 0 or frame_count <= 5:
                progress = (frame_count / max_frames) * 100
                mota = tracker.calculate_mota()
                motp = tracker.calculate_motp()
                
                print("=" * 80)
                print(f"MILITARY VEHICLE TRACKING PROGRESS: {frame_count}/{max_frames} ({progress:.1f}%)")
                print("=" * 80)
                print(f"PERFORMANCE METRICS:")
                print(f"   • Current FPS: {current_fps:.2f}")
                print(f"   • Active Military Tracks: {tracker.stats['active_tracks']}")
                print(f"   • Total Military Detections: {tracker.stats['total_detections']}")
                print(f"   • Data Records Saved: {len(tracker.frame_detections)}")
                
                print(f"\nACCURACY METRICS:")
                print(f"   • False Positives (FP): {tracker.stats['false_positives']}")
                print(f"   • False Negatives (FN): {tracker.stats['false_negatives']}")
                print(f"   • Identity Switches (IDS): {tracker.stats['identity_switches']}")
                print(f"   • MOTA: {mota:.2f}%")
                print(f"   • MOTP: {motp:.2f}%")
                
                if tracker.stats['threat_level_counts']:
                    print(f"\nMILITARY THREAT LEVEL SUMMARY:")
                    for threat, count in tracker.stats['threat_level_counts'].items():
                        if count > 0:
                            print(f"   • {threat}: {count}")
                
                if tracker.stats['class_counts']:
                    print(f"\nMILITARY VEHICLE CLASSES DETECTED:")
                    for cls_name, count in list(tracker.stats['class_counts'].items())[:5]:
                        vehicle_config = tracker.vehicle_config.get(cls_name, {})
                        vehicle_type = vehicle_config.get('type', 'UNKNOWN')
                        priority = vehicle_config.get('priority', 0)
                        print(f"   • {cls_name} ({vehicle_type}) Priority:{priority}: {count}")
                
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
    
    print("\n" + "="*80)
    print("SAVING FINAL MILITARY VEHICLE RESULTS TO LOCAL MACHINE")
    print("="*80)
    
    final_report = tracker.save_final_results(output_path)
    
    if final_report:
        print("\nMILITARY VEHICLE TRACKING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\nSESSION SUMMARY:")
        print(f"  • Session ID: {final_report['session_info']['session_id']}")
        print(f"  • Frames Processed: {final_report['session_info']['frames_processed']}")
        print(f"  • Average FPS: {final_report['session_info']['average_fps']:.2f}")
        print(f"  • Total Runtime: {final_report['session_info']['total_runtime_seconds']:.2f}s")
        
        print(f"\nMILITARY DETECTION RESULTS:")
        print(f"  • Total Military Detections: {final_report['detection_summary']['total_military_detections']}")
        print(f"  • Unique Military Tracks: {final_report['tracking_summary']['total_military_tracks']}")
        print(f"  • Max Concurrent: {final_report['tracking_summary']['max_concurrent_tracks']}")
        
        print(f"\nACCURACY METRICS:")
        print(f"  • False Positives (FP): {final_report['accuracy_metrics']['false_positives']}")
        print(f"  • False Negatives (FN): {final_report['accuracy_metrics']['false_negatives']}")
        print(f"  • Identity Switches (IDS): {final_report['accuracy_metrics']['identity_switches']}")
        print(f"  • MOTA: {final_report['accuracy_metrics']['mota_percentage']:.2f}%")
        print(f"  • MOTP: {final_report['accuracy_metrics']['motp_percentage']:.2f}%")
        
        print(f"\nMILITARY THREAT ANALYSIS:")
        print(f"  • High Threat Military Vehicles: {final_report['military_threat_analysis']['high_threat_military_vehicles']}")
        print(f"  • Total Military Vehicles: {final_report['military_threat_analysis']['total_military_vehicles_detected']}")
        print(f"  • Tanks Detected: {final_report['military_threat_analysis']['tanks_detected']}")
        print(f"  • Military Trucks Detected: {final_report['military_threat_analysis']['military_trucks_detected']}")
        
        print(f"\nFILES SAVED LOCALLY:")
        print(f"  Main Folder: {tracker.local_folder}")
        print(f"  Tracked Video: {tracker.subfolders['videos']}")
        print(f"  Analytics Data: {tracker.subfolders['analytics']}")
        print(f"  Reports: {tracker.subfolders['reports']}")
        print(f"  Raw Data: {tracker.subfolders['raw_data']}")
        
        print(f"\nMILITARY VEHICLE CLASSES DETECTED:")
        for cls_name, count in final_report['detection_summary']['military_class_distribution'].items():
            percentage = (count / final_report['detection_summary']['total_military_detections']) * 100
            print(f"  • {cls_name}: {count} ({percentage:.1f}%)")
        
        return output_path, final_report
    else:
        print("Error saving final results")
        return output_path, None


def main():
    """Main function to run the military vehicle tracking system"""
    print("MILITARY VEHICLE TRACKING SYSTEM")
    print("Tracking only military vehicles with priority > 8")
    
    # Configuration - Update these paths
    model_path = r"best.pt"  # Update this path
    video_path = r"demo.mp4"  # Update this path
    
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
        print(f"   • Military Classes to Track: Military Truck, Tank, Armored Tank")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run the processing
    try:
        output_path, report = process_video_with_enhanced_tracking(
            model_path=model_path,
            video_path=video_path,
            max_frames=400,  # Adjust as needed
            confidence=0.467,  # Adjust confidence threshold
            save_interval=50   # Save data every 50 frames
        )
        
        if output_path and report:
            print("\nSUCCESS: Military vehicle tracking complete!")
            print("Check your local results folder for all analytics and processed video.")
        else:
            print("\nProcessing completed but there may have been issues saving results.")
    
    except Exception as e:
        print(f"\nError during main execution: {e}")


if __name__ == "__main__":
    main()
