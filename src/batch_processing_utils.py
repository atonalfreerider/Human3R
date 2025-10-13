"""
Batch processing utilities for handling long videos
Splits videos into batches to prevent GPU memory issues
"""

import os
import cv2
import numpy as np
import tempfile
import shutil
import json
from typing import Dict, List, Any


class BatchProcessor:
    """Process videos in batches to manage GPU memory"""
    
    def __init__(self, batch_duration: float = 45.0, subsample: int = 1):
        self.batch_duration = batch_duration
        self.subsample = subsample
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def extract_batch_frames(self, video_path: str, start_time: float, 
                           end_time: float, output_dir: str) -> List[str]:
        """Extract frames for a specific time range"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        frame_paths = []
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx in range(start_frame, end_frame, self.subsample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        cap.release()
        return frame_paths
    
    def process_video_in_batches(self, video_path: str, output_dir: str, 
                                 model_args: Dict[str, Any]) -> Dict[str, Any]:
        """Process video in batches and merge results"""
        # Import here to avoid circular dependency
        from demo import run_batch_inference
        
        # Get video info
        video_info = self.get_video_info(video_path)
        print(f"📹 Video info: {video_info['duration']:.1f}s, "
              f"{video_info['fps']}fps, {video_info['total_frames']} frames")
        
        # Calculate batches
        num_batches = int(np.ceil(video_info['duration'] / self.batch_duration))
        print(f"🔄 Processing {num_batches} batches of {self.batch_duration}s each\n")
        
        all_results = []
        temp_dirs = []
        
        for batch_idx in range(num_batches):
            start_time = batch_idx * self.batch_duration
            end_time = min((batch_idx + 1) * self.batch_duration, video_info['duration'])
            
            print(f"🔄 Processing batch {batch_idx + 1}/{num_batches} "
                  f"(t={start_time:.1f}-{end_time:.1f}s)")
            
            # Create temp directory for this batch
            batch_temp = tempfile.mkdtemp(prefix=f"batch_{batch_idx}_")
            temp_dirs.append(batch_temp)
            
            # Extract frames
            print(f"  📷 Extracting frames...")
            frame_paths = self.extract_batch_frames(
                video_path, start_time, end_time, batch_temp
            )
            print(f"  ✓ Extracted {len(frame_paths)} frames")
            
            # Run inference on batch
            print(f"  🧠 Running inference...")
            try:
                batch_result = run_batch_inference(
                    frame_paths=frame_paths,
                    output_dir=os.path.join(output_dir, f"batch_{batch_idx}"),
                    **model_args
                )
                all_results.append(batch_result)
                print(f"  ✓ Batch {batch_idx + 1} complete\n")
            except Exception as e:
                print(f"  ❌ Batch {batch_idx} failed: {e}")
                import torch
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"  GPU Memory: {mem_allocated:.2f}GB allocated, "
                          f"{mem_reserved:.2f}GB reserved")
                raise
        
        # Merge results
        print("🔄 Merging batch results...")
        merged_output = self.merge_batch_results(all_results, output_dir)
        
        # Cleanup
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return merged_output
    
    def merge_batch_results(self, batch_results: List[Dict], 
                           output_dir: str) -> Dict[str, Any]:
        """Merge results from multiple batches"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Merge JSON outputs if they exist
        json_output = None
        if all('json_output' in r for r in batch_results):
            merged_json = self.merge_json_outputs(
                [r['json_output'] for r in batch_results]
            )
            json_output = os.path.join(output_dir, "poses.json")
            with open(json_output, 'w') as f:
                json.dump(merged_json, f, indent=2)
        
        return {
            'num_batches': len(batch_results),
            'json_output': json_output
        }
    
    def merge_json_outputs(self, json_paths: List[str]) -> Dict:
        """Merge multiple JSON pose files"""
        merged = {
            "metadata": {},
            "frames": {}
        }
        
        frame_offset = 0
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Merge metadata (use first batch)
            if not merged["metadata"]:
                merged["metadata"] = data["metadata"]
            
            # Merge frames with offset
            for frame_id, frame_data in data["frames"].items():
                new_frame_id = str(int(frame_id) + frame_offset)
                merged["frames"][new_frame_id] = frame_data
            
            frame_offset += len(data["frames"])
        
        return merged
