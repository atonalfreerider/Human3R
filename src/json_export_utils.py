"""
JSON Export Utilities for Human3R

Exports camera poses and human 3D poses (in camera space) to structured JSON format.
"""

import json
import numpy as np
import torch
from typing import List, Dict, Any
import os


def numpy_to_serializable(obj):
    """Convert numpy/torch objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def export_poses_to_json(
    output_path: str,
    cam_dict: Dict[str, np.ndarray],
    all_smpl_verts: List[torch.Tensor],
    smpl_shape: List[torch.Tensor],
    smpl_rotvec: List[torch.Tensor],
    smpl_transl: List[torch.Tensor],
    smpl_expression: List[torch.Tensor],
    smpl_id: List[torch.Tensor],
    smpl_jaw_pose: List[torch.Tensor] = None,
    smpl_leye_pose: List[torch.Tensor] = None,
    smpl_reye_pose: List[torch.Tensor] = None,
    smpl_left_hand_pose: List[torch.Tensor] = None,
    smpl_right_hand_pose: List[torch.Tensor] = None,
    video_metadata: Dict[str, Any] = None,
    subsample: int = 1,
    body_joints_camera: List[torch.Tensor] = None,
    face_joints_camera: List[torch.Tensor] = None,
    hand_joints_camera: List[torch.Tensor] = None
):
    """
    Export camera poses and human 3D poses (in camera space) to JSON format.
    
    All 3D joint positions are in CAMERA SPACE - your external program can convert to world space.
    
    Args:
        output_path: Path to save JSON file
        cam_dict: Dictionary with camera parameters (focal, pp, R, t)
        all_smpl_verts: List of SMPL vertices per frame (not exported)
        smpl_shape: List of SMPL shape parameters per frame
        smpl_rotvec: List of SMPL rotation vectors per frame (root + body)
        smpl_transl: List of SMPL translation vectors per frame (in camera space)
        smpl_expression: List of SMPL expression parameters per frame
        smpl_id: List of person IDs per frame
        smpl_jaw_pose: List of jaw pose parameters per frame (1x3)
        smpl_leye_pose: List of left eye pose parameters per frame (1x3)
        smpl_reye_pose: List of right eye pose parameters per frame (1x3)
        smpl_left_hand_pose: List of left hand pose parameters per frame (15x3)
        smpl_right_hand_pose: List of right hand pose parameters per frame (15x3)
        video_metadata: Metadata about the video
        subsample: Frame subsampling factor
        body_joints_camera: List of body joint positions in camera coords per frame (Nx22x3)
        face_joints_camera: List of face joint positions in camera coords per frame (Nx3x3)
        hand_joints_camera: List of hand joint positions in camera coords per frame (Nx30x3)
    """
    
    if video_metadata is None:
        video_metadata = {}
    
    num_frames = len(smpl_shape)
    frame_offset = video_metadata.get("start_frame_offset", 0)
    
    # Build the output structure
    output_data = {
        "metadata": {
            "video_path": video_metadata.get("video_path", "unknown"),
            "total_frames": num_frames,
            "subsample_factor": subsample,
            "effective_fps": video_metadata.get("fps", 30) / subsample if subsample > 0 else video_metadata.get("fps", 30),
            "original_fps": video_metadata.get("fps", 30),
            "coordinate_system": "camera",
            "units": "meters",
            "format_version": "2.0",
            "start_frame_offset": frame_offset,
            "smplx_model": "full",
            "includes_face": smpl_jaw_pose is not None,
            "includes_hands": smpl_left_hand_pose is not None,
            "includes_camera_joints": body_joints_camera is not None and len(body_joints_camera) > 0,
            "joint_topology": {
                "body_joints": 22,
                "face_joints": 3,
                "hand_joints": 30,
                "body_joint_names": [
                    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
                    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
                    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_wrist", "right_wrist"
                ],
                "face_joint_names": ["jaw", "left_eye", "right_eye"],
                "hand_joint_names": ["left_hand (15 joints)", "right_hand (15 joints)"]
            },
            "note": "All 3D coordinates are in CAMERA SPACE. Use camera pose (R, t) to transform to world space."
        },
        "frames": {}
    }
    
    # Debug: verify camera joints are present
    if body_joints_camera is not None and len(body_joints_camera) > 0:
        print(f"✓ Exporting camera-space joints to JSON:")
        print(f"  - Body joints: {len(body_joints_camera)} frames")
        if body_joints_camera[0].numel() > 0:
            print(f"    First frame shape: {body_joints_camera[0].shape}")
    else:
        print(f"⚠️ No camera-space joints to export!")

    # Process each frame
    for frame_idx in range(num_frames):
        frame_data = {
            "frame_number": frame_idx,
            "global_frame_number": frame_offset + frame_idx,
            "camera": {
                "focal_length": numpy_to_serializable(cam_dict["focal"][frame_idx]),
                "principal_point": numpy_to_serializable(cam_dict["pp"][frame_idx]),
                "rotation_matrix": numpy_to_serializable(cam_dict["R"][frame_idx]),  # 3x3 rotation (camera to world)
                "translation": numpy_to_serializable(cam_dict["t"][frame_idx]),  # 3D position (camera in world)
                "note": "R and t define camera-to-world transformation. To transform camera-space point p to world: p_world = R @ p + t"
            },
            "humans": []
        }
        
        # Get human data for this frame
        n_humans = smpl_shape[frame_idx].shape[0]
        
        for human_idx in range(n_humans):
            person_id = int(smpl_id[frame_idx][human_idx])
            
            # Split rotvec into root_pose (1x3) and body_pose (remaining)
            rotvec_full = smpl_rotvec[frame_idx][human_idx]
            root_pose = rotvec_full[:3] if len(rotvec_full.shape) == 1 else rotvec_full[0]
            body_pose = rotvec_full[3:] if len(rotvec_full.shape) == 1 else rotvec_full[1:]
            
            human_data = {
                "person_id": person_id,
                "smplx_parameters": {
                    "shape": numpy_to_serializable(smpl_shape[frame_idx][human_idx]),  # Beta parameters (10D or 11D)
                    "root_pose": numpy_to_serializable(root_pose),  # Global orientation (3D)
                    "body_pose": numpy_to_serializable(body_pose),  # Body joints (21x3 or 63D)
                    "translation": numpy_to_serializable(smpl_transl[frame_idx][human_idx]),  # Global translation in camera space (3D)
                }
            }
            
            # Add CAMERA-SPACE joint positions
            if body_joints_camera is not None and frame_idx < len(body_joints_camera):
                if body_joints_camera[frame_idx].numel() > 0 and human_idx < body_joints_camera[frame_idx].shape[0]:
                    human_data["camera_joints"] = {
                        "body": numpy_to_serializable(body_joints_camera[frame_idx][human_idx]),  # 22x3 in camera space
                    }
            
            if face_joints_camera is not None and frame_idx < len(face_joints_camera):
                if face_joints_camera[frame_idx].numel() > 0 and human_idx < face_joints_camera[frame_idx].shape[0]:
                    if "camera_joints" not in human_data:
                        human_data["camera_joints"] = {}
                    human_data["camera_joints"]["face"] = numpy_to_serializable(face_joints_camera[frame_idx][human_idx])  # 3x3 in camera space
            
            if hand_joints_camera is not None and frame_idx < len(hand_joints_camera):
                if hand_joints_camera[frame_idx].numel() > 0 and human_idx < hand_joints_camera[frame_idx].shape[0]:
                    if "camera_joints" not in human_data:
                        human_data["camera_joints"] = {}
                    hand_joints = hand_joints_camera[frame_idx][human_idx]
                    human_data["camera_joints"]["left_hand"] = numpy_to_serializable(hand_joints[:15])  # 15x3 in camera space
                    human_data["camera_joints"]["right_hand"] = numpy_to_serializable(hand_joints[15:])  # 15x3 in camera space
            
            # Add expression if available
            if smpl_expression[frame_idx] is not None and smpl_expression[frame_idx][human_idx] is not None:
                human_data["smplx_parameters"]["expression"] = numpy_to_serializable(
                    smpl_expression[frame_idx][human_idx]
                )
            
            # Add face parameters if available
            if smpl_jaw_pose is not None and smpl_jaw_pose[frame_idx] is not None:
                if human_idx < smpl_jaw_pose[frame_idx].shape[0]:
                    human_data["smplx_parameters"]["jaw_pose"] = numpy_to_serializable(
                        smpl_jaw_pose[frame_idx][human_idx]
                    )
            
            if smpl_leye_pose is not None and smpl_leye_pose[frame_idx] is not None:
                if human_idx < smpl_leye_pose[frame_idx].shape[0]:
                    human_data["smplx_parameters"]["left_eye_pose"] = numpy_to_serializable(
                        smpl_leye_pose[frame_idx][human_idx]
                    )
            
            if smpl_reye_pose is not None and smpl_reye_pose[frame_idx] is not None:
                if human_idx < smpl_reye_pose[frame_idx].shape[0]:
                    human_data["smplx_parameters"]["right_eye_pose"] = numpy_to_serializable(
                        smpl_reye_pose[frame_idx][human_idx]
                    )
            
            # Add hand parameters if available
            if smpl_left_hand_pose is not None and smpl_left_hand_pose[frame_idx] is not None:
                if human_idx < smpl_left_hand_pose[frame_idx].shape[0]:
                    human_data["smplx_parameters"]["left_hand_pose"] = numpy_to_serializable(
                        smpl_left_hand_pose[frame_idx][human_idx]
                    )
            
            if smpl_right_hand_pose is not None and smpl_right_hand_pose[frame_idx] is not None:
                if human_idx < smpl_right_hand_pose[frame_idx].shape[0]:
                    human_data["smplx_parameters"]["right_hand_pose"] = numpy_to_serializable(
                        smpl_right_hand_pose[frame_idx][human_idx]
                    )
            
            frame_data["humans"].append(human_data)
        
        output_data["frames"][str(frame_idx)] = frame_data
    
    # Write to JSON file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Exported poses to: {output_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Total frames: {num_frames}")
    print(f"   Frame offset: {frame_offset}")
    print(f"   Average humans per frame: {sum(len(frame['humans']) for frame in output_data['frames'].values()) / num_frames:.1f}")
    print(f"   Coordinate system: CAMERA SPACE")
    
    # Print what's included
    includes = []
    if output_data["metadata"]["includes_face"]:
        includes.append("face (jaw, eyes)")
    if output_data["metadata"]["includes_hands"]:
        includes.append("hands")
    if output_data["metadata"]["includes_camera_joints"]:
        includes.append("camera-space joint positions")
    if includes:
        print(f"   Includes: {', '.join(includes)}")
    
    return output_path
