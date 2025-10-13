#!/usr/bin/env python3
"""
Modified from CUT3R: https://github.com/CUT3R/CUT3R

Online Human-Scene Reconstruction Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D scene point clouds and SMPLX sequences with the SceneHumanViewer. 
Use the command-line arguments to adjust parameters 
such as the model checkpoint path, image sequence directory, image size, device, etc.

Example:
    python demo.py --model_path src/human3r.pth --size 512 \
        --seq_path examples/GoodMornin1.mp4 --subsample 1 --vis_threshold 2 \
        --downsample_factor 1 --use_ttt3r --reset_interval 100
"""

import os
import numpy as np
import torch

# Apply PyTorch 2.8+ compatibility patch before any model imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Apply compatibility patches
from pytorch28_compat import _patched_torch_load
try:
    from model_load_patch import patch_dust3r_model_loading
except ImportError:
    print("⚠️ Model loading patch not found, continuing anyway...")

import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import roma

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--msk_threshold",
        type=float,
        default=0.1,
        help="Mask threshold. Ranging from 0 to 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--save_smpl",
        action="store_true",
        help="Save smpl results (deprecated - only JSON output is generated).",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save smpl video (deprecated - video generation disabled to save memory).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=10,
        help="Point cloud downsample factor for the viewer",
    )
    parser.add_argument(
        "--smpl_downsample",
        type=int,
        default=1,
        help="SMPL sequence downsample factor for the viewer",
    )
    parser.add_argument(
        "--camera_downsample",
        type=int,
        default=1,
        help="Camera motion downsample factor for the viewer",
    )
    parser.add_argument(
        "--mask_morph",
        type=int,
        default=10,
        help="Mask morphology for the viewer",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help="Path to save JSON file with camera and human poses (default: output_dir/poses.json)",
    )
    parser.add_argument(
        "--batch_processing",
        action="store_true",
        help="Process video in 45-second batches to manage GPU memory",
    )
    parser.add_argument(
        "--batch_duration",
        type=float,
        default=45.0,
        help="Duration of each batch in seconds (default: 45.0)",
    )
    return parser.parse_args()


def prepare_input(
    img_paths, 
    img_mask, 
    size, 
    raymaps=None, 
    raymap_mask=None, 
    revisit=1, 
    update=True, 
    img_res=None, 
    reset_interval=100
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu") # if use pseudo K

    views = []
    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views

def prepare_output(
        outputs, outdir, revisit=1, use_pose=True, 
        save_smpl=False, save_video=False, img_res=None, subsample=1):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        save_smpl (bool): Deprecated - no longer used.
        save_video (bool): Deprecated - no longer used.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary, SMPL data for JSON export)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer
    from src.dust3r.utils.image import unpad_image

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    conf_other = [output["conf"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth - PER FRAME for zoom handling
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    
    # Per-frame focal length estimation to handle zoom
    focal_list = []
    for i, pts3d in enumerate(pts3ds_self):
        focal_i = estimate_focal_knowing_depth(pts3d.unsqueeze(0), pp[i:i+1], focal_mode="weiszfeld")
        focal_list.append(focal_i)
    focal = torch.cat(focal_list, dim=0)  # Shape: [B, 1]
    focal = focal.squeeze(-1)  # Now shape: [B]

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w.numpy(),
        "t": t_c2w.numpy(),
    }

    cam2world_tosave = torch.cat(pr_poses)
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # get SMPL parameters from outputs
    smpl_shape = [output.get(
        "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(
        output.get(
            "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get(
        "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get(
        "smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get(
        "smpl_id", torch.empty(1,0))[0] for output in outputs["pred"]]
    
    # Extract additional SMPLX parameters (face and hands)
    smpl_jaw_pose = [output.get("smpl_jaw_pose", [None])[0] for output in outputs["pred"]]
    smpl_leye_pose = [output.get("smpl_leye_pose", [None])[0] for output in outputs["pred"]]
    smpl_reye_pose = [output.get("smpl_reye_pose", [None])[0] for output in outputs["pred"]]
    smpl_left_hand_pose = [output.get("smpl_left_hand_pose", [None])[0] for output in outputs["pred"]]
    smpl_right_hand_pose = [output.get("smpl_right_hand_pose", [None])[0] for output in outputs["pred"]]

    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [output["msk"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    # SMPL layer for computing vertices and joints in CAMERA SPACE
    smpl_layer = SMPL_Layer(type='smplx', 
                            gender='neutral', 
                            num_betas=smpl_shape[0].shape[-1], 
                            kid=False, 
                            person_center='head')
    smpl_faces = smpl_layer.bm_x.faces

    # Extract CAMERA-SPACE joint positions (no world transformation)
    all_body_joints_camera = []
    all_face_joints_camera = []
    all_hand_joints_camera = []
    
    for f_id in range(B):
        n_humans_i = smpl_shape[f_id].shape[0]
        
        if n_humans_i > 0:
            # Use per-frame focal length for this frame's intrinsics
            frame_intrinsics = torch.eye(3).unsqueeze(0).repeat(n_humans_i, 1, 1)
            frame_intrinsics[:, 0, 0] = focal[f_id]
            frame_intrinsics[:, 1, 1] = focal[f_id]
            frame_intrinsics[:, 0, 2] = pp[f_id, 0]
            frame_intrinsics[:, 1, 2] = pp[f_id, 1]
            
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec[f_id], 
                    smpl_shape[f_id], 
                    smpl_transl[f_id], 
                    None, None, 
                    K=frame_intrinsics, 
                    expression=smpl_expression[f_id])
            
            # Extract joints in CAMERA SPACE (no world transformation)
            all_joints = smpl_out.get('smpl_j3d')
            
            if all_joints is not None and all_joints.shape[1] >= 55:
                # Body joints: first 22 joints (0-21) - CAMERA SPACE
                body_joints_camera = all_joints[:, :22, :]
                all_body_joints_camera.append(body_joints_camera)
                
                # Face joints: joints 22-24 (jaw, left_eye, right_eye) - CAMERA SPACE
                face_joints_camera = all_joints[:, 22:25, :]
                all_face_joints_camera.append(face_joints_camera)
                
                # Hand joints: joints 25-54 (15 left + 15 right) - CAMERA SPACE
                left_hand_joints = all_joints[:, 25:40, :]
                right_hand_joints = all_joints[:, 40:55, :]
                hand_joints_camera = torch.cat([left_hand_joints, right_hand_joints], dim=1)
                all_hand_joints_camera.append(hand_joints_camera)
            else:
                all_body_joints_camera.append(torch.zeros(n_humans_i, 22, 3))
                all_face_joints_camera.append(torch.zeros(n_humans_i, 3, 3))
                all_hand_joints_camera.append(torch.zeros(n_humans_i, 30, 3))
        else:
            all_body_joints_camera.append(torch.empty(0, 22, 3))
            all_face_joints_camera.append(torch.empty(0, 3, 3))
            all_hand_joints_camera.append(torch.empty(0, 30, 3))
    
    # Return SMPL data with CAMERA-SPACE joints
    smpl_data_for_json = {
        "smpl_shape": smpl_shape,
        "smpl_rotvec": smpl_rotvec,
        "smpl_transl": smpl_transl,
        "smpl_expression": smpl_expression,
        "smpl_id": smpl_id,
        "smpl_jaw_pose": smpl_jaw_pose,
        "smpl_leye_pose": smpl_leye_pose,
        "smpl_reye_pose": smpl_reye_pose,
        "smpl_left_hand_pose": smpl_left_hand_pose,
        "smpl_right_hand_pose": smpl_right_hand_pose,
        "body_joints_camera": all_body_joints_camera,
        "face_joints_camera": all_face_joints_camera,
        "hand_joints_camera": all_hand_joints_camera
    }
    
    return (
        pts3ds_other,
        colors, 
        conf_other, 
        cam_dict, 
        [],  # all_verts removed 
        smpl_faces,
        smpl_id,
        msks,
        smpl_data_for_json
    )

def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Check if batch processing is requested
    if args.batch_processing:
        print("🔄 Starting batch processing mode...")
        
        # Import batch processing utilities
        from src.batch_processing_utils import BatchProcessor
        
        # Create batch processor
        processor = BatchProcessor(
            batch_duration=args.batch_duration,
            subsample=args.subsample
        )
        
        # Prepare model arguments
        model_args = {
            "model_path": args.model_path,
            "device": args.device,
            "size": args.size,
            "use_ttt3r": args.use_ttt3r,
            "reset_interval": args.reset_interval,
            "subsample": args.subsample
        }
        
        # Process video in batches
        results = processor.process_video_in_batches(
            args.seq_path, args.output_dir, model_args
        )
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Processed {results['num_batches']} batches")
        if results['json_output']:
            print(f"   JSON output: {results['json_output']}")
        
        return
    
    # Original single-pass processing
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available. Switching to CPU.")
        device = "cpu"
    elif device == "cuda":
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Verify GPU is actually being used by creating a test tensor
        test_tensor = torch.zeros(1).to(device)
        assert test_tensor.is_cuda, "Failed to allocate tensor on GPU!"
        print(f"  ✓ GPU verification successful")
        del test_tensor
        torch.cuda.empty_cache()

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.json_export_utils import export_poses_to_json

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
    
    # Get video metadata
    video_metadata = {"video_path": args.seq_path}
    if not os.path.isdir(args.seq_path):
        cap = cv2.VideoCapture(args.seq_path)
        if cap.isOpened():
            video_metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
            video_metadata["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
    
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Prepare input views.
    print("Preparing input views...")
    img_res = getattr(model, 'mhmr_img_res', None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval
    )

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=args.use_ttt3r)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for JSON export only.
    print("Preparing JSON export...")
    (
        cam_dict, 
        smpl_data_for_json
    ) = prepare_output(
        outputs, args.output_dir, 1, True, 
        False, False, img_res, args.subsample
    )

    # Export to JSON
    json_output_path = args.json_output or os.path.join(args.output_dir, "poses.json")
    print(f"\nExporting poses to JSON: {json_output_path}")
    export_poses_to_json(
        output_path=json_output_path,
        cam_dict=cam_dict,
        all_smpl_verts=[],  # No vertices
        smpl_shape=smpl_data_for_json["smpl_shape"],
        smpl_rotvec=smpl_data_for_json["smpl_rotvec"],
        smpl_transl=smpl_data_for_json["smpl_transl"],
        smpl_expression=smpl_data_for_json["smpl_expression"],
        smpl_id=smpl_data_for_json["smpl_id"],
        smpl_jaw_pose=smpl_data_for_json.get("smpl_jaw_pose"),
        smpl_leye_pose=smpl_data_for_json.get("smpl_leye_pose"),
        smpl_reye_pose=smpl_data_for_json.get("smpl_reye_pose"),
        smpl_left_hand_pose=smpl_data_for_json.get("smpl_left_hand_pose"),
        smpl_right_hand_pose=smpl_data_for_json.get("smpl_right_hand_pose"),
        video_metadata=video_metadata,
        subsample=args.subsample,
        body_joints_camera=smpl_data_for_json.get("body_joints_camera"),
        face_joints_camera=smpl_data_for_json.get("face_joints_camera"),
        hand_joints_camera=smpl_data_for_json.get("hand_joints_camera")
    )

    # Print summary
    print("\n" + "="*60)
    print("✅ Processing complete!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"JSON output: {json_output_path}")
    print("="*60)


def run_batch_inference(frame_paths, output_dir, model_path, device, size, 
                       use_ttt3r, reset_interval, subsample):
    """
    Run inference on a batch of frames (used by batch processor)
    
    Args:
        frame_paths: List of image paths
        output_dir: Output directory for this batch
        model_path: Path to model checkpoint
        device: 'cuda' or 'cpu'
        size: Input image size
        use_ttt3r: Whether to use TTT3R
        reset_interval: Reset tracking interval
        subsample: Frame subsample factor
    
    Returns:
        Dictionary with output paths
    """
    import torch
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.json_export_utils import export_poses_to_json
    
    # Load model
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.eval()
    
    # Prepare input
    img_mask = [True] * len(frame_paths)
    img_res = getattr(model, 'mhmr_img_res', None)
    views = prepare_input(
        img_paths=frame_paths,
        img_mask=img_mask,
        size=size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=reset_interval
    )
    
    # Run inference
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=use_ttt3r
    )
    
    # Process outputs
    (
        pts3ds_other, 
        colors, 
        conf, 
        cam_dict, 
        all_smpl_verts, 
        smpl_faces,
        smpl_id,
        msks,
        smpl_data_for_json
    ) = prepare_output(
        outputs, output_dir, 1, True, 
        False, False, img_res, subsample
    )
    
    
    # Export JSON - NOW INCLUDING CAMERA JOINTS!
    json_output = os.path.join(output_dir, "poses.json")
    export_poses_to_json(
        output_path=json_output,
        cam_dict=cam_dict,
        all_smpl_verts=[],  # No vertices
        smpl_shape=smpl_data_for_json["smpl_shape"],
        smpl_rotvec=smpl_data_for_json["smpl_rotvec"],
        smpl_transl=smpl_data_for_json["smpl_transl"],
        smpl_expression=smpl_data_for_json["smpl_expression"],
        smpl_id=smpl_data_for_json["smpl_id"],
        smpl_jaw_pose=smpl_data_for_json["smpl_jaw_pose"],
        smpl_leye_pose=smpl_data_for_json["smpl_leye_pose"],
        smpl_reye_pose=smpl_data_for_json["smpl_reye_pose"],
        smpl_left_hand_pose=smpl_data_for_json["smpl_left_hand_pose"],
        smpl_right_hand_pose=smpl_data_for_json["smpl_right_hand_pose"],
        video_metadata={},
        subsample=subsample,
        body_joints_camera=smpl_data_for_json.get("body_joints_camera"),
        face_joints_camera=smpl_data_for_json.get("face_joints_camera"),
        hand_joints_camera=smpl_data_for_json.get("hand_joints_camera")
    )
    
    # Clear GPU memory
    del model, outputs, pts3ds_other, colors, conf
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        'json_output': json_output
    }


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteratively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
