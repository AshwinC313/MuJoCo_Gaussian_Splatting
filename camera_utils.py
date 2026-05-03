import numpy as np
import mujoco
import cv2
import json
import os
from PIL import Image as PILImage


def get_camera_image(renderer, data, camera_name):
    renderer.update_scene(data, camera=camera_name)
    return renderer.render()

def mujoco_to_opengl(T):
    flip = np.diag([1, -1, -1, 1])
    return T @ flip

def get_camera_pose(model, data, camera_name):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
    T = np.eye(4)
    T[:3, :3] = cam_rot
    T[:3, 3] = cam_pos
    return mujoco_to_opengl(T)

def save_for_gaussian_splat(frames, poses, model, camera_name,
                             output_dir="gaussian_splat_data",
                             width=320, height=240):
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    f = (height / 2) / np.tan(fovy / 2)

    intrinsics = {
        "fx": f, "fy": f,
        "cx": width / 2, "cy": height / 2,
        "width": width, "height": height
    }
    with open(f"{output_dir}/intrinsics.json", "w") as fp:
        json.dump(intrinsics, fp, indent=2)

    transforms = {
        "camera_angle_x": 2 * np.arctan(width / (2 * f)),
        "frames": []
    }

    for i, (frame, pose) in enumerate(zip(frames, poses)):
        img_path = f"{output_dir}/images/frame_{i:04d}.png"
        PILImage.fromarray(frame).save(img_path)
        transforms["frames"].append({
            "file_path": f"images/frame_{i:04d}.png",
            "transform_matrix": pose.tolist()
        })

    with open(f"{output_dir}/transforms.json", "w") as fp:
        json.dump(transforms, fp, indent=2)

    print(f"Saved {len(frames)} frames to {output_dir}/")