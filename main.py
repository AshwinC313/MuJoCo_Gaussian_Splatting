import mujoco
import mujoco.viewer
import numpy as np
import cv2

from config import *
from trajectory import get_object_positions, fit_circle_least_squares, circular_arc_trajectory
from ik_solver import ik_mujoco
from camera_utils import get_camera_image, get_camera_pose, save_for_gaussian_splat

# Load model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Renderers
wrist_renderer = mujoco.Renderer(model, height=RENDERER_HEIGHT, width=RENDERER_WIDTH)
table_renderer = mujoco.Renderer(model, height=RENDERER_HEIGHT, width=RENDERER_WIDTH)

# Get object positions and fit circle
mujoco.mj_forward(model, data)
points = get_object_positions(model, data, OBJECT_NAMES)
center, radius = fit_circle_least_squares(points)
trajectory, _ = circular_arc_trajectory(center, radius, SCAN_HEIGHT, SCAN_STEPS,
                                         scan_margin=SCAN_MARGIN)

# Scan storage
scan_frames = []
scan_poses = []
traj_idx = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        if traj_idx < len(trajectory):
            target_pos, target_quat = trajectory[traj_idx]
            ik_mujoco(model, data, target_pos, target_quat, SITE_NAME)

            # Capture frame and pose
            wrist_image = get_camera_image(wrist_renderer, data, CAMERA_WRIST)
            pose = get_camera_pose(model, data, CAMERA_WRIST)
            scan_frames.append(wrist_image.copy())
            scan_poses.append(pose.copy())

            traj_idx += 1
        else:
            print("Scan complete")
            save_for_gaussian_splat(scan_frames, scan_poses, model,
                                    CAMERA_WRIST, OUTPUT_DIR,
                                    RENDERER_WIDTH, RENDERER_HEIGHT)
            break

        mujoco.mj_step(model, data)

        # Display
        wrist_image = get_camera_image(wrist_renderer, data, CAMERA_WRIST)
        table_image = get_camera_image(table_renderer, data, CAMERA_TABLE)

        cv2.imshow("Wrist Camera", cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR))
        cv2.imshow("Table Camera", cv2.cvtColor(table_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        viewer.sync()

cv2.destroyAllWindows()