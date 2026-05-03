import numpy as np
import mujoco
import mujoco.viewer

def get_object_positions(model, data, object_names):

    positions = []
    for name in object_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[body_id]
        positions.append(pos[:2])
        print(f"{name}: x={pos[0]:.3f}, y={pos[1]:.3f}")
    
    return np.array(positions)

def fit_circle_least_squares(points):
    """
    Fit a circle to a set of 2D points using least squares.
    Solves: (x - cx)^2 + (y - cy)^2 = r^2
    Linearized as: 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2) = x^2 + y^2
    points: numpy array of shape (N, 2)
    returns: center (2,), radius (scalar)
    """
    x = points[:, 0]
    y = points[:, 1]

    # Build linear system A * [cx, cy, c]^T = b
    # where c = r^2 - cx^2 - cy^2
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2

    # Solve with least squares
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = result[0], result[1]
    c = result[2]

    # Recover radius
    r = np.sqrt(c + cx**2 + cy**2)

    print(f"Fitted circle — center: ({cx:.3f}, {cy:.3f}), radius: {r:.3f}m")
    return np.array([cx, cy]), r

def circular_arc_trajectory(center, radius, height, n_steps,
                             angle_start=0, angle_end=np.pi,
                             scan_margin=0.15):
    """
    Generates circular arc waypoints around fitted center
    scan_margin: extra distance added to radius so camera doesn't collide
    """
    scan_radius = radius + scan_margin
    angles = np.linspace(angle_start, angle_end, n_steps)
    waypoints = []

    for angle in angles:
        x = center[0] + scan_radius * np.cos(angle)
        y = center[1] + scan_radius * np.sin(angle)
        z = height

        pos = np.array([x, y, z])
        quat = np.array([0.0, 1.0, 0.0, 0.0])  # downward facing
        waypoints.append((pos, quat))

    return waypoints, scan_radius