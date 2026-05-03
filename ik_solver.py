import numpy as np
import mujoco


def ik_mujoco(model, data, target_pos, target_quat, site_name,
              max_iter=100, tol=1e-4, step_size=0.1, damping=1e-2):
    
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    
    # Store initial joint config — IK will stay close to this
    q_ref = data.qpos[:7].copy()

    for _ in range(max_iter):
        mujoco.mj_fwdPosition(model, data)

        current_pos = data.site_xpos[site_id].copy()
        current_mat = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_error = target_pos - current_pos

        target_mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(target_mat.ravel(), target_quat)
        rot_error_mat = target_mat @ current_mat.T
        rot_error = np.array([
            rot_error_mat[2, 1] - rot_error_mat[1, 2],
            rot_error_mat[0, 2] - rot_error_mat[2, 0],
            rot_error_mat[1, 0] - rot_error_mat[0, 1]
        ])

        error = np.concatenate([pos_error, rot_error])

        if np.linalg.norm(pos_error) < tol:
            break

        # Compute Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])
        J = J[:, :7]  # only FR3 joints, ignore fingers

        # Damped pseudoinverse
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))

        # Null-space projection — pulls joints toward reference config
        # This ensures a unique, stable solution
        null_space = np.eye(7) - J_pinv @ J
        q_null = 0.5 * (q_ref - data.qpos[:7])  # attraction to reference

        # Primary task + null-space secondary task
        dq = step_size * (J_pinv @ error + null_space @ q_null)

        # Clip joint delta to prevent large jumps
        dq = np.clip(dq, -0.05, 0.05)

        # Integrate
        data.qpos[:7] += dq
        
        # Clip to joint limits
        data.qpos[:7] = np.clip(
            data.qpos[:7],
            model.jnt_range[:7, 0],
            model.jnt_range[:7, 1]
        )

        data.ctrl[:7] = data.qpos[:7]

    # Update reference for next call — keeps motion smooth between waypoints
    q_ref = data.qpos[:7].copy()
    return data.qpos[:7].copy()