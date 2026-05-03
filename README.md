# MuJoCo Gaussian Splatting for Object Inspection

A simulation framework for generating **photorealistic 3D Gaussian Splats** of objects using a **Franka FR3 robot arm** in MuJoCo. The robot autonomously scans objects on a table using its wrist camera, capturing RGB images and camera poses to reconstruct high-fidelity 3D representations for inspection and downstream tasks.

---

##  Overview

This project combines:
- **MuJoCo** — physics simulation and robot control
- **Franka FR3** — 7-DOF robot arm with wrist-mounted RGB-D camera
- **Circular Arc Trajectory** — automated scanning with least-squares circle fitting
- **Gaussian Splatting (Nerfstudio)** — photorealistic 3D reconstruction from multi-view images

The pipeline enables automated object scanning and 3D reconstruction entirely in simulation, with direct compatibility with **Nerfstudio's splatfacto** trainer.

---

##  Project Structure

```
MuJoCo_Gaussian_Splatting/
├── main.py               # Main simulation and scanning loop
├── config.py             # All tunable parameters
├── trajectory.py         # Circular arc + least squares circle fitting
├── ik_solver.py          # MuJoCo Jacobian pseudoinverse IK with null-space projection
├── camera_utils.py       # RGB-D rendering, pose extraction, data saving
└── gaussian_splat_data/
    ├── images/           # Captured RGB frames (.png)
    ├── transforms.json   # Camera poses + intrinsics (Nerfstudio format)
    └── intrinsics.json   # Camera intrinsics
```

---

## ⚙️ Requirements

### System
- Ubuntu 20.04 / 22.04
- NVIDIA GPU with CUDA 12.1+
- Docker (for Nerfstudio training)

### Python Dependencies
```bash
pip install mujoco
pip install numpy opencv-python open3d Pillow
```

### MuJoCo Menagerie
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AshwinC313/MuJoCo_Gaussian_Splatting.git
cd MuJoCo_Gaussian_Splatting
```

### 2. Configure parameters
Edit `config.py` to match your setup:
```python
MODEL_PATH   = "/path/to/mujoco_menagerie/franka_fr3/fr3_table.xml"
SITE_NAME    = "attachment_site"
OBJECT_NAMES = ["apple", "banana", "blue_box"]
SCAN_HEIGHT  = 0.5
SCAN_STEPS   = 360
OUTPUT_DIR   = "gaussian_splat_data"
```

### 3. Run the scanning pipeline
```bash
python main.py
```

The robot will:
1. Read object positions from the simulation
2. Fit a circle around the objects using least squares
3. Generate a circular arc trajectory
4. Scan the scene, saving RGB frames and camera poses

---

## 🧠 How It Works

### Automatic Trajectory Generation
Object positions are read directly from MuJoCo and a **least-squares circle fit** determines the optimal scan center and radius:

```
Objects on table → fit circle → generate arc waypoints → IK → robot moves
```

### IK Solver
A **damped Jacobian pseudoinverse IK** with **null-space projection** ensures smooth, singularity-free motion and a unique solution at each waypoint.

### Data Collection
At each waypoint the pipeline captures:
- RGB image from the wrist camera
- 4×4 camera pose (converted to OpenGL convention for compatibility)

### 3D Reconstruction
Collected data is fed into **Nerfstudio's splatfacto** trainer to produce a Gaussian Splat — a photorealistic, view-dependent 3D representation of the scanned objects.

---

## 🏋️ Training Gaussian Splats

### Pull the Nerfstudio Docker image
```bash
docker run --gpus all \
    --name nerfstudio \
    -p 7007:7007 \
    -it ghcr.io/nerfstudio-project/nerfstudio:latest
```

### Copy scan data into the container
```bash
docker cp gaussian_splat_data nerfstudio:/workspace/data
```

### Train
```bash
ns-train splatfacto --data /workspace/data
```

### View result
```bash
ns-viewer --load-config outputs/gaussian_splat_data/splatfacto/<timestamp>/config.yml
```
Open `http://localhost:7007` in your browser.

### Export as .ply
```bash
ns-export gaussian-splat \
    --load-config outputs/gaussian_splat_data/splatfacto/<timestamp>/config.yml \
    --output-dir exports/
```

---

## 📷 Camera Setup

The FR3 is equipped with two cameras:

| Camera | Purpose |
|---|---|
| `wrist_cam` | Primary scan camera, mounted on end-effector |
| `table_cam` | Overhead depth camera for scene overview and point cloud |

---

## 💡 Tips for High Quality Splats

- Use **300-500 images** at **720p or higher** resolution
- Scan at **multiple heights** for full coverage
- Ensure **consistent lighting** — avoid dynamic shadows
- Verify camera poses form a clean arc before training
- Train for **50,000+ iterations** for best results

---

## 📦 Output Format

The scan pipeline outputs data in **Nerfstudio-compatible format**:

```json
{
  "fl_x": 320.0,
  "fl_y": 320.0,
  "cx": 160.0,
  "cy": 120.0,
  "w": 320,
  "h": 240,
  "frames": [
    {
      "file_path": "images/frame_0000.png",
      "transform_matrix": [[...], [...], [...], [...]]
    }
  ]
}
```

---

## 🔭 Future Work

- Integration with **OpenVLA / Octo** for language-conditioned pick and place
- **Real robot** deployment on physical FR3
- **Multi-object** reconstruction and segmentation
- **Sim-to-real** transfer of Gaussian Splat representations

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

##  Acknowledgements

- [MuJoCo](https://mujoco.org/) — physics simulation
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — robot models
- [Nerfstudio](https://nerf.studio/) — Gaussian Splatting training
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — original paper
