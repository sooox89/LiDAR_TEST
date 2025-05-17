import numpy as np
import open3d as o3d
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R

# ========== 설정 ==========
seq_name = "001"
root_dir = Path("/home/q/dataset/demo_output")
points_dir = root_dir / "points" / seq_name
pred_dir   = root_dir / "pred" / seq_name
gt_dir     = root_dir / "gt" / seq_name
pose_path = Path(f"/home/q/dataset/pandaset/{seq_name}/lidar/poses.json")
with open(pose_path, 'r') as f:
    poses = json.load(f)

frame_ids = sorted([f.stem for f in points_dir.glob("*.npy")])
frame_idx = 0

def create_open3d_box(center, size, yaw, color):
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = center
    obb.extent = size
    obb.R = obb.get_rotation_matrix_from_xyz((0, 0, yaw))
    obb.color = color
    return obb

def create_axes(pose):
    t = np.array([pose['position'][k] for k in ['x','y','z']])
    q = pose['heading']
    Rw = R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    R_inv = Rw.T
    R_norm = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    axis_ego = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0)
    axis_ego.rotate(R_norm, center=(0,0,0))

    axis_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0)
    axis_world.rotate(R_inv, center=(0,0,0))
    axis_world.translate(-R_inv @ t)
    axis_world.rotate(R_norm, center=(0,0,0))

    return axis_ego, axis_world

def get_frame_geometry(fid):
    pts = np.load(points_dir / f"{fid}.npy")
    preds = np.load(pred_dir / f"{fid}.npy")
    gts = np.load(gt_dir / f"{fid}.npy")
    pose = poses[int(fid)]

    geometries = []
    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.paint_uniform_color([0, 0.5, 1])
    geometries.append(pcd)

    for box in preds:
        geometries.append(create_open3d_box(box[:3], box[3:6], box[6], [1, 0, 0]))
    for box in gts:
        geometries.append(create_open3d_box(box[:3], box[3:6], box[6], [0, 1, 0]))

    axis_ego, axis_world = create_axes(pose)
    geometries += [axis_ego, axis_world]
    return geometries

# ========== 시각화 루프 ==========
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name=f"Sequence {seq_name}", width=1024, height=768)

def update_view(fid):
    vis.clear_geometries()
    geos = get_frame_geometry(fid)
    for g in geos:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()

def next_frame_callback(vis):
    global frame_idx
    frame_idx += 1
    if frame_idx >= len(frame_ids):
        print("✅ 마지막 프레임입니다.")
        return False
    print(f"▶ 프레임: {frame_ids[frame_idx]}")
    update_view(frame_ids[frame_idx])
    return False

# key: → (오른쪽 화살표) or N 키
vis.register_key_callback(ord("N"), next_frame_callback)
vis.register_key_callback(262, next_frame_callback)  # 262 = GLFW_KEY_RIGHT

# 초기 프레임 표시
print(f"▶ 시작 프레임: {frame_ids[frame_idx]}")
update_view(frame_ids[frame_idx])
vis.run()
vis.destroy_window()
