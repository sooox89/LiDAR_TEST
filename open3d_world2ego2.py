# test_pandaset_transform_rainbow.py

import gzip
import pickle
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ----------------------------------------
# 1) Pose 로드
# ----------------------------------------
with open('/home/q/dataset/pandaset/014/lidar/poses.json','r') as f:
    poses = json.load(f)

frame_idx = 60
pose = poses[frame_idx]
t_world = np.array([
    pose['position']['x'],
    pose['position']['y'],
    pose['position']['z']
])
w, x, y, z = (
    pose['heading']['w'],
    pose['heading']['x'],
    pose['heading']['y'],
    pose['heading']['z']
)
# quaternion → 회전행렬
rot_world = R.from_quat([x, y, z, w]).as_matrix()
R_inv = rot_world.T

# ----------------------------------------
# 2) Pandaset→Normative 스왑 매트릭스
# ----------------------------------------
R_norm = np.array([[ 0, 1, 0],
                   [-1, 0, 0],
                   [ 0, 0, 1]])

# ----------------------------------------
# 3) LiDAR 포인트 변환
# ----------------------------------------
lidar_path = '/home/q/dataset/pandaset/014/lidar/60.pkl.gz'
with gzip.open(lidar_path,'rb') as f:
    lidar_df = pickle.load(f)

# (1) world 좌표
pts_w = np.vstack((lidar_df['x'], lidar_df['y'], lidar_df['z'])).T

# (2) world→Ego
pts_center = pts_w - t_world
pts_ego    = (R_inv @ pts_center.T).T

# (3) Ego→Normative
pts_norm   = pts_ego[:, [1,0,2]]
pts_norm[:,1] *= -1

# ----------------------------------------
# 4) Cuboid 변환 (생략 가능, 시각화만 관심 있으면)
# ----------------------------------------
cuboids_path = '/home/q/dataset/pandaset/014/annotations/cuboids/60.pkl.gz'
with gzip.open(cuboids_path,'rb') as f:
    cub_df = pickle.load(f)

xs, ys, zs = (
    cub_df['position.x'].to_numpy(),
    cub_df['position.y'].to_numpy(),
    cub_df['position.z'].to_numpy()
)
dxs, dys, dzs = (
    cub_df['dimensions.x'].to_numpy(),
    cub_df['dimensions.y'].to_numpy(),
    cub_df['dimensions.z'].to_numpy()
)
yaws = cub_df['yaw'].to_numpy()

centers_w   = np.vstack((xs, ys, zs)).T
centers_ego = (R_inv @ (centers_w - t_world).T).T

# zrot 계산
p_world = np.array([[0,0,0],[0,1,0]])
p_ego   = (R_inv @ (p_world - t_world).T).T
yaxis   = p_ego[1] - p_ego[0]
zrot    = np.arctan2(-yaxis[0], yaxis[1])

ego_yaws = yaws + zrot

cen_norm = centers_ego[:, [1,0,2]]
cen_norm[:,1] *= -1

boxes = np.vstack([
    cen_norm[:,0], cen_norm[:,1], cen_norm[:,2],
    dys, dxs, dzs,
    ego_yaws
]).T.astype(np.float32)

# ----------------------------------------
# 5) Open3D 시각화 준비
# ----------------------------------------
# 5.1 Rainbow 색상 매핑: pts_norm 의 x→R, y→G, z→B
xyz = pts_norm  # shape (N,3)
mins = xyz.min(axis=0)
maxs = xyz.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1.0
colors = (xyz - mins) / ranges  # 정규화 후 0~1

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_norm)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 5.2 OrientedBoundingBox
obbs = []
for cx, cy, cz, dx, dy, dz, yaw in boxes:
    c, s = np.cos(yaw), np.sin(yaw)
    Rb = np.array([[ c,-s,0],[ s, c,0],[0,0,1]])
    obb = o3d.geometry.OrientedBoundingBox(
        center=(cx, cy, cz),
        R=Rb,
        extent=(dx, dy, dz)
    )
    obb.color = (1, 0, 0)
    obbs.append(obb)

# 5.3 좌표축: Ego(Normative)
axis_ego = o3d.geometry.TriangleMesh.create_coordinate_frame(size=12.0)
axis_ego.rotate(R_norm, center=(0,0,0))

# 5.4 좌표축: World → Normative Ego
axis_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=12.0)
axis_world.rotate(R_inv, center=(0,0,0))
axis_world.translate(-R_inv.dot(t_world))
axis_world.rotate(R_norm, center=(0,0,0))

# ----------------------------------------
# 6) 화면에 그리기
# ----------------------------------------
o3d.visualization.draw_geometries(
    [pcd, *obbs, axis_world, axis_ego],
    window_name=f"Rainbow Points & Axes (seq=014 frame={frame_idx})",
    width=720, height=768
)
