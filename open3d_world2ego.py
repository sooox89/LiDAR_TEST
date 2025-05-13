import gzip, pickle, json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# 1) Pose 로드
with open('/home/q/dataset/pandaset/014/lidar/poses.json','r') as f:
    poses = json.load(f)
frame_idx = 60
pose = poses[frame_idx]
t_world = np.array([pose['position']['x'],
                    pose['position']['y'],
                    pose['position']['z']])
q = pose['heading']
rot_world = R.from_quat([q['x'], q['y'], q['z'], q['w']])
R_inv = rot_world.inv().as_matrix()

# 2) World → Ego 변환
# 2.1 LiDAR
lidar_path = '/home/q/dataset/pandaset/014/lidar/60.pkl.gz'
with gzip.open(lidar_path,'rb') as f:
    df = pickle.load(f)
pts_w = np.vstack((df['x'], df['y'], df['z'])).T
pts_centered = pts_w - t_world
pts_ego = R_inv.dot(pts_centered.T).T
pcd_ego = o3d.geometry.PointCloud()
pcd_ego.points = o3d.utility.Vector3dVector(pts_ego)

# 2.2 Cuboids
cuboids_path = '/home/q/dataset/pandaset/014/annotations/cuboids/60.pkl.gz'
with gzip.open(cuboids_path,'rb') as f:
    cub_df = pickle.load(f)
obbs_ego = []
for _, row in cub_df.iterrows():
    cen_w = np.array([row['position.x'], row['position.y'], row['position.z']])
    cen_ego = R_inv.dot(cen_w - t_world)
    yaw = row['yaw']
    c,s = np.cos(yaw), np.sin(yaw)
    R_box = np.array([[ c, -s, 0],
                      [ s,  c, 0],
                      [ 0,  0, 1]])
    R_ego = R_inv.dot(R_box)
    extent = [row['dimensions.x'],
              row['dimensions.y'],
              row['dimensions.z']]
    obb = o3d.geometry.OrientedBoundingBox(cen_ego, R_ego, extent)
    obb.color = (1,0,0)
    obbs_ego.append(obb)

# 3) 축 생성
#   - axis_ego: 차량 기준 (origin=(0,0,0)), size=5
axis_ego = o3d.geometry.TriangleMesh.create_coordinate_frame(size=12.0)

#   - axis_world: 월드 원점(ego 기준으로 이동), size=5
axis_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=12.0)
axis_world.rotate(R_inv, center=(0,0,0))
axis_world.translate(-R_inv.dot(t_world))

# 4) 시각화
o3d.visualization.draw_geometries(
    [pcd_ego, *obbs_ego, axis_world, axis_ego],
    window_name=f"Ego vs World Axes (frame {frame_idx})",
    width=1024, height=768
)
