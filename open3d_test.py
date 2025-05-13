import numpy as np
import open3d as o3d

# 1) 데이터 로드
points     = np.load("/home/q/dataset/pandaset/demo_output/0000_points.npy")  # (N,4)
pred_boxes = np.load("/home/q/dataset/pandaset/demo_output/0000_pred.npy")    # (M_pred,7)
gt_boxes   = np.load("/home/q/dataset/pandaset/demo_output/0000_gt.npy")      # (M_gt,7)

# 2) PointCloud 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.paint_uniform_color([0, 0.5, 1])  # 파랑 계열

def create_open3d_box(center, size, yaw, color):
    """
    center: [x, y, z]
    size:   [dx, dy, dz]
    yaw:    rotation around Z axis (radians)
    color:  [r, g, b]
    """
    # yaw += np.pi/4
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = center
    obb.extent = size
    R_mat = obb.get_rotation_matrix_from_xyz((0, 0, yaw))
    obb.R = R_mat
    obb.color = color
    return obb

pred_o3d = []
for x, y, z, dx, dy, dz, yaw in pred_boxes:
    pred_o3d.append(create_open3d_box(
        center=[x, y, z],
        size=[dx, dy, dz],
        yaw=yaw,
        color=[1, 0, 0]
    ))

gt_o3d = []
for x, y, z, dx, dy, dz, yaw in gt_boxes:
    # yaw -= np.pi/4
    gt_o3d.append(create_open3d_box(
        center=[x, y, z],
        size=[dx, dy, dz],
        yaw = yaw,
        color=[0, 1, 0]
    ))

# 6) 좌표축 표시
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# 7) 시각화
o3d.visualization.draw_geometries(
    [pcd, *pred_o3d, *gt_o3d, axis],
    window_name="Predicted (Red) vs GT (Green) Boxes",
    width=1024, height=768
)
