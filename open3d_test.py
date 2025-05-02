import numpy as np
import open3d as o3d
import math


points = np.load("/home/q/dataset/pandaset/demo_output/0000_points.npy")   # N,4
boxes = np.load("/home/q/dataset/pandaset/demo_output/0000_pred.npy")                     # M,7
scores = np.load("/home/q/dataset/pandaset/demo_output/0000_score.npy")                  # M,
labels = np.load("/home/q/dataset/pandaset/demo_output/0000_label.npy")                  # M,
print("Boxes Center Mean:", np.mean(boxes[:, :3], axis=0))

print("Points Center Mean:", np.mean(points, axis=0))

# 포인트 클라우드 시각화 준비
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.paint_uniform_color([0, 0.5, 1])

# # 3D 바운딩 박스 생성 함수
def create_open3d_box(center, size, yaw, color=[1, 0, 0]):
    """
    Create a 3D bounding box in Open3D
    center: [x, y, z]
    size: [dx, dy, dz]
    yaw: rotation around Z axis (radians)
    """
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = center
    bbox.extent = size
    R = bbox.get_rotation_matrix_from_xyz((0, 0, yaw))
    bbox.R = R
    bbox.color = color
    return bbox

box_list = []
for box in boxes:
    x, y, z, dx, dy, dz, yaw = box
    bbox = create_open3d_box(center=[x, y, z], size=[dx, dy, dz], yaw=yaw, color=[1, 0, 0])
    box_list.append(bbox)

o3d.visualization.draw_geometries([pcd, *box_list])