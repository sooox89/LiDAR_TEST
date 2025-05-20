import numpy as np
import open3d as o3d

# 1) 데이터 로드
prefix = "/home/q/dataset/pandaset/demo_output/001/0000"
points = np.load(f"{prefix}_points.npy")[:, :3]
pred_boxes = np.load(f"{prefix}_pred.npy")
gt_boxes   = np.load(f"{prefix}_gt.npy")
gt_labels  = np.load(f"{prefix}_gt_label.npy", allow_pickle=True)
scores     = np.load(f"{prefix}_score.npy")
labels     = np.load(f"{prefix}_label.npy")

# 클래스 이름 및 색상 정의
class_names = ['Car', 'Pedestrian', 'Motorcycle']
class_colors = {
    'Car': [1, 0, 0],          # 빨간색
    'Pedestrian': [0, 0, 0],   # 검정
    'Motorcycle': [0.6, 0, 1]  # 보라색
}
score_threshold_map = {
    'Car': 0.5,
    'Pedestrian': 0.1,
    'Motorcycle': 0.1
}

# PointCloud 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0, 0.5, 1])

# Bounding box 생성 함수
def create_open3d_box(center, size, yaw, color):
    # Z축 회전 행렬 직접 계산
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])
    # OrientedBoundingBox(center, R, extent)
    obb = o3d.geometry.OrientedBoundingBox(center, R, size)
    obb.color = color
    return obb

# 박스 내 포인트 존재 여부 검사 함수
def has_points_in_box(box, points):
    x, y, z, dx, dy, dz, yaw = box
    # 회전 행렬 (z축 회전)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[ c, -s, 0],
                  [ s,  c, 0],
                  [ 0,  0, 1]])
    # 박스 중심으로 이동 후 회전
    pts_local = (points - np.array([x, y, z])) @ R
    inside = (
        (np.abs(pts_local[:, 0]) <= dx / 2) &
        (np.abs(pts_local[:, 1]) <= dy / 2) &
        (np.abs(pts_local[:, 2]) <= dz / 2)
    )
    return inside.any()

# 예측 박스 시각화 (클래스별 score threshold & 포인트 유무 필터 적용)
pred_o3d = []
for box, score, label_idx in zip(pred_boxes, scores, labels):
    if label_idx < 1 or label_idx > len(class_names):
        continue
    cls = class_names[label_idx - 1]
    if score < score_threshold_map[cls]:
        continue
    # 박스 안에 포인트가 하나라도 없으면 스킵
    if not has_points_in_box(box, points):
        continue
    color = class_colors.get(cls, [1, 1, 1])
    pred_o3d.append(create_open3d_box(center=box[:3], size=box[3:6], yaw=box[6], color=color))

# GT 박스 시각화
gt_o3d = []
for box, label in zip(gt_boxes, gt_labels):
    if label not in class_names:
        continue
    gt_o3d.append(create_open3d_box(center=box[:3], size=box[3:6], yaw=box[6], color=[0, 1, 0]))

# 좌표축
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

# 시각화 실행
o3d.visualization.draw_geometries(
    [pcd, *pred_o3d, *gt_o3d, axis],
    window_name="Predicted (Colored) vs GT (Green)",
    width=1024, height=768
)
