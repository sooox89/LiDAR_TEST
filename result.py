import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 쓰기 위해 필요

# 1) pickle 파일 로드
pkl_path = '/home/q/dataset/result.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 2) 시각화할 프레임 선택
frame_idx = 0
frame = data[frame_idx]

# 3) 포인트 클라우드와 박스 정보 추출
points = frame['points']  # shape: (N, 3)
boxes  = frame['boxes']   # shape: (M, 7): [x, y, z, length, width, height, yaw]

# 4) 3D 플롯 준비
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 포인트 클라우드 산점도
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)

# 5) 3D 바운딩 박스 그리기 함수
def draw_3d_box(ax, box, color='r', linewidth=1.5):
    x, y, z, l, w, h, yaw = box
    # 박스 중심 기준 8개 코너
    corners = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
    ])
    # Z축(Up) 중심 회전(수평 yaw)
    R = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0 ],
        [ np.sin(yaw),  np.cos(yaw), 0 ],
        [          0,            0, 1 ]
    ])
    # 회전 및 박스 중심으로 이동
    corners = (corners @ R.T) + np.array([x, y, z])
    # 엣지(간선) 정의 (8개 코너를 연결)
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # 상단 사각형
        (4,5),(5,6),(6,7),(7,4),  # 하단 사각형
        (0,4),(1,5),(2,6),(3,7)   # 수직 엣지
    ]
    for i0, i1 in edges:
        p0, p1 = corners[i0], corners[i1]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=color, linewidth=linewidth
        )

# 6) 모든 박스 렌더링
for box in boxes:
    draw_3d_box(ax, box, color='r')

# 축 레이블 및 시점 설정
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'3D Bounding Boxes - Frame {frame_idx}')
ax.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.show()
