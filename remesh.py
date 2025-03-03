import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# 加載點雲文件
pcd = o3d.io.read_point_cloud('multiway_registration_all.pcd')

# 計算點雲的法線
pcd.estimate_normals()  # 估算法線
pcd.orient_normals_consistent_tangent_plane(100)  # 使法線方向一致

# 使用泊松表面重建生成三角網格
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8)  # 使用泊松重建，深度設為8

# 輸出生成的網格信息
print(mesh)

# 可視化生成的網格
o3d.visualization.draw_geometries([mesh])

# 計算密度並進行可視化
print('visualize densities')
densities = np.asarray(densities)  # 將密度轉為NumPy數組
# 使用Plasma色彩映射將密度值轉換為顏色
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]  # 取前三個通道的顏色值

# 構建密度網格並添加顏色
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices  # 設置網格的頂點
density_mesh.triangles = mesh.triangles  # 設置網格的三角形
density_mesh.triangle_normals = mesh.triangle_normals  # 設置三角形法線
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)  # 設置頂點顏色

# 可視化密度網格
o3d.visualization.draw_geometries([density_mesh])

# 過濾低密度的頂點
vertices_to_remove = densities < np.quantile(densities, 0.032)  # 選取密度低於第3.2百分位的頂點
mesh.remove_vertices_by_mask(vertices_to_remove)  # 移除這些頂點

# 選擇性應用平滑過濾器（已註解）
# mesh = mesh.filter_smooth_simple(number_of_iterations=5)  # 簡單平滑
# mesh = mesh.filter_smooth_laplacian(number_of_iterations=50)  # Laplacian平滑

# 計算新的頂點法線並上色
mesh.compute_vertex_normals()  # 計算法線
mesh.paint_uniform_color([1, 0.706, 0])  # 統一上色為橙色

# 將重建的三角網格保存為STL文件
o3d.io.write_triangle_mesh("mesh-90.stl", mesh)

# 可視化最終的三角網格
o3d.visualization.draw_geometries([mesh])
