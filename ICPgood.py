

import open3d as o3d
import numpy as np

def load_point_clouds(voxel_size=0.0, pointlist=None):
    pcds = []
    for point in pointlist:
        pcd_down = point.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build PoseGraph")
            if target_id == source_id + 1:
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

def rotate_around_y_axis(pcd, rotation_angle):
    center = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-center)
    axis = np.array([0, 1, 0])
    rotation_vector = axis * rotation_angle
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    return pcd

mesh1 = o3d.io.read_triangle_mesh("./0001/rebbox/redata0001bbox-90.stl")
mesh2 = o3d.io.read_triangle_mesh("./0001/rebbox/redata0001bbox.stl")
mesh3 = o3d.io.read_triangle_mesh("./0001/rebbox/redata0001bbox--90.stl")

mesh1.paint_uniform_color([1, 0.706, 0])
mesh2.paint_uniform_color([0, 0.706, 1])
mesh3.paint_uniform_color([0, 1, 0])

source = o3d.geometry.PointCloud()
target = o3d.geometry.PointCloud()
third = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(np.array(mesh1.vertices))
target.points = o3d.utility.Vector3dVector(np.array(mesh2.vertices))
third.points = o3d.utility.Vector3dVector(np.array(mesh3.vertices))

rotation_angle = -np.pi / 2
source = rotate_around_y_axis(source, rotation_angle)
third = rotate_around_y_axis(third, -rotation_angle)

target.translate(-np.mean(np.asarray(target.points), axis=0))
source.translate(np.array([-5, 0, -2]))
third.translate(np.array([5, 0, -2]))

# o3d.visualization.draw_geometries([source, target, third], 
#                                   window_name="Point Clouds",
#                                   width=800, height=600)

source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=40))
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=40))
third.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=40))

voxel_size = 0.0002
pcds_down = load_point_clouds(voxel_size, [source, target])

max_correspondence_distance_coarse = 30
max_correspondence_distance_fine = 0.05

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down)

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.1,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

pcds = load_point_clouds(voxel_size, [source, target])
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds)):
    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds[point_id]

pcd_combined.translate(-np.mean(np.asarray(pcd_combined.points), axis=0))
pcd_combined.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=40))
pcds_down1 = load_point_clouds(voxel_size, [third, pcd_combined])

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph1 = full_registration(pcds_down1)

print("Optimizing PoseGraph1 ...")
option1 = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.1,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph1,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option1)
    
pcds1 = load_point_clouds(voxel_size, [third, pcd_combined])
pcd_combined1 = o3d.geometry.PointCloud()
for point_id in range(len(pcds1)):
    pcds1[point_id].transform(pose_graph1.nodes[point_id].pose)
    pcd_combined1 += pcds1[point_id]
o3d.visualization.draw_geometries([pcd_combined1])

o3d.io.write_point_cloud("multiway_registration_all.pcd", pcd_combined1)
