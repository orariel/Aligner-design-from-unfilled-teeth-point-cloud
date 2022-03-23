# This program find the orientation of the middle teeth

import numpy as np
import open3d as o3d
import pyvista as pv

#<--------------------------------------------------Load Data----------------------------------------------------->
tooth = pv.read("tooth_12_decimate.stl")

boundary = tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
p = pv.Plotter()
p.add_mesh(tooth)
p.add_mesh(boundary, color="red")
p.show()
boundary_arr = np.asarray(boundary.points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(boundary_arr)

# o3d.visualization.draw_geometries([pcd])
#<--------------------------------------------------Align tooth to same cordinate sys----------------------------------------------------->
cor_sys=o3d.geometry.TriangleMesh.create_coordinate_frame(size=5,origin=tooth.center)
tooth_o3d=o3d.io.read_triangle_mesh('tooth_7_decimate.stl')
tooth_o3d_9=o3d.io.read_triangle_mesh('tooth_9_decimate.stl')
tooth_o3d.compute_vertex_normals()
tooth_o3d_9.compute_vertex_normals()


o3d.visualization.draw_geometries([cor_sys,tooth_o3d,tooth_o3d_9],width=900)

#<-------------------------------------------------Clean PCD boundary----------------------------------------------------->
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    return inlier_cloud
cl, ind = pcd.remove_radius_outlier(nb_points=30, radius=2.5)
clean_boundary=display_inlier_outlier(pcd,ind)
# o3d.visualization.draw_geometries([clean_boundary])
boundary_arr=np.asarray(pcd.points)
points = pv.wrap(boundary_arr)
cloud=pv.PolyData(boundary_arr)
boundary_arr_shorted_z= boundary_arr[boundary_arr[:,2].argsort()]
local_maxima=boundary_arr_shorted_z[-1,:]
tooth_center=np.asarray(tooth.center)
p=pv.Plotter()
z_max_tooth = tooth.bounds[4]
# mid_plane.translate(tooth.center, inplace=True)
# mid_plane.translate([0,z_max_tooth, 0], inplace=True)
# side_L=tooth.clip(normal='y',origin=local_maxima,invert=False)
# side_L.plot()
# side_R=tooth.clip(normal='y',origin=local_maxima)
# side_R.plot()
#<---------------------------------Slice z crown------------------------------------------------------->
side_up=tooth.clip(normal='z',origin=local_maxima-1.5,invert=False)
side_up.plot()
normals_up=np.asarray(side_up.point_normals)
mean_normal_up=np.mean(normals_up,axis=0)
root_plane = pv.Circle(1, resolution=300)
root_plane.translate(tooth.center,inplace=True)
root_plane.translate([0,0,z_max_tooth-2],inplace=True)
projected =root_plane.project_points_to_plane(origin=root_plane.center,normal=-mean_normal_up)
Plo=pv.Plotter()
Plo.add_mesh(projected,color='Green')
Plo.add_mesh(tooth)
Plo.add_mesh(root_plane,color='blue')
Plo.show()
#<---------------------------------Mean normal calculation for side A------------------------------------------------------->
# normals_a=np.asarray(side_a.point_normals)
# mean_normal_a=np.mean(normals_a,axis=0)
# point_a=pv.PolyData(mean_normal_a)
# point_a.plot()
# #<---------------------------------Mean normal calculation for side B-------------------------------------------------------
# points_to_remove=[]
# points_b=np.asarray(side_b.points)
# points_b=points_b[points_b[:,2].argsort()]
# for i in range(points_b.shape[0]):
#     point=points_b[i,2]
#     if (point>local_maxima[2]-1):
#         points_to_remove.append([i])
# arr_clean = np.delete(points_b, points_to_remove, axis=0)
# side_b_clean=side_b.clip(normal='y',origin=arr_clean[-1,:],invert=False)
# side_b_clean.plot()
# poplibpp=pv.PolyData(side_b_clean)
# poplibpp.plot()
# normals_b=np.asarray(side_b_clean.point_normals)
# mean_normal_b=np.mean(normals_b,axis=0)
# mid_vector=mean_normal_a+mean_normal_b
# root_plane = pv.Circle(1, resolution=300)
#
# p_2=pv.Plotter()
# z_max_tooth = tooth.bounds[4]
# root_plane.translate(tooth.center,inplace=True)
# root_plane.translate([0,0,z_max_tooth-2],inplace=True)
#
#
#
# mid_plane=pv.Plane(center=tooth_center,direction=mid_vector,i_size=2, j_size=2)
# mid_plane.translate([0,0,z_max_tooth-2],inplace=True)
# # p_2.add_mesh(mid_plane)
# p_2.add_mesh(root_plane,color='red')
# p_2.add_mesh(tooth,color='blue')
# #
# # sphere = pv.Sphere(radius=1,center=(0,0,0),theta_resolution=100,phi_resolution=100)
# # projected = sphere.project_points_to_plane(normal=mid_vector)
# # projected.plot(show_edges=True, line_width=3)
# # pp=projected.points
# # pp=pv.PolyData(pp)
# # pp.translate(mid_plane.center,inplace=True)
# # origin =tooth.center
# # origin[-1] -=root_plane.length / 1
# projected =root_plane.project_points_to_plane(origin=root_plane.center,normal=-mid_vector)
#
# projected.translate([0,0,0],inplace=True)
# p_2.add_mesh(projected,color='Green')
#
# p_2.show()
# p_=pv.Plotter()
# p.add_mesh(tooth)
# p.add_mesh(projected)
# p.show()