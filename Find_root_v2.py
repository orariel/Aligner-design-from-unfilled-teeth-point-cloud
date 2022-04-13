import numpy as np
import pyvista as pv
import open3d as o3d
#<-------------------------------------------Function Declaration-------------------------------------------------------->
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    return inlier_cloud
#<-------------------------------------------Load Data-------------------------------------------------------->
tooth = pv.read("tooth_7_decimate.stl")
# boundary=tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
# boundary = boundary.compute_normals()
tooth_points=np.asarray(tooth.points)
tooth_normals_b=np.asarray(tooth.point_normals)
arr_index=[]
# for i in range(tooth_points.shape[0]):
#  if(any((a[:]==tooth_points[i]).all(1))==False):
#      arr_index.append([i])
# arr_clean_b= np.delete(tooth_normals_b, arr_index, axis=0)
# sum_nor_b=np.sum(arr_clean_b, axis=0)
# normalized_b=sum_nor_b/np.linalg.norm(sum_nor_b)

#<-------------------------------------------Remove upper part------------------------------------------------------->
tooth=tooth.elevation()
deriv = tooth.compute_derivative(gradient=True,divergence=False)
# deriv.plot(scalars='gradient')
deri_arr=np.asarray(deriv.active_scalars)
deri_arr=deri_arr.reshape(deri_arr.size,1)
# deriv.plot()
# boundary = tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
tooth_xyz=np.asarray(tooth.points)
pcd=pv.PolyData(tooth_xyz)
pcd.plot()
points_to_remove=[]

for i in range(tooth_xyz.shape[0]):
    if (deri_arr[i]>0.6):
        points_to_remove.append([i])
arr_clean = np.delete(tooth_xyz, points_to_remove, axis=0)
tooth_envlope=pv.PolyData(arr_clean)
tooth_envlope.compute_normals()
tooth_envlope.plot()
# pcd_o3d=o3d.geometry.PointCloud()
# pcd_o3d.points= o3d.utility.Vector3dVector(np.asarray(boundary.points))
# pcd_o3d.estimate_normals()

#<-------------------------------------------Clean remains of upper part------------------------------------------------------->
# cl, ind =pcd_o3d.remove_radius_outlier(nb_points=18, radius=0.55)
# clean_upper_surf=display_inlier_outlier(pcd_o3d,ind)
# clean_upper_surf.estimate_normals()
# o3d.visualization.draw_geometries([clean_upper_surf])
#<-------------------------------------------Normal vector calculation------------------------------------------------------>
normals=np.asarray(tooth.point_normals)
normals_clean = np.delete(normals, points_to_remove, axis=0)
sum_nor=np.sum(normals, axis=0)
normalized_x=sum_nor/np.linalg.norm(sum_nor)
arrow=pv.Arrow(start=tooth.center,direction=-normalized_x,shaft_radius=0.1,tip_length=0.55,scale=2)
plot=pv.Plotter()

#<-------------------------------------------Projection to plane------------------------------------------------------>

root_plane = pv.Circle(1, resolution=300)
root_plane.translate(tooth.center, inplace=True)
root_plane.translate([0, 0, -3], inplace=True)
projected = root_plane.project_points_to_plane(origin=root_plane.center,normal=normalized_x,inplace=False)
plot.add_bounding_box()
bboxx=np.asarray(tooth.bounds)
# root_plane_b.translate(bboxx[4], inplace=True)
#
arrow=pv.Arrow(start=tooth.center,direction=-normalized_x,shaft_radius=0.03,tip_length=0.15,scale=4)
plot.add_mesh(tooth)
plot.add_mesh(arrow,color='red')
plot.show()
tooth_and_root=tooth.merge(arrow)
tooth_and_root.save('tooth_7_root.ply')