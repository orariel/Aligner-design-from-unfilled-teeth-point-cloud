import numpy as np
import pyvista as pv
import open3d as o3d
#<-------------------------------------------Function Declaration-------------------------------------------------------->
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    return inlier_cloud
#<-------------------------------------------Load Data-------------------------------------------------------->
tooth = pv.read("full_tooth_2_r_5.stl")
tooth=tooth.triangulate()
# tooth=tooth.subdivide(1,subfilter='loop', inplace=False, progress_bar=False)
tooth.plot()
tooth_points=np.asarray(tooth.points)
tooth_normals_b=np.asarray(tooth.point_normals)

#<-------------------------------------------Remove upper part------------------------------------------------------->
tooth=tooth.elevation()
deriv = tooth.compute_derivative(gradient=True,divergence=False)
deri_arr=np.asarray(deriv.active_scalars)
deri_arr=deri_arr.reshape(deri_arr.size,1)
points_to_remove=[]
max_hight=float(deri_arr.max())*0.95
for i in range(deri_arr.shape[0]):
    if (deri_arr[i]>max_hight):
        points_to_remove.append([i])
tooth_envlope,rx=tooth.remove_points(points_to_remove)
tooth_envlope.plot()

#<------------------------------------------SLICING 2 SLICES------------------------------------------------------->
slice_z, R=tooth_envlope.clip(normal='z', origin=None, invert=True, value=2,return_clipped=True)
slice_z2,R2=slice_z.clip(normal='z', origin=None, invert=True, value=2+0.5,return_clipped=True)
R2.plot()




#<-------------------------------------------Clean remains of upper part------------------------------------------------------->
# cl, ind =pcd_o3d.remove_radius_outlier(nb_points=18, radius=0.55)
# clean_upper_surf=display_inlier_outlier(pcd_o3d,ind)
# clean_upper_surf.estimate_normals()
# o3d.visualization.draw_geometries([clean_upper_surf])
#<-------------------------------------------Normal vector calculation------------------------------------------------------>
# normals=np.asarray(tooth.point_normals)
# normals_clean = np.delete(normals, points_to_remove, axis=0)
# sum_nor=np.sum(normals, axis=0)
# normalized_x=sum_nor/np.linalg.norm(sum_nor)
# arrow=pv.Arrow(start=tooth.center,direction=-normalized_x,shaft_radius=0.1,tip_length=0.55,scale=2)
# plot=pv.Plotter()
#
# #<-------------------------------------------Projection to plane------------------------------------------------------>
#
# root_plane = pv.Circle(1, resolution=300)
# root_plane.translate(tooth.center, inplace=True)
# root_plane.translate([0, 0, -3], inplace=True)
# projected = root_plane.project_points_to_plane(origin=root_plane.center,normal=normalized_x,inplace=False)
# plot.add_bounding_box()
# bboxx=np.asarray(tooth.bounds)
# # root_plane_b.translate(bboxx[4], inplace=True)
# #
# arrow=pv.Arrow(start=tooth.center,direction=-normalized_x,shaft_radius=0.03,tip_length=0.15,scale=4)
# plot.add_mesh(tooth)
# plot.add_mesh(arrow,color='red')
# plot.show()
# tooth_and_root=tooth.merge(arrow)
# tooth_and_root.save('tooth_7_root.ply')