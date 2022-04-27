import numpy as np
import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt
# <-------------------------------------------Function Declaration-------------------------------------------------------->
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    return inlier_cloud
# <-------------------------------------------Load Data------------------------------------------------------------------->

tooh_number=2
file_name="tooth_"+str(tooh_number)+"_decimate.stl"
tooth = pv.read(file_name)
tooth_points = np.asarray(tooth.points)
tooth_normals_b = np.asarray(tooth.point_normals)
arr_index = []
# <-------------------------------------------Remove upper part------------------------------------------------------->
tooth = tooth.elevation()
deriv = tooth.compute_derivative(gradient=True, divergence=False)
deri_arr = np.asarray(deriv.active_scalars)
deri_arr = deri_arr.reshape(deri_arr.size, 1)
tooth_xyz = np.asarray(tooth.points)
pcd = pv.PolyData(tooth_xyz)
# pcd.plot()
points_to_remove = []
if(tooh_number>=1and tooh_number<=5):
    max_deri_value = np.amax(deri_arr) * 0.2
    threshold = 1.95
else:
    max_deri_value = np.amax(deri_arr) *0.2
    threshold = 0.2
for i in range(tooth_xyz.shape[0]):
    if (deri_arr[i] <max_deri_value):
        points_to_remove.append([i])
mesh_crown,ridx=tooth.remove_points(points_to_remove)
points_crown=np.asarray(mesh_crown.points)

arr_crown = np.delete(tooth_xyz, points_to_remove, axis=0)
pcd_crown = pv.PolyData(arr_crown)
pcd_crown.compute_normals()
 # <-------------------------------------------Normal vector calculation------------------------------------------------------>
normals = np.asarray(tooth.point_normals)
normals_clean = np.delete(normals, points_to_remove, axis=0)
sum_nor = np.sum(normals, axis=0)
normalized_x = sum_nor / np.linalg.norm(sum_nor)
plot = pv.Plotter()
# <-------------------------------------------Adding arrow and plot------------------------------------------------------>
arrow_root = pv.Arrow(start=tooth.center, direction=-normalized_x, shaft_radius=0.05, tip_length=0.15, scale=4)
plot.add_mesh(tooth)
plot.add_mesh(arrow_root, color='red')
plot.add_mesh(mesh_crown,color='blue')

# <-------------------------------------------Calc tangent vector, only for teeth 1-6 11-15 ------------------------------------------------------>
boundary=tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)

# deriv_boundary= boundary.compute_derivative(gradient=True, divergence=False)
# deri_boundary_arr = np.asarray(deriv_boundary.active_scalars)
# boundary_arr=np.asarray(boundary.points)
# points_to_remove = []
#
# for i in range(deri_boundary_arr.shape[0]):
#     if (deri_boundary_arr[i]<threshold):
#         points_to_remove.append([i])
#
# arr_clean_b= np.delete(boundary_arr, points_to_remove, axis=0)
# pcd_tan=o3d.geometry.PointCloud()
# pcd_tan.points=o3d.utility.Vector3dVector(arr_clean_b)
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         pcd_tan.cluster_dbscan(eps=1.8, min_points=10, print_progress=True))
#
# # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# # colors[labels < 0] = 0
# # pcd_tan.colors = o3d.utility.Vector3dVector(colors[:, :3])
# xyz_pcd_tan=np.asarray(pcd_tan.points)
# clustering_to_split_a=[]
# clustering_to_split_b=[]
# for i in range(xyz_pcd_tan.shape[0]):
#     if(labels[i]==0):
#         clustering_to_split_a.append([i])
#     if(labels[i]==1):
#         clustering_to_split_b.append([i])
# group_a= np.delete(xyz_pcd_tan, clustering_to_split_a, axis=0)
# group_b=np.delete(xyz_pcd_tan, clustering_to_split_b, axis=0)
# pcd_group_a=pv.PolyData(group_a)
# pcd_group_b=pv.PolyData(group_b)
# tan_vector=np.asarray(pcd_group_a.center)-np.asarray(pcd_group_b.center)
# tan_normal=tan_vector/np.linalg.norm(tan_vector)
# line = pv.Line((pcd_group_a.center), (pcd_group_b.center))
# arrow_tan = pv.Arrow(start=pcd_group_b.center, direction=tan_normal, shaft_radius=0.03, tip_length=0.15, scale=8)
# arrow_tan
normal_cr = np.asarray(mesh_crown.point_normals)
sum_nor_c = np.sum(normal_cr, axis=0)
normalized_c = sum_nor_c / np.linalg.norm(sum_nor_c)
arrow_cr = pv.Arrow(start=mesh_crown.center, direction=normalized_c, shaft_radius=0.03, tip_length=0.15, scale=8)
plot.add_mesh(tooth)
plot.add_mesh(mesh_crown,color='green')

plot.show()
mesh_crown = mesh_crown.elevation()
deriv_c =mesh_crown.compute_derivative(gradient=True, divergence=False)
deri_c_arr = np.asarray(deriv_c.active_scalars)
mesh_crown.plot()
