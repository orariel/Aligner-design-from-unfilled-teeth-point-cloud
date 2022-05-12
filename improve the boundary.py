import numpy as np
import pyvista as pv
import pymeshfix
import pymeshfix as mf
import open3d as o3d
from math import sqrt

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(1,len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    return inlier_cloud

t_1=pv.read("C:/Users/OR\PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_2_0.stl")
t_2=pv.read("C:/Users/OR\PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_2_1.stl")
t_3=pv.read("C:/Users/OR\PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_3_0.stl")
t_4=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_3_1.stl")
t_5=pv.read("C:/Users/OR\PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_4_0.stl")
t_6=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_4_1.stl")
t_7=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_5_0.stl")
t_8=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_5_1.stl")
t_9=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_6_0.stl")
t_10=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_6_1.stl")
t_11=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_7_0.stl")
t_12=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_8_0.stl")
before_seg=pv.read("final_pred_1.ply")
before_seg=before_seg.subdivide(2,subfilter='loop', inplace=True, progress_bar=False)
before_seg.save("final_pred_1_seg.ply")
arr_before_seg=np.asarray(before_seg.points)
np.savetxt("before_seg.txt",arr_before_seg)
#
t_1=t_1.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_2=t_2.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_3=t_3.subdivide(2,subfilter='loop', inplace=True, progress_bar=False)
t_4=t_4.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_5=t_5.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_6=t_6.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_7=t_7.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_8=t_8.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_9=t_9.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_10=t_10.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_11=t_11.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_12=t_12.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
#
# gum=pv.read("cube_dent_16_1.stl")
# gum=gum.subdivide(2,subfilter='loop', inplace=True, progress_bar=False)
#
# boundary_gum = gum.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_1=t_1.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_2=t_2.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_3=t_3.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_4=t_4.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_5=t_5.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_6=t_6.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_7=t_7.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_8=t_8.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_9=t_9.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_10=t_10.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_11=t_11.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_12=t_12.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
#
all_bounds=boundary_tooth_1.merge([boundary_tooth_1,boundary_tooth_2,boundary_tooth_3,boundary_tooth_4,boundary_tooth_5,boundary_tooth_6,boundary_tooth_7,boundary_tooth_8,boundary_tooth_9,boundary_tooth_10,boundary_tooth_11,boundary_tooth_12])


pp=pv.Plotter()
# pp.add_mesh(boundary_gum.points,color="red",line_width=2)
pp.add_mesh(all_bounds.points,color="blue",line_width=2)


before_seg = before_seg.elevation()
deriv = before_seg.compute_derivative(gradient=True, divergence=False)
deriv_arr = np.asarray(deriv.active_scalars)
points_to_remove=[]

curv_arr=np.asarray(before_seg.curvature(curv_type='Minimum', progress_bar=False))
for i in range(curv_arr.size):
    if(curv_arr[i]>-0.5):
        points_to_remove.append([i])
before_seg_curv,rx=before_seg.remove_points(points_to_remove)
before_seg_curv.plot()
arr_before_seg=np.asarray(before_seg.points)
arr_of_teeth_b=np.asarray(all_bounds.points)

before_seg_curv = before_seg_curv.elevation()
deriv = before_seg_curv.compute_derivative(gradient=True, divergence=False)
deriv_arr = np.asarray(deriv.active_scalars)
before_seg_curv.plot()
points_to_remove=[]
for i in range(deriv_arr.size):
    if(deriv_arr[i]<5.5):
        points_to_remove.append([i])
before_seg_derv,rx2=before_seg_curv.remove_points(points_to_remove)
before_seg_derv.plot()

conn = before_seg_derv.connectivity(largest=False)
conn.plot(cmap=['red', 'blue'])
before_seg_derv_b=before_seg_derv.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
before_seg_derv_b.plot()
# pp.add_mesh(before_seg.points,color="yellow",scalars=curv)
# pp.add_mesh(boundary_tooth_2,line_width=2)
# pp.add_mesh(boundary_tooth_3,line_width=2)
# pp.add_mesh(boundary_tooth_4,line_width=2)
# pp.add_mesh(boundary_tooth_5,line_width=2)
# pp.add_mesh(boundary_tooth_6,line_width=2)
# pp.add_mesh(boundary_tooth_7,line_width=2)
# pp.add_mesh(boundary_tooth_8,line_width=2)
# pp.add_mesh(boundary_tooth_9,line_width=2)
# pp.add_mesh(boundary_tooth_10,line_width=2)
# pp.add_mesh(boundary_tooth_11,line_width=2)
# pp.add_mesh(boundary_tooth_12,line_width=2)

#

dataset =arr_before_seg
dataset.shape

boudary = arr_of_teeth_b
boudary.shape

liste_neigbors = []
for i in range(arr_of_teeth_b.shape[0]):
    neighbors = get_neighbors(dataset, arr_of_teeth_b[i,:], 20)
    liste_neigbors.append(neighbors)
liste_neigbors=np.concatenate(liste_neigbors)
pcd_knn=pv.PolyData(liste_neigbors)
pp.add_mesh(pcd_knn,color="black")
pp.show()
#
arr_a=np.asarray(before_seg_derv_b.points)

corrected_b = np.asarray([x for x in arr_a if x in liste_neigbors])
pcd_to_outlier=o3d.geometry.PointCloud()
pcd_to_outlier.points=o3d.utility.Vector3dVector(corrected_b)
cl, ind = pcd_to_outlier.remove_radius_outlier(nb_points=30, radius=0.05)
display_inlier_outlier(pcd_to_outlier, ind)


pcd_knn=pv.PolyData(corrected_b)
pcd_knn.plot()
np.savetxt("pcd.txt",corrected_b)
b=np.loadtxt("pcd.txt")
pcd_to_outlier=o3d.geometry.PointCloud()
pcd_to_outlier.points=o3d.utility.Vector3dVector(b)
cl, ind = pcd_to_outlier.remove_radius_outlier(nb_points=12, radius=2.45)
pcd_final=display_inlier_outlier(pcd_to_outlier, ind)
o3d.visualization.draw_geometries([pcd_final])
final=pv.read("final_pred_1.ply")
final=final.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
pcdd=pv.PolyData(b)
p__=pv.Plotter()
p__.add_mesh(final)
p__.add_mesh(pcdd)
p__.show()

