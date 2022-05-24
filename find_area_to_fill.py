# <-----------------------------------imports packages------------------------------------------------------->
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pyvista as pv
import open3d as o3d
import pymeshfix as mf
import numpy as np
from math import sqrt
import copy
import for_or_order_list
import matplotlib.pyplot as plt
# <-----------------------------------loading files------------------------------------------------------->

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


gum=pv.read("cube_dent_16_1.stl")

# boundary_gum.plot()
# gum.plot()
pp=pv.Plotter()
# pp.add_mesh(gum)
# pp.add_mesh(before_seg)
# pp.add_mesh(t_1)
# pp.add_mesh(t_2)
# pp.add_mesh(t_3)
# pp.add_mesh(t_4)
# pp.add_mesh(t_5)
# pp.add_mesh(t_6)
# pp.add_mesh(t_7)
# pp.add_mesh(t_8)
# pp.add_mesh(t_9)
# pp.add_mesh(t_10)
# pp.add_mesh(t_11)
# pp.add_mesh(t_12)
# pp.show()
#
# # <-----------------------------------subdivide files------------------------------------------------------->
t_1=t_1.subdivide(2,subfilter='loop', inplace=True, progress_bar=False)
t_2=t_2.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_3=t_3.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_4=t_4.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_5=t_5.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_6=t_6.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_7=t_7.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_8=t_8.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_9=t_9.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_10=t_10.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_11=t_11.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
t_12=t_12.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)

gum=gum.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)



# <--------------------------------------------boundary extract----------------------------------------------->

boundary_gum = gum.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
# boundary_tooth_1=t_1.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
# boundary_tooth_2=t_2.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
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
# <-----------------------------------separate the gum boundary and tooth boundary---------------------------->
tooth=t_11
tooth.plot()
boundary_tooth=tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth.plot()
# pp.add_mesh(tooth)
# pp.add_mesh(boundary_tooth_3.points,color="red")
# pp.add_mesh(boundary_gum.points,color="blue")
# pp.add_mesh(gum,color="yellow")
# pp.show()
b_tooth_points=np.asarray(boundary_tooth.points)
b_gum_points=np.asarray(boundary_gum.points)
b_of_area_to_fill = np.asarray([ind for ind in b_tooth_points if ind not in b_gum_points])
pcd=pv.PolyData(b_of_area_to_fill)
pcd.plot()
p__1=pv.Plotter()
#
# # <-----------------------------------clustering the PCD of the area to fill---------------------------->
pcd_to_cluster=o3d.geometry.PointCloud()
pcd_to_cluster.points=o3d.utility.Vector3dVector(b_of_area_to_fill)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd_to_cluster.cluster_dbscan(eps=2, min_points=5, print_progress=True))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd_to_cluster.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd_to_cluster])
p__2 = pv.Plotter()
# # <-------------------------------separate the cluster PCD-------------------------------->
if (max_label!=0):
    points_to_remove=[]
    for i in range(labels.size):
       if(labels[i]==0):
           points_to_remove.append([i])
    points_to_remove=np.concatenate(points_to_remove)
    arr_cluster_1= np.delete(b_of_area_to_fill, points_to_remove, axis=0)
    points_to_remove=[]
    for i in range(labels.size):
      if(labels[i]==1):
           points_to_remove.append([i])
    points_to_remove=np.concatenate(points_to_remove)
    arr_cluster_2= np.delete(b_of_area_to_fill, points_to_remove, axis=0)
    pcd_1=pv.PolyData(arr_cluster_1)
    pcd_1=pcd_1.elevation()
    deriv = pcd_1.compute_derivative(gradient=True,divergence=False)
    # deriv.plot()
    deri_arr=np.asarray(deriv.active_scalars)
    deri_arr=deri_arr.reshape(deri_arr.size,1)


#<-----------------------------------------sorting the bounds for part 1------------------------------------------------------>
    order_list = for_or_order_list.get_sorted_arr(arr_cluster_1, 0)

# <---------------------------------------Addinf points for cluster_1----------------------------------------------------------------------->

    arr_sorted=[]
    order_list=np.asarray(order_list)
    for i in range(order_list.size):
        arr_sorted.append(arr_cluster_1[order_list[i]])
    points_to_add=[]
    arr_sorted=np.asarray(arr_sorted)
    for i in range(order_list.size-1):
        line=pv.Line(pointa=arr_sorted[i],pointb=arr_sorted[i+1],resolution=10)
        points_to_add.append(arr_sorted[i])
        for j in range(10):
          points_to_add.append(line.points[j])
        points_to_add.append(arr_sorted[i+1])
    line = pv.Line(pointa=points_to_add[0], pointb=points_to_add[-1], resolution=200)
    points_to_add=np.concatenate(points_to_add).reshape(len(points_to_add), 3)

    lines_to_add=[]
    for i in range(int(points_to_add.shape[0]/2)):
        if i==0:
            line=pv.Line(pointa=points_to_add[0],pointb=points_to_add[-1],resolution=200)
            lines_to_add.append(line.points)
        else:
            line = pv.Line(pointa=points_to_add[i], pointb=points_to_add[-1*i],resolution=200)
            lines_to_add.append(line.points)
    lines_to_add=np.concatenate(lines_to_add)
    all_points=np.concatenate((lines_to_add,points_to_add),axis=0)
    # points_to_add=np.concatenate(points_to_add)
    # lines_to_add=np.asarray(lines_to_add)
    # pcd_sort = pv.PolyData(lines_to_add)
    # pcd_sort.plot()
    # line = pv.Line(pointa=arr_sorted[0], pointb=arr_sorted[-1], resolution=200)
    # lines_to_add.append(line.points)
    # lines_to_add=np.concatenate(lines_to_add)
    # arr_lines=np.concatenate((arr_sorted,lines_to_add), axis=0)

    arr_to_add=[]
    # for i in range(1,arr_cluster_2.shape[0]):
    #
    #      line = pv.Line(pointa=arr_sorted[i], pointb=arr_sorted[-1*i], resolution=100)
    #      arr_to_add.append(line.points)
    # line = pv.Line(pointa=arr_sorted[0], pointb=arr_sorted[-1], resolution=200)
    # arr_to_add.append(line.points)
    # arr_to_add=np.concatenate(arr_to_add)


    # arr_lines=np.concatenate((arr_lines,arr_to_add), axis=0)
    pcd_sort=pv.PolyData(all_points)
    # pcd_sort.plot()
    # <-----------------------------------------PCD TO MESH PYVISTA------------------------------------------------------>
    points = pv.wrap(all_points)
    pcd_to_mesh_1=pv.PolyData(all_points)
    surf_1 = points.delaunay_2d()
    surf_1.save("pv_pcd_to_mesh.stl")

    # surf_1.plot()

    # <-----------------------------------------sorting the bounds for part 2------------------------------------------------------>
    order_list = for_or_order_list.get_sorted_arr(arr_cluster_2, 0)
    # print(order_list)
    # <---------------------------------------Addinf points for cluster_2----------------------------------------------------------------------->

    arr_sorted = []
    order_list = np.asarray(order_list)
    for i in range(order_list.size):
        arr_sorted.append(arr_cluster_2[order_list[i]])
    points_to_add = []
    arr_sorted = np.asarray(arr_sorted)
    for i in range(order_list.size - 1):
        line = pv.Line(pointa=arr_sorted[i], pointb=arr_sorted[i + 1], resolution=10)
        points_to_add.append(arr_sorted[i])
        for j in range(10):
            points_to_add.append(line.points[j])
        points_to_add.append(arr_sorted[i + 1])
    line = pv.Line(pointa=points_to_add[0], pointb=points_to_add[-1], resolution=200)
    points_to_add = np.concatenate(points_to_add).reshape(len(points_to_add), 3)

    lines_to_add = []
    for i in range(int(points_to_add.shape[0] / 2)):
        if i == 0:
            line = pv.Line(pointa=points_to_add[0], pointb=points_to_add[-1], resolution=200)
            lines_to_add.append(line.points)
        else:
            line = pv.Line(pointa=points_to_add[i], pointb=points_to_add[-1 * i], resolution=200)
            lines_to_add.append(line.points)
    lines_to_add = np.concatenate(lines_to_add)
    all_points = np.concatenate((lines_to_add, points_to_add), axis=0)

    arr_to_add = []
    pcd_sort = pv.PolyData(all_points)
    # pcd_sort.plot()

    # <-----------------------------------------PCD TO MESH PYVIST_2------------------------------------------------------>
    points = pv.wrap(all_points)
    pcd_to_mesh_2=pv.PolyData(all_points)
    surf_2 = points.delaunay_2d()
    surf_2.save("pv_pcd_to_mesh.stl")
    # surf_2.plot()
    p__=pv.Plotter()
    p__.add_mesh(pcd_to_mesh_1,color="red")
    p__.add_mesh(pcd_to_mesh_2,color="red")
    p__.add_mesh(tooth,color="tan")
    p__.export_gltf('t_12_pcd.gltf')
    p__.export_obj('t_12_pcd.obj')
    p__.show()

# # <-------------------------------convert PCD to mesh-------------------------------->
#     pcd_to_mesh=o3d.geometry.PointCloud()
#     pcd_to_mesh.points=o3d.utility.Vector3dVector(all_points)
#     pcd_to_mesh.estimate_normals()
#     # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_to_mesh, depth=9, width=0, scale=2.1, linear_fit=False)[0]
#     o3d.visualization.draw_geometries([pcd_to_mesh])
#     pcd_to_mesh.estimate_normals()
#     distances = pcd_to_mesh.compute_nearest_neighbor_distance()
#     avg_dist = np.mean(distances)
#     radius =5* avg_dist
#     bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_to_mesh,o3d.utility.DoubleVector([radius,radius*2]))
#     bpa_mesh.remove_degenerate_triangles()
#     bpa_mesh.remove_duplicated_triangles()
#     bpa_mesh.remove_duplicated_vertices()
#     bpa_mesh.remove_non_manifold_edges()
#     o3d.visualization.draw_geometries([bpa_mesh])
#     o3d.io.write_triangle_mesh("ss.stl",bpa_mesh)
    # bpa_mesh =bpa_mesh.subdivide_loop(number_of_iterations=1)
    # pcd_to_mesh_2 = o3d.geometry.PointCloud()
    # pcd_to_mesh_2.points = o3d.utility.Vector3dVector(arr_cluster_2)
    # pcd_to_mesh_2.estimate_normals()
    # distances = pcd_to_mesh_2.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius =5* avg_dist
    # bpa_mesh_2 = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_to_mesh_2,
    #                                                                              o3d.utility.DoubleVector(
    #                                                                                  [radius, radius * 2]))
    # # dec_mesh = bpa_mesh_2.simplify_quadric_decimation(1000000)
    # bpa_mesh_2.remove_degenerate_triangles()
    # bpa_mesh_2.remove_duplicated_triangles()
    # bpa_mesh_2.remove_duplicated_vertices()
    # bpa_mesh_2.remove_non_manifold_edges()
    # o3d.visualization.draw_geometries([bpa_mesh_2])
#     o3d.io.write_triangle_mesh("area_filled_2.ply", bpa_mesh_2)
#
#     area_1 = pv.read("area_filled.ply")
#     area_2 = pv.read("area_filled_2.ply")
#
#     p__2.add_mesh(area_2, color='red')
#     p__2.add_mesh(t_3)
#     p__2.add_mesh(area_1, color="red")
#     p__2.add_mesh(boundary_gum)
#
#     p__2.show()
#     merged = t_3.merge([area_1, area_2])
#     merged.save("full_tooth_12_r_5.stl")
# points = pv.wrap(arr_cluster_2)
# surf = points.delaunay_2d()
# surf = surf.translate((10, 0, 0), inplace=False)
# t_3_offset = t_3.translate((10, 0, 0), inplace=False)
#
# pcd_s = pv.PolyData(surf.points)
# p__3 = pv.Plotter()
# p__3.add_mesh(merged)
# p__3.add_mesh(t_3_offset)
# p__3.add_mesh(surf, color="green")
#
# p__3.show()
#
# print("only one cluster")
#
# #
# if (max_label == 0):
#     pcd_to_mesh = pcd_to_cluster
#     pcd_to_mesh.estimate_normals()
#     distances = pcd_to_mesh.compute_nearest_neighbor_distance()
#     avg_dist = np.mean(distances)
#     radius = 5 * avg_dist
#     bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_to_mesh, o3d.utility.DoubleVector(
#         [radius, radius * 2]))
#     bpa_mesh.remove_degenerate_triangles()
#     bpa_mesh.remove_duplicated_triangles()
#     bpa_mesh.remove_duplicated_vertices()
#     bpa_mesh.remove_non_manifold_edges()
#
#     o3d.visualization.draw_geometries([bpa_mesh])
#     o3d.io.write_triangle_mesh("area_filled.ply", bpa_mesh)
#     area_1 = pv.read("area_filled.ply")
#     p__2.add_mesh(t_9)
#     p__2.add_mesh(area_1, color="red")
#     p__2.add_mesh(boundary_gum)
#     p__2.show()
#     merged = t_9.merge([area_1])
#     merged.save("full_tooth_10_r_5.stl")

# o3d.visualization.draw_geometries([bpa_mesh])
#     o3d.io.write_triangle_mesh("area_filled.ply", bpa_mesh)


# arr_of_boundary=np.asarray(bo)
#
# collision_1, ncol = t_2.collision(t_3, cell_tolerance=1.0, generate_scalars=True)
# collision_1.plot()
#
#
# num_of_orig_faces=np.asarray(t_2.n_cells)
# all_faces = np.arange(num_of_orig_faces)
# arr_of_collision_1 = np.asarray(collision_1['ContactCells'])
# arr_to_delete = np.setdiff1d(all_faces, arr_of_collision_1)
# area_to_fill= t_2.remove_cells(arr_to_delete)
#
# area_to_fill.save("area_to_fill.stl")
# area_to_fill=pv.read("area_to_fill.stl")
# area_to_fill.plot()
# area_to_fill_normals=np.asarray(area_to_fill.point_normals)
# sum_nor=np.sum(area_to_fill_normals, axis=0)
# normalized_ex=sum_nor/np.linalg.norm(sum_nor)
# normalized_ex[2]=normalized_ex[2]-10
# area_to_fill_e=area_to_fill.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
# area_to_fill_e.plot()
#
# area_to_fill_e=area_to_fill.extrude((0.245, -0.47, 0.8458-5), capping=True)
# area_to_fill_e.save("area_to_fill_e.stl")
# area_to_fill_e=pv.read("area_to_fill_e.stl")
# area_to_fill_e.plot(line_width=0.4, show_edges=True)
# p1_=pv.Plotter()
# bbox=pv.Box(bounds=np.asarray(area_to_fill.bounds))
#
#
# area_to_fill_e=area_to_fill_e.clip('z',value=2.5,invert=False)
#
# area_to_fill_e =area_to_fill_e.triangulate(inplace=False, progress_bar=False)
# area_to_fill_e =area_to_fill_e.elevation()
#
# deriv.plot()
# deriv_arr = np.asarray(area_to_fill_e.active_scalars)
# points_to_remove = []
#
# clip=area_to_fill_e.clip_box(np.asarray(bbox.bounds),invert=False)
#
#
# p1_.add_mesh(t_2)
# p1_.add_mesh(clip)
#
# p1_.show()
#
# #

