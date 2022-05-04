import numpy as np
import pyvista as pv
import pymeshfix as mf
import open3d as o3d

#
t_1=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_2_0.stl")
t_2=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_2_1.stl")
t_3=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_3_0.stl")
t_4=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_3_1.stl")
t_5=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_4_0.stl")
t_6=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_4_1.stl")
t_7=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_5_0.stl")
t_8=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_5_1.stl")
t_9=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_6_0.stl")
t_10=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_6_1.stl")
t_11=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_7_0.stl")
t_12=pv.read("C:/Users/OR/PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_8_0.stl")
gum=pv.read("cube_dent_16_1.stl")


t_1=t_1.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_2=t_2.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_3=t_3.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_4=t_4.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_5=t_5.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_6=t_6.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_7=t_7.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_8=t_8.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_9=t_9.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_10=t_10.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_11=t_11.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
t_12=t_12.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)
gum=gum.subdivide(1,subfilter='loop', inplace=True, progress_bar=False)

boundary_gum = gum.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_2=t_2.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
boundary_tooth_3=t_3.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)

p__=pv.Plotter()
p__.add_mesh(t_1)
p__.add_mesh(t_2,color="green")
p__.add_mesh(t_3,color="red")
p__.add_mesh(t_4)
p__.add_mesh(t_5)
p__.add_mesh(t_6)
p__.add_mesh(t_7)
p__.add_mesh(t_8)
p__.add_mesh(t_9)
p__.add_mesh(t_10)
p__.add_mesh(t_11)
p__.add_mesh(t_12)
# p__.add_mesh(gum)
p__.add_mesh(boundary_gum.points,color="blue",line_width=3)
p__.add_mesh(boundary_tooth_2.points,color="yellow",line_width=3)
p__.add_mesh(boundary_tooth_3.points,color="orange",line_width=3)
p__.show()

# arr_of_boundary=np.asarray(bo)


collision_1, ncol = t_2.collision(t_3, cell_tolerance=1.0, generate_scalars=True)
collision_1.plot()




num_of_orig_faces=np.asarray(t_2.n_cells)
all_faces = np.arange(num_of_orig_faces)
arr_of_collision_1 = np.asarray(collision_1['ContactCells'])
arr_to_delete = np.setdiff1d(all_faces, arr_of_collision_1)
area_to_fill= t_2.remove_cells(arr_to_delete)

area_to_fill.save("area_to_fill.stl")
area_to_fill=pv.read("area_to_fill.stl")
area_to_fill.plot()
area_to_fill_normals=np.asarray(area_to_fill.point_normals)
sum_nor=np.sum(area_to_fill_normals, axis=0)
normalized_ex=sum_nor/np.linalg.norm(sum_nor)
normalized_ex[2]=normalized_ex[2]-10
area_to_fill_e=area_to_fill.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
area_to_fill_e.plot()

area_to_fill_e=area_to_fill.extrude((0.245, -0.47, 0.8458-5), capping=True)
area_to_fill_e.save("area_to_fill_e.stl")
area_to_fill_e=pv.read("area_to_fill_e.stl")
area_to_fill_e.plot(line_width=0.4, show_edges=True)
p1_=pv.Plotter()
bbox=pv.Box(bounds=np.asarray(area_to_fill.bounds))


# area_to_fill_e=area_to_fill_e.clip('z',value=2.5,invert=False)

area_to_fill_e =area_to_fill_e.triangulate(inplace=False, progress_bar=False)
area_to_fill_e =area_to_fill_e.elevation()

# deriv.plot()
deriv_arr = np.asarray(area_to_fill_e.active_scalars)
points_to_remove = []

clip=area_to_fill_e.clip_box(np.asarray(bbox.bounds),invert=False)


p1_.add_mesh(t_2)
p1_.add_mesh(clip)

p1_.show()

#
# <-----------------------------For removing gum------------------------------------------>
# f=pv.read("final_pred_1.ply")
# f=f.subdivide(2,subfilter='loop', inplace=True, progress_bar=False)
# f.plot()
# color_arr = np.asarray(f.active_scalars)
# p__=pv.Plotter()
# p__.add_mesh(f)
# p__.show()
# points_to_remove=[]
# for i in range(color_arr.shape[0]):
#     if (color_arr[i]<=180):
#         points_to_remove.append([i])
# fo, ridx=f.remove_points(points_to_remove)
# fo.plot()
# color_arr_2 = np.asarray(f.active_scalars)
#
# m=o3d.io.read_triangle_mesh("final_pred_1.ply")
# m.compute_vertex_normals()
# o3d.visualization.draw_geometries([m])