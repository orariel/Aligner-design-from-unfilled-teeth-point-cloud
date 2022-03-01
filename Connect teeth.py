import numpy as np
import open3d as o3d
import pyvista as pv

circle=pv.Circle(0.5)
circle.plot(show_edges=True,line_width=5)

def clean_array(pcd):
    unique_pcd = np.unique(pcd.view([('', pcd.dtype)] * pcd.shape[1]))
    return unique_pcd.view(pcd.dtype).reshape((unique_pcd.shape[0], pcd.shape[1]))


source_arr = np.loadtxt('TXT/1-5_13 bas_int.txt')

source_arr = clean_array(source_arr)
source_arr = source_arr[source_arr[:, 3].argsort()]
p = np.hsplit(source_arr, 4)
# ----------------------------------Creating PCD of all teeth------------------------#
dem = int(source_arr.size / 4)
x_cor = p[0]
y_cor = p[1]
z_cor = p[2]
index = p[3]
xyz_cor_source = np.concatenate((x_cor, y_cor, z_cor), axis=1)
# xyz_cor_source = clean_array(xyz_cor_source)
source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(xyz_cor_source)
source_pcd.estimate_normals()
# o3d.visualization.draw_geometries([source_pcd])

# -------------------------------slicing arry and making list of teeth-(each element contain np.arry of tooth)------------------------#
dem = int(x_cor.size)
teeth_seg = p[3]
list_of_index = clean_array(teeth_seg)
list_tooth = []
p = 0
for j in range(dem - 1):
    if teeth_seg[j] != teeth_seg[j + 1]:
        list_tooth.append(xyz_cor_source[p:j + 1, :])
        p = j + 1
list_tooth.append(xyz_cor_source[p:dem, :])
num_OF_tooth = len((list_tooth))

# -------------------------------Looping on each two tooth------------------------#
# flag=1
pcd_tooth_1 = o3d.geometry.PointCloud()
pcd_pair = o3d.geometry.PointCloud()
pcd_tooth_2 = o3d.geometry.PointCloud()
pcd_tooth = o3d.geometry.PointCloud()
i = 8

# while(i<num_OF_tooth):
#     pcd_tooth.points=o3d.utility.Vector3dVector(list_tooth[i])
#     density =1.1  # 0.08
#     tengent_plane =50 # 100
#     depth_mesh = 9  # 9
#     pcd_tooth.estimate_normals()
#     pcd_tooth.orient_normals_consistent_tangent_plane(tengent_plane)
#     p, t = pcd_tooth.compute_convex_hull()
#     p.orient_triangles()
#     p.compute_vertex_normals()
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_tooth, depth=depth_mesh,linear_fit=False)
#     mesh = mesh.simplify_quadric_decimation(100000)
#     mesh.remove_degenerate_triangles()
#     mesh.remove_duplicated_triangles()
#     mesh.remove_duplicated_vertices()
#     # mesh.remove_non_manifold_edges()
#     bbox = pcd_tooth.get_axis_aligned_bounding_box()
#     p_mesh_crop = mesh.crop(bbox)
#     o3d.io.write_triangle_mesh('tooth_' + str(i) + '.ply', p_mesh_crop)
#     i=i+1
tooth_1 = list_tooth[i]
tooth_2 = list_tooth[i + 1]
pcd_tooth_1.points = o3d.utility.Vector3dVector(tooth_1)
pcd_tooth_2.points = o3d.utility.Vector3dVector(tooth_2)
bbox_1 = pcd_tooth_1.get_axis_aligned_bounding_box()
bbox_1.color = (1, 0, 0)

bbox_2 = pcd_tooth_2.get_axis_aligned_bounding_box()
bbox_2.color = (1, 0, 0)
pcd_tooth_1.estimate_normals()
pcd_tooth_2.estimate_normals()
o3d.visualization.draw_geometries([pcd_tooth_1,pcd_tooth_2,bbox_2,bbox_1])
# while (i < num_OF_tooth - 1):
#     tooth_1 = list_tooth[i]
#     tooth_2 = list_tooth[i + 1]
#     pcd_tooth_1.points = o3d.utility.Vector3dVector(tooth_1)
#     pcd_tooth_2.points = o3d.utility.Vector3dVector(tooth_2)
#     bbox_1 = pcd_tooth_1.get_axis_aligned_bounding_box()
#     bbox_1.color = (1, 0, 0)
#     bbox_2 = pcd_tooth_2.get_axis_aligned_bounding_box()
#     bbox_2.color = (1, 0, 0)
#     # o3d.visualization.draw_geometries([pcd_tooth_1,pcd_tooth_2,bbox_2,bbox_1])
#     # ------------------------------convreting to mesh using Poassion------------------------#
#     pair_tooth_xyz = np.concatenate((tooth_1, tooth_2), axis=0)
#     pcd_pair.points = o3d.utility.Vector3dVector(pair_tooth_xyz)
#     pcd_pair.estimate_normals()
#     density = 2.2  # 0.08
#     tengent_plane = 50  # 100
#     depth_mesh = 8  # 9
#     pcd_pair.orient_normals_consistent_tangent_plane(tengent_plane)
#     p, t = pcd_pair.compute_convex_hull()
#     p.orient_triangles()
#     p.compute_vertex_normals()
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_pair, depth=depth_mesh,
#                                                                                 linear_fit=False)
#     mesh = mesh.simplify_quadric_decimation(100000)
#     mesh.remove_degenerate_triangles()
#     mesh.remove_duplicated_triangles()
#     mesh.remove_duplicated_vertices()
#     mesh.remove_non_manifold_edges()
#
#     # # mesh=mesh.crop(bbox_1)
#     # mesh =mesh.crop(bbox_2)
#     bbox = pcd_pair.get_axis_aligned_bounding_box()
#     p_mesh_crop = mesh.crop(bbox)
#     o3d.io.write_triangle_mesh('pair_mesh_' + str(i) + '.ply', p_mesh_crop)
#     i = i + 1

# tooth_1 = pv.read('pair_mesh_0.ply')
# tooth_2 = pv.read('pair_mesh_1.ply')
# tooth_3 = pv.read('pair_mesh_2.ply')
# tooth_4 = pv.read('pair_mesh_3.ply')
# tooth_5 = pv.read('pair_mesh_4.ply')
# tooth_6 = pv.read('pair_mesh_5.ply')
# tooth_7 = pv.read('pair_mesh_6.ply')
# tooth_8 = pv.read('pair_mesh_7.ply')
# tooth_9 = pv.read('pair_mesh_8.ply')
# tooth_10 = pv.read('pair_mesh_9.ply')
# p = pv.Plotter()
# p.add_mesh(tooth_1)
# p.add_mesh(tooth_2)
# p.add_mesh(tooth_3)
# p.add_mesh(tooth_4)
# p.add_mesh(tooth_5)
# p.add_mesh(tooth_6)
# p.show()
# p.add_mesh(tooth_1)
