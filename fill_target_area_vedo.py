import numpy as np
import open3d as o3d
import pyvista as pv



mesh_p = pv.read("TEST_color_nativevrai.ply")

mesh_p_cur=mesh_p.curvature(curv_type='Maximum', progress_bar=False)
curv =mesh_p.curvature()
mesh_p.plot(scalars=curv)
mesh_t = mesh_p.elevation()
deriv = mesh_t.compute_derivative(gradient=True, divergence=False)
deri_arr = np.asarray(deriv.active_scalars)
color_arr = np.asarray(mesh_p.active_scalars)
points_to_remove = []
arr_xyz = np.asarray(mesh_p.points)
for i in range(color_arr.shape[0]):
    if (color_arr[i]>260):
        points_to_remove.append([i])

points_to_remove=np.concatenate(points_to_remove)
re,rix = mesh_p.remove_points(points_to_remove)
re.plot()
points_to_remove=[]
index=np.asarray(re.points).shape[0]
for i in range(index):
    if (deri_arr[i]<17):
        points_to_remove.append([i])
# arr_clean_b = np.delete(arr_xyz, points_to_remove, axis=0)
points_to_remove=np.concatenate(points_to_remove)
re,rix = re.remove_points(points_to_remove)
re.plot()
# deriv.plot()


#
#
# # Create the mesh
# cube = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(triangles):
#     for j in range(3):
#         cube.vectors[i][j] = vertices[f[j],:]
#
# # Write the mesh to file "cube.stl"
# cube.save('cube.stl')
#
# # <-------------------------------------------------------------------------->
# # tooth_1 = pv.read("tooth_1_decimate.stl")
# # tooth_2 = pv.read("tooth_2_decimate.stl")
# # p = pv.Plotter()
# # collision_1, ncol = tooth_1.collision(tooth_2, cell_tolerance=2.5001, generate_scalars=True)
# # p.add_mesh(collision_1)
# # p.show()
# # all_faces = np.arange(13594)
# # arr_of_collision_1 = np.asarray(collision_1['ContactCells'])
# # arr_to_delete = np.setdiff1d(all_faces, arr_of_collision_1)
# # area_to_fill= tooth_1.remove_cells(arr_to_delete)
# # area_to_fill.plot()
# # full=area_to_fill.fill_holes(1000)
# # # deriv = full.compute_derivative(gradient=True, divergence=False)
# # # deri_arr = np.asarray(deriv.active_scalars)
# # # deriv.plot()
# # p_1=pv.Plotter()
# # p_1.add_mesh(tooth_1)
# # p_1.add_mesh(tooth_2)
# #
# # p_1.show()
