import numpy as np
import pyvista as pv
import open3d as o3d


#
# mesh_p = pv.read("TEST_color_nativevrai.ply")
# mesh_p.plot()# if we dont plot the mesh first so we get ARR_OF_COLOR[(N,3] not [N,1]
# mesh_t = mesh_p.elevation()
# deriv = mesh_t.compute_derivative(gradient=True, divergence=False)
# deriv_arr = np.asarray(deriv.active_scalars)
# color_arr = np.asarray(mesh_p.active_scalars)
# points_to_remove = []
# arr_xyz = np.asarray(mesh_p.points)
# for i in range(color_arr.shape[0]):
#     if (color_arr[i] > 240 and deriv_arr[i]<28):
#         points_to_remove.append([i])
# points_to_remove = np.concatenate(points_to_remove)
# re, rix = mesh_p.remove_points(points_to_remove)
# re.save("rr.ply")
# re.plot()
# deriv.plot()for ELI

##

tooth_1 = pv.read("tooth_1_decimate.stl")
tooth_2 = pv.read("tooth_2_decimate.stl")
p = pv.Plotter()
collision_1, ncol = tooth_1.collision(tooth_2, cell_tolerance=2.5001, generate_scalars=True)
p.add_mesh(collision_1)
p.show()
all_faces = np.arange(13594)
arr_of_collision_1 = np.asarray(collision_1['ContactCells'])
arr_to_delete = np.setdiff1d(all_faces, arr_of_collision_1)
area_to_fill= tooth_1.remove_cells(arr_to_delete)
area_to_fill.plot()
p_1=pv.Plotter()
p_1.add_mesh(collision_1)
p_1.add_mesh(tooth_2)

p_1.show()
