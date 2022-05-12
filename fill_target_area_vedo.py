import numpy as np
import pyvista as pv
import pymeshfix
import pymeshfix as mf
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

tooth_1 = pv.read("tooth_7_decimate.stl")
# tooth_1=tooth_1.subdivide(2,subfilter='loop', inplace=False, progress_bar=False)
mesh =tooth_1.extrude((0, 0, 15), capping=True)
# mesh.plot()
# tooth_1.plot()
tooth_2 = pv.read("tooth_8_decimate.stl")
p = pv.Plotter()
collision_1, ncol = tooth_1.collision(tooth_2, cell_tolerance=0.0000001, generate_scalars=True)


num_of_orig_faces=np.asarray(tooth_1.n_cells)
all_faces = np.arange(num_of_orig_faces)
arr_of_collision_1 = np.asarray(collision_1['ContactCells'])
arr_to_delete = np.setdiff1d(all_faces, arr_of_collision_1)
area_to_fill= tooth_1.remove_cells(arr_to_delete)

area_to_fill.save("area_to_fill.stl")
area_to_fill=pv.read("area_to_fill.stl")
area_to_fill_e=area_to_fill.extrude((0, 0, 10), capping=True)
area_to_fill_e=area_to_fill_e.clip('z',value=2.5,invert=False)
area_to_fill_e=area_to_fill_e.smooth(0, feature_smoothing=False)
area_to_fill_e=area_to_fill_e.translate((0, 0,-10), inplace=False)
p.add_mesh(area_to_fill,color='red')
p.add_mesh(area_to_fill_e)
p.show()
area_to_fill_e.plot()
area_to_fill_arr=np.asarray(area_to_fill.points)
area_to_fill_pcd=pv.PolyData(area_to_fill_arr)
area_to_fill_pcd.plot()


# # point_a=area_to_fill_arr[10,:]
# # point_b=area_to_fill_arr[60,:]
# #
# # line=pv.Line(pointa=point_a,pointb=point_b)
# # line.save("line_a.stl")
# # coor=point_b-point_a
# # area_to_fill.merge(line)
# # area_to_fill.save("line_a1.stl")
# # plane=pv.Plane(center=area_to_fill.center, direction=(0,0,1), i_size=1, j_size=3.5, i_resolution=5, j_resolution=5)
# # plane=plane.triangulate()
# # plane=plane.merge(area_to_fill)
# # plane.save("ttre.stl")
# # pp=pv.Plotter()
# # pp.add_mesh(line)
# # pp.add_mesh(area_to_fill)
# # pp.add_mesh(plane,line_width=5,show_edges=True)
# # pp.show()
# #
# meshfix = mf.MeshFix(tooth_1)
# holes = meshfix.extract_holes()
# meshfix.repair(verbose=True)
# meshfix.plot()
# meshfix.repair(verbose=True)
# meshfix.plot()
#
# # # area_to_fill_pcd = area_to_fill_pcd.elevation()
# # # area_to_fill_pcd.plot()
# # # deriv = area_to_fill_pcd.compute_derivative(gradient=True, divergence=False)
# # # deri_arr = np.asarray(deriv.active_scalars)
# # # deri_arr = deri_arr.reshape(deri_arr.size, 1)
# # # # points_to_remove=[]
# # # # for i in range(deri_arr.shape[0]):
# # # #     if (deri_arr[i]<1.3):
# # # #         points_to_remove.append([i])
# # # # points_to_remove=np.concatenate(points_to_remove)
# # # # arr_clean_b= np.delete(area_to_fill_arr, points_to_remove, axis=0)
# # # # area_to_fill_pcd_2=pv.PolyData(arr_clean_b)
# # #
# # # # area_to_fill_pcd_2.plot()
# # # # mesh_fix.fill_small_boundaries(nbe=100, refine=True)
# # # # mesh_fix.plot()
# # # # mesh_fix.save("filled_area.stl")
# # # # mesh_fix=pv.read("filled_area.stl")
# # # # p_1=pv.Plotter()
# # # # p_1.add_mesh(tooth_1)
# # # # p_1.add_mesh(tooth_2)
# # # # p_1.add_mesh(boundary_tooth_1,color="red")
# # # # p_1.add_mesh(area_to_fill_pcd,color="blue")
# # # # p_1.show()
# # # # # #
# # # # # convex_mesh=o3d.io.read_triangle_mesh('area_to_fill.stl')
# # # # # convex_mesh.compute_vertex_normals()
# # # # # o3d.visualization.draw_geometries([convex_mesh])
# # # # #
# # # # # pcl = convex_mesh.sample_points_poisson_disk(number_of_points=2000)
# # # # # hull, _ = pcl.compute_convex_hull()
# # # # # hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
# # # # # hull_ls.paint_uniform_color((1, 0, 0))
# # # # # o3d.visualization.draw_geometries([pcl, hull_ls])
# # # # # poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcl, depth=8, width=0, scale=1.1, linear_fit=False)[0]
# # # # # bbox = pcl.get_axis_aligned_bounding_box()
# # # # # p_mesh_crop = poisson_mesh.crop(bbox)
# # # # # hull.compute_vertex_normals()
# # # # # o3d.visualization.draw_geometries([hull])
# # # # # o3d.io.write_triangle_mesh('colse_and_filled.stl',hull)
# # # #
# # # # mesh2=pv.read('colse_and_filled.stl')
# # # # mesh1=pv.read('tooth_7_decimate.stl')
# # # # p__=pv.Plotter()
# # # # p__.add_mesh(mesh1)
# # # # p__.add_mesh(mesh2)
# # # # p__.show()
# # # #
# # # #
