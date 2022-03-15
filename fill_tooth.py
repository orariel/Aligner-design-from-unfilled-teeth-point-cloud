import math
import open3d as o3d
import numpy as np
import pyvista as pv

# <------------------------------------------------load mesh create circle------------------------------------------>
tooth_to_fill = pv.read('tooth_11_decimate.stl')
boundary = tooth_to_fill.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
plott = pv.Plotter()
plott.add_mesh(boundary, color='blue')
plott.add_mesh(tooth_to_fill)
points_boundary = boundary.points
root_border = pv.Circle(1, resolution=points_boundary.shape[0])
arr_border=np.asarray(root_border.points)
arr_border=arr_border[arr_border[:,1].argsort()]
c=np.asarray(boundary.points)
c = c[c[:,1].argsort()]
point_cloud_c = pv.PolyData(arr_border)

points_boundary=c


# pp.add_mesh(point_cloud)
# point_cloud_c = pv.PolyData(arr_border)
# pp.add_mesh(point_cloud_c)
# pp.show()


# <------------------------------------------------Add points to circle------------------------------------------>
x_c = []
y_c = []
z_max_tooth = tooth_to_fill.bounds[4]

beta = np.linspace(0, 2 * np.pi, num=10000, endpoint=False)
for i in range(beta.size):
    p = np.random.random(1)
    x_c.append(1 * p * math.cos(beta[i]))
    y_c.append(1 * p * math.sin(beta[i]))
x_c = np.asarray(x_c)
y_c = np.asarray(y_c)
z_c = np.full((len(x_c), 1), 0)

xyz_to_add_c = np.concatenate((x_c, y_c, z_c), axis=1)
# arr_root = np.concatenate((xyz_to_add_c,arr_border), axis=0)
point_cloud = pv.PolyData(arr_border)
point_inside = pv.PolyData(xyz_to_add_c)
root_border.translate(tooth_to_fill.center, inplace=True)
point_inside.translate(tooth_to_fill.center, inplace=True)
root_border.translate([0, 0, z_max_tooth], inplace=True)
point_inside.translate([0, 0, z_max_tooth], inplace=True)
point_cloud_c.translate([0, 0, z_max_tooth], inplace=True)
point_cloud_c.translate(tooth_to_fill.center, inplace=True)
xyz_to_add_c =np.asarray(point_inside.points)

root_border = np.asarray(root_border.points)

arr_border=np.asarray(point_cloud_c.points)
# <------------------------------------------------RAY-trace----------------------------------------->

# p = pv.Plotter()
tooth_to_fill = tooth_to_fill.merge(point_cloud)

for i in range(np.asarray(boundary.points).shape[0]):
     stop = points_boundary[i, :]
     start = arr_border[i, :]
     points, ind = tooth_to_fill.ray_trace(start, stop)
     ray = pv.Line(start, stop, resolution=50)
     intersection = pv.PolyData(points)
     tooth_to_fill = tooth_to_fill.merge(ray)
#
# tooth_to_fill.plot()
full_tooth_arr = np.asarray(tooth_to_fill.points)
point_cloud = pv.PolyData(full_tooth_arr)

point_cloud.plot(eye_dome_lighting=True)



pcd = o3d.geometry.PointCloud()
# pcd_inside_c=o3d.geometry.PointCloud()
# pcd_inside_c = o3d.utility.Vector3dVector(xyz_to_add_c)

full_tooth_arr = np.concatenate((xyz_to_add_c,full_tooth_arr), axis=0)
pcd.points = o3d.utility.Vector3dVector(full_tooth_arr)

o3d.visualization.draw_geometries([pcd])
pcd.estimate_normals()
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)
o3d.visualization.draw_geometries([p_mesh_crop])
o3d.io.write_triangle_mesh("tt.ply",p_mesh_crop)

