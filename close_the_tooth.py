import math

import alphashape as al
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv

root = pv.Circle(1, resolution=200)
arr_root = np.asarray(root.points)

x_c = []
y_c = []
beta = np.linspace(0, 2 * np.pi, num=10000, endpoint=False)
for i in range(beta.size):
   p = np.random.random(1)
   x_c.append(1 * p * math.cos(beta[i]))
   y_c.append(1 * p * math.sin(beta[i]))

x_c = np.asarray(x_c)
y_c = np.asarray(y_c)
z_c = np.zeros((len(x_c), 1))
xyz_to_add_c = np.concatenate((x_c, y_c, z_c), axis=1)
arr_root = np.concatenate((xyz_to_add_c, arr_root), axis=0)

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
xy_cor_source = np.concatenate((x_cor, y_cor), axis=1)
# xyz_cor_source = clean_array(xyz_cor_source)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_cor_source)
pcd.estimate_normals()

# # -------------------------------slicing arry and making list of teeth-(each element contain np.arry of tooth)------------------------#
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
arr_tooth = np.asarray(list_tooth[5])
p = np.hsplit(arr_tooth, 3)
xy_tooth = np.concatenate((p[0], p[1]), axis=1)
pcd_tooth = o3d.geometry.PointCloud()
pcd_tooth.points = o3d.utility.Vector3dVector(arr_tooth)
pcd_tooth.estimate_normals()

alpha_shape = al.alphashape(xy_tooth, 2.0)
bounds = np.asarray(alpha_shape.boundary)

bbox = pcd_tooth.get_axis_aligned_bounding_box().get_min_bound()
bbox_center=np.asarray(pcd_tooth.get_axis_aligned_bounding_box().get_center())
z = bbox[2]

z_vec = np.repeat(z, bounds[:, 0].size, axis=0)
z_vec = z_vec.reshape((bounds[:, 0].size, 1))

# y_f=interp1d(bounds[:,0],bounds[:,1],'linear')
# x=np.linspace(6,8,100)
# y=y_f(x)

#
#
# plt.plot(bounds[6:288,0],bounds[6:288,1],'b+')
#
# plt.show()
# plt.plot(bounds[1:50,0],bounds[1:50,1],'r+')
x = []
y = []
threshold = 0.001
i = 0
j = 0
max_ = 0.1
x = []
y = []
j = 0

for i in range(bounds[:, 0].size - 1):
   x_1 = bounds[i, 0]
   y_1 = bounds[i, 1]
   x_2 = bounds[i + 1, 0]
   y_2 = bounds[i + 1, 1]
   x_new = x_1 + (x_2 - x_1) / 2
   y_new = y_1 + (y_2 - y_1) / 2
   if (abs(x_new - x_1) / 2 > threshold):
      xy_new = np.array([x_new, y_new])
      x_new_2 = x_1 + (x_new - x_1) / 2
      y_new_2 = y_1 + (y_new - y_1) / 2
      x_new_3 = x_new + (x_new_2 - x_new) / 2
      y_new_3 = y_new + (y_new_2 - y_new) / 2
      x_new_4 = x_new + (x_2 - x_new) / 2
      y_new_4 = y_new + (y_2 - y_new) / 2
      x_new_5 = x_new_2 + (x_2 - x_new_2) / 2
      y_new_5 = y_new_2 + (y_2 - y_new_2) / 2
      x.append(x_new)
      y.append(y_new)
      x.append(x_new_2)
      y.append(y_new_2)
      x.append(x_new_3)
      y.append(y_new_3)
      x.append(x_new_4)
      y.append(y_new_4)
      x.append(x_new_5)
      y.append(y_new_5)
x = np.array(x).reshape((len(x), 1))
y = np.array(y).reshape((len(x), 1))
xy_to_add = np.concatenate((x, y), axis=1)

plt.plot(x, y, 'r+')
plt.plot(bounds[:, 0], bounds[:, 1], 'b+')
# plt.show()

pcd_border = o3d.geometry.PointCloud()

z = bbox[2]
bounds = np.concatenate((bounds, xy_to_add), axis=0)
z_vec = np.repeat(z, bounds[:, 0].size, axis=0)
z_vec = z_vec.reshape((bounds[:, 0].size, 1))

xyz_bounds = np.concatenate((bounds, z_vec), axis=1)

pcd_border.points = o3d.utility.Vector3dVector(xyz_bounds)
pcd_border.estimate_normals()
#
pcd_root = o3d.geometry.PointCloud()
pcd_root.points = o3d.utility.Vector3dVector(arr_root)
pcd_root.translate(bbox_center)
arr_root=np.asarray(pcd_root.points)
zz=arr_tooth.min()/4
pcd_root.translate([0,0,zz])
# R =pcd_root.get_rotation_matrix_from_axis_angle((0,np.radians(10), 0))
# pcd_root= pcd_root.rotate(R)
total_arr=np.concatenate((arr_tooth,arr_root),axis=0)
pcd_total=o3d.geometry.PointCloud()
pcd_total.points=o3d.utility.Vector3dVector(total_arr)
pcd_total.estimate_normals()
o3d.visualization.draw_geometries([pcd_total])
# o3d.io.write_triangle_mesh("bpa_mesh.ply",p_mesh_crop)

density = 2.00008  # 0.08
tengent_plane =3  # 100
depth_mesh = 9 # 9
pcd_tooth.orient_normals_consistent_tangent_plane(tengent_plane)
p, t = pcd_tooth.compute_convex_hull()
p.orient_triangles()
p.compute_vertex_normals()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_tooth, depth=depth_mesh,linear_fit=False)
o3d.visualization.draw_geometries([mesh])
bbox = pcd_tooth.get_axis_aligned_bounding_box()
mesh= mesh.crop(bbox)
o3d.io.write_triangle_mesh('poasso.ply',mesh)