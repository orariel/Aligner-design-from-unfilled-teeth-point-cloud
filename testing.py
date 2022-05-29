import numpy as np
import surf2stl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib import cm
import pyvista as pv
import for_or_order_list

#Create the profile
Radii = [0.2, 1, 1.2, 1.4,1.6, 1.8, 2]
Zradii=[-2.0,5, 8, 11, 14,17, 20]

radius = CubicSpline(Zradii, Radii, bc_type=((1, 0.5), (1, 0.0)))

# Make data
thetarange = np.linspace(0, 2*np.pi, 50)
zrange = np.linspace(min(Zradii), max(Zradii),50)
X = [radius(z)*np.cos(thetarange) for z in zrange]
Y = [radius(z)*np.sin(thetarange) for z in zrange]
Z = np.array([[z] for z in zrange])

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(0, 20)
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

#Plot the circles
for zz in Zradii:
    XX = radius(zz)*np.cos(thetarange)
    YY = radius(zz)*np.sin(thetarange)
    # ax.plot(XX,YY,zz, lw=1, color='k')

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim3d(-2, 2)
# ax.set_ylim3d(-2, 2)
# ax.set_zlim3d(0, 20)
# ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
#

#
# pcd=o3d.geometry.PointCloud()
# pcd.points=o3d.utility.Vector3dVector(arr_xyz)
tooth=pv.read("C:/Users/OR\PycharmProjects/Aligner design from unfilled teeth point cloud/cube_dent_3_0.stl")

tooth=tooth.subdivide(3,subfilter='loop', inplace=True, progress_bar=False)
tooth_b=tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
tooth_b.plot()
side_1=tooth_b.clip(normal='x', origin=tooth_b.center, invert=True, value=0)

side_2=tooth_b.clip(normal='-x', origin=tooth_b.center, invert=True, value=0)
p__=pv.Plotter()
# p__.add_mesh(side_1)
# p__.add_mesh(side_2,color="red")
# p__.show()

arr_b_1=np.asarray(side_2.points)
order_list = for_or_order_list.get_sorted_arr(arr_b_1, 0)
order_list=np.asarray(order_list)
arr_sorted=[]
points_to_add=[]
for i in range(order_list.size):
    arr_sorted.append(arr_b_1[order_list[i]])
arr_sorted=np.asarray(arr_sorted)
pcd_test=pv.PolyData(arr_sorted)
pcd_test.plot()
for i in range(order_list.size - 1):
    line = pv.Line(pointa=arr_sorted[i], pointb=arr_sorted[i + 1], resolution=10)
    points_to_add.append(arr_sorted[i])
    for j in range(10):
        points_to_add.append(line.points[j])
    # points_to_add.append(arr_sorted[i + 1])
points_to_add=np.concatenate(points_to_add).reshape(len(points_to_add), 3)
pcd_test=pv.PolyData(points_to_add)
pcd_test.plot()
# # #<--------------------------------------------------SORTING SIDE B------------------------------------------------->
# # arr_b_2=np.asarray(side_2.points)
# # max_index=np.hsplit(arr_b_2,3)[2].max()
# # max_index=np.argmax(max_index)
# # order_list_2 = for_or_order_list.get_sorted_arr(arr_b_2,0)
# # order_list_2=np.asarray(order_list_2)
# # arr_sorted_2=[]
# # points_to_add_2=[]
# # for i in range(order_list_2.size):
# #     arr_sorted_2.append(arr_b_2[order_list_2[i]])
# # arr_sorted_2=np.asarray(arr_sorted_2)
# # for i in range(order_list_2.size - 1):
# #     line = pv.Line(pointa=arr_sorted_2[i], pointb=arr_sorted_2[i + 1], resolution=10)
# #     points_to_add_2.append(arr_sorted_2[i])
# #     for j in range(10):
# #         points_to_add_2.append(line.points[j])
# #     points_to_add_2.append(arr_sorted_2[i + 1])
# # points_to_add_2=np.concatenate(points_to_add_2).reshape(len(points_to_add_2), 3)
# # pcd_test_2=pv.PolyData(points_to_add_2[0:1900,:])
# # p__=pv.Plotter()
# # p__.add_mesh(pcd_test)
# # p__.add_mesh(pcd_test_2,color="red")
# # p__.show()
# #
# #
#
#
#
# # create x,y,z data for 3d surface plot
# # x = np.linspace(-6, 6, 30)
# # y = np.linspace(-6, 6, 30)
# # X, Y = np.meshgrid(x, y)
# # Z = np.sqrt(X ** 4 + Y ** 4)
# # # draw surface plot
# # ax = plt.axes(projection='3d')
# #
# # # export surface to a stl format file
# # surf2stl.write('3d-sinusoidal.stl', X, Y, Z)