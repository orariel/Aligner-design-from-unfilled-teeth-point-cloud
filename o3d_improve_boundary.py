import numpy as np
import pyvista as pv
import pymeshfix
import pymeshfix as mf
import open3d as o3d
from math import sqrt

mesh_color=o3d.io.read_triangle_mesh("final_pred_1.ply")
mesh_color.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_color])
pcd =mesh_color.sample_points_uniformly(number_of_points=70000)
pcd.estimate_normals()
o3d.visualization.draw_geometries([pcd])