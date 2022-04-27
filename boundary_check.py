import numpy as np
import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt


tooth_number=6
file_name="tooth_"+str(tooth_number)+"_decimate.stl"
tooth = pv.read(file_name)
tooth_points = np.asarray(tooth.points)
tooth_normals_b = np.asarray(tooth.point_normals)
arr_index = []


