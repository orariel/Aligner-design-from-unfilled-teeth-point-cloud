import open3d as o3d

mesh=o3d.io.read_triangle_mesh('tooth_6_decimate.stl')
aabb = mesh.get_axis_aligned_bounding_box()

obb =mesh.get_oriented_bounding_box()
obb.color=(1,0,0)
# boundary=tooth.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
# boundary = boundary.compute_normals()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh,obb,aabb])