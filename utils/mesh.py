import open3d as o3d
import numpy as np
import math
import torch
import potpourri3d as pp3d

def visu_pts(surf, idx_points, colors):
    solver = pp3d.MeshHeatMethodDistanceSolver(surf.vertices, surf.faces)
    color_array = np.zeros(surf.vertices.shape)
    for i in range(len(idx_points)):
        i_v = idx_points[i]
        dist = solver.compute_distance(i_v)*2
        color_array += np.clip(1-dist, 0, np.inf)[:, None]*colors[i][None, :]
    color_array = np.clip(color_array, 0, 255.)
    return color_array

def load_mesh(filepath, scale=True, return_vnormals=False):


    mesh = o3d.io.read_triangle_mesh(filepath)

    tmat = np.identity(4, dtype=np.float32)
    center = mesh.get_center()
    tmat[:3, 3] = -center
    if scale:
        smat = np.identity(4, dtype=np.float32)
        area = mesh.get_surface_area()
        smat[:3, :3] = np.identity(3, dtype=np.float32) / math.sqrt(area)
        tmat = smat @ tmat
    mesh.transform(tmat)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if return_vnormals:
        mesh.compute_vertex_normals()
        vnormals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if scale:
            return vertices, faces, vnormals, area, center
        return vertices, faces, vnormals
    else:
        return vertices, faces

def getMeshFromData(mesh, Rho=None, color=None):
    """
    Performs midpoint subdivision. Order determines the number of iterations
    """
    V = mesh[0]
    F = mesh[1]
    # mesh=o3d.geometry.TriangleMesh(o3d.cpu.pybind.utility.Vector3dVector(V),o3d.cpu.pybind.utility.Vector3iVector(F))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))

    if Rho is not None:
        Rho = np.squeeze(Rho)
        col = np.stack((Rho, Rho, Rho))
        mesh.vertex_colors = o3d.utility.Vector3dVector(col.T)

    if color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    return mesh


def decimate_mesh(V,F,target):
    """
    Decimates mesh given by V,F to have number of faces approximatelyu equal to target
    """
    mesh=getMeshFromData([V,F])
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh=mesh.simplify_quadric_decimation(target)
    VS = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    FS = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    return VS, FS

def opt_rot(surf_1, surf_2):
    areas = np.linalg.norm(surf_1.surfel, axis=1)
    q_1 = surf_1.surfel / (1e-8 + np.linalg.norm(surf_1.surfel, axis=1, keepdims=True))
    q_2 = surf_2.surfel / (1e-8 + np.linalg.norm(surf_2.surfel, axis=1, keepdims=True))
    to_sum = q_1[:, :, np.newaxis] * q_2[:, np.newaxis, :]
    A = (to_sum * areas[:, np.newaxis, np.newaxis]).sum(axis=0)
    u, _, v = np.linalg.svd(A)
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.sign(np.linalg.det(A))]])
    O = u @ a @ v
    print(u, v, u@v, A)
    new_vertices = np.dot(surf_2.vertices, O.T)
    return O.T

def opt_rot_points(pts_1, pts_2):
    center_1 = pts_1.mean(axis=0)
    pts_c1 = pts_1 - center_1
    center_2 = pts_2.mean(axis=0)
    pts_c2 = pts_2 - center_2 
    to_sum = pts_c1[:, :, np.newaxis] * pts_c2[:, np.newaxis, :]
    A = np.dot(pts_c1.T, pts_c2)
    #A = to_sum.sum(axis=0)
    u, _, v = np.linalg.svd(A)
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.sign(np.linalg.det(A))]])
    O = u @ a @ v
    return O.T


def opt_rot_points_torch(pts_1, pts_2, device="cuda"):
    center_1 = pts_1.mean(dim=0)
    pts_c1 = pts_1 - center_1
    center_2 = pts_2.mean(dim=0)
    pts_c2 = pts_2 - center_2 
    to_sum = pts_c1[:, :, None] * pts_c2[:, None, :]
    A = pts_c1.T @ pts_c2
    #A = to_sum.sum(axis=0)
    u, _, v = torch.linalg.svd(A)
    a = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, torch.sign(torch.linalg.det(A))]]).float().to(device)
    O = u @ a @ v
    return O.T