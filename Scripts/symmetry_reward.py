import numpy as np
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os, sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from Configuration.Config import texpref_config, texgen_config, common_config

# ===============================
# 1. Mesh Loading and 3D PCA Functions
# ===============================
def load_mesh(mesh_path):
    """Load a 3D mesh from file using trimesh."""
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh

def compute_3d_pca(vertices):
    """
    Computes PCA on a set of 3D vertices.

    Returns:
        centroid (np.ndarray): Mean of the vertices.
        eigenvalues (np.ndarray): Sorted eigenvalues (ascending).
        eigenvectors (np.ndarray): Corresponding eigenvectors (as columns).
    """
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return centroid, eigenvalues, eigenvectors

def compute_symmetry_features(centroid, eigenvalues, eigenvectors):
    """
    Computes symmetry features in 3D:
      - Uses the smallest eigenvector as the symmetry plane normal.
      - Projects the dominant eigenvector (largest eigenvalue) onto the symmetry plane
        to yield a symmetry line.
      
    Returns:
        line_point (np.ndarray): A point on the symmetry line (the centroid).
        line_direction (np.ndarray): Unit vector for the symmetry line.
        plane_normal (np.ndarray): Estimated symmetry plane normal.
    """
    plane_normal = eigenvectors[:, 0]
    dominant = eigenvectors[:, -1]
    proj = dominant - np.dot(dominant, plane_normal) * plane_normal
    norm_proj = np.linalg.norm(proj)
    line_direction = proj / norm_proj if norm_proj > 0 else dominant
    return centroid, line_direction, plane_normal

# ===============================
# 2. UV Mapping Functions (Using the Mesh UVs)
# ===============================
def barycentric_coords(P, A, B, C):
    """
    Computes barycentric coordinates of point P with respect to triangle ABC.
    Returns an array of weights that sum to 1.
    """
    v0 = B - A
    v1 = C - A
    v2 = P - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if np.abs(denom) < 1e-8:
        return np.array([1/3, 1/3, 1/3])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])

def project_point_to_uv(mesh, point):
    """
    Projects a 3D point to UV space using the mesh's UV mapping.
    Finds the closest surface point and uses barycentrics to interpolate the face's UVs.
    """
    closest_data = mesh.nearest.on_surface([point])
    closest_point = closest_data[0][0]
    face_idx = closest_data[2][0]
    face = mesh.faces[face_idx]
    verts = mesh.vertices[face]
    face_uv = mesh.visual.uv[face]
    bary = barycentric_coords(closest_point, verts[0], verts[1], verts[2])
    uv = bary[0] * face_uv[0] + bary[1] * face_uv[1] + bary[2] * face_uv[2]
    return uv

# ===============================
# 3. Mirror Computation in 3D and UV Correspondence
# ===============================
def reflect_vertex(vertex, centroid, symmetry_plane_normal):
    """
    Reflects a 3D point "vertex" across the plane defined by point
    "centroid" and unit normal "symmetry_plane_normal".
    """
    diff = vertex - centroid
    return vertex - 2 * np.dot(diff, symmetry_plane_normal) * symmetry_plane_normal


def compute_vertex_mirror_pairs(mesh, c, n, visualize=False, delay=0.1):
    """
    Computes UV pairs for only the vertices on one side of the symmetry plane.

    Args:
        mesh (trimesh.Trimesh): The mesh with mesh.visual.uv present.
        c (np.ndarray): A point on the symmetry plane (3,).
        n (np.ndarray): Unit normal of the symmetry plane (3,).
        visualize (bool): If True, shows each original/mirror UV pair.
        delay (float): Pause between visualized steps (seconds).

    Returns:
        uv_orig (np.ndarray): UVs for vertices on the chosen half (M,2).
        uv_mirror (np.ndarray): UVs for their mirrors (M,2).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    verts = mesh.vertices           # (N,3)
    uvs_all = np.array(mesh.visual.uv)  # (N,2)

    # 1) signed distances to plane
    diffs = verts - c[None, :]      # (N,3)
    dists = diffs.dot(n)            # (N,)

    # 2) pick only one half
    mask = dists > 0                # choose positive side; flip sign for the other side
    idxs = np.nonzero(mask)[0]      # indices of vertices we'll process

    uv_orig = uvs_all[mask]         # (M,2)
    uv_mirror = []

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
        ax.set_xlabel("U"); ax.set_ylabel("V")

    # 3) for each selected vertex compute mirror
    for i in idxs:
        P = verts[i]
        P_mirror = reflect_vertex(P, c, n)
        uv_m = project_point_to_uv(mesh, P_mirror)
        uv_mirror.append(uv_m)

        if visualize:
            ax.clear()
            ax.scatter(*uvs_all[i],  color='gray', label='Original UV')
            ax.scatter(*uv_m,       color='blue', label='Mirror UV')
            ax.legend()
            plt.draw()
            time.sleep(delay)

    if visualize:
        plt.ioff()
        plt.show()

    uv_mirror = np.vstack(uv_mirror)  # (M,2)
    return uv_orig, uv_mirror


# ===============================
# 4. PyVista Helpers for 3D Visualization
# ===============================
def trimesh_to_pyvista(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    nfaces = faces.shape[0]
    faces_pv = np.hstack([np.full((nfaces, 1), 3), faces]).astype(np.int64)
    return pv.PolyData(vertices, faces_pv)

def create_symmetry_plane_patch(centroid, plane_normal, size):
    arbitrary = np.array([1, 0, 0])
    if np.allclose(np.abs(np.dot(arbitrary, plane_normal)), 1.0):
        arbitrary = np.array([0, 1, 0])
    in_plane1 = np.cross(plane_normal, arbitrary)
    in_plane1 /= np.linalg.norm(in_plane1)
    in_plane2 = np.cross(plane_normal, in_plane1)
    in_plane2 /= np.linalg.norm(in_plane2)
    
    corners = []
    for dx in [-size, size]:
        for dy in [-size, size]:
            pt = centroid + dx * in_plane1 + dy * in_plane2
            corners.append(pt)
    corners = np.array(corners)
    faces = np.hstack([[4, 0, 1, 3, 2]])
    plane_patch = pv.PolyData(corners, faces)
    return plane_patch

# ===============================
# 5. PyVista 3D Visualization Routine
# ===============================
def visualize_3d(mesh_pv, line_point, line_direction, plane_normal):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_pv, color='white')
    
    bounds = mesh_pv.bounds
    max_extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    t_val = max_extent * 1.5
    
    line_start = line_point - t_val * line_direction
    line_end = line_point + t_val * line_direction
    sym_line = pv.Line(line_start, line_end, resolution=200)
    plotter.add_mesh(sym_line, color='red', line_width=4, label="Symmetry Line")
    
    normal_start = line_point - t_val * plane_normal
    normal_end = line_point + t_val * plane_normal
    sym_normal = pv.Line(normal_start, normal_end, resolution=50)
    plotter.add_mesh(sym_normal, color='green', line_width=2, label="Symmetry Normal")
    
    plane_patch = create_symmetry_plane_patch(line_point, plane_normal, size=t_val/2)
    plotter.add_mesh(plane_patch, color='blue', opacity=0.3, label="Symmetry Plane")
    
    plotter.add_legend()
    plotter.show_grid()
    plotter.show()



def load_intrinsic_symmetry_uv_pairs(filepath):
    """
    Load UV symmetry pairs from a text file.
    Returns two arrays of shape (N, 2):
        - uv_orig: the original UV coordinates
        - uv_mirror: the corresponding mirrored UV coordinates
    """
    uv_orig = []
    uv_mirror = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip comments or empty lines
            tokens = line.split()
            if len(tokens) != 4:
                continue  # skip malformed lines

            u1, v1, u2, v2 = map(float, tokens)
            uv_orig.append([u1, v1])
            uv_mirror.append([u2, v2])

    return np.array(uv_orig), np.array(uv_mirror)



# Example usage:
# filepath = "C:/Users/User/Desktop/Differentiable-Texture-Learning/Assets/sym_pairs.txt"
# uv_orig, uv_mirror = load_uv_symmetry_pairs(filepath)
# print(uv_orig.shape, uv_mirror.shape)

def visualize_3d_with_texture(mesh_pv, line_point, line_direction, plane_normal, texture_path=None):
    plotter = pv.Plotter()
    
    # Load and apply texture if provided
    if texture_path is not None:
        texture = pv.read_texture(texture_path)
        plotter.add_mesh(mesh_pv, texture=texture)
    else:
        plotter.add_mesh(mesh_pv, color='white')

    # Determine length of visual elements
    bounds = mesh_pv.bounds
    max_extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    t_val = max_extent * 1.2

    # Symmetry line
    line_start = line_point - t_val * line_direction
    line_end = line_point + t_val * line_direction
    sym_line = pv.Line(line_start, line_end, resolution=200)
    # plotter.add_mesh(sym_line, color='red', line_width=4, label="Symmetry Line")

    # Normal vector
    normal_start = line_point - t_val * plane_normal
    normal_end = line_point + t_val * plane_normal
    sym_normal = pv.Line(normal_start, normal_end, resolution=50)
    # plotter.add_mesh(sym_normal, color='green', line_width=2, label="Symmetry Normal")

    # Symmetry plane patch (you need to have this function defined)
    plane_patch = create_symmetry_plane_patch(line_point, plane_normal, size=t_val/2)
    plotter.add_mesh(plane_patch, color='blue', opacity=0.3, label="Symmetry Plane")

    # Display
    plotter.add_legend()
    # plotter.show_grid()
    plotter.show()

# ===============================
# 6. Precompute Pixel Coordinates from UVs
# ===============================
def uv_to_pixel_coords(uv, texture_shape):
    """
    Converts UV coordinates in [0,1] to pixel coordinates for a texture image.
    
    Args:
        uv (np.ndarray): Array of UV coordinates (N,2) in [0,1].
        texture_shape (tuple): (H, W) of the texture image.
        
    Returns:
        torch.Tensor: Pixel coordinates (N,2) as floats.
    """
    H, W = texture_shape
    x = uv[:,0] * (W - 1)
    y = uv[:,1] * (H - 1)
    return torch.tensor(np.stack([x, y], axis=-1), dtype=torch.float32)

# ===============================
# 7. Differentiable Sampling Using grid_sample (with precomputed locations)
# ===============================
def sample_texture_at_uv(T, uv_coords):
    """
    Samples texture T at specific UV positions using F.grid_sample.
    
    Args:
        T (torch.Tensor): Texture image, shape [B, C, H, W].
        uv_coords (torch.Tensor): UV coordinates (N,2) in [0,1].
        
    Returns:
        torch.Tensor: Sampled pixel values, shape [B, C, N].
    """
    # Convert UV coordinates from [0,1] to normalized coordinates [-1, 1]
    # Why do we convert UV coordinates from [0, 1] to [-1, 1]?
    # It's specifically because of how "torch.nn.functional.grid_sample" works.
    # "grid_sample" expects the sampling grid coordinates to be in the normalized
    # coordinate system of [-1, 1], not the texture image pixel space or UV space.

    B, C, H, W = T.shape
    uv_norm = (uv_coords * 2) - 1  # [N,2]
    # grid_sample expects grid of shape [B, N, 1, 2]
    grid = uv_norm.view(1, -1, 1, 2).expand(B, -1, -1, -1)
    samples = F.grid_sample(T, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return samples.squeeze(-1)  # shape [B, C, N]


def sample_texture_at_uv(texture_image, uv_coords, patch_size):
    """
    Samples texture T at specific UV positions using F.grid_sample.
    Supports patch-based sampling for symmetry-aware rewards.

    Args:
        T (torch.Tensor): Texture image, shape [B, C, H, W].
        uv_coords (torch.Tensor): UV coordinates (N, 2) in [0,1].
        patch_size (int): Size of square patch to sample (default = 1 for single pixel).

    Returns:
        torch.Tensor: 
            If patch_size == 1: shape [B, C, N].
            If patch_size > 1: shape [B, C, N, patch_size, patch_size].
    """
    B, C, H, W = texture_image.shape
    N = uv_coords.shape[0]
    
    if patch_size > 1: # Offsets for local patch sampling
        offset = torch.linspace(-(patch_size // 2), patch_size // 2, patch_size, device=uv_coords.device)
        dx, dy = torch.meshgrid(offset, offset, indexing='ij')
        dx = dx.flatten() / (W - 1)
        dy = dy.flatten() / (H - 1)
        offset_grid = torch.stack([dx, dy], dim=1)  # (patch_size^2, 2)

        uv_offsets = uv_coords[:, None, :] + offset_grid[None, :, :]  # (N, patch_size^2, 2)
        uv_offsets = uv_offsets.clamp(0, 1)  # stay within valid range
        # Convert UV coordinates from [0,1] to normalized coordinates [-1, 1]
        # Why do we convert UV coordinates from [0, 1] to [-1, 1]?
        # It's specifically because of how "torch.nn.functional.grid_sample" works.
        # "grid_sample" expects the sampling grid coordinates to be in the normalized
        # coordinate system of [-1, 1], not the texture image pixel space or UV space.
        uv_norm = uv_offsets * 2 - 1  # normalize to [-1, 1]
        uv_grid = uv_norm.view(1, N * patch_size * patch_size, 1, 2).expand(B, -1, -1, -1)

        # Resample
        samples = F.grid_sample(texture_image, uv_grid, mode='bilinear', padding_mode='border', align_corners=True)
        samples = samples.view(B, C, N, patch_size, patch_size)  # (B, C, N, p, p)
    else: # Single point sampling
        # Convert UV coordinates from [0,1] to normalized coordinates [-1, 1]
        # Why do we convert UV coordinates from [0, 1] to [-1, 1]?
        # It's specifically because of how "torch.nn.functional.grid_sample" works.
        # "grid_sample" expects the sampling grid coordinates to be in the normalized
        # coordinate system of [-1, 1], not the texture image pixel space or UV space.
        uv_norm = (uv_coords * 2) - 1  # [N, 2]
        grid = uv_norm.view(1, N, 1, 2).expand(B, -1, -1, -1)
        samples = F.grid_sample(texture_image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        samples = samples.squeeze(-1)  # (B, C, N)

    return samples

# ===============================
# 8. Differentiable Symmetry Reward Function Based on Vertex Mirror Correspondence
# ===============================
def symmetry_vertex_loss(texture_image, pixel_uv_orig, pixel_uv_mirror, patch_size):
    """
    Computes the symmetry loss for generated texture T by comparing RGB values
    sampled at UV coordinates corresponding to original vertices and their mirror.
    
    Args:
        T (torch.Tensor): Generated texture image, shape [B, C, H, W].
        pixel_uv_orig (torch.Tensor): UV coordinates for original vertices, shape [N,2] in [0,1].
        pixel_uv_mirror (torch.Tensor): UV coordinates for mirror vertices, shape [N,2] in [0,1].
        
    Returns:
        torch.Tensor: Scalar loss value.
    """
    samples_orig = sample_texture_at_uv(texture_image, pixel_uv_orig, patch_size)   # shape [B, C, N]
    samples_mirror = sample_texture_at_uv(texture_image, pixel_uv_mirror, patch_size) # shape [B, C, N]
    loss = F.mse_loss(samples_orig, samples_mirror)
    return loss

# ===============================
# 9. Texture Image Loader
# ===============================
def load_texture_image(image_path, texture_size, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((texture_size, texture_size)),
        transforms.ToTensor(),  # Converts to [0,1]
    ])
    texture_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
    return texture_tensor

# ===============================
# 10. Main Routine
# ===============================
if __name__ == "__main__":
    device = 'cuda'  # or 'cpu'
    
    # --- Load Mesh ---
    mesh_path = texgen_config["mesh"]  # Adjust as needed.
    mesh = load_mesh(mesh_path)
    vertices = mesh.vertices
    print(len(vertices))

    # --- Compute 3D PCA and symmetry features ---
    centroid, eigenvalues, eigenvectors = compute_3d_pca(vertices)
    line_point, line_direction, plane_normal = compute_symmetry_features(centroid, eigenvalues, eigenvectors)
    print("3D symmetry line passes through:", line_point)
    print("3D symmetry line direction:", line_direction)
    print("Symmetry plane normal:", plane_normal)

    # --- 3D Visualization ---
    mesh_pv = trimesh_to_pyvista(mesh)
    visualize_3d(mesh_pv, line_point, line_direction, plane_normal)

    # --- Use the mesh's UV mapping (must be present) ---
    if not hasattr(mesh.visual, "uv"):
        raise ValueError("Mesh does not have UV coordinates!")
    uvs = np.array(mesh.visual.uv)  # shape (N,2) in [0,1]
    
    # --- Compute mirror UV correspondences ---
    uv_orig, uv_mirror = compute_vertex_mirror_pairs(mesh, centroid, plane_normal)

    print("uv_orig: ", uv_orig.shape)
    print("uv_mirror: ", uv_mirror.shape)
    
    # Visualize the UV correspondences for inspection.
    plt.figure(figsize=(6,6))
    plt.scatter(uv_orig[:,0], uv_orig[:,1], s=10, c='gray', alpha=0.7, label="Original UVs")
    plt.scatter(uv_mirror[:,0], uv_mirror[:,1], s=10, c='blue', alpha=0.7, label="Mirror UVs")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.title("Original vs. Mirror UV Coordinates")
    plt.axis('equal')
    plt.legend()
    plt.show()
    
    # --- Precompute UV Coordinates as Tensors (for grid_sample) ---
    # These remain fixed across iterations.
    pixel_uv_orig = torch.tensor(uv_orig, dtype=torch.float32, device=device)  # shape (N,2), in [0,1]
    pixel_uv_mirror = torch.tensor(uv_mirror, dtype=torch.float32, device=device)  # shape (N,2), in [0,1]
    
    # --- Load Real Generated Texture ---
    texture_res = 1024
    texture_image_path = r"C:\Users\User\Desktop\Texture-DPO\Scripts\balloon_albedo.png"
    generated_texture = load_texture_image(texture_image_path, texture_size=texture_res, device=device)

    # --- Compute Symmetry Loss ---
    loss = symmetry_vertex_loss(generated_texture, pixel_uv_orig, pixel_uv_mirror, patch_size=1)
    print("Symmetry Vertex Loss:", loss.item())

    # In your training loop, you would use the same fixed pixel_uv_orig and pixel_uv_mirror to sample from the generated texture
    # and add the resulting loss to your overall objective.
