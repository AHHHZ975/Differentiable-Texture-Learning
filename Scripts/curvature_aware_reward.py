import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import igl  # For curvature computation
import nvdiffrast.torch as dr
from mesh import Mesh
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
import kiui

script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))

from Configuration.Config import texgen_config


class CurvatureReward(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.mesh = None

        # Background color
        self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.bg_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        if not texgen_config["gui"] or os.name == 'nt':
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

    @torch.no_grad()
    def load_mesh(self, path):
        self.mesh = Mesh.load(path, 
                              front_dir=texgen_config["front_dir"],
                              retex=texgen_config["retex"],
                              remesh=texgen_config["remesh"],
                              device=self.device)
        
        # Extract vertices and faces
        vertices = self.mesh.v.cpu().numpy() # Vertices of the mesh
        faces = self.mesh.f.cpu().numpy()    # Triangles (facet) of the mesh 
        uv_coords = self.mesh.vt.cpu().numpy() if self.mesh.vt is not None else None # 2D coordinates in the image space correponding to each vertex
        uv_faces = self.mesh.ft.cpu().numpy() if self.mesh.ft is not None else None  # 2D coordinates in the image space correponding to each facet

        return vertices, faces, uv_coords, uv_faces


    @torch.no_grad()
    def export_mesh(self, path):
        """
        Export the loaded mesh normally (without modifying vertex colors).
        """
        self.mesh.write_obj(path)

    def export_mesh_with_curvature_colors(self, path):
        """
        Computes curvature-based vertex colors, bakes them into a texture, and exports the mesh.
        """
        if self.mesh is None:
            print("[ERROR] No mesh loaded!")
            return

        # Extract vertices and faces
        vertices = self.mesh.v.cpu().numpy() # Vertices of the mesh
        faces = self.mesh.f.cpu().numpy()    # Triangles (facet) of the mesh 
        uv_coords = self.mesh.vt.cpu().numpy() if self.mesh.vt is not None else None # 2D coordinates in the image space correponding to each vertex
        uv_faces = self.mesh.ft.cpu().numpy() if self.mesh.ft is not None else None  # 2D coordinates in the image space correponding to each facet

        # Compute principal curvature using libigl
        v1, v2, k1, k2 = igl.principal_curvature(vertices, faces)
        h2 = 0.5 * (k1 + k2)  # Mean curvature

        # Define threshold and min/max values
        threshold = 1.9
        h_min = h2[h2 < threshold].min() if np.any(h2 < threshold) else threshold
        h_max = h2[h2 >= threshold].max() if np.any(h2 >= threshold) else threshold

        # Define gradient colors for the curvature range
        color_low_start = np.array([0.0, 0.0, 1.0])  # dark blue        
        color_low_end = np.array([0.4, 0.4, 1.0])    # light blue
        color_high_start = np.array([1.0, 0.4, 0.4])  # light red    
        color_high_end = np.array([1.0, 0.0, 0.0])   # pure red

        # Compute interpolation for the color mapping
        t_low = (h2 - h_min) / (threshold - h_min)  # for h2 < threshold
        t_high = (h2 - threshold) / (h_max - threshold)  # for h2 >= threshold
        t = np.where(h2 < threshold, t_low, t_high)

        # Interpolate colors
        colors = np.where(
            h2[:, None] < threshold,
            (1 - t[:, None]) * color_low_start + t[:, None] * color_low_end,
            (1 - t[:, None]) * color_high_start + t[:, None] * color_high_end
        )

        # Bake colors into a texture
        texture_resolution = 1024  # Adjust as needed
        baked_texture = self.bake_texture(uv_coords, faces, colors, uv_faces, texture_resolution)

        # Assign the generated texture to the mesh
        self.mesh.albedo = torch.tensor(baked_texture, dtype=torch.float32, device=self.device)

        # Export the mesh using `write_obj`
        self.mesh.write_obj(path)
        print(f"[INFO] Mesh with curvature-based colors and texture saved to '{path}'")

    def export_mesh_with_curvature_directions(self, path, step=10, texture_resolution=1024):
        """
        Uses the 2D curvature_map tensor to bake a heatmap+quiver texture
        and exports the mesh with that texture.

        Args:
            path: output .obj path
            curvature_map: torch.Tensor [1,1,H,W] normalized [0,1]
            step: quiver sampling step
            texture_resolution: size of texture
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import torch.nn.functional as F

        # Extract vertices and faces
        mesh_vertices, mesh_faces, uv_coords, uv_faces = self.load_mesh(texgen_config["mesh"])

        # Compute the mean curvature for all vertices of the input mesh object
        mean_curvature_3d = self.compute_vertex_curvature(mesh_vertices, mesh_faces)
        H = W = int(texgen_config["texture_size"])
        curvature_map = self.build_curvature_texture(uv_coords, uv_faces, mesh_faces, mean_curvature_3d, H, W)


        if self.mesh is None:
            print("[ERROR] No mesh loaded!")
            return

        # Prepare curvature_map for NumPy extraction and Sobel conv
        if curvature_map.ndim == 2:            
            curv_np = curvature_map.cpu().numpy()
            curv_t  = curvature_map.unsqueeze(0).unsqueeze(0).to(curvature_map.device)  # [1,1,H,W]
        elif curvature_map.ndim == 4:            
            curv_t  = curvature_map
            curv_np = curvature_map[0,0].cpu().numpy()
        else:            
            # assume [1,H,W]
            curv_np = curvature_map[0].cpu().numpy()
            curv_t  = curvature_map.unsqueeze(1)

        
        H, W = curv_np.shape



        # Flip vertically so top of map aligns with top of texture
        curv_np = np.flipud(curv_np)

        # Compute Sobel gradients on GPU
        device = curv_t.device
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device).view(1,1,3,3)/8.0
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device).view(1,1,3,3)/8.0
        with torch.no_grad():
            gx_t = F.conv2d(curv_t, sobel_x, padding=1)
            gy_t = F.conv2d(curv_t, sobel_y, padding=1)
        grad_x = gx_t[0,0].cpu().numpy()
        grad_y = gy_t[0,0].cpu().numpy()

        # Flip gradients to match flipped heatmap
        grad_x = np.flipud(grad_x)
        grad_y = np.flipud(grad_y)

        # Normalize vector field
        mag = np.sqrt(grad_x*grad_x + grad_y*grad_y) + 1e-8
        ux = grad_x / mag
        uy = grad_y / mag

        # Create figure of exact texture_resolution
        fig = plt.figure(figsize=(1,1), dpi=texture_resolution)
        ax = fig.add_axes([0,0,1,1])
        # Use origin='upper' because we flipped array
        ax.imshow(curv_np, cmap='viridis', origin='lower')
        ax.quiver(
            np.arange(0,W,step), np.arange(0,H,step),
            ux[0:H:step,0:W:step], uy[0:H:step,0:W:step],
            color='white', angles='xy', scale_units='xy', scale=None, width=0.0015
        )
        ax.axis('off')
        fig.canvas.draw()

        # Extract image buffer
        buf = fig.canvas.tostring_rgb()
        img = np.frombuffer(buf, dtype=np.uint8).reshape((texture_resolution, texture_resolution, 3))
        baked_texture = img.astype(np.float32) / 255.0
        plt.close(fig)

        # Assign and export
        self.mesh.albedo = torch.tensor(baked_texture, dtype=torch.float32, device=self.device)
        self.mesh.write_obj(path)
        print(f"[INFO] Mesh with baked curvature texture saved to '{path}'")

    def bake_texture(self, uv_coords, faces, vertex_colors, uv_faces, resolution=512):
        """
        Renders the per-vertex colors into a texture image using UV mapping.
        Ensures correct color interpolation and encoding.
        """

        # Initialize an empty white texture
        texture = np.ones((resolution, resolution, 3), dtype=np.float32)

        # Normalize UV coordinates     
        uv_coords = (uv_coords * (resolution - 1)).astype(np.int32)  # Scale to pixel space


        # Iterate over each face directly: 
        # Use the UV connectivity provided by uv_faces so that 
        # each triangle is baked exactly as it appears in the UV map
        # This ensures that we’re not re-triangulating the entire set of UVs,
        # and that each face gets its intended color.
        for face_indices, vertex_indices in zip(uv_faces, faces):
            coordinates2D = uv_coords[face_indices]  # Three UV points for the face
            # Get corresponding vertex colors (make sure the ordering matches)
            face_colors = vertex_colors[vertex_indices]
            # For simplicity, use the average color for the face
            color_rgb = face_colors.mean(axis=0)
            # Fill the triangle in the texture
            cv2.fillConvexPoly(texture, coordinates2D, color_rgb)


        # This part is added to resolve a minor visual issue for the texture:
        # So, when looking at the final 3D object with the texture, white seams
        # were observed. So, these seams typically indicate that there are gaps
        # in my texture where no triangle was filled. So, I am not sure why they
        # are exactly happening (IDK the root cause!) but at least I could remove
        # these seams by just simply do the erosion operation to cover the gaps and white seams.
        kernel = np.ones((2, 2), np.uint8)
        texture = cv2.erode(texture, kernel, iterations=5)

        return texture


    def project_3Dcurvatures_into_texture(self):

        V = np.array(self.mesh.v.cpu())  # Vertex positions (N x 3)
        F = np.array(self.mesh.f.cpu())  # Faces (M x 3)
        UV = np.array(self.mesh.vt.cpu())  # UV coordinates per vertex (N x 2)
        
        # -------------------------------
        # 2. Compute principal curvatures with libigl
        # -------------------------------
        d1, d2, k1, k2 = igl.principal_curvature(V, F)
        print("Curvature k1 (max):", k1)
        print("Curvature k2 (min):", k2)
        print("Principal directions d1 (max):", d1)
        print("Principal directions d2 (min):", d2)

        # -------------------------------
        # 3. Build a mapping from vertex to an incident face
        # -------------------------------        
        vertex_faces = defaultdict(list)
        for f in F:
            for v in f:
                vertex_faces[v].append(f)

        # -------------------------------
        # 4. For each vertex: compute the UV tangent basis and project curvature directions
        # -------------------------------
        curvature_2D = np.zeros((V.shape[0], 2, 2))
        for i in range(V.shape[0]):
            # Get the two 3D principal curvature directions for vertex i.
            d1_i = d1[i, :]
            d2_i = d2[i, :]

            # Find the best incident face for vertex i using the largest absolute UV Jacobian determinant.
            best_face = None
            best_denom = 0
            for f in vertex_faces[i]:
                uv0, uv1, uv2 = UV[f[0]], UV[f[1]], UV[f[2]]
                duv1 = uv1 - uv0
                duv2 = uv2 - uv0
                denom = duv1[0] * duv2[1] - duv2[0] * duv1[1]
                if abs(denom) > best_denom:
                    best_denom = abs(denom)
                    best_face = f
            if best_face is None:
                continue  # Skip if vertex is isolated
            f = best_face

            # Ensure vertex i is first in the face ordering for consistency.
            if f[0] != i:
                idx = np.where(f == i)[0][0]
                f = np.roll(f, -idx)

            # Extract the 3D positions and UV coordinates of the face vertices.
            v0, v1, v2 = V[f[0]], V[f[1]], V[f[2]]
            uv0, uv1, uv2 = UV[f[0]], UV[f[1]], UV[f[2]]

            # Compute the edge vectors in 3D.
            e1 = v1 - v0
            e2 = v2 - v0

            # Compute the differences in UV coordinates.
            duv1 = uv1 - uv0
            duv2 = uv2 - uv0

            # Compute the determinant for the UV mapping Jacobian.
            denom = duv1[0] * duv2[1] - duv2[0] * duv1[1]
            if abs(denom) < 1e-6:
                # If the UV mapping is degenerate, fall back on an arbitrary tangent basis.
                n = igl.per_vertex_normals(V, F)[i]
                tangent = np.cross(n, [1, 0, 0])
                if np.linalg.norm(tangent) < 1e-6:
                    tangent = np.cross(n, [0, 1, 0])
                tangent /= np.linalg.norm(tangent)
                bitangent = np.cross(n, tangent)
            else:
                f_val = 1.0 / denom
                tangent = f_val * (duv2[1] * e1 - duv1[1] * e2)
                bitangent = f_val * (-duv2[0] * e1 + duv1[0] * e2)
                tangent /= np.linalg.norm(tangent)
                bitangent /= np.linalg.norm(bitangent)

            # Debug: Print the tangent and bitangent for the first few vertices.
            if i < 3:
                print(f"Vertex {i} tangent:", tangent)
                print(f"Vertex {i} bitangent:", bitangent)
                print(f"Dot product (should be ~0):", np.dot(tangent, bitangent))

            # -------------------------------
            # 5. Project the 3D curvature directions to the UV space.
            # -------------------------------
            proj_d1 = np.array([np.dot(d1_i, tangent), np.dot(d1_i, bitangent)])
            proj_d2 = np.array([np.dot(d2_i, tangent), np.dot(d2_i, bitangent)])
            curvature_2D[i, 0, :] = proj_d1
            curvature_2D[i, 1, :] = proj_d2

        # -------------------------------
        # 6. Output example results in text
        # -------------------------------
        print("Example curvature projections in UV space:")
        for i in range(min(5, V.shape[0])):
            print(f"\nVertex {i}:")
            print(f"  3D principal directions:\n    d1 = {d1[i, :]}\n    d2 = {d2[i, :]}")
            print(f"  2D projected directions:\n    proj_d1 = {curvature_2D[i, 0, :]}\n    proj_d2 = {curvature_2D[i, 1, :]}")

        # -------------------------------
        # 7. Visualization of the curvature directions
        # -------------------------------        
        # For clarity, sample a subset of vertices for quiver plots.
        num_samples = min(50, V.shape[0])
        sample_idx = np.random.choice(V.shape[0], num_samples, replace=False)

        # Compute a scale for 3D arrows based on the bounding box size.
        bbox_size = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
        scale_3d = bbox_size * 0.02
        uv_arrow_scale = 0.1  # Arbitrary scale factor for UV arrows

        # 3D Visualization: Plot the mesh vertices and the two curvature directions.
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(V[:, 0], V[:, 1], V[:, 2], color='k', s=5, alpha=0.5)
        ax1.quiver(V[sample_idx, 0], V[sample_idx, 1], V[sample_idx, 2],
                d1[sample_idx, 0], d1[sample_idx, 1], d1[sample_idx, 2],
                length=scale_3d, color='r', normalize=True, label='d1')
        ax1.quiver(V[sample_idx, 0], V[sample_idx, 1], V[sample_idx, 2],
                d2[sample_idx, 0], d2[sample_idx, 1], d2[sample_idx, 2],
                length=scale_3d, color='b', normalize=True, label='d2')
        ax1.set_title("3D Curvature Directions")
        ax1.legend()

        # 2D Visualization in UV space: Scatter UVs and draw projected curvature vectors.
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(UV[:, 0], UV[:, 1], color='k', s=5, alpha=0.5)
        # d1 projection (red arrows)
        ax2.quiver(UV[sample_idx, 0], UV[sample_idx, 1],
                curvature_2D[sample_idx, 0, 0], curvature_2D[sample_idx, 0, 1],
                color='r', angles='xy', scale_units='xy', scale=1/uv_arrow_scale, label='proj_d1')
        # d2 projection (blue arrows)
        ax2.quiver(UV[sample_idx, 0], UV[sample_idx, 1],
                curvature_2D[sample_idx, 1, 0], curvature_2D[sample_idx, 1, 1],
                color='b', angles='xy', scale_units='xy', scale=1/uv_arrow_scale, label='proj_d2')
        ax2.set_title("2D Projected Curvature Directions (UV Space)")
        ax2.set_aspect('equal')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def compute_vertex_curvature(self, mesh_vertices, mesh_Faces) -> np.ndarray:
        """
        Compute mean curvature at each vertex using libigl.
        v: [V,3] vertices
        f: [F,3] face indices
        Returns curv_v: [V] mean curvature per vertex.
        """

        # Compute principal curvature using libigl
        v1, v2, k1, k2 = igl.principal_curvature(mesh_vertices, mesh_Faces)
        # curv_v = 0.5 * (k1 + k2)  # per-vertex signed mean curvature

        curv_v = k2 # per-vertex signed mean curvature
        
        # Normalize to [-1,1]
        mn, mx = curv_v.min(), curv_v.max()
        curv_v = (curv_v - mn) / (mx - mn + 1e-20)  # [0,1]
        curv_v = curv_v * 2.0 - 1.0                 # [-1,1]
        return curv_v
    
    def compute_max_principal_direction(self, mesh_vertices, mesh_faces) -> np.ndarray:
        """
        Compute maximum principal curvature direction at each vertex using libigl.

        Args:
            mesh_vertices: [V, 3] array of 3D vertex positions
            mesh_faces: [F, 3] array of face indices

        Returns:
            v1: [V, 3] array of maximum principal curvature direction vectors per vertex
        """
        # Compute principal directions and curvatures using libigl
        v1, v2, k1, k2 = igl.principal_curvature(mesh_vertices, mesh_faces)

        # v1 is the direction of maximum curvature (k1), already unit-length
        return v1

    def build_curvature_texture(self, vt: np.ndarray, ft_uv: np.ndarray,
                                mesh_faces: np.ndarray, curv_v: np.ndarray,
                                H: int, W: int) -> np.ndarray:
        """
        vt: [Vt,2] UV coords in [0,1]
        ft_uv: [F,3] UV face indices into vt
        mesh_f: [F,3] 3D face indices into mesh.v
        curv_v: [V] per-vertex curvature for mesh.v
        """
        ######################### Step 1: Initialization of the Output Arrays #####################
        # tex: A 2D array that will accumulate the curvature values for each pixel (in the texture space).
        # weight: A 2D array that records how many times a pixel has been "covered" by triangles (for averaging).
        tex = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        ######################### Step 2: Mapping UV Coordinates to Pixels #########################
        # The UV coordinates in "vt" are assumed to be normalized in the range [0,1].
        # Multiplying by [W-1, H-1] scales them to pixel indices (for instance, if W and H are 1024, values are mapped to [0,1023]).
        # "uv_pix" now represents the pixel positions for each UV vertex.        
        uv_pix = (vt * np.array([[W-1, H-1]])).astype(np.float32)  # [Vt,2]

        ######################### Step 3: Loop Over Triangles in UV Space #########################
        # The function loops over each triangle in the UV domain.
        # For each triangle, "tri_uv" contains the indices (into vt, now uv_pix) of the triangle’s vertices.
        # "tri_v" holds the corresponding indices into the 3D mesh (for fetching curvature values from "curv_v").

        # Precompute barycentric inv matrices per face in UV space
        for tri_uv, tri_v in zip(ft_uv, mesh_faces):
            # UV triangle corners
            pts_uv = uv_pix[tri_uv]  # shape [3,2]

            ######################### Step 4: Compute the Barycentric Coordinate Mapping for the Triangle #########################
            # A 3×3 matrix A is constructed where:
            # The first row contains the x-coordinates of the triangle’s corners.
            # The second row contains the y-coordinates.
            # The third row is all ones (for homogeneous coordinates).
            # By inverting A, we obtain a matrix "invA" that, when multiplied by a homogeneous coordinate 
            # [x,y,1], gives the barycentric coordinates for that pixel inside the triangle.
            # If the matrix is singular (degenerate triangle), the function skips that triangle.
            A = np.array([
                [pts_uv[0,0], pts_uv[1,0], pts_uv[2,0]],
                [pts_uv[0,1], pts_uv[1,1], pts_uv[2,1]],
                [1.0,         1.0,         1.0       ]
            ], dtype=np.float32)
            try:
                invA = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue  # skip degenerate UV triangles

            ######################### Step 5: Compute a Bounding Box for the Triangle #########################
            # To avoid testing every pixel in the whole texture, compute a tight bounding box that encloses the triangle in UV space.
            # The bounding box is then clipped to the texture boundaries.            
            min_x, max_x = int(pts_uv[:,0].min()), int(pts_uv[:,0].max())+1
            min_y, max_y = int(pts_uv[:,1].min()), int(pts_uv[:,1].max())+1
            min_x, max_x = np.clip([min_x, max_x], 0, W-1)
            min_y, max_y = np.clip([min_y, max_y], 0, H-1)

            ######################### Step 6: Rasterize the Triangle onto the Texture #########################
            # For every pixel in the bounding box, calculate its barycentric coordinates by multiplying invA with 
            # [x,y,1].
            # If all barycentric coordinates are nonnegative (within a small tolerance), the pixel lies inside (or on the edge of) the triangle.
            # Compute the curvature value for that pixel by taking a weighted sum of the curvature at each of the triangle’s vertices:
            # val = bary[0] × curv_v[tri_v[0]] + bary[1] × curv_v[tri_v[1]] + bary[2]×curv_v[tri_v[2]]
            # Add that value to tex[y,x] and increment a weight counter so that if a pixel is covered by multiple triangles, you can average them later.
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    bary = invA @ np.array([x, y, 1.0], dtype=np.float32)
                    if (bary >= -1e-4).all():
                        # Interpolate curvature from 3D-vertex curv_v[tri_v]
                        val = float(bary[0] * curv_v[tri_v[0]] +
                                    bary[1] * curv_v[tri_v[1]] +
                                    bary[2] * curv_v[tri_v[2]])
                        tex[y, x] += val
                        weight[y, x] += 1.0

        ######################### Step 7: Normalization and Return #########################
        # After looping over all triangles, each pixel in tex has an accumulated curvature value and a weight (the number of triangles that contributed).
        # For every pixel where weight > 0, we divide by the weight to obtain an average curvature value.
        # Finally, convert the resulting NumPy array to a PyTorch tensor on the desired device and return it.
        mask = weight > 0
        tex[mask] /= weight[mask]
        
        curvature_map = torch.from_numpy(tex).to(device='cuda')

        return curvature_map

    def build_direction_map_texture(self, 
                                    v: np.ndarray,
                                    vt: np.ndarray,
                                    f: np.ndarray,
                                    ft_uv: np.ndarray,
                                    principal_dir: np.ndarray,
                                    H: int,
                                    W: int) -> torch.Tensor:
        """
        Rasterizes 3D principal curvature directions into UV space.

        v: [V,3]             3D mesh vertices
        vt: [Vt,2]           UV coordinates (in [0,1])
        f: [F,3]             face indices into v
        ft_uv: [F,3]         face indices into vt
        principal_dir: [V,3] 3D direction vectors (e.g., max principal curvature)
        H, W:                texture resolution

        Returns:
            direction_map: [H, W, 2] PyTorch tensor with 2D direction vectors in UV space.
        """
        tex = np.zeros((H, W, 2), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        uv_pix = vt * np.array([[W - 1, H - 1]])  # map UV to pixel coords

        for face, face_uv in zip(f, ft_uv):
            verts = v[face]                # [3,3]
            dirs = principal_dir[face]     # [3,3]
            uvs = uv_pix[face_uv]          # [3,2]

            # 3D edges
            edge1_3d = verts[1] - verts[0]
            edge2_3d = verts[2] - verts[0]

            # UV edges
            edge1_uv = uvs[1] - uvs[0]
            edge2_uv = uvs[2] - uvs[0]

            # Build the 2×2 UV Jacobian
            uv_matrix = np.stack([edge1_uv, edge2_uv], axis=1)  # [2,2]
            try:
                uv_inv = np.linalg.inv(uv_matrix)
            except np.linalg.LinAlgError:
                continue  # degenerate UV triangle

            # Compute tangent basis T,B in R³
            TB = np.stack([edge1_3d, edge2_3d], axis=1)  # [3,2]
            dxy_duv = TB @ uv_inv                        # [3,2]
            T = dxy_duv[:, 0]
            B = dxy_duv[:, 1]

            # projection matrix 3D→UV
            proj_matrix = np.stack([T, B], axis=1)       # [3,2]

            # bounding box in pixel space
            min_x, max_x = int(uvs[:, 0].min()), int(uvs[:, 0].max()) + 1
            min_y, max_y = int(uvs[:, 1].min()), int(uvs[:, 1].max()) + 1
            min_x, max_x = np.clip([min_x, max_x], 0, W - 1)
            min_y, max_y = np.clip([min_y, max_y], 0, H - 1)

            # barycentric inv matrix
            A = np.array([
                [uvs[0, 0], uvs[1, 0], uvs[2, 0]],
                [uvs[0, 1], uvs[1, 1], uvs[2, 1]],
                [1.0,       1.0,       1.0]
            ], dtype=np.float32)
            try:
                invA = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue

            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    bary = invA @ np.array([x, y, 1.0], dtype=np.float32)
                    if (bary >= -1e-4).all():
                        dir_3d = (
                            bary[0] * dirs[0] +
                            bary[1] * dirs[1] +
                            bary[2] * dirs[2]
                        )
                        # project to UV
                        dir_2d = dir_3d @ proj_matrix  # [2]
                        tex[y, x] += dir_2d
                        weight[y, x] += 1.0

        # normalize interpolated directions
        mask = weight > 0
        tex[mask] /= weight[mask, None]

        # —— OVERRIDE HERE: make every pixel a horizontal unit vector ——
        # comment out these two lines to restore real projection!
        tex[..., 0] = -1.0  # x component = 1
        tex[..., 1] = 1.0  # y component = 0

        direction_map = torch.from_numpy(tex).to(device='cuda')  # [H, W, 2]
        return direction_map

    def visualize_curvature_map(self, curvature_map: torch.Tensor, show_quiver=True, step=10):
        """
        Plot a 2D curvature heatmap (correctly oriented) with optional quiver.
        In summary, this function takes a 2D curvature heatmap curv, 
        computes the local slope vectors, normalizes them, 
        samples them at a regular grid, and draws arrows showing 
        “this is the direction and orientation of the curvature flow” 
        on top of the heatmap. It is a convenient way to visually verify that
        our curvature extraction is correct and that the “flow lines” match the shape features.
        """
        import torch.nn.functional as F
        import numpy as np

        # Ensure curvature_map is [1, 1, H, W]
        if curvature_map.ndim == 3:
            curv_t = curvature_map[0:1, None, ...]
        elif curvature_map.ndim == 2:
            curv_t = curvature_map[None, None, ...]
        else:
            curv_t = curvature_map.unsqueeze(1)

        device = curv_t.device
        # Define Sobel kernels on same device
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=device).view(1,1,3,3) / 8.0
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1,-2,-1]], dtype=torch.float32, device=device).view(1,1,3,3) / 8.0
        sobel_x.requires_grad_(False)
        sobel_y.requires_grad_(False)

        # Compute gradients via conv2d
        grad_x = F.conv2d(curv_t, sobel_x, padding=1)
        grad_y = F.conv2d(curv_t, sobel_y, padding=1)
        # Convert to NumPy for plotting
        gx = grad_x[0,0].cpu().numpy()
        gy = grad_y[0,0].cpu().numpy()

        # Convert curvature to numpy heatmap and flip vertically so top becomes bottom
        curv = curv_t[0,0].cpu().numpy()
        curv = np.flipud(curv)
        H, W = curv.shape

        # 1) Heatmap (origin='upper' since we've flipped data)
        plt.figure(figsize=(6,6))
        plt.imshow(curv, cmap='viridis', origin='lower')
        plt.colorbar(label='Curvature')
        plt.title("2D Curvature Heatmap + Sobel Gradients")

        # 2) Optional: overlay gradient quiver
        if show_quiver:
            # flip gradients to match flipped heatmap
            gx = np.flipud(gx)
            gy = np.flipud(gy)
            mag = np.sqrt(gx*gx + gy*gy) + 1e-8
            ux = gx / mag
            uy = gy / mag
            
            # # Force a constant vector field (1,1) everywhere
            # ux = np.ones_like(ux)
            # uy = np.ones_like(ux)
            # # In flipped coords, uy=+1 now points upward            

            # Sample grid
            Y, X = np.mgrid[0:H:step, 0:W:step]
            U = ux[0:H:step, 0:W:step]
            V = uy[0:H:step, 0:W:step]

            plt.quiver(X, Y, U, V,
                       color='white',
                       angles='xy',
                       scale_units='xy',
                       scale=None,
                       width=0.0015)
                    #    headwidth=1,
                    #    headlength=4)
        plt.show()



    def visualize_direction_map(self, direction_map: torch.Tensor, step=50, normalize=True):
        """
        Visualize a 2D vector field (direction map) as arrows.

        Args:
            direction_map: Tensor of shape [H, W, 2]
            step: int, spacing between vectors (larger = sparser)
            normalize: bool, whether to normalize vectors for visualization
            autoscale: bool, whether to auto-scale arrows based on magnitude
        """
        if direction_map.ndim != 3 or direction_map.shape[2] != 2:
            raise ValueError("Expected shape [H, W, 2]")

        H, W = direction_map.shape[:2]
        dx = direction_map[..., 0]  # [H, W]
        dy = direction_map[..., 1]  # [H, W]

        if normalize:
            norm = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
            dx = dx / (norm + 1e-8)
            dy = dy / (norm + 1e-8)

        # Sample vectors at intervals
        Y, X = torch.meshgrid(torch.arange(0, H, step), torch.arange(0, W, step), indexing='ij')
        U = dx[Y, X].cpu().numpy()
        V = dy[Y, X].cpu().numpy()    

        U = Y * 0 + 0.1
        V = X * 0

        X = X.cpu().numpy()
        Y = Y.cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', width=0.003)
        plt.gca().set_aspect('equal')
        plt.title("2D Principal Curvature Directions (Projected)")
        plt.xlabel("U (Texture X)")
        plt.ylabel("V (Texture Y)")
        plt.grid(True)
        plt.show()

    def colorize_curvature_map(self, curv_map: torch.Tensor,
                               thresh=-0.6) -> np.ndarray:
        """
        curv_map: torch.Tensor of shape [1,1,H,W], values in [-1,1]
        Returns an RGB image as a NumPy array of shape [H,W,3] in [0,1].
        """

        v = curv_map.squeeze().cpu().numpy()  # [H,W]
        
        red_mask = (v > thresh).astype(np.float32)
        blue_mask = 1.0 - red_mask  # inverse of red_mask

        red   = red_mask
        green = np.zeros_like(red)  # no green
        blue  = blue_mask


        img = np.stack([red, green, blue], axis=-1)  # [H,W,3]
        img = torch.tensor(img, dtype=torch.float32, device='cuda')
        return img
    
    def curvature_reward_smooth(self, pred_tex: torch.Tensor, curv_map: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
        """
        pred_tex: [1,3,H,W]  This is the texture image (unwrapped from 3D), with 3 channels (RGB), values ∈ [0, 1]
        curv_map: [1,1,H,W]  This is the UV-space curvature map, with values ∈ [-1, 1] where:
                             -1 could represent low/negative curvature (valleys),
                             +1 could represent high/positive curvature (peaks).
        Returns scalar reward.
        """
        red  = pred_tex[:,0:1] # Extract Red channel -> [1,1,H,W]
        blue = pred_tex[:,2:3] # Extract Blue channel -> [1,1,H,W]

        # This "red - blue" value is used as a very basic measure of "color direction":
        # If red dominates: red - blue > 0
        # If blue dominates: red - blue < 0
        red_blue = red - blue  # Difference between red and blue -> still [1,1,H,W]


        # This is the key part:
        # High positive curvature multiplied by strong red color -> positive reward.
        # Low (negative) curvature multiplied by strong blue color -> also positive reward.
        # So the model gets rewarded when:
        # Red is painted in high curvature areas.
        # Blue is painted in low curvature areas.
        per_pixel_reward = curv_map * red_blue   # [H,W] * [H,W]


        # Averages the contribution across all pixels to produce a single scalar reward.
        return per_pixel_reward.mean()
    
    def curvature_reward_sharp(self, pred_tex: torch.Tensor, curv_map: torch.Tensor,
                     curvature_thresh: float = -0.3, red_blue_thresh: float = 0.4) -> torch.Tensor:
        """
        pred_tex: [1,3,H,W] RGB in [0,1]
        curv_map: [1,1,H,W] in [-1,1]

        curvature_thresh: threshold to decide high vs low curvature
        red_blue_thresh: threshold to decide whether a pixel is red or blue

        Returns scalar reward encouraging clear red-blue coloring based on curvature.
        """
        red  = pred_tex[:, 0:1]  # [1,1,H,W]
        blue = pred_tex[:, 2:3]
        red_blue_diff = red - blue  # Positive → red, Negative → blue

        # Binary decisions
        high_curv_mask = (curv_map > curvature_thresh).float()      # Should be red
        low_curv_mask  = (curv_map <= curvature_thresh).float()     # Should be blue

        is_red  = (red_blue_diff > red_blue_thresh).float()
        is_blue = (red_blue_diff < -red_blue_thresh).float()

        # Reward: +1 for correct, -1 for wrong
        reward_high_curv = high_curv_mask * (is_red - is_blue)
        reward_low_curv  = low_curv_mask  * (is_blue - is_red)

        total_reward = reward_high_curv + reward_low_curv

        # Normalize
        return total_reward.mean()


    def load_vec_tex(self, path: str, device: torch.device) -> torch.Tensor:
        """
        Load a vec_tex.txt containing N lines of "u,v" (per-UV vector) into
        a torch.FloatTensor of shape (2, N), inverting the v-component to match
        your coordinate convention.

        Args:
        path   : Path to comma-separated vec_tex file
        device : torch.device for the returned tensor

        Returns:
        vec_uv : torch.Tensor of shape (2, N)
        """
        # Load data as (N,2)
        data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        # Invert the v-component
        data[:, 1] *= -1.0

        # Transpose and convert to a PyTorch tensor of shape (2, N):
        vec_uv = torch.from_numpy(data.T).to(device)
        return vec_uv


from PIL import Image
from torchvision import transforms

def load_texture_image(path, H, W, device='cuda'):
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),  # Converts to [C,H,W], range [0,1]
    ])
    img = Image.open(path).convert("RGB")
    tex = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    return tex

def GENERATE_SUPERVISED_CURVATURE_REWARD():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curvatureReward = CurvatureReward(device)

    # Load the mesh
    curvatureReward.load_mesh(texgen_config["mesh"])

    # renderer.project_3Dcurvatures_into_texture()

    # Export the mesh with curvature-based vertex colors
    curvatureReward.export_mesh_with_curvature_colors(texgen_config["save_path"])

def GENERATE_DIRECT_MEAN_CURVATURE_MAP():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curvatureReward = CurvatureReward(device)

    # Load the mesh
    mesh_vertices, mesh_faces, uv_coords, uv_faces = curvatureReward.load_mesh(texgen_config["mesh"])

    print("The mesh loaded!")

    # Compute curvature
    curv_v = curvatureReward.compute_vertex_curvature(mesh_vertices, mesh_faces)
    
    print("The curvature map computed!")
    print(curv_v)


    # 4. Rasterize curvature to UV map
    H = W = int(texgen_config["texture_size"])

    curv_map = curvatureReward.build_curvature_texture(uv_coords, uv_faces, mesh_faces, curv_v, H, W)    
    kiui.vis.plot_image(curv_map)

    # Colorize the curvature texture map with blue and red (red for high- and blue for low-curvature parts)
    curv_map_colorized = curvatureReward.colorize_curvature_map(curv_map)
    curvatureReward.mesh.albedo = curv_map_colorized
    curvatureReward.mesh.write_obj(texgen_config["save_path"])


    # 5. In your training loop, after you generate a predicted texture `pred_tex` [1,3,H,W]:
    texture_path = r'C:\Users\User\Desktop\Texture-DPO\Texture-DPO\Texture-DPO\Scripts\Cow_albedo.png'
    H = W = 1024
    pred_tex = load_texture_image(texture_path, H, W, device='cuda')
    kiui.vis.plot_image(pred_tex)
    reward = curvatureReward.curvature_reward_smooth(pred_tex, curv_map)

    print(curv_map.shape)
    print(pred_tex.shape)
    print(reward)


def GENERATE_3DOBJ_WITH_CURVATURE_GRADIENTS_VECTOR_ON_IT():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curvatureReward = CurvatureReward(device)
    
    # Export the mesh with curvature-based vertex colors
    curvatureReward.export_mesh_with_curvature_directions(texgen_config["save_path"])



def GENERATE_DIRECT_MAX_CURVATURE_MAP():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curvatureReward = CurvatureReward(device)

    # Load the mesh
    mesh_vertices, mesh_faces, uv_coords, uv_faces = curvatureReward.load_mesh(texgen_config["mesh"])

    print("The mesh loaded!")

    # Compute curvature
    curv_max = curvatureReward.compute_max_principal_direction(mesh_vertices, mesh_faces)
    
    print("The curvature map computed!")
    print(curv_max)
    print(curv_max.shape)


    # 4. Rasterize curvature to UV map
    H = W = int(texgen_config["texture_size"])

    curv_map = curvatureReward.build_direction_map_texture(mesh_vertices, uv_coords, mesh_faces, uv_faces, curv_max, H, W)    
    # torch.set_printoptions(threshold=torch.inf)
    # print(curv_map)
    # print(curv_map.shape)
    curvatureReward.visualize_direction_map(curv_map)

    # Colorize the curvature texture map with blue and red (red for high- and blue for low-curvature parts)
    curv_map_colorized = curvatureReward.colorize_curvature_map(curv_map)
    curvatureReward.mesh.albedo = curv_map_colorized
    curvatureReward.mesh.write_obj(texgen_config["save_path"])


    # 5. In your training loop, after you generate a predicted texture `pred_tex` [1,3,H,W]:
    texture_path = r'C:\Users\User\Desktop\Texture-DPO\Texture-DPO\Texture-DPO\Scripts\Cow_albedo.png'
    H = W = 1024
    pred_tex = load_texture_image(texture_path, H, W, device='cuda')
    kiui.vis.plot_image(pred_tex)
    reward = curvatureReward.curvature_reward_smooth(pred_tex, curv_map)

    print(curv_map.shape)
    print(pred_tex.shape)
    print(reward)



def VISUALIZE_3DCURVATURES_ON_MESH():
    import igl
    import numpy as np
    import open3d as o3d  # for visualization

    # === 1. Load mesh ===
    V, F = igl.read_triangle_mesh(texgen_config["mesh"])  # replace with your mesh path

    # === 2. Compute principal curvatures ===
    # Output: PD1/PD2 = principal directions (3D), PV1/PV2 = principal values (scalar)
    maximal_curvature, minimal_curvature, PV1, PV2 = igl.principal_curvature(V, F)


    # === 3. Normalize directions ===
    maximal_curvature_normalized = maximal_curvature / (np.linalg.norm(maximal_curvature, axis=1, keepdims=True) + 1e-9)
    # minimal_curvature_normalized = minimal_curvature / (np.linalg.norm(minimal_curvature, axis=1, keepdims=True) + 1e-9)
    

    # === 4. Create arrows for minimum curvature directions ===
    arrow_length = 7
    arrow_radius = 0.4
    arrow_resolution = 20
    arrow_cone_ratio = 4  # larger = smaller cone head

    arrow_objs = []
    for i in range(0, V.shape[0], 2):  # downsample every 20th vertex for clarity
        origin = V[i]
        # direction = minimal_curvature_normalized[i]
        direction = maximal_curvature_normalized[i]

        # Create base arrow aligned with z-axis
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=arrow_radius,
            cone_radius=arrow_radius * 1.5,
            cylinder_height=arrow_length * (arrow_cone_ratio / (arrow_cone_ratio + 1)),
            cone_height=arrow_length / (arrow_cone_ratio + 1),
            resolution=arrow_resolution,
            cylinder_split=1,
            cone_split=1
        )
        arrow.paint_uniform_color([0, 0, 1])  # blue arrows

        # Compute rotation from (0,0,1) → direction
        z_axis = np.array([0.0, 0.0, 1.0])
        dir_norm = direction / np.linalg.norm(direction)
        rotation_axis = np.cross(z_axis, dir_norm)
        angle = np.arccos(np.clip(np.dot(z_axis, dir_norm), -1.0, 1.0))
        if np.linalg.norm(rotation_axis) > 1e-6:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis / np.linalg.norm(rotation_axis) * angle)
        else:
            R = np.eye(3)

        arrow.rotate(R, center=(0, 0, 0))
        arrow.translate(origin)
        arrow_objs.append(arrow)

    # === 5. Convert the mesh to Open3D format ===
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.compute_vertex_normals()

    # === 6. Visualize ===
    o3d.visualization.draw_geometries([mesh] + arrow_objs)


if __name__== "__main__":
    
    # GENERATE_3DOBJ_WITH_CURVATURE_GRADIENTS_VECTOR_ON_IT()

    VISUALIZE_3DCURVATURES_ON_MESH()




















