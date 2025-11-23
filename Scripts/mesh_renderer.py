import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from mesh import Mesh, safe_normalize

from Configuration.Config import texgen_config

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)



class Renderer(nn.Module):
    def __init__(self, device):
        
        super().__init__()

        self.device = device

        self.mesh = None

        if texgen_config["bg_image"] is not None and os.path.exists(texgen_config["bg_image"]):
            # load an image as the background
            bg_image = cv2.imread(texgen_config["bg_image"])
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            bg_image = torch.from_numpy(bg_image.astype(np.float32) / 255).to(self.device)
            self.bg = F.interpolate(bg_image.permute(2, 0, 1).unsqueeze(0), (texgen_config["render_resolution"], texgen_config["render_resolution"]), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).contiguous()
        else:
            # default as blender grey
            # self.bg = 0.807 * torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
            self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.bg_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        if not texgen_config["gui"]  or os.name == 'nt':
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

    @torch.no_grad()
    def load_mesh(self, path):
        if not os.path.exists(path):
            # try downloading from objaverse (treat path as uid)
            import objaverse
            objects = objaverse.load_objects(uids=[path], download_processes=1)
            path = objects[path]
            print(f'[INFO] load Objaverse from {path}')

        self.mesh = Mesh.load(path, front_dir=texgen_config["front_dir"], retex=texgen_config["retex"], remesh=texgen_config["remesh"] , device=self.device)

        # Extract vertices and faces
        vertices = self.mesh.v.cpu().numpy() # Vertices of the mesh
        faces = self.mesh.f.cpu().numpy()    # Triangles (facet) of the mesh 
        uv_coords = self.mesh.vt.cpu().numpy() if self.mesh.vt is not None else None # 2D coordinates in the image space correponding to each vertex
        uv_faces = self.mesh.ft.cpu().numpy() if self.mesh.ft is not None else None  # 2D coordinates in the image space correponding to each facet

        return vertices, faces, uv_coords, uv_faces


    @torch.no_grad()
    def export_mesh(self, path):
        self.mesh.write(path)
        
    def render(self, pose, proj, h, w):

        # This "result" dictionary is the image buffer that the InteX paper talks about:
        # From the InteX paper: (page 6, section 3.2 Iterative Texture Synthesis)
        # "Rendering: Firstly, we render the 3D shape with the current texture image
        # to obtain the necessary image buffers, including:
        # 1- an RGB image with inpainting mask        
        # 2- a depth map,
        # 3- a normal map,
        # 4- UV coordinates."
        results = {}

        # Get the vertices of the input mesh
        v = self.mesh.v

        # get v_clip and render rgb
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        if isinstance(proj, np.ndarray):
            proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)        

        # Representing the vertices in the camera space
        # "F.pad" is performing some padding on the input tensor which is the vertices of the input mesh. But WHY???
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        # "rast" has shape [minibatch_size, height, width, 4] and contains the main rasterizer output in order (u, v, z/w, triangle_id)
        # "rast_db" has have shape [minibatch_size, height, width, 4] and contain said derivatives in order (du/dX, du/dY, dv/dX, dv/dY)
        # "v_clip" is the vertices of the input mesh which is projected into the camera space (see a couple of lines above)
        # "self.mesh.f" is the triangle tensor with shape [num_triangles, 3]
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))


        # actually disparity (1 / depth), to align with controlnet
        disp = -1 / (v_cam[..., [2]] + 1e-20)
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-20) # pre-normalize
        depth, _ = dr.interpolate(disp, rast, self.mesh.f) # [1, H, W, 1]
        depth = depth.clamp(0, 1).squeeze(0) # [H, W, 1]

        alpha = (rast[..., 3:] > 0).float()

        # rgb texture
        texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]

        # get vn and render normal
        vn = self.mesh.vn        
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]

        # rot normal z axis is exactly viewdir-normal cosine
        viewcos = rot_normal[..., [2]].abs() # double-sided

        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

        # replace background
        albedo = alpha * albedo + (1 - alpha) * self.bg
        normal = alpha * normal + (1 - alpha) * self.bg_normal
        rot_normal = alpha * rot_normal + (1 - alpha) * self.bg_normal

        # extra texture (hard coded)
        if hasattr(self.mesh, 'cnt'):
            # The texture function is used for texture sampling. This function allows you to sample a texture 
            # (typically an image) at specified UV coordinates. The function performs bilinear filtering by default
            # but also supports other filtering modes. It can be used to map a texture onto a 3D model or in other
            # rendering tasks where you need to retrieve texture data based on coordinates.
            cnt = dr.texture(self.mesh.cnt.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            
            # The antialias function is used to smooth out the edges of rasterized primitives, reducing visual artifacts
            # like jagged edges (aliasing) that occur when rendering high-contrast edges. This function applies anti-aliasing
            # techniques to improve the visual quality of the rendered image, making it appear smoother and more visually appealing.
            cnt = dr.antialias(cnt, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            cnt = alpha * cnt + (1 - alpha) * 1 # 1 means no-inpaint in background
            results['cnt'] = cnt

        if hasattr(self.mesh, 'viewcos_cache'):
            viewcos_cache = dr.texture(self.mesh.viewcos_cache.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            viewcos_cache = dr.antialias(viewcos_cache, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            results['viewcos_cache'] = viewcos_cache

        if hasattr(self.mesh, 'ori_albedo'):
            ori_albedo = dr.texture(self.mesh.ori_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            ori_albedo = dr.antialias(ori_albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            ori_albedo = alpha * ori_albedo + (1 - alpha) * self.bg
            results['ori_image'] = ori_albedo


        # all shaped as [H, W, C]

        # This is the final rendered image, typically a 2D array (or tensor) representing the RGB color values
        # for each pixel in the image. This is the primary output you would see on the screen after the mesh 
        # has been rasterized and all shading and lighting calculations have been applied.
        results['image'] = albedo

        # This variable usually stores the alpha channel of the rendered image, representing the transparency
        # of each pixel. An alpha value of 1 indicates full opacity, while 0 indicates full transparency.
        # This can be useful for compositing images or handling objects with transparent surfaces.
        results['alpha'] = alpha

        # This is the depth buffer or z-buffer, which contains the depth value for each pixel in the image.
        # The depth value indicates how far a pixel is from the camera in the scene. This buffer is crucial
        # for determining which surfaces are visible and which are hidden behind other objects (occlusion).
        results['depth'] = depth

        # This variable stores the normal vectors for each pixel in the image. A normal vector is a 3D vector
        # that is perpendicular to the surface of the mesh at that pixel. Normals are essential for lighting
        # calculations, as they determine how light interacts with the surface, affecting shading and reflection.
        results['normal'] = normal # in [-1, 1]

        # This likely represents the normals that have been rotated or transformed according to some operation,
        # such as a view transformation or a rotation applied to the entire mesh. These rotated normals might be
        # used in specific shading or lighting models where the orientation of the normals needs to be adjusted
        # relative to the camera or light sources.
        results['rot_normal'] = rot_normal # in [-1, 1]

        # This variable typically stores the cosine of the angle between the view vector (from the camera to the
        # surface point) and the surface normal. The viewcos is often used in shading models, particularly in 
        # specular highlights and reflection calculations, to determine how the surface is oriented relative to the viewer.
        results['viewcos'] = viewcos

        # This array contains the UV coordinates for each pixel, which map points on the 3D surface of the mesh to
        # points on a 2D texture. UV mapping is how textures are applied to 3D models. The uvs array allows you to
        # determine which part of the texture corresponds to each pixel on the rendered mesh.
        results['uvs'] = texc.squeeze(0)

        return results
    

    def render1(self, mesh_state, pose, proj, h, w):

        # This "result" dictionary is the image buffer that the InteX paper talks about:
        # From the InteX paper: (page 6, section 3.2 Iterative Texture Synthesis)
        # "Rendering: Firstly, we render the 3D shape with the current texture image
        # to obtain the necessary image buffers, including:
        # 1- an RGB image with inpainting mask        
        # 2- a depth map,
        # 3- a normal map,
        # 4- UV coordinates."
        results = {}

        # print(mesh_state['albedo0'].shape)

        mesh_state_albedo = mesh_state['albedo0']
        mesh_state_cnt = mesh_state['cnt0'] 
        mesh_state_viewcos_cache = mesh_state['viewcos_cache']

        # Get the vertices of the input mesh
        v = self.mesh.v

        # get v_clip and render rgb
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        if isinstance(proj, np.ndarray):
            proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # Representing the vertices in the camera space
        # "F.pad" is performing some padding on the input tensor which is the vertices of the input mesh. But WHY???
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        # "rast" has shape [minibatch_size, height, width, 4] and contains the main rasterizer output in order (u, v, z/w, triangle_id)
        # "rast_db" has have shape [minibatch_size, height, width, 4] and contain said derivatives in order (du/dX, du/dY, dv/dX, dv/dY)
        # "v_clip" is the vertices of the input mesh which is projected into the camera space (see a couple of lines above)
        # "self.mesh.f" is the triangle tensor with shape [num_triangles, 3]
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))


        # actually disparity (1 / depth), to align with controlnet
        disp = -1 / (v_cam[..., [2]] + 1e-20)
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-20) # pre-normalize
        depth, _ = dr.interpolate(disp, rast, self.mesh.f) # [1, H, W, 1]
        depth = depth.clamp(0, 1).squeeze(0) # [H, W, 1]

        alpha = (rast[..., 3:] > 0).float()

        # rgb texture
        texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]
        
        # get vn and render normal
        vn = self.mesh.vn        
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]

        # rot normal z axis is exactly viewdir-normal cosine
        viewcos = rot_normal[..., [2]].abs() # double-sided

        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

        # replace background
        albedo = alpha * albedo + (1 - alpha) * self.bg
        normal = alpha * normal + (1 - alpha) * self.bg_normal
        rot_normal = alpha * rot_normal + (1 - alpha) * self.bg_normal

        # extra texture (hard coded)
        if hasattr(self.mesh, 'cnt'):
            # The texture function is used for texture sampling. This function allows you to sample a texture 
            # (typically an image) at specified UV coordinates. The function performs bilinear filtering by default
            # but also supports other filtering modes. It can be used to map a texture onto a 3D model or in other
            # rendering tasks where you need to retrieve texture data based on coordinates.
            cnt = dr.texture(mesh_state_cnt.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            
            # The antialias function is used to smooth out the edges of rasterized primitives, reducing visual artifacts
            # like jagged edges (aliasing) that occur when rendering high-contrast edges. This function applies anti-aliasing
            # techniques to improve the visual quality of the rendered image, making it appear smoother and more visually appealing.
            cnt = dr.antialias(cnt, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            cnt = alpha * cnt + (1 - alpha) * 1 # 1 means no-inpaint in background
            results['cnt'] = cnt
        
        if hasattr(self.mesh, 'viewcos_cache'):
            viewcos_cache = dr.texture(mesh_state_viewcos_cache.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            viewcos_cache = dr.antialias(viewcos_cache, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            results['viewcos_cache'] = viewcos_cache

        if hasattr(self.mesh, 'ori_albedo'):
            ori_albedo = dr.texture(self.mesh.ori_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
            ori_albedo = dr.antialias(ori_albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
            ori_albedo = alpha * ori_albedo + (1 - alpha) * self.bg
            results['ori_image'] = ori_albedo


        # all shaped as [H, W, C]

        # This is the final rendered image, typically a 2D array (or tensor) representing the RGB color values
        # for each pixel in the image. This is the primary output you would see on the screen after the mesh 
        # has been rasterized and all shading and lighting calculations have been applied.
        results['image'] = albedo

        # This variable usually stores the alpha channel of the rendered image, representing the transparency
        # of each pixel. An alpha value of 1 indicates full opacity, while 0 indicates full transparency.
        # This can be useful for compositing images or handling objects with transparent surfaces.
        results['alpha'] = alpha

        # This is the depth buffer or z-buffer, which contains the depth value for each pixel in the image.
        # The depth value indicates how far a pixel is from the camera in the scene. This buffer is crucial
        # for determining which surfaces are visible and which are hidden behind other objects (occlusion).
        results['depth'] = depth

        # This variable stores the normal vectors for each pixel in the image. A normal vector is a 3D vector
        # that is perpendicular to the surface of the mesh at that pixel. Normals are essential for lighting
        # calculations, as they determine how light interacts with the surface, affecting shading and reflection.
        results['normal'] = normal # in [-1, 1]

        # This likely represents the normals that have been rotated or transformed according to some operation,
        # such as a view transformation or a rotation applied to the entire mesh. These rotated normals might be
        # used in specific shading or lighting models where the orientation of the normals needs to be adjusted
        # relative to the camera or light sources.
        results['rot_normal'] = rot_normal # in [-1, 1]

        # This variable typically stores the cosine of the angle between the view vector (from the camera to the
        # surface point) and the surface normal. The viewcos is often used in shading models, particularly in 
        # specular highlights and reflection calculations, to determine how the surface is oriented relative to the viewer.
        results['viewcos'] = viewcos

        # This array contains the UV coordinates for each pixel, which map points on the 3D surface of the mesh to
        # points on a 2D texture. UV mapping is how textures are applied to 3D models. The uvs array allows you to
        # determine which part of the texture corresponds to each pixel on the rendered mesh.
        results['uvs'] = texc.squeeze(0)

        return results


    def render2(self, mesh_state, pose, proj, h, w):

        # This "result" dictionary is the image buffer that the InteX paper talks about:
        # From the InteX paper: (page 6, section 3.2 Iterative Texture Synthesis)
        # "Rendering: Firstly, we render the 3D shape with the current texture image
        # to obtain the necessary image buffers, including:
        # 1- an RGB image with inpainting mask
        # 2- a depth map,
        # 3- a normal map,
        # 4- UV coordinates."
        results = {}

        # print(mesh_state['albedo0'].shape)

        # mesh_state_albedo = mesh_state['albedo0']
        mesh_state_albedo = mesh_state

        # Get the vertices of the input mesh
        v = self.mesh.v

        # get v_clip and render rgb
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        if isinstance(proj, np.ndarray):
            proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # Representing the vertices in the camera space
        # "F.pad" is performing some padding on the input tensor which is the vertices of the input mesh. But WHY???
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        # "rast" has shape [minibatch_size, height, width, 4] and contains the main rasterizer output in order (u, v, z/w, triangle_id)
        # "rast_db" has have shape [minibatch_size, height, width, 4] and contain said derivatives in order (du/dX, du/dY, dv/dX, dv/dY)
        # "v_clip" is the vertices of the input mesh which is projected into the camera space (see a couple of lines above)
        # "self.mesh.f" is the triangle tensor with shape [num_triangles, 3]
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))


        # actually disparity (1 / depth), to align with controlnet
        disp = -1 / (v_cam[..., [2]] + 1e-20)
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-20) # pre-normalize
        depth, _ = dr.interpolate(disp, rast, self.mesh.f) # [1, H, W, 1]
        depth = depth.clamp(0, 1).squeeze(0) # [H, W, 1]

        alpha = (rast[..., 3:] > 0).float()

        # rgb texture
        texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
        albedo = dr.texture(mesh_state_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear') # [1, H, W, 3]

        # get vn and render normal
        vn = self.mesh.vn        
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]


        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]

        # replace background
        albedo = alpha * albedo + (1 - alpha) * self.bg
        normal = alpha * normal + (1 - alpha) * self.bg_normal
        rot_normal = alpha * rot_normal + (1 - alpha) * self.bg_normal

 

        # all shaped as [H, W, C]

        # This is the final rendered image, typically a 2D array (or tensor) representing the RGB color values
        # for each pixel in the image. This is the primary output you would see on the screen after the mesh 
        # has been rasterized and all shading and lighting calculations have been applied.
        results['image'] = albedo

        # This variable usually stores the alpha channel of the rendered image, representing the transparency
        # of each pixel. An alpha value of 1 indicates full opacity, while 0 indicates full transparency.
        # This can be useful for compositing images or handling objects with transparent surfaces.
        results['alpha'] = alpha

        return results

