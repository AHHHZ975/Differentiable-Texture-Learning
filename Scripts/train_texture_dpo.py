import contextlib
import os
import copy
import datetime
import time
import sys
import torch.utils
import torch.utils.checkpoint
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from Configuration.Config import texpref_config, texgen_config, common_config
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import prompts
import rewards
import cv2
import torch
import torch.nn.functional as F
from functools import partial
import tqdm
import copy
import numpy as np
from cam_utils import orbit_camera, orbit_camera_differentiable, undo_orbit_camera, undo_orbit_camera_differentiable, OrbitCamera
from mesh_renderer import Renderer
from grid_put import mipmap_linear_grid_put_2d
import kiui
import random
from matplotlib import pyplot as plt
from curvature_aware_reward import CurvatureReward
from symmetry_reward import load_mesh, compute_3d_pca, compute_symmetry_features, visualize_3d, visualize_3d_with_texture, compute_vertex_mirror_pairs

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)






def dilate_image(image, mask, iterations):
    # image: [H, W, C], current image
    # mask: [H, W], region with content (~mask is the region to inpaint)
    # iterations: int
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    if mask.dtype != bool:
        mask = mask > 0.5

    inpaint_region = binary_dilation(mask, iterations=iterations)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    image[tuple(inpaint_coords.T)] = image[tuple(search_coords[indices[:, 0]].T)]
    return image

# Support multi-dimensional comparison. Default demension is 1. You can add many rewards instead of only one to judge the preference of images.
# For example: A: clipscore-30 blipscore-10 LAION aesthetic score-6.0 ; B: 20, 8, 5.0  then A is prefered than B
# if C: 40, 4, 4.0 since C[0] = 40 > A[0] and C[1] < A[1], we do not think C is prefered than A or A is prefered than C 
def compare(a, b):
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
    if len(a.shape)==1:
        a = a[...,None]
        b = b[...,None]

    a_dominates = torch.logical_and(torch.all(a <= b, dim=1), torch.any(a < b, dim=1))
    b_dominates = torch.logical_and(torch.all(b <= a, dim=1), torch.any(b < a, dim=1))


    c = torch.zeros([a.shape[0],2],dtype=torch.float,device=a.device)

    c[a_dominates] = torch.tensor([-1., 1.],device=a.device)
    c[b_dominates] = torch.tensor([1., -1.],device=a.device)

    return c


class Differentiable_Texture_Learning:
    def __init__(self):

        self.forwar_pass_visit_count = 0

        # Camera resolution (800 * 800)
        self.W = texgen_config["W"]
        self.H = texgen_config["H"]

        # Camera class instantiation
        self.cam = OrbitCamera(self.W, self.H, r=texgen_config["radius"], fovy=texgen_config["fovy"])        

        self.device = torch.device("cuda")

        # The diffusion model(+control commands) which will be used to generate the texture 
        self.guidance = None

        # Renderer -> This clafss uses the differentiable rasterizer from Nvidia (nvdiffrast)
        self.renderer0 = Renderer(self.device)
        self.renderer1 = Renderer(self.device)
        self.renderer_reward = Renderer(self.device)

        # Load the mesh
        # There is a "self.mesh" variable inside the "Renderer" class 
        # which will be initialized with the mesh provided by the user.
        # All the information/variables related to the mesh object such as
        # vertices, faces, normals, albedo, etc are of type torch.tensor
        # and differentiable so that we could do the backpropagatation through them.
        if texgen_config["mesh"] is not None:
            mesh_vertices, mesh_faces, uv_coords, uv_faces = self.renderer0.load_mesh(texgen_config["mesh"])
            self.renderer1.load_mesh(texgen_config["mesh"])
            self.renderer_reward.load_mesh(texgen_config["reward_mesh"])

        # Input text prompt
        self.negative_prompt = texgen_config["nega_prompt"]

        ################## Geometry-Aware Texture Colorization Reward ##################
        if (texpref_config["reward_fn"] == "curvature_score_sharp") or (texpref_config["reward_fn"] == "curvature_score_smooth"):
            self.curvatureReward = CurvatureReward(self.device)
            # Compute the mean curvature for all vertices of the input mesh object
            self.curvature_in_3d = self.curvatureReward.compute_vertex_curvature(mesh_vertices, mesh_faces)
            H = W = int(texgen_config["texture_size"])
            self.curvature_in_2d = self.curvatureReward.build_curvature_texture(uv_coords, uv_faces, mesh_faces, self.curvature_in_3d, H, W)

        ################## Texture Features Emphasis Reward ##################
        if texpref_config["reward_fn"] == "mean_curvature_texture_alignment_reward":
            self.curvatureReward = CurvatureReward(self.device)
            # Compute the mean curvature for all vertices of the input mesh object
            mean_curvature_3d = self.curvatureReward.compute_vertex_curvature(mesh_vertices, mesh_faces)
            H = W = int(texgen_config["texture_size"])
            self.curvature_in_2d = self.curvatureReward.build_curvature_texture(uv_coords, uv_faces, mesh_faces, mean_curvature_3d, H, W)

            self.curvature_in_3d = self.curvatureReward.compute_max_principal_direction(mesh_vertices, mesh_faces)
            H = W = int(texgen_config["texture_size"])
            self.direction_in_2d = self.curvatureReward.build_direction_map_texture(mesh_vertices, uv_coords, mesh_faces, uv_faces, self.curvature_in_3d, H, W)

            self.curvatureReward.visualize_curvature_map(self.curvature_in_2d, show_quiver=True, step=30)

            self.curvatureReward.visualize_direction_map(self.direction_in_2d)

        ################## Geometry-Texture Alignment Reward ##################
        if texpref_config["reward_fn"] == "minimum_curvature_texture_alignment_reward":
            self.curvatureReward = CurvatureReward(self.device)
            # Load the minimum curvature for all vertices of the input mesh object
            self.curvature_in_2d = self.curvatureReward.load_vec_tex(path=texgen_config['vector_texture_path'], device=self.device)

            import igl
            _, self.UV, _, _, _, _ = igl.read_obj(texgen_config["mesh"])
            self.UV = self.UV.T
            self.UV = torch.from_numpy(self.UV).to(device=self.device, dtype=torch.float32)

        ################## Symmetry-Aware Texture Generation Reward ##################
        if (texpref_config["reward_fn"] == "symmetry_score"):
            import pyvista as pv

            # Load Mesh
            mesh_path = texgen_config["mesh"]
            mesh = load_mesh(mesh_path)
            vertices = mesh.vertices

            # Compute 3D PCA and symmetry features
            centroid, eigenvalues, eigenvectors = compute_3d_pca(vertices)
            line_point, line_direction, plane_normal = compute_symmetry_features(centroid, eigenvalues, eigenvectors)
            print("3D symmetry line passes through:", line_point)
            print("3D symmetry line direction:", line_direction)
            print("Symmetry plane normal:", plane_normal)

            # 3D Visualization
            mesh_pv = pv.wrap(mesh)
            visualize_3d_with_texture(mesh_pv, line_point, line_direction, plane_normal)
            
            # Use the mesh's UV mapping (must be presented)
            if not hasattr(mesh.visual, "uv"):
                raise ValueError("Mesh does not have UV coordinates!")
            uvs = np.array(mesh.visual.uv)  # shape (N,2) in [0,1]

            # Compute mirror UV correspondences
            uv_orig, uv_mirror = compute_vertex_mirror_pairs(mesh, centroid, plane_normal)

            print("uv_orig: ", uv_orig.shape)
            print("uv_mirror: ", uv_mirror.shape)
            
            # Precompute UV Coordinates as Tensors (for grid_sample)
            # These remain fixed across iterations.
            self.pixel_uv_orig =  torch.tensor(uv_orig, dtype=torch.float32, device=self.device)  # shape (N,2), in [0,1]
            self.pixel_uv_mirror = torch.tensor(uv_mirror, dtype=torch.float32, device=self.device)  # shape (N,2), in [0,1]

            

    @torch.no_grad()        
    def backup0(self):
        self.backup_albedo0 = self.albedo0.clone()
        self.backup_cnt0 = self.cnt0.clone()
        self.backup_viewcos_cache0 = self.renderer0.mesh.viewcos_cache.clone()
    
    @torch.no_grad()
    def backup1(self):
        self.backup_albedo1 = self.albedo1.clone()
        self.backup_cnt1 = self.cnt1.clone()
        self.backup_viewcos_cache1 = self.renderer1.mesh.viewcos_cache.clone()
    
    @torch.no_grad()
    def update_mesh_albedo0(self):
        mask = self.cnt0.squeeze(-1) > 0
        cur_albedo = self.albedo0.clone()        
        cur_albedo[mask] /= self.cnt0[mask].repeat(1, 3)
        self.renderer0.mesh.albedo = cur_albedo

    @torch.no_grad()
    def update_mesh_albedo0_rlhf(self):
        mask = self.cnt0.squeeze(-1) > 0
        cur_albedo = self.albedo0.clone()        
        # cur_albedo[mask] /= self.cnt0[mask].repeat(1, 3)
        self.renderer0.mesh.albedo = cur_albedo
    
    @torch.no_grad()
    def update_mesh_albedo1(self):
        mask = self.cnt1.squeeze(-1) > 0
        cur_albedo = self.albedo1.clone()
        cur_albedo[mask] /= self.cnt1[mask].repeat(1, 3)
        self.renderer1.mesh.albedo = cur_albedo

    
    @torch.no_grad()
    def dilate_texture(self):
        h = w = int(texgen_config["texture_size"])

        ############################# Mesh 0 #############################
        self.backup0()
        mask = self.cnt0.squeeze(-1) > 0        
        ## dilate texture
        mask = mask.view(h, w)
        mask = mask.detach().cpu().numpy()
        self.albedo0 = dilate_image(self.albedo0, mask, iterations=int(h*0.2))
        self.cnt0 = dilate_image(self.cnt0, mask, iterations=int(h*0.2))        
        self.update_mesh_albedo0()

        ############################# Mesh 1 #############################
        self.backup1()
        mask = self.cnt1.squeeze(-1) > 0        
        ## dilate texture
        mask = mask.view(h, w)
        mask = mask.detach().cpu().numpy()
        self.albedo1 = dilate_image(self.albedo1, mask, iterations=int(h*0.2))
        self.cnt1 = dilate_image(self.cnt1, mask, iterations=int(h*0.2))        
        self.update_mesh_albedo1()
    
    @torch.no_grad()
    def deblur_texture(self, ratio=2):
        h = w = int(texgen_config["texture_size"])

        ############################# Mesh 0 #############################
        self.backup0()

        # overall deblur by LR then SR
        # kiui.vis.plot_image(self.albedo0)
        cur_albedo = self.renderer0.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
        cur_albedo = cur_albedo.astype(np.float32) / 255
        # kiui.vis.plot_image(cur_albedo)
        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)

        self.renderer0.mesh.albedo = cur_albedo

        ############################# Mesh 1 #############################

        self.backup1()

        # overall deblur by LR then SR
        # kiui.vis.plot_image(self.albedo)
        cur_albedo = self.renderer1.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
        cur_albedo = cur_albedo.astype(np.float32) / 255
        # kiui.vis.plot_image(cur_albedo)
        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)

        self.renderer1.mesh.albedo = cur_albedo

    @torch.no_grad()
    def dilate_texture_rlhf(self):
        h = w = int(texgen_config["texture_size"])

        ############################# Mesh 0 #############################
        self.backup0()
        mask = self.cnt0.squeeze(-1) > 0
        ## dilate texture
        mask = mask.view(h, w)
        mask = mask.detach().cpu().numpy()
        # kiui.vis.plot_image(self.albedo0.detach().cpu().numpy())
        self.albedo0 = dilate_image(self.albedo0, mask, iterations=int(h*0.2))
        self.cnt0 = dilate_image(self.cnt0, mask, iterations=int(h*0.2))
        self.update_mesh_albedo0_rlhf()

    @torch.no_grad()
    def deblur_texture_rlhf(self, ratio=2):
        h = w = int(texgen_config["texture_size"])

        ############################# Mesh 0 #############################
        self.backup0()

        # overall deblur by LR then SR
        # kiui.vis.plot_image(self.albedo0)
        cur_albedo = self.renderer0.mesh.albedo.detach().cpu().numpy()
        cur_albedo = (cur_albedo * 255).astype(np.uint8)
        cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
        cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
        cur_albedo = cur_albedo.astype(np.float32) / 255
        # kiui.vis.plot_image(cur_albedo)
        cur_albedo = torch.from_numpy(cur_albedo).to(self.device)

        self.renderer0.mesh.albedo = cur_albedo

    def save_model(self, prompt: str, epoch: int, all_reward, all_loss, seed):
        # Creating the "logs" folder in the directory of the project
        logDir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'logs'))
        
        epochDir = os.path.join(logDir, f'Epoch{epoch}')

        sample0_path = os.path.join(epochDir, "sample0")        
        os.makedirs(sample0_path, exist_ok=True)        

        # Save the mesh objects and their texture with a standard format.
        obj0 = os.path.join(sample0_path, texgen_config["save_path"])        
        self.renderer0.export_mesh(obj0)        

        if common_config['training']:
            # Save the loss figure
            plt.plot([i.cpu().detach().numpy() for i in all_loss])
            plt.title('Loss')
            lossDir = os.path.join(epochDir, "loss.png")
            plt.savefig(lossDir)
            plt.close()
            
            # Save the reward figures
            plt.plot([i.cpu().detach().numpy() for i in all_reward])
            plt.title('Reward')
            rewardDir = os.path.join(epochDir, "reward.png")
            plt.savefig(rewardDir)
            plt.close()

        # Save the text prompt as a text file along with two generated texture samples.
        prompt_path = os.path.join(epochDir, prompt["default"][0] + ".txt")
        f = open(prompt_path, "a")

        # Save the seeds as two text files along with two generated texture samples.
        seed0_path = os.path.join(sample0_path,  f"{seed[0]}.txt")        
        f = open(seed0_path, "a")        
        f.close()

        print(f"[INFO] save models to {obj0}.")

        return obj0

    def accelerate_initialization(self):
        # Basic Accelerate and logging setup
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

        if texpref_config["resume_from"]:
            texpref_config["resume_from"] = os.path.normpath(os.path.expanduser(texpref_config["resume_from"]))
            if "checkpoint_" not in os.path.basename(texpref_config["resume_from"]):
                # get the most recent checkpoint in this directory
                checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(texpref_config["resume_from"])))
                if len(checkpoints) == 0:
                    raise ValueError("No checkpoints found in" + texpref_config["resume_from"])
                texpref_config["resume_from"] = os.path.join(
                    texpref_config["resume_from"],
                    sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
                )

        # When training a PyTorch model with Hugging Face Accelerate, you may often want to save and continue a state of training.
        # Doing so requires saving and loading the model, optimizer, RNG generators, and the GradScaler.
        # Inside Hugging Face Accelerate are two convenience functions to achieve this quickly:
        #   1- Use save_state() for saving everything mentioned above to a folder location
        #   2- Use load_state() for loading everything stored from an earlier save_state
        # To further customize where and how states are saved through save_state() the ProjectConfiguration class can be used.
        # For example if "automatic_checkpoint_naming" is enabled each saved checkpoint will be located then at Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}.
        # It should be noted that the expectation is that those states come from the same training script, they should not be from two separate scripts.
        # [Reference: https://huggingface.co/docs/accelerate/en/usage_guides/checkpoint]
        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(texpref_config["logdir"], unique_id),
            automatic_checkpoint_naming=True,
            total_limit = texpref_config["num_checkpoint_limit"],
        )


        # Gradient accumulation is a technique where you can train on
        # bigger batch sizes than your machine would normally be able to fit into memory.
        # This is done by accumulating gradients over several batches, 
        # and only stepping the optimizer after a certain number of batches have been performed.
        # While technically standard gradient accumulation code would work fine in a distributed setup,
        # it is not the most efficient method for doing so and you may experience considerable slowdowns!
        # All that is left now is to let Hugging Face Accelerate handle the gradient accumulation for us.
        # To do so you should pass in a "gradient_accumulation_steps" parameter to Accelerator, 
        # dictating the number of steps to perform before each call to step() and how to automatically
        # adjust the loss during the call to backward(). From here you can use the accumulate() context
        # manager from inside your training loop to automatically perform the gradient accumulation for you!
        # You just need to wrap it around the entire training part of the code.
        # [Reference: https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation]        
        accelerator = Accelerator(
            # log_with="wandb",                 # Uncomment this to enable monitoring and logging
            mixed_precision = texpref_config["mixed_precision"],
            project_config=accelerator_config,            
                gradient_accumulation_steps = texpref_config["train_gradient_accumulation_steps"], #4
        )

        if accelerator.is_main_process:
            accelerator.init_trackers(project_name="texture-dpo", config = texpref_config,
                                    init_kwargs={"wandb": {"name": texpref_config["run_name"]}})


        return accelerator

    def sd_initialization(self, accelerator: Accelerator):
        # Load the stable diffusion model
        from diffusers_patch.sd_utils import StableDiffusion
        self.guidance = StableDiffusion(self.device, 
                                        control_mode = texgen_config["control_mode"], 
                                        model_key = common_config["model_key"])

        # Whether or not to use xFormers to reduce memory usage.
        if texpref_config["use_xformers"]:
            self.guidance.pipe.enable_xformers_memory_efficient_attention()

        # Freeze parameters of models to save more memory
        self.guidance.pipe.vae.requires_grad_(False)
        self.guidance.pipe.text_encoder.requires_grad_(False)
        self.guidance.pipe.unet.requires_grad_(not texpref_config["use_lora"])

        # Whether or not to use the gradient checkpointing to save more memory
        # This comes at the cost of more computation time as it recomputes the
        # intermediates tensors/activations instead of storing them into the memory
        # during the backward pass.
        if texpref_config["train_activation_checkpoint"]:
            self.guidance.pipe.unet.enable_gradient_checkpointing()

        
        # The "pipeline.safety_checker" in Stable Diffusion pipelines (or other diffusion-based pipelines) 
        # refers to a component responsible for ensuring that generated images meet certain safety criteria and
        # to identify and filter out potentially NSFW (Not Safe For Work) content from generated images.  
        # This is particularly important when models are deployed to prevent the generation of harmful, explicit,
        # or undesirable content. Setting "pipeline.safety_checker = None" disables this functionality.
        # [Reference: https://huggingface.co/docs/diffusers/v0.6.0/en/api/pipelines/stable_diffusion?utm_source=chatgpt.com#diffusers.StableDiffusionPipeline.safety_checker]
        self.guidance.pipe.safety_checker = None

        # Make the progress bar nicer
        self.guidance.pipe.set_progress_bar_config(
            position=1,
            disable=not accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # Move unet, vae and text_encoder to device (GPU) and 
        # cast their parameters to the proper dtype
        self.guidance.pipe.vae.to(accelerator.device, dtype=self.guidance.dtype)
        self.guidance.pipe.text_encoder.to(accelerator.device, dtype=self.guidance.dtype)
        self.guidance.pipe.unet.to(accelerator.device, dtype=self.guidance.dtype)
        
        # Copy.deepcopy() creates a copy of the original variable
        # with a completely different location at memory.
        # Here the "ref" is a copy of the UNet existing in the stable diffusion model,
        # referring to as the refernce model.
        ref =  copy.deepcopy(self.guidance.pipe.unet)
        for param in ref.parameters():
            param.requires_grad = False # We keep the parameters of the reference model unchanged during the fine-tuning process.
                                        # TODO: During the training process, check whether these actually remain unchanged.

        
        if texpref_config["prompt_tuning"]: # If we use Prompt-Tuning, set correct learnable tokens/parameters for fine-tuning
            # Define the number of learnable tokens and their dimensionality
            num_learnable_tokens = 5
            hidden_size = self.guidance.pipe.text_encoder.config.hidden_size

            # Create learnable prompt embeddings
            trainable_layers = torch.nn.Parameter(torch.randn(num_learnable_tokens, hidden_size), requires_grad=True)
            # This was added to make the learnable prompts consistent with the precision of other parts of the model
            # trainable_layers = trainable_layers.clone().detach().requires_grad_(True)

            # We will keep the rest of the model frozen
            for param in self.guidance.pipe.unet.parameters():
                param.requires_grad = False

        elif texpref_config["use_lora"]: # If we use LORA, set correct lora layers/parameters for fine-tuning
            # Now we will add new LoRA weights to the attention layers
            # It's important to realize here how many attention weights will be added and of which sizes
            # The sizes of the attention layers consist only of two different variables:
            # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
            # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

            # Let's first see how many attention processors we will have to set.
            # For Stable Diffusion, it should be equal to:
            # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
            # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
            # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
            # => 32 layers

            # [Reference: https://github.com/huggingface/diffusers/blob/2dad462d9bf9890df09bfb088bf0a446c6074bec/examples/research_projects/lora/train_text_to_image_lora.py#L520]
            lora_attn_procs = {}
            for name in self.guidance.pipe.unet.attn_processors.keys():
                cross_attention_dim = (
                    None if name.endswith("attn1.processor") else self.guidance.pipe.unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.guidance.pipe.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.guidance.pipe.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.guidance.pipe.unet.config.block_out_channels[block_id]

                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            self.guidance.pipe.unet.set_attn_processor(lora_attn_procs)
            trainable_layers = AttnProcsLayers(self.guidance.pipe.unet.attn_processors)
            
        else: # If we don't use LORA, use all layers in the UNet architecture for fine-tuning
            trainable_layers = self.guidance.pipe.unet

        return trainable_layers, ref
    
    def text_embed_initialization(self, prompt_fn, learnable_tokens=None):
        # Generate 8 prompts and their embeddings
        # Specifically, we are going to generate two prompts which are identical
        # and later we feed these two prompts to the stable diffusion model to
        # generate two different samples using different seed.
        # NOTE that this is done in a batch way, meaning that instead of generating
        # only 1 prompt for prompt1 and prompt2, we generate a batch (8) prompts for
        # prompt1 and then initialize the prompt2 with the prompt1.
        temp_prompt, prompt_metadata = zip(*[prompt_fn(**texpref_config["prompt_fn_kwargs"])])
        prompt = {}
        prompt_embed = {}
        prompt['default'] = temp_prompt


        if texpref_config["prompt_tuning"]:
            # Get the modified positive prompt embeddings
            input_ids = self.guidance.pipe.tokenizer(
                prompt['default'],
                padding="max_length",
                truncation=False,
                max_length=self.guidance.pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            prompt_embed['default'] = self.prepend_learnable_tokens(self.guidance.pipe, learnable_tokens, input_ids)
            prompt_embed['default'] = prompt_embed['default'] # This was added to make the learnable prompts consistent with the precision of other parts of the model

            # Get the modified negative prompt embeddings
            input_ids = self.guidance.pipe.tokenizer(
                [self.negative_prompt],
                padding="max_length",
                truncation=False,
                max_length=self.guidance.pipe.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            negative_prompt_embed = self.prepend_learnable_tokens(self.guidance.pipe, learnable_tokens, input_ids)
            negative_prompt_embed = negative_prompt_embed # This was added to make the learnable prompts consistent with the precision of other parts of the model

            # Make the positive and negative prompt embeddings differentiable lead nodes in our computation graph.
            prompt_embed['default'].requires_grad_(True)
            negative_prompt_embed.requires_grad_(True)

            
        else:
            prompt_embed['default'] = self.guidance.get_text_embeds(prompt['default'])
            negative_prompt_embed = self.guidance.get_text_embeds([self.negative_prompt])
            # Make the positive and negative prompt embeddings differentiable lead nodes in our computation graph.
            prompt_embed['default'].requires_grad_(True)
            negative_prompt_embed.requires_grad_(True)


        for d in ['front', 'side', 'back', 'top', 'bottom']:
            edited_prompt = tuple(texgen_config["posi_prompt"] + ', ' + x + f', {d} view' for x in prompt['default']) # This is how I added the positive_prompt to the prompt value
            prompt[d] = edited_prompt
            prompt_embed[d] = self.guidance.get_text_embeds(edited_prompt)
            prompt_embed[d].requires_grad_(True)


        sample_neg_prompt_embeds = negative_prompt_embed.repeat(1, 1, 1)
        train_neg_prompt_embeds = negative_prompt_embed.repeat(1, 1, 1)

        output = (prompt, prompt_embed, negative_prompt_embed, sample_neg_prompt_embeds, train_neg_prompt_embeds)
        return output

    def mesh_initialization(self):
        h = w = int(texgen_config["texture_size"]) # AHZ: Texture image resolution: 1024*1024

        # AHZ: Create some variables related to the 3D object with respect to the size of the texture image
        # To understand this part, read the section "3.2. Iterative Texture Synthesis" of the InteX paper (page 6)
        # From the page 6 of the paper:
        # "To begin with, the 3D shape is initialized with:
        # 1- an empty texture image T ∈ H×W×3,
        # 2- a weight image W ∈ H×W,
        # 3- and a view cosine cache image V ∈ H×W."

        ############################ Mesh 0 ############################
        self.albedo0 = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32) # Texture image
        self.cnt0 = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32) # weight image
        self.viewcos_cache0 = - torch.ones((h, w, 1), device=self.device, dtype=torch.float32) # view cosine cache image

        self.renderer0.mesh.albedo = self.albedo0
        self.renderer0.mesh.cnt = self.cnt0 
        self.renderer0.mesh.viewcos_cache = self.viewcos_cache0


        ############################ Mesh 1 ############################
        self.albedo1 = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32) # Texture image
        self.cnt1 = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32) # weight image
        self.viewcos_cache1 = - torch.ones((h, w, 1), device=self.device, dtype=torch.float32) # view cosine cache image

        self.renderer1.mesh.albedo = self.albedo1
        self.renderer1.mesh.cnt = self.cnt1
        self.renderer1.mesh.viewcos_cache = self.viewcos_cache1


        if texpref_config["kl_divergence"] == True:
            self.kl_divergence_squared = torch.zeros(1, device=self.device, dtype=torch.float32) # Texture image
    

    def camera_initialization_for_texture_generation(self):
        # Hard-coded 11 camera positions/orientations in the 3D space for rendering the
        # mesh object from 11 different viewpoints
        if texgen_config["camera_path_texture_generation"]  == 'default':
            camera_pos_verticals = [-15] * 8 + [-89.9, 89.9] + [45]
            camera_pos_horizontals = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif texgen_config["camera_path_texture_generation"]  == 'front':
            camera_pos_verticals = [0] * 8 + [-89.9, 89.9] + [45]
            camera_pos_horizontals = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif texgen_config["camera_path_texture_generation"] == 'top':
            camera_pos_verticals = [0, -45, 45, -89.9, 89.9] + [0] + [0] * 6
            camera_pos_horizontals = [0] * 5 + [180] + [45, -45, 90, -90, 135, -135]
        elif texgen_config["camera_path_texture_generation"] == 'side':
            camera_pos_verticals = [0, 0, 0, 0, 0] + [-45, 45, -89.9, 89.9] + [-45, 0]
            camera_pos_horizontals = [0, 45, -45, 90, -90] + [0, 0, 0, 0] + [180, 180]
        elif texgen_config["camera_path_texture_generation"] == 'top-back':
            camera_pos_verticals = [0, -45, -45,  0,   0, -89.9,  0,   0, 89.9,   0,    0]
            camera_pos_horizontals = [0, 180,   0, 45, -45,     0, 90, -90,    0, 135, -135]
        ################## The following camera trajectories are fine-tuned for the 3D balloon mesh objects ###########
        elif texgen_config["camera_path_texture_generation"]  == 'front_rabbit':
            camera_pos_verticals = [0] * 7 + [-79.9, 89.9]
            camera_pos_horizontals = [0, 45, -45, 90, -90, 135, -135] + [0, 0]
        elif texgen_config["camera_path_texture_generation"]  == 'front_balloon':
            camera_pos_verticals = [0] * 7 + [-79.9, 89.9]
            camera_pos_horizontals = [0, 45, -45, 90, -90, 135, -135] + [0, 0]
        elif texgen_config["camera_path_texture_generation"]  == 'front_balloon_asymmetric':
            camera_pos_verticals = [0] * 6 + [-89.9, 89.9]
            camera_pos_horizontals = [25, -75, 70, -90, 105, -95] + [0, 0]
        elif texgen_config["camera_path_texture_generation"]  == 'top_balloon':
            camera_pos_verticals = [0, -45, 45, -89.9, 89.9] + [0] + [0] * 4
            camera_pos_horizontals = [0] * 5 + [180] + [45, -45, 135, -135]
        ################## The following two camera trajectories are fine-tuned for the 3D head mesh objects ###########
        elif texgen_config["camera_path_texture_generation"]  == 'front_short':
            camera_pos_verticals = [0, -35, -35, -35, -35, -35]
            camera_pos_horizontals = [0, 95, 135, 180, -135, -95]
        elif texgen_config["camera_path_texture_generation"]  == 'tilted_short':
            camera_pos_verticals = [0, -25, -25, -15, -25, -25]
            camera_pos_horizontals = [0, 95, 135, 180, -135, -95]
        elif texgen_config["camera_path_texture_generation"]  == 'front_3view':
            camera_pos_verticals = [0, -15, -15]
            camera_pos_horizontals = [0, 60, -60]
        elif texgen_config["camera_path_texture_generation"]  == 'front_4view':
            camera_pos_verticals = [0, 0, 0, 0]
            camera_pos_horizontals = [0, 60, -60, 180]
        elif texgen_config["camera_path_texture_generation"]  == 'tilted_4view':
            camera_pos_verticals = [-25, -15, -25, -15]
            camera_pos_horizontals = [0, 60, -60, 180]
        elif texgen_config["camera_path_texture_generation"]  == 'tilted':
            camera_pos_verticals = [-15, -35, -35, -35,  -15]
            camera_pos_horizontals = [60, 135, 180, -135, -60]
        elif texgen_config["camera_path_texture_generation"]  == 'tilted_2view':
            camera_pos_verticals = [0, 0]
            camera_pos_horizontals = [90, -90]
            # camera_pos_verticals = [0, 0]
            # camera_pos_horizontals = [60, -60]
        elif texgen_config["camera_path_texture_generation"] == 'debug':
            camera_pos_verticals = [0]
            camera_pos_horizontals = [0]
        else:
            raise NotImplementedError('camera path ' + texgen_config["camera_path"] + ' not implemented!')

        return camera_pos_horizontals,camera_pos_verticals

    def camera_initialization_for_reward_computation(self):
        # Hard-coded 11 camera positions/orientations in the 3D space for rendering the
        # mesh object from 11 different viewpoints
        if texgen_config["camera_path_reward_computation"]  == 'default':
            camera_pos_verticals = [-15] * 8 + [-89.9, 89.9] + [45]
            camera_pos_horizontals = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif texgen_config["camera_path_reward_computation"]  == 'front':
            camera_pos_verticals = [0] * 8 + [-89.9, 89.9] + [45]
            camera_pos_horizontals = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif texgen_config["camera_path_reward_computation"] == 'top':
            camera_pos_verticals = [0, -45, 45, -89.9, 89.9] + [0] + [0] * 6
            camera_pos_horizontals = [0] * 5 + [180] + [45, -45, 90, -90, 135, -135]
        elif texgen_config["camera_path_reward_computation"] == 'side':
            camera_pos_verticals = [0, 0, 0, 0, 0] + [-45, 45, -89.9, 89.9] + [-45, 0]
            camera_pos_horizontals = [0, 45, -45, 90, -90] + [0, 0, 0, 0] + [180, 180]
        elif texgen_config["camera_path_reward_computation"] == 'top-back':
            camera_pos_verticals = [0, -45, -45,  0,   0, -89.9,  0,   0, 89.9,   0,    0]
            camera_pos_horizontals = [0, 180,   0, 45, -45,     0, 90, -90,    0, 135, -135]
        ################## The following two camera trajectories are fine-tuned for the 3D head mesh objects ###########
        elif texgen_config["camera_path_reward_computation"]  == 'front_short':
            camera_pos_verticals = [0, -35, -35, -35, -35, -35]
            camera_pos_horizontals = [0, 95, 135, 180, -135, -95]
        elif texgen_config["camera_path_reward_computation"]  == 'tilted_short':
            camera_pos_verticals = [0, -25, -25, -15, -25, -25]
            camera_pos_horizontals = [0, 95, 135, 180, -135, -95]
        elif texgen_config["camera_path_reward_computation"]  == 'front_3view':
            camera_pos_verticals = [0, -15, -15]
            camera_pos_horizontals = [0, 60, -60]
        elif texgen_config["camera_path_reward_computation"]  == 'tilted_4view':
            camera_pos_verticals = [-25, -15, -25, -15]
            camera_pos_horizontals = [0, 60, -60, 180]
        elif texgen_config["camera_path_reward_computation"]  == 'tilted':
            camera_pos_verticals = [-15, -35, -35, -35,  -15]
            camera_pos_horizontals = [65, 135, 180, -135, -65]
        elif texgen_config["camera_path_reward_computation"] == 'debug':
            camera_pos_verticals = [0]
            camera_pos_horizontals = [0]
        else:
            raise NotImplementedError('camera path ' + texgen_config["camera_path"] + ' not implemented!')

        return camera_pos_horizontals,camera_pos_verticals

    def render_and_compute_reward_inference(self, reward_function, pose, prompts, sampleName:str):
        with torch.no_grad():
            h = w = int(texgen_config["texture_size"]) ### generated texture image resolution
            H = W = int(texgen_config["render_resolution"]) ### camera rendering resolution (choose from [512, 1024, 2048, 4096], larger than 512 will incur Real-ESRGAN super-resolution)

            # From the InteX paper: (page 6, section 3.2 Iterative Texture Synthesis)
            # "Rendering: Firstly, we render the 3D shape with the current texture image
            # to obtain the necessary image buffers, including:
            # 1- an RGB image with inpainting mask
            # 2- a depth map,
            # 3- a normal map,
            # 4- UV coordinates."
            
            ################################################# RENDERING ##########################################
            # "pose" is the camera position matrix in the world space.
            # "self.cam.perspective" is the projection (perspective)         

            if(sampleName == 'sample0'):
                out = self.renderer0.render(pose, self.cam.perspective, H, W)
                unwrapped_texture = self.renderer0.mesh.albedo
            
            unwrapped_texture_reward = self.renderer_reward.mesh.albedo

            # valid crop region with fixed aspect ratio
            valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
            min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
            min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()

            size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
            h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
            w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

            min_h = int(h_start)
            min_w = int(w_start)
            max_h = int(min_h + size)
            max_w = int(min_w + size)

            # crop region is outside rendered image: do not crop at all.
            if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
                min_h = 0
                min_w = 0
                max_h = H
                max_w = W

            def _zoom(x, mode='bilinear', size=(H, W)):
                return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)

            image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]

            # kiui.vis.plot_image(image)
            # kiui.vis.plot_image(unwrapped_texture)
            # kiui.vis.plot_image(unwrapped_texture_reward)

            if texpref_config["reward_fn"] == "curvature_score_mse":
                reward = reward_function(unwrapped_texture, unwrapped_texture_reward, prompts, None)[0]
            else:
                reward = reward_function(image, prompts, None)[0]
            
            return reward

    def render_and_compute_reward_train(self, unwrapped_texture, reward_function, pose, prompts):
        with torch.enable_grad():
            h = w = int(texgen_config["texture_size"]) ### generated texture image resolution
            H = W = int(texgen_config["render_resolution"]) ### camera rendering resolution (choose from [512, 1024, 2048, 4096], larger than 512 will incur Real-ESRGAN super-resolution)

            # From the InteX paper: (page 6, section 3.2 Iterative Texture Synthesis)
            # "Rendering: Firstly, we render the 3D shape with the current texture image
            # to obtain the necessary image buffers, including:
            # 1- an RGB image with inpainting mask
            # 2- a depth map,
            # 3- a normal map,
            # 4- UV coordinates."

            ################################################# RENDERING ##########################################
            out = self.renderer0.render2(unwrapped_texture, pose, self.cam.perspective, H, W)


            # valid crop region with fixed aspect ratio
            valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
            min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
            min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()

            size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
            h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
            w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

            min_h = int(h_start)
            min_w = int(w_start)
            max_h = int(min_h + size)
            max_w = int(min_w + size)

            # crop region is outside rendered image: do not crop at all.
            if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
                min_h = 0
                min_w = 0
                max_h = H
                max_w = W

            def _zoom(x, mode='bilinear', size=(H, W)):
                return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)

            image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]


            reward = reward_function(image, prompts, None)[0]


            return reward


    def all_initialization(self):
        accelerator = self.accelerate_initialization()

        # Properly loading the diffusion model and the controlnet model.
        trainable_layers, ref = self.sd_initialization(accelerator)

        # Set up diffusers-friendly checkpoint saving with Accelerate
        def save_model_hook(models, weights, output_dir):
            assert len(models) == 1
            if texpref_config["use_lora"] and isinstance(models[0], AttnProcsLayers):
                self.guidance.pipe.unet.save_attn_procs(output_dir)
            elif not texpref_config["use_lora"] and isinstance(models[0], UNet2DConditionModel):
                models[0].save_pretrained(os.path.join(output_dir, "unet"))
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

        def load_model_hook(models, input_dir):
            assert len(models) == 1
            if texpref_config["use_lora"] and isinstance(models[0], AttnProcsLayers):
                tmp_unet = UNet2DConditionModel.from_pretrained(
                    common_config["model_key"], revision=common_config["pretrained_revision"], subfolder="unet"
                )
                tmp_unet.load_attn_procs(input_dir)
                models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
                del tmp_unet
            elif not texpref_config["use_lora"] and isinstance(models[0], UNet2DConditionModel):
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                models[0].register_to_config(**load_model.config)
                models[0].load_state_dict(load_model.state_dict())
                del load_model
            else:
                raise ValueError(f"Unknown model type {type(models[0])}")
            models.pop()  # ensures that accelerate doesn't try to handle loading of the model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if texpref_config["allow_tf32"]:
            # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
            # in PyTorch 1.12 and later.
            torch.backends.cuda.matmul.allow_tf32 = True
            # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
            torch.backends.cudnn.allow_tf32 = True


        # Hard-coded camera positions/orientations texture generation
        camera_pos_horizontals_texture_generation, camera_pos_verticals_texture_generation = self.camera_initialization_for_texture_generation()
        # Hard-coded camera positions/orientations for reward computation
        camera_pos_horizontals_reward_computation, camera_pos_verticals_reward_computation = self.camera_initialization_for_reward_computation()

        if texpref_config["learn_camera_viewpoints"] == True:
            # Make the camera viewpoints learnable parameters
            # use float32 so they can interact with your renderer's floats
            # camera_pos_horizontals_texture_generation = torch.tensor(camera_pos_horizontals_texture_generation, dtype=torch.float32, device='cuda')
            camera_pos_verticals_texture_generation = torch.tensor(camera_pos_verticals_texture_generation, dtype=torch.float32, device='cuda')

            # camera_pos_horizontals_texture_generation = torch.nn.Parameter(camera_pos_horizontals_texture_generation, requires_grad=True)
            camera_pos_verticals_texture_generation   = torch.nn.Parameter(camera_pos_verticals_texture_generation, requires_grad=True)

        # Initialize the AdamW optimizer
        if texpref_config["prompt_tuning"]:
            optimizer = torch.optim.AdamW(
                [trainable_layers],
                lr=texpref_config["train_learning_rate"],
                betas=(texpref_config["train_adam_beta1"], texpref_config["train_adam_beta2"]),
                weight_decay=texpref_config["train_adam_weight_decay"],
                eps=texpref_config["train_adam_epsilon"],
            )
        else:
            if texpref_config["learn_camera_viewpoints"] == True:
                optimizer = torch.optim.AdamW(
                    [
                        # group 1: LoRA / SD parameters
                        {
                            "params": trainable_layers.parameters(),
                            "lr": texpref_config["train_learning_rate"],
                        },
                        # group 2: camera viewpoint parameters
                        {
                            # "params": [camera_pos_horizontals_texture_generation, camera_pos_verticals_texture_generation],
                            # "params": [camera_pos_horizontals_texture_generation],
                            "params": [camera_pos_verticals_texture_generation],
                            "lr": texpref_config["camera_learning_rate"],
                        },
                    ],
                    betas=(
                        texpref_config["train_adam_beta1"],
                        texpref_config["train_adam_beta2"],
                    ),
                    weight_decay=texpref_config["train_adam_weight_decay"],
                    eps=texpref_config["train_adam_epsilon"],
                )
            else: # Only learn the SD model's parameters
                optimizer = torch.optim.AdamW(
                    trainable_layers.parameters(),
                    lr=texpref_config["train_learning_rate"],
                    betas=(texpref_config["train_adam_beta1"], texpref_config["train_adam_beta2"]),
                    weight_decay=texpref_config["train_adam_weight_decay"],
                    eps=texpref_config["train_adam_epsilon"],
                )
                

        # Read prompts and reward function
        # getattr(object, name): Return the value of the named attribute
        # of object. "name" must be a string. If the string is the name of
        # one of the object’s attributes, the result is the value of that
        # attribute. For example, getattr(x, 'foobar') is equivalent to x.foobar.
        # Refernece: https://docs.python.org/3/library/functions.html#getattr
        prompt_fn = getattr(prompts, texpref_config["prompt_fn"]) # This is equivalent to "prompt_fn = prompts.simple_animals()"
        reward_fn = getattr(rewards, texpref_config["reward_fn"])() # This is equivalent to "reward_fn = rewards.jpeg_compressibility()"

        # For some reason, autocast is necessary for non-lora training 
        # but for lora training it isn't necessary and it uses more memory
        autocast = contextlib.nullcontext if texpref_config["use_lora"] else accelerator.autocast

        # Prepare all objects passed in `args` for distributed training 
        # and mixed precision, then return them in the same order.        
        trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

        # Continue from a checkpoint
        if texpref_config["resume_from"]:
            print("##################################################################################################")
            print("Starting from a checkpoint (pre-trained model)")
            print("##################################################################################################")
            accelerator.load_state(texpref_config["resume_from"])

        # Store the models with the initial weights into the memory storage
        self.save_initial_weights(trainable_layers)

        output = (ref, prompt_fn, reward_fn,
                  camera_pos_horizontals_texture_generation, camera_pos_verticals_texture_generation,
                  camera_pos_horizontals_reward_computation, camera_pos_verticals_reward_computation,
                  autocast, accelerator, trainable_layers, optimizer)
        return output

    def isFineTuningCorrect(self):
        if texpref_config["prompt_tuning"]:
            # Load the learnable tokens from before and after fine-tuning
            learnable_tokens_before = torch.load('previous_finetuned_weights.pth')
            learnable_tokens_after = torch.load('latest_finetuned__weights.pth')

            # Check if every parameter has changed
            if torch.equal(learnable_tokens_before, learnable_tokens_after):
                print("No parameters of the learnable tokens have changed.")
            else:
                print("Some or all parameters of the learnable tokens have changed.")

        else:
            # Load and compare
            initial_weights = torch.load('previous_finetuned_weights.pth')
            fine_tuned_weights = torch.load('latest_finetuned_weights.pth')

            for key in initial_weights:
                if not torch.equal(initial_weights[key], fine_tuned_weights[key]):
                    print(f"Parameter {key} has been updated.")
                else:
                    print(f"Parameter {key} has not been updated.")

    def save_initial_weights(self, trainable_layers):
        if texpref_config["prompt_tuning"]:
            torch.save(trainable_layers, 'initial_weights.pth')
            torch.save(trainable_layers, 'previous_finetuned_weights.pth')
        elif texpref_config["use_lora"]:
            torch.save(trainable_layers.state_dict(), 'initial_weights.pth')
            torch.save(trainable_layers.state_dict(), 'previous_finetuned_weights.pth')
        else:
            torch.save(self.guidance.pipe.unet.state_dict(), 'initial_weights.pth')
            torch.save(self.guidance.pipe.unet.state_dict(), 'previous_finetuned_weights.pth')


    @torch.enable_grad()
    def multi_view_forward(self, reference_unet_model, epoch, seed, camera_pos_verticals_texture_generation, camera_pos_horizontals_texture_generation,
                           camera_pos_horizontals_reward_computation, camera_pos_verticals_reward_computation,
                           prompt, prompt_embed, negative_prompt_embed, reward_fn):

        import gc

        # Step 0: Initialize mesh state
        h = w = int(texgen_config["texture_size"]) ### generated texture image resolution
        mesh_state = {
            'albedo0': torch.zeros((h, w, 3), device=self.device, dtype=torch.float32).requires_grad_(True),
            'cnt0': torch.zeros((h, w, 1), device=self.device, dtype=torch.float32).requires_grad_(True),
            'viewcos_cache': -torch.ones((h, w, 1), device=self.device, dtype=torch.float32).requires_grad_(True)
        }
        # Set the renderer's mesh to use our mesh state
        self.renderer0.mesh.albedo = mesh_state['albedo0']
        self.renderer0.mesh.cnt = mesh_state['cnt0']
        self.renderer0.mesh.viewcos_cache = mesh_state['viewcos_cache']


        total_loss = 0.0
        total_reward = 0.0
        num_views_texture_generation = len(camera_pos_verticals_texture_generation)
        num_views_reward_computation = len(camera_pos_verticals_reward_computation)

        ################################################################################################################
        ################### Step 1 - Texture Generation: Generate the texture for all viewpoints #######################
        ################################################################################################################
        for current_view_num, (ver, hor) in enumerate(tqdm(zip(camera_pos_verticals_texture_generation, camera_pos_horizontals_texture_generation), total=len(camera_pos_verticals_texture_generation))):            
            # [Reference: https://pytorch.org/docs/stable/checkpoint.html]
            # Activation checkpointing is a technique that trades compute for memory.
            # Instead of keeping tensors needed for backward alive until they are used
            # in gradient computation during backward, forward computation in checkpointed
            # regions omits saving tensors for backward and recomputes them during the 
            # backward pass. Activation checkpointing can be applied to any part of a model.

            # There are currently two checkpointing implementations available, determined by
            # the use_reentrant parameter. It is recommended that you use use_reentrant=False.

            # The reentrant variant of checkpoint (use_reentrant=True) and the non-reentrant variant
            # of checkpoint (use_reentrant=False) differ in the following ways:

            # 1- Non-reentrant checkpoint stops recomputation as soon as all needed intermediate activations
            # have been recomputed. This feature is enabled by default, but can be disabled with set_checkpoint_early_stop().
            # Reentrant checkpoint always recomputes function in its entirety during the backward pass.

            # 2- The reentrant variant does not record the autograd graph during the forward pass, as it
            # runs with the forward pass under torch.no_grad(). The non-reentrant version does record the
            # autograd graph, allowing one to perform backward on the graph within checkpointed regions.

            # 3- The reentrant checkpoint only supports the torch.autograd.backward() API for the backward
            # pass without its inputs argument, while the non-reentrant version supports all ways of performing
            # the backward pass.

            # 4- At least one input and output must have requires_grad=True for the reentrant variant. If this
            # condition is unmet, the checkpointed part of the model will not have gradients. The non-reentrant
            # version does not have this requirement.

            # 5- The reentrant version does not consider tensors in nested structures (e.g., custom objects, lists, dicts, etc)
            # as participating in autograd, while the non-reentrant version does.

            # 6- The reentrant checkpoint does not support checkpointed regions with detached tensors from the
            # computational graph, whereas the non-reentrant version does. For the reentrant variant, if the checkpointed
            # segment contains tensors detached using detach() or with torch.no_grad(), the backward pass will raise an error.
            # This is because checkpoint makes all the outputs require gradients and this causes issues when a tensor is defined
            # to have no gradient in the model. To avoid this, detach the tensors outside of the checkpoint function.

            # Two important arguments of the checkpoint function:

            # 1- use_reentrant (bool) – specify whether to use the activation checkpoint variant that requires reentrant autograd.
            # This parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed.
            # If use_reentrant=False, checkpoint will use an implementation that does not require reentrant autograd. 
            # This allows checkpoint to support additional functionality, such as working as expected with torch.autograd.grad and
            # support for keyword arguments input into the checkpointed function.

            # 2- preserve_rng_state (bool, optional) – Omit stashing and restoring the RNG state during each checkpoint. 
            # Note that under torch.compile, this flag doesn’t take effect and we always preserve RNG state. Default: True

            # 3- debug (bool, optional) – If True, error messages will also include a trace of the operators ran during the
            # original forward computation as well as the recomputation. This argument is only supported if use_reentrant=False.

            # Wrap the single-view processing with checkpoint. All arguments must be of type torch.tensors.
            mesh_state = torch.utils.checkpoint.checkpoint(
                self.process_viewpoint, 
                reference_unet_model,seed, ver, hor, prompt_embed, 
                negative_prompt_embed, mesh_state, num_views_texture_generation,
                use_reentrant=False
                # preserve_rng_state=True
            )
            
        ### End of the texture generation loop

        #######################################################################################################################################
        ################### Step 2 - Texture Preference Learning: Render the object from different viewpoints to compute the reward ####################
        #######################################################################################################################################        
        if texpref_config["reward_fn"] == "curvature_score_mse":
            unwrapped_texture_reward = self.renderer_reward.mesh.albedo
            # kiui.vis.plot_image(unwrapped_texture_reward.detach().cpu().numpy())
            # kiui.vis.plot_image(self.albedo0.detach().cpu().numpy())
            total_reward = reward_fn(self.albedo0, unwrapped_texture_reward, prompt, None)[0]            
            total_loss = -total_reward
        elif (texpref_config["reward_fn"] == "curvature_score_smooth") or (texpref_config["reward_fn"] == "curvature_score_sharp"):
            total_reward = reward_fn(self.albedo0, self.curvature_in_2d, prompt, None)[0]
            total_loss = -total_reward
        elif (texpref_config["reward_fn"] == "mean_curvature_texture_alignment_reward"):
            total_reward = reward_fn(self.albedo0, self.curvature_in_2d, self.direction_in_2d, prompt, None)[0]
            total_loss = -total_reward
        elif (texpref_config["reward_fn"] == "minimum_curvature_texture_alignment_reward"):
            total_reward = reward_fn(epoch, self.albedo0, self.UV, self.curvature_in_2d, prompt, None)[0]
            total_loss = -total_reward
        elif texpref_config["reward_fn"] == "symmetry_score":
            patch_size = 3
            total_reward = reward_fn(self.albedo0, self.pixel_uv_orig, self.pixel_uv_mirror, patch_size, prompt, None)[0]
            total_loss = -total_reward
        else:
            for current_view_num, (ver, hor) in enumerate(tqdm(zip(camera_pos_verticals_reward_computation, camera_pos_horizontals_reward_computation), total=num_views_reward_computation)):
                # AHZ: Put the camera in different locations and compute the camera position matrix in the world space
                viewpoint = orbit_camera_differentiable(ver, hor, self.cam.radius)

                # Render the object into an image from a specific viewpoint and generate the texture using the Intex method                
                reward = self.render_and_compute_reward_train(self.albedo0, reward_fn, viewpoint, prompt)
                loss = -reward

                # Accumulate the reward across viewpoints            
                total_reward += reward
                total_loss += loss

                # This was added to resolve the memory leak issue that happened
                # when running the experiments using the aesthetic reward.
                del reward
                gc.collect()
                torch.cuda.empty_cache()

            total_reward = total_reward / num_views_reward_computation
            total_loss = total_loss / num_views_reward_computation


        if texpref_config["kl_divergence"] == True:
            self.kl_divergence_squared = self.kl_divergence_squared / num_views_texture_generation
            total_loss = total_loss + (texpref_config["kl_divergence_beta"] * self.kl_divergence_squared)


        print("Reward: ", total_reward)
        print("Loss: ", total_loss)
        

        return total_loss, total_reward


    # Define a function that processes a single viewpoint.
    # Note: All inputs to checkpoint must be tensors.
    def process_viewpoint(self, reference_unet_model, seed, ver, hor, prompt_embed, negative_prompt_embed, mesh_state, num_views):

        # AHZ: Put the camera in different locations and compute the camera position matrix in the world
        # NOTE: "ver" and "hor" are leaf variables in our computation graph.
        viewpoint = orbit_camera_differentiable(ver, hor, self.cam.radius)

        # pose to view dir
        if texgen_config["text_dir"]:
            v, h, _ = undo_orbit_camera_differentiable(viewpoint)
            if v <= -60: d = 'top'
            elif v >= 60: d = 'bottom'
            else:
                if abs(h) < 30: d = 'front'
                elif abs(h) < 90: d = 'side'
                else: d = 'back'
            prompt_embeds1 = prompt_embed[d]
        else:
            prompt_embeds1 = prompt_embed["default"]

        # Render the object into an image from a specific viewpoint and generate the texture using the Intex method
        # IMPORTANT: The renderer uses mesh_state and updates it.
        result0 = self.inpaint_view_train(reference_unet_model, viewpoint, negative_prompt_embed, prompt_embeds1, mesh_state, num_views, seed=seed[0])
        # result0 = self.inpaint_view_train(viewpoint, negative_prompt_embed, prompt_embeds1, mesh_state, num_views, seed=0)
        if result0 is None:
            raise ValueError("inpaint_view_train returned None for viewpoint")
        rendering, latents, noise_preds0, unwrapped_texture, texture_reward, unwrapped_texture_reward, updated_mesh_state = result0

        return updated_mesh_state

    @torch.no_grad
    def inpaint_view_train(self, reference_unet_model, pose, negative_prompt_embed, positive_prompt_embed, mesh_state, num_views, seed):
        """
        A differentiable, input-dependent inpainting function.
        
        Args:
            pose: Camera pose (tensor) used for rendering.
            negative_prompt_embed: Negative prompt embeddings.
            positive_prompt_embed: Positive prompt embeddings.
            mesh_state: A dictionary with keys 'albedo0', 'cnt0', and 'viewcos_cache' representing the current mesh state.
            force_inpaint (bool): If True, forces inpainting even if the region appears textured.
            seed (int): Random seed for rendering.
            
        Returns:
            sample: The output sample (texture and other outputs).
            updated_mesh_state: The updated mesh state after inpainting.
        """

        h = w = int(texgen_config["texture_size"]) ### generated texture image resolution
        H = W = int(texgen_config["render_resolution"]) ### camera rendering resolution (choose from [512, 1024, 2048, 4096], larger than 512 will incur Real-ESRGAN super-resolution)

        # From the InteX paper: (page 6, section 3.2 Iterative Texture Synthesis)
        # "Rendering: Firstly, we render the 3D shape with the current texture image
        # to obtain the necessary image buffers, including:
        # 1- an RGB image with inpainting mask
        # 2- a depth map,
        # 3- a normal map,
        # 4- UV coordinates."


        ################################################# RENDERING ##########################################
        # "pose" is the camera position matrix in the world space.
        # "self.cam.perspective" is the projection (perspective)

        out_reward = self.renderer_reward.render(pose, self.cam.perspective, H, W)

        with torch.enable_grad():
            # out = self.renderer0.render(pose, self.cam.perspective, H, W)
            out = self.renderer0.render1(mesh_state, pose, self.cam.perspective, H, W)


        # valid crop region with fixed aspect ratio
        valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
        min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
        min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()

        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        min_h = int(h_start)
        min_w = int(w_start)
        max_h = int(min_h + size)
        max_w = int(min_w + size)

        # crop region is outside rendered image: do not crop at all.
        if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
            min_h = 0
            min_w = 0
            max_h = H
            max_w = W

        def _zoom(x, mode='bilinear', size=(H, W)):
            return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)

        with torch.enable_grad():
            image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]


        # We don't need to track the gradients of the ground truth image.
        image_reward = _zoom(out_reward['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]

 
        # Leverage the cosine map and the weight image to partition the image into a trimap denoted as M_generate, M_refine, and M_keep.        
        with torch.enable_grad():
            # 1- M_generate -> use a sigmoid to produce a soft mask:
            # Here, when "cnt_zoom" is below 0.1, sigmoid(k*(cnt_zoom - 0.1))
            # is near 0 so "mask_generate" is near 1, and vice versa.
            k = 100.0  # A large constant for steepness
            cnt_zoom = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='bilinear')  # Use bilinear for smoothness
            mask_generate = 1 - torch.sigmoid(k * (cnt_zoom - 0.1))

            # 2- M_refine
            viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
            viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
            # Compute soft differences. 
            # For example, I define a soft comparison between viewcos_old and viewcos as:
            viewcos_diff = torch.sigmoid(k * (viewcos - viewcos_old))

            # Then, if we want to consider regions where viewcos_old is less than viewcos, viewcos_diff will be high.
            # Next, instead of the bitwise NOT on mask_generate, we use:
            mask_generate_soft = 1 - mask_generate  # Already soft
            mask_refine = viewcos_diff * mask_generate_soft


            # 3- M_keep
            mask_keep = (1 - mask_generate) * (1 - mask_refine)

        mask_generate = mask_generate.float()
        mask_refine = mask_refine.float()
        mask_keep = mask_keep.float()

        mask_generate_blur = mask_generate


        if not (mask_generate > 0.5).any():
            return None, mesh_state


        control_images = {}

        with torch.enable_grad():
            # construct normal control
            if 'normal' in texgen_config["control_mode"]:
                rot_normal = out['rot_normal'] # [H, W, 3]
                rot_normal[..., 0] *= -1 # align with normalbae: blue = front, red = left, green = top
                control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512)) # [1, 3, H, W]
            
            # construct depth control
            if 'depth' in texgen_config["control_mode"]:
                depth = out['depth']
                control_images['depth'] = _zoom(depth.view(1, 1, H, W), size=(512, 512)).repeat(1, 3, 1, 1) # [1, 3, H, W]

            # construct ip2p control
            if 'ip2p' in texgen_config["control_mode"]:
                ori_image = _zoom(out['ori_image'].permute(2, 0, 1).unsqueeze(0).contiguous(), size=(512, 512)) # [1, 3, H, W]
                control_images['ip2p'] = ori_image

            # construct inpaint control
            if 'inpaint' in texgen_config["control_mode"]:
                image_generate = image.clone()
                image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region
                image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
                control_images['inpaint'] = image_generate

                # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
                latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                control_images['latents_mask'] = latents_mask
                control_images['latents_mask_refine'] = latents_mask_refine
                control_images['latents_mask_keep'] = latents_mask_keep
                control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]


            # construct depth-aware-inpaint control
            if 'depth_inpaint' in texgen_config["control_mode"]:
                # "image" is the texture which was rendered by the diff-rasterizer previously
                # image.clone() returns a copy of "image". This function is differentiable, so gradients
                # will flow back from the result of this operation to input. To create a tensor without
                # an autograd relationship to input see detach().
                image_generate = image.clone()

                # This line just mask out those pixels in the original rendered image where needed to be textured
                image_generate[mask_keep.repeat(1, 3, 1, 1) < 0.5] = -1 # -1 is inpaint region
                # image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region

                # This line interpolates the 1024*1024 masked rendered image to 512*512
                image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)

                # This line only converts the depth map corresponding to a image rendered from a specific viewpoint into a depth map with the size of 512*512
                depth = _zoom(out['depth'].view(1, 1, H, W), size=(512, 512)).clamp(0, 1).repeat(1, 3, 1, 1) # [1, 3, H, W]

                # Concatenate the masked image and the depth map
                control_images['depth_inpaint'] = torch.cat([image_generate, depth], dim=1) # [1, 6, H, W]

                # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
                latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                control_images['latents_mask'] = latents_mask
                control_images['latents_mask_refine'] = latents_mask_refine
                control_images['latents_mask_keep'] = latents_mask_keep

                # image_fill = image.clone()
                # image_fill = dilate_image(image_fill, mask_generate_blur, iterations=int(H*0.2))

                control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]

        ########################################## SD and CONTROLNET MODEL ####################################
        rgbs_reward = image_reward
        with torch.enable_grad():
            if texpref_config["lora_scaling"] == True:
                rgbs, latents, noise_preds, KL_term = self.guidance.pipeline_with_logprob_train(
                    reference_unet_model=reference_unet_model,
                    prompt_embeds=positive_prompt_embed,
                    negative_prompt_embeds=negative_prompt_embed,
                    num_inference_steps=texpref_config["sample_num_steps"],
                    guidance_scale=texpref_config["sample_guidance_scale"],
                    eta=texpref_config["sample_eta"],
                    output_type="pt",
                    control_images=control_images,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    refine_strength=texgen_config["refine_strength"],
                    cross_attention_kwargs={"scale": texpref_config["lora_scale_alpha"]} # When we merge the LoRA weights with the frozen pretrained model
                                                        # weights, we can optionally adjust how much of the weights to merge
                                                        # with the scale parameter. So, a scale value of 0 is the same as not
                                                        # using your LoRA weights and we’re only using the base model weights,
                                                        # and a scale value of 1 means we’re only using the fully finetuned LoRA
                                                        # weights. Values between 0 and 1 interpolates between the two weights.
                                                        # [Reference: https://huggingface.co/docs/diffusers/v0.21.0/en/training/lora]
                )
            else:
                rgbs, latents, noise_preds, KL_term = self.guidance.pipeline_with_logprob_train(
                    reference_unet_model=reference_unet_model,
                    prompt_embeds=positive_prompt_embed,
                    negative_prompt_embeds=negative_prompt_embed,
                    num_inference_steps=texpref_config["sample_num_steps"],
                    guidance_scale=texpref_config["sample_guidance_scale"],
                    eta=texpref_config["sample_eta"],
                    output_type="pt",
                    control_images=control_images,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                    refine_strength=texgen_config["refine_strength"]
                )
        
        # A naive differentiable version of the super-resolution step we had before.
        # I checked several examples and it looks fine. But, to make sure, check this
        # function "kiui.sr.sr" for a more accurate re-implementation (TODO)
        # https://github.com/ashawkey/kiuikit/blob/73e34203fd3bd4b5979dc3a38057dff117b39521/kiui/sr.py#L533 
        with torch.enable_grad():
            if rgbs.shape[-1] != W or rgbs.shape[-2] != H:
                rgbs = F.interpolate(rgbs, size=(H, W), mode='bilinear', align_corners=False)

        sample = (rgbs, latents, noise_preds)
        ################################################ INPAINTING ##########################################
        with torch.enable_grad():
            # Apply mask to make sure non-inpaint region is not changed
            # The keep region remains strictly fixed through mask blending in the image space
            # See page 7, equation 4 to understand the following line
            rgbs = rgbs * (1 - mask_keep) + image * mask_keep
            # rgbs = rgbs * mask_generate_blur + image * (1 - mask_generate_blur)


        # Visualize the intermediate results
        if texgen_config["vis"]:
            if 'depth' in control_images:
                kiui.vis.plot_image(control_images['depth'])
            if 'normal' in control_images:
                kiui.vis.plot_image(control_images['normal'])
            if 'ip2p' in control_images:
                kiui.vis.plot_image(ori_image)
            # kiui.vis.plot_image(mask_generate)
            if 'inpaint' in control_images:
                kiui.vis.plot_image(control_images['inpaint'].clamp(0, 1))
                # kiui.vis.plot_image(control_images['inpaint_refine'].clamp(0, 1))
            if 'depth_inpaint' in control_images:
                # kiui.vis.plot_image(control_images['depth_inpaint'][:, :3].clamp(0, 1))
                # kiui.vis.plot_image(control_images['depth_inpaint'][:, 3:].clamp(0, 1))
                pass
            kiui.vis.plot_image(rgbs)
            kiui.vis.plot_image(rgbs_reward)

        # grid put
        with torch.enable_grad():
            # project-texture mask
            proj_mask = (out['alpha'] > 0) & (out['viewcos'] > texgen_config["cos_thresh"])  # [H, W, 1]
            # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
            # kiui.vis.plot_image(proj_mask.squeeze(-1).detach().cpu().numpy())
            proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
            uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')
            uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
            rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]



        ### NOTE: We don't need to track the gradients of all information related to the reward and ground truth
        proj_mask_reward = (out_reward['alpha'] > 0) & (out_reward['viewcos'] > texgen_config["cos_thresh"])  # [H, W, 1]
        # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
        # kiui.vis.plot_image(proj_mask.squeeze(-1).detach().cpu().numpy())
        proj_mask_reward = _zoom(proj_mask_reward.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        uvs_reward = _zoom(out_reward['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')
        uvs_reward = uvs_reward.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask_reward]

        rgbs_reward = rgbs_reward.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask_reward]

        ################################################# UPDATING #########################################
        with torch.enable_grad():
            # uvs is the tensor of shape [..., 2]
            # Create an index tensor for swapping: it must be on the same device as uvs.
            index = torch.tensor([1, 0], device=uvs.device)
            # Use torch.index_select along the last dimension to swap the channels.
            swapped_uvs = torch.index_select(uvs, dim=-1, index=index)

            cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, swapped_uvs * 2 - 1, rgbs, min_resolution=128, return_count=True)
            # cur_albedo, cur_cnt = linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, return_count=True)


        cur_albedo_reward, cur_cnt_reward = mipmap_linear_grid_put_2d(h, w, uvs_reward[..., [1, 0]] * 2 - 1, rgbs_reward, min_resolution=128, return_count=True)

        # Adding the unwrapped texture to the output
        sample = sample + (cur_albedo, image_reward, cur_albedo_reward,)


        with torch.enable_grad():

            # Instead of updating global variables, update the mesh_state dictionary.
            new_albedo = mesh_state['albedo0'].clone()
            new_cnt = mesh_state['cnt0'].clone()
            mask_update = cur_cnt.squeeze(-1) > 0
            new_albedo[mask_update] += cur_albedo[mask_update]
            new_cnt[mask_update] += cur_cnt[mask_update]


            # update viewcos cache
            viewcos = viewcos.view(-1, 1)[proj_mask]
            cur_viewcos = mipmap_linear_grid_put_2d(h, w, swapped_uvs * 2 - 1, viewcos, min_resolution=256)            
            new_viewcos_cache = torch.maximum(self.renderer0.mesh.viewcos_cache, cur_viewcos)


            updated_mesh_state = {
                'albedo0': new_albedo,
                'cnt0': new_cnt,
                'viewcos_cache': new_viewcos_cache
            }

        # This part is just to track whether we are in the actual
        # forward pass or in the "virtual" forward pass. The forward
        # pass is the one which the torch.utils.checkpoint.checkpoint
        # calls during the backward. So, we detect if we're in the first
        # forward pass (the one which is not in the backward), then we only
        # update the information related to the 3D object that we plan to save
        # after each epoch. This is the first solution came to my mind to prevent
        # the 3D object from untexturing during the virtual forward pass.       
        self.forwar_pass_visit_count += 1
        if(self.forwar_pass_visit_count <= num_views):
            # Update mesh texture for rendering
            mask = updated_mesh_state['cnt0'].squeeze(-1) > 0
            cur_albedo = updated_mesh_state['albedo0'].clone()
            cur_albedo[mask] /= updated_mesh_state['cnt0'][mask].repeat(1, 3)
            self.renderer0.mesh.albedo = cur_albedo
            self.renderer0.mesh.viewcos_cache = new_viewcos_cache
            self.renderer0.mesh.cnt = updated_mesh_state['cnt0']

        # We also update our global variable called "self.albedo0".
        # This is going to be used as input to the "Step 2: Texture Preference Learning"
        # of our pipeline. The Texture Preference Learning module, takes this "self.albedo0",
        # as input and and renders the 3D object + its texture and gives a scalar,
        # representing how good the texture is in terms of the reward function.
        with torch.enable_grad():
            mask = updated_mesh_state['cnt0'].squeeze(-1) > 0
            cur_albedo = updated_mesh_state['albedo0'].clone()
            cur_albedo[mask] /= updated_mesh_state['cnt0'][mask].repeat(1, 3)
            self.albedo0 = cur_albedo


            # Update the global variable called "self.kl_divergence_squared"
            if texpref_config["kl_divergence"] == True:
                self.kl_divergence_squared = self.kl_divergence_squared + KL_term

        sample = sample + (updated_mesh_state,)

        return sample

    def inpaint_view_inference(self, reference_model, pose, negative_prompt_embed, positive_prompt_embed, sampleName:str, seed=0):
        with torch.no_grad():
            h = w = int(texgen_config["texture_size"]) ### generated texture image resolution
            H = W = int(texgen_config["render_resolution"]) ### camera rendering resolution (choose from [512, 1024, 2048, 4096], larger than 512 will incur Real-ESRGAN super-resolution)

            # From the InteX paper: (page 6, section 3.2 Iterative Texture Synthesis)
            # "Rendering: Firstly, we render the 3D shape with the current texture image
            # to obtain the necessary image buffers, including:
            # 1- an RGB image with inpainting mask
            # 2- a depth map,
            # 3- a normal map,
            # 4- UV coordinates."


            ################################################# RENDERING ##########################################
            # "pose" is the camera position matrix in the world space.
            # "self.cam.perspective" is the projection (perspective)
            out_reward = self.renderer_reward.render(pose, self.cam.perspective, H, W)            

            if(sampleName == 'sample0'):
                out = self.renderer0.render(pose, self.cam.perspective, H, W)
            elif(sampleName == 'sample1'):
                out = self.renderer1.render(pose, self.cam.perspective, H, W)

            # valid crop region with fixed aspect ratio
            valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
            min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
            min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()

            size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
            h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
            w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

            min_h = int(h_start)
            min_w = int(w_start)
            max_h = int(min_h + size)
            max_w = int(min_w + size)

            # crop region is outside rendered image: do not crop at all.
            if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
                min_h = 0
                min_w = 0
                max_h = H
                max_w = W

            def _zoom(x, mode='bilinear', size=(H, W)):
                return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)

            image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]
            
            image_reward = _zoom(out_reward['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]


            # From the paper: (page 6, section 3.2 Iterative Texture Synthesis)
            # "Following [3,36], we leverage this cosine map and the weight image 
            # to partition the image into a trimap denoted as M_generate, M_refine, and M_keep."
            
            # 1- M_generate
            mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest') < 0.1 # [1, 1, H, W]

            # 2- M_refine
            viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
            viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
            mask_refine = ((viewcos_old < viewcos) & ~mask_generate)

            # 3- M_keep
            mask_keep = (~mask_generate & ~mask_refine)

            mask_generate = mask_generate.float()
            mask_refine = mask_refine.float()
            mask_keep = mask_keep.float()


            ################ Display the image buffers generated by the differentiable rasterizer ############
            # kiui.vis.plot_image(out['viewcos_cache'])
            # kiui.vis.plot_image(out['viewcos'])
            # print(mask_generate)
            # print(type(mask_generate))
            # print(mask_generate.shape)
            # kiui.vis.plot_image(mask_generate)
            # kiui.vis.plot_image(out['image'])
            # ahz_image =  mask_keep.squeeze(0).cpu().numpy().transpose(1, 2, 0)                
            # cv2.imshow("image", ahz_image)
            # cv2.waitKey()

            # dilate and blur mask
            # blur_size = 9
            # mask_generate_blur = dilation(mask_generate, kernel=torch.ones(blur_size, blur_size, device=mask_generate.device))
            # mask_generate_blur = gaussian_blur(mask_generate_blur, kernel_size=blur_size, sigma=5) # [1, 1, H, W]
            # mask_generate[mask_generate > 0.5] = 1 # do not mix any inpaint region
            mask_generate_blur = mask_generate

            # weight map for mask_generate
            # mask_weight = (mask_generate > 0.5).float().cpu().numpy().squeeze(0).squeeze(0)
            # mask_weight = ndimage.distance_transform_edt(mask_weight)#.clip(0, 30) # max pixel dist hardcoded...
            # mask_weight = (mask_weight - mask_weight.min()) / (mask_weight.max() - mask_weight.min() + 1e-20)
            # mask_weight = torch.from_numpy(mask_weight).to(self.device).unsqueeze(0).unsqueeze(0)

            # kiui.vis.plot_matrix(mask_generate, mask_refine, mask_keep)

            if not (mask_generate > 0.5).any():
                return

            control_images = {}

            # construct normal control
            if 'normal' in texgen_config["control_mode"]:
                rot_normal = out['rot_normal'] # [H, W, 3]
                rot_normal[..., 0] *= -1 # align with normalbae: blue = front, red = left, green = top
                control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512)) # [1, 3, H, W]
            
            # construct depth control
            if 'depth' in texgen_config["control_mode"]:
                depth = out['depth']
                control_images['depth'] = _zoom(depth.view(1, 1, H, W), size=(512, 512)).repeat(1, 3, 1, 1) # [1, 3, H, W]
            
            # construct ip2p control
            if 'ip2p' in texgen_config["control_mode"]:
                ori_image = _zoom(out['ori_image'].permute(2, 0, 1).unsqueeze(0).contiguous(), size=(512, 512)) # [1, 3, H, W]
                control_images['ip2p'] = ori_image

            # construct inpaint control
            if 'inpaint' in texgen_config["control_mode"]:
                image_generate = image.clone()
                image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region
                image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
                control_images['inpaint'] = image_generate

                # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
                latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                control_images['latents_mask'] = latents_mask
                control_images['latents_mask_refine'] = latents_mask_refine
                control_images['latents_mask_keep'] = latents_mask_keep
                control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]

            # construct depth-aware-inpaint control
            if 'depth_inpaint' in texgen_config["control_mode"]:
                # "image" is the texture which was rendered by the diff-rasterizer previously
                # image.clone() returns a copy of "image". This function is differentiable, so gradients
                # will flow back from the result of this operation to input. To create a tensor without
                # an autograd relationship to input see detach().
                image_generate = image.clone()


                # This line just mask out those pixels in the original rendered image where needed to be textured
                image_generate[mask_keep.repeat(1, 3, 1, 1) < 0.5] = -1 # -1 is inpaint region
                # image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region

                # This line interpolates the 1024*1024 masked rendered image to 512*512
                image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)

                # This line only converts the depth map corresponding to a image rendered from a specific viewpoint into a depth map with the size of 512*512
                depth = _zoom(out['depth'].view(1, 1, H, W), size=(512, 512)).clamp(0, 1).repeat(1, 3, 1, 1) # [1, 3, H, W]

                # Concatenate the masked image and the depth map
                control_images['depth_inpaint'] = torch.cat([image_generate, depth], dim=1) # [1, 6, H, W]

                # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
                latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                control_images['latents_mask'] = latents_mask
                control_images['latents_mask_refine'] = latents_mask_refine
                control_images['latents_mask_keep'] = latents_mask_keep

                # image_fill = image.clone()
                # image_fill = dilate_image(image_fill, mask_generate_blur, iterations=int(H*0.2))

                control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]

                ################ Display the image buffers generated by the differentiable rasterizer
                # ahz_image = latents_mask.squeeze(0).cpu().numpy().transpose(1, 2, 0)            
                # print(ahz_image.shape)
                # cv2.imshow("image", ahz_image)            
                # cv2.waitKey()

            # kiui.vis.plot_image(control_images['latents_original'].detach().cpu().numpy())
            # kiui.vis.plot_image(control_images['depth_inpaint'].detach().cpu().numpy())
            # kiui.vis.plot_image(control_images['latents_mask'].detach().cpu().numpy())
            # kiui.vis.plot_image(control_images['latents_mask_refine'].detach().cpu().numpy())
            # kiui.vis.plot_image(control_images['latents_mask_keep'].detach().cpu().numpy())

        ########################################## SD and CONTROLNET MODEL ####################################
            rgbs_reward = image_reward

            rgbs, latents, log_probs, mid_samples, down_samples, noise_preds = self.guidance.pipeline_with_logprob_inference(
                prompt_embeds=positive_prompt_embed,
                negative_prompt_embeds=negative_prompt_embed,
                num_inference_steps=texpref_config["sample_num_steps"],
                guidance_scale=texpref_config["sample_guidance_scale"],
                eta=texpref_config["sample_eta"],
                output_type="pt",
                control_images=control_images,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                refine_strength=texgen_config["refine_strength"]
            )

            # Upscale the generated texture (assume 2/4/8x)
            if rgbs.shape[-1] != W or rgbs.shape[-2] != H:
                scale = W // rgbs.shape[-1]
                rgbs = rgbs.detach().cpu().squeeze(0).permute(1, 2, 0).contiguous().numpy()
                rgbs = (rgbs * 255).astype(np.uint8)
                rgbs = kiui.sr.sr(rgbs, scale=scale)
                rgbs = rgbs.astype(np.float32) / 255
                rgbs = torch.from_numpy(rgbs).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
            sample = (rgbs, latents, log_probs, control_images, mid_samples, down_samples, noise_preds)

            # kiui.vis.plot_image(rgbs.detach().cpu().numpy())
            ################################################ INPAINTING ##########################################

            # Apply mask to make sure non-inpaint region is not changed
            # The keep region remains strictly fixed through mask blending in the image space
            # See page 7, equation 4 to understand the following line
            rgbs = rgbs * (1 - mask_keep) + image * mask_keep
            # rgbs = rgbs * mask_generate_blur + image * (1 - mask_generate_blur)

            # Visualizing the texture generated from mipmap interpolation
            # if sampleName == 'sample0':
            #     print('$$$$$$$$$$$$$')
            #     print('sample0')
            #     print('$$$$$$$$$$$$$')
            # elif sampleName == 'sample1':
            #     print('$$$$$$$$$$$$$')
            #     print('sample1')
            #     print('$$$$$$$$$$$$$')

            # kiui.vis.plot_image(rgbs.detach().cpu().numpy())
            # kiui.vis.plot_image(rgbs_reward.detach().cpu().numpy())


            # Visualize the intermediate results
            if texgen_config["vis"]:
                if 'depth' in control_images:
                    # kiui.vis.plot_image(control_images['depth'])
                    pass
                if 'normal' in control_images:
                    kiui.vis.plot_image(control_images['normal'])
                if 'ip2p' in control_images:
                    kiui.vis.plot_image(ori_image)
                # kiui.vis.plot_image(mask_generate)
                if 'inpaint' in control_images:
                    kiui.vis.plot_image(control_images['inpaint'].clamp(0, 1))
                    # kiui.vis.plot_image(control_images['inpaint_refine'].clamp(0, 1))
                if 'depth_inpaint' in control_images:
                    # kiui.vis.plot_image(control_images['depth_inpaint'][:, :3].clamp(0, 1))
                    # kiui.vis.plot_image(control_images['depth_inpaint'][:, 3:].clamp(0, 1))
                    pass
                kiui.vis.plot_image(rgbs)
                # kiui.vis.plot_image(rgbs_reward)

            # grid put

            # project-texture mask
            proj_mask = (out['alpha'] > 0) & (out['viewcos'] > texgen_config["cos_thresh"])  # [H, W, 1]
            # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
            # kiui.vis.plot_image(proj_mask.squeeze(-1).detach().cpu().numpy())
            proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
            uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')
            uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask]
            rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask]


            proj_mask_reward = (out_reward['alpha'] > 0) & (out_reward['viewcos'] > texgen_config["cos_thresh"])  # [H, W, 1]
            # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
            # kiui.vis.plot_image(proj_mask.squeeze(-1).detach().cpu().numpy())
            proj_mask_reward = _zoom(proj_mask_reward.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
            uvs_reward = _zoom(out_reward['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')
            uvs_reward = uvs_reward.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)[proj_mask_reward]

            rgbs_reward = rgbs_reward.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)[proj_mask_reward]



            ################################################# UPDATING ##########################################

            # print(f'[INFO] processing {ver} - {hor}, {rgbs.shape}')            
            cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, min_resolution=128, return_count=True)
            # cur_albedo, cur_cnt = linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, return_count=True)


            cur_albedo_reward, cur_cnt_reward = mipmap_linear_grid_put_2d(h, w, uvs_reward[..., [1, 0]] * 2 - 1, rgbs_reward, min_resolution=128, return_count=True)

            # Adding the unwrapped texture to the output
            sample = sample + (cur_albedo, image_reward, cur_albedo_reward,)

            # Visualizing the texture generated from mipmap interpolation
            # kiui.vis.plot_image(cur_albedo.detach().cpu().numpy())
            # kiui.vis.plot_image(cur_cnt.detach().cpu().numpy())

            # kiui.vis.plot_image(cur_albedo_reward.detach().cpu().numpy())
            # kiui.vis.plot_image(cur_cnt_reward.detach().cpu().numpy())



            # albedo += cur_albedo
            # cnt += cur_cnt

            # mask = cnt.squeeze(-1) < 0.1
            # albedo[mask] += cur_albedo[mask]
            # cnt[mask] += cur_cnt[mask]

            if(sampleName == 'sample0'):
                self.backup0()

                mask = cur_cnt.squeeze(-1) > 0
                self.albedo0[mask] += cur_albedo[mask]
                self.cnt0[mask] += cur_cnt[mask]

                # Update mesh texture for rendering
                self.update_mesh_albedo0()


                # update viewcos cache
                viewcos = viewcos.view(-1, 1)[proj_mask]
                cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
                self.renderer0.mesh.viewcos_cache = torch.maximum(self.renderer0.mesh.viewcos_cache, cur_viewcos)

            elif(sampleName == 'sample1'):
                self.backup1()

                mask = cur_cnt.squeeze(-1) > 0
                self.albedo1[mask] += cur_albedo[mask]
                self.cnt1[mask] += cur_cnt[mask]

                # Update mesh texture for rendering
                self.update_mesh_albedo1()


                # update viewcos cache
                viewcos = viewcos.view(-1, 1)[proj_mask]
                cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
                self.renderer1.mesh.viewcos_cache = torch.maximum(self.renderer1.mesh.viewcos_cache, cur_viewcos)
        
            return sample


    @torch.no_grad
    def train(self):

        all_reward = []
        all_loss = []

        ######################### Initialization #########################
        initialization_output = self.all_initialization()
        (reference_unet_model, prompt_fn, reward_fn,
         camera_pos_horizontals_texture_generation, camera_pos_verticals_texture_generation,
         camera_pos_horizontals_reward_computation, camera_pos_verticals_reward_computation,
         autocast, accelerator, trainable_layers, optimizer) = initialization_output

        with torch.enable_grad():
            # Do track gradients in the forward pass
            self.guidance.pipe.unet.train()

        ######################### Texture-DPO Algorithm #########################        
        for epoch in range(texpref_config["num_epochs"]):
            print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
            # Gradient accumulation is a technique where you can train on
            # bigger batch sizes than your machine would normally be able to fit into memory.
            # This is done by accumulating gradients over several batches, 
            # and only stepping the optimizer after a certain number of batches have been performed.
            # While technically standard gradient accumulation code would work fine in a distributed setup,
            # it is not the most efficient method for doing so and you may experience considerable slowdowns!
            # All that is left now is to let Hugging Face Accelerate handle the gradient accumulation for us.
            # To do so you should pass in a gradient_accumulation_steps parameter to Accelerator, 
            # dictating the number of steps to perform before each call to step() and how to automatically
            # adjust the loss during the call to backward(). From here you can use the accumulate() context
            # manager from inside your training loop to automatically perform the gradient accumulation for you!
            # You just wrap it around the entire training part of our code.
            # [Reference: https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation]
            # with torch.autograd.set_detect_anomaly(True):
            with accelerator.accumulate(self.guidance.pipe.unet):
                with autocast():
                    # Initialize the parameters related to the 3D mesh object
                    # Creating and initializing some variables related to the 3D object with respect to the size of the texture image (e.g. albedo)
                    self.mesh_initialization()

                    start_t = time.time() # This is used to measure the time takes to run the texturing algorithm

                    with torch.enable_grad():
                        text_information = self.text_embed_initialization(prompt_fn, trainable_layers)
                        (prompt, prompt_embed, negative_prompt_embed, _, _) = text_information

                    print(f"###############################################################################################################################################")
                    print(f"################################################################ EPOCH {epoch} ################################################################")
                    print(f"###############################################################################################################################################")

                    # Generate two random and different seeds for two samples we're going to generate!
                    seed = random.sample(range(0, 1000000), 1)
                    print("seed for sample0: ", seed[0])
                    print("Prompt: ", prompt["default"])

                    if texpref_config["prompt_tuning"]:
                        self.guidance.pipe.unet.half()

                    # Randomly changing the order of viewpoints that the differentiable renderer renders the 3D object
                    if common_config['randomize_camera_pose_order']:
                        # Get a list of indices and shuffle it
                        indices = list(range(len(camera_pos_verticals_texture_generation)))
                        random.shuffle(indices)

                        # Reorder both lists according to the shuffled indices
                        shuffled_camera_pos_verticals_texture_generation = [camera_pos_verticals_texture_generation[i] for i in indices]
                        shuffled_camera_pos_horizontals_texture_generation = [camera_pos_horizontals_texture_generation[i] for i in indices]

                        camera_pos_verticals_texture_generation = shuffled_camera_pos_verticals_texture_generation
                        camera_pos_horizontals_texture_generation = shuffled_camera_pos_horizontals_texture_generation

                    print("Camera pose verticals: ", camera_pos_verticals_texture_generation)
                    print("Camera pose horizontals: ", camera_pos_horizontals_texture_generation)


                    if texpref_config["learn_camera_viewpoints"] == True:
                        with torch.enable_grad():
                            # camera_pos_verticals_texture_generation = torch.tensor(camera_pos_verticals_texture_generation, device='cuda').float().requires_grad_(True)
                            camera_pos_horizontals_texture_generation = torch.tensor(camera_pos_horizontals_texture_generation, device='cuda').float().requires_grad_(True)
                            camera_pos_verticals_reward_computation = torch.tensor(camera_pos_verticals_reward_computation, device='cuda').float().requires_grad_(True)
                            camera_pos_horizontals_reward_computation = torch.tensor(camera_pos_horizontals_reward_computation, device='cuda').float().requires_grad_(True)

                    else:
                        with torch.enable_grad():
                            camera_pos_verticals_texture_generation = torch.tensor(camera_pos_verticals_texture_generation, device='cuda').float().requires_grad_(True)
                            camera_pos_horizontals_texture_generation = torch.tensor(camera_pos_horizontals_texture_generation, device='cuda').float().requires_grad_(True)
                            camera_pos_verticals_reward_computation = torch.tensor(camera_pos_verticals_reward_computation, device='cuda').float().requires_grad_(True)
                            camera_pos_horizontals_reward_computation = torch.tensor(camera_pos_horizontals_reward_computation, device='cuda').float().requires_grad_(True)


                    self.forwar_pass_visit_count = 0
                    avg_loss, avg_reward = self.multi_view_forward(reference_unet_model, epoch, seed, camera_pos_verticals_texture_generation, camera_pos_horizontals_texture_generation,
                                                                camera_pos_horizontals_reward_computation, camera_pos_verticals_reward_computation,
                                                                prompt, prompt_embed, negative_prompt_embed, reward_fn)

                    print("\n RLHF optimized - avg_loss: ", avg_loss.requires_grad)
                    print("\n RLHF optimized - avg_loss: ", avg_loss.grad_fn)
                    print("\n RLHF optimized - avg_reward: ", avg_reward.requires_grad)
                    print("\n RLHF optimized - avg_reward: ", avg_reward.grad_fn)


                    self.albedo0 = self.renderer0.mesh.albedo
                    self.cnt0 = self.renderer0.mesh.cnt
                    self.viewcos_cache0 = self.renderer0.mesh.viewcos_cache


                    # Post-processing the texture image
                    # self.dilate_texture_rlhf()
                    # self.deblur_texture_rlhf()
                    torch.cuda.synchronize()

                    #####################################################################################################################################
                    ################### Step 3 - Texture Enhancement: Compute the loss and backpropagate it through the model #######################
                    #####################################################################################################################################
                    with torch.enable_grad():
                        # backward pass 
                        accelerator.backward(avg_loss)
                        if accelerator.sync_gradients:
                            if texpref_config["is_grad_norm_clipping"]:
                                accelerator.clip_grad_norm_(trainable_layers.parameters(), max_norm=texpref_config["train_max_grad_norm"])
                            elif texpref_config["is_grad_value_clipping"]:
                                accelerator.clip_grad_value_(trainable_layers.parameters(), clip_value=texpref_config["train_max_grad_value"])
                            else:
                                pass
                            print("\n Performing optimizer step and synchronizing gradients")
                        optimizer.step()
                        optimizer.zero_grad()


            # IMPORTANT: You need to call the detach() function on both the loss and reward values before storing them into the following list.
            # Otherwise, what is going to happen is that the "all_loss" and "all_reward" lists will store all the information
            # related to the gradient computations. So, because these two lists holds tensors with gradients and also because these two
            # tensors never gets back-propagated, we will never clear the computational graph and so the computational graph just
            # keeps growing and growing. That was how I had the memory leak issue.
            # So, to debug this I experimented many things but finally the following link helped me a lot to find where the memory leak comes from.
            # [Reference: https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339]
            # So, the way I debugged this, was to use torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated() to print a percent of used
            # memory at the top of the training loop. Then I looked at my training loop, added a "continue" statement right below the first line and
            # run the training loop. So, every time memory usage holds steady, I moved the continue to the next line and so on until I found out that
            # the leak is related to these two lists "all_loss" and "all_reward". The goal of this approach was to just isolate each line individually
            # until I find the part with the memory leak.
            all_loss.append(avg_loss.detach())
            all_reward.append(avg_reward.detach())

            f = open("reward_loss_values.txt", "a")
            f.write(f"{avg_reward.cpu().detach().numpy()}, {avg_loss.cpu().detach().numpy()} \n")
            f.close()

            # AHZ: This is used to measure the time takes to run the texturing algorithm
            end_t = time.time()            
            print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

            # Save the fine-tuned model (checkpointing)
            if (epoch) % texpref_config["save_freq"] == 0 and accelerator.is_main_process:
                # Generate and save the mesh object and its texture with a standard format.
                self.save_model(prompt, epoch, all_reward, all_loss, seed)
                accelerator.save_state()


            # Save fine-tuned weights
            if texpref_config["prompt_tuning"]:
                torch.save(trainable_layers, 'latest_finetuned_weights.pth')
            elif texpref_config["use_lora"]:
                torch.save(trainable_layers.state_dict(), 'latest_finetuned_weights.pth')
            else:
                torch.save(self.guidance.pipe.unet.state_dict(), 'latest_finetuned_weights.pth')

            # torch.save(self.guidance.pipe.unet.state_dict(), 'all_weights.pth')

            self.isFineTuningCorrect()


            # Update the previous fine-tuned model
            if texpref_config["prompt_tuning"]:
                torch.save(trainable_layers, 'previous_finetuned_weights.pth')
            elif texpref_config["use_lora"]:
                torch.save(trainable_layers.state_dict(), 'previous_finetuned_weights.pth')
            else:
                torch.save(self.guidance.pipe.unet.state_dict(), 'previous_finetuned_weights.pth')

            # Clear cached memory after each epoch
            torch.cuda.empty_cache()

    def inference(self, seed):
        all_reward = []
        all_loss = []

        ######################### Initialization #########################
        initialization_output = self.all_initialization()
        (ref, prompt_fn, reward_fn,
         camera_pos_horizontals_texture_generation, camera_pos_verticals_texture_generation,
         camera_pos_horizontals_reward_computation, camera_pos_verticals_reward_computation,
         autocast, accelerator, trainable_layers, optimizer) = initialization_output

        for iteration in range(texpref_config["num_generate_samples"]): # This loop runs 1000 times

            # Initialize the parameters related to the 3D mesh object
            # Creating and initializing some variables related to the 3D object with respect to the size of the texture image (e.g. albedo)
            self.mesh_initialization()

            start_t = time.time() # This is used to measure the time takes to run the texturing algorithm

            text_information = self.text_embed_initialization(prompt_fn, trainable_layers)
            (prompt, prompt_embed, negative_prompt_embed,
            sample_neg_prompt_embeds, train_neg_prompt_embeds) = text_information
            
            # Generate two random and different seeds for two samples we're going to generate!            
            print("seed for sample0: ", seed[0])            
            print("Prompt: ", prompt["default"])
            
            if texpref_config["prompt_tuning"]:
                self.guidance.pipe.unet.half()

            total_reward_sample0 = 0.0            
            
            ################################################################################################################
            ################### Step 1 - Texture Generation: Generate the texture for all viewpoints #######################
            ################################################################################################################
            for current_view_num, (ver, hor) in enumerate(tqdm(zip(camera_pos_verticals_texture_generation, camera_pos_horizontals_texture_generation), total=len(camera_pos_verticals_texture_generation))):

                # AHZ: Put the camera in different locations and compute the camera position matrix in the world
                viewpoint = orbit_camera(ver, hor, self.cam.radius)

                # pose to view dir
                if texgen_config["text_dir"]:
                    v, h, _ = undo_orbit_camera(viewpoint)
                    if v <= -60: d = 'top'
                    elif v >= 60: d = 'bottom'
                    else:
                        if abs(h) < 30: d = 'front'
                        elif abs(h) < 90: d = 'side'
                        else: d = 'back'
                    prompts2 = prompts1 = prompt[d]
                    prompt_embeds2 = prompt_embeds1 = prompt_embed[d]
                else:
                    prompts2 = prompts1 = prompt["default"]
                    prompt_embeds2 = prompt_embeds1 = prompt_embed["default"]


                # Render the object into an image from a specific viewpoint and generate the texture using the Intex method             
                result0 = self.inpaint_view_inference(ref, viewpoint, negative_prompt_embed, prompt_embeds1, sampleName='sample0', seed=seed[0])                

                # This one
                if result0 is None:
                    print("\n Warning: inpaint_view returned None")
                    continue  # Skip to the next iteration

                rendering0, latents0, log_probs0, control_images0, mid_samples0, down_samples0, noise_preds0, unwrapped_texture0, texture_reward, unwrapped_texture_reward = result0                

                # kiui.vis.plot_image(rendering0.detach().cpu().numpy())
                # kiui.vis.plot_image(rendering1.detach().cpu().numpy())                
                # kiui.vis.plot_image(unwrapped_texture0.detach().cpu().numpy())
                # kiui.vis.plot_image(unwrapped_texture1.detach().cpu().numpy())

            # Post-processing the texture image
            # self.dilate_texture()
            self.deblur_texture()
            torch.cuda.synchronize()

            #######################################################################################################################################
            ################### Step 2 - Texture Evaluation: Render the object from different viewpoints to compute the reward ####################
            #######################################################################################################################################
            for current_view_num, (ver, hor) in enumerate(tqdm(zip(camera_pos_verticals_reward_computation, camera_pos_horizontals_reward_computation), total=len(camera_pos_verticals_reward_computation))):
                # AHZ: Put the camera in different locations and compute the camera position matrix in the world space
                viewpoint = orbit_camera(ver, hor, self.cam.radius)

                # Render the object into an image from a specific viewpoint and generate the texture using the Intex method             
                reward0 = self.render_and_compute_reward_inference(reward_fn, viewpoint, prompts1, sampleName='sample0')

                total_reward_sample0 += reward0                

            total_reward_sample0 /= len(camera_pos_verticals_reward_computation)            

            # AHZ: This is used to measure the time takes to run the texturing algorithm
            end_t = time.time()            
            print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')
            print("Reward for the sample 0: ", total_reward_sample0)            

            # Generated and save the mesh object and its texture with a standard format.
            self.save_model(prompt, iteration, all_reward, all_loss, seed)


    def prepend_learnable_tokens(self, learnable_tokens, input_ids):
        # Get the original prompt embeddings from the text encoder
        original_prompt_embeddings = self.guidance.pipe.text_encoder(input_ids.to(self.device))[0]

        # Repeat learnable tokens for the batch size
        batch_size = input_ids.shape[0]
        repeated_tokens = learnable_tokens.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)

        # Concatenate learnable tokens to original prompt embeddings
        modified_prompt_embeddings = torch.cat([repeated_tokens, original_prompt_embeddings], dim=1)

        return modified_prompt_embeddings 

   

if __name__ == "__main__":

    texture_generetor = Differentiable_Texture_Learning()

    if common_config['training']: # Perform the fine-tuning process                 
        texture_generetor.train()

    elif common_config['inference']: # Perform the inference texture-generation process
        seed = [350250, 0]
        texture_generetor.inference(seed)



    # Clear GPU memory
    torch.cuda.empty_cache()
