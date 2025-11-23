from collections import defaultdict

# Defining the dict and passing
# lambda as default_factory argument
texpref_config = defaultdict(lambda: "Not Present")
texgen_config = defaultdict(lambda: "Not Present")
common_config = defaultdict(lambda: "Not Present")


################################################ Pretrained Model ############################################################
# base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
common_config["pretrained_revision"] = "main"
common_config["model_key"] = "philz1337/revanimated" # Other options:
                                                     # 1- "CompVis/stable-diffusion-v1-4"  
                                                     # 2- "stabilityai/stable-diffusion-2-1"
                                                     # 3-  "runwayml/stable-diffusion-v1-5"

common_config['randomize_camera_pose_order'] = False # Whether to shuffle the camera positions
common_config['training'] = True
common_config['inference'] = False

############ General ############
# top-level logging directory for checkpoint saving.
texpref_config["logdir"] = "logs"
# number of epochs to train for. each epoch is one round of sampling from the model followed by training on those samples.
texpref_config["num_epochs"] = 20000
# number of epochs between saving model checkpoints.
texpref_config["save_freq"] = 1
# number of checkpoints to keep before overwriting old ones.
texpref_config["num_checkpoint_limit"] = 50
# mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
texpref_config["mixed_precision"] = "fp16"
# allow tf32 on Ampere GPUs, which can speed up training.
texpref_config["allow_tf32"] = True
# resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
# containing checkpoints, in which case the latest one will be used. `d3po_config.use_lora` must be set to the same value
# as the run that generated the saved checkpoint.
texpref_config["resume_from"] = ""
# Whether or not to use Prompt-Tuning
texpref_config["prompt_tuning"] = False
# whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
# attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
# about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
# files will also be large.
if texpref_config["prompt_tuning"]:
    texpref_config["use_lora"] = False
else:
    texpref_config["use_lora"] = True

# whether or not to use xFormers to reduce memory usage.
texpref_config["use_xformers"] = False

texpref_config["num_generate_samples"] = 1
# number of sampler inference steps.
# Why do we have this if condition?
# The reason is that when training the texture-generation, we backpropagate
# through all the intermediate steps including all the textures that are
# generated on the 3D object from different camera viewpoints. So, as we
# had memory issues during this process, I had to decrease the number of
# diffusion steps (this was one of the temporary solution to resolve the issue.
# However, I am working on it and it should be systematically resolved in near future!)
if common_config["inference"]:
    texpref_config["sample_num_steps"] = 10
else:
    texpref_config["sample_num_steps"] = 10

texpref_config["cfg_steps"] = texpref_config["sample_num_steps"]
# eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
# being fully deterministic and 1.0 being equivalent to the DDPM sampler.
texpref_config["sample_eta"] = 1.0
# classifier-free guidance weight. 1.0 is no guidance.
texpref_config["sample_guidance_scale"] = 7.5
# save interval
texpref_config["sample.save_interval"] = 100
# eval epoch
texpref_config["sample_eval_epoch"] = 10
############ Training ############
# learning rate.
texpref_config["train_learning_rate"] = 0.00005 #2e-5
# Adam beta1.
texpref_config["train_adam_beta1"] = 0.9
# Adam beta2.
texpref_config["train_adam_beta2"] = 0.999
# Adam weight decay.
texpref_config["train_adam_weight_decay"] = 1e-4
# Adam epsilon.
texpref_config["train_adam_epsilon"] = 1e-8
# number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
# gradient_accumulation_steps`.
texpref_config["train_gradient_accumulation_steps"] = 1 # = 8*32

texpref_config["is_grad_norm_clipping"] = False
texpref_config["is_grad_value_clipping"] = False
# maximum gradient norm for gradient clipping.
texpref_config["train_max_grad_norm"] = 0.001
# maximum gradient value for gradient clipping.
texpref_config["train_max_grad_value"] = 0.001
# enable activation checkpointing or not.
# this reduces memory usage at the cost of some additional compute.
texpref_config["train_activation_checkpoint"] = True
texpref_config["train_activation_checkpoint_controlnet"] = True
# whether enable the KL-divergence regularization or not
# this keeps the fine-tuning model as close as possible to the pre-trained model
texpref_config["kl_divergence"] = False
texpref_config["kl_divergence_beta"] = 1.0
# whether enable the LoRA-scaling regularization or not
# this keeps the fine-tuning model as close as possible to the pre-trained model:
# The more the lora_scale_alpha is, the more the LoRA parameters contribute to generate the output image
# The less the lora_scale_alpha is, the more the original pre-trained diffusion model (stable diffusion v1.4) contributes to generate the output image
texpref_config["lora_scaling"] = True
texpref_config["lora_scale_alpha"] = 1.0

# Whether to learn camera viewpoints or not
texpref_config["learn_camera_viewpoints"] = True
texpref_config["camera_learning_rate"] = 5.0

# whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
# sampling will be used during training.
texpref_config["train_cfg"] = True

############ Prompt Function ############
# prompt function to use. see `prompts.py` for available prompt functisons.
texpref_config["prompt_fn"] = "dragon_texture"
# kwargs to pass to the prompt function.
texpref_config["prompt_fn_kwargs"] = {}
############ Reward Function ############
# reward function to use. see `rewards.py` for available reward functions.
# if the reward_fn is "jpeg_compressibility" or "jpeg_incompressibility", using the default d3po_config can reproduce our results.
# if the reward_fn is "aesthetic_score" and you want to reproduce our results,
# set d3po_config.num_epochs = 1000, sample.num_batches_per_epoch=1, sample.batch_size=8 and sample.eval_batch_size=8
texpref_config["reward_fn"] = "aesthetic_score_differentiable" # "mean_curvature_texture_alignment_reward"  "curvature_score_sharp" "curvature_score_smooth" "curvature_score_mse" "aesthetic_score_differentiable" "symmetry_score" #"aesthetic_score"
                                                               # "minimum_curvature_texture_alignment_reward"
texpref_config["use_ddim_scheduler"] = True
texpref_config["use_unipc_scheduler"] = False # TODO: This one does not work for now. Add the file "unipc_with_logprob" similar to the "ddim_with_logprob"

######################################################### Texture-Generation Config ############################
texgen_config['mesh_name'] = "Russian_Soldier.obj"
texgen_config['vector_texture_path'] = '../Assets/Minimum_Curvature_Texture_Alignment_Reward/Rabbit/After_smoothing(=500)/rabbit_vec_tex_500.txt'
# texgen_config['vector_texture_path'] = '../Assets/Minimum_Curvature_Texture_Alignment_Reward/Rabbit_Different_UV/After_smoothing(=500)/rabbitdiffUV_vec_tex_500.txt'
# texgen_config['vector_texture_path'] = '../Assets/Minimum_Curvature_Texture_Alignment_Reward/Balloon/After_smoothing(=1000)/balloon_vec_tex_1000.txt'
# texgen_config['vector_texture_path'] = '../Assets/Minimum_Curvature_Texture_Alignment_Reward/Balloon/Before_smoothing/balloon_vec_tex_0.txt'
texgen_config["mesh"] = f"../Assets/{texgen_config['mesh_name']}"
texgen_config["reward_mesh"] = "../Assets/3DReward/Rabbit/Rabbit.obj"
texgen_config["reward_texture"] = "../Assets/3DReward/Rabbit/Rabbit_albedo.png"
### mesh's front-facing direction, check mesh.py for details
texgen_config["front_dir"] = "+z"
texgen_config["outdir"] = "../logs"
texgen_config["save_path"] = texgen_config['mesh_name']
### additional positive/negative prompt
texgen_config["posi_prompt"] = "masterpiece, high quality"
texgen_config["nega_prompt"] = "bad quality, worst quality, shadows"
### controlnet model
texgen_config["control_mode"] = ['depth_inpaint']
### whether append 'front/side/back/top/bottom view' for the prompt, only use if the mesh's front dir is correctly set.
texgen_config["text_dir"] = True
### camera path to generate the texture
texgen_config["camera_path_texture_generation"] = "front_short"
### camera path to generate the texture
texgen_config["camera_path_reward_computation"] = "tilted_short"
### generated texture image resolution
texgen_config["texture_size"] = 1024
### threshold for view cosine value
texgen_config["cos_thresh"] = 0
### refine strength (larger value means less strong refinement)
texgen_config["refine_strength"] = 0
### camera rendering resolution (choose from [512, 1024, 2048, 4096], larger than 512 will incur Real-ESRGAN super-resolution)
texgen_config["render_resolution"] = 1024
### visualize intermediate results
texgen_config["vis"] = False


### gui resolution
texgen_config["H"] = 800
texgen_config["W"] = 800
### camera setting
texgen_config["radius"] = 2.5
texgen_config["fovy"] = 50
### recalculate uv coordinate for mesh
texgen_config["retex"] = False
texgen_config["remesh"] = False # This does not work for now. It gives me this error when setting it to True:
                               # "RuntimeError: CUDA error: device-side assert triggered
                               # CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
                               # For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
                               # Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions."


