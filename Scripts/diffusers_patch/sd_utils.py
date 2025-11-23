import os
import torch.utils
import torch.utils.checkpoint
from transformers import logging


from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from .ddim_with_logprob import ddim_step_with_logprob, ddim_step_rlhf


# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from Configuration.Config import texpref_config


class StableDiffusion(nn.Module):
    def __init__(self, device, control_mode, model_key):
        
        super().__init__()

        self.device = device
        self.control_mode = control_mode
        
        # A comprehensive comparison between FP16, BF16, and FP32 floating-point formats:
        #   1- FP32 (Single Precision): Allocates 8 bits for the exponent and 23 bits for the mantissa, providing a wide dynamic range and high precision.
        #   2- FP16 (Half Precision): Uses 5 bits for the exponent and 10 bits for the mantissa, resulting in a narrower dynamic range and reduced precision compared to FP32.
        #   3- BF16 (Brain Floating Point): Shares the same 8-bit exponent as FP32 but reduces the mantissa to 7 bits, balancing dynamic range with lower precision.
                
        # a) Dynamic Range:
        #   1- FP32: Offers a vast dynamic range due to its 8-bit exponent, accommodating very large and very small numbers.
        #   2- FP16: Has a limited dynamic range because of its 5-bit exponent, which can lead to overflow or underflow in computations involving extreme values.
        #   3- BF16: Maintains the same dynamic range as FP32 with its 8-bit exponent, reducing the risk of overflow or underflow.
        
        # b) Precision:
        #   1- FP32: Provides high precision with its 23-bit mantissa, allowing for detailed representation of fractional values.
        #   2- FP16: Offers lower precision due to its 10-bit mantissa, which may result in quantization errors in sensitive computations.
        #   3- BF16: Further reduces precision with a 7-bit mantissa, potentially leading to higher quantization errors but often acceptable in many machine learning applications.

        # c) Memory Usage: 
        #   1- Both FP16 and BF16 consume half the memory of FP32, enabling larger models or batch sizes to fit into the same memory footprint.
        
        # d) Computational Speed: 
        #   Modern hardware accelerators, such as NVIDIA's Tensor Cores and Google's TPUs, are optimized for lower-precision formats:
        #   1- FP16: Supported by many GPUs, offering significant speedups in compatible operations.
        #   2- BF16: Gaining support in newer hardware, providing a balance between speed and numerical stability.

        # [References:
        #  1- https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407
        #  2- https://stats.stackexchange.com/questions/637988/understanding-the-advantages-of-bf16-vs-fp16-in-mixed-precision-training
        #  3- https://www.exxactcorp.com/blog/hpc/what-is-fp64-fp32-fp16]
        if texpref_config["mixed_precision"] is "fp16":
            self.dtype = torch.float16            
        elif texpref_config["mixed_precision"] is "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32


        # Loading the pretrained stable diffusion model
        if os.path.exists(model_key):            
            # treat as local ckpt
            pipe = StableDiffusionPipeline.from_single_file(model_key, torch_dtype=self.dtype)
        else:            
            # huggingface ckpt
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype)

        pipe.to(device)
        self.pipe = pipe


        # Loading the ControlNet model to give more flexibility to the diffusio model
        if self.control_mode is not None:
            self.controlnet = {}
            self.controlnet_conditioning_scale = {}
            
            if "normal" in self.control_mode:
                self.controlnet['normal'] = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_normalbae",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['normal'] = 1.0
            if "depth" in self.control_mode:
                self.controlnet['depth'] = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['depth'] = 1.0
            if "ip2p" in self.control_mode:
                self.controlnet['ip2p'] = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_ip2p",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['ip2p'] = 1.0
            if "inpaint" in self.control_mode:
                self.controlnet['inpaint'] = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint",torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['inpaint'] = 1.0
            if "depth_inpaint" in self.control_mode:
                self.controlnet['depth_inpaint'] = ControlNetModel.from_pretrained("ashawkey/control_v11e_sd15_depth_aware_inpaint",torch_dtype=self.dtype).to(self.device)
                # self.controlnet['depth_inpaint'] = ControlNetModel.from_pretrained("ashawkey/controlnet_depth_aware_inpaint_v11", torch_dtype=self.dtype).to(self.device)
                
                # TODO: Look at this thread to see how to implement the lighter version of the ControlNet called "ControlNetLora" by StabilityAI
                # [Reference: https://github.com/huggingface/diffusers/issues/4679]
                # self.controlnet['depth_inpaint'] = ControlNetModel.from_single_file(r"C:\Users\User\Desktop\Texture-DPO\Texture-DPO\Texture-DPO\models\control-lora-depth-rank128.safetensors", torch_dtype=self.dtype).to(self.device)
                self.controlnet_conditioning_scale['depth_inpaint'] = 1.0

                # To prevent updates, remove ControlNet parameters from our optimizer
                for param in self.controlnet['depth_inpaint'].parameters():
                    param.requires_grad = False # We keep the parameters of the ControlNet model unchanged during the fine-tuning process.
                                        # TODO: During the training process, check whether these actually remain unchanged.

                # self.controlnet['depth_inpaint'].enable_gradient_checkpointing()



        # Using UniPC scheduler instead of DDIM scheduler as it is faster
        if texpref_config["use_ddim_scheduler"]:
            self.pipe.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.dtype)

    
    @torch.no_grad()
    def get_text_embeds(self, prompt, truncation=False):
        # prompt: [str]

        # For tokenizer see: https://stackoverflow.com/questions/65246703/how-does-max-length-padding-and-truncation-arguments-work-in-huggingface-bertt#:~:text=Add%20a%20comment-,1%20Answer,-Sorted%20by%3A
        inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            truncation=truncation,
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.pipe.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def decode_latents(self, latents):
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        imgs = self.pipe.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.pipe.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()   
    def __call__(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=texpref_config["sample_num_steps"],
        guidance_scale=texpref_config["sample_guidance_scale"],
        guidance_rescale=0,
        control_images=None,
        latents=None,
        strength=0,
        refine_strength=0,
    ):
        # Send the text embedding and control signals to GPU
        text_embeddings = text_embeddings.to(self.dtype)
        for k in control_images:
            control_images[k] = control_images[k].to(self.dtype)

        # Initialize the "latents" variable with some random value with the size of (?, 4, 64, 64)
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, 4, height // 8, width // 8,), dtype=self.dtype, device=self.device)
        
        if strength != 0: # The strength is zero and the code never comes here
            full_num_inference_steps = int(num_inference_steps / (1 - strength))
            self.pipe.scheduler.set_timesteps(full_num_inference_steps)
            init_step = full_num_inference_steps - num_inference_steps
            latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents), self.pipe.scheduler.timesteps[init_step])
        else:             # The code always comes here
            self.pipe.scheduler.set_timesteps(num_inference_steps)
            init_step = 0

        # Performing the denoising process
        for i, t in enumerate(self.pipe.scheduler.timesteps[init_step:]):
            
            # inpaint mask blend
            if 'latents_mask' in control_images:
                if i < num_inference_steps * refine_strength:
                    # fix keep + refine at early steps
                    mask_keep = 1 - control_images['latents_mask']
                else:
                    # 'keep' region in the image remains fixed through mask blending at later steps
                    mask_keep = control_images['latents_mask_keep']

                latents_original = control_images['latents_original']
                latents_original_noisy = self.pipe.scheduler.add_noise(latents_original, torch.randn_like(latents_original), t)
                
                # Perform the mask blending the latent space (see equation 2 and 4 of the paper to understand this)
                latents = latents * (1 - mask_keep) + latents_original_noisy * mask_keep

            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            # Execute the ControlNet
            if self.control_mode is not None and control_images is not None:

                noise_pred = 0

                # This for loop only runs once because there is only "depth_inpaint" item in the self.controlnet variable
                for mode, controlnet in self.controlnet.items():
                    # may omit control mode if input is not provided
                    if mode not in control_images: continue
                    
                    control_image = control_images[mode] # the "control_image" is initialized with the rendered image and the depth map obtained from the differentiable rasterizer
                    weight = 1 / len(self.controlnet) # This "weight" is 1 as there is only the "depth_inpaint" in the self.controlnet variable

                    # Expand the control image if we are doing classifier-free guidance to avoid doing two forward passes.
                    control_image_input = torch.cat([control_image] * 2)

                    # Run the ControlNet model
                    down_samples, mid_sample = controlnet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        controlnet_cond=control_image_input, 
                        conditioning_scale=self.controlnet_conditioning_scale[mode],
                        return_dict=False
                    )

                    # predict the noise residual
                    noise_pred_cur = self.pipe.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        down_block_additional_residuals=down_samples, 
                        mid_block_additional_residual=mid_sample
                    ).sample

                    # merge after unet
                    noise_pred = noise_pred + weight * noise_pred_cur
                
            else:
                noise_pred = self.pipe.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings,
                ).sample

            # Perform guidance by incorporating the guidance scale
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            if guidance_rescale > 0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        return imgs
    
    # Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    # with the following modifications:
    # - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
    #   `ddim` scheduler.
    # - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.    
    @torch.no_grad()
    def pipeline_with_logprob_inference(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = texpref_config["sample_num_steps"],
        guidance_scale: float = texpref_config["sample_guidance_scale"] ,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = texpref_config["sample_eta"],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        control_images=None,
        refine_strength=0
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(prompt, 
                            height=height, 
                            width=width, 
                            callback_steps=callback_steps, 
                            negative_prompt=negative_prompt, 
                            prompt_embeds=prompt_embeds, 
                            negative_prompt_embeds=negative_prompt_embeds)


        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0        

        # 3. Encode input prompt
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        prompt_embeds_temp = prompt_embeds

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Denoising loop

        # Send the control signals to GPU
        if self.control_mode is not None:
            for k in control_images:
                control_images[k] = control_images[k].to(self.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipe.scheduler.order

        all_latents = [latents]
        all_log_probs = []
        all_mid_samples = []
        all_down_samples = []
        all_noise_preds = []

        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for diffusion_step, t in enumerate(timesteps):

                # Inpaint mask blend
                if self.control_mode is not None and control_images is not None:
                    if 'latents_mask' in control_images:
                        if diffusion_step < num_inference_steps * refine_strength:
                            # fix keep + refine at early steps
                            mask_keep = 1 - control_images['latents_mask']
                        else:
                            # 'keep' region in the image remains fixed through mask blending at later steps
                            mask_keep = control_images['latents_mask_keep']

                        latents_original = control_images['latents_original']
                        latents_original_noisy = self.pipe.scheduler.add_noise(latents_original, torch.randn_like(latents_original), t)
                        
                        # Perform the mask blending the latent space (see equation 2 and 4 of the paper to understand this)
                        latents = latents * (1 - mask_keep) + latents_original_noisy * mask_keep

                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    if diffusion_step < texpref_config["cfg_steps"]:
                        latent_model_input = torch.cat([latents] * 2)  # Prepare conditional and unconditional embeddings for CFG                        
                    else:
                        # Revert to single batch size for prompt_embeds
                        _, prompt_embeds = prompt_embeds_temp.chunk(2)

                        # Use original latents for remaining steps
                        latent_model_input = latents
                
                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual                            
                if self.control_mode is not None and control_images is not None: # Execute the ControlNet                    

                    # This for loop only runs once because there is only "depth_inpaint" item in the self.controlnet variable
                    for mode, controlnet in self.controlnet.items():
                        # may omit control mode if input is not provided
                        if mode not in control_images: continue
                        
                        control_image = control_images[mode] # the "control_image" is initialized with the rendered image and the depth map obtained from the differentiable rasterizer
                        weight = 1 / len(self.controlnet) # This "weight" is 1 as there is only the "depth_inpaint" in the self.controlnet variable

                        # Expand the control image if we are doing classifier-free guidance to avoid doing two forward passes.
                        if do_classifier_free_guidance:
                            if diffusion_step < texpref_config["cfg_steps"]:
                                control_image_input = torch.cat([control_image] * 2)
                            else:
                                control_image_input = control_image

                        # Run the ControlNet model
                        down_samples, mid_sample = controlnet(
                            latent_model_input, t, encoder_hidden_states=prompt_embeds, 
                            controlnet_cond=control_image_input, 
                            conditioning_scale=self.controlnet_conditioning_scale[mode],
                            return_dict=False
                        )

                        # predict the noise residual
                        noise_pred_cur = self.pipe.unet(
                            latent_model_input, 
                            t, 
                            encoder_hidden_states=prompt_embeds, 
                            down_block_additional_residuals=down_samples, 
                            mid_block_additional_residual=mid_sample
                        ).sample

                        # merge after unet
                        noise_pred = weight * noise_pred_cur

                        all_noise_preds.append(noise_pred)
                else:
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    if diffusion_step < texpref_config["cfg_steps"]:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        # For steps after cfg_steps, revert to unconditional prediction
                        noise_pred = noise_pred.chunk(2, dim=0)[0]  # Use only the unconditional prediction

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents, log_prob = ddim_step_with_logprob(self = self.pipe.scheduler,
                                                           model_output = noise_pred,
                                                           timestep = t,
                                                           sample = latents,
                                                           eta = eta,
                                                           generator = generator
                                                           )

                all_latents.append(latents)
                all_log_probs.append(log_prob)
                if self.control_mode is not None and control_images is not None: # Execute the ControlNet
                    all_mid_samples.append(mid_sample)
                    all_down_samples.append(down_samples)

                # call the callback, if provided
                if diffusion_step == len(timesteps) - 1 or ((diffusion_step + 1) > num_warmup_steps and (diffusion_step + 1) % self.pipe.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and diffusion_step % callback_steps == 0:
                        callback(diffusion_step, t, latents)

        if not output_type == "latent":
            image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0].float()
            # image = (image / 2 + 0.5).clamp(0, 1) # This line was added from the Intex code to here            
        else:
            image = latents            

        do_denormalize = [True] * image.shape[0]
        image = self.pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.final_offload_hook.offload()

        return image, all_latents, all_log_probs, all_mid_samples, all_down_samples, all_noise_preds

    @torch.no_grad()
    def pipeline_with_logprob_train(
        self,
        reference_unet_model,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = texpref_config["sample_num_steps"],
        guidance_scale: float = texpref_config["sample_guidance_scale"] ,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = texpref_config["sample_eta"],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        control_images=None,
        refine_strength=0
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """        
        # 0. Default height and width to unet
        height = height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(prompt, 
                            height=height, 
                            width=width, 
                            callback_steps=callback_steps, 
                            negative_prompt=negative_prompt, 
                            prompt_embeds=prompt_embeds, 
                            negative_prompt_embeds=negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0        

        # 3. Encode input prompt
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        with torch.enable_grad():
            prompt_embeds = self.pipe._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
            )

        prompt_embeds_temp = prompt_embeds

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # To make the "latents" a differntiable leaf node in our computation graph
        # TODO: Check what is happening inside the "self.pipe.prepare_latens" function.
        # Maybe, there is something that you need to make differentiable and then if 
        # that's the case, you don't need to do the following line.
        latents.requires_grad_(True)


        # 7. Denoising loop

        # Send the control signals to GPU
        with torch.enable_grad():
            if self.control_mode is not None:
                for k in control_images:
                    control_images[k] = control_images[k].to(self.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipe.scheduler.order

        all_latents = [latents]
        all_noise_preds = []

        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for diffusion_step, t in enumerate(timesteps):

                # Inpaint mask blend
                with torch.enable_grad():
                    if self.control_mode is not None and control_images is not None:
                        if 'latents_mask' in control_images:
                            if diffusion_step < num_inference_steps * refine_strength:
                                # fix keep + refine at early steps
                                mask_keep = 1 - control_images['latents_mask']
                            else:
                                # 'keep' region in the image remains fixed through mask blending at later steps
                                mask_keep = control_images['latents_mask_keep']

                            latents_original = control_images['latents_original']
                            latents_original_noisy = self.pipe.scheduler.add_noise(latents_original, torch.randn_like(latents_original), t)
                            
                            # Perform the mask blending the latent space (see equation 2 and 4 of the paper to understand this)
                            latents = latents * (1 - mask_keep) + latents_original_noisy * mask_keep


                    # expand the latents if we are doing classifier free guidance
                    if do_classifier_free_guidance:
                        if diffusion_step < texpref_config["cfg_steps"]:
                            latent_model_input = torch.cat([latents] * 2)  # Prepare conditional and unconditional embeddings for CFG                        
                        else:
                            # Revert to single batch size for prompt_embeds
                            _, prompt_embeds = prompt_embeds_temp.chunk(2)

                            # Use original latents for remaining steps
                            latent_model_input = latents
                    
                    latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual                            
                    if self.control_mode is not None and control_images is not None: # Execute the ControlNet                    

                        # This for loop only runs once because there is only "depth_inpaint" item in the self.controlnet variable
                        for mode, controlnet in self.controlnet.items():
                            # may omit control mode if input is not provided
                            if mode not in control_images: continue
                            
                            control_image = control_images[mode] # the "control_image" is initialized with the rendered image and the depth map obtained from the differentiable rasterizer
                            weight = 1 / len(self.controlnet) # This "weight" is 1 as there is only the "depth_inpaint" in the self.controlnet variable

                            # Expand the control image if we are doing classifier-free guidance to avoid doing two forward passes.
                            if do_classifier_free_guidance:
                                if diffusion_step < texpref_config["cfg_steps"]:
                                    control_image_input = torch.cat([control_image] * 2)
                                else:
                                    control_image_input = control_image
                            

                            # Run the ControlNet model
                            if (texpref_config["train_activation_checkpoint"] == True) and (texpref_config["train_activation_checkpoint_controlnet"] == True):
                                down_samples, mid_sample = torch.utils.checkpoint.checkpoint(
                                                            controlnet,
                                                            latent_model_input, t, encoder_hidden_states=prompt_embeds, 
                                                            controlnet_cond=control_image_input, 
                                                            conditioning_scale=self.controlnet_conditioning_scale[mode],
                                                            return_dict=False,
                                                            use_reentrant=False
                                                            )
                            else:
                                down_samples, mid_sample = controlnet(
                                    latent_model_input, t, encoder_hidden_states=prompt_embeds, 
                                    controlnet_cond=control_image_input, 
                                    conditioning_scale=self.controlnet_conditioning_scale[mode],
                                    return_dict=False
                                )


                            # predict the noise residual
                            noise_pred_cur = self.pipe.unet(
                                latent_model_input, 
                                t, 
                                encoder_hidden_states=prompt_embeds, 
                                down_block_additional_residuals=down_samples, 
                                mid_block_additional_residual=mid_sample,
                                cross_attention_kwargs=cross_attention_kwargs
                            ).sample

                            # merge after unet
                            noise_pred = weight * noise_pred_cur

                            if texpref_config["kl_divergence"] == True: # if we want to use KL-divergence regularization                                
                                if t == timesteps[-1]: # if the current diffusion step is the last one, compute the KL-divergence term.                                    
                                     # NOTE: No gradient computation is needed, because we are only computing the noise predicted by the pre-trained model (epsilon_pt).
                                    noise_pred_cur_refernece_model = reference_unet_model(
                                        latent_model_input, 
                                        t, 
                                        encoder_hidden_states=prompt_embeds, 
                                        down_block_additional_residuals=down_samples, 
                                        mid_block_additional_residual=mid_sample,
                                        cross_attention_kwargs=cross_attention_kwargs
                                    ).sample
                                    noise_pred_refernece_model = weight * noise_pred_cur_refernece_model

                                    # KL-regularization at the final denoising step
                                    with torch.enable_grad():
                                        KL_term = ((noise_pred - noise_pred_refernece_model) ** 2).flatten(1).sum(dim=1).mean()
                                
                                else: # if this is not the last diffusion step, we just pass because we compute the KL-divergence term only at the last denoising diffusion step.
                                    pass

                            all_noise_preds.append(noise_pred)
                    else:
                        noise_pred = self.pipe.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        if diffusion_step < texpref_config["cfg_steps"]:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        else:
                            # For steps after cfg_steps, revert to unconditional prediction
                            noise_pred = noise_pred.chunk(2, dim=0)[0]  # Use only the unconditional prediction

                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)


                    # compute the previous noisy sample x_t -> x_t-1
                    latents = ddim_step_rlhf(self = self.pipe.scheduler,
                                                            model_output = noise_pred,
                                                            timestep = t,
                                                            sample = latents,
                                                            eta = eta,
                                                            generator = generator
                                                            )

                    all_latents.append(latents)

                # call the callback, if provided
                if diffusion_step == len(timesteps) - 1 or ((diffusion_step + 1) > num_warmup_steps and (diffusion_step + 1) % self.pipe.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and diffusion_step % callback_steps == 0:
                        callback(diffusion_step, t, latents)

        with torch.enable_grad():
            if not output_type == "latent":
                image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0].float()
                # image = (image / 2 + 0.5).clamp(0, 1) # This line was added from the Intex code to here            
            else:
                image = latents            

            do_denormalize = [True] * image.shape[0]
            image = self.pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.final_offload_hook.offload()

        if texpref_config["kl_divergence"] == True:
            return image, all_latents, all_noise_preds, KL_term
        else:
            return image, all_latents, all_noise_preds, None
