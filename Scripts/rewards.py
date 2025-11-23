from PIL import Image
import io
import numpy as np
import torch
from Configuration.Config import texgen_config, texpref_config
import torch.nn.functional as F

def light_reward():
    def _fn(images, prompts, metadata):
        reward = images.reshape(images.shape[0],-1).mean(1)
        return np.array(reward.cpu().detach()),{}
    return _fn


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from aesthetic_scorer import AestheticScorer
    import matplotlib.pyplot as plt

    scorer = AestheticScorer(dtype=torch.float32).cuda()    
    
    # This was added to prevent the parameters of the reward model from getting updated 
    # during the backpropagation process.
    for param in scorer.parameters():
        param.requires_grad = False

    def _fn(images, prompts, metadata):
        # NOTE: The "round()" operation prevent from updating the 
        # parameters of the UNet during the backpropagation. TBH,
        # I still don't know why this is happening and I need to 
        # figure out its root cause.
        # As a simple workaround, I just tried to remove it for now.
        # But, it's important to know whether this affect the way the
        # aesthetic reward rates the images or not.
        
        # plt.imshow(images.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        images = (images * 255).round().clamp(0, 255)

        scores = scorer(images)
        return scores, scorer, {}

    return _fn


def aesthetic_score_differentiable():
    from aesthetic_scorer import AestheticScorer_Differentiable    

    scorer = AestheticScorer_Differentiable(dtype=torch.float32).cuda()    

    # This was added to prevent the parameters of the reward model from getting updated 
    # during the backpropagation process.
    for param in scorer.parameters():
        param.requires_grad = False

    def _fn(images, prompts, metadata):
        scores = scorer(images)
        return scores, scorer, {}

    return _fn



def curvature_score_mse():
    def _fn(image, image_reward, prompts, metadata):        
        
        # Ensure both images are in the same dtype and range
        # image = (image * 255).round().clamp(0, 255)
        # image_reward = (image_reward * 255).round().clamp(0, 255)

        # Compute L2 (Mean Squared Error) loss using PyTorch tensors
        mse_loss = torch.nn.MSELoss()
        reward = -mse_loss(image, image_reward)
        
        return reward, {}

    return _fn


def curvature_score_smooth():
    def _fn(image, curvature_map, prompts, metadata):
        
        """
        image: [1,3,H,W]  This is the texture image (unwrapped from 3D), with 3 channels (RGB), values ∈ [0, 1]
        curvature_map: [1,1,H,W]  This is the UV-space curvature map, with values ∈ [-1, 1] where:
                             -1 could represent low/negative curvature (valleys),
                             +1 could represent high/positive curvature (peaks).
        Returns scalar reward.
        """
        # Rearrange to [C, H, W] then add batch dimension to get [1, C, H, W]
        image = image.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, 1024, 1024]

        red  = image[:,0:1] # Extract Red channel -> [1,1,H,W]
        blue = image[:,2:3] # Extract Blue channel -> [1,1,H,W]

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
        per_pixel_reward = curvature_map * red_blue   # [H,W] * [H,W]

        # Averages the contribution across all pixels to produce a single scalar reward.
        reward = per_pixel_reward.mean()
        
        return reward, {}

    return _fn

# Below is one solution that replaces all hard comparisons with smooth (sigmoid) approximations
# so that the entire reward function is differentiable. In this version, we “soft‐threshold” both
# the curvature and color difference values.

# In our goal we want:
# - For regions with high curvature (curv_map > curvature_thresh), we want the reward to be high if
#  the red channel dominates (red‐blue is high) and low (or negative) if the blue channel dominates.
# - For regions with low curvature (curv_map ≤ curvature_thresh), we want the reward to be high if 
# blue dominates and low if red dominates.

# To do this in a differentiable way we use two sigmoid functions:

# 1- A sigmoid on curvature to get a soft mask:

# - high_curv_mask = sigmoid(k * (curv_map – curvature_thresh))
# - low_curv_mask = 1 – high_curv_mask

# 2- A sigmoid on the color difference (red – blue) to detect red versus blue:
# - is_red = sigmoid(k2 * (red_blue_diff – red_blue_thresh))
# - is_blue = sigmoid(k2 * (–red_blue_diff – red_blue_thresh))

# Then, for each pixel:
# - If it is in a high-curvature region, we want reward = is_red – is_blue.
# - If it is in a low-curvature region, reward = is_blue – is_red. Finally, we average over pixels.
def curvature_score_sharp():
    def _fn(image, curvature_map, prompts, metadata):
        """
        Computes a differentiable curvature reward that encourages:
        - In high-curvature regions (curv_map > threshold): the texture should be red.
        - In low-curvature regions (curv_map <= threshold): the texture should be blue.

        All operations are made smooth using steep sigmoid functions so that gradients flow.

        Args:
            pred_tex: [1,3,H,W] tensor with RGB values in [0,1] (the predicted texture).
            curv_map: [1,1,H,W] tensor with curvature values in [-1,1].
            threshold: Curvature threshold (default 0.3) separating high from low curvature.
            k: Steepness for the curvature threshold sigmoid (higher means closer to a hard threshold).
            k2: Steepness for (red-blue) thresholding (if you want to enforce a minimum difference; here not used).
            color_thresh: (Optional) A further threshold on red-blue difference; if 0, not used.
            
        Returns:
            A scalar tensor representing the mean reward.
        """

        # Rearrange to [C, H, W] then add batch dimension to get [1, C, H, W]
        image = image.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, 1024, 1024]

        threshold = -0.7
        k = 100.0


        red  = image[:, 0:1]  # [1,1,H,W]
        blue = image[:, 2:3]
        red_blue_diff = red - blue  # Positive → red, Negative → blue in [-1, 1]

        # Use a steep sigmoid to create soft masks for high vs low curvature.
        # Where curv_map is much higher than threshold, high_mask ~ 1; 
        # where curv_map is much less than threshold, high_mask ~ 0.
        high_mask = torch.sigmoid(k * (curvature_map - threshold))
        low_mask  = 1.0 - high_mask

        # Optionally, if you want to enforce a minimum red-blue difference, you can do:
        # red_blue_mask = torch.sigmoid(k2 * (red_blue - color_thresh))   # near 1 if red_blue is high
        # blue_diff_mask = torch.sigmoid(k2 * (-red_blue - color_thresh))  # near 1 if red_blue is low
        # Otherwise, we assume the raw red_blue difference is used.

        # For high-curvature, we want red-blue to be high:
        reward_high = (red_blue_diff * high_mask).sum() / (high_mask.sum() + 1e-8)
        # For low-curvature, we want red-blue to be low (i.e., blue dominates), so negative red_blue:
        reward_low = -(red_blue_diff * low_mask).sum() / (low_mask.sum() + 1e-8)

        # Combine the two:
        total_reward = 0.5 * (reward_high + reward_low)


        return total_reward, {}
    
    return _fn



def mean_curvature_texture_alignment_reward():
    import kiui
    import torchvision.transforms.functional as TF

    """
    Computes a reward encouraging alignment between texture features and curvature map.

    Args:
        texture: Tensor of shape [1, 3, H, W], RGB texture image with values in [0, 1].
        curvature_map: Tensor of shape [1, 1, H, W], curvature values normalized to [0, 1].

    Returns:
        reward: Scalar tensor representing the alignment reward.
    """
    def _fn(texture_image, curvature_map, vector_field, prompts, metadata):

        ############################ 1- Convert texture to grayscale #######################################
        # Rearrange to [C, H, W] then add batch dimension to get [1, C, H, W]
        texture_image = texture_image.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, 1024, 1024]
        # Convert texture to grayscale
        texture_gray = TF.rgb_to_grayscale(texture_image, num_output_channels=1)
        # kiui.vis.plot_image(texture_gray.detach().cpu().numpy())

        ############################ 2- Compute gradients of the texture image ############################
        # Define Sobel filters (used for both texture and curvature gradient)
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=texture_image.device).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=texture_image.device).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_x.requires_grad_(False)
        sobel_y.requires_grad_(False)

        grad_x = F.conv2d(texture_gray, sobel_x, padding=1)
        grad_y = F.conv2d(texture_gray, sobel_y, padding=1)
        texture_grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        # kiui.vis.plot_image(texture_grad.detach().cpu().numpy())

        ################# 3- Compute gradient of curvature to approximate direction map ###################
        curvature_map = curvature_map.to(texture_image.device)
        if curvature_map.ndim == 2:
            curvature_map = curvature_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        curvature_dx = F.conv2d(curvature_map, sobel_x, padding=1)
        curvature_dy = F.conv2d(curvature_map, sobel_y, padding=1)
        curvature_grad = torch.sqrt(curvature_dx ** 2 + curvature_dy ** 2 + 1e-8)
        direction_map = torch.cat([curvature_dx / curvature_grad,
                                   curvature_dy / curvature_grad], dim=1)  # [1, 2, H, W]
        
    
        # Force a fixed, constant direction everywhere (e.g. purely horizontal to the right)
        # direction_map has shape [1,2,H,W]: [:,0] is x-component, [:,1] is y-component
        direction_map = torch.zeros_like(direction_map)
        direction_map[:, 0, :, :] = 0.0   # x = +1
        direction_map[:, 1, :, :] = 1.0   # y =  0


        ########################### 4- Compute alignment losses ###########################################
        # Normalize texture gradients and curvature map to [0, 1]
        # texture_grad_norm = (texture_grad - texture_grad.mean()) / (texture_grad.std() + 1e-6)
        # curvature_norm = (curvature_map - curvature_map.mean()) / (curvature_map.std() + 1e-6)

        texture_grad_norm = (texture_grad - texture_grad.min()) / (texture_grad.max() - texture_grad.min() + 1e-8)
        curvature_norm = (curvature_map - curvature_map.min()) / (curvature_map.max() - curvature_map.min() + 1e-8)

        # (a) Magnitude alignment
        loss_magnitude = F.mse_loss(texture_grad_norm, curvature_norm)

        # (b) Directional alignment
        dx, dy = direction_map[:, 0:1], direction_map[:, 1:2]
        proj = grad_x * dx + grad_y * dy
        proj = proj / (texture_grad + 1e-6)

        # Optional: weight directional loss only in high-curvature areas
        threshold = 0.6
        mask = (curvature_map > threshold).float()

        # Normalize texture gradients and curvature map to [0, 1]
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)

        loss_direction = F.mse_loss(proj_norm * mask, mask)


        
        # (c) Colorfulness loss (Hasler–Süsstrunk)
        R,G,B = texture_image[0,0], texture_image[0,1], texture_image[0,2]
        rg     = R - G
        yb     = 0.5*(R+G) - B
        μ_rg   = rg.mean()
        μ_yb   = yb.mean()
        sigma_rg   = rg.std(unbiased=False)
        sigma_yb   = yb.std(unbiased=False)
        colorfulness = sigma_rg + sigma_yb + 0.3*(μ_rg.abs() + μ_yb.abs())
        loss_color   = -colorfulness

        ########################### 5- Final reward ###########################################
        alpha_magnitutde = 1.0
        alpha_direction = 1.0
        alpha_color = 0.001
        total_loss = (alpha_magnitutde * loss_magnitude) + (alpha_direction * loss_direction) + (alpha_color * loss_color)

        reward = -total_loss

        return reward, {}

    return _fn



def minimum_curvature_texture_alignment_reward():
    import kiui
    import torchvision.transforms.functional as TF    

    """
    Computes a reward encouraging alignment between texture features and curvature map.
    This ensures our texture generation model is steered, via gradients, toward producing
    local image patterns whose edge directions align with the curvature‐derived vectors.

    Args:
        texture: Tensor of shape [1, 3, H, W], RGB texture image with values in [0, 1].
        curvature_map: Tensor of shape [1, 1, H, W], curvature values normalized to [0, 1].

    Returns:
        reward: Scalar tensor representing the alignment reward.
    """

    def sample_texture_gradients_at_uv(
        tex: torch.Tensor,
        uv: torch.Tensor,
        mode: str = 'bilinear',
        align_corners: bool = True
    ) -> torch.Tensor:
        """
        Sample unit gradient directions of `tex` at UV coords.
        This is because of the fact that the number of curvature
        vectors we have is less than the texture gradients. So,
        we sample the texture gradients at the same UV locations
        that the curvature gradients exist.

        Args:
        tex           : (C,H,W) or (1,H,W) texture tensor
        uv            : (2,N) UV coords in [0,1]
        mode          : sampling mode
        align_corners : grid_sample option

        Returns:
        grad_dirs     : (2,N) tensor of unit gradient vectors
        """
        _, C, H, W = tex.shape

        # Sobel kernels
        sobel = torch.tensor([[1, 0, -1], 
                              [2, 0, -2], 
                              [1, 0., -1]], dtype=tex.dtype, device=tex.device)
        sobel_u = sobel.view(1,1,3,3)
        sobel_v = sobel.t().view(1,1,3,3)

        # Sobel Filters: Applies horizontal and vertical 3×3 kernels to compute
        # du and dv with padding to keep shape (H, W).
        du = F.conv2d(tex, sobel_u, padding=1)[0,0]
        dv = F.conv2d(tex, sobel_v, padding=1)[0,0]

        # Normalization: ensures each sampled direction is a unit vector.
        mag = torch.sqrt(du*du + dv*dv + 1e-6)
        gu = du / mag
        gv = dv / mag
        grads = torch.stack([gu, gv], dim=0)  # (2,H,W)


        # Converts UV coordinates in [0,1] to [-1,1] grid coordinates.
        u_norm = uv[0] * 2 - 1
        v_norm = uv[1] * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1,-1,1,2)

        # Calls F.grid_sample(grads.unsqueeze(0), grid, ...) to interpolate at
        # our exact UV points which results in having a (2, N) tensor of texture direction vectors.
        sampled = F.grid_sample(
            grads.unsqueeze(0), grid,
            mode=mode,
            align_corners=align_corners
        )  # (1,2,N,1)
        grad_dirs = sampled[0,:, :, 0]  # (2,N)

        return grad_dirs


    def _fn(epoch, texture_image, uv, vec_uv, prompts, metadata):
        """
        Differentiable reward: alignment of texture gradients with curvature vectors.

        Args:
        tex                : (C,H,W) tensor
        uv                 : (2,N) UV coords
        vec_uv             : (2,N) target vectors (v already inverted)
        weight_by_magnitude: if True, weight alignment by gradient magnitude
        eps                : small constant

        Returns:
        reward             : scalar tensor
        """        
        DEBUG_FLAG = True

        ############################ 1- Convert texture to grayscale #######################################
        # Rearrange to [C, H, W] then add batch dimension to get [1, C, H, W]
        texture_image = texture_image.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, 1024, 1024]
        # Convert texture to grayscale
        texture_gray = TF.rgb_to_grayscale(texture_image, num_output_channels=1)
        # kiui.vis.plot_image(texture_gray.detach().cpu().numpy())

        ############################ 2- Compute gradients of the texture image ############################

        # Sample gradient dirs at UV points
        grad_uv = sample_texture_gradients_at_uv(texture_gray, uv)

        
        # Normalize curvature directions to unit length
        curv_mag = torch.sqrt((vec_uv**2).sum(dim=0, keepdim=True) + 1e-6)
        vec_uv = vec_uv / (curv_mag + 1e-6)

        # Computes squared dot‐product between sampled texture gradient and target curvature vectors.
        # This line is computing, for each UV sample point, the squared dot‐product (i.e. squared cosine similarity)
        # between your two 2D vectors: the texture‐gradient direction and the curvature direction. Here’s the breakdown:
        # 1- grad_uv * vec_uv
        #   Both grad_uv and vec_uv are shape (2, N)—two components (u‑ and v‑directions) for each of N points.
        #   The * does element‑wise multiplication, yielding a (2, N) tensor whose first row is grad_u * curv_u and second row is grad_v * curv_v
        # 2- .sum(dim=0)
        #   Summing along dimension 0 collapses those two rows into a single row of length N.
        align = (grad_uv * vec_uv).sum(dim=0).pow(2)


        # This caps the maximum per‐pixel reward at 0.8, so once a dot‑product’s magnitude reaches that threshold (i.e. |cos⁡𝜃| ≥ 0.8), it no longer increases your reward. In other words:
        # If two vectors are very well aligned or anti‑aligned (∣cos𝜃∣ above 0.8), you get the full 0.8 reward.
        # If they’re only moderately aligned (∣cos⁡𝜃∣ below 0.8), you get a proportional score equal to ∣cos⁡𝜃∣.
        # This has two practical effects:
        # 1- Prevents “over‐rewarding” already‐good alignments so that outlier pixels don’t dominate the average.
        # 2- Emphasizes poorly aligned regions, since their ∣cos𝜃∣ value directly scales the reward up until the 0.8 cap.
        
        # Why you might do this
        # - Reward shaping / saturation: In reinforcement‐style setups, it’s often helpful to saturate your reward beyond some “good enough” threshold. That focuses learning on bringing the worst‐aligned points up, rather than chasing diminishing gains on already‐good points.
        # - Robustness to noise: If certain UV locations are noisy—say, tiny texture‐gradient fluctuations—you don’t want extreme 
        # cos𝜃≈1 (which can happen by chance) to overwhelm the loss. Capping at 0.8 keeps the reward stable.
        # - Ignoring orientation flips: By taking absolute value, you say “I only care that the gradient lies along the curvature axis, not which direction along that axis.” If your textures are symmetric (e.g.\ stripes), flipping the gradient sign may be visually acceptable or indistinguishable, so you don’t want to penalize it.
        # - Linear vs. quadratic: Squaring cos𝜃 can make small misalignments contribute very little gradient early on (
        # cos𝜃 ≈ 0.1 → 0.01 squared). Using the linear (and clamped) version gives a stronger gradient signal for moderate misalignments until they reach the cap.

        # In summary, your new reward:
        # Measures “axis alignment” (via ∣cos𝜃∣) rather than full orientation.
        # Provides proportional reward for moderate alignments.
        # Saturates at 0.8 so that very‑well‑aligned pixels don’t drown out the “harder” cases.
        # Keeps gradients strong for mid‑range angles (down to ∣cos𝜃∣≈0), helping the network focus on bringing up the poorest alignments.
        # align = (grad_uv * vec_uv).sum(dim=0) 
        # align = torch.min(torch.abs(align), torch.tensor(0.4, device=align.device))

        r = align

        # Averages per-UV rewards to get a scalar.
        reward_direction = r.mean()




        ########################### 5- Final reward ###########################################
        alpha_direction = 1.0
        # alpha_colorfulness = 0.0
        # reward = (alpha_direction * reward_direction) + (alpha_colorfulness * reward_colorfulness)
        reward = (alpha_direction * reward_direction)


        # Draw red arrows for the texture gradients and blue arrows for the target vectors on top.
        if DEBUG_FLAG and ((epoch) % texpref_config["save_freq"] == 0):

            import matplotlib.pyplot as plt
            import os

            # 1) Convert texture to H×W×3 uint8 NumPy
            img_np = (texture_image.squeeze(0).detach().cpu().numpy() * 255).astype('uint8')
            img_np = img_np.transpose(1, 2, 0)  # (H,W,3)
            H, W, _ = img_np.shape

            # 2) UV → pixel coords
            px = uv[0].detach().cpu().numpy() * (W - 1)
            py = uv[1].detach().cpu().numpy() * (H - 1)

            # 3) Gradient & target vectors
            gu = grad_uv[0].detach().cpu().numpy()
            gv = grad_uv[1].detach().cpu().numpy()
            tx = vec_uv[0].detach().cpu().numpy()
            ty = vec_uv[1].detach().cpu().numpy()

            # 4) Subsample indices (every arrow or change step as needed)
            idxs = np.arange(0, uv.shape[1], 2)

            # 5) Build the figure
            fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
            # ax.imshow(img_np)  # default origin='upper', so image is not mirrored
            # ax.axis('off')

            # 6) Plot texture gradients in red
            ax.quiver(px[idxs], py[idxs],
                    gu[idxs], gv[idxs],
                    color='red', width=0.001, scale=60,
                    label='Texture Grad')

            # 7) Plot curvature vectors in blue
            ax.quiver(px[idxs], py[idxs],
                    tx[idxs], ty[idxs],
                    color='blue', width=0.001, scale=60,
                    label='Min Curvature')


            ax.legend(loc='upper right')
            plt.tight_layout()


            DATA_DIR = "../logs"
            out_path = os.path.join(DATA_DIR, f'{epoch}.png')
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0)

            # plt.show()

        return reward, {}

    return _fn



def symmetry_score():

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

    def _fn(texture_image, pixel_uv_orig, pixel_uv_mirror, patch_size, prompts, metadata):
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
            # Rearrange to [C, H, W] then add batch dimension to get [1, C, H, W]
            texture_image = texture_image.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, 1024, 1024]
            
            # 1) symmetry loss
            # Computes the symmetry loss for generated texture by comparing RGB values
            # sampled at UV coordinates corresponding to original vertices and their mirror.
            samples_orig = sample_texture_at_uv(texture_image, pixel_uv_orig, patch_size)   # shape [B, C, N]
            samples_mirror = sample_texture_at_uv(texture_image, pixel_uv_mirror, patch_size) # shape [B, C, N]
            loss_symmetry = F.mse_loss(samples_orig, samples_mirror)

            # 2) variance regularization (encourage non-uniform texture)
            # Variance term penalizes zero‑variance outputs, so pure black (zero everywhere) is not optimal.
            # We want the texture to have high variance (i.e. not collapse to a constant), which is equivalent
            # to maximizing the variance. So, to turn a “maximize Var” objective into something we minimize,
            # we negate it.
            var = texture_image.var()
            lambda_var = 0.5
            loss_var = -lambda_var * var

            # 3) Encourage detail by bossting high-frequency information
            # High‑frequency term rewards edges/detail, steering the network away from smooth constants.
            dx = texture_image[:, :, :, 1:] - texture_image[:, :, :, :-1]
            dy = texture_image[:, :, 1:, :] - texture_image[:, :, :-1, :]
            hf = -(dx.abs().mean() + dy.abs().mean())
            lambda_hf = 0.5
            loss_hf = lambda_hf * hf

            # total
            # loss = loss_symmetry + loss_var + loss_hf
            loss = loss_symmetry
            reward = -loss
            return reward, {}
    
    return _fn