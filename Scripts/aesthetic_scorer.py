# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import kiui
import matplotlib.pyplot as plt

ASSETS_PATH = resources.files("Assets")

def differentiable_preprocess(image_tensor, target_size=(224, 224)):
    # Assume image_tensor is a float tensor in [0, 1] with shape [B, C, H, W]
    # Resize the image with bilinear interpolation (which is differentiable)
    with torch.enable_grad():
        processed = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
        # Apply normalization (e.g., subtract mean, divide by std)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_tensor.device).view(1, -1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_tensor.device).view(1, -1, 1, 1)
        processed = (processed - mean) / std

    return processed

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
    
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()
    
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}

        # plt.imshow(inputs["pixel_values"].squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()
        # kiui.vis.plot_image(inputs["pixel_values"].detach().cpu().numpy())

        with torch.enable_grad():
            embed = self.clip.get_image_features(**inputs)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True) # normalize embedding
            return self.mlp(embed).squeeze(1)        



class AestheticScorer_Differentiable(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()
    
    def __call__(self, images):
        # NOTE: Added this to replace the non-differentiable preprocessor that we had
        # before from the "CLIPProcessor". Just keep the previous code here for the reference:
        # device = next(self.parameters()).device
        # inputs = self.processor(images=images, return_tensors="pt")
        # inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        # To know how I actually was able to write the following function, you need to 
        # read the implementation of the "CLIPPreprocessor.__call__" function. Here is the
        # reference for that:
        inputs = {"pixel_values": differentiable_preprocess(images)}

        # plt.imshow(inputs["pixel_values"].squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()


        with torch.enable_grad():
            embed = self.clip.get_image_features(**inputs)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True) # normalize embedding
            return self.mlp(embed).squeeze(1)       