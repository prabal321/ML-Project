"""
encoders.py - frozen CLIP visual + text encoders.

We use Hugging Face transformers to load OpenAI's CLIP. Both encoders are
kept in eval() mode and have requires_grad=False - they will not be
fine-tuned. The training pipeline only learns the fusion + GRU + policy
modules on top of these features.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_DIM = 512


class CLIPEncoders(nn.Module):
    """Wraps a frozen CLIP model providing image and text encoding."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = "cpu"):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.clip.eval()
        self.device = torch.device(device)
        self.clip.to(self.device)
        self.feature_dim = FEATURE_DIM

    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        if isinstance(images, torch.Tensor):
            np_images = images.detach().to(torch.uint8).cpu().numpy()
            np_images = [np_images[i].transpose(1, 2, 0) for i in range(np_images.shape[0])]
        else:
            np_images = images
        inputs = self.processor(images=np_images, return_tensors="pt").to(self.device)
        feats = self.clip.get_image_features(**inputs)
        return feats

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        feats = self.clip.get_text_features(**inputs)
        return feats


if __name__ == "__main__":
    print("Loading CLIP encoders (will download ~600 MB on first run)...")
    enc = CLIPEncoders(device="cpu")
    print(f"Loaded. Feature dim = {enc.feature_dim}")

    texts = ["walk to the kitchen", "stop at the red door"]
    txt_feats = enc.encode_text(texts)
    print(f"Text features shape: {txt_feats.shape}  (expected (2, 512))")

    fake_img = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8).float()
    img_feats = enc.encode_image(fake_img)
    print(f"Image features shape: {img_feats.shape}  (expected (2, 512))")

    a = enc.encode_text(["a photo of a kitchen"])
    b = enc.encode_text(["a photo of a stove"])
    c = enc.encode_text(["a photo of the ocean"])
    cos = torch.nn.functional.cosine_similarity
    print(f"\nSanity check (CLIP knows semantics):")
    print(f"  cos(kitchen, stove) = {cos(a, b).item():+.3f}  (should be > 0.7)")
    print(f"  cos(kitchen, ocean) = {cos(a, c).item():+.3f}  (should be < 0.7)")

    print("\nencoders.py smoke test complete.")
