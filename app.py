"""
Fake Dataset Factory — Gradio App
Synthetic Medical Chest X-Ray Generation

A clean, modern interface for generating and evaluating synthetic chest X-rays.
"""

import os
import json
import random
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
import math

# conditional imports
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False

try:
    from diffusers import StableDiffusionImg2ImgPipeline, UNet2DModel, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "notebooks" / "outputs"

IMG_SIZE = 64
NOISE_DIM = 100
FEATURE_G = 64


# =============================================================================
# Custom CSS for a clean, modern look
# =============================================================================

CUSTOM_CSS = """
/* Main container */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Header styling */
.header-container {
    text-align: center;
    padding: 20px 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #e0e0e0;
}

/* Stats cards */
.stats-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
}

/* Model cards */
.model-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    transition: box-shadow 0.3s ease;
}

.model-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Best model highlight */
.best-model {
    border: 2px solid #10b981 !important;
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
}

/* Metric display */
.metric-display {
    font-size: 2em;
    font-weight: bold;
    color: #1f2937;
}

.metric-label {
    font-size: 0.9em;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Button styling */
.generate-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    border-radius: 8px !important;
    transition: transform 0.2s ease !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
}

/* Gallery styling */
.gallery-container {
    border-radius: 12px;
    overflow: hidden;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #6b7280;
    font-size: 0.9em;
    border-top: 1px solid #e0e0e0;
    margin-top: 40px;
}
"""


# =============================================================================
# Model Definitions
# =============================================================================

class DCGANGenerator(nn.Module):
    def __init__(self, noise_dim=100, channels=1, feature_g=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class WGANGPGenerator(nn.Module):
    def __init__(self, noise_dim=100, channels=1, feature_g=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# VQ-VAE components
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embed_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embed_dim)
        d = (z_flat ** 2).sum(dim=1, keepdim=True) + \
            (self.embedding.weight ** 2).sum(dim=1) - \
            2 * z_flat @ self.embedding.weight.t()
        indices = d.argmin(dim=1)
        z_q = self.embedding(indices).view(z.shape)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, codebook_loss, commitment_loss, indices


class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)


class VQVAEDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, embed_dim=64, num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = VQVAEEncoder(in_channels, hidden_dim, embed_dim)
        self.vq = VectorQuantizer(num_embeddings, embed_dim, beta)
        self.decoder = VQVAEDecoder(embed_dim, hidden_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, codebook_loss, commitment_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, codebook_loss, commitment_loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)


# Flow Matching v2 components
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + residual


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.enc1 = ResBlock(in_ch, base_ch, time_emb_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        self.enc4 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)
        self.attn_enc3 = SelfAttention(base_ch * 4, num_heads=4)
        self.attn_enc4 = SelfAttention(base_ch * 4, num_heads=4)
        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.mid_attn = SelfAttention(base_ch * 4, num_heads=4)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.up4 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, 2, 1)
        self.dec4 = ResBlock(base_ch * 8, base_ch * 4, time_emb_dim)
        self.attn_dec4 = SelfAttention(base_ch * 4, num_heads=4)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.dec3 = ResBlock(base_ch * 4, base_ch * 2, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.dec2 = ResBlock(base_ch * 2, base_ch, time_emb_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        e3 = self.attn_enc3(e3)
        e4 = self.enc4(self.down3(e3), t_emb)
        e4 = self.attn_enc4(e4)
        m = self.mid1(e4, t_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb)
        d4 = self.dec4(torch.cat([self.up4(m), e3], dim=1), t_emb)
        d4 = self.attn_dec4(d4)
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1), t_emb)
        return self.out(self.out_act(self.out_norm(d2)))


class ODEFunc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        t_batch = t.expand(x.shape[0])
        return self.model(x, t_batch)


# =============================================================================
# Model Registry
# =============================================================================

MODEL_CONFIGS = {
    "flow_matching_v2": {
        "name": "Flow Matching v2",
        "emoji": "🥇",
        "fid": 6.20,
        "description": "Best model — Attention + EMA + 100 epochs",
        "checkpoint": OUTPUTS_DIR / "06_flow_matching_v2" / "checkpoints" / "checkpoint_epoch_100.pt",
        "type": "flow_matching_v2"
    },
    "ddpm": {
        "name": "DDPM",
        "emoji": "🥈",
        "fid": 8.96,
        "description": "Diffusion model baseline",
        "checkpoint": OUTPUTS_DIR / "05_ddpm" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "ddpm"
    },
    "wgan_gp": {
        "name": "WGAN-GP",
        "emoji": "🥉",
        "fid": 11.10,
        "description": "Stable GAN with gradient penalty",
        "checkpoint": OUTPUTS_DIR / "03_wgan_gp" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "wgan_gp"
    },
    "dcgan": {
        "name": "DCGAN",
        "emoji": "4️⃣",
        "fid": 15.24,
        "description": "Classic GAN baseline",
        "checkpoint": OUTPUTS_DIR / "02_dcgan" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "dcgan"
    },
    "vqvae": {
        "name": "VQ-VAE",
        "emoji": "5️⃣",
        "fid": 46.59,
        "description": "Discrete codebook (no prior)",
        "checkpoint": OUTPUTS_DIR / "04_vqvae" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "vqvae"
    },
    "stable_diffusion": {
        "name": "Stable Diffusion",
        "emoji": "6️⃣",
        "fid": 94.71,
        "description": "img2img mode (different input)",
        "checkpoint": None,
        "check_path": OUTPUTS_DIR / "01_stable_diffusion" / "metrics.json",
        "type": "stable_diffusion"
    }
}

loaded_models = {}


def check_model_available(model_id: str) -> bool:
    config = MODEL_CONFIGS[model_id]
    if config["type"] == "stable_diffusion":
        check_path = config.get("check_path")
        return check_path and check_path.exists()
    checkpoint = config.get("checkpoint")
    return checkpoint and checkpoint.exists()


def get_available_models() -> dict:
    return {mid: check_model_available(mid) for mid in MODEL_CONFIGS}


def load_model(model_id: str):
    if model_id in loaded_models:
        return loaded_models[model_id]

    if not check_model_available(model_id):
        return None

    config = MODEL_CONFIGS[model_id]
    model_type = config["type"]

    if model_type == "stable_diffusion":
        if not DIFFUSERS_AVAILABLE:
            return None
        loaded_models[model_id] = "stable_diffusion"
        return "stable_diffusion"

    elif model_type == "dcgan":
        model = DCGANGenerator(NOISE_DIM, 1, FEATURE_G)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["netG_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "wgan_gp":
        model = WGANGPGenerator(NOISE_DIM, 1, FEATURE_G)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["netG_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "vqvae":
        model = VQVAE(in_channels=1, hidden_dim=128, embed_dim=64, num_embeddings=512, beta=0.25)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "ddpm":
        if not DIFFUSERS_AVAILABLE:
            return None
        model = UNet2DModel(
            sample_size=IMG_SIZE, in_channels=1, out_channels=1, layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        loaded_models[model_id] = {"model": model, "scheduler": scheduler}
        return loaded_models[model_id]

    elif model_type == "flow_matching_v2":
        if not TORCHDIFFEQ_AVAILABLE:
            return None
        model = AttentionUNet(in_ch=1, base_ch=64, time_emb_dim=256)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        # load EMA weights if available
        if "ema_shadow" in checkpoint:
            for name, param in model.named_parameters():
                if name in checkpoint["ema_shadow"]:
                    param.data = checkpoint["ema_shadow"][name]
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        loaded_models[model_id] = model
        return model

    return None


# =============================================================================
# Evaluation
# =============================================================================

def load_xrv_model():
    if not XRV_AVAILABLE:
        return None, None
    xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    xrv_model.to(device).eval()
    feature_extractor = nn.Sequential(*list(xrv_model.features.children()))
    feature_extractor.to(device).eval()
    return xrv_model, feature_extractor


def preprocess_for_xrv(img: Image.Image) -> torch.Tensor:
    img = img.convert('L').resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 255.0) * 2048 - 1024
    return torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    if len(real_features) < 2 or len(fake_features) < 2:
        return None
    mu_real, mu_fake = np.mean(real_features, axis=0), np.mean(fake_features, axis=0)
    sigma_real, sigma_fake = np.cov(real_features, rowvar=False), np.cov(fake_features, rowvar=False)
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps
    diff = mu_real - mu_fake
    try:
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        if not np.isfinite(covmean).all():
            return None
        fid = float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))
        return fid if np.isfinite(fid) else None
    except Exception:
        return None


# =============================================================================
# Generation Functions
# =============================================================================

def generate_gan_images(model, n_samples: int) -> list:
    images = []
    with torch.no_grad():
        for i in range(n_samples):
            noise = torch.randn(1, NOISE_DIM, 1, 1, device=device)
            fake = model(noise)
            arr = ((fake.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
    return images


def generate_sd_images(n_samples: int, target_class: str) -> list:
    if not DIFFUSERS_AVAILABLE:
        return []
    class_dir = DATA_DIR / target_class.lower()
    if not class_dir.exists():
        return []
    real_paths = sorted([p for p in class_dir.iterdir() if p.suffix == '.png'])[:n_samples]
    if not real_paths:
        return []

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None, requires_safety_checker=False
    ).to(device)
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()

    images = []
    prompt = "chest x-ray, medical radiograph, grayscale"
    neg_prompt = "color, artistic, painting, drawing, cartoon"

    for i, img_path in enumerate(real_paths):
        img = Image.open(img_path).convert('RGB').resize((512, 512), Image.LANCZOS)
        with torch.no_grad():
            result = pipe(prompt=prompt, negative_prompt=neg_prompt, image=img,
                          strength=0.7, guidance_scale=7.5, num_inference_steps=50,
                          generator=torch.Generator(device=device).manual_seed(SEED + i))
        gen_img = result.images[0].convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        images.append(gen_img)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images


def generate_vqvae_images(model, n_samples: int) -> list:
    images = []
    with torch.no_grad():
        for i in range(n_samples):
            indices = torch.randint(0, 512, (16 * 16,), device=device)
            z_q = model.vq.embedding(indices).view(1, 16, 16, 64).permute(0, 3, 1, 2).contiguous()
            fake = model.decode(z_q)
            arr = (fake.squeeze().cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
    return images


def generate_ddpm_images(model_dict, n_samples: int) -> list:
    model, scheduler = model_dict["model"], model_dict["scheduler"]
    images = []
    with torch.no_grad():
        for i in range(n_samples):
            torch.manual_seed(SEED + i)
            sample = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)
            for t in scheduler.timesteps:
                model_output = model(sample, t).sample
                sample = scheduler.step(model_output, t, sample).prev_sample
            arr = ((sample.squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
    return images


def generate_flow_matching_images(model, n_samples: int) -> list:
    if not TORCHDIFFEQ_AVAILABLE:
        return []
    images = []
    ode_func = ODEFunc(model)
    with torch.no_grad():
        for i in range(n_samples):
            torch.manual_seed(SEED + i)
            x0 = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)
            t_span = torch.tensor([0.0, 1.0], device=device)
            x1 = odeint(ode_func, x0, t_span, method='dopri5', rtol=1e-5, atol=1e-5)[-1]
            arr = ((x1.squeeze().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
    return images


def generate_images(model_id: str, target_class: str, n_samples: int, progress=gr.Progress()):
    model = load_model(model_id)
    if model is None:
        return None, None, None, None, "Model not available"

    config = MODEL_CONFIGS[model_id]
    progress(0.1, desc=f"Generating with {config['name']}...")

    # generate
    if config["type"] == "stable_diffusion":
        images = generate_sd_images(n_samples, target_class)
    elif config["type"] in ["dcgan", "wgan_gp"]:
        images = generate_gan_images(model, n_samples)
    elif config["type"] == "vqvae":
        images = generate_vqvae_images(model, n_samples)
    elif config["type"] == "ddpm":
        images = generate_ddpm_images(model, n_samples)
    elif config["type"] == "flow_matching_v2":
        images = generate_flow_matching_images(model, n_samples)
    else:
        return None, None, None, None, "Unknown model type"

    if not images:
        return None, None, None, None, "Generation failed"

    progress(0.6, desc="Evaluating...")

    # evaluate
    fid_score, tstr_accuracy = None, None
    if XRV_AVAILABLE:
        xrv_model, feature_extractor = load_xrv_model()
        fake_features, pneumonia_scores = [], []
        pathology_names = xrv_model.pathologies
        pneumonia_idx = next((i for i, n in enumerate(pathology_names) if 'lung opacity' in n.lower()), None)

        for img in images:
            inp = preprocess_for_xrv(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = feature_extractor(inp).mean(dim=[2, 3])
                fake_features.append(feat.cpu().numpy())
                pred = xrv_model(inp)
                if pneumonia_idx is not None:
                    pneumonia_scores.append(pred[0, pneumonia_idx].item())

        fake_features = np.concatenate(fake_features, axis=0)

        class_dir = DATA_DIR / target_class.lower()
        if class_dir.exists():
            real_paths = sorted([p for p in class_dir.iterdir() if p.suffix == '.png'])
            real_sample = random.sample(real_paths, min(n_samples, len(real_paths)))
            real_features = []
            for p in real_sample:
                img = Image.open(p)
                inp = preprocess_for_xrv(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = feature_extractor(inp).mean(dim=[2, 3])
                    real_features.append(feat.cpu().numpy())
            if real_features:
                real_features = np.concatenate(real_features, axis=0)
                fid_score = compute_fid(real_features, fake_features)

        if pneumonia_scores:
            threshold = 0.5
            if target_class.lower() == "pneumonia":
                tstr_accuracy = sum(s > threshold for s in pneumonia_scores) / len(pneumonia_scores) * 100
            else:
                tstr_accuracy = sum(s <= threshold for s in pneumonia_scores) / len(pneumonia_scores) * 100

    progress(0.9, desc="Creating download...")

    # zip
    tmp_dir = tempfile.mkdtemp()
    zip_dir = Path(tmp_dir) / "generated"
    zip_dir.mkdir()

    for i, img in enumerate(images):
        img.save(zip_dir / f"{i:04d}.png", "PNG")

    zip_path = Path(tmp_dir) / f"{model_id}_{target_class}_generated.zip"
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', zip_dir)

    status = f"Generated {len(images)} images with {config['name']}"
    return images, fid_score, tstr_accuracy, str(zip_path), status


# =============================================================================
# Gradio Interface
# =============================================================================

def create_app():
    available = get_available_models()
    n_available = sum(available.values())

    with gr.Blocks(title="Fake Dataset Factory", css=CUSTOM_CSS, theme=gr.themes.Soft()) as app:

        # Header
        gr.Markdown("""
        <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid #e0e0e0;">
            <h1 style="margin: 0; font-size: 2.5em;">Fake Dataset Factory</h1>
            <p style="margin: 10px 0 0 0; color: #6b7280; font-size: 1.1em;">
                Synthetic Medical Chest X-Ray Generation
            </p>
        </div>
        """)

        # Stats row
        with gr.Row():
            gr.Markdown(f"""
            <div style="display: flex; justify-content: center; gap: 40px; padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 12px; margin: 20px 0;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5em; font-weight: bold; color: #0369a1;">6.20</div>
                    <div style="color: #6b7280; font-size: 0.9em;">Best FID Score</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5em; font-weight: bold; color: #0369a1;">6</div>
                    <div style="color: #6b7280; font-size: 0.9em;">Models Compared</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5em; font-weight: bold; color: #0369a1;">{n_available}</div>
                    <div style="color: #6b7280; font-size: 0.9em;">Models Ready</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2.5em; font-weight: bold; color: #0369a1;">{"GPU" if torch.cuda.is_available() else "CPU"}</div>
                    <div style="color: #6b7280; font-size: 0.9em;">Device</div>
                </div>
            </div>
            """)

        # Main interface
        with gr.Row():
            # Left panel - controls
            with gr.Column(scale=1):
                gr.Markdown("### Select Model")

                model_dropdown = gr.Dropdown(
                    choices=[
                        (f"{c['emoji']} {c['name']} (FID: {c['fid']:.2f})", mid)
                        for mid, c in MODEL_CONFIGS.items()
                        if available.get(mid, False)
                    ],
                    value="flow_matching_v2" if available.get("flow_matching_v2") else None,
                    label="Model",
                    info="Lower FID = Better quality"
                )

                class_dropdown = gr.Dropdown(
                    choices=["Pneumonia", "Normal"],
                    value="Pneumonia",
                    label="Target Class"
                )

                n_samples = gr.Slider(
                    minimum=1, maximum=100, value=16, step=1,
                    label="Number of Images"
                )

                generate_btn = gr.Button(
                    "Generate Images",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("---")

                # Results metrics
                gr.Markdown("### Results")

                with gr.Row():
                    fid_display = gr.Number(label="FID Score", precision=2)
                    tstr_display = gr.Number(label="TSTR %", precision=1)

                status_text = gr.Textbox(label="Status", interactive=False)

                download_btn = gr.File(label="Download ZIP")

            # Right panel - gallery
            with gr.Column(scale=2):
                gr.Markdown("### Generated Images")

                gallery = gr.Gallery(
                    label="",
                    columns=4,
                    rows=4,
                    height=500,
                    object_fit="contain",
                    show_label=False
                )

        # Leaderboard
        gr.Markdown("---")
        gr.Markdown("### Model Leaderboard")

        leaderboard_data = [
            [c["emoji"], c["name"], f"{c['fid']:.2f}", c["description"], "Ready" if available.get(mid) else "Not trained"]
            for mid, c in MODEL_CONFIGS.items()
        ]

        gr.Dataframe(
            value=leaderboard_data,
            headers=["Rank", "Model", "FID", "Description", "Status"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )

        # Footer
        gr.Markdown("""
        <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 0.9em; border-top: 1px solid #e0e0e0; margin-top: 40px;">
            <p><strong>Fake Dataset Factory</strong> — CSET419 Generative AI Project</p>
            <p>Evaluation uses domain-adapted FID with torchxrayvision DenseNet121 features</p>
        </div>
        """)

        # Event handler
        def on_generate(model_id, target_class, n):
            if not model_id:
                return [], None, None, None, "Please select a model"
            images, fid, tstr, zip_path, status = generate_images(model_id, target_class, int(n))
            return images or [], fid, tstr, zip_path, status

        generate_btn.click(
            fn=on_generate,
            inputs=[model_dropdown, class_dropdown, n_samples],
            outputs=[gallery, fid_display, tstr_display, download_btn, status_text]
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
