import torch
from tqdm import trange
from typing_extensions import override
from comfy_api.latest import io
import comfy.utils
import comfy.model_management

from .src.kandinsky.magcache_utils import set_magcache_params
from .src.kandinsky.models.utils import fast_sta_nabla

@torch.no_grad()
def get_sparse_params(conf, batch_shape, device):
    F_cond, H_cond, W_cond, C_cond = batch_shape
    patch_size = conf.model.dit_params.patch_size
    
    T = F_cond // patch_size[0]
    H = H_cond // patch_size[1]
    W = W_cond // patch_size[2]

    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(T, H // 8, W // 8, conf.model.attention.wT,
                                  conf.model.attention.wH, conf.model.attention.wW, device=device)
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
        }
    else:
        sparse_params = None

    return sparse_params

@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
):
    model_input = x
    if dit.visual_cond:
        visual_cond_zeros = torch.zeros_like(x)
        visual_cond_mask = torch.zeros([*x.shape[:-1], 1], dtype=x.dtype, device=x.device)
        model_input = torch.cat([x, visual_cond_zeros, visual_cond_mask], dim=-1)

    pred_velocity = dit(
        model_input,
        text_embeds["text_embeds"],
        text_embeds["pooled_embed"],
        t * 1000,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=conf.metrics.scale_factor,
        sparse_params=sparse_params,
    )
    
    if abs(guidance_weight - 1.0) > 1e-6:
        uncond_pred_velocity = dit(
            model_input,
            null_text_embeds["text_embeds"],
            null_text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            null_text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
        )
        pred_velocity = uncond_pred_velocity + guidance_weight * (pred_velocity - uncond_pred_velocity)
        
    return pred_velocity

@torch.no_grad()
def generate(
    diffusion_model, 
    device, 
    shape, 
    steps, 
    text_embed, 
    null_embed,
    visual_rope_pos, 
    text_rope_pos, 
    null_text_rope_pos,
    cfg, 
    scheduler_scale, 
    conf,
    seed,
    pbar
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    
    model_dtype = next(diffusion_model.parameters()).dtype
    current_latent = torch.randn(shape, generator=g, device=device, dtype=model_dtype)
    
    sparse_params = get_sparse_params(conf, shape, device)

    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=model_dtype)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)
    
    for i in range(steps):
        t_now = timesteps[i]
        t_next = timesteps[i+1]
        dt = t_next - t_now
        
        pred_velocity = get_velocity(
            diffusion_model,
            current_latent,
            t_now.unsqueeze(0),
            text_embed,
            null_embed,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            cfg,
            conf,
            sparse_params=sparse_params
        )
        
        current_latent = current_latent + dt * pred_velocity
        pbar.update(1)
    
    return current_latent

class KandinskySampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KandinskyV5_Sampler",
            display_name="Kandinsky 5 Sampler",
            category="Kandinsky",
            description="Performs the specific Flow Matching sampling loop for Kandinsky-5 models.",
            inputs=[
                io.Model.Input("model", tooltip="The Kandinsky 5 model patcher from the Kandinsky 5 Loader."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=200, tooltip="Number of sampling steps."),
                io.Float.Input("cfg", default=5.0, min=1.0, max=20.0, step=0.1),
                io.Float.Input("scheduler_scale", default=5.0, min=1.0, max=20.0, step=0.1),
                io.Conditioning.Input("positive", tooltip="Positive conditioning from Kandinsky 5 Text Encode."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning from Kandinsky 5 Text Encode."),
                io.Latent.Input("latent_image", tooltip="Empty latent from Empty Kandinsky 5 Latent."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, model, seed, steps, cfg, scheduler_scale, positive, negative, latent_image) -> io.NodeOutput:
        patcher = model
        
        comfy.model_management.load_model_gpu(patcher)
        k_handler = patcher.model
        diffusion_model = k_handler.diffusion_model
        conf = k_handler.conf
        device = patcher.load_device
        model_dtype = next(diffusion_model.parameters()).dtype

        if conf.get('use_magcache', False):
            if hasattr(conf, "magcache"):
                set_magcache_params(diffusion_model, conf.magcache.mag_ratios, steps, conf.model.guidance_weight == 1.0)

        latent = latent_image["samples"].to(device)
        B, C, F, H, W = latent.shape
        
        pos_cond = positive[0][1].get("kandinsky_embeds")
        neg_cond = negative[0][1].get("kandinsky_embeds")
        
        for key in pos_cond:
            pos_cond[key] = pos_cond[key].to(device=device, dtype=model_dtype)
            neg_cond[key] = neg_cond[key].to(device=device, dtype=model_dtype)
            
        patch_size = conf.model.dit_params.patch_size
        visual_rope_pos = [
            torch.arange(F // patch_size[0], device=device),
            torch.arange(H // patch_size[1], device=device),
            torch.arange(W // patch_size[2], device=device)
        ]
        
        text_rope_pos = torch.arange(pos_cond["text_embeds"].shape[0], device=device)
        null_text_rope_pos = torch.arange(neg_cond["text_embeds"].shape[0], device=device)

        output_latents = []
        pbar = comfy.utils.ProgressBar(steps * B)
        for i in range(B):
            current_seed = seed + i

            final_latent_unbatched = generate(
                diffusion_model, 
                device, 
                (F, H, W, C), 
                steps, 
                pos_cond, 
                neg_cond,
                visual_rope_pos, 
                text_rope_pos, 
                null_text_rope_pos,
                cfg, 
                scheduler_scale, 
                conf,
                current_seed,
                pbar
            )
            output_latents.append(final_latent_unbatched.permute(3, 0, 1, 2))

        final_latents = torch.stack(output_latents, dim=0)

        scaling_factor = 0.476986
        scaled_latents = final_latents / scaling_factor

        return io.NodeOutput({"samples": scaled_latents.to(comfy.model_management.intermediate_device())})