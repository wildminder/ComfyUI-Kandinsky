import torch
import gc
import comfy.model_patcher
import comfy.model_management as model_management
import comfy.utils

from .src.kandinsky.models.dit import DiffusionTransformer3D

KANDINSKY_CONFIGS = {
    "sft_5s": {"config": "config_5s_sft.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_5s.safetensors"},
    "sft_10s": {"config": "config_10s_sft.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_10s.safetensors"},
    "pretrain_5s": {"config": "config_5s_pretrain.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_pretrain_5s.safetensors"},
    "pretrain_10s": {"config": "config_10s_pretrain.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_pretrain_10s.safetensors"},
    "nocfg_5s": {"config": "config_5s_nocfg.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_nocfg_5s.safetensors"},
    "nocfg_10s": {"config": "config_10s_nocfg.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_sft_10s.safetensors"},
    "distil_5s": {"config": "config_5s_distil.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_distilled16steps_5s.safetensors"},
    "distil_10s": {"config": "config_10s_distil.yaml", "ckpt": "kandinsky/kandinsky5lite_t2v_distilled16steps_10s.safetensors"},
}

class KandinskyModelHandler(torch.nn.Module):
    """
    A lightweight placeholder for the Kandinsky DiT model.
    """
    def __init__(self, conf, ckpt_path):
        super().__init__()
        self.conf = conf
        self.ckpt_path = ckpt_path
        self.diffusion_model = None
        self.size = int(conf.model.dit_params.model_dim * 12 * 24 * 1.5)

class KandinskyPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher to load, patch, and manage the Kandinsky DiT model.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    @property
    def is_loaded(self) -> bool:
        return hasattr(self, 'model') and self.model is not None and self.model.diffusion_model is not None

    def patch_model(self, device_to=None, *args, **kwargs):
        if self.is_loaded:
            self.model.diffusion_model.to(self.load_device)
            return

        # print(f"Loading Kandinsky DiT model to {self.load_device}...")
        
        model_dtype = model_management.unet_dtype()

        dit_params = self.model.conf.model.dit_params
        model = DiffusionTransformer3D(**dit_params)
        
        model.to(dtype=model_dtype)
        
        sd = comfy.utils.load_torch_file(self.model.ckpt_path)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print("Kandinsky missing keys:", m)
        if len(u) > 0:
            print("Kandinsky unexpected keys:", u)

        model.eval()
        model.to(self.load_device)

        if model_management.force_channels_last():
            model.to(memory_format=torch.channels_last)

        self.model.diffusion_model = model
        # print("Kandinsky DiT model loaded.")
        return

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if self.is_loaded:
            # print(f"Offloading Kandinsky DiT model to {self.offload_device}...")
            self.model.diffusion_model.to(self.offload_device)

        if unpatch_weights:
             if self.is_loaded:
                del self.model.diffusion_model
                self.model.diffusion_model = None
             gc.collect()
             model_management.soft_empty_cache()
        return