<!-- Improved compatibility of back to top link -->
<a id="readme-top"></a>

<div align="center">
  <h1 align="center">ComfyUI-Kandinsky</h1>

<img width="80%" alt="ComfyUI-Kandinsky logo" src="https://github.com/user-attachments/assets/6011acb2-b8ab-4979-ae5e-a5825025d32c" />

  
  <p align="center">
    A custom node for ComfyUI that integrates <strong>Kandinsky 5.0</strong>, a powerful family of open-source text-to-video diffusion models.
    <br />
    <br />
    <a href="https://github.com/wildminder/ComfyUI-Kandinsky/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/wildminder/ComfyUI-Kandinsky/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]

</div>

<br>

## About The Project

This project brings the state-of-the-art **Kandinsky 5.0 T2V Lite** text-to-video model into the ComfyUI ecosystem. Kandinsky 5 is a latent diffusion pipeline built on a Flow Matching and Diffusion Transformer (DiT) backbone, capable of generating high-quality video from text prompts.

It leverages a powerful combination of **Qwen2.5-VL** and **CLIP** for text conditioning and the **HunyuanVideo 3D VAE** for latent space encoding, enabling a nuanced understanding of prompts and impressive visual results.

<div align="center">
  
</div>
  
This custom node suite provides all the necessary tools to run the Kandinsky 5 pipeline natively in ComfyUI, including a custom sampler for its specific inference loop and efficient memory management to run on consumer-grade hardware.

**✨ Key Features:**
*   **Native Kandinsky 5.0 Integration**
*   **High-Quality Video Generation**
*   **Custom Sampler Node**
*   **Efficient Memory Management**
*   **Multiple Model Variants:** Supports SFT (high quality), no-CFG (faster), and distilled (fastest) model versions.
*   **Familiar ComfyUI Workflow**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🚀 Getting Started

The easiest way to install is via **ComfyUI Manager**. Search for `ComfyUI-Kandinsky` and click "Install".

Alternatively, to install manually:

1.  **Clone the Repository:**
    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```sh
    git clone https://github.com/wildminder/ComfyUI-Kandinsky.git
    ```

2.  **Install Dependencies:**
    This node relies on packages from the original Kandinsky repository. Navigate into the cloned `ComfyUI-Kandinsky` directory and install the required dependencies:
    ```sh
    cd ComfyUI-Kandinsky
    pip install -r requirements.txt
    ```

3.  **Download Models:**
    This node does **not** automatically download models. You must download the required models and place them in the correct ComfyUI directories. See the **Model Zoo** table below for links.
    *   Place **Kandinsky DiT models** (`.safetensors`) in `ComfyUI/models/diffusion_models/kandinsky/`.
    *   Place the **HunyuanVideo VAE** in `ComfyUI/models/vae/`.
    *   Place the **CLIP-L** and **Qwen2.5-VL** text encoders in `ComfyUI/models/clip/`.

4.  **Start/Restart ComfyUI:**
    Launch ComfyUI. The Kandinsky nodes will appear under the `Kandinsky` category.

## Model Zoo

The `Kandinsky 5 Loader` node uses the config name to identify the correct checkpoint file from the `kandinsky/` subdirectory in your `diffusion_models` folder.

### Kandinsky DiT Models

| Model                               | Config Name              | Duration | Hugging Face Link |
|:------------------------------------|:-------------------------|:--------:|:------------------|
| Kandinsky 5.0 T2V Lite SFT 5s       | `config_5s_sft.yaml`     | 5s       |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s/tree/main/model) |
| Kandinsky 5.0 T2V Lite SFT 10s      | `config_10s_sft.yaml`    | 10s      |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s/tree/main/model) |
| Kandinsky 5.0 T2V Lite pretrain 5s  | `config_5s_pretrain.yaml`| 5s       |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s/tree/main/model) |
| Kandinsky 5.0 T2V Lite pretrain 10s | `config_10s_pretrain.yaml`| 10s      |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-10s/tree/main/model) |
| Kandinsky 5.0 T2V Lite no-CFG 5s    | `config_5s_nocfg.yaml`   | 5s       |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s/tree/main/model) |
| Kandinsky 5.0 T2V Lite no-CFG 10s   | `config_10s_nocfg.yaml`  | 10s      |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-10s/tree/main/model) |
| Kandinsky 5.0 T2V Lite distill 5s   | `config_5s_distil.yaml`  | 5s       |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s/tree/main/model) |
| Kandinsky 5.0 T2V Lite distill 10s  | `config_10s_distil.yaml` | 10s      |🤗 [HF](https://huggingface.co/ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-10s/tree/main/model) |

### Required Dependency Models
These are common models used in many ComfyUI workflows and are required for the Kandinsky.

| Model                               | Purpose                     | Hugging Face Link |
|:------------------------------------|:----------------------------|:------------------|
| **HunyuanVideo VAE**                | Latent Encoding/Decoding    |🤗 [HF](https://huggingface.co/hunyuanvideo-community/HunyuanVideo/tree/main/vae) |
| **HunyuanVideo VAE bf16**                | Latent Encoding/Decoding    |🤗 [HF ComfyUI](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/vae) |
| **CLIP-ViT-L-14**                   | Text Conditioning  |🤗 [HF](https://huggingface.co/openai/clip-vit-large-patch14) |
| **Qwen2.5-VL-7B fp8 scaled**                   | Text Conditioning  |🤗 [HF ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders) |
| **Qwen2.5-VL-7B bf16**                   | Text Conditioning   |🤗 [HF Kijai](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Qwen/Qwen2.5_7B_instruct_bf16.safetensors) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🛠️ Node Parameters

> [!NOTE]
> The quality of the generated video is highly dependent on the quality of your prompt. The output strongly depends on both the user prompt and the underlying system prompt used by the Qwen2.5-VL encoder. Experiment with descriptive phrasing to achieve the best results.

### Kandinsky 5 Loader
*   `variant`: Select the Kandinsky DiT model variant to load. The name corresponds to the config files.

### Kandinsky 5 Text Encode
*   `clip`: The standard CLIP-L model.
*   `qwen_vl`: The Qwen2.5-VL model. **Must be loaded with the `qwen_image` type in the CLIPLoader node.**
*   `text`: The positive text prompt describing the desired video.
*   `negative_text`: The negative text prompt describing what to avoid.
*   `content_type`: Sets the internal prompt template for either `video` or `image` generation.

### Empty Kandinsky 5 Latent
*   `width`/`height`: The dimensions of the video to be generated.
*   `time_length`: The desired duration of the video in seconds. Set to `0` for single image generation.
*   `batch_size`: The number of videos to generate in one run.

### Kandinsky 5 Sampler
*   `seed`: The random seed used for creating the initial noise.
*   `steps`: The number of sampling steps. Should generally match the model type (e.g., 50 for `sft` models, 16 for `distill` models).
*   `cfg`: Classifier-Free Guidance scale. Higher values increase adherence to the prompt.
*   `scheduler_scale`: Controls the timestep distribution during sampling.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📊 Performance

Video generation is computationally intensive. As a baseline, generating a 5-second video (768x512) with the **`pretrain_5s`** model on an **NVIDIA 4070Ti (16GB VRAM)** can take approximately **15 minutes**. Distilled models will be significantly faster.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ⚠️ Risks and Limitations
*   **Potential for Misuse:** The ability to generate video from text could be misused. Users of this node must not use it to create content that infringes upon the rights of individuals or is intended to mislead or harm. It is strictly forbidden to use this for any illegal or unethical purposes.
*   **Technical Limitations:** The model may occasionally struggle with very long, complex prompts or maintaining perfect temporal consistency.
*   **Language Support:** The model is trained primarily on English and has a strong understanding of Russian concepts. Performance on other languages is not guaranteed.
*   This node is released for research and development purposes. Please use it responsibly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="center">══════════════════════════════════</p>

Beyond the code, I believe in the power of community and continuous learning. I invite you to join the 'TokenDiff AI News' and 'TokenDiff Community Hub'

<table border="0" align="center" cellspacing="10" cellpadding="0">
  <tr>
    <td align="center" valign="top">
      <h4>TokenDiff AI News</h4>
      <a href="https://t.me/TokenDiff">
        <img width="50%" alt="tokendiff-tg-qw" src="https://github.com/user-attachments/assets/e29f6b3c-52e5-4150-8088-12163a2e1e78" />
      </a>
      <p><sub>🗞️ AI for every home, creativity for every mind!</sub></p>
    </td>
    <td align="center" valign="top">
      <h4>TokenDiff Community Hub</h4>
      <a href="https://t.me/TokenDiff_hub">
        <img width="50%" alt="token_hub-tg-qr" src="https://github.com/user-attachments/assets/da544121-5f5b-4e3d-a3ef-02272535929e" />
      </a>
      <p><sub>💬 questions, help, and thoughtful discussion.</sub> </p>
    </td>
  </tr>
</table>

<p align="center">══════════════════════════════════</p>

<!-- LICENSE -->
## License

This custom node is subject to its own repository license. The Kandinsky 5 model and its components are subject to the license provided by the original authors at the [AI Forever Kandinsky-5 repository](https://github.com/ai-forever/Kandinsky-5).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

*   **The AI Forever team** for creating and open-sourcing the incredible [Kandinsky 5](https://github.com/ai-forever/Kandinsky-5) project.
*   **Qwen Team** for [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL).
*   **OpenAI** for [CLIP](https://github.com/openai/CLIP).
*   **Tencent** for the [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) VAE.
*   **The ComfyUI team** for their powerful and extensible platform.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-Kandinsky.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-Kandinsky/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-Kandinsky.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-Kandinsky/issues
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-Kandinsky.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-Kandinsky/network/members
