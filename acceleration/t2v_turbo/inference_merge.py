import os
import random
import time
import torch
import torchvision
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from scheduler.t2v_turbo_scheduler import T2VTurboScheduler
from pipeline.t2v_turbo_vc2_pipeline import T2VTurboVC2Pipeline
from utils.utils import instantiate_from_config


class conv3d_to_conv2d(nn.Module):
    """Converts Conv3D layers to Conv2D for better compatibility."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.out_channels = out_channels
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        b, c, f, h, w = x.size()
        return self.conv2d(x.flatten(-2)).view(b, self.out_channels, f, h, w)


def replace_module(model):
    """Recursively replaces Conv3D layers with Conv2D where applicable."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv3d) and module.kernel_size[-1] == 1:
            transfer_conv = conv3d_to_conv2d(
                module.in_channels, module.out_channels,
                module.kernel_size[:2], module.stride[:2], module.padding[:2]
            )
            transfer_conv.conv2d.weight.data = module.weight.data.squeeze(-1)
            transfer_conv.conv2d.bias.data = module.bias.data
            setattr(model, name, transfer_conv)
        else:
            replace_module(module)


class InferencePipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.current_model = None
        self.pipeline = None
        self.models = {
            "7B": {
                "config": "configs/inference_t2v_512_v2.0_distil_07B.yaml",
                "weights": "07B_merged_all.pt"
            },
            "9B": {
                "config": "configs/inference_t2v_512_v2.0_distil_09B.yaml",
                "weights": "09B_merged_all.pt"
            }
        }

    def load_model(self, model_version):
        """Loads a model and ensures only one stays in memory at a time."""
        if self.current_model == model_version:
            print(f"Model {model_version} is already loaded. Skipping reload.")
            return

        # Offload previous model
        if self.pipeline is not None:
            del self.pipeline
            torch.cuda.empty_cache()
            self.pipeline = None

        print(f"Loading model {model_version}...")

        config_path = self.models[model_version]["config"]
        model_path = self.models[model_version]["weights"]
        print(model_path)
        print(config_path)
        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        print("Model Configuration:")
        print(model_config)

        pretrained_t2v = instantiate_from_config(model_config)
        print("passed 1")
        pretrained_t2v = torch.load(model_path, map_location=self.device)
        print("passed 2")
        pretrained_t2v.eval()
        print("passed 3")

        # Apply Conv3D -> Conv2D replacement if needed
        replace_module(pretrained_t2v)

        scheduler = T2VTurboScheduler(
            linear_start=model_config["params"]["linear_start"],
            linear_end=model_config["params"]["linear_end"],
        )
        self.pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
        self.pipeline.to(self.device)

        self.current_model = model_version
        print(f"Model {model_version} loaded successfully!")

    def generate_video(self, prompt, model_version, precision, seed, guidance_scale, num_inference_steps, num_frames, fps, save_path):
        """Generates a video given a prompt."""
        self.load_model(model_version)  # Ensure correct model is loaded

        seed_everything(seed)
        torch.manual_seed(seed)

        if precision == 16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = self.pipeline(
                    prompt=prompt,
                    frames=num_frames,
                    fps=fps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    num_videos_per_prompt=1,
                )
        else:
            result = self.pipeline(
                prompt=prompt,
                frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_videos_per_prompt=1,
            )

        video_path = self.save_videos(result, prompt, seed, guidance_scale, num_inference_steps, fps, save_path)
        return video_path, seed

    def save_videos(self, video_array, prompt, seed, guidance_scale, num_inference_steps, fps, save_path):
        """Saves video and returns the file path."""
        os.makedirs(save_path, exist_ok=True)
        metadata = {
            "prompt": prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }
        with ThreadPoolExecutor() as executor:
            paths = list(executor.map(
                self.save_video,
                video_array,
                [metadata] * len(video_array),
                [save_path] * len(video_array),
                [fps] * len(video_array),
            ))
        return paths[0]

    def save_video(self, vid_tensor, metadata, root_path, fps):
        """Helper function to save video to disk."""
        if metadata['prompt'].strip():
            unique_name = metadata['prompt'].strip().replace(" ", "_") + ".mp4"
        unique_name = os.path.join(root_path, unique_name)

        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(1, 0, 2, 3)  # t,c,h,w
        video = (video + 1.0) / 2.0
        video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

        torchvision.io.write_video(
            unique_name, video, fps=fps, video_codec="h264", options={"crf": "10"}
        )
        return unique_name
