# felix-diffusion-cli
# Felix's application for running SDXL/Illustrious models
import sys
import torchvision.transforms.functional as torchvision_tf
sys.modules['torchvision.transforms.functional_tensor'] = torchvision_tf

import os
import warnings
import logging
import gc
from datetime import datetime
import time
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import re
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from safetensors.torch import load_file

import sysconfig # for printing OS info only
import platform
import subprocess
import copy
import hashlib
import urllib.request
from safetensors import safe_open

from effects import effects_menu

THIS_DIR = Path(__file__).parent
OUTPUT_DIR = THIS_DIR / "output"
UPSCALE_OUTPUT_DIR = OUTPUT_DIR / "upscale_output"
RESOURCES_DIR = Path("/home/felix/Desktop/models/")
CHECKPOINT_DIR = RESOURCES_DIR / "checkpoints"
EMBEDDINGS_DIR = RESOURCES_DIR / "embeddings"
UPSCALERS_DIR = RESOURCES_DIR / "upscalers"
LORA_DIR = RESOURCES_DIR / "loras"
SETTINGS_JSON_PATH = THIS_DIR / "user_settings.json"

if os.path.exists(THIS_DIR / "ai-cache"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPSCALE_OUTPUT_DIR, exist_ok=True)
os.environ["HF_HOME"] = str(THIS_DIR / "ai-cache")
os.environ["HF_HUB_CACHE"] = str(THIS_DIR / "ai-cache" / "hub")
os.environ["DIFFUSERS_CACHE"] = str(THIS_DIR / "ai-cache" / "diffusers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import transformers
import diffusers
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, EulerDiscreteScheduler
from safetensors.torch import load_file
from compel import CompelForSDXL, DiffusersTextualInversionManager
transformers.logging.set_verbosity_error()
diffusers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*Already found a `peft_config` attribute.*")
warnings.filterwarnings("ignore", message=r".*upcast_vae is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"diffusers(\..*)?")

DEFAULT_SETTINGS = {
    "model": "waiIllustriousSDXL_v170.safetensors",
    "width": 1024,
    "height": 1344,
    "sampler": "Euler a",
    "steps": 28,
    "cfg": 5,
    "batch_size": 1,
    "lora": {
        "[style] CunnyfunkyV7": 0.8,
        "[style] CG look and anime aesthetics-MSS": 0.55,
        "[style] 0__11Xx": 0.55,
        "[style] Smooth_Booster_v4": 0.2,
    },
    "positive_prompt_a": (
        "lazypos, lazyup, Smooth_Quality, very aesthetic, cinematic lighting, ultra-detailed, beautiful detailed eyes,"
    ),
    "positive_prompt_b": (
        ", masterpiece, best quality, amazing quality, absurdres, highres, newest, 0__11Xx_illu"
    ),
    "negative_prompt": (
        "lazyneg, lazydn, lazyhand, lazyloli, lazywet, "
        "(worst quality, bad quality)1.2, low quality, (sweat)1.2, sketch, censor, low contrast, "
        "transparent, logo, text, jpeg artifacts, bad anatomy, old, early, "
        "copyright name, watermark, artist name, signature, bad hands, bad finger, "
        "lazyreal, 3d,"
    ),
    "auto_open_output": True,
    "force_enable_vae_tiling": False,
    "force_enable_model_cpu_offload": False,
    "force_disable_unet_compilation": False,
}

SAMPLER_OPTIONS = ["Euler", "Euler a", "DPM++ 2M SDE", "DPM++ SDE Karras"]

user_settings = {}
user_settings = copy.deepcopy(DEFAULT_SETTINGS) # avoid modifying the original default settings (eg. if I remove a lora in settings menu it actually mutates the original dictionary)

GRAY = "\033[90m"
RED = "\033[91m" # Error
GREEN = "\033[92m" # Finish, OK
YELLOW = "\033[93m" # Loading, doing something, menu
BLUE = "\033[94m" # Special
MAGENTA = "\033[95m" # Info/user input
CYAN = "\033[96m" # Settings/OS info
RESET = "\033[0m"

def prompt_builder(input: str) -> str:
    return f"{user_settings['positive_prompt_a']} {input} {user_settings['positive_prompt_b']}"

def generate_image(pipe, compel, prompt: str):
    with torch.no_grad():
        full_prompt = prompt_builder(prompt)
        if "pony" not in user_settings["model"]:
            neg_prompt = user_settings["negative_prompt"]
        else:
            neg_prompt = "score_6, score_5, score_4, " + user_settings["negative_prompt"]
        c = compel(full_prompt, negative_prompt=neg_prompt)

    base_seed = random.randint(0, 2**32 - 1) # Stable Diffusion seeds are uint32 integers (0 to 4,294,967,295)
    generator = torch.Generator("cuda").manual_seed(base_seed)
    print(YELLOW + f"Generating image... (CFGSCALE: {user_settings['cfg']}, STEPS: {user_settings['steps']}, SAMPLER: {user_settings['sampler']}, CLIPSKIP: 2)")
    with torch.inference_mode():
        image = pipe(
            prompt_embeds=c.embeds,
            pooled_prompt_embeds=c.pooled_embeds,
            negative_prompt_embeds=c.negative_embeds,
            negative_pooled_prompt_embeds=c.negative_pooled_embeds,
            num_inference_steps=user_settings["steps"],
            guidance_scale=user_settings["cfg"],
            # clip_skip=2, Compel already did PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED. Which is clip skip 2
            width=user_settings["width"],
            height=user_settings["height"],
            generator=generator,
            num_images_per_prompt=user_settings["batch_size"],
        ).images

    print(f"{GREEN}[  OK  ] Image generated.{RESET}")

    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")

    for i, img in enumerate(image):
        current_seed = base_seed + i # PyTorch increments seed by 1 for each image in a batch
        file_name = f"output_{time_str}_{i+1}.png"
        full_path = OUTPUT_DIR / file_name

        metadata = PngInfo()
        metadata.add_text("parameters", (
            f"{full_prompt}\n"
            f"Negative prompt: {user_settings["negative_prompt"]}\n"
            f"Steps: {user_settings["steps"]}, Sampler: {user_settings["sampler"]}, CFG scale: {user_settings["cfg"]}, Clip skip: 2, Seed: {current_seed}, Size: {user_settings["width"]}x{user_settings["height"]}, Model: {user_settings["model"]}"
        ))
        if user_settings["lora"]:
            text = ""
            for key, value in user_settings["lora"].items():
                text += f"{key}: {value}\n"
            metadata.add_text("lora", text)
        metadata.add_text("Software", "felix-diffusion-cli")
        img.save(full_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved image {i+1}/{len(image)}: {full_path}{RESET}")
        if user_settings["auto_open_output"]:
            if platform.system() == "Linux":
                subprocess.Popen(["xdg-open", str(full_path)])
            elif platform.system() == "Darwin": # Well...just in case... (MacOS)
                subprocess.Popen(["open", str(full_path)])
            elif platform.system() == "Windows":
                os.startfile(str(full_path))

    del image
    del c
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"{MAGENTA}[  INFO  ] VRAM after cleanup: {(total-free)/1024**3:.2f} / {total/1024**3:.2f} GiB used{RESET}")

    return full_path

def load_or_create_user_settings():
    if SETTINGS_JSON_PATH.exists():
        print(f"{YELLOW}[ LOAD ] Loading user settings from JSON.{RESET}")
        try:
            with open(SETTINGS_JSON_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"{RED}[ ERROR] Failed to load user settings: {e}{RESET}")
    else:
        print(f"{YELLOW}[ CREATE] Creating new user settings with default values.{RESET}")
        try:
            with open(SETTINGS_JSON_PATH, "w") as f:
                json.dump(user_settings, f, indent=4)
            print(f"{GREEN}[  OK  ] New user settings created and saved to JSON.{RESET}")
            return user_settings
        except Exception as e:
            print(f"{RED}[ ERROR] Failed to create new user settings: {e}{RESET}")

print(f"{BLUE}[ START] Starting felix-diffusion-cli...{RESET}")
user_settings = load_or_create_user_settings()
if not user_settings:
    print(f"{RED}[ ERROR] Failed to load or create user settings.{RESET}")
    exit(1)
print(f"{GREEN}[  OK  ] User settings loaded.{RESET}")

def load_sdxl_embedding(pipe):
    embeddings = [f for f in os.listdir(EMBEDDINGS_DIR) if os.path.isfile(os.path.join(EMBEDDINGS_DIR, f)) and f.endswith(".safetensors")]
    number_of_embeddings = len(embeddings)
    print(f"{YELLOW}    Detected {number_of_embeddings} embeddings in EMBEDDINGS_DIR...{RESET}")
    if number_of_embeddings == 0:
        print(f"{YELLOW}No embeddings, skipping the process...{RESET}")
    else:
        i = 1
        for emb in embeddings:
            if i != number_of_embeddings:
                print(f"{YELLOW}    ├─ Loading embeddings: {i}/{number_of_embeddings}... ({emb}){RESET}")
            else:
                print(f"{YELLOW}    └─ Loading embeddings: {i}/{number_of_embeddings}... ({emb}){RESET}")
            try:
                state = load_file(str(EMBEDDINGS_DIR / emb))
                # CLIP-L (text_encoder)
                pipe.load_textual_inversion(
                    state["clip_l"],
                    token=emb,
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                )
                # CLIP-G (text_encoder_2)
                pipe.load_textual_inversion(
                    state["clip_g"],
                    token=emb,
                    text_encoder=pipe.text_encoder_2,
                    tokenizer=pipe.tokenizer_2,
                )
            except Exception as e:
                print(f"{RED}[ ERROR] Failed to load embedding ({emb}): Not a SDXL embedding{RESET}")
            i = i + 1

def load_lora(pipe, lora_settings):
    number_of_lora = len(lora_settings)
    print(f"{YELLOW}    Detected {number_of_lora} LoRA in settings...{RESET}")
    if number_of_lora == 0:
        print(f"{YELLOW}No LoRA, skipping the process...{RESET}")
    else:
        failed_keys = []
        i = 1
        for key, weight in lora_settings.items():
            if i != number_of_lora:
                print(f"{YELLOW}    ├─ Loading LoRA: {i}/{number_of_lora}... ({key}){RESET}")
            else:
                print(f"{YELLOW}    └─ Loading LoRA: {i}/{number_of_lora}... ({key}){RESET}")
            try:
                lora_path = str(LORA_DIR / f"{key}.safetensors")

                # Load raw state dict from file
                state_dict = load_file(lora_path)
                
                # Filter out keys that cause shape mismatches with SDXL UNet / TE2
                unet_state_dict = pipe.unet.state_dict()
                filtered_state_dict = {}
                
                for k, v in state_dict.items():
                    # Check if key maps to UNet and verify shape compatibility
                    unet_key = k.replace("lora_unet_", "").replace("_", ".") # Basic translation
                    if unet_key in unet_state_dict:
                        if unet_state_dict[unet_key].shape == v.shape:
                            filtered_state_dict[k] = v
                    else:
                        filtered_state_dict[k] = v

                # Pass converted/filtered state dict to diffusers
                pipe.load_lora_weights(
                    filtered_state_dict,
                    adapter_name=key,
                )
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    print(f"{RED}[ ERROR] Failed to load LoRA ({key}): {e}{RESET}")
                else:
                    print(f"{RED}[ ERROR] Failed to load LoRA ({key}): Not a SDXL LoRA{RESET}")
                failed_keys.append(key)
            i = i + 1
        for key in failed_keys:
            del lora_settings[key]
            print(f"{YELLOW}[ LOAD ] Removed unsupported LoRA from settings: {key}{RESET}")
        if lora_settings:
            print(f"{YELLOW}[ LOAD ] Fusing LoRA...{RESET}")
            pipe.set_adapters(
                list(lora_settings.keys()),
                adapter_weights=list(lora_settings.values()),
            )
            pipe.fuse_lora()
            print(f"{GREEN}[  OK  ] Done fusing LoRA.{RESET}")
            print(f"{YELLOW}[ LOAD ] Unloading LoRA weights and clearing VRAM...{RESET}")
            pipe.unload_lora_weights()
        else:
            print(f"{YELLOW}[ LOAD ] No valid LoRA adapters loaded.{RESET}")
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"{MAGENTA}[  INFO  ] VRAM after cleanup: {(total-free)/1024**3:.2f} / {total/1024**3:.2f} GiB used{RESET}")

def upscale(
    image_path: str,
    model_path: str,
    output_path: str,
    device: str | None = None,
    use_fp16: bool = False,
    size_factor: float = 4.0,
):
    print(f"{YELLOW}[ LOAD ] Now upscaling...") # Not resetting the color so that the progress remains yellow during upscaling
    device_type = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_path_str = str(model_path)

    if "4x-UltraSharp.pth" in model_path_str:
        num_block_arg = 23
    else:
        num_block_arg = 6

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=num_block_arg,
        num_grow_ch=32,
        scale=4,
    )

    # To future me: 4x-UltraSharp uses old ESRGAN naming style, modern RRDBNet does not accept it so I translate it to the newer naming style
    if "4x-UltraSharp.pth" in model_path_str:
        print("Loading 4x-UltraSharp with proper old→new key conversion...")
        loadnet = torch.load(model_path_str, map_location="cpu", weights_only=False)
        if "params_ema" in loadnet:
            state_dict = loadnet["params_ema"]
        elif "params" in loadnet:
            state_dict = loadnet["params"]
        else:
            state_dict = loadnet
        new_state = {}
        for old_key, value in state_dict.items():
            if old_key.startswith("module."):
                old_key = old_key[7:] # strip possible "module."
            if old_key.startswith("model.0."):
                new_key = old_key.replace("model.0.", "conv_first.")
            elif old_key.startswith("model.1.sub.23."):
                new_key = old_key.replace("model.1.sub.23.", "conv_body.")
            elif old_key.startswith("model.1.sub."):
                parts = old_key.split(".")
                block_idx = parts[3]
                rdb = parts[4].lower()
                conv = parts[5]
                param = parts[-1]
                new_key = f"body.{block_idx}.{rdb}.{conv}.{param}"
            elif old_key.startswith("model.3."):
                new_key = old_key.replace("model.3.", "conv_up1.")
            elif old_key.startswith("model.6."):
                new_key = old_key.replace("model.6.", "conv_up2.")
            elif old_key.startswith("model.8."):
                new_key = old_key.replace("model.8.", "conv_hr.")
            elif old_key.startswith("model.10."):
                new_key = old_key.replace("model.10.", "conv_last.")
            else:
                new_key = old_key   # should not happen
            new_state[new_key] = value

        model.load_state_dict(new_state, strict=False) # Fill the network with the 4x-UltraSharp weights
        model.eval() # Put the network into inference/reading mode
        model = model.to(device_type) # Move the network onto GPU

        # To future me: I manually create a RealESRGANer object
        # RealESRGANer.__init__ is hard-coded to load the weights itself and it always uses strict=True.
        # Because UltraSharp doesn’t have the expected keys, the constructor kept exploding.
        upsampler = RealESRGANer.__new__(RealESRGANer)
        upsampler.scale = 4
        upsampler.tile_size = 512
        upsampler.tile_pad = 10
        upsampler.pre_pad = 0
        upsampler.mod_scale = None
        upsampler.half = False
        upsampler.device = torch.device(device_type)
        upsampler.model = model

    else:
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path_str,
            dni_weight=None,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=use_fp16,
            device=device_type,
        )

    # To future me: RealESRGANer was written assuming OpenCV-style BGR input. So we convert img to RGB -> numpy array -> convert to BGR
    raw_img = Image.open(image_path).convert("RGB")
    img_np = np.array(raw_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    output_bgr, _ = upsampler.enhance(img_bgr, outscale=size_factor)

    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(output_rgb)
    new_width, new_height = output_img.size

    metadata = PngInfo()
    for key, value in raw_img.info.items():
        if isinstance(value, str):
            if key == "parameters":
                new_params = re.sub(
                    r"Size: \d+x\d+", f"Size: {new_width}x{new_height}", value
                )
                metadata.add_text(key, new_params)
            else:
                metadata.add_text(key, value)

    output_img.save(output_path, pnginfo=metadata)
    print(f"{GREEN}[  OK  ] Upscaled -> {output_path} ({raw_img.size} -> {output_img.size}){RESET}")

    if user_settings["auto_open_output"]:
        if platform.system() == "Linux":
            subprocess.Popen(["xdg-open", str(output_path)])
        elif platform.system() == "Darwin": # Well...just in case... (MacOS)
            subprocess.Popen(["open", str(output_path)])
        elif platform.system() == "Windows":
            os.startfile(str(output_path))

    del upsampler, model, raw_img, img_np, img_bgr, output_bgr, output_img
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def inspect_metadata(file_path):
    if not os.path.exists(file_path):
        print(f"{RED}[ ERROR] File not found: {file_path}{RESET}")
        time.sleep(1)
        return
    if not file_path.endswith(".safetensors"):
        try:
            img = Image.open(file_path)
            print(f"{CYAN}Keys found in image metadata:{RESET}", img.info.keys())
            for key in img.info.keys():
                print(f"{CYAN}Key:{RESET} {key}")
                print(f"{img.info[key]}")
            print()
        except Exception as e:
            print(f"{RED}[ ERROR] Error opening image file: {e}{RESET}")
            time.sleep(1)
    else:
        try:
            with safe_open(file_path, framework="pt") as f: # pt = PyTorch
                meta = f.metadata() or {}
        except Exception as e:
            print(f"{RED}[ ERROR] Error opening file: {e}{RESET}")
            time.sleep(1)
            return

        # Calculates the SHA256 hash of the safetensors file.
        sha256 = hashlib.sha256()
        print(f"{YELLOW}[ LOAD ] Calculating SHA256 hash of the file...{RESET}")
        with open(file_path, "rb") as f:
            block_size = 1048576
            while chunk := f.read(block_size):
                sha256.update(chunk)
        file_hash = sha256.hexdigest().upper() # Now the whole file is represented by its SHA256 hash
        print(f"{GREEN}[  OK  ] File SHA256:{RESET} {file_hash}")

        # Queries Civitai API using the file hash.
        url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
        civitai_url = None
        try:
            print(f"{YELLOW}[ LOAD ] Querying Civitai API...{RESET}")
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    model_id = data.get("modelId")
                    version_id = data.get("id")
                    civitai_url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
                    model_name = data.get("model")
                    baseModel = data.get("baseModel")
        except Exception:
            print(f"{RED}[ ERROR] Error querying Civitai API.{RESET}")
            time.sleep(1)

        if civitai_url:
            print(f"{GREEN}[  OK  ] Civitai URL found via hash lookup.{RESET}")
            print(f"\n{CYAN}Civitai URL Found:{RESET} {civitai_url}")
            print(f"{CYAN}Model Name:{RESET} {model_name["name"]}")
            print(f"{CYAN}Base Model:{RESET} {baseModel}\n")
            #print("DEBUG: ")
            #for key, value in data.items():
            #    print(f"{key}: {value}")
        else:
            print(f"\n{GREEN}[  OK  ] No Civitai URL found via hash lookup.{RESET}\n")

        # Inspect the metadata for tag frequency information.
        ss_tag_frequency = meta.get("ss_tag_frequency") # This gives string not dictionary
        if not ss_tag_frequency:
            print(f"{CYAN}Metadata (no ss_tag_frequency found):{RESET}")
            for key, value in meta.items():
                print(f"{key}: {value}")
            print()
            return

        try:
            tag_dict = (
                json.loads(ss_tag_frequency) # Convert the string to a dictionary if it's a string
                if isinstance(ss_tag_frequency, str)
                else ss_tag_frequency
            )

            for folder_name, tags in tag_dict.items():
                sorted_tags = sorted(tags.keys(), key=tags.get, reverse=True) # Sort tags by frequency (highest count first)
                # print("DEBUG: Unsorted tags:", tags)
                top_tags = sorted_tags[:30] # Slice to get only the top 30 tags
                print(f"{CYAN}Dataset Folder:{RESET} {folder_name}")
                print(f"{CYAN}Top {30} Tags:{RESET}\n"
                    f"{', '.join(top_tags)}"
                )
                print()
        except json.JSONDecodeError:
            print(f"{RED}[ ERROR] Error decoding tag JSON.{RESET}\n")
            time.sleep(1)

def start(reload: bool=False):
    if not reload:
        print(f"{MAGENTA}[ INFO ] Warming up the system...{RESET}")
    else:
        print(f"{MAGENTA}[ INFO ] Reloading the system...{RESET}")
        free, total = torch.cuda.mem_get_info()
        print(f"{MAGENTA}[  INFO  ] VRAM after cleanup: {(total-free)/1024**3:.2f} / {total/1024**3:.2f} GiB used{RESET}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"{YELLOW}[ LOAD ] Initializing pipeline...") # Not resetting the color so that the bar would remain yellow during loading
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(CHECKPOINT_DIR / user_settings["model"]),
        torch_dtype=torch.float16,
    )
    pipe.enable_vae_slicing()
    print(f"{GREEN}[  OK  ] Pipeline initialized with model: ({user_settings["model"]}).{RESET}")

    print(f"{YELLOW}[ LOAD ] Configuring scheduler...{RESET}")
    if user_settings["sampler"] not in SAMPLER_OPTIONS:
        raise ValueError(f"{RED}[ ERROR] Invalid sampler option: {user_settings['sampler']}. Available options are: {SAMPLER_OPTIONS}{RESET}")
    if user_settings["sampler"] == "Euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif user_settings["sampler"] == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif user_settings["sampler"] == "DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
        )
    elif user_settings["sampler"] == "DPM++ SDE Karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True
        )
    print(f"{GREEN}[  OK  ] Scheduler configured, using sampler: ({user_settings["sampler"]}).{RESET}")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    is_low_vram = vram_gb < 12
    print(f"{MAGENTA}[ INFO ] Detected VRAM: {vram_gb:.2f} GB{RESET}")
    if is_low_vram:
        print(f"{YELLOW}[ LOAD ] Low VRAM detected. Enabling VAE tiling and CPU offload...{RESET}")
        pipe.enable_vae_tiling()
        pipe.enable_model_cpu_offload()
        print(f"{GREEN}[  OK  ] VAE tiling and model CPU offload enabled.{RESET}")
    else:
        if user_settings["force_enable_vae_tiling"]:
            print(f"{YELLOW}[ LOAD ] Forcing VAE tiling...{RESET}")
            pipe.enable_vae_tiling()
            print(f"{GREEN}[  OK  ] VAE tiling forced.{RESET}")
        if user_settings["force_enable_model_cpu_offload"]:
            print(f"{YELLOW}[ LOAD ] Forcing model CPU offload...{RESET}")
            pipe.enable_model_cpu_offload()
            print(f"{GREEN}[  OK  ] Model CPU offload forced. (Embeddings and LoRA might take longer to load.){RESET}")
        else:
            print(f"{YELLOW}[ LOAD ] Sufficient VRAM detected. Moving pipeline to GPU...{RESET}")
            pipe.to("cuda")
            print(f"{GREEN}[  OK  ] Pipeline moved to GPU.{RESET}")

    print(f"{YELLOW}[ LOAD ] Loading embeddings...{RESET}")
    load_sdxl_embedding(pipe)
    print(f"{GREEN}[  OK  ] Embeddings loaded.{RESET}")

    print(f"{YELLOW}[ LOAD ] Loading LoRA models...{RESET}")
    load_lora(pipe, user_settings["lora"])
    print(f"{GREEN}[  OK  ] LoRA models loaded.{RESET}")

    if not is_low_vram and not user_settings["force_disable_unet_compilation"]:
        print(f"{YELLOW}[ LOAD ] Compiling UNet with Torch...{RESET}")
        pipe.unet = torch.compile(
            pipe.unet,
            mode="reduce-overhead",
            fullgraph=True,
        )
        print(f"{GREEN}[  OK  ] UNet compiled with Torch. (First generation will take longer.){RESET}")

    print(f"{YELLOW}[ LOAD ] Initializing textual inversion manager and Compel wrapper...{RESET}")
    _ti_manager = DiffusersTextualInversionManager(pipe)
    compel = CompelForSDXL(
        pipe,
        textual_inversion_manager=_ti_manager,
        device="cuda",
    )
    print(f"{GREEN}[  OK  ] Textual inversion manager and Compel wrapper initialized.{RESET}")

    print(f"{BLUE}[ READY] System warmup complete.{RESET}")
    time.sleep(1)
    return pipe, compel, is_low_vram

MAIN_LOGO = [
    "    ,...       ,,    ,,                       ,,    ,,      ,...  ,...                    ,,",
    "  .d' \"\"     `7MM    db                     `7MM    db    .d' \"\".d' \"\"                    db",
    "  dM`          MM                             MM          dM`   dM`",
    " mMMmm.gP\"Ya   MM  `7MM  `7M'   `MF'     ,M\"\"bMM  `7MM   mMMmm mMMmm`7MM  `7MM  ,pP\"Ybd `7MM  ,pW\"Wq.`7MMpMMMb.",
    "  MM ,M'   Yb  MM    MM    `VA ,V'     ,AP    MM    MM    MM    MM    MM    MM  8I   `\"   MM 6W'   `Wb MM    MM",
    "  MM 8M\"\"\"\"\"\"  MM    MM      XMX mmmmm 8MI    MM    MM    MM    MM    MM    MM  `YMMMa.   MM 8M     M8 MM    MM",
    "  MM YM.    ,  MM    MM    ,V' VA.     `Mb    MM    MM    MM    MM    MM    MM  L.   I8   MM YA.   ,A9 MM    MM",
    ".JMML.`Mbmmd'.JMML..JMML..AM.   .MA.    `Wbmd\"MML..JMML..JMML..JMML.  `Mbod\"YML.M9mmmP' .JMML.`Ybmd9'.JMML  JMML.",
]
GENERATE_LOGO = [
    "  .g8\"\"\"bgd                                              mm",
    ".dP'     `M                                              MM",
    "dM'       `   .gP\"Ya `7MMpMMMb.  .gP\"Ya `7Mb,od8 ,6\"Yb.mmMMmm .gP\"Ya",
    "MM           ,M'   Yb  MM    MM ,M'   Yb  MM' \"'8)   MM  MM  ,M'   Yb",
    "MM.    `7MMF'8M\"\"\"\"\"\"  MM    MM 8M\"\"\"\"\"\"  MM     ,pm9MM  MM  8M\"\"\"\"\"\"",
    "`Mb.     MM  YM.    ,  MM    MM YM.    ,  MM    8M   MM  MM  YM.    ,",
    "  `\"bmmmdPY   `Mbmmd'.JMML  JMML.`Mbmmd'.JMML.  `Moo9^Yo.`Mbmo`Mbmmd'",
]

UPSCALE_LOGO = [
    "                                                    ,,",
    "`7MMF'   `7MF'                                    `7MM",
    "  MM       M                                        MM",
    "  MM       M `7MMpdMAo. ,pP\"Ybd  ,p6\"bo   ,6\"Yb.    MM  .gP\"Ya",
    "  MM       M   MM   `Wb 8I   `\" 6M'  OO  8)   MM    MM ,M'   Yb",
    "  MM       M   MM    M8 `YMMMa. 8M        ,pm9MM    MM 8M\"\"\"\"\"\"",
    "  YM.     ,M   MM   ,AP L.   I8 YM.    , 8M   MM    MM YM.    ,",
    "   `bmmmmd\"'   MMbmmd'  M9mmmP'  YMbmd'  `Moo9^Yo..JMML.`Mbmmd'",
    "               MM",
    "             .JMML."
]

EFFECTS_LOGO = [
    "                ,...  ,...",
    "`7MM\"\"\"YMM    .d' \"\".d' \"\"                mm",
    "  MM    `7    dM`   dM`                   MM",
    "  MM   d     mMMmm mMMmm.gP\"Ya   ,p6\"bo mmMMmm ,pP\"Ybd",
    "  MMmmMM      MM    MM ,M'   Yb 6M'  OO   MM   8I   `\"",
    "  MM   Y  ,   MM    MM 8M\"\"\"\"\"\" 8M        MM   `YMMMa.",
    "  MM     ,M   MM    MM YM.    , YM.    ,  MM   L.   I8",
    ".JMMmmmmMMM .JMML..JMML.`Mbmmd'  YMbmd'   `MbmoM9mmmP'"
]

METADATA_LOGO = [
    "                                             ,,",
    "`7MMM.     ,MMF'         mm                `7MM           mm",
    "  MMMb    dPMM           MM                  MM           MM",
    "  M YM   ,M MM  .gP\"Ya mmMMmm  ,6\"Yb.   ,M\"\"bMM   ,6\"Yb.mmMMmm  ,6\"Yb.",
    "  M  Mb  M' MM ,M'   Yb  MM   8)   MM ,AP    MM  8)   MM  MM   8)   MM",
    "  M  YM.P'  MM 8M\"\"\"\"\"\"  MM    ,pm9MM 8MI    MM   ,pm9MM  MM    ,pm9MM",
    "  M  `YM'   MM YM.    ,  MM   8M   MM `Mb    MM  8M   MM  MM   8M   MM",
    ".JML. `'  .JMML.`Mbmmd'  `Mbmo`Moo9^Yo.`Wbmd\"MML.`Moo9^Yo.`Mbmo`Moo9^Yo."
]


SETTINGS_LOGO = [
    "                                  ,,",
    " .M\"\"\"bgd           mm     mm     db",
    ",MI    \"Y           MM     MM",
    "`MMb.      .gP\"Ya mmMMmm mmMMmm `7MM  `7MMpMMMb.  .P\"Ybmmm ,pP\"Ybd",
    "  `YMMNq. ,M'   Yb  MM     MM     MM    MM    MM :MI  I8   8I   `\"",
    ".     `MM 8M\"\"\"\"\"\"  MM     MM     MM    MM    MM  WmmmP\"   `YMMMa.",
    "Mb     dM YM.    ,  MM     MM     MM    MM    MM 8M        L.   I8",
    "P\"Ybmmd\"   `Mbmmd'  `Mbmo  `Mbmo.JMML..JMML  JMML.YMMMMMb  M9mmmP'",
    "                                                 6'     dP",
    "                                                 Ybmmmd'"
]

def print_logo(logo, start_rgb=(196, 113, 245), end_rgb=(250, 113, 205)):
    os.system("cls" if os.name == "nt" else "clear")
    print()
    r1, g1, b1 = start_rgb
    r2, g2, b2 = end_rgb
    length = len(max(logo, key=len))

    # R: 196-222 (223) 224-250
    # i: 30-56 (57) 58-84

    for line in logo:
        result = []
        j = 1
        for i, char in enumerate(line):
            if i < 30:
                r, g, b = r1, g1, b1
            elif i >= 30 and i <= 84:
                t = j / (length + 1)
                r = int(r1 + (r2 - r1) * t)
                g = int(g1 + (g2 - g1) * t)
                b = int(b1 + (b2 - b1) * t)
                j = j + 1
            else:
                r, g, b = r2, g2, b2
            result.append(f"\033[38;2;{r};{g};{b}m{char}")
        
        print("".join(result) + "\033[0m")
    print()

def main():
    global user_settings
    pipe, compel, is_low_vram = start(reload=False)
    while True:
        print_logo(MAIN_LOGO)
        print(f"{BLUE}Welcome to felix-diffusion-cli!{RESET}")
        print(f"{CYAN}OS: {GRAY}{sysconfig.get_platform()}{RESET} ")
        print(f"{CYAN}Output directory: {GRAY}{OUTPUT_DIR}{RESET}")
        print(f"{CYAN}Model: {GRAY}{user_settings['model']}{RESET}")
        print()
        print(f"{YELLOW}Menu:{RESET}\n"
            "   (1) Generate Image\n"
            "   (2) Upscale Image\n"
            "   (3) Apply Effects\n"
            "   (4) Read Metadata (Image or LoRA)\n"
            "   (s) Settings\n"
            "   (q) Quit\n"
            )
        user_input = input(MAGENTA + "[ INPUT] Your choice: " + RESET)

        if user_input.lower() not in ["1", "2", "3", "4", "s", "q"]:
            print(f"{RED}[ ERROR] Invalid input.{RESET}")
            time.sleep(1)

        # Generate Image
        if user_input == "1":
            print_logo(GENERATE_LOGO)
            while True:
                user_input = input(MAGENTA + "[ INPUT] Enter your prompt (or 'b' to back): " + RESET)
                if user_input.lower() == 'b':
                    break

                original_image = generate_image(pipe, compel, user_input)
                print()
                print(f"{YELLOW}Upscale Options:\n{RESET}"
                    "   (1) RealESRGAN_x4plus_anime_6B\n"
                    "   (2) 4x-UltraSharp\n"
                    "   (n) No upscale"
                )
                user_input = input(MAGENTA + "[ INPUT] Your choice: " + RESET)
                if user_input.lower() not in ["1", "2", "n"]:
                    print(f"{RED}[ ERROR] Invalid input.{RESET}")
                elif user_input == "1":
                    upscaler = UPSCALERS_DIR / "RealESRGAN_x4plus_anime_6B.pth"
                    use_fp16 = True
                elif user_input == "2":
                    upscaler = UPSCALERS_DIR / "4x-UltraSharp.pth"
                    use_fp16 = False

                if user_input in ["1", "2"]:
                    size_factor = input(MAGENTA + "[ INPUT] Input the multipler (default/max 4, meaning 4x the size) or input `b` to cancel: " + RESET).strip()
                    if size_factor.lower() != "b":
                        if size_factor == "":
                            size_factor = 4.0
                        else:
                            try:
                                size_factor = float(size_factor)
                                if size_factor > 4.0 or size_factor <= 1:
                                    print(f"{RED}[ ERROR] Invalid size factor. Defaulting to 4{RESET}")
                                    size_factor = 4.0
                            except ValueError:
                                print(f"{RED}[ ERROR] Invalid size factor. Defaulting to 4{RESET}")
                                size_factor = 4.0
                        print(f"{YELLOW}[ LOAD ] Moving pipe to cpu temporarily to free VRAM for upscaler...{RESET}")
                        pipe.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"{GREEN}[  OK  ] Done.{RESET}")
                        
                        now = datetime.now()
                        time_str = now.strftime("%Y%m%d_%H%M%S")
                        file_name = f"output_{time_str}_upscaled.png"
                        output_path = UPSCALE_OUTPUT_DIR / file_name
                        upscale(image_path=original_image, model_path=upscaler, output_path=output_path, use_fp16=use_fp16, size_factor=size_factor)

                        print(f"{YELLOW}[ LOAD ] Restoring SDXL pipeline...{RESET}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        if is_low_vram:
                            pipe.enable_model_cpu_offload()
                        else:                    
                            pipe.to("cuda")
                        print(f"{GREEN}[  OK  ] Done.{RESET}")
                print()

        # Upscale Image
        elif user_input == "2":
            print_logo(UPSCALE_LOGO)
            while True:
                original_image = input(MAGENTA + "[ INPUT] Enter target path (or 'b' to back): " + RESET)
                if original_image.lower() == 'b':
                    break
                if not os.path.exists(original_image):
                    print(f"{RED}[ ERROR] File does not exist.{RESET}")
                    print()
                    time.sleep(1)
                    continue
                print(f"{YELLOW}Upscale Options:{RESET}\n"
                    "   (1) RealESRGAN_x4plus_anime_6B\n"
                    "   (2) 4x-UltraSharp\n"
                    "   (b) Back"
                )
                user_input = input(MAGENTA + "[ INPUT] Your choice: " + RESET)
                if user_input.lower() not in ["1", "2", "b"]:
                    print(f"{RED}[ ERROR] Invalid input.{RESET}")
                    print()
                elif user_input == "1":
                    upscaler = UPSCALERS_DIR / "RealESRGAN_x4plus_anime_6B.pth"
                    use_fp16 = True
                elif user_input == "2":
                    upscaler = UPSCALERS_DIR / "4x-UltraSharp.pth"
                    use_fp16 = False
                    
                if user_input in ["1", "2"]:
                    size_factor = input(MAGENTA + "[ INPUT] Input the multipler (default/max 4, meaning 4x the size) or input `b` to cancel: " + RESET).strip()
                    if size_factor.lower() != "b":
                        if size_factor == "":
                            size_factor = 4.0
                        else:
                            try:
                                size_factor = float(size_factor)
                                if size_factor > 4.0 or size_factor <= 1:
                                    print(f"{RED}[ ERROR] Invalid size factor. Defaulting to 4{RESET}")
                                    size_factor = 4.0
                            except ValueError:
                                print(f"{RED}[ ERROR] Invalid size factor. Defaulting to 4{RESET}")
                                size_factor = 4.0
                        print(f"{YELLOW}[ LOAD ] Moving pipe to cpu temporarily to free VRAM for upscaler...{RESET}")
                        pipe.to("cpu")
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"{GREEN}[  OK  ] Done.{RESET}")

                        now = datetime.now()
                        time_str = now.strftime("%Y%m%d_%H%M%S")
                        file_name = f"output_{time_str}_upscaled.png"
                        output_path = UPSCALE_OUTPUT_DIR / file_name
                        upscale(image_path=original_image, model_path=upscaler, output_path=output_path, use_fp16=use_fp16, size_factor=size_factor)

                        print(f"{YELLOW}[ LOAD ] Restoring SDXL pipeline...{RESET}")
                        gc.collect()
                        torch.cuda.empty_cache()
                        if is_low_vram:
                            pipe.enable_model_cpu_offload()
                        else:
                            pipe.to("cuda")
                        print(f"{GREEN}[  OK  ] Done.{RESET}")
                        print()

        elif user_input == "3":
            effects_menu(func=print_logo, logo=EFFECTS_LOGO, auto_open_bool=user_settings["auto_open_output"])

        # Metadata inspection
        elif user_input == "4":
            while True:
                print_logo(METADATA_LOGO)
                path = input(f"{MAGENTA}[ INPUT] Input path (or input 'b' to back): {RESET}")
                if path == "b":
                    break
                inspect_metadata(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")
                continue

        # Settings menu
        elif user_input == "s":
            need_reload = False
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print_logo(SETTINGS_LOGO)
                for key, value in user_settings.items():
                    if isinstance(value, dict):
                        print(f"{CYAN}{key}: {RESET}")
                        for sub_key, sub_value in value.items():
                            print(f"{CYAN}      {sub_key}: {RESET}{sub_value}")
                    else:
                        print(f"{CYAN}{key}: {RESET}{value}")
                print()
                print(f"{CYAN}Detected embeddings in EMBEDDINGS_DIR:{RESET}")
                embeddings = [f for f in os.listdir(EMBEDDINGS_DIR) if os.path.isfile(os.path.join(EMBEDDINGS_DIR, f)) and f.endswith(".safetensors")]
                embeddings.sort()
                embeddings_display = [f.replace(".safetensors", "") for f in embeddings]
                for emb in embeddings_display:
                    print(f"    {emb}")
                print()
                print(f"{YELLOW}Menu:{RESET}\n"
                    "   (1) model\n"
                    "   (2) width\n"
                    "   (3) height\n"
                    "   (4) sampler\n"
                    "   (5) steps\n"
                    "   (6) cfg\n"
                    "   (7) batch size\n"
                    "   (8) lora\n"
                    "   (9) positive prompt a\n"
                    "   (10) positive prompt b\n"
                    "   (11) negative prompt\n"
                    "   (12) auto-open output\n"
                    "   (13) force enable VAE tiling\n"
                    "   (14) force enable model CPU offload\n"
                    "   (15) force disable UNet compilation\n"
                    "   (d) reset to defaults (stable)\n"
                    "   (b) back and apply changes\n"
                )
                user_input = input(MAGENTA + "[ INPUT] Your choice: " + RESET)

                # Model Settings
                if user_input == "1":
                    before_change = user_settings.get("model", "")
                    available_models = [f for f in os.listdir(CHECKPOINT_DIR) if os.path.isfile(os.path.join(CHECKPOINT_DIR, f))]
                    available_models.sort()
                    available_models_display = [f.replace(".safetensors", "") for f in available_models]
                    print(f"{CYAN}Available models:{RESET}")
                    for i, name in enumerate(available_models_display, start=1):
                        print(f"    {i}. {name}")
                    user_input_index = input(f"{MAGENTA}[ INPUT] Enter the model number (or 'b' to back): {RESET}")
                    if user_input_index == "b":
                        continue
                    if not user_input_index.isdigit() or int(user_input_index) < 1 or int(user_input_index) > len(available_models_display):
                        print(f"{RED}[ ERROR] Invalid model selected.{RESET}")
                        time.sleep(1)
                        continue
                    user_settings["model"] = available_models_display[int(user_input_index) - 1] + ".safetensors"
                    if before_change != user_settings.get("model", ""):
                        print(f"{GREEN}[  OK  ] Model updated to {user_settings['model']}.{RESET}")
                        time.sleep(1)
                        need_reload = True

                # Width Settings
                elif user_input == "2":
                    before_change = user_settings.get("width", 0)
                    new_value = input(f"{MAGENTA}[ INPUT] Enter width (or 'b' to back): {RESET}")
                    if new_value == "b":
                        continue
                    try:
                        new_value = int(new_value)
                        if new_value <= 0:
                            print(f"{RED}[ ERROR] Width must be a positive integer.{RESET}")
                            time.sleep(1)
                            continue
                        if new_value % 8 != 0:
                            print(f"{RED}[ ERROR] Width must be a multiple of 8.{RESET}")
                            time.sleep(1)
                            continue
                        user_settings["width"] = new_value
                        if before_change != user_settings.get("width", 0):
                            print(f"{GREEN}[  OK  ] Width updated to {user_settings['width']}.{RESET}")
                            time.sleep(1)
                    except ValueError:
                        print(f"{RED}[ ERROR] Invalid width. Please enter a valid integer.{RESET}")
                        time.sleep(1)

                # Height Settings
                elif user_input == "3":
                    before_change = user_settings.get("height", 0)
                    new_value = input(f"{MAGENTA}[ INPUT] Enter height (or 'b' to back): {RESET}")
                    if new_value == "b":
                        continue
                    try:
                        new_value = int(new_value)
                        if new_value <= 0:
                            print(f"{RED}[ ERROR] Height must be a positive integer.{RESET}")
                            time.sleep(1)
                            continue
                        if new_value % 8 != 0:
                            print(f"{RED}[ ERROR] Height must be a multiple of 8.{RESET}")
                            time.sleep(1)
                            continue
                        user_settings["height"] = new_value
                        if before_change != user_settings.get("height", 0):
                            print(f"{GREEN}[  OK  ] Height updated to {user_settings['height']}.{RESET}")
                            time.sleep(1)
                    except ValueError:
                        print(f"{RED}[ ERROR] Invalid height. Please enter a valid integer.{RESET}")
                        time.sleep(1)

                # Sampler Settings
                elif user_input == "4":
                    before_change = user_settings.get("sampler", "")
                    print(f"{YELLOW}Available samplers:{RESET}")
                    for i, name in enumerate(SAMPLER_OPTIONS, start=1):
                        print(f"    {i}. {name}")
                    user_input_index = input(f"{MAGENTA}[ INPUT] Enter the sampler number (or 'b' to back): {RESET}")
                    if user_input_index == "b":
                        continue
                    if not user_input_index.isdigit() or int(user_input_index) < 1 or int(user_input_index) > len(SAMPLER_OPTIONS):
                        print(f"{RED}[ ERROR] Invalid sampler selected.{RESET}")
                        time.sleep(1)
                        continue
                    new_value = SAMPLER_OPTIONS[int(user_input_index) - 1]
                    user_settings["sampler"] = new_value
                    if before_change != user_settings.get("sampler", ""):
                        print(f"{GREEN}[  OK  ] Sampler updated to {user_settings['sampler']}.{RESET}")
                        time.sleep(1)
                        need_reload = True

                # Steps Settings
                elif user_input == "5":
                    before_change = user_settings.get("steps", 0)
                    new_value = input(f"{MAGENTA}[ INPUT] Enter steps (or 'b' to back): {RESET}")
                    if new_value == "b":
                        continue
                    try:
                        new_value = int(new_value)
                        if new_value <= 0:
                            print(f"{RED}[ ERROR] Steps must be a positive integer.{RESET}")
                            time.sleep(1)
                            continue
                        user_settings["steps"] = new_value
                        if before_change != user_settings["steps"]:
                            print(f"{GREEN}[  OK  ] Steps updated to {user_settings['steps']}.{RESET}")
                            time.sleep(1)
                    except ValueError:
                        print(f"{RED}[ ERROR] Invalid value. Please enter a valid integer.{RESET}")
                        time.sleep(1)

                # CFG Settings
                elif user_input == "6":
                    before_change = user_settings.get("cfg", 0)
                    new_value = input(f"{MAGENTA}[ INPUT] Enter cfg (or 'b' to back): {RESET}")
                    if new_value == "b":
                        continue
                    try:
                        new_value = float(new_value)
                        if new_value <= 0:
                            print(f"{RED}[ ERROR] Cfg must be a positive number.{RESET}")
                            time.sleep(1)
                            continue
                        user_settings["cfg"] = new_value
                        if before_change != user_settings["cfg"]:
                            print(f"{GREEN}[  OK  ] Cfg updated to {user_settings['cfg']}.{RESET}")
                            time.sleep(1)
                    except ValueError:
                        print(f"{RED}[ ERROR] Invalid cfg. Please enter a valid number.{RESET}")
                        time.sleep(1)

                # Batch Size Settings
                elif user_input == "7":
                    before_change = user_settings.get("batch_size", 0)
                    new_value = input(f"{MAGENTA}[ INPUT] Enter batch size (or 'b' to back): {RESET}")
                    if new_value == "b":
                        continue
                    try:
                        new_value = int(new_value)
                        if new_value <= 0:
                            print(f"{RED}[ ERROR] Batch size must be a positive integer.{RESET}")
                            time.sleep(1)
                            continue
                        user_settings["batch_size"] = new_value
                        if before_change != user_settings["batch_size"]:
                            print(f"{GREEN}[  OK  ] Batch size updated to {user_settings['batch_size']}.{RESET}")
                            time.sleep(1)
                    except ValueError:
                        print(f"{RED}[ ERROR] Invalid batch size. Please enter a valid integer.{RESET}")
                        time.sleep(1)

                # LoRA Settings
                elif user_input == "8":
                    before_change = copy.deepcopy(user_settings.get("lora", {}))
                    while True:
                        os.system("cls" if os.name == "nt" else "clear")
                        print_logo(SETTINGS_LOGO)
                        for key, value in user_settings.items():
                            if isinstance(value, dict):
                                print(f"{CYAN}{key}: {RESET}")
                                for sub_key, sub_value in value.items():
                                    print(f"{CYAN}      {sub_key}: {RESET}{sub_value}")
                            else:
                                print(f"{CYAN}{key}: {RESET}{value}")
                        print()
                        print(f"{CYAN}Detected embeddings in EMBEDDINGS_DIR:{RESET}")
                        embeddings = [f for f in os.listdir(EMBEDDINGS_DIR) if os.path.isfile(os.path.join(EMBEDDINGS_DIR, f)) and f.endswith(".safetensors")]
                        embeddings.sort()
                        embeddings_display = [f.replace(".safetensors", "") for f in embeddings]
                        for emb in embeddings_display:
                            print(f"    {emb}")
                        print()
                        available_models = [f for f in os.listdir(LORA_DIR) if os.path.isfile(os.path.join(LORA_DIR, f))]
                        available_models.sort()
                        available_models_display = [f.replace(".safetensors", "") for f in available_models]
                        print(f"{YELLOW}Available models:{RESET}")
                        print(f"    0. Remove all existing LoRAs")
                        for i, name in enumerate(available_models_display, start=1):
                            print(f"    {i}. {name}")
                        print(f"{GRAY}Hint: Set value to 0 to remove a LoRA from settings.{RESET}")
                        lora_index = input(f"{MAGENTA}[ INPUT] Enter the model number (or 'b' to back): {RESET}")
                        if lora_index == "b":
                            break
                        if lora_index == "0":
                            user_settings["lora"].clear()
                            need_reload = True
                            print(f"{GREEN}[  OK  ] All LoRAs removed.{RESET}")
                            time.sleep(1)
                            continue
                        if not lora_index.isdigit() or int(lora_index) < 0 or int(lora_index) > len(available_models_display):
                            print(f"{RED}[ ERROR] Invalid model number '{lora_index}'. Please enter a valid number from the list.{RESET}")
                            time.sleep(1)
                            continue
                        lora_key = available_models_display[int(lora_index) - 1]

                        lora_value_input = input(f"{MAGENTA}[ INPUT] Enter lora value (empty value to cancel or 'b' to go back to settings menu): {RESET}")
                        if lora_value_input == "b":
                            break
                        if lora_value_input == "":
                            print(f"{RED}[ ERROR] Empty value detected.{RESET}")
                            time.sleep(1)
                            continue

                        try:
                            lora_value = float(lora_value_input)
                        except ValueError:
                            print(f"{RED}[ ERROR] Invalid lora value. Please enter a valid decimal like 0.8.{RESET}")
                            time.sleep(1)
                            continue

                        if lora_value == 0:
                            if lora_key in user_settings["lora"]:
                                del user_settings["lora"][lora_key]
                                need_reload = True
                                print(f"{GREEN}[  OK  ] Lora key '{lora_key}' removed due to 0 value.{RESET}")
                                time.sleep(1)
                                continue
                            else:
                                print(f"{RED}[ ERROR] Detected 0 value for new Lora key '{lora_key}'. Skipping...{RESET}")
                                time.sleep(1)
                                continue
                            
                        if lora_key not in user_settings["lora"]:
                            print(f"{GREEN}[  OK  ] New lora key '{lora_key}' added with value {lora_value}{RESET}")
                        else:
                            print(f"{GREEN}[  OK  ] Existing lora key '{lora_key}' updated from {user_settings['lora'][lora_key]} to {lora_value}{RESET}")
                        
                        user_settings["lora"][lora_key] = lora_value
                        time.sleep(1)
                        if before_change != user_settings.get("lora", {}):
                            need_reload = True

                # Positive Prompt A Settings
                elif user_input == "9":
                    before_change = copy.deepcopy(user_settings.get("positive_prompt_a"))
                    new_positive_prompt_a = input(f"{MAGENTA}[ INPUT] Enter positive prompt A ('d' to restore default or 'b' to back): {RESET}")
                    if new_positive_prompt_a == "b":
                        continue
                    if new_positive_prompt_a == "d":
                        user_settings["positive_prompt_a"] = DEFAULT_SETTINGS["positive_prompt_a"]
                        print(f"{GREEN}[  OK  ] Positive prompt A restored to default.{RESET}")
                        time.sleep(1)
                    else:
                        user_settings["positive_prompt_a"] = new_positive_prompt_a
                    if before_change != user_settings.get("positive_prompt_a"):
                        print(f"{GREEN}[  OK  ] Positive prompt A updated.{RESET}")
                        time.sleep(1)

                # Positive Prompt B Settings
                elif user_input == "10":
                    before_change = copy.deepcopy(user_settings.get("positive_prompt_b"))
                    new_positive_prompt_b = input(f"{MAGENTA}[ INPUT] Enter positive prompt B ('d' to restore default or 'b' to back): {RESET}")
                    if new_positive_prompt_b == "b":
                        continue
                    if new_positive_prompt_b == "d":
                        user_settings["positive_prompt_b"] = DEFAULT_SETTINGS["positive_prompt_b"]
                        print(f"{GREEN}[  OK  ] Positive prompt B restored to default.{RESET}")
                        time.sleep(1)
                    else:
                        user_settings["positive_prompt_b"] = new_positive_prompt_b
                    if before_change != user_settings.get("positive_prompt_b"):
                        print(f"{GREEN}[  OK  ] Positive prompt B updated.{RESET}")
                        time.sleep(1)

                # Negative Prompt Settings
                elif user_input == "11":
                    before_change = copy.deepcopy(user_settings.get("negative_prompt", ""))
                    new_negative_prompt = input(f"{MAGENTA}[ INPUT] Enter negative prompt ('d' to restore default or 'b' to back): {RESET}")
                    if new_negative_prompt == "b":
                        continue
                    if new_negative_prompt == "d":
                        user_settings["negative_prompt"] = DEFAULT_SETTINGS["negative_prompt"]
                        print(f"{GREEN}[  OK  ] Negative prompt restored to default.{RESET}")
                        time.sleep(1)
                    else:
                        user_settings["negative_prompt"] = new_negative_prompt
                    if before_change != user_settings.get("negative_prompt", ""):
                        print(f"{GREEN}[  OK  ] Negative prompt updated.{RESET}")
                        time.sleep(1)

                # Auto-Open Output Setting
                elif user_input == "12":
                    user_input_auto_open = input(f"{MAGENTA}[ INPUT] Enable auto-open output? (y/n or 'b' to back): {RESET}").lower()
                    if user_input_auto_open == "b":
                        continue
                    user_settings["auto_open_output"] = user_input_auto_open == "y"
                    print(f"{GREEN}[  OK  ] Auto-open output setting set to {user_settings['auto_open_output']}.{RESET}")
                    time.sleep(1)

                # Force Enable VAE Tiling Setting
                elif user_input == "13":
                    before_change = copy.deepcopy(user_settings.get("force_enable_vae_tiling"))
                    print(f"{GRAY}Hint: This could save VRAM but might potentially create artifacts.{RESET}")
                    user_input = input(f"{MAGENTA}[ INPUT] Enable force enable VAE tiling? (y/n or 'b' to back): {RESET}").lower()
                    if user_input == "b":
                        continue
                    user_settings["force_enable_vae_tiling"] = user_input == "y"
                    status = "enabled" if user_settings["force_enable_vae_tiling"] else "disabled"
                    print(f"{GREEN}[  OK  ] Force enable VAE tiling {status}.{RESET}")
                    time.sleep(1)
                    if before_change != user_settings.get("force_enable_vae_tiling"):
                        need_reload = True
                
                # Force Enable Model CPU Offload Setting
                elif user_input == "14":
                    before_change = copy.deepcopy(user_settings.get("force_enable_model_cpu_offload"))
                    print(f"{GRAY}Hint: This could save VRAM but might slow down the image generation.{RESET}")
                    user_input = input(f"{MAGENTA}[ INPUT] Enable force enable model CPU offload? (y/n or 'b' to back): {RESET}").lower()
                    if user_input == "b":
                        continue
                    user_settings["force_enable_model_cpu_offload"] = user_input == "y"
                    status = "enabled" if user_settings["force_enable_model_cpu_offload"] else "disabled"
                    print(f"{GREEN}[  OK  ] Force enable model CPU offload {status}.{RESET}")
                    time.sleep(1)
                    if before_change != user_settings.get("force_enable_model_cpu_offload"):
                        need_reload = True

                # Force Disable UNet Compilation Setting
                elif user_input == "15":
                    before_change = copy.deepcopy(user_settings.get("force_disable_unet_compilation"))
                    print(f"{GRAY}Hint: [Only affect high VRAM setups] Force disabling UNet compilation might save VRAM and skip long loading time for *first* image generation but might reduce generation speed.{RESET}")
                    user_input = input(f"{MAGENTA}[ INPUT] Enable force disable UNet compilation? (y/n or 'b' to back): {RESET}").lower()
                    if user_input == "b":
                        continue
                    user_settings["force_disable_unet_compilation"] = user_input == "y"
                    status = "enabled" if user_settings["force_disable_unet_compilation"] else "disabled"
                    print(f"{GREEN}[  OK  ] Force disable UNet compilation {status}.{RESET}")
                    time.sleep(1)
                    if before_change != user_settings.get("force_disable_unet_compilation"):
                        need_reload = True

                # Reset to Default Settings
                elif user_input == "d":
                    before_change = copy.deepcopy(user_settings)
                    user_settings = copy.deepcopy(DEFAULT_SETTINGS)
                    if before_change != user_settings:
                        need_reload = True
                    print(f"{GREEN}[  OK  ] Settings reset to default.{RESET}")
                    time.sleep(1)

                # Back to Main Menu
                elif user_input == "b":
                    if need_reload:
                        print(f"{YELLOW}Applying new settings...{RESET}")
                        print(f"{YELLOW}Cleaning up old pipeline from VRAM...{RESET}")
                        del compel
                        del pipe
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"{GREEN}[  OK  ] VRAM memory freed!{RESET}")
                        pipe, compel, is_low_vram = start(reload=True)
                    try:
                        with open(SETTINGS_JSON_PATH, "w") as f:
                            json.dump(user_settings, f, indent=4)
                        print(f"{GREEN}[  OK  ] Settings saved to local JSON successfully.{RESET}")
                    except Exception as e:
                        print(f"{RED}[ ERROR] Failed to save settings: {e}{RESET}")
                    time.sleep(1)
                    break

        elif user_input == "q":
            print()
            print(f"{BLUE}[ EXIT ] Exiting! :D{RESET}")
            time.sleep(1)
            os.system("cls" if os.name == "nt" else "clear")
            break

if __name__ == "__main__":
    main()