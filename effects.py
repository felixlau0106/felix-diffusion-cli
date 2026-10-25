from datetime import datetime
import os
from pathlib import Path
import platform
import subprocess
from PIL.PngImagePlugin import PngInfo
import cv2
import torch
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps
import time

THIS_DIR = Path(__file__).parent
AI_CACHE = THIS_DIR / "ai-cache"
EFFECTS_OUTPUT_DIR = THIS_DIR / "output" / "effects_ouput"
if not EFFECTS_OUTPUT_DIR.exists():
    EFFECTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAY = "\033[90m"
RED = "\033[91m" # Error
GREEN = "\033[92m" # Finish, OK
YELLOW = "\033[93m" # Loading, doing something, menu
BLUE = "\033[94m" # Special
MAGENTA = "\033[95m" # Info/user input
CYAN = "\033[96m" # Settings/OS info
RESET = "\033[0m"

def disable_std():
    import sys
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")

def enable_std():
    import sys
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

disable_std()

os.environ["HF_HOME"] = str(AI_CACHE / "huggingface")
os.environ["HF_HUB_CACHE"] = str(AI_CACHE / "huggingface" / "hub")
os.environ["DIFFUSERS_CACHE"] = str(AI_CACHE / "huggingface" / "diffusers")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TORCH_HOME"] = str(AI_CACHE / "torch")
os.environ["MPLCONFIGDIR"] = str(AI_CACHE / "matplotlib")


from depth_anything_3.api import DepthAnything3
print(f"{YELLOW}[ LOAD ] Loading DepthAnything3 model...{RESET}")
da_model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
device = "cuda" if torch.cuda.is_available() else "cpu"
da_model.to(device)
print(f"{GREEN}[  OK  ] DepthAnything3 model loaded.{RESET}")
enable_std()

def generate_depth_map(path, da_model, style):
    print(f"{YELLOW}[ LOAD ] Generating depth map...{RESET}")
    disable_std()
    input_img = Image.open(path)
    orig_w, orig_h = input_img.size
    prediction = da_model.inference(image=[input_img])
    raw_depth = prediction.depth

    clean_depth = np.squeeze(raw_depth) # Squeeze out all empty 1-sized dimensions to get a clean 2D grid (H, W)
    clean_depth = cv2.resize(clean_depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    depth_min = clean_depth.min()
    depth_max = clean_depth.max()

    # For future me:
    # whole map - min = every pixel - min, it makes the closest point in the image become 0
    # max - min = total range of depth values
    # by dividing by the total range, it is basically a percentage
    # eg: if clean_depth has values ranging from 2 to 10
    # clean_depth - min = [0, 8], dividing by 8 = [0, 1]
    normalized_depth = (clean_depth - depth_min) / (depth_max - depth_min + 1e-8) # between 0.0 and 1.0, add 1e-8 to avoid division by zero

    depth_pixels = (normalized_depth * 255).astype(np.uint8) # Scale to 0–255 integer pixel values. uint8 is unsigned 8-bit integer 2^8 = 256 possible values (0-255)
    enable_std()

    metadata = PngInfo()
    metadata.add_text("Software", "felix-diffusion-cli")

    now = datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    
    if style == "8bit_grayscale":
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_depth_map_visual.png"
        output_img = Image.fromarray(depth_pixels)
        output_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved visual depth map to {output_path}{RESET}")

    depth_16bit = (normalized_depth * 65535).astype(np.uint16)
    depth_16bit_img = Image.fromarray(depth_16bit)
    if style == "16bit_grayscale":
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_depth_map_16bit.png"
        depth_16bit_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved 16-bit depth map to {output_path}{RESET}")

    inferno_depth = cv2.applyColorMap(depth_pixels, cv2.COLORMAP_INFERNO)
    inferno_depth_img = Image.fromarray(inferno_depth)
    if style == "inferno":
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_depth_map_INFERNO.png"
        inferno_depth_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved colored depth map to {output_path}{RESET}")

    magma_depth = cv2.applyColorMap(depth_pixels, cv2.COLORMAP_MAGMA)
    magma_depth_img = Image.fromarray(magma_depth)
    if style == "magma":
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_depth_map_MAGMA.png"
        magma_depth_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved colored depth map to {output_path}{RESET}")

    turbo_depth = cv2.applyColorMap(depth_pixels, cv2.COLORMAP_TURBO)
    turbo_depth_img = Image.fromarray(turbo_depth)
    if style == "turbo":
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_depth_map_TURBO.png"
        turbo_depth_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved colored depth map to {output_path}{RESET}")

    if style is None:
        print(f"{GREEN}[  OK  ] No visual style selected; only the depth_pixels is returned.{RESET}")
    return depth_pixels

def apply_edge_softening(input, es_radius, es_alpha, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying edge softening...{RESET}")
    # Radius is how wide the blur effect will be
    # (1 - alpha) is the opacity of the blurred image when blending with the original image. Eg. if alpha = 0.6, the blurred copy is painted at 40% strength over the original drawing
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input
    img_array = np.array(img)
    blurred_img = img.filter(ImageFilter.GaussianBlur(es_radius))
    blurred_array = np.array(blurred_img)
    blended_float = (img_array * es_alpha) + (blurred_array * (1.0 - es_alpha))
    softened_array = blended_float.astype(np.uint8)
    softened_img = Image.fromarray(softened_array)
    if save:
        print(f"{YELLOW}[ LOAD ] Saving edge softened result...{RESET}")
        metadata = PngInfo()
        if img.info.items():
            existing_effects = img.info.get("effects", "")
            new_effects = existing_effects + f"edge_softening: es_radius={es_radius}, es_alpha={es_alpha}\n"
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_edge_softened.png"
        softened_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved edge softened result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Edge softening applied. Passing the PIL Image object to the next stage.{RESET}")
        return softened_img

def apply_simple_depth_blur(input, depth_pixels, blur_radius, invert: bool = False, target_depth: int = None, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying simple depth blur...{RESET}")
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input
    blurred_img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    depth_mask = Image.fromarray(depth_pixels).convert("L") # Image.composite requires a mask in "L" mode (a PIL Image object), without this depth_pixels is just a numpy array and won't work as a mask.
    if invert:
        depth_mask = ImageOps.invert(depth_mask)
        # For future me:
        # Inverted Mask = 255 - depth_pixels
        # darkest pixels: value 0 -> 255 - 0 = 255 (pure white)
        # brightest pixels: value 255 -> 255 - 255 = 0 (pure black)
        # Dark gray pixels: eg value 50 -> 255 - 50 = 205 (light gray)
    if target_depth is not None:
        # if a pixel value = target_depth, the resulting mask will be 0 (black)
        # if a pixel value is far from target_depth, the resulting mask will be higher (lighter)
        distance = np.abs(depth_pixels.astype(np.int16) - target_depth) # change the depth pixels from unsigned 8-bit to signed 16-bit. abs = absolute value, so the result is always non-negative
        focused_depth_pixels = distance.astype(np.uint8)
        depth_mask = Image.fromarray(focused_depth_pixels).convert("L")
    result_img = Image.composite(blurred_img, img, depth_mask)
    if save:
        print(f"{YELLOW}[ LOAD ] Saving simple depth blur result...{RESET}")
        metadata = PngInfo()
        if img.info.items():
            existing_effects = img.info.get("effects", "")
            new_effects = existing_effects + f"simple_depth_blur: blur_radius={blur_radius}, invert={invert}, target_depth={target_depth}\n"
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_blurred_result.png"
        result_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved blurred result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Simple depth blur applied. Passing the PIL Image object to the next stage.{RESET}")
        return result_img

def apply_bloom(input, bloom_threshold: float = 165.0, bloom_radius: float = 25, bloom_intensity: float = 0.05, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying bloom effect...{RESET}")
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input
    img = img.convert("RGB")
    luminance = img.convert("L") #0.299 * r + 0.587 * g + 0.114 * b

    bright_mask = np.array(luminance) > bloom_threshold
    img_np = np.array(img)
    img_np[~bright_mask] = 0

    bright_img = Image.fromarray(img_np)
    blurred_bright_img = bright_img.filter(ImageFilter.GaussianBlur(radius=bloom_radius))

    original_float = np.array(img, dtype=np.float32)
    glow_float = np.array(blurred_bright_img, dtype=np.float32)
    result_float = original_float + glow_float * bloom_intensity
    result_float = np.clip(result_float, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_float)

    if save:
        print(f"{YELLOW}[ LOAD ] Saving bloom result...{RESET}")
        metadata = PngInfo()
        if img.info.items():
            existing_effects = img.info.get("effects", "")
            new_effects = existing_effects + f"bloom: bloom_threshold={bloom_threshold}, bloom_radius={bloom_radius}, bloom_intensity={bloom_intensity}\n"
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_bloom.png"
        result_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved bloom result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Bloom effect applied. Passing the PIL Image object to the next stage.{RESET}")
        return result_img

def apply_radial_chromatic_aberration(input, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying radial chromatic aberration...{RESET}")
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input
    img = img.convert("RGB")
    r_channel, g_channel, b_channel = img.split()
    width, height = img.size
    # For future me:
    # Red Channel becomes slightly larger then gets cropped back to the original size, making red shift outward
    # Blue Channel becomes slightly smaller then gets padded back to the original size, making blue shift inward
    r_new_width = int(width * 1.004)
    r_new_height = int(height * 1.004)
    r_stretched = r_channel.resize((r_new_width, r_new_height), resample=Image.Resampling.LANCZOS)
    left = (r_new_width - width) // 2 # eg. 1000 became 1004, left would be 2
    top = (r_new_height - height) // 2 # eg. 1000 became 1004, top would be 2
    right = left + width # eg. 2 + 1000 = 1002
    r_cropped = r_stretched.crop((left, top, right, top + height)) # eg. crop box would be (2, 2, 1002, 1002), meaning skip 2 pixels on the left then stop at pixel 1002(right) and skip 2 pixels at the top then stop at 1002(bottom)

    b_new_width = int(width * 0.996)
    b_new_height = int(height * 0.996)
    b_shrunk = b_channel.resize((b_new_width, b_new_height), resample=Image.Resampling.LANCZOS)
    pad_left = (width - b_new_width) // 2
    pad_top = (height - b_new_height) // 2
    b_padded = Image.new("L", (width, height), 0)
    b_padded.paste(b_shrunk, (pad_left, pad_top))

    result_img = Image.merge("RGB", (r_cropped, g_channel, b_padded))

    if save:
        print(f"{YELLOW}[ LOAD ] Saving radial chromatic aberration result...{RESET}")
        existing_effects = img.info.get("effects", "")
        new_effects = existing_effects + f"radial_chromatic_aberration\n"
        metadata = PngInfo()
        if img.info.items():
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_radial_chromatic_aberration.png"
        result_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved radial chromatic aberration result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Radial chromatic aberration applied. Passing the PIL Image object to the next stage.{RESET}")
        return result_img

def apply_vignette_non_np(input, vignette_radius: float = 0.85, vignette_intensity: float = 0.35, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying vignette...{RESET}")
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input

    width, height = img.size
    center_x = width // 2
    center_y = height // 2
    vignette = Image.new("L", (width, height), 0)
    max_distance = (center_x ** 2 + center_y ** 2) ** 0.5
    total_pixels = width * height
    current_pixel = 0
    for y in range(height):
        for x in range(width):
            current_pixel += 1
            print(f"\r[ LOAD ] Processing pixel: {current_pixel}/{total_pixels}", end="", flush=True)
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            # For future me (this is so fucking annoying to learn):
            # Radius (standard 0.0 to 1.0 BUT can technically exceed 1.0 for pushing the shadow further outside the screen)
            # Intensity (Strictly 0.0 to 1.0, controls how dark the outer corners get)
            # Fraction = distance / (max_distance * radius)
            # Brightness = 1.0 - (Fraction^2 * intensity)
            # Example:
            # lets say radius is 1.0 and intensity is 0.5
            # Imagine max distance from center to a corner is 100 pixels
            # the current pixel's distance from the center is 50 pixels
            # Fraction = 50 / (100 * 1.0) = 0.5
            # Brightness = 1.0 - (0.5 * 0.5) = 1.0 - 0.25 = 0.75 (75% light)
            normalized_distance = distance / (max_distance * vignette_radius)
            factor = 1.0 - (normalized_distance ** 2) * vignette_intensity
            factor = max(0.0, min(factor, 1.0))
            vignette_value = int(255 * factor)
            vignette.putpixel((x, y), vignette_value)
    black_img = Image.new("RGB", (width, height), (0, 0, 0))
    result_img = Image.composite(img, black_img, vignette)

    if save:
        print(f"{YELLOW}[ LOAD ] Saving vignette result...{RESET}")
        existing_effects = img.info.get("effects", "")
        new_effects = existing_effects + f"vignette_non_np: vignette_radius={vignette_radius}, vignette_intensity={vignette_intensity}\n"
        metadata = PngInfo()
        if img.info.items():
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_vignette.png"
        result_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved vignette result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Vignette applied. Passing the PIL Image object to the next stage.{RESET}")
        return result_img


def apply_vignette(input, vignette_radius: float = 0.85, vignette_intensity: float = 0.35, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying vignette...{RESET}")
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input

    width, height = img.size
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2

    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_distance = (center_x ** 2 + center_y ** 2) ** 0.5

    normalized_distance = distance / (max_distance * vignette_radius)
    factor = 1.0 - (normalized_distance ** 2) * vignette_intensity
    factor = np.clip(factor, 0.0, 1.0)

    img_array = np.array(img, dtype=np.float32)
    result_array = (img_array * factor[..., np.newaxis]).clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_array)

    if save:
        print(f"{YELLOW}[ LOAD ] Saving vignette result...{RESET}")
        existing_effects = img.info.get("effects", "")
        new_effects = existing_effects + f"vignette: vignette_radius={vignette_radius}, vignette_intensity={vignette_intensity}\n"
        metadata = PngInfo()
        if img.info.items():
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_vignette.png"
        result_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved vignette result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Vignette applied. Passing the PIL Image object to the next stage.{RESET}")
        return result_img

def apply_film_grain_simple(input, film_grain_intensity, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying simple film grain...{RESET}")
    # Simple but it works. However the grain creates random colors due to the added random noise on all channels.
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input
    img_array = np.array(img)
    noise = np.random.randint(-film_grain_intensity,film_grain_intensity + 1, size=img_array.shape) # This would create random noise for each pixel and each color channel
    noisy_img_array = img_array.astype(np.int16) + noise
    clipped_noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(clipped_noisy_img_array)
    if save:
        print(f"{YELLOW}[ LOAD ] Saving simple film grain result...{RESET}")
        metadata = PngInfo()
        if img.info.items():
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        existing_effects = img.info.get("effects", "")
        new_effects = existing_effects + f"simple_film_grain: film_grain_intensity={film_grain_intensity}\n"
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_film_grain.png"
        result_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved film grain result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Simple film grain applied. Passing the PIL Image object to the next stage.{RESET}")
        return result_img

def apply_film_grain(input, film_grain_intensity, grain_scale, save: bool = True):
    print(f"{YELLOW}[ LOAD ] Applying film grain...{RESET}")
    # For future me: This approach ensures that the noise affects all color channels equally.
    if isinstance(input, str) or isinstance(input, Path):
        img = Image.open(input)
    else:
        img = input
    img_array = np.array(img)
    h, w, _ = img_array.shape
    # For future me:
    # If the grain_scale is greater than 1, the noise will be generated at a smaller resolution and then upscaled to match the image size.
    # Which naturally creates larger grain patterns when upscaled
    noise_h, noise_w = h // grain_scale, w // grain_scale 
    small_noise = np.random.randint(-film_grain_intensity, film_grain_intensity + 1, size=(noise_h, noise_w, 1)) # Create small noise for the image, with a single channel to ensure equal noise across all color channels
    # + 128 to avoid negative values (since Image.fromarray expects unsigned 8-bit values)
    # Since NumPy 1-channel noise shape is (H, W, 1), using [:, :, 0] (a slicing operation) to throw away the extra dimensions (making it a 2D grid that matches the PIL grayscale "L" mode)
    noise_img = Image.fromarray((small_noise + 128).astype(np.uint8)[:, :, 0], mode="L") 
    noise_img = noise_img.resize((w, h), Image.Resampling.BILINEAR) # if grain_scale > 1, resize the small noise to match the image size. Bilinear is edge softening
    noise = np.array(noise_img).astype(np.int16)[:, :, np.newaxis] - 128 # convert back to signed 16-bit and add the extra channel dimension, -128 to restore the original noise range
    clipped_noisy_img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(clipped_noisy_img_array)
    if save:
        print(f"{YELLOW}[ LOAD ] Saving film grain result...{RESET}")
        existing_effects = img.info.get("effects", "")
        new_effects = existing_effects + f"film_grain: film_grain_intensity={film_grain_intensity}, grain_scale={grain_scale}\n"
        metadata = PngInfo()
        if img.info.items():
            for key, value in img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", new_effects)
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_film_grain.png"
        result_img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved film grain result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Film grain applied. Passing the PIL Image object to the next stage.{RESET}")
        return result_img

def apply_full_chain(path, es_radius, es_alpha, depth_pixels, blur_radius, invert, target_depth, bloom_threshold, bloom_radius, bloom_intensity, vignette_radius, vignette_intensity, film_grain_intensity, grain_scale, save_intermediate):
    img = apply_edge_softening(path, es_radius, es_alpha, save=save_intermediate)
    img = apply_simple_depth_blur(img, depth_pixels=depth_pixels, blur_radius=blur_radius, invert=invert, target_depth=target_depth, save=save_intermediate)
    img = apply_bloom(img, bloom_threshold=bloom_threshold,bloom_radius=bloom_radius, bloom_intensity=bloom_intensity, save=save_intermediate)
    img = apply_radial_chromatic_aberration(img, save=save_intermediate)
    img = apply_vignette(img, vignette_radius=vignette_radius, vignette_intensity=vignette_intensity, save=save_intermediate)
    img = apply_film_grain(img, film_grain_intensity=film_grain_intensity, grain_scale=grain_scale, save=save_intermediate)
    if not save_intermediate:
        print(f"{YELLOW}[ LOAD ] Saving full chain result...{RESET}")
        source_img = Image.open(path)
        existing_effects = source_img.info.get("effects", "")
        metadata = PngInfo()
        if source_img.info.items():
            for key, value in source_img.info.items():
                metadata.add_text(key, str(value))
        else:        
            metadata.add_text("Software", "felix-diffusion-cli")
        metadata.add_text("effects", (
            f"edge_softening: es_radius={es_radius}, es_alpha={es_alpha}\n"
            f"simple_depth_blur: blur_radius={blur_radius}, invert={invert}, target_depth={target_depth}\n"
            f"bloom: bloom_threshold={bloom_threshold}, bloom_radius={bloom_radius}, bloom_intensity={bloom_intensity}\n"
            f"radial_chromatic_aberration\n"
            f"vignette: vignette_radius={vignette_radius}, vignette_intensity={vignette_intensity}\n"
            f"film_grain: film_grain_intensity={film_grain_intensity}, grain_scale={grain_scale}"
        ))
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = EFFECTS_OUTPUT_DIR / f"output_{time_str}_full_chain.png"
        img.save(output_path, pnginfo=metadata)
        print(f"{GREEN}[  OK  ] Saved full chain result to {output_path}{RESET}")
        return output_path
    else:
        print(f"{GREEN}[  OK  ] Full chain applied. Final result is {img}.{RESET}") # if save_intermediate, film grain will return the path of the resulting image
        return img
# Implementation Plan
# 1. Edge softening [done]
# 2. Bokeh Depth Blur [TBD] / Simple Depth Blur [done]
# 3. Bloom [done]
# 4. Chromatic aberration [done]
# 5. Vignette [done]
# 6. Film grain [done]

def auto_open_output(path):
    if platform.system() == "Linux":
        subprocess.Popen(["xdg-open", str(path)])
    elif platform.system() == "Darwin": # Well...just in case... (MacOS)
        subprocess.Popen(["open", str(path)])
    elif platform.system() == "Windows":
        os.startfile(str(path))

def simplified_print_logo(text):
    os.system("cls" if os.name == "nt" else "clear")
    print(text)

def effects_menu(func=None, logo=None, auto_open_bool=False):
    if func and logo and auto_open_bool is not None:
        print_logo = func
        logo_text = logo
        auto_open = auto_open_bool
    while True:
        print_logo(logo_text)
        print(f"{CYAN}Depth Anything:{RESET} {"Loaded" if da_model else 'Not loaded'}")
        print(f"{YELLOW}Menu:\n{RESET}"
            "   (1) Apply edge softening\n"
            "   (2) Apply blur\n"
            "   (3) Apply bloom\n"
            "   (4) Apply chromatic aberration\n"
            "   (5) Apply vignette\n"
            "   (6) Apply film grain\n"
            "   (f) Full effects-chain\n"
            "   (b) Back\n"
        )

        user_input = input(f"{MAGENTA}[ INPUT] Choose an option: {RESET}").strip()
        if user_input not in ["1", "2", "3", "4", "5", "6", "f", "b", ""]:
            print(f"{RED}[ ERROR] Invalid choice. Please try again.{RESET}")
            time.sleep(1)
            continue
        
        elif user_input == "":
            continue

        elif user_input == "b":
            break

        # Edge softening effect
        elif user_input == "1":
            while True:
                print_logo(logo_text)
                print(f"{YELLOW}Edge Softening Effect{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Enter the image path (or b to go back): {RESET}")
                if user_input.lower() == "b":
                    break
                if not user_input:
                    print(f"{RED}[ ERROR] Image path cannot be empty.{RESET}")
                    time.sleep(1)
                    continue
                path = user_input
                if not os.path.exists(path):
                    print(f"{RED}[ ERROR] Image path does not exist.{RESET}")
                    time.sleep(1)
                    continue
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the edge softening radius (default 7, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        radius = 7
                    else:
                        radius = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default radius of 7.{RESET}")
                    radius = 7
                user_input = input(f"{MAGENTA}[ INPUT] Enter the alpha value (higher alpha = lower opacity of the blurred copy) for edge softening (default 0.6, blank for default, b to cancel): {RESET}")
                if user_input.lower() == "b":
                    continue
                try:
                    if not user_input:
                        alpha = 0.6
                    else:
                        alpha = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default alpha of 0.6.{RESET}")
                    alpha = 0.6
                path = apply_edge_softening(path, radius, alpha, save=True)
                if auto_open:
                    auto_open_output(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")

        # Blur effect
        elif user_input == "2":
            while True:
                print_logo(logo_text)
                print(f"{YELLOW}Blur Effect{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Enter the image path (or b to go back): {RESET}")
                if user_input.lower() == "b":
                    break
                if not user_input:
                    print(f"{RED}[ ERROR] Image path cannot be empty.{RESET}")
                    time.sleep(1)
                    continue
                path = user_input
                if not os.path.exists(path):
                    print(f"{RED}[ ERROR] Image path does not exist.{RESET}")
                    time.sleep(1)
                    continue            
                print(f"{YELLOW}Menu:\n{RESET}"
                            "   (1) 8-bit Grayscale\n"
                            "   (2) 16-bit Grayscale\n"
                            "   (3) Inferno\n"
                            "   (4) Magma\n"
                            "   (5) Turbo\n"
                            "   (b) Back"
                        )
                print(f"{GRAY}Hint: Choose a style if you want to output a depth map PNG.{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Choose a style or leave empty (b to cancel): {RESET}")
                if user_input.lower() == "b":
                    continue
                style_map = {
                    "1": "8bit_grayscale",
                    "2": "16bit_grayscale",
                    "3": "inferno",
                    "4": "magma",
                    "5": "turbo",
                }
                if user_input in style_map:
                    style = style_map[user_input]
                    print(f"{MAGENTA}[ INFO ] Selected style: {style}{RESET}")
                else:
                    style = None
                    print(f"{MAGENTA}[ INFO] Continuing without saving depth map PNG.{RESET}")
                depth_pixels = generate_depth_map(path, da_model=da_model, style=style)
                print(f"{GREEN}[  OK  ] Generated depth map.{RESET}")
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the blur radius (default 15, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        blur_radius = 15
                    else:
                        blur_radius = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default blur radius of 15.{RESET}")
                    blur_radius = 15
                user_input = input(f"{MAGENTA}[ INPUT] (Optional) Custom target depth? (0-255, lower values are closer; leave blank for none, b to cancel): {RESET}")
                if user_input.lower() == "b":
                    continue
                try:
                    target_depth = int(user_input) if user_input else None
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. No target depth will be used.{RESET}")
                    target_depth = None
                invert = False
                if target_depth is None: # if a target depth is specified, everything is focused around that target depth, so inverting doesn't do anything meaningful
                    user_input = input(f"{MAGENTA}[ INPUT] Invert depth mask? (y/n, leave blank for no, b to cancel): {RESET}").strip().lower()
                    if user_input == "y":
                        invert = True
                    elif user_input == "n" or not user_input:
                        invert = False
                    elif user_input == "b":
                        continue
                    else:
                        print(f"{RED}[ ERROR] Invalid input. Using default of not inverting.{RESET}")
                        invert = False
                    print(f"{MAGENTA}[ INFO ] Invert depth mask: {invert}{RESET}")
                path = apply_simple_depth_blur(path, depth_pixels, blur_radius, invert=invert, target_depth=target_depth, save=True)
                if auto_open:
                    auto_open_output(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")

        # Bloom effect
        elif user_input == "3":
            while True:
                print_logo(logo_text)
                print(f"{YELLOW}Bloom Effect{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Enter the image path (or b to go back): {RESET}")
                if user_input.lower() == "b":
                    break
                if not user_input:
                    print(f"{RED}[ ERROR] Image path cannot be empty.{RESET}")
                    time.sleep(1)
                    continue
                path = user_input
                if not os.path.exists(path):
                    print(f"{RED}[ ERROR] Image path does not exist.{RESET}")
                    time.sleep(1)
                    continue
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the bloom threshold (default 165.0, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        bloom_threshold = 165.0
                    else:
                        bloom_threshold = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default bloom threshold of 165.0.{RESET}")
                    bloom_threshold = 165.0
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the bloom blur radius (default 25, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        bloom_radius = 25
                    else:
                        bloom_radius = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default bloom blur radius of 25.{RESET}")
                    bloom_radius = 25
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the bloom intensity (default 0.05, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        bloom_intensity = 0.05
                    else:
                        bloom_intensity = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default bloom intensity of 0.05.{RESET}")
                    bloom_intensity = 0.05
                path = apply_bloom(path, bloom_threshold=bloom_threshold, bloom_radius=bloom_radius, bloom_intensity=bloom_intensity, save=True)
                if auto_open:
                    auto_open_output(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")

        # Chromatic aberration effect
        elif user_input == "4":
            while True:
                print_logo(logo_text)
                print(f"{YELLOW}Chromatic Aberration Effect{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Enter the image path (or b to go back): {RESET}")
                if user_input.lower() == "b":
                    break
                if not user_input:
                    print(f"{RED}[ ERROR] Image path cannot be empty.{RESET}")
                    time.sleep(1)
                    continue
                path = user_input
                if not os.path.exists(path):
                    print(f"{RED}[ ERROR] Image path does not exist.{RESET}")
                    time.sleep(1)
                    continue
                path = apply_radial_chromatic_aberration(path, save=True)
                if auto_open:
                    auto_open_output(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")

        # Vignette effect
        elif user_input == "5":
            while True:
                print_logo(logo_text)
                print(f"{YELLOW}Vignette Effect{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Enter the image path (or b to go back): {RESET}")
                if user_input.lower() == "b":
                    break
                if not user_input:
                    print(f"{RED}[ ERROR] Image path cannot be empty.{RESET}")
                    time.sleep(1)
                    continue
                path = user_input
                if not os.path.exists(path):
                    print(f"{RED}[ ERROR] Image path does not exist.{RESET}")
                    time.sleep(1)
                    continue
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the vignette radius (default 0.85, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        vignette_radius = 0.85
                    else:
                        vignette_radius = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input for radius. Using default value 0.85.{RESET}")
                    vignette_radius = 0.85
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the vignette intensity (default 0.35, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        vignette_intensity = 0.35
                    else:
                        vignette_intensity = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default values for vignette.{RESET}")
                    vignette_intensity = 0.35
                path = apply_vignette(path, vignette_radius=vignette_radius, vignette_intensity=vignette_intensity, save=True)
                if auto_open:
                    auto_open_output(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")
            
        # Film grain effect
        elif user_input == "6":
            while True:
                print_logo(logo_text)
                print(f"{YELLOW}Film Grain Effect{RESET}")
                print(f"{GRAY}Hint: Default value is for upcaled large images like 4096x4096. Adjust the values for smaller images accordingly.{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Enter the image path (or b to go back): {RESET}")
                if user_input.lower() == "b":
                    break
                if not user_input:
                    print(f"{RED}[ ERROR] Image path cannot be empty.{RESET}")
                    time.sleep(1)
                    continue
                path = user_input
                if not os.path.exists(path):
                    print(f"{RED}[ ERROR] Image path does not exist.{RESET}")
                    time.sleep(1)
                    continue
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter intensity value for the film grain effect (default 15, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        film_grain_intensity = 15
                    else:
                        film_grain_intensity = int(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default film_grain_intensity.{RESET}")
                    film_grain_intensity = 15
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter grain scale (grain size) for the film grain effect (default 1, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        grain_scale = 1
                    else:
                        grain_scale = int(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default grain_scale.{RESET}")
                    grain_scale = 1
                path = apply_film_grain(path, film_grain_intensity=film_grain_intensity, grain_scale=grain_scale, save=True)
                if auto_open:
                    auto_open_output(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")
            
        # Full chain of effects
        elif user_input == "f":
            while True:
                print_logo(logo_text)
                print(f"{YELLOW}Full Chain of Effects{RESET}")
                print(f"{GRAY}Hint: Default value of film grain is for upcaled large images like 4096x4096. Adjust the values for smaller images accordingly.{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Enter the image path (or b to go back): {RESET}")
                if user_input.lower() == "b":
                    break
                if not user_input:
                    print(f"{RED}[ ERROR] Image path cannot be empty.{RESET}")
                    time.sleep(1)
                    continue
                path = user_input
                if not os.path.exists(path):
                    print(f"{RED}[ ERROR] Image path does not exist.{RESET}")
                    time.sleep(1)
                    continue
            
                print(f"{YELLOW}Menu:\n{RESET}"
                            "   (1) 8-bit Grayscale\n"
                            "   (2) 16-bit Grayscale\n"
                            "   (3) Inferno\n"
                            "   (4) Magma\n"
                            "   (5) Turbo\n"
                            "   (b) Back"
                        )
                print(f"{GRAY}Hint: Choose a style if you want to output a depth map PNG.{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Choose a style or leave empty (b to cancel): {RESET}")
                if user_input.lower() == "b":
                    continue
                style_map = {
                    "1": "8bit_grayscale",
                    "2": "16bit_grayscale",
                    "3": "inferno",
                    "4": "magma",
                    "5": "turbo",
                }
                if user_input in style_map:
                    style = style_map[user_input]
                    print(f"{MAGENTA}[ INFO ] Selected style: {style}{RESET}")
                else:
                    style = None
                    print(f"{MAGENTA}[ INFO] Continuing without saving depth map PNG.{RESET}")
                depth_pixels = generate_depth_map(path, da_model=da_model, style=style)
                print(f"{GREEN}[  OK  ] Generated depth map.{RESET}")
                save_steps = input(f"{MAGENTA}[ INPUT] Save intermediate steps? (y/n, default n, blank for default, b to cancel): {RESET}")
                if save_steps.lower() == "b":
                    continue
                if not save_steps:
                    save_steps = "n"
                if save_steps.lower() == "y":
                    save_intermediate = True
                    print(f"{MAGENTA}[ INFO ] Intermediate steps will be saved.{RESET}")
                else:
                    save_intermediate = False
                    print(f"{MAGENTA}[ INFO ] Intermediate steps will not be saved.{RESET}")
                user_input = input(f"{MAGENTA}[ INPUT] Use default values for all effects? (y/n, default y, blank for default, b to cancel): {RESET}")
                if user_input.lower() == "b":
                    continue
                if user_input.lower() != "n":
                    path =apply_full_chain(
                        path=path, 
                        es_radius=7, 
                        es_alpha=0.6, 
                        depth_pixels=depth_pixels, 
                        blur_radius=15,
                        invert=False,
                        target_depth=None,
                        bloom_threshold=165.0,
                        bloom_radius=25, 
                        bloom_intensity=0.05, 
                        vignette_radius=0.85, 
                        vignette_intensity=0.35, 
                        film_grain_intensity=15, 
                        grain_scale=1, 
                        save_intermediate=save_intermediate)
                    if auto_open:
                        auto_open_output(path)
                    input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")
                    continue

                # Chain - edge softening effect
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the edge softening radius (default 7, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        es_radius = 7
                    else:
                        es_radius = float(user_input)
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the alpha value (higher alpha = loweropacity of the blurred copy) for edge softening (default 0.6, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        es_alpha = 0.6
                    else:
                        es_alpha = float(user_input)                    
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default values for edge softening.{RESET}")
                    es_radius = 7
                    es_alpha = 0.6

                # Chain - simple depth blur effect
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the blur radius (default 15, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        blur_radius = 15
                    else:
                        blur_radius = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default blur radius of 15.{RESET}")
                    blur_radius = 15
                user_input = input(f"{MAGENTA}[ INPUT] (Optional) Custom target depth? (0-255, lower values are closer; leave blank for none, b to cancel): {RESET}")
                if user_input.lower() == "b":
                    continue
                try:
                    target_depth = int(user_input) if user_input else None
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. No target depth will be used.{RESET}")
                    target_depth = None
                invert = False
                if target_depth is None: # if a target depth is specified, everything is focused around that target depth, so inverting doesn't do anything meaningful
                    user_input = input(f"{MAGENTA}[ INPUT] Invert depth mask? (y/n, leave blank for no, b to cancel): {RESET}").strip().lower()
                    if user_input == "y":
                        invert = True
                    elif user_input == "n" or not user_input:
                        invert = False
                    elif user_input == "b":
                        continue
                    else:
                        print(f"{RED}[ ERROR] Invalid input. Using default of not inverting.{RESET}")
                        invert = False
                    print(f"{MAGENTA}[ INFO ] Invert depth mask: {invert}{RESET}")

                # Chain - bloom effect
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the bloom threshold (default 165.0, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        bloom_threshold = 165.0
                    else:
                        bloom_threshold = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default bloom threshold of 165.0.{RESET}")
                    bloom_threshold = 165.0
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the bloom blur radius (default 25, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        bloom_radius = 25
                    else:
                        bloom_radius = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default bloom blur radius of 25.{RESET}")
                    bloom_radius = 25
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the bloom intensity (default 0.05, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        bloom_intensity = 0.05
                    else:
                        bloom_intensity = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default bloom intensity of 0.05.{RESET}")
                    bloom_intensity = 0.05

                # Chain - vignette effect
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the vignette radius (default 0.85, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        vignette_radius = 0.85
                    else:
                        vignette_radius = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input for vignette radius. Using default value 0.85.{RESET}")
                    vignette_radius = 0.85
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the vignette intensity (default 0.35, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        vignette_intensity = 0.35
                    else:
                        vignette_intensity = float(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default values for vignette.{RESET}")
                    vignette_intensity = 0.35

                # Chain - film grain effect
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter the intensity for film grain effect (default 15, blank for default, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        film_grain_intensity = 15
                    else:
                        film_grain_intensity = int(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default intensity of 15 for film grain effect.{RESET}")
                    film_grain_intensity = 15
                try:
                    user_input = input(f"{MAGENTA}[ INPUT] Enter grain scale (grain size) for the film grain effect (default 1, b to cancel): {RESET}")
                    if user_input.lower() == "b":
                        continue
                    if not user_input:
                        grain_scale = 1
                    else:
                        grain_scale = int(user_input)
                except ValueError:
                    print(f"{RED}[ ERROR] Invalid input. Using default grain scale of 1.{RESET}")
                    grain_scale = 1
                path = apply_full_chain(
                    path=path, 
                    es_radius=es_radius,
                    es_alpha=es_alpha,
                    depth_pixels=depth_pixels,
                    blur_radius=blur_radius,
                    invert=invert,
                    target_depth=target_depth,
                    bloom_threshold=bloom_threshold,
                    bloom_radius=bloom_radius,
                    bloom_intensity=bloom_intensity,
                    vignette_radius=vignette_radius,
                    vignette_intensity=vignette_intensity,
                    film_grain_intensity=film_grain_intensity,
                    grain_scale=grain_scale,
                    save_intermediate=save_intermediate)
                if auto_open:
                    auto_open_output(path)
                input(f"{MAGENTA}[ INPUT] Press Enter to continue...{RESET}")

if __name__ == "__main__":
    effects_menu(simplified_print_logo, f"{BLUE}Effects{RESET}")