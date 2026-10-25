# Felix's Diffusion CLI

*Felix's application for running SDXL/Illustrious models.*

I am developing this project in Arch Linux but I *might* make it a cross-platform thing someday. (That's why in the main.py you might see some code that looks like it's gonna support Windows and MacOS in future)

## 1. Core Features
In the main menu you will be able to choose between:
- `(1) Generate Image`
- `(2) Upscale Image`
- `(3) Apply Effects`
- `(4) Read Metadata (Image or LoRA)`
- `(s) Settings`
- `(q) Quit`

----------------------------------

### 1.1 Generate Image
**Flow:** Generates an image using the resolution specified in user_settings.json -> optionally upscale the result

Since the application automatically adds quality control prompts to your input. You simply input the prompt like:
```plaintext
1girl, slight smile
```
and it will become:

```plaintext
lazypos, lazyup, Smooth_Quality, very aesthetic, cinematic lighting, ultra-detailed, beautiful detailed eyes, 1girl, slight smile, masterpiece, best quality, amazing quality, absurdres, highres, newest, 0__11Xx_illu"
```
If you want to change the quality control prompts you can change `positive prompt a` and `positive prompt b` in settings. The final prompt format is:

```plaintext
`positive prompt a` + `your input` + `positive prompt b`
```

Negative Prompt can be changed in `Settings` (see below).

----------------------------------

### 1.2 Upscale Image
Input the absolute path of the target file. Then choose from:
- RealESRGAN_x4plus_anime_6B
- 4x-UltraSharp

Default multiplier is 4 (meaning the image will upscale to 4x)

`tile` is set to 512, if you want to change it you can edit the code in the `upscale` function.

----------------------------------

### 1.3 Apply Effects
See section 2.

----------------------------------

### 1.3 Read Metadata (image or LoRA)
Input the absolute path of the target file. It will print the metadata of the target file. It is mainly for reading images generated with this application or LoRA. If you use this to read an image generated with ComfyUI the metadata would look very messy (it would still print but hard to read).

----------------------------------

### 1.4 Settings
Settings are saved in `user_settings.json`. This application automatically creates a JSON file if one doesn't exist in the project root. This application currently does *not* actively monitor changes in the json file. It reads the file once when start up. Then it uses the in-app memory(and copy the new settings to the json file so that next startup would keep your settings).

When you want to change the settings in the current session, use the `(s) settings` in main menu.

tldr: Change settings using `(s) settings` in menu. Manually tweaking the json file while the application is running will NOT affect the settings.

### For now you can change these settings:

#### Model
Selects the checkpoint for generation. The Settings Menu will list out models inside `CHECKPOINT_DIR`.

#### Width/Height
Sets the resolution (dimensions in pixels) of the generated output image.

#### Sampler
The sampling algorithm used to denoise the image. 
* Supported: `Euler`, `Euler a`, `DPM++ 2M SDE`, `DPM++ SDE Karras`

#### Steps
The number of sampling iterations performed to generate the image.

#### CFG (Classifier-Free Guidance)
Controls how strictly the AI adheres to your text prompt.

#### Batch Size
(WIP) KEEP THIS AT 1. Batch generation has not been tested. Might not work properly.

#### LoRA
You can add new lora or adjust existing lora value. To remove a lora you can set the value to 0. The Settings Menu will list out LoRA inside `LORA_DIR`. You can also input index `0` to clear all existing LoRAs in settings.

#### Positive Prompt A
Positive prompt that sits before your input in image generation.

#### Positive Prompt B
Positive prompt that sits after your input in image generation.

#### Negative Prompt
The default one is quite good for most use cases. You can change it if the default one gets into your way.

#### Auto-Open Output
Default = True. Meaning whenever the application generates/upscale an image it will open the file with your image viewer.

#### Force Enable VAE Tiling
For high VRAM setup VAE tiling is default set to disabled. This option is for forcing it to be enabled. Enabling VAE tiling could potentially solve out of VRAM issue but it slower the image generation speed and brings a minor risk of boundary artifacts.

#### Force Enable Model CPU offload
For high VRAM setup model CPU offload is default set to disabled and model is living in `cuda`. This option is for forcing model CPU offload to be enabled. This could potentially solve some out of VRAM issue but it slower the image generation speed and makes embeddings/LoRAs loading time longer.

#### Force Disable UNet Compilation
For high VRAM setup it is default set to compile the UNet. It improves the image generation speed *after* the first generation but requires a long loading time for the before the first image being generated. This option is for disabling the compilation.

#### *If you mess up so badly with the settings you can always restore the default settings using `(d)` in the Settings Menu.

#### *Embeddings are automatically loaded during startup. Just put your embeddings in `EMBEDDINGS_DIR`

----------------------------------

## 2. Effects
This app can apply these effects to images:
- `(1) Edge Softening`
- `(2) Depth Blur`
- `(3) Bloom`
- `(4) Chromatic Aberration`
- `(5) Vignette`
- `(6) Film Grain`

### 2.1 Edge Softening
Blend the image with a blurred copy to create a softer, slightly dreamy look.

Parameters:
- `es_radius`: How far the blur spreads
- `es_alpha`: How much of the original image is kept. (Higher alpha means less visible blur)

### 2.2 Depth Blur
Uses a depth map to blur parts of the image. Mimics a camera focusing on a subject while leaving other distances out of foccus.

Parameters:
- `blur_radius`: How strong the blur is
- `invert`: (Optional) Reverses the blurred areas
- `target_depth`: (Optional) Depth-map value to focus on (0–255). Areas with similar values stay sharper, while areas with increasingly different values become more blurred.

### 2.3 Bloom
Adds a soft glow around bright areas, inspired by light blooming in camera lens.

Parameters:
- `bloom_threshold`: How bright a pixel must be to contribute to the glow
- `bloom_radius`:  How far the glow spreads
- `bloom_intensity`: How strong the glow appears

### 2.4 Chromatic Aberration
Mimics lens distortion by scaling red and blue channels then cropping/padding them. This gives a real-world camera optics feel-a soft, retro, cinematic feel.

### 2.5 Vignette
Darkens the image toward the edges, mimics the light falloff sometimes seen around the corners of a camera image.

Parameters:
- `vignette_radius`: How far from the center the darkening becomes noticeable. Higher values keep the center brighter over a wider area.
- `vignette_intensity`: How dark the corners become

### 2.6 Film Grain
Adds fine, film-like grain to give the image a textured, less digitally smooth appearance.

Parameters:
- `film_grain_intensity`: How strong the brightness variation is
- `grain_scale`: Grain size

----------------------------------

## 3. Installation

### 3.1 Clone The Folder
```bash
git clone https://github.com/felixlau0106/felix-diffusion-cli
```

### 3.2 Your Resources
By default this application read `checkpoints`, `embeddings`, `upscalers`, `loras` in one resources directory. *You can change this in the code*.

### **Please change the `RESOURCES_DIR` in main.py to your actual path**

```bash
# Default(hardcoded, please change to your path): /home/felix/Desktop/models/
models/
├── checkpoints/
├── embeddings/
├── loras/
└── upscalers/
```

### 3.3 Run with uv

Install uv and Python 3.12 or newer, then run these commands `*inside*` the project folder:

```sh
# `uv sync` installs the dependencies listed in `pyproject.toml` using the pinned versions in `uv.lock`
uv sync
uv run main.py
# When adding a dependency, use `uv add package-name`; uv updates both files.
```

----------------------------------

## 4. Recommended Resources

I use the resources below (available on civitai.red) for anime-style images. The default `Positive Prompt A`, `Positive Prompt B`, and `Negative Prompt` settings are configured to use them. If you use different resources—or don’t use these—edit their defaults in `main.py`. The `(d)` option in the Settings menu restores the defaults defined there.

*Some of the model/LoRA pages below might show NSFW example outputs, but the models generate verey good SFW anime-style images too.

### Model
- [waiIllustriousSDXL_v170](https://civitai.red/models/827184/wai-illustrious-sdxl?modelVersionId=2883731)

### LoRA
- [CunnyfunkyV7](https://civitai.red/models/2324040/cunnystyle?modelVersionId=2614427)
- [CG look and anime aesthetics-MSS](https://civitai.red/models/1060551/illustrious-style-pack?modelVersionId=1306061)
- [0__11Xx](https://civitai.red/models/1197562/011xx-or-shiiros-styles-or-niji?modelVersionId=1348455)
- [Smooth_Booster_v4](https://civitai.red/models/1145743/smooth-detailer-booster-animasdxlpony?modelVersionId=2196453)

### Embeddings
- [lazypos](https://civitai.red/models/1302719/lazy-embeddings-for-all-illustrious-noobai-pony-sdxl-models-lazypositive-lazynegative-positive-and-negative-plus-more)
- lazyneg
- lazyup
- lazydn
- lazyhand
- lazyreal
- lazywet
- lazyloli (for negative prompt)
- [Smooth_Quality](https://civitai.red/models/1065154/smooth-embeddings?modelVersionId=1470551)

----------------------------------

## 5. Additional Information

The cache generated by huggingface is saved in the project folder (inside an auto generated folder `ai-cache`).

This application was designed to be super clean. You use uv to create venv (so the dependencies are installed *inside* the project folder), the ai related cache is inside the folder. When you want to delete this application, just remove the whole folder and everything will be gone.

----------------------------------
