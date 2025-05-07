import requests
from PIL import Image
import numpy as np
import torch
import json
import os

def get_config_path():
    # Path of wordpress_config.json next to this file
    return os.path.join(os.path.dirname(__file__), "wordpress_config.json")

def load_wordpress_config():
    config_path = get_config_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"wordpress_config.json file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ["wordpress_site_url", "wordpress_user", "wordpress_pass"]:
        if not data.get(key):
            raise ValueError(f"Key '{key}' must have a value in wordpress_config.json.")
    return data["wordpress_site_url"], data["wordpress_user"], data["wordpress_pass"]

class SaveToWordPressNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "title": ("STRING", {"default": "Sample Post Title","multiline": True,"lines": 3}),
                "content": ("STRING", {"default": "Post content text for WordPress","multiline": True,"lines": 5}),
                "tags": ("STRING", {"default": "comfyui,ai,sample","multiline": True,"lines": 3}),
                "category": ("STRING", {"default": ""}),
                "meta_data": ("STRING",{"default": "", "forceInput": True} ),
                "positive_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (
    ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
     "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
     "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp",
     "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
     "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
     "res_multistep_ancestral_cfg_pp", "gradient_estimation", "er_sde", "seeds_2", "seeds_3",
     "ddim", "uni_pc", "uni_pc_bh2"],
 ),
"scheduler": (
    ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta",
     "linear_quadratic", "kl_optimal"],
 ),

                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_image_in_post": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "save_and_post"
    OUTPUT_NODE = True
    CATEGORY = "custom"

    def get_ksampler_table(self, **kwargs):
        field_defs = [
            ("seed", "Seed"),
            ("steps", "Steps"),
            ("cfg", "CFG"),
            ("sampler_name", "Sampler"),
            ("scheduler", "Scheduler"),
            ("denoise", "Denoise"),
        ]
        items = []
        for key, label in field_defs:
            value = str(kwargs.get(key, "")).strip()
            if value != '':
                items.append((label, value))
        if not items:
            return ""
        html = "<table border='1' style='border-collapse:collapse;direction: ltr;'><tr><th>Parameter</th><th>Value</th></tr>"
        for label, value in items:
            html += f"<tr><td>{label}</td><td>{value}</td></tr>"
        html += "</table>"
        return html

    def save_and_post(self,
                      image, title, content, tags, category, meta_data,
                      positive_prompt, negative_prompt,
                      seed, steps, cfg, sampler_name, scheduler,
                      denoise,
                      add_image_in_post):
        try:
            print("="*40)
            print("Uploading image to WordPress process started...")

            # --- Here ← Only read from wordpress_config.json ---
            wp_site_url, wp_user, wp_pass = load_wordpress_config()

            img_np = image
            if isinstance(img_np, list) or isinstance(img_np, tuple):
                img_np = img_np[0]
            if isinstance(img_np, torch.Tensor):
                img_np = img_np.cpu().numpy()
            while len(img_np.shape) > 3:
                img_np = img_np[0]
            if img_np.shape[-1] != 3:
                raise Exception(f"Unexpected shape after squeezing: {img_np.shape}")

            img_uint8 = (img_np * 255).astype("uint8")
            pil_img = Image.fromarray(img_uint8)
            print("Converted image to PIL format.")

            img_file_path = "comfyui_output.png"
            pil_img.save(img_file_path)
            print(f"Image saved successfully: {img_file_path}")

            # ======= 1. Upload image =======
            img_upload_url = wp_site_url.rstrip("/") + "/wp-json/wp/v2/media"
            with open(img_file_path, "rb") as img_file:
                image_headers = {
                    "Content-Disposition": f'attachment; filename="comfyui_output.png"',
                }
                response = requests.post(
                    img_upload_url,
                    headers=image_headers,
                    auth=(wp_user, wp_pass),
                    files={"file": img_file}
                )

            if response.status_code not in [200, 201]:
                print("Error uploading image! Server message:")
                print(response.text)
                return (False,)

            img_json = response.json()
            img_url = img_json.get('source_url', '')
            img_id = img_json.get('id', None)

            if not img_url or not img_id:
                print("Failed to get image url or id!")
                return (False,)
            print(f"Uploaded image URL: {img_url} | ID: {img_id}")

            # ======= 2. Tags =======
            tags = tags.strip()
            tag_list = [tag.strip() for tag in tags.replace("\n",",").split(",") if tag.strip()] if tags else []
            tag_ids = []
            if tag_list:
                tag_api_url = wp_site_url.rstrip("/") + "/wp-json/wp/v2/tags"
                for tag_name in tag_list:
                    params = {"search": tag_name}
                    tag_response = requests.get(tag_api_url, params=params, auth=(wp_user, wp_pass))
                    tag_data = tag_response.json() if tag_response.status_code == 200 else []
                    tag_id = None
                    if tag_data:
                        for t in tag_data:
                            if t.get('name', "").lower() == tag_name.lower():
                                tag_id = t.get('id')
                                break
                    if not tag_id:
                        data = {"name": tag_name, "slug": tag_name.replace(" ","-")}
                        tag_post = requests.post(tag_api_url, auth=(wp_user, wp_pass), json=data)
                        if tag_post.status_code in [200,201]:
                            tag_id = tag_post.json().get('id')
                        else:
                            print(f"[!] Tag creation failed for {tag_name}: {tag_post.status_code} {tag_post.text}")
                            continue
                    if tag_id:
                        tag_ids.append(tag_id)

            # ======= 3. Category =======
            category_ids = []
            category = category.strip()
            if category:
                cat_api_url = wp_site_url.rstrip("/") + "/wp-json/wp/v2/categories"
                if category.isdigit():
                    category_ids = [int(category)]
                else:
                    params = {"search": category}
                    cat_response = requests.get(cat_api_url, params=params, auth=(wp_user, wp_pass))
                    cat_data = cat_response.json() if cat_response.status_code == 200 else []
                    cat_id = None
                    if cat_data:
                        for c in cat_data:
                            if c.get('name', "").lower() == category.lower():
                                cat_id = c.get('id')
                                break
                    if not cat_id:
                        data = {"name": category, "slug": category.replace(" ","-")}
                        cat_post = requests.post(cat_api_url, auth=(wp_user, wp_pass), json=data)
                        if cat_post.status_code in [200,201]:
                            cat_id = cat_post.json().get('id')
                        else:
                            print(f"[!] Category creation failed {category}: {cat_post.status_code} {cat_post.text}")
                    if cat_id:
                        category_ids = [cat_id]

            # ======= 4. Build ksampler table =======
            ksampler_html = self.get_ksampler_table(
                seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
                scheduler=scheduler, denoise=denoise
            )

            # ======= 5. Prepare content =======
            content_html = f"<h2>{title}</h2>"
            content_html = f"<div>{content}</div>"

            if meta_data:
                content_html += f"<p style=\"direction: ltr;\">Meta information: {meta_data}</p>"

            if positive_prompt:
                content_html += f"<h4 style=\"direction: ltr;\">🟢 Positive Prompt:</h4><p>{positive_prompt}</p>"
            if negative_prompt:
                content_html += f"<h4 style=\"direction: ltr;\">🔴 Negative Prompt:</h4><p>{negative_prompt}</p>"

            if ksampler_html:
                content_html += "<div style=\"direction: ltr;\"> <h4>KSampler Settings Table:</h4>\n" + ksampler_html+"</div>"

            if add_image_in_post:
                content_html += f"<hr><img src='{img_url}' style='max-width:100%;'>"

            # ======= 6. Create the post =======
            wp_api_url = wp_site_url.rstrip("/") + "/wp-json/wp/v2/posts"
            data = {
                "title": title,
                "content": content_html,
                "status": "publish",
                "featured_media": img_id,
            }
            if tag_ids:
                data["tags"] = tag_ids
            if category_ids:
                data["categories"] = category_ids

            response = requests.post(
                wp_api_url,
                json=data,
                auth=(wp_user, wp_pass)
            )

            print(f"WordPress response status: {response.status_code}")
            if response.status_code == 201:
                print("✅ Post published to WordPress.")
                print("=" * 40)
                return (True,)
            else:
                print("❌ Error publishing post!")
                print(response.text)
                print("=" * 40)
                return (False,)

        except Exception as e:
            print("❌ Exception occurred:")
            print(e)
            print("=" * 40)
            return (False,)

NODE_CLASS_MAPPINGS = {
    "SaveToWordPressNode": SaveToWordPressNode
}
