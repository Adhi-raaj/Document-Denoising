"""
Flask frontend for Document Denoising.
Run locally after downloading the trained model from Google Drive/GitHub.
"""

import io
import os
import sys
import time
import math

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import DenoisingUNet
from src.utils import psnr, ssim, denoise_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# ─── Model Loading ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "best_model.pth")
DEVICE = "cpu"  # CPU for local inference

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading model...")
        _model = DenoisingUNet(features=[32, 64, 128, 256])

        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            if "model_state" in state:
                state = state["model_state"]
            key_map = {
                "enc."      : "encoder_blocks.",
                "dec."      : "decoder_blocks.",
                "out_conv." : "output_conv.",
            }
            remapped = {}
            for k, v in state.items():
                new_k = k
                for old, new in key_map.items():
                    if k.startswith(old):
                        new_k = new + k[len(old):]
                        break
                remapped[new_k] = v
            _model.load_state_dict(remapped)
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠ Model not found at {MODEL_PATH}")

        _model.eval()
        # ── Speed up CPU inference ──
        torch.set_num_threads(os.cpu_count())
        _model = torch.jit.script(_model)  # TorchScript compile
        print(f"✓ Model optimized for CPU")
    return _model

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ready = os.path.exists(MODEL_PATH)
    return render_template("index.html", model_ready=model_ready)


@app.route("/denoise", methods=["POST"])
def denoise():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    try:
        # Load image
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
        original_size = pil_img.size  # (W, H)

        # Resize if too large (for fast CPU inference)
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            ratio = max_dim / max(pil_img.size)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        img_np = np.array(pil_img, dtype=np.float32) / 255.0

        # Run inference
        model = get_model()
        t0 = time.time()

        patch_size = 256 if max(img_np.shape) > 256 else 128
        denoised = denoise_image(model, img_np, patch_size=patch_size, overlap=32, device=DEVICE)

        inference_time = time.time() - t0

        # Compute metrics
        noisy_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        clean_t = torch.from_numpy(denoised).unsqueeze(0).unsqueeze(0)
        psnr_val = psnr(clean_t, noisy_t)
        ssim_val = ssim(clean_t, noisy_t)

        # Convert denoised image to PNG bytes
        denoised_uint8 = (denoised * 255).clip(0, 255).astype(np.uint8)
        out_pil = Image.fromarray(denoised_uint8, mode="L").convert("RGB")
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG", optimize=True)
        buf.seek(0)

        return jsonify({
            "success": True,
            "denoised_image": _img_to_b64(buf),
            "original_size": f"{original_size[0]}×{original_size[1]}",
            "processed_size": f"{pil_img.width}×{pil_img.height}",
            "inference_time": f"{inference_time:.2f}s",
            "psnr": f"{psnr_val:.2f} dB" if not math.isinf(psnr_val) else "∞ dB",
            "ssim": f"{ssim_val:.4f}",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _img_to_b64(buf: io.BytesIO) -> str:
    import base64
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


@app.route("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "ok",
        "model_loaded": model_ready,
        "device": DEVICE,
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Document Denoising — Local Server")
    print("="*50)
    get_model()  # Pre-load model
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
