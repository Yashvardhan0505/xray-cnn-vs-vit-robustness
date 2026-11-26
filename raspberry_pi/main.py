import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError, ImageFile

from run_inference import (
    CustomBTPNet,
    get_densenet121,
    get_resnet50,
    get_efficientnet_b0,
    get_vgg16,
    load_checkpoint,
    run_benchmark,
    transform,
    DEVICE,
)

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB; helps protect Pi memory
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
ImageFile.LOAD_TRUNCATED_IMAGES = False

app = FastAPI(
    title="Chest X-ray AI Lab",
    description="Lightweight local interface for comparing available image models.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

MODEL_CONFIGS = {
    "Custom BTP CNN": (CustomBTPNet, "best_custom_btp_net_3class/data.pkl"),
    "DenseNet-121": (get_densenet121, "best_densenet121_3class/data.pkl"),
    "ResNet-50": (get_resnet50, "best_resnet50_3class/data.pkl"),
    "EfficientNet-B0": (get_efficientnet_b0, "best_efficientnet_b0_3class/data.pkl"),
    "VGG-16": (get_vgg16, "best_vgg16_3class/data.pkl"),
}

# Models are loaded once at startup, not for each upload.
# On a Raspberry Pi with limited RAM, consider enabling only the models you need.
MODELS = {}
MODEL_LOAD_ERRORS = {}
INFERENCE_LOCK = Lock()


def load_models():
    """Load checkpoints relative to this file, independent of the working directory."""
    MODELS.clear()
    MODEL_LOAD_ERRORS.clear()
    for name, (build_fn, relative_path) in MODEL_CONFIGS.items():
        checkpoint_path = BASE_DIR / relative_path
        if not checkpoint_path.is_file():
            MODELS[name] = None
            MODEL_LOAD_ERRORS[name] = "Checkpoint file not found."
            print(f"[models] Skipping {name}: checkpoint not found at {checkpoint_path}")
            continue
        try:
            model = build_fn(3)
            model = load_checkpoint(model, str(checkpoint_path))
            if hasattr(model, "eval"):
                model.eval()
            MODELS[name] = model
            print(f"[models] Loaded {name}")
        except Exception as exc:
            MODELS[name] = None
            MODEL_LOAD_ERRORS[name] = "Model could not be loaded."
            print(f"[models] Failed to load {name}: {exc}")


@app.on_event("startup")
def startup_event():
    # Inference is CPU-friendly when PyTorch uses a modest number of threads on Pi.
    try:
        cpu_count = os.cpu_count() or 2
        torch.set_num_threads(max(1, min(2, cpu_count)))
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass
    load_models()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def list_models():
    return {
        "device": str(DEVICE),
        "models": [
            {
                "name": name,
                "available": MODELS.get(name) is not None,
                "message": None if MODELS.get(name) is not None else MODEL_LOAD_ERRORS.get(name, "Unavailable"),
            }
            for name in MODEL_CONFIGS
        ],
    }


def _run_inference(image_bytes: bytes):
    """CPU/model work runs in a worker thread; lock prevents simultaneous RAM-heavy runs."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image_format = (opened.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=400, detail="Use a JPEG, PNG, WEBP, or BMP image.")
            opened.verify()

        with Image.open(io.BytesIO(image_bytes)) as opened:
            pil_image = opened.convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        output = {}
        with INFERENCE_LOCK, torch.inference_mode():
            for name in MODEL_CONFIGS:
                model = MODELS.get(name)
                if model is None:
                    output[name] = {
                        "status": "unavailable",
                        "message": MODEL_LOAD_ERRORS.get(name, "Checkpoint file not found."),
                    }
                    continue
                try:
                    result = run_benchmark(model, name, image_tensor)
                    # Keep response JSON-safe even if a helper returns a tensor/NumPy value.
                    if isinstance(result, torch.Tensor):
                        result = result.detach().cpu().tolist()
                    elif hasattr(result, "item") and callable(result.item):
                        try:
                            result = result.item()
                        except Exception:
                            result = str(result)
                    output[name] = {"status": "success", "result": result}
                except Exception as exc:
                    print(f"[inference] {name} failed: {exc}")
                    output[name] = {
                        "status": "error",
                        "message": "Inference failed for this model.",
                    }
        return output
    finally:
        try:
            pil_image.close()
        except Exception:
            pass


@app.post("/")
@app.post("/api/predict")
async def get_output(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    contents = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum upload size is 10 MB.")
    return await run_in_threadpool(_run_inference, contents)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}
