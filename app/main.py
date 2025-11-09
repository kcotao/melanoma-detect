import io, os, time
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

from src.model_def import load_model_from_state, CLASS_NAMES
from src.preprocess import preprocess_pil

# Variables entorno

DEVICE = "cpu"
MODEL_PATH = os.getenv("MODEL_PATH", "weights/resnet18_finetune_best.pth")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

# Fastapi
app = FastAPI(title="melanoma-detector", version=MODEL_VERSION)
app.add_middleware(CORSORMiddleware := CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = load_model_from_state(MODEL_PATH, DEVICE)

def _predict_pil(img: Image.Image):
    x = preprocess_pil(img).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        logits = _model(x)[0]
        probs = torch.softmax(logits, dim=0).cpu().detach().numpy()
    ms = int((time.time() - t0) * 1000)

    order = np.argsort(probs)[::-1]
    top = [(CLASS_NAMES[i], float(probs[i])) for i in order]
    return top, ms

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
async def predoct(file: UploadFile = File(...)):
    try:
        b = await file.read()
        img = Image.open(io.BytesIO(b))
    except Exception:
        raise HTTPException(400, "Archivo no es una imagen válida")
    top, ms = _predict_pil(img)
    return {
        "top1": {"label": top[0][0], "prob": top[0][1]},
        "probs": [{"label": l, "prob": p} for l, p in top],
        "inference_ms": ms,
        "disclaimer": "Soporte a la decisión; no es diagnóstico clinico."
    }

# Gradio UI
def gradio_infer(img: Image.Image):
    top, ms = _predict_pil(img)
    # filas: [Clase, Probabilidad]
    rows = [[t[0], round(t[1], 4)] for t in top]
    title = f"Predicción: {top[0][0]} ({top[0][1]:.2%}) • {ms} ms"
    return title, rows



with gr.Blocks(title="melanoma-detect") as demo:
    gr.Markdown("# melanoma-detect\nSube una imagen y obtén probabilidades por clase\n*(no reemplaza evaluación médica)*")
    inp = gr.Image(type="pil", label="Imagen de la lesión")
    out_title = gr.Markdown()
    out_table = gr.Dataframe(headers=["Clase", "Probabilidad"], row_count=3, col_count=2, interactive=False)
    btn = gr.Button("Predecir")
    btn.click(fn=gradio_infer, inputs=inp, outputs=[out_title, out_table])

# Montar Gradio en la raíz ("/")
app = gr.mount_gradio_app(app, demo, path="/")