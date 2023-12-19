from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests
import io
from io import BytesIO

app = FastAPI(title="Visual QA API", version="0.0.1")


# Vilt model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, text):
    try:
        # load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")
        # prepare input
        encoding = processor(img, text, return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label.get(idx, "")

    except Exception as e:
        return str(e)

    return answer


#fastapi code
@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/answer")
async def process_image(image: UploadFile = File(...), text: str = None):
    try:
        answer = get_answer(await image.read(), text)
        return JSONResponse({"answer": answer})
    
    except Exception as e:
        return JSONResponse({"error": str(e)})