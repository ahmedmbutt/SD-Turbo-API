from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware

from PIL import Image
from io import BytesIO
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    text2img = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo").to(
        "cpu"
    )

    img2img = AutoPipelineForImage2Image.from_pipe(text2img).to("cpu")

    inpaint = AutoPipelineForInpainting.from_pipe(img2img).to("cpu")

    yield {"text2img": text2img, "img2img": img2img, "inpaint": inpaint}

    del text2img
    del img2img
    del inpaint


app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"Hello": "World"}


@app.post("/text-to-image/")
async def text_to_image(
    request: Request,
    prompt: str = Form(...),
    num_inference_steps: int = Form(1),
):
    image = request.state.text2img(
        prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0
    ).images[0]

    bytes = BytesIO()
    image.save(bytes, "PNG")
    bytes.seek(0)
    return StreamingResponse(bytes, media_type="image/png")


@app.post("/image-to-image/")
async def image_to_image(
    request: Request,
    prompt: str = Form(...),
    init_image: UploadFile = File(...),
    num_inference_steps: int = Form(2),
    strength: float = Form(1.0),
):
    init_bytes = await init_image.read()
    init_image = Image.open(BytesIO(init_bytes))
    init_width, init_height = init_image.size
    init_image = init_image.convert("RGB").resize((512, 512))

    image = request.state.img2img(
        prompt,
        image=init_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=0.0,
    ).images[0]
    image = image.resize((init_width, init_height))

    bytes = BytesIO()
    image.save(bytes, "PNG")
    bytes.seek(0)
    return StreamingResponse(bytes, media_type="image/png")


@app.post("/inpainting/")
async def inpainting(
    request: Request,
    prompt: str = Form(...),
    init_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    num_inference_steps: int = Form(2),
    strength: float = Form(1.0),
):
    init_bytes = await init_image.read()
    init_image = Image.open(BytesIO(init_bytes))
    init_width, init_height = init_image.size
    init_image = init_image.convert("RGB").resize((512, 512))
    mask_bytes = await mask_image.read()
    mask_image = Image.open(BytesIO(mask_bytes))
    mask_image = mask_image.convert("RGB").resize((512, 512))

    image = request.state.inpaint(
        prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=0.0,
    ).images[0]
    image = image.resize((init_width, init_height))

    bytes = BytesIO()
    image.save(bytes, "PNG")
    bytes.seek(0)
    return StreamingResponse(bytes, media_type="image/png")
