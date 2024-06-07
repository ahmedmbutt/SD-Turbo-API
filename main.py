from fastapi import FastAPI, Request, UploadFile, Form, File
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
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker


@asynccontextmanager
async def lifespan(app: FastAPI):
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    )

    text2img = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
    ).to("cpu")

    img2img = AutoPipelineForImage2Image.from_pipe(text2img).to("cpu")

    inpaint = AutoPipelineForInpainting.from_pipe(img2img).to("cpu")

    yield {"text2img": text2img, "img2img": img2img, "inpaint": inpaint}

    del inpaint
    del img2img
    del text2img

    del safety_checker
    del feature_extractor


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
    results = request.state.text2img(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
    )

    if not results.nsfw_content_detected[0]:
        image = results.images[0]
    else:
        image = Image.new("RGB", (512, 512), "black")

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

    results = request.state.img2img(
        prompt,
        image=init_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=0.0,
    )

    if not results.nsfw_content_detected[0]:
        image = results.images[0].resize((init_width, init_height))
    else:
        image = Image.new("RGB", (512, 512), "black")

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

    results = request.state.inpaint(
        prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=0.0,
    )

    if not results.nsfw_content_detected[0]:
        image = results.images[0].resize((init_width, init_height))
    else:
        image = Image.new("RGB", (512, 512), "black")

    bytes = BytesIO()
    image.save(bytes, "PNG")
    bytes.seek(0)
    return StreamingResponse(bytes, media_type="image/png")
