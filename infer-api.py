# infer-api.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import os
import argparse
import uvicorn
import logging
import traceback
import time
import uuid

from acestep.pipeline_ace_step import ACEStepPipeline

# ---------- logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
)
log = logging.getLogger("ace.api")

app = FastAPI(title="ACEStep Pipeline API")

# ---------- models ----------
class GenerateInput(BaseModel):
    output_path: Optional[str] = None
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    # accept either (match CLI behavior)
    manual_seeds: Optional[List[int]] = None
    actual_seeds: Optional[List[int]] = None
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0

class GenerateOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str

# ---------- pipeline init ----------
def init_pipeline(
    checkpoint_path: str,
    bf16: bool,
    torch_compile: bool,
    cpu_offload: bool,
    overlapped_decode: bool,
    device_id: int,
) -> ACEStepPipeline:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    log.info(
        "Initializing pipeline | ckpt=%s | bf16=%s | torch_compile=%s | cpu_offload=%s | overlapped_decode=%s | device_id=%s",
        checkpoint_path, bf16, torch_compile, cpu_offload, overlapped_decode, device_id,
    )
    t0 = time.time()
    pipe = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )
    log.info("Pipeline initialized in %.2fs", time.time() - t0)
    return pipe

# ---------- simple request logging ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
        dt = (time.time() - t0) * 1000
        log.info('%s %s -> %s (%.1f ms)', request.method, request.url.path, response.status_code, dt)
        return response
    except Exception:
        dt = (time.time() - t0) * 1000
        log.error("Unhandled error for %s %s after %.1f ms\n%s",
                  request.method, request.url.path, dt, traceback.format_exc())
        raise

# ---------- endpoints ----------
@app.post("/generate", response_model=GenerateOutput)
async def generate_audio(body: GenerateInput):
    model: ACEStepPipeline = app.state.model

    # lists -> comma-joined strings; actual_seeds -> manual_seeds (CLI parity)
    seeds_list = body.manual_seeds if body.manual_seeds is not None else (body.actual_seeds or [])
    manual_seeds_str = ", ".join(map(str, seeds_list))
    oss_steps_str = ", ".join(map(str, body.oss_steps))

    # ensure save_path has a real directory that doesn't exist yet (their mkdir('') crashes; mkdir('.') may raise)
    if body.output_path and os.path.dirname(body.output_path):
        out_path = body.output_path
    else:
        out_dir = os.path.join(os.getcwd(), f"ace_out_{uuid.uuid4().hex}")
        out_name = (body.output_path if body.output_path and os.path.basename(body.output_path)
                    else "output.wav")
        out_path = os.path.join(out_dir, out_name)

    log.info(
        "GEN start | dur=%.3fs steps=%d cfg=%.2f omega=%.2f sched=%s cfg_type=%s seeds=%s oss_steps=%s save=%s",
        body.audio_duration,
        body.infer_step,
        body.guidance_scale,
        body.omega_scale,
        body.scheduler_type,
        body.cfg_type,
        manual_seeds_str,
        oss_steps_str,
        out_path,
    )

    t0 = time.time()
    try:
        model(
            audio_duration=body.audio_duration,
            prompt=body.prompt,
            lyrics=body.lyrics,
            infer_step=body.infer_step,
            guidance_scale=body.guidance_scale,
            scheduler_type=body.scheduler_type,
            cfg_type=body.cfg_type,
            omega_scale=body.omega_scale,
            manual_seeds=manual_seeds_str,
            guidance_interval=body.guidance_interval,
            guidance_interval_decay=body.guidance_interval_decay,
            min_guidance_scale=body.min_guidance_scale,
            use_erg_tag=body.use_erg_tag,
            use_erg_lyric=body.use_erg_lyric,
            use_erg_diffusion=body.use_erg_diffusion,
            oss_steps=oss_steps_str,
            guidance_scale_text=body.guidance_scale_text,
            guidance_scale_lyric=body.guidance_scale_lyric,
            save_path=out_path,
        )
        dt = time.time() - t0
        log.info("GEN done in %.2fs -> %s", dt, out_path)
        return GenerateOutput(status="success", output_path=out_path, message="OK")
    except Exception as e:
        log.error("GEN failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating audio: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ---------- main ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--torch_compile", action="store_true", default=False)
    p.add_argument("--cpu_offload", action="store_true", default=False)
    p.add_argument("--overlapped_decode", action="store_true", default=False)
    args = p.parse_args()

    try:
        app.state.model = init_pipeline(
            checkpoint_path=args.checkpoint_path,
            bf16=args.bf16,
            torch_compile=args.torch_compile,
            cpu_offload=args.cpu_offload,
            overlapped_decode=args.overlapped_decode,
            device_id=args.device_id,
        )
    except Exception as e:
        log.error("Pipeline init failed: %s\n%s", e, traceback.format_exc())
        raise

    log.info("Serving on 0.0.0.0:%d (LOG_LEVEL=%s)", args.port, LOG_LEVEL)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=LOG_LEVEL.lower())
