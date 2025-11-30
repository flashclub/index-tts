import os
import uuid
import wave
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from indextts.infer import IndexTTS


def chunk_text(text: str, max_len: int) -> List[str]:
    seps = {".", "!", "?", "。", "！", "？", "；", ";", "，", ",", "\n", " "}
    res: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_len, n)
        k = j
        while k > i and text[k - 1] not in seps:
            k -= 1
        if k == i:
            k = j
        res.append(text[i:k].strip())
        i = k
    return [s for s in res if s]


def merge_wavs(paths: List[str], out_path: str) -> None:
    if not paths:
        raise ValueError("no input wavs")
    with wave.open(paths[0], "rb") as w0:
        nch = w0.getnchannels()
        sampwidth = w0.getsampwidth()
        fr = w0.getframerate()
    with wave.open(out_path, "wb") as out:
        out.setnchannels(nch)
        out.setsampwidth(sampwidth)
        out.setframerate(fr)
        for p in paths:
            with wave.open(p, "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))


class TTSService:
    def __init__(
        self,
        model_dir: str,
        cfg_path: str,
        use_fp16: bool = True,
        use_cuda_kernel: bool = False,
    ) -> None:
        self.tts = IndexTTS(
            model_dir=model_dir,
            cfg_path=cfg_path,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
        )

    def infer_chunk(
        self,
        text: str,
        voice_path: str,
        out_path: str,
        verbose: bool = False,
        max_text_tokens_per_segment: Optional[int] = None,
    ) -> str:
        kwargs = {}
        if max_text_tokens_per_segment is not None:
            kwargs["max_text_tokens_per_segment"] = max_text_tokens_per_segment
        self.tts.infer_fast(
            audio_prompt=voice_path,
            text=text,
            output_path=out_path,
            verbose=verbose,
            **kwargs,
        )
        return out_path


app = FastAPI()

MODEL_DIR = os.getenv("INDEXTTS_MODEL_DIR", "checkpoints")
CFG_PATH = os.getenv("INDEXTTS_CFG_PATH", os.path.join(MODEL_DIR, "config.yaml"))
CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", "300"))
service = TTSService(MODEL_DIR, CFG_PATH)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/synthesize")
async def synthesize(
    file: UploadFile = File(...),
    voice_path: str = Form(...),
    drive_dir: Optional[str] = Form(None),
    chunk_length: int = Form(CHUNK_LENGTH),
    session_id: Optional[str] = Form(None),
) -> JSONResponse:
    content = await file.read()
    text = content.decode("utf-8")
    sid = session_id or str(uuid.uuid4())
    base_dir = drive_dir or os.getenv("GOOGLE_DRIVE_DIR", "outputs")
    out_dir = os.path.join(base_dir, sid)
    os.makedirs(out_dir, exist_ok=True)
    chunks = chunk_text(text, chunk_length)
    chunk_files: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        name = f"chunk_{idx:04d}.wav"
        path = os.path.join(out_dir, name)
        service.infer_chunk(
            text=chunk,
            voice_path=voice_path,
            out_path=path,
            verbose=False,
        )
        chunk_files.append(path)
    final_path = os.path.join(out_dir, "final.wav")
    merge_wavs(chunk_files, final_path)
    return JSONResponse(
        {
            "session_id": sid,
            "chunk_count": len(chunk_files),
            "chunk_files": chunk_files,
            "final_file": final_path,
        }
    )


def main() -> None:
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    print(f"IndexTTS API running: http://{host}:{port}")
    print(f"Docs: http://127.0.0.1:{port}/docs")
    print(
        f"curl -F \"file=@/path/text.txt\" -F \"voice_path=/path/voice.wav\" -F \"drive_dir=/content/drive/MyDrive/IndexTTS/outputs\" http://127.0.0.1:{port}/synthesize"
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
