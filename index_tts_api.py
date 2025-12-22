import os
import uuid
import wave
import argparse
import re
import shutil
import sys
import uvicorn
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pyngrok import ngrok
import boto3
from botocore.exceptions import ClientError
from supabase import create_client, Client

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(
    description="IndexTTS API Server",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 for inference if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
cmd_args = parser.parse_args()

from indextts.infer_v2 import IndexTTS2

# --- Helper Functions ---

def move_punctuation_for_english_quotes(text: str) -> str:
    """
    专门处理英文引号(" " 和 ' ')，不使用正则表达式，将引号内末尾的标点移动到引号外
    """
    punctuations = {'.', '。', '!', '！', '?', '？'}
    quote_chars = {'"', "'"}
    
    char_list = list(text)
    n = len(char_list)
    
    i = 0
    while i < n:
        current_char = char_list[i]
        if current_char in quote_chars:
            left_quote_index = i
            right_quote_index = -1
            for j in range(left_quote_index + 1, n):
                if char_list[j] == current_char:
                    right_quote_index = j
                    break # 找到后立即停止
            if right_quote_index != -1:
                char_before_right_quote_index = right_quote_index - 1
                if char_before_right_quote_index > left_quote_index and char_list[char_before_right_quote_index] in punctuations:
                    punc = char_list[char_before_right_quote_index]
                    quote = char_list[right_quote_index]
                    char_list[char_before_right_quote_index] = quote
                    char_list[right_quote_index] = punc
                i = right_quote_index + 1
            else:
                i += 1
        else:
            i += 1
    return "".join(char_list)

def chunk_text(text, max_length=250, min_length=15) -> List[str]:
    """
    Splits the text into chunks with robust handling for the final chunk.
    """
    text = text.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"').replace("·", "").replace("…", "，")
    text = re.sub(r'《([^》]+)》', lambda m: f"《{m.group(1).replace(',', ' ')}》", text)
    # 去除所有空白符（空格、tab、换行符等）
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'([。！？!?，,])+', r'\1', text)
    text = move_punctuation_for_english_quotes(text)
    # 在连续英文与数字之间插入连字符，例如 U2 -> U-2，3M -> 3-M；纯数字或纯英文不处理
    text = re.sub(r'([A-Za-z])(?=\d)', r'\1-', text)
    text = re.sub(r'(\d)(?=[A-Za-z])', r'\1-', text)
    sentences = re.split(r'(?<=[。！？!?])', text)

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if current_chunk and len(current_chunk) + len(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        if chunks and len(current_chunk.strip()) < min_length:
            chunks[-1] += " " + current_chunk.strip()
        else:
            chunks.append(current_chunk.strip()) 

    return chunks

def merge_wavs(paths: List[str], out_path: str) -> None:
    """Merges multiple WAV files into a single file."""
    if not paths:
        raise ValueError("No input WAV files to merge.")
    with wave.open(paths[0], "rb") as w0:
        params = w0.getparams()
    with wave.open(out_path, "wb") as out:
        out.setparams(params)
        for p in paths:
            with wave.open(p, "rb") as w:
                out.writeframes(w.readframes(w.getnframes()))

class TTSService:
    """A service class to handle Text-to-Speech inference."""
    def __init__(
        self,
        model_dir: str,
        cfg_path: str,
        use_fp16: bool = True,
        use_cuda_kernel: bool = False,
    ) -> None:
        self.tts = IndexTTS2(
            model_dir=model_dir,
            cfg_path=cfg_path,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
        )
        self._sr = int(self.tts.cfg.s2mel['preprocess_params']['sr'])
        self._hop = int(self.tts.cfg.s2mel['preprocess_params']['spect_params']['hop_length'])
        self._code_to_frame = 1.72

    def infer_chunk(
        self,
        text: str,
        voice_path: str,
        out_path: str,
        duration_sec: Optional[float] = None,
    ) -> str:
        kwargs = {}
        if duration_sec is not None and duration_sec > 0:
            frames = duration_sec * (self._sr / self._hop)
            code_len = int(max(32, round(frames / self._code_to_frame)))
            kwargs["max_mel_tokens"] = code_len
        return self.tts.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=out_path,
            **kwargs,
        )

# --- FastAPI Application ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the TTS model on application startup and handle shutdown.
    Aligns with webui.py model loading logic.
    """
    global tts_service
    print("--- Lifespan startup ---")

    # 1. Check if model directory exists
    if not os.path.isdir(cmd_args.model_dir):
        print(f"\033[91mError: Model directory '{cmd_args.model_dir}' not found. Please specify a valid path with --model_dir.\033[0m")
        sys.exit(1)

    # 2. Check for required model files
    config_path = os.path.join(cmd_args.model_dir, "config.yaml")
    required_files = ["bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt", "config.yaml"]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(cmd_args.model_dir, f))]

    if missing_files:
        print(f"\033[91mError: The following required files are missing from '{cmd_args.model_dir}':\033[0m")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    # 3. Load the TTS model
    try:
        print("Loading TTS model...")
        tts_service = TTSService(
            model_dir=cmd_args.model_dir,
            cfg_path=config_path,
            use_fp16=cmd_args.fp16,
            use_cuda_kernel=cmd_args.cuda_kernel,
        )
        print("\033[92mTTS Service loaded successfully.\033[0m")
    except Exception as e:
        print(f"\033[91mError loading TTS Service: {e}\033[0m")
        sys.exit(1)

    # 4. Initialize R2 client (optional)
    global r2_client, supabase_client
    r2_account_id = os.environ.get("R2_ACCOUNT_ID")
    r2_access_key = os.environ.get("R2_ACCESS_KEY_ID")
    r2_secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    r2_bucket_name = os.environ.get("R2_BUCKET_NAME")
    
    if all([r2_account_id, r2_access_key, r2_secret_key, r2_bucket_name]):
        try:
            print("Initializing R2 client...")
            r2_client = boto3.client(
                service_name='s3',
                endpoint_url=f'https://{r2_account_id}.r2.cloudflarestorage.com',
                aws_access_key_id=r2_access_key,
                aws_secret_access_key=r2_secret_key,
                region_name='auto'
            )
            print("\033[92mR2 client initialized successfully.\033[0m")
        except Exception as e:
            print(f"\033[93mWarning: Failed to initialize R2 client: {e}\033[0m")
            print("\033[93m/api/synthesize_with_storage endpoint will not be available.\033[0m")
            r2_client = None
    else:
        print("\033[93mR2 configuration not found. /api/synthesize_with_storage endpoint will not be available.\033[0m")
    
    # 5. Initialize Supabase client (optional)
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if all([supabase_url, supabase_key]):
        try:
            print("Initializing Supabase client...")
            supabase_client = create_client(supabase_url, supabase_key)
            print("\033[92mSupabase client initialized successfully.\033[0m")
        except Exception as e:
            print(f"\033[93mWarning: Failed to initialize Supabase client: {e}\033[0m")
            print("\033[93m/api/synthesize_with_storage endpoint will not be available.\033[0m")
            supabase_client = None
    else:
        print("\033[93mSupabase configuration not found. /api/synthesize_with_storage endpoint will not be available.\033[0m")
        
    yield
    
    print("--- Lifespan shutdown ---")
    # Clean up the ML models and release the resources
    tts_service = None
    r2_client = None
    supabase_client = None

app = FastAPI(title="IndexTTS API", description="A pure API for IndexTTS2 using FastAPI and ngrok.", lifespan=lifespan)

tts_service: Optional[TTSService] = None
r2_client = None
supabase_client: Optional[Client] = None

@app.post("/api/synthesize")
async def synthesize(
    voice_path: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    duration_sec: Optional[float] = Form(None),
):
    """
    Synthesize speech from text or a text file with chunking.
    Saves intermediate and final files to a specified output directory
    with a filename based on the input file.
    """
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service is not available. Check model path.")

    if not text and not file:
        raise HTTPException(status_code=400, detail="Either 'text' (form field) or 'file' (upload) must be provided.")

    input_text = ""
    base_filename = ""
    if file and file.filename:
        input_text = (await file.read()).decode("utf-8")
        base_filename = os.path.splitext(file.filename)[0]
    elif text:
        input_text = text
        base_filename = f"synthesis_{uuid.uuid4()}"

    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # Use the specified final output directory for all files
    output_dir = "/content/drive/MyDrive/Index-TTS/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    wav_paths = []
    try:
        chunks = chunk_text(input_text, max_length=250)
        print(f"Starting synthesis for {len(chunks)} chunks, using base name: '{base_filename}'")

        for i, chunk in enumerate(chunks):
            # Intermediate files are named based on the base filename and chunk index
            out_path = os.path.join(output_dir, f"{base_filename}_{i+1}.wav")
            try:
                tts_service.infer_chunk(
                    text=chunk,
                    voice_path=voice_path,
                    out_path=out_path,
                    duration_sec=duration_sec if i == 0 else None
                )
                wav_paths.append(out_path)
                print(f"  - Chunk {i+1}/{len(chunks)} saved to: {out_path}")
            except Exception as e:
                print(f"\033[91mError processing chunk {i+1}: {e}\033[0m")
                raise HTTPException(status_code=500, detail=f"Error processing chunk {i+1}: {str(e)}")

        if not wav_paths:
            raise HTTPException(status_code=500, detail="Synthesis failed, no audio chunks were generated.")

        # Final merged file is named based on the base filename
        final_out_path = os.path.join(output_dir, f"{base_filename}.wav")
        
        merge_wavs(wav_paths, final_out_path)
        print(f"\033[92mSynthesis complete. Final audio at: {final_out_path}\033[0m")

        return {"output_path": final_out_path, "message": "Synthesis successful."}

    finally:
        # Clean up intermediate chunk files after merging
        if wav_paths:
            print("Cleaning up intermediate files...")
            for p in wav_paths:
                try:
                    os.remove(p)
                    print(f"  - Removed {p}")
                except OSError as e:
                    print(f"\033[91mError removing intermediate file {p}: {e}\033[0m")

# --- Helper Functions for R2 and Supabase ---

async def upload_to_r2(file_path: str, filename: str) -> str:
    """
    Upload a file to Cloudflare R2 and return the public URL.
    
    Args:
        file_path: Local path to the file to upload
        filename: Desired filename in R2
        
    Returns:
        Public URL of the uploaded file
        
    Raises:
        HTTPException: If upload fails
    """
    if r2_client is None:
        raise HTTPException(status_code=503, detail="R2 service is not configured.")
    
    bucket_name = os.environ.get("R2_BUCKET_NAME")
    r2_public_url = os.environ.get("R2_PUBLIC_URL", "")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 📤 Uploading to R2: {filename}")
        
        with open(file_path, 'rb') as f:
            r2_client.upload_fileobj(
                f,
                bucket_name,
                filename,
                ExtraArgs={'ContentType': 'audio/wav'}
            )
        
        # Construct public URL
        public_url = f"{r2_public_url.rstrip('/')}/{filename}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] ✅ R2 upload successful: {public_url}")
        return public_url
        
    except ClientError as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] ❌ R2 upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"R2 upload failed: {str(e)}")
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] ❌ Unexpected error during R2 upload: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

async def save_to_supabase(filename: str, url: str) -> dict:
    """
    Save file metadata to Supabase audio_url table.
    
    Args:
        filename: Name of the audio file
        url: Public URL of the file
        
    Returns:
        Dictionary containing the inserted record
        
    Raises:
        HTTPException: If database operation fails
    """
    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase service is not configured.")
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 💾 Saving to Supabase: {filename}")
        
        data = {
            "filename": filename,
            "url": url,
        }
        
        response = supabase_client.table("audio_url").insert(data).execute()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] ✅ Supabase save successful. Record ID: {response.data[0].get('id') if response.data else 'N/A'}")
        
        return response.data[0] if response.data else {}
        
    except Exception as e:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] ❌ Supabase save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Supabase save failed: {str(e)}")

@app.post("/api/synthesize_with_storage")
async def synthesize_with_storage(
    voice_path: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    duration_sec: Optional[float] = Form(None),
):
    """
    Synthesize speech from text or a text file with chunking,
    upload to R2, and save metadata to Supabase.
    The local file will be deleted after successful upload to save space.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"[{timestamp}] 🚀 Starting synthesize_with_storage request")
    print(f"{'='*60}")
    
    # Check service availability
    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service is not available. Check model path.")
    
    if r2_client is None or supabase_client is None:
        raise HTTPException(
            status_code=503, 
            detail="R2 or Supabase service is not configured. Please set environment variables."
        )

    if not text and not file:
        raise HTTPException(status_code=400, detail="Either 'text' (form field) or 'file' (upload) must be provided.")

    # Parse input
    input_text = ""
    base_filename = ""
    if file and file.filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 📄 Reading uploaded file: {file.filename}")
        input_text = (await file.read()).decode("utf-8")
        base_filename = os.path.splitext(file.filename)[0]
    elif text:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 📝 Using text input (length: {len(text)} chars)")
        input_text = text
        base_filename = f"synthesis_{uuid.uuid4()}"

    if not input_text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # Use the specified final output directory for all files
    output_dir = "/content/drive/MyDrive/Index-TTS/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    wav_paths = []
    final_out_path = None
    
    try:
        # Step 1: Generate audio
        chunks = chunk_text(input_text, max_length=250)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 🎵 Starting TTS synthesis for {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            out_path = os.path.join(output_dir, f"{base_filename}_{i+1}.wav")
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"[{timestamp}] 🔊 Processing chunk {i+1}/{len(chunks)}...")
                
                tts_service.infer_chunk(
                    text=chunk,
                    voice_path=voice_path,
                    out_path=out_path,
                    duration_sec=duration_sec if i == 0 else None
                )
                wav_paths.append(out_path)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"[{timestamp}] ✅ Chunk {i+1}/{len(chunks)} complete: {out_path}")
            except Exception as e:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"[{timestamp}] ❌ Error processing chunk {i+1}: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing chunk {i+1}: {str(e)}")

        if not wav_paths:
            raise HTTPException(status_code=500, detail="Synthesis failed, no audio chunks were generated.")

        # Step 2: Merge chunks
        final_out_path = os.path.join(output_dir, f"{base_filename}.wav")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 🔗 Merging {len(wav_paths)} chunk(s) into final audio...")
        
        merge_wavs(wav_paths, final_out_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] ✅ Audio merge complete: {final_out_path}")

        # Step 3: Upload to R2
        r2_filename = f"{base_filename}.wav"
        r2_url = await upload_to_r2(final_out_path, r2_filename)

        # Step 4: Save to Supabase
        supabase_record = await save_to_supabase(r2_filename, r2_url)

        # Step 5: Delete local file to save space
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[{timestamp}] 🗑️  Deleting local file to save space: {final_out_path}")
        try:
            os.remove(final_out_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[{timestamp}] ✅ Local file deleted successfully")
        except OSError as e:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[{timestamp}] ⚠️  Warning: Failed to delete local file: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n{'='*60}")
        print(f"[{timestamp}] 🎉 Synthesis with storage complete!")
        print(f"{'='*60}\n")

        return {
            "message": "Synthesis with storage successful.",
            "r2_url": r2_url,
            "supabase_record": supabase_record,
            "filename": r2_filename,
        }

    finally:
        # Clean up intermediate chunk files
        if wav_paths:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"[{timestamp}] 🧹 Cleaning up intermediate chunk files...")
            for p in wav_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"  ✓ Removed: {p}")
                except OSError as e:
                    print(f"  ⚠️  Error removing {p}: {e}")

def main():

    """Sets up ngrok tunnel and starts the FastAPI server."""
    # Get config from environment variables
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", 7890))
    ngrok_authtoken = os.environ.get("NGROK_AUTHTOKEN")

    if ngrok_authtoken:
        ngrok.set_auth_token(ngrok_authtoken)
    
    # Set up ngrok tunnel
    public_url = ngrok.connect(port, "http")
    
    print("\n" + "="*60)
    print("🚀 IndexTTS API is running!")
    print(f"✅ Public API Endpoint (ngrok): {public_url.public_url}/api/synthesize")
    print(f"✅ Public Docs URL (ngrok):   {public_url.public_url}/docs")
    print(f"✅ Local Docs URL:            http://{host}:{port}/docs")
    print("="*60 + "\n")
    print("Hint: Use a POST request to the API endpoint with form data:")
    print("  - 'voice_path': Path to the voice sample in the remote (e.g., Colab) environment.")
    print("  - 'file':       Text file to synthesize (e.g., @local_file.txt).")
    print("  - 'text':       Alternatively, a string of text to synthesize.")
    print("-"*60)

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
