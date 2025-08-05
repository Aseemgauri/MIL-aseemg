import os
# Suppress HuggingFace Hub symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import warnings
# Suppress torch.cuda.amp deprecation warning from SpeechBrain
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*", category=FutureWarning)

import torch
import sounddevice as sd
import queue
import numpy as np
import time
import gc
import threading
from speechbrain.inference.ASR import StreamingASR
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from collections import deque
from transformers import pipeline

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA not available. Using CPU.")
    device = "cpu"
else:
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    device = "cuda"

# === Load ASR Model ===
model_dir = os.path.join(os.path.expanduser("~"), ".speechbrain", "asr_model")
os.makedirs(model_dir, exist_ok=True)

def check_model_exists(model_dir):
    """Check if all required model files exist locally"""
    required_files = ["hyperparams.yaml", "model.ckpt", "normalizer.ckpt", "tokenizer.ckpt"]
    
    if not os.path.exists(model_dir):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    
    return True

# Load or download model
try:
    if check_model_exists(model_dir):
        # Load from local cache
        asr_model = StreamingASR.from_hparams(
            source=model_dir,
            run_opts={"device": device}
        )
    else:
        # Download model
        try:
            from huggingface_hub import hf_hub_download
            
            required_files = [
                "hyperparams.yaml",
                "model.ckpt", 
                "normalizer.ckpt",
                "tokenizer.ckpt"
            ]
            
            for file in required_files:
                try:
                    hf_hub_download(
                        repo_id="speechbrain/asr-streaming-conformer-librispeech",
                        filename=file,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    # Only print error if it's a critical file
                    if file in required_files:
                        print(f"Error downloading {file}: {e}")
            
            asr_model = StreamingASR.from_hparams(
                source=model_dir,
                run_opts={"device": device}
            )
            
        except Exception as download_e:
            # Fallback to direct download
            asr_model = StreamingASR.from_hparams(
                source="speechbrain/asr-streaming-conformer-librispeech",
                savedir=model_dir,
                run_opts={"device": device}
            )
            
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# === Load Qwen3-0.6B for text analysis ===
print("Loading Qwen3-0.6B for text analysis...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    
    # Load tokenizer and model directly for better control
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    # Move to device if not using device_map
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("LLM loaded on GPU.")
    else:
        model = model.to("cpu")
        print("LLM loaded on CPU.")
    
except Exception as e:
    print(f"Error loading Qwen3-0.6B: {e}")
    print("Falling back to pipeline approach...")
    try:
        llm_pipeline = pipeline(
            "text-generation",
            model="gpt2",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("Fallback GPT-2 loaded successfully!")
        model = None
        tokenizer = None
    except Exception as e:
        print(f"Error loading fallback model: {e}")
        exit(1)

# === Audio Config ===
sr = 16000
chunk_config = DynChunkTrainConfig(chunk_size=24, left_context_size=96)
expected_chunk_frames = asr_model.get_chunk_size_frames(chunk_config)
chunk_samples = expected_chunk_frames

# Calculate context parameters
context_duration_ms = 3840  # Target context duration
context_samples = int(context_duration_ms * sr / 1000)  # 61,440 samples
max_context_chunks = context_samples // chunk_samples  # Maximum chunks to keep in context

# === Enhanced Context Manager ===
class FixedContextManager:
    def __init__(self, asr_model, chunk_config, max_context_chunks):
        self.asr_model = asr_model
        self.chunk_config = chunk_config
        self.max_context_chunks = max_context_chunks
        self.context_buffer = deque(maxlen=max_context_chunks)
        self.streaming_context = None
        self.reset_context()
        
    def reset_context(self):
        """Reset the streaming context"""
        self.streaming_context = self.asr_model.make_streaming_context(self.chunk_config)
        self.context_buffer.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def should_reset_context(self, chunk_count):
        """Determine if context should be reset"""
        return chunk_count > 0 and chunk_count % 50 == 0  # Reset every 50 chunks
        
    def transcribe_chunk(self, chunk_tensor, chunk_count):
        """Transcribe chunk with proper context management"""
        
        # Check if we need to reset context
        if self.should_reset_context(chunk_count):
            self.reset_context()
            
        # Transcribe chunk
        with torch.no_grad():
            audio_batch = chunk_tensor.unsqueeze(0)
            text_chunks = self.asr_model.transcribe_chunk(self.streaming_context, audio_batch)
            text = text_chunks[0] if text_chunks else ""
        
        return text

# === Qwen3-0.6B Text Analysis Manager ===
class TextAnalysisManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.transcription_buffer = []
        self.analysis_interval_seconds = 5  # 5 seconds of audio content
        self.buffer_lock = threading.Lock()
        self.analysis_thread = None
        self.running = True
        
    def add_transcription(self, text):
        """Add new transcription to buffer (includes both speech and silence markers)"""
        if text.strip():
            with self.buffer_lock:
                self.transcription_buffer.append(text.strip())
                
    def analyze_text_async(self, text_content):
        """Analyze text content using Qwen3-0.6B in a separate thread"""
        def analysis_worker():
            try:
                if not text_content.strip():
                    return
                    
                # Check if we have the model loaded
                if self.model is None or self.tokenizer is None:
                    print("Model not loaded, skipping analysis")
                    return
                    
                import time
                start_time = time.time()
                # Create messages for Qwen3-0.6B chat template
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "Based on this speech transcription, output ONLY ONE SENTENCE that summarizes the main topic of the text. "
                            "Do not output any lists, numbers, or extra explanation. Be concise and specific.\n\n"
                            "Note: The transcription includes [silence] markers to show when no speech was detected.\n\n"
                            f"Transcription: \"{text_content}\""
                        )
                    }
                ]
                
                # Apply chat template with thinking mode enabled
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Disable thinking mode for direct answer
                )
                
                # Tokenize input with proper attention mask
                model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
                if torch.cuda.is_available():
                    model_inputs = model_inputs.to(self.model.device)
                
                # Generate response with optimized parameters for Qwen3-0.6B
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=50,  # Lowered for faster response
                        temperature=0.6,     # Optimal for thinking mode
                        top_p=0.95,         # Optimal for thinking mode
                        top_k=20,           # Optimal for thinking mode
                        min_p=0.0,          # Optimal for thinking mode
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
                    )
                    
                # Decode the response
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Parse the response (may contain thinking content)
                summary = self._parse_qwen_response(response)
                elapsed = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"✅ QWEN3-0.6B ANALYSIS COMPLETE (15 seconds processed):")
                print(f"Transcription: {text_content[:100]}{'...' if len(text_content) > 100 else ''}")
                print(f"Analysis: {summary}")
                print(f"⏱️ LLM processing time: {elapsed:.2f} seconds")
                print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"\n❌ Error in Qwen3-0.6B analysis: {e}")
                print(f"Transcription was: {text_content[:100]}{'...' if len(text_content) > 100 else ''}")
                print("Continuing with next audio batch...\n")
                
        # Start analysis in background thread
        if self.analysis_thread is None or not self.analysis_thread.is_alive():
            self.analysis_thread = threading.Thread(target=analysis_worker)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            
    def _parse_qwen_response(self, response):
        # Qwen3-0.6B may include thinking content wrapped in <think> tags
        # Remove thinking content and extract the actual response
        if "<think>" in response and "</think>" in response:
            # Extract content after thinking
            parts = response.split("</think>")
            if len(parts) > 1:
                actual_response = parts[1].strip()
            else:
                actual_response = response
        else:
            actual_response = response
            
        # Clean up the response
        actual_response = actual_response.strip()
        
        # If response is too long, truncate it
        if len(actual_response) > 300:
            actual_response = actual_response[:300] + "..."
            
        return actual_response
            
    def check_for_analysis(self):
        with self.buffer_lock:
            if (
                len(self.transcription_buffer) >= self.analysis_interval_seconds
                and (self.analysis_thread is None or not self.analysis_thread.is_alive())
            ):
                # Take exactly 5 seconds of audio content (first 5 items)
                audio_batch = self.transcription_buffer[:self.analysis_interval_seconds]
                # Remove those 5 seconds from buffer for streaming
                self.transcription_buffer = self.transcription_buffer[self.analysis_interval_seconds:]
                # Combine the 5 seconds of transcription
                combined_text = " ".join(audio_batch)
                print(f"\n🔄 Analyzing 5 seconds of audio content...")
                print(f"Buffer content: {combined_text[:50]}{'...' if len(combined_text) > 50 else ''}")
                # Analyze text asynchronously
                self.analyze_text_async(combined_text)
            
    def stop(self):
        self.running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)

# === Streaming Setup ===
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(f"Audio error: {status}")
    q.put(indata.copy())

stream = sd.InputStream(
    samplerate=sr,
    channels=1,
    dtype='float32',
    callback=callback,
    blocksize=chunk_samples
)

# === Main Streaming Loop ===
context_manager = FixedContextManager(asr_model, chunk_config, max_context_chunks)

# Initialize text analyzer with Qwen3-0.6B model
if 'model' in locals() and model is not None:
    text_analyzer = TextAnalysisManager(model, tokenizer)
    print("\nStarting real-time ASR-to-Qwen3-0.6B test...")
    print("Speak into your microphone. Qwen3-0.6B analysis will occur every 5 seconds.")
else:
    print("\nQwen3-0.6B model not loaded. Analysis will be skipped.")
    text_analyzer = None

chunk_count = 0
print("Press Ctrl+C to stop.\n")

stream.start()

try:
    while True:
        try:
            chunk = q.get(timeout=1.0)
        except queue.Empty:
            # No need to check for analysis on timeout since we're counting buffer items, not time
            continue
            
        chunk_count += 1
        chunk_tensor = torch.from_numpy(chunk.flatten()).float().to(device)
        
        # Normalize
        chunk_tensor = chunk_tensor / max(chunk_tensor.abs().max(), 1e-6)
        
        # Transcribe with context management
        text = context_manager.transcribe_chunk(chunk_tensor, chunk_count)
        
        # Print real-time results and add to buffer
        # Note: Each chunk represents exactly 1 second of audio content
        if text.strip():
            print(f"[ASR] {text}")
            # Add to analysis buffer if analyzer is available
            if text_analyzer is not None:
                text_analyzer.add_transcription(text)
        else:
            print("[silence]")
            # Add silence periods to analysis buffer as well - helps with pattern analysis
            if text_analyzer is not None:
                text_analyzer.add_transcription("[silence]")
            
        # Check if we have 5 seconds of audio content ready for analysis
        if text_analyzer is not None:
            text_analyzer.check_for_analysis()

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"Error: {e}")
    
finally:
    if text_analyzer is not None:
        text_analyzer.stop()
    stream.stop()
    stream.close()
    print("System stopped.") 