#!/usr/bin/env python3
"""
Dos Amigos - Dual-Model Speech-to-Text with Whisper Small MLX and Parakeet MLX.
"""

import sys
import threading
import time
import subprocess
import tempfile
import os
import argparse
from pathlib import Path

try:
    import sounddevice as sd
    import numpy as np
    import scipy.io.wavfile as wavfile
    from pynput import keyboard
    import mlx.core as mx
except ImportError as e:
    print(f"Missing core dependency: {e}")
    print("Run the setup script first: python scripts/setup_offline.py")
    sys.exit(1)

def get_model_config(amigo_type):
    """Map amigo types to actual model configurations"""
    model_configs = {
        "ligero": {
            "type": "whisper",
            "name": "whisper-small-mlx",
            "repo": "mlx-community/whisper-small-mlx",
            "description": "Lightweight & fast"
        },
        "preciso": {
            "type": "parakeet",
            "name": "parakeet-tdt-0.6b-v3",
            "repo": "mlx-community/parakeet-tdt-0.6b-v3",
            "description": "Maximum accuracy"
        }
    }
    return model_configs.get(amigo_type)


class DosAmigos:
    def __init__(self, amigo_type="ligero", model_path=None):
        self.amigo_type = amigo_type.lower()
        self.model_config = get_model_config(self.amigo_type)
        
        if not self.model_config:
            raise ValueError(f"Unknown amigo type: {self.amigo_type}")
        
        self.actual_model_type = self.model_config["type"]
        self.model_name = self.model_config["name"]
        self.model_path = model_path
        self.model = None
        self.is_recording = False
        self.recording_data = []
        self.sample_rate = 16000
        self.hotkey = keyboard.Key.alt_r  # Right Option key as toggle
        
        print(f"Initializing Dos Amigos with {self.amigo_type.upper()} model...")
        self.load_model()
        print(f"✓ {self.amigo_type.upper()} amigo loaded successfully!")
        print(f"✓ Toggle key: Right Option")
        print("✓ Ready! Press Right Option to start/stop recording.")
    
    def find_local_model(self, amigo_type):
        """Find the local model directory for the specified amigo"""
        model_config = get_model_config(amigo_type)
        if not model_config:
            return None
    
        script_dir = Path(__file__).parent
        model_dirs = [
            script_dir / "models",
            script_dir.parent / "models", 
            Path.cwd() / "models"
        ]
    
        
        # Look for specific model type
        model_patterns = {
                "ligero": ["whisper-small-mlx"],
                "preciso": ["parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v2"],
            }

        for model_dir in model_dirs:
            if model_dir.exists():
                # Look for the specific model type
                for pattern in model_patterns.get(amigo_type, []):
                    matches = list(model_dir.glob(pattern))
                    for subdir in model_dir.glob(pattern):
                        if subdir.is_dir():
                            model_files = (
                                list(subdir.glob("*.safetensors")) + 
                                list(subdir.glob("*.mlx")) + 
                                list(subdir.glob("*.bin")) +
                                list(subdir.glob("*.npz")) +
                                list(subdir.rglob("*.safetensors")) +
                                list(subdir.rglob("*.mlx")) +
                                list(subdir.rglob("*.npz"))
                            )
                            # Check for model files
                            if model_files:
                                print(f"Found local {amigo_type} model at: {subdir}")
                                return str(subdir)
                            else:
                                print(f"DEBUG: No model files found in {subdir}")
        
        return None
    
    def load_model(self):
        try:
            if self.actual_model_type == "parakeet":
                self.load_parakeet_model()
            elif self.actual_model_type == "whisper":
                self.load_whisper_model()
            else:
                raise ValueError(f"Unsupported model type: {self.actual_model_type}")
        except Exception as e:
            print(f"Error loading {self.amigo_type} amigo: {e}")
            print("Make sure you've run the setup script and have the model files.")

        # Suggest fallback
            if self.amigo_type != "ligero":
                print("Try using --model ligero for the lightweight amigo.")
            sys.exit(1)
    
    def load_parakeet_model(self):
        """Load Parakeet MLX model"""
        try:
            from parakeet_mlx import from_pretrained
        except ImportError:
            raise ImportError("parakeet-mlx not installed. Install with: uv pip install parakeet-mlx")
        
        if self.model_path:
            model_path = self.model_path
        else:
            model_path = self.find_local_model(self.amigo_type)
        
        if model_path and Path(model_path).exists():
            self.model = from_pretrained(model_path)
        else:
            print("Local Preciso model not found, attempting online download...")
            self.model = from_pretrained(self.model_config["repo"])

    def load_whisper_model(self):
        """Load Whisper MLX model"""
        try:
            import mlx_whisper
        except ImportError:
            raise ImportError("mlx-whisper not installed. Install with: uv pip install mlx-whisper")
        
        if self.model_path:
            self.model_path_or_repo = self.model_path
        else:
            local_path = self.find_local_model(self.amigo_type)
            if local_path and Path(local_path).exists():
                self.model_path_or_repo = local_path
            else:
                # Use the configured online repository
                print(f"Local {self.amigo_type.title()} model not found, using {self.model_config['repo']}...")
                self.model_path_or_repo = self.model_config["repo"]
        
        # For MLX Whisper, we don't pre-load the model, we pass the path to transcribe()
        print(f"{self.amigo_type.title()} model configured: {self.model_path_or_repo}")
        self.model = self.model_path_or_repo  # Store the path/repo for transcribe()


    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_recording:
            self.recording_data.append(indata.copy())
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return
        
        print("🎤 Recording... Press Right Option again to stop.")
        self.is_recording = True
        self.recording_data = []
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback,
            blocksize=1024
        )
        self.stream.start()
    
    def stop_recording(self):
        """Stop recording and transcribe"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        if not self.recording_data:
            print("No audio recorded.")
            return
        
        print(f"🛑 Recording stopped. Transcribing with {self.amigo_type.upper()} amigo...")
        
        # Concatenate recorded audio
        audio_data = np.concatenate(self.recording_data, axis=0)
        audio_data = audio_data.flatten()
        
        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Close the file handle so we can write to it
            
        # Convert to 16-bit PCM WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(temp_path, self.sample_rate, audio_int16)
        
        # Debug: Check audio file info
        audio_duration = len(audio_data) / self.sample_rate
        audio_max = np.max(np.abs(audio_data))
        
        try:
            # Transcribe using the selected model
            
            # Debug: Check if the method exists and is callable
            
            # Debug: Check the method directly
            method = getattr(self, 'transcribe_audio')
            
            # Try calling it directly with detailed error handling
            try:
                # Bypass the broken transcribe_audio method
                if self.amigo_type == "ligero":
                    transcription = self.transcribe_with_whisper(temp_path)
                elif self.amigo_type == "preciso":
                    transcription = self.transcribe_with_parakeet(temp_path)
                else:
                    transcription = ""

            except Exception as inner_e:
                import traceback
                traceback.print_exc()
                transcription = None
                
                
            if transcription and transcription.strip():
                print(f"📝 Transcription: {transcription}")
                self.paste_text(transcription)
            else:
                print("❌ No speech detected or transcription failed.")
        
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()  # This will show the full error
        
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Debug: Error cleaning up temp file: {e}")
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using the selected model"""
        try:
            if self.amigo_type == "preciso":
                return self.transcribe_with_parakeet(audio_path)
            elif self.amigo_type == "ligero":
                return self.transcribe_with_whisper(audio_path)
        except Exception as e:
            print(f"Transcription error with {self.amigo_type}: {e}")
            return ""
    
    def transcribe_with_parakeet(self, audio_path):
        """Transcribe using Parakeet MLX"""
        result = self.model.transcribe(audio_path)

        if hasattr(result, 'text'):
            text = result.text.strip()
        elif hasattr(result, 'sentences'):
            text = ' '.join(result.sentences).strip()
        else:
            text = str(result).strip()

        # Remove filler words like 'um' from the transcription
        text = self.remove_filler_words(text)

        return text
    
    def transcribe_with_whisper(self, audio_path):
        """Transcribe using Whisper MLX"""
        
        # Validate audio file exists and has content
        if not os.path.exists(audio_path):
            print(f"Error: Audio file does not exist: {audio_path}")
            return ""
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            print(f"Error: Audio file is empty: {audio_path}")
            return ""
        
        
        try:
            import mlx_whisper
        except ImportError as e:
            print(f"Import error: {e}")
            return ""
        
        try:
            
            # Use the correct MLX Whisper API
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=self.model,  # This is the model path or repo
                language="en",  # English optimized
                fp16=True       # Use half precision for memory efficiency
            )
            
            
            # Handle different possible return types
            if result is None:
                print("Warning: Transcription returned None - likely audio loading failed")
                return ""
            
            if isinstance(result, dict):
                if 'text' in result:
                    text = result['text'].strip()
                    return text
                elif 'segments' in result and result['segments']:
                    # Extract text from segments
                    text_parts = []
                    for segment in result['segments']:
                        if 'text' in segment:
                            text_parts.append(segment['text'])
                    combined_text = ' '.join(text_parts).strip()
                    return combined_text
                else:
                    print(f"Warning: Unexpected result format: {result}")
                    return str(result).strip()
            else:
                # Fallback - result might be a string or other type
                fallback_text = str(result).strip()
                return fallback_text
                
        except Exception as e:
            print(f"MLX Whisper transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def remove_filler_words(self, text):
        """Remove filler words like 'um' from transcription"""
        import re

        # Remove 'um' as a standalone word (case-insensitive)
        # This pattern matches 'um' with word boundaries to avoid removing 'um' from words like 'umbrella'
        text = re.sub(r'\b[Uu]m\b', '', text)

        # Clean up multiple spaces left after removal
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        return text.strip()

    def paste_text(self, text):
        """Paste text to the current application using clipboard"""
        try:
            # Copy to clipboard
            process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-8'))
            
            # Simulate Cmd+V to paste
            subprocess.run([
                'osascript', '-e',
                'tell application "System Events" to keystroke "v" using command down'
            ])
            
            print("✅ Text pasted to active application!")
            
        except Exception as e:
            print(f"❌ Failed to paste text: {e}")
            print(f"📋 Copied to clipboard: {text}")
    
    def on_hotkey_press(self, key):
        """Handle hotkey press events"""
        if key == self.hotkey:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print(f"🎙️  Dos Amigos RUNNING ({self.amigo_type.upper()} AMIGO)")
        print("="*60)
        print("Toggle key: Right Option")
        print("Press Ctrl+C to quit")
        print("="*60 + "\n")
        
        # Set up keyboard listener
        with keyboard.Listener(on_press=self.on_hotkey_press) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                if self.is_recording:
                    self.stop_recording()
                sys.exit(0)

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Dos Amigos Offline - Dual model ASR with auto paste.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The Two Amigos:
  🪽 Ligero: whisper-small (~500MB)
  🎯 Preciso: parakeet-tdt-0.6Bb (~2GB)

Examples:
  python dos_amigos.py                    # Use El Ligero (default)
  python dos_amigos.py --model ligero     # Use El Ligero explicitly  
  python dos_amigos.py --model preciso    # Use El Preciso
  python dos_amigos.py --path /custom/model/path  # Use custom model path
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['ligero', 'preciso'],
        default='ligero',
        help='Amigo to use for transcription (default: ligero)'
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        help='Custom path to model directory'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available local models and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("🔍 Scanning for local models...")
        script_dir = Path(__file__).parent
        models_dir = script_dir / "models"
        
        if models_dir.exists():
            print(f"\nModels directory: {models_dir}")
            
            # Look for Parakeet models
            parakeet_models = []
            for path in models_dir.rglob("*parakeet*"):
                if path.is_dir() and any(path.glob("*.safetensors")):
                    parakeet_models.append(path)
            
            # Look for Whisper models  
            whisper_models = []
            for pattern in ["*whisper*", "*distil-whisper*"]:
                for path in models_dir.glob(pattern):
                    if path.is_dir():
                        model_files = (
                            list(path.glob("*.safetensors")) + 
                            list(path.glob("*.mlx")) + 
                            list(path.glob("*.bin")) +
                            list(path.glob("*.npz")) +
                            list(path.rglob("*.safetensors")) +
                            list(path.rglob("*.mlx")) +
                            list(path.rglob("*.npz"))
                        )
                        if model_files:
                            whisper_models.append(path)
            
            # Remove duplicates
            whisper_models = list(set(whisper_models))
            
            print(f"\n📦 Parakeet models found: {len(parakeet_models)}")
            for model in parakeet_models:
                size_mb = sum(f.stat().st_size for f in model.rglob("*") if f.is_file()) / (1024*1024)
                print(f"  {model.name} ({size_mb:.1f} MB)")
            
            print(f"\n🎵 Whisper models found: {len(whisper_models)}")
            for model in whisper_models:
                size_mb = sum(f.stat().st_size for f in model.rglob("*") if f.is_file()) / (1024*1024)
                print(f"  {model.name} ({size_mb:.1f} MB)")
                
            if not parakeet_models and not whisper_models:
                print("\n❌ No local models found. Run the setup script to download models.")
        else:
            print("❌ Models directory not found.")
        
        return
    
    try:
        app = DosAmigos(amigo_type=args.model, model_path=args.path)
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()