import os
import csv
import numpy as np
import torch
import torchaudio
from inference_wrapper import InferenceWrapper

# ----- Constants -----
# Audio processing constants
AUDIO_SAMPLE_RATE = 16000
AUDIO_MAX_VALUE = 32768.0  # 16-bit signed max value

# Feature extraction constants
DEFAULT_INPUT_TDIM = 1024
DEFAULT_MEL_BINS = 128
FRAME_SHIFT_MS = 10

# Normalization constants (from model training)
FBANK_MEAN = -4.2677393
FBANK_STD = 4.5689974 * 2

# Target classes for classification
TARGET_CLASSES = ["Baby cry", "Cat", "Rooster", "Cricket", "Dog"]

class MdlModel:
    """
    Model wrapper for AST ONNX model, providing audio preprocessing and inference.
    """
    def __init__(self, model_path, label_csv, input_tdim=DEFAULT_INPUT_TDIM, 
                 mel_bins=DEFAULT_MEL_BINS, sampling_rate=AUDIO_SAMPLE_RATE, debug=False):
        """
        Initialize the model wrapper.
        
        Args:
            model_path (str): Path to ONNX model file.
            label_csv (str): Path to CSV file with class labels.
            input_tdim (int): Number of time frames for input spectrogram.
            mel_bins (int): Number of mel bins for spectrogram.
            sampling_rate (int): Audio sampling rate (Hz).
            debug (bool): Enable debug output.
        """
        # Validate file paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(label_csv):
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        
        # Initialize model components
        self.wrapper = InferenceWrapper(model_path)
        self.input_tdim = input_tdim
        self.mel_bins = mel_bins
        self.sampling_rate = sampling_rate
        self.debug = debug
        self.labels = self._load_labels(label_csv)
        self.target_classes = TARGET_CLASSES.copy()
        
        # Validate target classes and show status
        self._validate_target_classes()

    def _load_labels(self, label_csv):
        """Load class labels from CSV file."""
        labels = []
        name_to_index = {}
        
        with open(label_csv, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                label = row['display_name']
                labels.append(label)
                name_to_index[label] = i
        
        self.name_to_index = name_to_index
        return labels
    
    def _validate_target_classes(self):
        """Validate target classes and show initialization status."""
        missing_classes = []
        found_classes = []
        
        for target_class in self.target_classes:
            class_index = self._find_class_index(target_class)
            if class_index != -1:
                found_classes.append((target_class, self.labels[class_index], class_index))
            else:
                missing_classes.append(target_class)
        
        print(f"🎯 Model loaded: {len(self.labels)} total classes")
        print(f"✅ Found {len(found_classes)}/{len(self.target_classes)} target classes")
        
        if self.debug:
            print("\n[DEBUG] Target class mapping:")
            for target, actual, idx in found_classes:
                print(f"  {target} → '{actual}' (index {idx})")
            if missing_classes:
                print(f"  Missing: {missing_classes}")
            print(f"[DEBUG] Model I/O: {self.wrapper.input_names} → {self.wrapper.output_names}")
        elif missing_classes:
            print(f"⚠️  Missing classes: {missing_classes}")

    def _find_class_index(self, target_class):
        """
        Find class index using exact match first, then partial match.
        Args:
            target_class (str): The class name to find
        Returns:
            int: Class index, or -1 if not found
        """
        # First try exact match (O(1) lookup)
        if target_class in self.name_to_index:
            return self.name_to_index[target_class]
        
        # If exact match fails, try partial match (O(n) fallback)
        target_lower = target_class.lower()
        
        for i, label in enumerate(self.labels):
            label_lower = label.lower()
            if target_lower in label_lower:
                return i
        
        return -1  # Not found

    def extract_features(self, audio):
        """
        Extract log-mel spectrogram features from audio.
        
        Args:
            audio (np.ndarray): 1D float32 array, normalized to [-1, 1]
        Returns:
            np.ndarray: (1, input_tdim, mel_bins) float32 spectrogram
        """
        # Extract mel-scale filter bank features
        fbank = torchaudio.compliance.kaldi.fbank(
            torch.tensor(audio).unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.mel_bins,
            dither=0.0,
            frame_shift=FRAME_SHIFT_MS
        )
        
        # Pad or truncate to required time dimension
        n_frames = fbank.shape[0]
        frame_diff = self.input_tdim - n_frames
        if frame_diff > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, frame_diff))
        elif frame_diff < 0:
            fbank = fbank[:self.input_tdim, :]
        
        # Apply normalization (from training statistics)
        fbank = (fbank - FBANK_MEAN) / FBANK_STD
        
        return fbank.unsqueeze(0).numpy().astype(np.float32)

    def callback(self, audio_bytes):
        """
        Process audio and return classification scores for target classes.
        
        Args:
            audio_bytes (bytes): Raw PCM 16-bit mono audio at 16kHz.
        Returns:
            list: Target class labels and scores as list of tuples.
        """
        try:
            # Convert raw bytes to normalized float32 audio
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            
            if audio_np.size == 0:
                if self.debug:
                    print("[DEBUG] Empty audio buffer!")
                return [(cls, 0.0) for cls in self.target_classes]
            
            # Normalize to [-1, 1] range
            audio_float = audio_np.astype(np.float32) / AUDIO_MAX_VALUE
            
            if self.debug:
                rms = np.sqrt(np.mean(audio_float**2))
                print(f"[DEBUG] Audio: {len(audio_bytes)} bytes, RMS: {rms:.4f}")
            
            # Extract features and run inference
            feats = self.extract_features(audio_float)
            input_name = self.wrapper.input_names[0]
            outputs = self.wrapper.infer({input_name: feats})
            output = list(outputs.values())[0][0]
            
            # Apply sigmoid activation
            probabilities = torch.sigmoid(torch.tensor(output)).numpy()
            
            if self.debug:
                print(f"[DEBUG] Features: {feats.shape}, Output: {output.shape}")
            
            # Extract scores for target classes
            target_results = []
            for target_class in self.target_classes:
                class_index = self._find_class_index(target_class)
                if class_index != -1:
                    score = float(probabilities[class_index])
                    target_results.append((target_class, score))
                    if self.debug:
                        print(f"[DEBUG] {target_class}: {score:.4f}")
                else:
                    target_results.append((target_class, 0.0))
                    if self.debug:
                        print(f"[DEBUG] {target_class}: NOT FOUND")
            
            return target_results
            
        except Exception as e:
            print(f"❌ Model inference error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return [(cls, 0.0) for cls in self.target_classes]

    def get_target_classes(self):
        """Get the list of target class names."""
        return self.target_classes.copy()