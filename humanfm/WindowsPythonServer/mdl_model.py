import numpy as np
import torch
import torchaudio
from inference_wrapper import InferenceWrapper
import os
import csv

class MdlModel:
    """
    Model wrapper for AST ONNX model, providing audio preprocessing and inference.
    """
    def __init__(self, model_path, label_csv, input_tdim=1024, mel_bins=128, sampling_rate=16000):
        """
        Args:
            model_path (str): Path to ONNX model file.
            label_csv (str): Path to CSV file with class labels.
            input_tdim (int): Number of time frames for input spectrogram.
            mel_bins (int): Number of mel bins for spectrogram.
            sampling_rate (int): Audio sampling rate (Hz).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(label_csv):
            raise FileNotFoundError(f"Label CSV not found: {label_csv}")
        self.wrapper = InferenceWrapper(model_path)
        self.input_tdim = input_tdim
        self.mel_bins = mel_bins
        self.sampling_rate = sampling_rate
        self.labels = self._load_labels(label_csv)
        
        # Define the 5 target classes we care about (must match Node.js server)
        self.target_classes = ["Baby cry", "Cat", "Rooster", "Cricket", "Dog"]
        
        # Debug: Check which target classes are found in the loaded labels
        print(f"[DEBUG] Loaded {len(self.labels)} labels from CSV")
        print(f"[DEBUG] Labels around index 99: {self.labels[97:102] if len(self.labels) > 102 else 'Not enough labels'}")
        print("[DEBUG] Target class availability:")
        for target_class in self.target_classes:
            class_index = self._find_class_index(target_class)
            if class_index != -1:
                actual_label = self.labels[class_index]
                print(f"  {target_class} → '{actual_label}' (index {class_index})")
            else:
                print(f"  {target_class}: NOT FOUND!")
        print(f"[DEBUG] First 10 labels: {self.labels[:10]}")
        print(f"[DEBUG] Model input names: {self.wrapper.input_names}")
        print(f"[DEBUG] Model output names: {self.wrapper.output_names}")

    def _load_labels(self, label_csv):
        labels = []
        name_to_index = {}  # For efficient lookup
        
        with open(label_csv, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                label = row['display_name']  # This properly handles quoted fields
                labels.append(label)
                name_to_index[label] = i  # Create reverse lookup
        
        # Store both for different use cases
        self.name_to_index = name_to_index
        return labels

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
            np.ndarray: (1, input_tdim, mel_bins) float32
        """
        fbank = torchaudio.compliance.kaldi.fbank(
            torch.tensor(audio).unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.mel_bins,
            dither=0.0,
            frame_shift=10
        )
        n_frames = fbank.shape[0]
        p = self.input_tdim - n_frames
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        elif p < 0:
            fbank = fbank[:self.input_tdim, :]
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank.unsqueeze(0).numpy().astype(np.float32)

    def callback(self, audio_bytes):
        """
        Frame processing function for server use.
        Args:
            audio_bytes (bytes): Raw PCM 16-bit mono audio at 16kHz.
        Returns:
            list: Target 5 class labels and scores as list of tuples, e.g. [('Speech', 0.87), ('Music', 0.65), ...]
        """
        try:
            # Convert raw bytes to 16-bit PCM, then to float32 [-1, 1]
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            print(f"[DEBUG] Audio buffer size: {len(audio_bytes)} bytes, samples: {audio_np.size}")
            
            if audio_np.size == 0:
                print("[DEBUG] Empty audio buffer!")
                return [(cls, 0.0) for cls in self.target_classes]
            
            audio_float = audio_np.astype(np.float32) / 32768.0
            print(f"[DEBUG] Audio range: [{audio_float.min():.4f}, {audio_float.max():.4f}], RMS: {np.sqrt(np.mean(audio_float**2)):.4f}")
            
            feats = self.extract_features(audio_float)
            print(f"[DEBUG] Feature shape: {feats.shape}, range: [{feats.min():.4f}, {feats.max():.4f}]")
            
            input_name = self.wrapper.input_names[0]
            print(f"[DEBUG] Model input name: {input_name}")
            
            outputs = self.wrapper.infer({input_name: feats})
            output = list(outputs.values())[0][0]  # shape: (n_classes,)
            print(f"[DEBUG] Raw model output shape: {output.shape}, range: [{output.min():.4f}, {output.max():.4f}]")
            
            output = torch.sigmoid(torch.tensor(output)).numpy()
            print(f"[DEBUG] After sigmoid: range: [{output.min():.4f}, {output.max():.4f}]")
            
            # Get scores for the 5 target classes only
            target_results = []
            for target_class in self.target_classes:
                class_index = self._find_class_index(target_class)
                if class_index != -1:
                    score = output[class_index]
                    actual_label = self.labels[class_index]
                    print(f"[DEBUG] {target_class} → '{actual_label}' (index {class_index}): {score:.4f}")
                    target_results.append((target_class, score))
                else:
                    # If target class not found in model labels, return 0 score
                    print(f"[DEBUG] {target_class}: NOT FOUND in labels!")
                    target_results.append((target_class, 0.0))
            
            return target_results
        except Exception as e:
            print(f"[DEBUG] Exception in callback: {e}")
            import traceback
            traceback.print_exc()
            return [(cls, 0.0) for cls in self.target_classes]

    def get_target_classes(self):
        """
        Get the list of target class names.
        Returns:
            list: List of target class names
        """
        return self.target_classes.copy()

# Example usage (uncomment to test):
# model = MdlModel("AST_realTime_main/pretrained_models/audio_mdl.onnx", "AST_realTime_main/egs/audioset/data/class_labels_indices.csv")
# result = model.callback(audio_bytes)
# print(result) 