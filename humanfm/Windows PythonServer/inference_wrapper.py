import onnxruntime as ort
import numpy as np

class InferenceWrapper:
    """
    General-purpose ONNX inference wrapper for audio and other models.
    """
    def __init__(self, model_path):
        """
        Initialize the ONNX Runtime session and gather input/output info.
        """
        self.session = ort.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}
        self.output_shapes = {out.name: out.shape for out in self.session.get_outputs()}

    def infer(self, input_dict):
        """
        Run inference on the model.
        Args:
            input_dict (dict): {input_name: np.ndarray}
        Returns:
            dict: {output_name: np.ndarray}
        """
        ort_inputs = {name: np.asarray(arr, dtype=np.float32) for name, arr in input_dict.items()}
        outputs = self.session.run(self.output_names, ort_inputs)
        return {name: out for name, out in zip(self.output_names, outputs)}

    def get_input_shape(self, input_name):
        return self.input_shapes[input_name]

    def get_output_shape(self, output_name):
        return self.output_shapes[output_name] 