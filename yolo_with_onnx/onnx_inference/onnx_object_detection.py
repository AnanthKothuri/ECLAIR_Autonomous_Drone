from typing import Tuple, Union

import numpy as np
import onnxruntime as ort
from typing import List

from onnx_inference.onnx_base import OnnxBase


class OnnxObjectDetection(OnnxBase):
    """ONNX Base class."""
    weight_path: str = None
    session: ort.InferenceSession = None

    input_size: Tuple[int, int] = None
    input_name: str = None
    output_name: str = None

    def __init__(self, weight_path: str, classnames: List[str] = None, cuda: bool = False) -> None:
        """Initialize class.

        Args:
            weight_path: Location of the weight file
        """
        super().__init__(weight_path=weight_path, cuda=cuda)
        self.weight_path: str = weight_path
        self.classnames = classnames
        self.cuda = cuda

    def predict_object_detection(self, input_data: np.ndarray) -> Union[np.ndarray, None]:
        """OCR model predict code, independent of weight format.

        Args:
            input_data: Input data

        Returns:
            Resulting predictions
        """
        return self.predict(input_data)[0]
