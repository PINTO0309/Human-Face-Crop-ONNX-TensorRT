#!/usr/bin/env python

"""
runtime: https://github.com/microsoft/onnxruntime

pip install onnxruntime or pip install onnxruntime-gpu
pip install opencv-contrib-python==4.9.0.80
"""
from __future__ import annotations
import os
import re
import sys
import copy
import cv2
import requests
import subprocess
import numpy as np
from enum import Enum
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod

from glob import glob

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    looked_score: float
    x1: int
    y1: int
    x2: int
    y2: int
    landmarks: np.ndarray

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    _mean: np.ndarray = np.array([0.000, 0.000, 0.000], dtype=np.float32)
    _std: np.ndarray = np.array([1.000, 1.000, 1.000], dtype=np.float32)

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap: Tuple = (2, 0, 1)
    _h_index: int = 2
    _w_index: int = 3
    _norm_shape: List = [1,3,1,1]
    _class_score_th: float

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
        mean: Optional[np.ndarray] = np.array([0.000, 0.000, 0.000], dtype=np.float32),
        std: Optional[np.ndarray] = np.array([1.000, 1.000, 1.000], dtype=np.float32),
        class_score_th: float = 0.35,
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]
            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3
            self._norm_shape = [1,3,1,1]

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shapes = [
                input.get('shape', None) for input in self._input_details
            ]
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2
            self._norm_shape = [1,1,1,3]

        self._mean = mean.reshape(self._norm_shape)
        self._std = std.reshape(self._norm_shape)
        self._class_score_th = class_score_th

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

class YOLOX(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolox_n_body_head_hand_post_0461_0.4428_1x3x480x640.onnx',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """YOLOX

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """YOLOX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, x1, y1, x2, y2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]

        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self._input_shapes[0][self._w_index]),
                int(self._input_shapes[0][self._h_index]),
            )
        )
        resized_image = resized_image.transpose(self._swap)
        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]
        result_boxes: List[Box] = []
        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]
            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    result_boxes.append(
                        Box(
                            classid=int(box[1]),
                            score=float(score),
                            looked_score=0.0,
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            landmarks=None,
                        )
                    )
        return result_boxes

class RetinaFace(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'retinaface_mbn025_with_postprocess_480x640_max20_th0.70.onnx',
        class_score_th: Optional[float] = 0.15,
        providers: Optional[List] = None,
    ):
        """RetinaFace

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            mean=np.asarray([104, 117, 123], dtype=np.float32),
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
        boxes: List[Box],
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        boxes: List[Box]
            Head boxes

        Returns
        -------
        batchno_classid_score_x1y1x2y2_landms: np.ndarray
            [N, [batchno, classid, score, x1, y1, x2, y2, landms0, ..., landms9]]
        """
        temp_image = copy.deepcopy(image)
        temp_boxes = copy.deepcopy(boxes)
        # PreProcess
        inferece_images = \
            self._preprocess(
                image=temp_image,
                boxes=temp_boxes,
            )
        # Inference
        outputs = super().__call__(input_datas=[inferece_images])
        batchno_classid_score_x1y1x2y2_landms = outputs[0]
        # PostProcess
        face_boxes = \
            self._postprocess(
                face_boxes=batchno_classid_score_x1y1x2y2_landms,
                head_boxes=temp_boxes,
            )
        return face_boxes

    def _preprocess(
        self,
        image: np.ndarray,
        boxes: List[Box],
        swap: Optional[Tuple[int,int,int,int]] = (0, 3, 1, 2),
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        cropped_boxes = [image[box.y1:box.y2, box.x1:box.x2, :] for box in boxes]
        # Normalization + BGR->RGB
        resized_image_list: List[np.ndarray] = []
        for cropped_box in cropped_boxes:
            h, w, c = cropped_box.shape
            if h > 0 and w > 0:
                resized_image = cv2.resize(
                    cropped_box,
                    (
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
                resized_image = resized_image[..., ::-1] # BGR->RGB
                resized_image_list.append(resized_image)
        resized_images = np.asarray(resized_image_list, dtype=self._input_dtypes[0])
        resized_images = resized_images.transpose(swap)
        resized_images = (resized_images - self._mean)
        return resized_images

    def _postprocess(
        self,
        face_boxes: np.ndarray,
        head_boxes: List[Box],
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        result_boxes: List[Box] = []
        if len(face_boxes) > 0:
            scores = face_boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = face_boxes[keep_idxs, :]
            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    batchno = int(box[0])
                    head_w = abs(head_boxes[batchno].x2 - head_boxes[batchno].x1)
                    head_h = abs(head_boxes[batchno].y2 - head_boxes[batchno].y1)
                    x_min = int(max(0, box[3]) * head_w / self._input_shapes[0][self._w_index]) + head_boxes[batchno].x1
                    y_min = int(max(0, box[4]) * head_h / self._input_shapes[0][self._h_index]) + head_boxes[batchno].y1
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * head_w / self._input_shapes[0][self._w_index]) + head_boxes[batchno].x1
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * head_h / self._input_shapes[0][self._h_index]) + head_boxes[batchno].y1
                    landmarks: np.ndarray = box[7:]
                    landmarks = landmarks.reshape(-1, 2).astype(np.int32)
                    landmarks[:, 0] = landmarks[:, 0] * head_w / self._input_shapes[0][self._w_index] + head_boxes[batchno].x1
                    landmarks[:, 1] = landmarks[:, 1] * head_h / self._input_shapes[0][self._h_index] + head_boxes[batchno].y1
                    result_boxes.append(
                        Box(
                            classid=int(box[1]),
                            score=float(score),
                            looked_score=0.0,
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            landmarks=landmarks,
                        )
                    )
        return result_boxes

class DBFace(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'dbface_Nx3x192x192_post_max100_th030.onnx',
        class_score_th: Optional[float] = 0.15,
        providers: Optional[List] = None,
    ):
        """DBFace

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for DBFace. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for DBFace

        class_score_th: Optional[float]
            Score threshold. Default: 0.15

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            mean=np.asarray([0.408, 0.447, 0.470], dtype=np.float32),
            std=np.asarray([0.289, 0.274, 0.278], dtype=np.float32),
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
        boxes: List[Box],
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        boxes: List[Box]
            Head boxes

        Returns
        -------
        batchno_classid_score_x1y1x2y2_landms: np.ndarray
            [N, [batchno, classid, score, x1, y1, x2, y2, landms0, ..., landms9]]
        """
        temp_image = copy.deepcopy(image)
        temp_boxes = copy.deepcopy(boxes)
        # PreProcess
        inferece_images = \
            self._preprocess(
                image=temp_image,
                boxes=temp_boxes,
            )
        # Inference
        outputs = super().__call__(input_datas=[inferece_images])
        batchno_classid_score_x1y1x2y2_landms = outputs[0]
        # PostProcess
        face_boxes = \
            self._postprocess(
                face_boxes=batchno_classid_score_x1y1x2y2_landms,
                head_boxes=temp_boxes,
            )
        return face_boxes

    def _preprocess(
        self,
        image: np.ndarray,
        boxes: List[Box],
        swap: Optional[Tuple[int,int,int,int]] = (0, 3, 1, 2),
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        cropped_boxes = [image[box.y1:box.y2, box.x1:box.x2, :] for box in boxes]
        # Normalization + BGR->RGB
        resized_image_list: List[np.ndarray] = []
        for cropped_box in cropped_boxes:
            h, w, c = cropped_box.shape
            if h > 0 and w > 0:
                resized_image = cv2.resize(
                    cropped_box,
                    (
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
                # resized_image = resized_image[..., ::-1] # BGR->RGB
                resized_image_list.append(resized_image)
        resized_images = np.asarray(resized_image_list, dtype=self._input_dtypes[0])
        resized_images = resized_images.transpose(swap)
        resized_images = (resized_images - self._mean)
        return resized_images

    def _postprocess(
        self,
        face_boxes: np.ndarray,
        head_boxes: List[Box],
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        result_boxes: List[Box] = []
        if len(face_boxes) > 0:
            scores = face_boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = face_boxes[keep_idxs, :]
            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    batchno = int(box[0])
                    head_w = abs(head_boxes[batchno].x2 - head_boxes[batchno].x1)
                    head_h = abs(head_boxes[batchno].y2 - head_boxes[batchno].y1)
                    x_min = int(max(0, box[3]) * head_w / self._input_shapes[0][self._w_index]) + head_boxes[batchno].x1
                    y_min = int(max(0, box[4]) * head_h / self._input_shapes[0][self._h_index]) + head_boxes[batchno].y1
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * head_w / self._input_shapes[0][self._w_index]) + head_boxes[batchno].x1
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * head_h / self._input_shapes[0][self._h_index]) + head_boxes[batchno].y1
                    landmarks: np.ndarray = box[7:]
                    landmarks = landmarks.reshape(-1, 2).astype(np.int32)
                    landmarks[:, 0] = landmarks[:, 0] * head_w / self._input_shapes[0][self._w_index] + head_boxes[batchno].x1
                    landmarks[:, 1] = landmarks[:, 1] * head_h / self._input_shapes[0][self._h_index] + head_boxes[batchno].y1
                    result_boxes.append(
                        Box(
                            classid=int(box[1]),
                            score=float(score),
                            looked_score=0.0,
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            landmarks=landmarks,
                        )
                    )
        return result_boxes

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def download_file(url, folder, filename):
    """
    Download a file from a URL and save it to a specified folder.
    If the folder does not exist, it is created.

    :param url: URL of the file to download.
    :param folder: Folder where the file will be saved.
    :param filename: Filename to save the file.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Full path for the file
    file_path = os.path.join(folder, filename)
    # Download the file
    print(f"{Color.GREEN('Downloading...')} {url} to {file_path}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"{Color.GREEN('Download completed:')} {file_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def get_nvidia_gpu_model() -> List[str]:
    try:
        # Run nvidia-smi command
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)

        # Extract GPU model numbers using regular expressions
        models = re.findall(r'GPU \d+: (.*?)(?= \(UUID)', output)
        return models
    except Exception as e:
        print(f"Error: {e}")
        return []

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)

def calculate_reduced_area(
    *,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    scale: float=0.80,
) -> Tuple[int, int, int, int]:
    """
    Calculate the coordinates of a reduced area centered around the same center point
    as the original area, but with a scale factor (default is 80%).

    Parameters:
    y1, y2, x1, x2 (int): Coordinates of the original area.
    scale (float): Scale factor for the size reduction (default is 0.8 for 80%).

    Returns:
    tuple: Coordinates of the reduced area (y1', y2', x1', x2').
    """
    # Calculate the center of the original area
    center_y = (y1 + y2) / 2
    center_x = (x1 + x2) / 2
    # Calculate the height and width of the original area
    height = y2 - y1
    width = x2 - x1
    # Calculate the height and width of the new, reduced area
    new_height = height * scale
    new_width = width * scale
    # Calculate the coordinates of the new area
    y1_prime = int(center_y - new_height / 2 * 0.7)
    y2_prime = int(center_y + new_height / 2)
    x1_prime = int(center_x - new_width / 2 * 0.8)
    x2_prime = int(center_x + new_width / 2 * 0.8)
    return y1_prime, y2_prime, x1_prime, x2_prime

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-hdm',
        '--head_detection_model',
        type=str,
        default='yolox_m_body_head_hand_post_0299_0.5263_1x3x480x640.onnx',
        choices=[
            'yolox_n_body_head_hand_post_0461_0.4428_1x3x480x640.onnx',
            'yolox_t_body_head_hand_post_0299_0.4522_1x3x480x640.onnx',
            'yolox_s_body_head_hand_post_0299_0.4983_1x3x480x640.onnx',
            'yolox_m_body_head_hand_post_0299_0.5263_1x3x480x640.onnx',
            'yolox_l_body_head_hand_0299_0.5420_post_1x3x480x640.onnx',
            'yolox_x_body_head_hand_0102_0.5533_post_1x3x480x640.onnx',
        ],
        help='ONNX/TFLite file path for YOLOX.',
    )
    parser.add_argument(
        '-fdm',
        '--face_detection_model',
        type=str,
        default='retinaface_mbn025_with_postprocess_Nx3x192x192_max001_th0.15.onnx',
        choices=[
            'retinaface_mbn025_with_postprocess_Nx3x64x64_max001_th0.15.onnx',
            'retinaface_mbn025_with_postprocess_Nx3x96x96_max001_th0.15.onnx',
            'retinaface_mbn025_with_postprocess_Nx3x128x128_max001_th0.15.onnx',
            'retinaface_mbn025_with_postprocess_Nx3x160x160_max001_th0.15.onnx',
            'retinaface_mbn025_with_postprocess_Nx3x192x192_max001_th0.15.onnx',
            'retinaface_mbn025_with_postprocess_Nx3x224x224_max001_th0.15.onnx',
            'retinaface_mbn025_with_postprocess_Nx3x256x256_max001_th0.15.onnx',
        ],
        help='ONNX/TFLite file path for RetinaFace.',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
        help='Video file path or camera index.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='cuda',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    args = parser.parse_args()

    # runtime check
    head_detection_model_file: str = args.head_detection_model
    head_detection_model_ext: str = os.path.splitext(head_detection_model_file)[1][1:].lower()
    face_detection_model_file: str = args.face_detection_model
    face_detection_model_ext: str = os.path.splitext(face_detection_model_file)[1][1:].lower()

    if head_detection_model_ext != face_detection_model_ext:
        print(Color.RED('ERROR: head_detection_model and face_detection_model must be files with the same extension.'))
        sys.exit(0)
    runtime: str = None
    if head_detection_model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif head_detection_model_ext == 'tflite':
        if is_package_installed('tflite_runtime'):
            runtime = 'tflite_runtime'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: tflite_runtime or tensorflow is not installed.'))
            print(Color.RED('ERROR: https://github.com/PINTO0309/TensorflowLite-bin'))
            print(Color.RED('ERROR: https://github.com/tensorflow/tensorflow'))
            sys.exit(0)

    WEIGHT_FOLDER_PATH = '.'

    # Download head detection onnx
    weight_file = os.path.basename(head_detection_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/NITEC-ONNX-TensorRT/releases/download/onnx/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)

    # Download face detection onnx
    weight_file = os.path.basename(face_detection_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/NITEC-ONNX-TensorRT/releases/download/onnx/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)

    execution_provider: str = args.execution_provider
    providers: List[Tuple[str, Dict] | str] = None
    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

    # Model initialization
    head_detection_model = \
        YOLOX(
            runtime=runtime,
            model_path=head_detection_model_file,
            providers=providers,
        )

    if 'retinaface' in face_detection_model_file:
        face_detection_model = \
            RetinaFace(
                runtime=runtime,
                model_path=face_detection_model_file,
                class_score_th=0.85,
                providers=providers,
            )
    elif 'dbface' in face_detection_model_file:
        face_detection_model = \
            DBFace(
                runtime=runtime,
                model_path=face_detection_model_file,
                class_score_th=0.35,
                providers=[
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider',
                ],
            )

    os.makedirs('outputs', exist_ok=True)

    for image_file in sorted(glob(f'inputs/*.jpg')):
        image = cv2.imread(image_file)
        base_file_name_wo_ext = os.path.splitext(os.path.basename(image_file))[0]
        ext = os.path.splitext(os.path.basename(image_file))[1]

        debug_image = copy.deepcopy(image)
        # debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        boxes = head_detection_model(image=debug_image)
        head_boxes: List[Box] = [box for box in boxes if box.classid == 1]
        other_boxes: List[Box] = [box for box in boxes if box.classid != 1]
        face_boxes: List[Box] = []
        if len(head_boxes) > 0:
            face_boxes = face_detection_model(image=debug_image, boxes=head_boxes)

        for idx, face_box in enumerate(face_boxes):
            output_file_path = f'outputs/{base_file_name_wo_ext}_{idx:04d}{ext}'
            try:
                cv2.imwrite(filename=output_file_path, img=debug_image[face_box.y1:face_box.y2, face_box.x1:face_box.x2, :])
            except:
                pass
            if os.path.isfile(output_file_path) and os.path.getsize(output_file_path) == 0:
                os.remove(output_file_path)

        for box in face_boxes:
            color = (255,255,255)
            if box.classid == 0:
                color = (255,0,0)
            elif box.classid == 1:
                color = (0,0,255)
            elif box.classid == 2:
                color = (0,255,0)
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)
            for landmark in box.landmarks:
                cv2.circle(debug_image, (landmark[0], landmark[1]), 2, (0,255,0), -1)

        for box in head_boxes:
            color = (255,255,255)
            if box.classid == 0:
                color = (255,0,0)
            elif box.classid == 1:
                color = (0,0,255)
            elif box.classid == 2:
                color = (0,255,0)
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
            cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)

        # for box in other_boxes:
        #     color = (255,255,255)
        #     if box.classid == 0:
        #         color = (0,233,245)
        #     elif box.classid == 1:
        #         color = (0,0,255)
        #     elif box.classid == 2:
        #         color = (0,255,0)
        #     draw_dashed_rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2, 5)
        #     draw_dashed_rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1, 5)

        cv2.imshow("test", debug_image)
        key = cv2.waitKey(0)
        if key == 27: # ESC
            break





if __name__ == "__main__":
    main()
