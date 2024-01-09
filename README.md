# Human-Face-Crop-ONNX-TensorRT
Simply crop the face from the image at high speed and save.

```bash
# Default:
#   yolox_x_body_head_hand_0102_0.5533_post_1x3x480x640.onnx
#   retinaface_resnet50_with_postprocess_Nx3x96x96_max001_th015.onnx

python demo_face_crop_onnx_tflite.py
```

```
usage: demo_face_crop_onnx_tflite.py \
[-h] \
[-hdm {
    yolox_n_body_head_hand_post_0461_0.4428_1x3x480x640.onnx,
    yolox_t_body_head_hand_post_0299_0.4522_1x3x480x640.onnx,
    yolox_s_body_head_hand_post_0299_0.4983_1x3x480x640.onnx,
    yolox_m_body_head_hand_post_0299_0.5263_1x3x480x640.onnx,
    yolox_l_body_head_hand_0299_0.5420_post_1x3x480x640.onnx,
    yolox_x_body_head_hand_0102_0.5533_post_1x3x480x640.onnx
  }
] \
[-fdm {
    retinaface_mbn025_with_postprocess_Nx3x64x64_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x96x96_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x128x128_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x160x160_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x192x192_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x224x224_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x256x256_max001_th0.15.onnx,
    retinaface_resnet50_with_postprocess_Nx3x64x64_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x96x96_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x128x128_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x160x160_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x192x192_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x224x224_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x256x256_max001_th015.onnx
  }
] \
[-v VIDEO] \
[-ep {cpu,cuda,tensorrt}] \
[-dvw]

options:
  -h, --help
    show this help message and exit
  -hdm {...}, --head_detection_model {...}
    ONNX/TFLite file path for YOLOX.
  -fdm {...}, --face_detection_model {...}
    ONNX/TFLite file path for RetinaFace.
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated
    with automatic recording to MP4. Devices that use a MicroSD card or
    similar for main storage can speed up overall processing.
```

## INPUT

![12_Group_Group_12_Group_Group_12_10](https://github.com/PINTO0309/Human-Face-Crop-ONNX-TensorRT/assets/33194443/4a3d8ec8-5f7d-4358-b7e9-c7835099dbdc)

## OUTPUT

![test_screenshot_08 01 2024](https://github.com/PINTO0309/Human-Face-Crop-ONNX-TensorRT/assets/33194443/1510685f-ee1c-4240-a56d-01951e7ac83c)

![image](https://github.com/PINTO0309/Human-Face-Crop-ONNX-TensorRT/assets/33194443/12a35a59-b5ce-4030-b5b2-7c720cc0dd39)

![test_screenshot_09 01 2024](https://github.com/PINTO0309/Human-Face-Crop-ONNX-TensorRT/assets/33194443/0986892c-9c54-4016-bea8-4d7ae7defe7c)

![test_screenshot_09 01 20242](https://github.com/PINTO0309/Human-Face-Crop-ONNX-TensorRT/assets/33194443/e0ce0752-e38e-45e4-a6fc-892a4d60962f)

![test_screenshot_09 01 20243](https://github.com/PINTO0309/Human-Face-Crop-ONNX-TensorRT/assets/33194443/359ffe38-7ee4-4d3d-b4b1-938bb414b590)
