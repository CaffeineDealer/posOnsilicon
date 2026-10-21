# posOnSilicon

Live human pose estimation on the iPhone, comparing three approaches side by side on the same camera feed:

1. **Apple Vision** 3D body pose (`VNDetectHumanBodyPose3DRequest`), built into iOS.
2. **ViTPose-H** (vision transformer, 17 COCO keypoints), converted to Core ML.
3. **SAM 3D Body** (Meta; image encoder + body decoder, 70 keypoints), converted to Core ML.

Everything runs on the phone. The goal is to compare speed and accuracy of transformer-based pose models with Apple's built-in pose, for tracking body movement from a camera facing a person (for example a driver or a sim racer).

## The app: PoseCam

SwiftUI app, iPhone only, in `PoseCam/`.

1. Camera at 1920 x 1080, switchable between 60, 120 and 240 fps with the fps button.
2. One on/off button per model (Apple, ViTPose, SAM3D); all start off. A model is loaded only when switched on and released when switched off.
3. Overlays: Apple's joints and skeleton in green, ViTPose keypoints in orange, SAM 3D Body keypoints in magenta, and a side view (depth vs height) of Apple's 3D joints in cyan.
4. Stats box: camera and processing fps, time per model, the difference between Apple and ViTPose on 12 shared joints, app and system CPU and memory, thermal state, and each model's status. For SAM 3D Body: the delay per result, results per second, encoder and decoder times, and a few diagnostics.
5. A small debug view shows the 512 x 512 crop SAM 3D Body actually sees, with its keypoints drawn on it.

Source files (`PoseCam/PoseCam/`):

| File | What it does |
|---|---|
| `CameraManager.swift` | camera capture and frame rate, runs the models per frame, publishes keypoints, stats and status |
| `ContentView.swift` | the screen: camera preview, overlays, stats, buttons |
| `ViTPoseRunner.swift` | runs the ViTPose Core ML model |
| `SAM3DPreprocess.swift` | SAM 3D Body pre-processing: crop, camera values, ray map, the decoder's fixed inputs |
| `SAM3DRunner.swift` | runs the SAM 3D Body encoder |
| `SAM3DDecoderRunner.swift` | runs the SAM 3D Body decoder, returns 70 keypoints in frame pixels |
| `SystemStats.swift` | app and system CPU and memory |
| `CameraPreview.swift`, `MyApp.swift` | preview layer and app entry |

## Models (not in this repo)

The Core ML models are large and git-ignored. Put them in `PoseCam/PoseCam/` and add them to the PoseCam target in Xcode:

| File | Model | How to get it |
|---|---|---|
| `vitpose_base.mlpackage` | ViTPose-plus-huge (`usyd-community/vitpose-plus-huge` on Hugging Face), 256 x 192 image in, 17 keypoint heatmaps out | `convert_vitpose.py` in this repo |
| `sam3d_backbone.mlpackage` | SAM 3D Body image encoder (DINOv3), 512 x 512 image in, `[1, 1280, 32, 32]` features out (about 1.7 GB) | the companion repo `sam3d-coreml` |
| `sam3d_decoder.mlpackage` | SAM 3D Body body decoder (6 layers with the body rig and camera head inside), about 480 MB | the companion repo `sam3d-coreml` (`test_decoder.py`) |

SAM 3D Body's weights are gated on Hugging Face and come under Meta's license; they are not redistributed here. `sam3d-coreml` documents the conversion and every issue met on the way.

Copy a model folder with `cp -R model.mlpackage destination/` (no trailing slash after the package name, or its contents spill into the destination folder).

## Setup

### Python (ViTPose conversion)

```bash
# create and activate a virtual environment (uv)
uv venv --seed
source .venv/bin/activate
# install the conversion dependencies
uv pip install transformers coremltools torch timm huggingface_hub
# download ViTPose-plus-huge and convert it to Core ML
python convert_vitpose.py
```

`model.py` is an earlier exploration script (loading and tracing ViTPose); it is not needed for the app.

### Xcode

1. Open `PoseCam/PoseCam.xcodeproj`.
2. In Signing & Capabilities, choose your own team and a unique bundle identifier.
3. Build Settings: "User Script Sandboxing" must be No.
4. Add the model files (see above) and run on an iPhone (iOS 18 or later).

## Measured on an iPhone (so far)

| Model | Time per frame | Notes |
|---|---|---|
| Apple Vision 3D pose | about 13 ms | |
| ViTPose-H | about 43 ms | about 1.8 GB of memory once loaded |
| SAM 3D Body (encoder + decoder) | about 1.0-1.4 s per result | encoder about 400 ms, decoder about 870 ms; about one new pose per second |

Thermal state reaches "serious" with several models running at high frame rates.

## SAM 3D Body: settings and known limits

1. Compute units matter more than expected. The encoder runs with `.cpuAndGPU`: with the Neural Engine allowed, its Neural Engine compile fails, the first load takes about 4-5 minutes and the keypoints come out shifted; on the CPU alone the keypoints collapse after a fraction of a second. The decoder runs with `.all`: without the Neural Engine it crashes inside Core ML. The root causes are not known yet.
2. Loading: encoder about 2 s, decoder about 25 s the first time after installing (later loads are faster).
3. No person detector yet: the whole frame is used as the person's box (as Meta does without a detector), so the person is small in the model's input. The model always outputs a body, even when nobody is in view.
4. Camera values are Meta's defaults (focal length = image diagonal), not the phone's real intrinsics.
5. Speed: running encoder and decoder on different frames at the same time was tried; they share the GPU and both slowed down, so the app processes one frame at a time.

## Other known limits

1. ViTPose's image input uses one averaged standard deviation instead of ImageNet's three per-channel values (a small approximation).
2. The camera preview crops the frame to fit the screen, while keypoints are computed on the full frame, so overlays can be slightly misaligned.
3. Neural Engine and GPU usage are not visible in the app; use Xcode Instruments (Core ML and Metal System Trace templates).

## License

My code in this repo is under the MIT License (see `LICENSE`). The models keep their own licenses: ViTPose (from its authors) and SAM 3D Body (Meta).
