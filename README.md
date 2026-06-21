# posOnSilicon

Run ViTPose-H (human pos estimation) on iPad via Core ML.

## Status
- [x] uv venv set up
- [x] Dependencies installed: transformers, coremltools, torch, timm, huggingface_hub
- [x] ViTPose-H loaded from 'usyd-community/vitpose-plus-huge'
- [] Convert to Core ML (.mlpackge)
- [] Build Xcode app with Vision/Core ML integration
- [] Deploy to iPad (M4, 8GB)

## Setup
\'\'\'bash
uv venv --seed
source .venv/bin/activate
uv pip install transformers coremltools torch timm huggingface_hub
\'\'\'

# Model
- Source: usyd-community/vitpose-plus-huge (Hugging Face)
- ~632M params, runs comfortably in 8GB RAM