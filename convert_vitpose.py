import torch  # PyTorch: runs the model
from transformers import VitPoseForPoseEstimation  # Hugging Face ViTPose implementation
import coremltools as ct  # converts the traced PyTorch model to Core ML
import numpy as np  # for comparing the two outputs

# Base-size ViTPose, 17 COCO keypoints (downloads on first run)
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge")
model.eval()  # inference mode


# One person crop: batch 1, 3 color channels, height 256, width 192
dummy = torch.randn(1, 3, 256, 192)
dataset_index = torch.tensor([0], dtype=torch.long)
with torch.no_grad():
    out = model(pixel_values=dummy, dataset_index=dataset_index)

print(out.heatmaps.shape)  # one heatmap per keypoint


# Wrapper so the model returns a plain tensor (tracing needs that)
class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("ds_idx", torch.tensor([0], dtype=torch.long))
    def forward(self, x):
        return self.m(pixel_values=x, dataset_index=self.ds_idx).heatmaps

traced = torch.jit.trace(Wrapper(model).eval(), dummy, strict=False)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=(1, 3, 256, 192),
        color_layout=ct.colorlayout.RGB,
        scale=1 / (255 * 0.226),                          # ImageNet std (average of the 3 channels)
        bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],  # ImageNet mean / std per channel
    ),],
    outputs=[ct.TensorType(name="heatmaps")],             # readable output name
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18,
)

mlmodel.save("vitpose_base.mlpackage")
print("saved")