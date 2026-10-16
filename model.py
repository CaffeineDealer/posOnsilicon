import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

# Handle preprocessing/postprocessing: resizing image, normalizing pixels
processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
# Download model weights
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge")

dummy_pixel_values = torch.rand(1, 3, 256, 192)
dataset_index = torch.tensor([0])


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, dataset_index):
        outputs = self.model(pixel_values, dataset_index=dataset_index, return_dict=False)
        return outputs[0]
    
wrapped = WrappedModel(model)
wrapped.eval()

traced_model = torch.jit.trace(wrapped, (dummy_pixel_values, dataset_index))
