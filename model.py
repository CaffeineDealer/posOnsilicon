from transformers import AutoProcessor, VitPoseForPoseEstimation

# Handle preprocessing/postprocessing: resizing image, normalizing pixels
processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-huge")
# Download model weights
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-huge")
