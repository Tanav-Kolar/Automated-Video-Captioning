import cv2
import os
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Setup BLIP-2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 if device=="cuda" else torch.float32)
model.to(device)

# Provide your video files here
video_paths = [
    "mp4/Video 1.mp4", "mp4/Video 2.mp4", "mp4/Video 3.mp4", "mp4/Video 4.mp4", "mp4/Video 5.mp4", "mp4/Video 6.mp4",
    "mp4/Video 7.mp4","mp4/Video 8.mp4", "mp4/Video 9.mp4","mp4/Video 10.mp4"
]

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def generate_caption(image):
    prompt = "Describe the scene in detail:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=60)
    return processor.decode(outputs[0], skip_special_tokens=True).strip()

# Generate captions
for path in video_paths:
    if not os.path.exists(path):
        print(f"{path} not found.")
        continue
    print(f"\n{path}")
    image = extract_middle_frame(path)
    if image:
        caption = generate_caption(image)
        print(f"Caption: {caption}")
    else:
        print("Failed to extract frame.")

