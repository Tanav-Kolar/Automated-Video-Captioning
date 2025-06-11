import cv2
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
from PIL import Image
import torch
import os

device = "cpu"

#Load BLIP-2 Model
print("Loading BLIP-2 model...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

# === Summarization Pipeline ===
summarizer = pipeline("summarization")

# === Frame Extraction ===
def extract_frames(video_path, interval=90):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

# === Caption a Single Frame ===
def generate_caption(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# === Summarize Multiple Captions ===
def summarize_captions(captions):
    joined = " ".join(captions)
    result = summarizer(joined, max_length=60, min_length=20, do_sample=False)
    return result[0]['summary_text']

# === Caption the Entire Video ===
def caption_video(video_path, max_frames=5):
    print(f"Processing video: {video_path}")
    frames = extract_frames(video_path)
    selected_frames = frames[:max_frames]

    print(f"Generating captions for {len(selected_frames)} frames...")
    captions = [generate_caption(frame) for frame in selected_frames]
    
    print("Raw frame captions:")
    for i, cap in enumerate(captions):
        print(f"[Frame {i}] {cap}")

    final_summary = summarize_captions(captions)
    print("\n📋 Final Video Summary:\n", final_summary)
    return final_summary

# === Run Script ===
if __name__ == "__main__":
    video_file = "your_video.mp4"  # Replace with your video file path
    caption_video(video_file)
