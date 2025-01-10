import os
import random
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import torch.nn.functional as F

# Emotion mapping
emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgusted', 'Surprised', 'Calm']

# Randomly select a video from dataset
def select_random_video(dataset_path):
    video_files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]
    selected_video = random.choice(video_files)
    video_path = os.path.join(dataset_path, selected_video)
    label = int(selected_video.split('-')[2]) - 1  # Extract label from filename
    return video_path, label

# Load and process video
def load_video(video_path):
    """Load video data from .npy file."""
    video_data = np.load(video_path)  # Expected shape: [num_frames, height, width, channels]
    if video_data.ndim == 4:  # Ensure the shape matches
        video_data = np.transpose(video_data, (0, 3, 1, 2))  # Convert to [num_frames, channels, height, width]
    return video_data


# Mock prediction function
def predict(video_tensor, model):
    """
    Perform inference on the video tensor using dummy audio input.
    """
    model.eval()
    with torch.no_grad():
        # Debug: Print original shape
        print(f"Original video tensor shape: {video_tensor.shape}")
        
        # Combine batch_size and num_frames for Conv2D input
        if video_tensor.ndim == 4:  # Ensure video_tensor has correct dimensions
            video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension

        batch_size, num_frames, channels, height, width = video_tensor.shape
        video_tensor = video_tensor.view(batch_size * num_frames, channels, height, width)

        print(f"Processed video tensor shape: {video_tensor.shape}")

        # Create dummy audio input with correct shape [batch_size, channels, sequence_length]
        dummy_audio = torch.randn(1, 10, 16000)  # Fake audio input

        # Perform forward pass
        output = model(x_visual=video_tensor, x_audio=dummy_audio)

        # Apply softmax for class probabilities
        output = F.softmax(output, dim=1)
        return output


# Function to visualize video and predictions
def visualize_results(video_data, pred, label):
    root = tk.Tk()
    root.title("Results and Predictions")

    # Automatically set window size
    window_width, window_height = 800, 600
    root.geometry(f"{window_width}x{window_height}")

    canvas = tk.Canvas(root, width=window_width, height=window_height, bg="black")
    canvas.pack()

    pred_label = emotions[torch.argmax(pred).item()]
    video_label = tk.Label(root, text=f"Video Prediction: {pred_label}", font=('helvetica', 14), fg="white", bg="black")
    truth_label = tk.Label(root, text=f"Ground Truth: {emotions[label]}", font=('helvetica', 14), fg="gray", bg="black")
    canvas.create_window(window_width // 2, 25, window=video_label)
    canvas.create_window(window_width // 2, 50, window=truth_label)

    # Video display section
    video_canvas = tk.Label(root, bg="black")
    video_canvas.place(relx=0.5, rely=0.6, anchor="center")  # Centered placement

    # Adjust frame display scaling
    def play_video(index=0):
        if index < len(video_data):
            frame = video_data[index].transpose(1, 2, 0)  # Convert to HWC format
            frame = (frame * 255).astype(np.uint8)  # Convert back to range 0-255
            frame = Image.fromarray(frame)

            # Dynamically scale to 50% of window width and height
            frame_width = int(window_width * 0.6)
            frame_height = int(window_height * 0.6)
            frame = frame.resize((frame_width, frame_height), Image.ANTIALIAS)

            frame_tk = ImageTk.PhotoImage(frame)
            video_canvas.config(image=frame_tk)
            video_canvas.image = frame_tk

            root.after(100, play_video, index + 1)  # Recursively display next frame
        else:
            close_button = tk.Button(root, text='Close', command=root.destroy, font=('helvetica', 12, 'bold'))
            close_button.place(relx=0.5, rely=0.9, anchor="center")  # Centered close button

    play_video()
    root.mainloop()

# Load model
def load_model(model_path):
    """
    Load a pretrained model and adjust its state_dict if needed.
    """
    from model import generate_model
    from opts import parse_opts

    opt = parse_opts()  # Load options
    opt.device = 'cpu'  # Force model to run on CPU
    model, _ = generate_model(opt)  # Create model architecture
    state_dict = torch.load(model_path, map_location=torch.device(opt.device))

    # Remove `module.` prefix if necessary and filter unexpected keys
    new_state_dict = {}
    model_keys = set(model.state_dict().keys())
    for key, value in state_dict['state_dict'].items():
        new_key = key.replace('module.', '')  # Adjust key name
        if new_key in model_keys:  # Only keep keys that exist in the model
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    print(f"Loaded pretrained model from: {model_path}")
    return model


# Main function
def main():
    dataset_path = '/Users/anujpatel/Downloads/Transformer-Based-Multimodal-Emotion-Recognition-System/RAVDESS/videos/Actor_01'  # Replace with your video dataset path
    model_path = '/Users/anujpatel/Downloads/Transformer-Based-Multimodal-Emotion-Recognition-System/pretrained_model/ia_1head_moddrop_2.pth'  # Replace with your model path

    # Load model
    model = load_model(model_path)

    # Select random video
    video_path, label = select_random_video(dataset_path)
    print(f"Randomly selected video: {video_path}, Label: {emotions[label]}")

    # Load video data
    video_data = load_video(video_path)

    # Convert video to tensor
    video_tensor = torch.tensor(video_data).float() / 255.0  # Normalize to [0, 1]

    # Predict
    pred = predict(video_tensor, model)

    # Visualize results
    visualize_results(video_data, pred, label)

if __name__ == "__main__":
    main()
