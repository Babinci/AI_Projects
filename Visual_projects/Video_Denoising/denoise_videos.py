from pathlib import Path
import sys
import numpy as np
import cv2
import oidn
import subprocess
import tempfile
import os

# Original denoise_video function (unchanged)
def denoise_video(input_video_path, output_video_path):
    # Open the video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Using mp4v codec for .mov compatibility
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Initialize OIDN device and filter once (reused for all frames)
    device = oidn.NewDevice()
    oidn.CommitDevice(device)
    filter = oidn.NewFilter(device, "RT")

    # Process each frame
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to float32 and normalize to [0, 1]
        img = frame.astype(np.float32) / 255.0

        # Prepare output array
        result = np.zeros_like(img, dtype=np.float32)

        # Set input and output for OIDN filter
        oidn.SetSharedFilterImage(
            filter, "color", img, oidn.FORMAT_FLOAT3, width, height
        )
        oidn.SetSharedFilterImage(
            filter, "output", result, oidn.FORMAT_FLOAT3, width, height
        )
        oidn.CommitFilter(filter)
        oidn.ExecuteFilter(filter)

        # Convert back to uint8 format for video writing
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        # Write the denoised frame to the output video
        out.write(result)

        print(f"Processed frame {frame_idx + 1}/{frame_count}")

    # Cleanup
    oidn.ReleaseFilter(filter)
    oidn.ReleaseDevice(device)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Denoised video saved to {output_video_path}")

# Updated wrapper function to preserve audio
def denoise_video_with_audio(input_video_path, output_video_path):
    # Create temporary files for audio and video
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    with tempfile.NamedTemporaryFile(suffix=".mov", delete=False) as temp_video_file:
        temp_video_path = temp_video_file.name

    try:
        # Step 1: Extract audio from the input video
        subprocess.check_call([
            "ffmpeg",
            "-y",
            "-i", str(input_video_path),
            "-map", "0:a",          # Select audio stream
            "-c:a", "copy",         # Copy audio without re-encoding
            temp_audio_path
        ])

        # Step 2: Denoise video and save to temporary file
        denoise_video(input_video_path, temp_video_path)

        # Step 3: Combine denoised video with original audio
        subprocess.check_call([
            "ffmpeg",
            "-y",
            "-i", temp_video_path,  # Denoised video
            "-i", temp_audio_path,  # Original audio
            "-c:v", "copy",         # Copy video stream without re-encoding
            "-c:a", "copy",         # Copy audio stream without re-encoding
            "-shortest",            # Match duration to the shortest stream
            str(output_video_path)
        ])

    finally:
        # Step 4: Clean up temporary files
        os.remove(temp_audio_path)
        os.remove(temp_video_path)

# Function to process the directory structure
def process_directory(input_dir, output_dir):
    # Iterate through all .mov files recursively
    for input_video in input_dir.rglob("*.mov"):
        # Get the relative path of the input video to the input directory
        rel_path = input_video.relative_to(input_dir)
        # Construct the output subdirectory path
        output_subdir = output_dir / rel_path.parent
        # Create the subdirectory if it doesn’t exist
        output_subdir.mkdir(parents=True, exist_ok=True)
        # Construct the output video path with '_denoised' appended
        output_video = output_subdir / (input_video.stem + "_denoised" + input_video.suffix)
        print(f"Processing {input_video} -> {output_video}")
        # Denoise the video with audio preservation
        denoise_video_with_audio(input_video, output_video)

# Main execution block
if __name__ == "__main__":
    # Define input and output directories using Path
    input_dir = Path("/mnt/d/All_media_projects_synchronized_nextcloud/Computer Shared Video_Images Projects/Babinci Dnb3 Liberati Clips/PRZEBRANE")
    output_dir = Path("/mnt/d/All_media_projects_synchronized_nextcloud/Computer Shared Video_Images Projects/Babinci Dnb3 Liberati Clips/PRZEBRANE_denoised")
    # Process the directory
    process_directory(input_dir, output_dir)