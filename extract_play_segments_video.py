import argparse
import json
import cv2

def extract_segments_from_video(video_path, segments, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Extracting segments to {output_path} ...")
    frame_idx = 0
    segment_idx = 0
    total_segments = len(segments)

    current_segment = segments[segment_idx] if segments else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_segment is None:
            # No more segments to extract
            break

        start_frame = current_segment["start"]
        end_frame = current_segment["end"]

        if frame_idx > end_frame:
            segment_idx += 1
            if segment_idx >= total_segments:
                break
            current_segment = segments[segment_idx]
            start_frame = current_segment["start"]
            end_frame = current_segment["end"]

        if start_frame <= frame_idx <= end_frame:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished writing shortened video: {output_path}")

def main(args):
    with open(args.segments_json, "r") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("No segments found in JSON.")
        return

    extract_segments_from_video(args.video_path, segments, args.output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract play segments from video using segment JSON")
    parser.add_argument("--video_path", required=True, help="Path to the input full video")
    parser.add_argument("--segments_json", required=True, help="Path to JSON file with play segments")
    parser.add_argument("--output_video", required=True, help="Path to save the extracted short video")

    args = parser.parse_args()
    main(args)
