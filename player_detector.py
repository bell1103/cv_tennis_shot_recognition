# code from past project 
from ultralytics import YOLO
import sys
import json
import os
import cv2

def detect_and_save_players(video_path, output_dir):
    model = YOLO('models/player_detector_2_best.pt')
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_json_path = os.path.join(output_dir, f"{base_name}_players.json")

    results = model.track(video_path, classes=[1, 2], persist=True, save=False, stream=True)

    players_per_frame = {}
    frame_idx = 0
    for result in results:
        frame_players = []
        if result.boxes is not None:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, track_id, cls_id = box[:6]
                if int(cls_id) == 0:  # Only players (class 0)
                    frame_players.append([float(x1), float(y1), float(x2), float(y2)])
        players_per_frame[str(frame_idx)] = frame_players
        frame_idx += 1

    with open(output_json_path, "w") as f:
        json.dump(players_per_frame, f, indent=2)

    print(f"Saved JSON: {output_json_path}")
    return output_json_path, players_per_frame

def run_player_detection(video_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    return detect_and_save_players(video_path, output_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python player_detector.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"[INFO] Detecting players in: {video_path}")
    run_player_detection(video_path)
