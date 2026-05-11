# code from past project 

import json
import math
import argparse

court_data = {}

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

def server_and_returner_position_ok (bboxes, target_pos, returner_left, returner_right, server_threshold, returner_threshold, server_pos):
    # print(f"    DEBUG server_and_returner_position_ok: checking {len(bboxes)} player boxes")
    global court_data

    
    for srv_idx, bbox in enumerate(bboxes):
        sx = (bbox[0] + bbox[2]) / 2
        sy = bbox[3]  # bottom center y (for server)
        tx = target_pos[0]
        ty = target_pos[1]
        
        # print(f"      Player {srv_idx}: bbox={bbox}, center=({sx:.1f}, {sy:.1f})")
        # print(f"      Target position: ({tx:.1f}, {ty:.1f})")

        if server_pos == "top":
            server_threshold_y = 30
            if sx < target_pos[0]:
                side = "deuce"
            else:
                side = "ad"
        elif server_pos == "bottom":
            server_threshold_y = (court_data["bottom_T_line"][1] - court_data["top_T_line"][1]) * 0.7
            if sx < target_pos[0]:
                side = "ad"
            else:
                side = "deuce"
        
        
        
        if abs(tx - sx) <= server_threshold and abs(ty - sy) <= server_threshold_y:
            r_idx = 1 - srv_idx
            rbox = bboxes[r_idx]
            rcx = (rbox[0] + rbox[2]) / 2
            if server_pos == "top":
                rcy = (rbox[1] + rbox[3]) / 2  # returner CENTER
            elif server_pos == "bottom":
                rcy = rbox[3]

            
            
            dist_left = distance((rcx, rcy), returner_left)
            dist_right = distance((rcx, rcy), returner_right)
           
            
            # Check each condition separately for debugging
            cond1 = (server_pos == "top" and side == "deuce" and dist_right <= returner_threshold)
            cond2 = (server_pos == "top" and side == "ad" and dist_left <= returner_threshold)
            cond3 = (server_pos == "bottom" and side == "deuce" and dist_left <= returner_threshold)
            cond4 = (server_pos == "bottom" and side == "ad" and dist_right <= returner_threshold)
            
            
            
            result = cond1 or cond2 or cond3 or cond4
            print(f"      Final result: {result}")
            
            return result
        else:
            print(f"      ❌ Server {srv_idx} not in correct position")
    
    print(f"    DEBUG: No valid server position found")
    return False

def detect_points_in_play(ball_positions, player_boxes, bounces, target_top, target_bottom, target_threshold_top, target_threshold_bottom, max_missing=30):
    global court_data

    point_in_play = False
    missing_ball = 0
    missing_players = 0
    alerts = []
    prev_x, prev_y = None, None
    
    # Initialize these variables outside the loop to persist between frames
    target_pos = None
    target_label = None
    server_threshold = None
    returner_threshold = None
    returner_left = None
    returner_right = None
    
    # Create a set of bounce frames for quick lookup
    bounce_frames = set()
    bounce_lookup = {}
    for bounce_data in bounces:
        bounce_frame = bounce_data[0]
        bounce_point = bounce_data[1:3]  # Ensure we only get x, y coordinates
        bounce_frames.add(bounce_frame)
        bounce_lookup[bounce_frame] = bounce_point
        print(f"DEBUG: Bounce at frame {bounce_frame}, point: {bounce_point}")

    ball_frames = set(frame for frame, x, y in ball_positions)
    print(f"Total ball frames: {len(ball_frames)}")
    print(f"Total bounce frames: {len(bounce_frames)}")
    print(f"Bounce frames in ball data: {len(bounce_frames & ball_frames)}")
    print(f"Bounce frames NOT in ball data: {len(bounce_frames - ball_frames)}")   
       
    ball_position_dict = {frame: (x, y) for frame, x, y in ball_positions}
    all_frames = sorted(set(ball_position_dict.keys()) | bounce_frames)

    for frame in all_frames:
        x, y = ball_position_dict.get(frame, (0, 0))  

        players_present = frame in player_boxes and len(player_boxes[frame]) >= 2
        no_players = frame in player_boxes and len(player_boxes[frame]) == 0
        ball_visible = (x != 0 and y != 0)

       

        # Check for point start only when not currently in play
        if not point_in_play and frame in bounce_frames:
            bounce_point = bounce_lookup[frame]
            print(f"DEBUG: Processing bounce at frame {frame}, bounce_point: {bounce_point}")
            
            dist_to_top = distance(bounce_point, target_top)
            dist_to_bottom = distance(bounce_point, target_bottom)
            # print(f"    Distance to top target: {dist_to_top:.1f}, threshold: {target_threshold_top}")  
            # print(f"    Distance to bottom target: {dist_to_bottom:.1f}, threshold: {target_threshold_bottom}") 
            # print(f"    DEBUG: dist_to_top <= target_threshold_top: {dist_to_top} <= {target_threshold_top} = {dist_to_top <= target_threshold_top}")
            # print(f"    DEBUG: dist_to_bottom <= target_threshold_bottom: {dist_to_bottom} <= {target_threshold_bottom} = {dist_to_bottom <= target_threshold_bottom}")

            # Reset target variables for this bounce check
            target_pos = None
            target_label = None
            
            if dist_to_top <= target_threshold_top:
                print(f"  ✓ Bounce near TOP serve target at frame {frame}")
                target_pos, target_label = target_top, "top"
                server_threshold = target_threshold_top
                returner_threshold = target_threshold_bottom
                returner_left = court_data["bottom_left_single"] 
                returner_right = court_data["bottom_right_single"]
            elif dist_to_bottom <= target_threshold_bottom:
                print(f"  ✓ Bounce near BOTTOM serve target at frame {frame}")
                target_pos, target_label = target_bottom, "bottom"
                server_threshold = target_threshold_bottom
                returner_threshold = target_threshold_top
                returner_left = court_data["top_left_single"]
                returner_right = court_data["top_right_single"] 
            else:
                print(f"  Bounce not near serve target at frame {frame} (top: {dist_to_top:.1f}, bottom: {dist_to_bottom:.1f})")
                print(f"    DEBUG: Neither condition met - this should not happen if dist_to_bottom <= target_threshold_bottom!")

            # Only proceed if we have a valid target position
            if target_pos is not None:
                # print(f"   Checking player positions for {target_label} serve...")
                # print(f"    DEBUG: frame {frame} in player_boxes: {frame in player_boxes}")
                if frame in player_boxes:
                    print(f"    DEBUG: player_boxes[{frame}] length: {len(player_boxes[frame])}")
                    print(f"    DEBUG: player_boxes[{frame}]: {player_boxes[frame]}")
                
                if frame in player_boxes and len(player_boxes[frame]) >= 2:
                    bboxes = player_boxes[frame]
                    # print(f"    DEBUG: Calling server_and_returner_position_ok with:")
                    # print(f"      bboxes: {bboxes}")
                    # print(f"      target_pos: {target_pos}")
                    # print(f"      returner_left: {returner_left}")
                    # print(f"      returner_right: {returner_right}")
                    # print(f"      server_threshold: {server_threshold}")
                    # print(f"      returner_threshold: {returner_threshold}")
                    # print(f"      target_label: {target_label}")
                    
                    ok = server_and_returner_position_ok(bboxes, target_pos, returner_left, returner_right, server_threshold, returner_threshold, target_label)
                    print(f"    DEBUG: server_and_returner_position_ok returned: {ok}")
                    print(f"    DEBUG: players_present: {players_present}")
                    
                    if players_present and ok:
                        point_in_play = True
                        missing_ball = 0
                        missing_players = 0
                        print(f"*** Point START at frame {frame} ({target_label} side) ***")
                    else:
                        print(f"  ❌ Players not in correct positions at frame {frame}")
                        if not players_present:
                            print(f"    → players_present is False")
                        if not ok:
                            print(f"    → server_and_returner_position_ok returned False")
                else:
                    print(f"  ❌ Not enough player boxes to evaluate at frame {frame}")
            else:
               print(f"  → Skipping frame {frame} - bounce not near any serve target")

        # Handle point continuation and ending
        if point_in_play:
            if not ball_visible:
                # Only check player positions if we have valid target info
                if (target_pos is not None and frame in player_boxes and len(player_boxes[frame]) >= 2):
                    bboxes = player_boxes[frame]
                    if not server_and_returner_position_ok(bboxes, target_pos, returner_left, returner_right, server_threshold, returner_threshold, target_label):
                        missing_ball += 1
                    else: 
                        missing_ball = 0
                else:
                    missing_ball += 1
            else:
                missing_ball = 0

            if not players_present:
                missing_players += 1
            else:
                missing_players = 0

            if missing_ball > max_missing:
                print(f"Point ends due to missing ball for {missing_ball} frames (max allowed: {max_missing})")
                point_in_play = False
                print(f"Point ENDS at frame {frame}")
                # Reset variables for next point
                target_pos = None
                target_label = None
            elif missing_players > max_missing:
                print(f"Point ends due to missing players for {missing_players} frames (max allowed: {max_missing})")
                point_in_play = False
                print(f"Point ENDS at frame {frame}")
                # Reset variables for next point
                target_pos = None
                target_label = None
            elif no_players:
                print(f"Point ends due to no players detected at frame {frame}")
                point_in_play = False
                print(f"Point ENDS at frame {frame}")
                # Reset variables for next point
                target_pos = None
                target_label = None

        alerts.append((frame, point_in_play))
        prev_x, prev_y = x, y

    print(f"Total alerts generated: {len(alerts)}")
    return alerts

def extract_play_segments_from_alerts(alerts):
    segments = []
    segment_start = None

    for frame, in_play in alerts:
        if in_play and segment_start is None:
            segment_start = frame
        elif not in_play and segment_start is not None:
            segments.append({"start": segment_start, "end": frame - 1})
            segment_start = None

    # Handle if last segment goes till end
    if segment_start is not None:
        segments.append({"start": segment_start, "end": alerts[-1][0]})

    return segments

def main(args):
    global court_data  # needed in detect_points_in_play for server_and_returner_position_ok

    with open(args.ball_json, "r") as f:
        ball_data = json.load(f)

    with open(args.players_json, "r") as f:
        player_data_raw = json.load(f)
    player_data = {int(k): v for k, v in player_data_raw.items()}

    with open(args.bounces_json, "r") as f:
        bounces_data = json.load(f)

    with open(args.court_keypoints_json, "r") as f:
        court_data = json.load(f)

    serve_position_top = midpoint(court_data["top_left_single"], court_data["top_right_single"])
    serve_position_bottom = midpoint(court_data["bottom_left_single"], court_data["bottom_right_single"])

    # Find the threshold using size of the court
    target_threshold_top = abs(court_data["top_right_single"][0] - serve_position_top[0]) * 0.65
    target_threshold_bottom = abs(serve_position_bottom[0] - court_data["bottom_right_single"][0]) *0.65

    alerts = detect_points_in_play(
        ball_data,
        player_data,
        bounces_data,
        serve_position_top,
        serve_position_bottom,
        target_threshold_top,
        target_threshold_bottom,
        max_missing=args.max_missing
    )

    play_segments = extract_play_segments_from_alerts(alerts)

    print(f"Detected {len(play_segments)} play segments:")
    for seg in play_segments:
        print(f"  Start: {seg['start']}, End: {seg['end']}")

    # Save segments
    with open(args.output_json, "w") as f:
        json.dump({"segments": play_segments}, f, indent=2)
    print(f"Saved play segments to {args.output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect play segments in tennis video data")

    parser.add_argument("--ball_json", required=True, help="Path to ball coordinates JSON")
    parser.add_argument("--players_json", required=True, help="Path to players JSON")
    parser.add_argument("--bounces_json", required=True, help="Path to bounces JSON")
    parser.add_argument("--court_keypoints_json", required=True, help="Path to court keypoints JSON")

    # parser.add_argument("--target_threshold_top", type=float, default=150.0, help="Distance threshold for top side serve")
    # parser.add_argument("--target_threshold_bottom", type=float, default=350.0, help="Distance threshold for bottom side serve")

    parser.add_argument("--max_missing", type=int, default=50, help="Max missing frames to consider point ended")

    parser.add_argument("--output_json", required=True, help="Output JSON file path for play segments")

    args = parser.parse_args()
    main(args)