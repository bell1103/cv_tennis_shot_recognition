# code from past project 

import cv2
import json
import argparse

class LabelKeypoints:
    def __init__(self, video_path = None, output_path="court_keypoints.json", display_size=(1280,720)):
        self.output_path = output_path
        self.keypoint_names = [
            "top_left", "top_right", "bottom_left", "bottom_right", 
            "top_left_single", "top_right_single", "bottom_left_single",
            "bottom_right_single", "top_left_service", "top_right_service", "bottom_left_service", 
            "bottom_right_service", "top_T_line", "bottom_T_line"
        ]
        self.keypoints = {}     # dict: name -> (x,y)
        self.index = 0
        self.display_size = display_size
        self.video_path = video_path
        
        # load frame from video
        cap = cv2.VideoCapture(self.video_path)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError(f"Failed to read frame from video: {self.video_path}")
        
        self.image = frame
        self.orig_h, self.orig_w = self.image.shape[:2]
        self.display_w, self.display_h = self.display_size
        self.scale_x = self.orig_w / self.display_w
        self.scale_y = self.orig_h / self.display_h
        self.display_image = cv2.resize(self.image, self.display_size)
        

    # def get_frames(video_path):
    #     """Gets a frame from the video"""
    #     cap = cv2.VideoCapture(video_path)
    #     frames = []
    #     while True:
    #         # when ret is false (no more frames) loop ends
    #         ret, frame = cap.read()
    #         if not ret:
    #             break 
    #         frames.append(frame)
    #     cap.release()
    #     return frames

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.index < len(self.keypoint_names):
            # Convert display coords to original image coords
            real_x = int(x * self.scale_x)
            real_y = int(y * self.scale_y)
            kp_name = self.keypoint_names[self.index]
            self.keypoints[kp_name] = [real_x, real_y]
            print(f"{kp_name}: ({real_x}, {real_y})")
            self.index += 1
            self.redraw()

    def redraw(self):
        temp = self.display_image.copy()
        for name, (real_x, real_y) in self.keypoints.items():
            disp_x = int(real_x / self.scale_x)
            disp_y = int(real_y / self.scale_y)
            cv2.circle(temp, (disp_x, disp_y), 6, (0, 255, 0), -1)
            cv2.putText(temp, name, (disp_x + 10, disp_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("Label Court Keypoints", temp)

    def run(self, save_to_files=False):
        image = self.image
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        print("Click in this order:", self.keypoint_names)
        cv2.imshow("Label Court Keypoints", self.display_image)
        cv2.setMouseCallback("Label Court Keypoints", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if save_to_files:
            with open(self.output_path, 'w') as f:
                json.dump(self.keypoints, f, indent=4)
            print(f"Saved keypoints to {self.output_path}")

        # Return keypoints as ordered list of (x,y)
        return [tuple(self.keypoints[name]) for name in self.keypoint_names]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, default='videos/', help='Path to the video or folder of videos')
    parser.add_argument('--output_path', type=str, default='court_keypoints.json', help='Path to save the output keypoints JSON')

    args = parser.parse_args()

    labeler = LabelKeypoints(video_path=args.video_path, output_path=args.output_path)

        
       