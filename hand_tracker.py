# hand_tracker.py
# This file handles everything related to the camera and hand detection.
# It uses MediaPipe to find the hand and read finger positions.

import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # Start hand detection
        # max_num_hands=1 because we only need one hand
        # min_detection_confidence=0.7 means 70% sure before it counts as a hand
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Open the webcam (0 = default camera)
        self.cap = cv2.VideoCapture(0)

        # Store the 21 landmark points of the detected hand
        self.landmarks = None

    def update(self):
        # Read one frame from the camera
        ret, frame = self.cap.read()

        if not ret:
            return None  # if camera fails, return nothing

        # Mirror the frame so it feels natural (like a mirror)
        frame = cv2.flip(frame, 1)

        # MediaPipe needs RGB, but OpenCV gives BGR — so we convert
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = self.hands.process(rgb)

        self.landmarks = None  # reset each frame

        if results.multi_hand_landmarks:
            # Take the first hand found
            hand = results.multi_hand_landmarks[0]

            # Draw the hand skeleton on the frame
            self.mp_draw.draw_landmarks(
                frame, hand, self.mp_hands.HAND_CONNECTIONS
            )

            # Save the 21 landmark points for gesture detection
            self.landmarks = hand.landmark

        return frame  # return the frame with drawing on it

    def get_landmarks(self):
        # Other files can call this to get the current hand landmarks
        return self.landmarks

    def release(self):
        # Properly close camera when program ends
        self.cap.release()