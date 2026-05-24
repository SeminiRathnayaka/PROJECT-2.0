#hand_tracker.py
#This file is responsible for opening the camera and detecting the hand land marks using Mediapipe

import cv2                              # open opencv for camera
import mediapipe as mp                  # open mediapipe for detecting hand 21 land marks

class HandTracker:                   # everything related to hand detection under here 
    """this class opens the camera, detects hand landmarks ,draws the skeleton ,stores the landmark positions"""
    def __init__(self):
        self.mp_hands = mp.solutions.hands           # gets hand detection tool from mediapipe and saves 
        self.mp_draw = mp.solutions.drawing_utils    # drawing skeleton lines on your hand
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )                                            # Actually starts the hand detector. max_num_hands=1 means only track one hand. 0.7 means it must be 70% confident before it says "yes that's a hand

        self.cap = None                              # camera not opened yet
        self.landmarks = None                        # create an empty variable to store 21 hand points. start as none because no hand detects yet
    
    def start(self):
        self.cap = cv2.VideoCapture(0)               # open the webcam
        
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found")

    def update(self):                                 # runs every single frame. read cam,detect hand,update landmarks
        #read image from eb cam
        ret, frame = self.cap.read()                  #ret=true frame is actual image
        if not ret: #if camera fails 
            return None

        frame = cv2.flip(frame, 1)                     # mirror frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # CONVERT bgr to rgb , opencv usee bgr but mediapipe want rgb
        results = self.hands.process(rgb)              # return hand detection results 
        self.landmarks = None                          # reset landmark in every frame
        
        
        if results.multi_hand_landmarks:
            # loop through detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,                            # camera frame
                    hand_landmarks,                   # hand land marks 
                    self.mp_hands.HAND_CONNECTIONS    # skeleton connections
                    )
                   # save landmark positions
                self.landmarks = [
                    (lm.x ,lm.y ,lm.z ) for lm in hand_landmarks.landmark
                ]
        #return final frame            
        return frame

    def get_landmarks(self):
        return self.landmarks    

    def release(self):
        if self.cap:
            self.cap.release()            #close camera
        self.hands.close()    
        cv2.destroyAllWindows()           #close all    
    
