# test_tracker.py
from hand_tracker import HandTracker
import cv2

tracker = HandTracker()
tracker.start()

while True:
    frame = tracker.update()
    if frame is None:
        break
    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.release()