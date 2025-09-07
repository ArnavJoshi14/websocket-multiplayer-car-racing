import cv2
import mediapipe as mp
import time
from directkeys import PressKey, ReleaseKey, W, A, D, S, SPACE


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


smoothed_diff = 0.0
smoothing_factor = 0.5
steering_deadzone = 0.1

# gesture detection functions
def is_fist(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    folded = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y > landmarks[pip].y:
            folded += 1
    return folded >= 3

def is_palm(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    extended = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            extended += 1
    return extended >= 3

def release_all_keys():
    for key in [W, A, D, S, SPACE]:
        ReleaseKey(key)

# main
try:
    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        key_pressed = []

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            hand_landmarks = result.multi_hand_landmarks
            handedness = result.multi_handedness

            # assign left and right hands
            if handedness[0].classification[0].label == 'Left':
                left_lm = hand_landmarks[0].landmark
                right_lm = hand_landmarks[1].landmark
            else:
                left_lm = hand_landmarks[1].landmark
                right_lm = hand_landmarks[0].landmark

            # draw landmarks on the hands
            for handLms in hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # gesture detection
            left_fist = is_fist(left_lm)
            right_fist = is_fist(right_lm)
            left_palm = is_palm(left_lm)
            right_palm = is_palm(right_lm)

            # accelerate
            if left_fist and right_fist:
                PressKey(W)
                key_pressed.append(W)

            # reverse
            elif left_palm and right_palm:
                PressKey(S)
                key_pressed.append(S)

            # brake
            elif (left_palm and right_fist) or (right_palm and left_fist):
                PressKey(SPACE)
                key_pressed.append(SPACE)

            # EMA smoothed steering
            left_y = left_lm[0].y
            right_y = right_lm[0].y
            y_diff = left_y - right_y

            # EMA
            smoothed_diff = smoothed_diff * (1 - smoothing_factor) + y_diff * smoothing_factor

            if smoothed_diff > steering_deadzone:
                PressKey(A)
                key_pressed.append(A)
                ReleaseKey(D)
            elif smoothed_diff < -steering_deadzone:
                PressKey(D)
                key_pressed.append(D)
                ReleaseKey(A)
            else:
                ReleaseKey(A)
                ReleaseKey(D)

        # release non-triggered keys
        for key in [W, A, D, S, SPACE]:
            if key not in key_pressed:
                ReleaseKey(key)

        cv2.imshow("Steering", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except KeyboardInterrupt:
    print("Interrupted. Cleaning up...")

finally:
    release_all_keys()
    cap.release()
    cv2.destroyAllWindows()