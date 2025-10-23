import cv2
import mediapipe as mp
import pygame
import random
import threading

# ================== Snake Game (pygame) =====================

pygame.init()
width, height = 1280, 960
win = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 35)

snake_block = 20
snake_speed = 8


def draw_snake(snake_list):
    for x in snake_list:
        pygame.draw.rect(win, (0, 255, 0), [x[0], x[1], snake_block, snake_block])


def message(msg, color):
    mesg = font.render(msg, True, color)
    win.blit(mesg, [width / 6, height / 3])


# Shared direction and tracking signal
current_direction = "RIGHT"
tracking_started = threading.Event()  # ✅ used to signal that face tracking started


def game_loop():
    global current_direction

    x1 = width // 2
    y1 = height // 2

    x1_change = snake_block
    y1_change = 0

    snake_list = []
    length_of_snake = 1

    foodx = round(random.randrange(0, width - snake_block) / 20.0) * 20.0
    foody = round(random.randrange(0, height - snake_block) / 20.0) * 20.0

    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Update direction from head tracking
        if current_direction == "LEFT" and x1_change == 0:
            x1_change = -snake_block
            y1_change = 0
        elif current_direction == "RIGHT" and x1_change == 0:
            x1_change = snake_block
            y1_change = 0
        elif current_direction == "UP" and y1_change == 0:
            x1_change = 0
            y1_change = -snake_block
        elif current_direction == "DOWN" and y1_change == 0:
            x1_change = 0
            y1_change = snake_block

        x1 += x1_change
        y1 += y1_change

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
            game_over = True

        win.fill((0, 0, 0))
        pygame.draw.rect(win, (255, 0, 0), [foodx, foody, snake_block, snake_block])

        snake_head = [x1, y1]
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for x in snake_list[:-1]:
            if x == snake_head:
                game_over = True

        draw_snake(snake_list)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - snake_block) / 20.0) * 20.0
            foody = round(random.randrange(0, height - snake_block) / 20.0) * 20.0
            length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()


# ================== Head Tracking (OpenCV + Mediapipe) =====================

# def head_tracker():
#     global current_direction

#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh()
#     cap = cv2.VideoCapture(0)

#     while True:
#         success, image = cap.read()
#         if not success:
#             continue

#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image_rgb)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Nose tip landmark (index 1)
#                 nose = face_landmarks.landmark[1]
#                 nose_x = nose.x
#                 nose_y = nose.y

#                 # ✅ Trigger tracking event on first valid nose detection
#                 if not tracking_started.is_set():
#                     print("✅ Face tracking detected! Starting game...")
#                     tracking_started.set()

#                 # Centered around 0.5, adjust sensitivity
#                 if nose_x < 0.55:
#                     current_direction = "LEFT"
#                 elif nose_x > 0.45:
#                     current_direction = "RIGHT"
#                 elif nose_y < 0.47:
#                     current_direction = "UP"
#                 elif nose_y > 0.60:
#                     current_direction = "DOWN"

#                 break  # We only need the first detected face

#         # Show webcam feed with debug info
#         cv2.putText(image,
#                     f'Direction: {current_direction}',
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (0, 255, 0), 2)

#         if not tracking_started.is_set():
#             cv2.putText(image,
#                         "Waiting for face...",
#                         (10, 70),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (0, 0, 255), 2)

#         cv2.imshow('Head Direction Tracker', image)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

def head_tracker():
    global current_direction

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,  # improves landmark precision
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    # Use a small buffer to smooth nose motion
    recent_x = []
    recent_y = []
    SMOOTHING_FRAMES = 5

    while True:
        success, image = cap.read()
        if not success:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                nose_x = nose.x
                nose_y = nose.y

                # Add to smoothing buffer
                recent_x.append(nose_x)
                recent_y.append(nose_y)
                if len(recent_x) > SMOOTHING_FRAMES:
                    recent_x.pop(0)
                    recent_y.pop(0)

                avg_x = sum(recent_x) / len(recent_x)
                avg_y = sum(recent_y) / len(recent_y)

                # Signal start once we have stable readings
                if not tracking_started.is_set():
                    print("✅ Face tracking detected! Starting game...")
                    tracking_started.set()

                # Add a dead zone in the middle to prevent flicker
                DEAD_ZONE = 0.07  # ~7% of the frame center zone

                if avg_x < 0.5 - DEAD_ZONE:
                    current_direction = "RIGHT"
                elif avg_x > 0.5 + DEAD_ZONE:
                    current_direction = "LEFT"
                elif avg_y < 0.5 - DEAD_ZONE:
                    current_direction = "UP"
                elif avg_y > 0.5 + DEAD_ZONE:
                    current_direction = "DOWN"

                # Draw debug info
                h, w, _ = image.shape
                nose_px = int(nose_x * w)
                nose_py = int(nose_y * h)
                cv2.circle(image, (nose_px, nose_py), 5, (0, 255, 255), -1)

                break  # only use first face

        # Show info
        cv2.putText(image,
                    f'Direction: {current_direction}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        if not tracking_started.is_set():
            cv2.putText(image, "Waiting for face...",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        cv2.imshow('Head Direction Tracker', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ================== Run Game & Tracker in Parallel =====================

# Start the tracker thread
tracker_thread = threading.Thread(target=head_tracker, daemon=True)
tracker_thread.start()

# Wait until face tracking is confirmed
print("Waiting for valid face tracking to begin...")
tracking_started.wait()  # <-- this blocks until Mediapipe detects a nose

# Once valid tracking is detected, start the game
game_loop()
