import cv2
import mediapipe as mp
import pygame
import random
import threading

# ================== Setup (Pygame) =====================
pygame.init()
width, height = 1080, 720
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Head-Controlled Snake 🐍")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 40)

snake_block = 20
snake_speed = 8

# Shared direction variable
current_direction = "RIGHT"
tracking_started = threading.Event()  # signals that face tracking started


# ================== Helper Functions =====================

def draw_snake(snake_list):
    for x in snake_list:
        pygame.draw.rect(win, (0, 255, 0), [x[0], x[1], snake_block, snake_block])


def message(msg, color, x=None, y=None):
    mesg = font.render(msg, True, color)
    if x is None or y is None:
        x, y = width / 6, height / 3
    win.blit(mesg, [x, y])


def draw_button(text, x, y, w, h, color, hover_color):
    """Draws a clickable button; returns True if clicked."""
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    action = False

    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(win, hover_color, (x, y, w, h))
        if click[0] == 1:
            action = True
    else:
        pygame.draw.rect(win, color, (x, y, w, h))

    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(x + w / 2, y + h / 2))
    win.blit(text_surface, text_rect)
    return action


# ================== Game Logic =====================

def show_game_over_screen(score):
    """Displays a fail screen with score and restart/quit buttons."""
    waiting = True
    while waiting:
        win.fill((20, 20, 20))
        message(f"💀 Game Over! Your Score: {score}", (255, 0, 0), width / 4, height / 3)

        restart_clicked = draw_button("Restart", width / 4, height / 1.8, 200, 60,
                                      (0, 128, 0), (0, 200, 0))
        quit_clicked = draw_button("Quit", width / 1.8, height / 1.8, 200, 60,
                                   (128, 0, 0), (200, 0, 0))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    cv2.destroyAllWindows()
                    exit()
                if event.key == pygame.K_r:
                    return True

        if restart_clicked:
            return True
        elif quit_clicked:
            pygame.quit()
            cv2.destroyAllWindows()
            exit()

        clock.tick(15)


def game_loop():
    """Runs one game round."""
    global current_direction

    x1 = width // 2
    y1 = height // 2
    x1_change = snake_block
    y1_change = 0

    snake_list = []
    length_of_snake = 1

    foodx = round(random.randrange(0, width - snake_block) / 20.0) * 20.0
    foody = round(random.randrange(0, height - snake_block) / 20.0) * 20.0

    score = 0
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cv2.destroyAllWindows()
                exit()

        # Movement direction (based on tracking)
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

        # Display score
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        win.blit(score_text, [10, 10])

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - snake_block) / 20.0) * 20.0
            foody = round(random.randrange(0, height - snake_block) / 20.0) * 20.0
            length_of_snake += 1
            score += 1

        clock.tick(snake_speed)

    # Show fail screen
    return show_game_over_screen(score)


def run_interactive_game():
    """Main loop that handles restarting/ending."""
    while True:
        restart = game_loop()
        if not restart:
            break


# ================== Head Tracking (OpenCV + Mediapipe) =====================

def head_tracker():
    global current_direction

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    recent_x, recent_y = [], []
    SMOOTHING_FRAMES = 5
    DEAD_ZONE = 0.07  # sensitivity around center

    while True:
        success, image = cap.read()
        if not success:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[1]
                nose_x, nose_y = nose.x, nose.y

                # Smooth motion
                recent_x.append(nose_x)
                recent_y.append(nose_y)
                if len(recent_x) > SMOOTHING_FRAMES:
                    recent_x.pop(0)
                    recent_y.pop(0)

                avg_x = sum(recent_x) / len(recent_x)
                avg_y = sum(recent_y) / len(recent_y)

                # Signal tracking start
                if not tracking_started.is_set():
                    print("✅ Face tracking detected! Starting game...")
                    tracking_started.set()

                # Determine direction
                if avg_x < 0.5 - DEAD_ZONE:
                    current_direction = "RIGHT"
                elif avg_x > 0.5 + DEAD_ZONE:
                    current_direction = "LEFT"
                elif avg_y < 0.5 - DEAD_ZONE:
                    current_direction = "UP"
                elif avg_y > 0.5 + DEAD_ZONE:
                    current_direction = "DOWN"

                # Debug nose display
                h, w, _ = image.shape
                cv2.circle(image, (int(nose_x * w), int(nose_y * h)), 5, (0, 255, 255), -1)
                break

        # Debug info
        cv2.putText(image, f'Direction: {current_direction}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not tracking_started.is_set():
            cv2.putText(image, "Waiting for face...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Head Direction Tracker', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ================== Run Game & Tracker in Parallel =====================

tracker_thread = threading.Thread(target=head_tracker, daemon=True)
tracker_thread.start()

print("Waiting for valid face tracking to begin...")
tracking_started.wait()  # wait for Mediapipe detection

try:
    run_interactive_game()
except KeyboardInterrupt:
    print("👋 Exiting gracefully...")
finally:
    pygame.quit()
    cv2.destroyAllWindows()
