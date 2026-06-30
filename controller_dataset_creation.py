# Conditional Imitation Learning Dataset Collector
# Webots

from controller import Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
import os
import csv
import time
import glob
import re
# -----------------------------------
# CONFIGURACION
# -----------------------------------

MAX_ANGLE = 0.8
ANGLE_STEP = 0.02

MAX_SPEED = 30
MIN_SPEED = 0
SPEED_STEP = 2

CAPTURE_EVERY_N_STEPS = 5

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = os.path.join(DATASET_DIR, "driving_log.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)

# -----------------------------------
# CSV
# -----------------------------------

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "image",
            "timestamp",
            "speed",
            "steering",
            "cmd_left",
            "cmd_straight",
            "cmd_right",
            "cmd_follow"
        ])

# -----------------------------------
# ROBOT
# -----------------------------------

robot = Car()
driver = Driver()

timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)



# -----------------------------------
# FUNCIONES
# -----------------------------------

def get_image():

    raw = camera.getImage()

    image = np.frombuffer(
        raw,
        np.uint8
    ).reshape(
        (
            camera.getHeight(),
            camera.getWidth(),
            4
        )
    )

    return image

def encode_command(command):

    if command == "left":
        return [1,0,0,0]

    elif command == "straight":
        return [0,1,0,0]

    elif command == "right":
        return [0,0,1,0]

    return [0,0,0,1]

def save_sample(image, steering, speed, command):

    global sample_id

    filename = f"img_{sample_id:06d}.jpg"

    path = os.path.join(
        IMAGE_DIR,
        filename
    )

    bgr = cv2.cvtColor(
        image,
        cv2.COLOR_BGRA2BGR
    )

    cv2.imwrite(path, bgr)

    cmd = encode_command(command)

    with open(CSV_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            filename,
            time.time(),
            speed,
            steering,
            cmd[0],
            cmd[1],
            cmd[2],
            cmd[3]
        ])

    sample_id += 1

def get_next_sample_id():

    files = glob.glob(
        os.path.join(
            IMAGE_DIR,
            "img_*.jpg"
        )
    )

    if len(files) == 0:
        return 0

    max_id = -1

    for file in files:

        match = re.search(
            r"img_(\d+)\.jpg",
            os.path.basename(file)
        )

        if match:

            idx = int(match.group(1))

            max_id = max(
                max_id,
                idx
            )

    return max_id + 1


# -----------------------------------
# ESTADOS
# -----------------------------------

speed = 20
steering = 0.0

current_command = "follow"

sample_id = get_next_sample_id()

print(
    f"Starting from sample {sample_id}"
)
step_counter = 0

recording = False

last_enter_time = 0

# -----------------------------------
# MAIN LOOP
# -----------------------------------

while robot.step() != -1:

    key = keyboard.getKey()
    steering_key_pressed = False

    # ---------------------------------
    # LEER TODAS LAS TECLAS PRESIONADAS
    # ---------------------------------

    while key != -1:

        # -------- CONDUCCION --------

        if key == Keyboard.UP:

            speed += SPEED_STEP

        elif key == Keyboard.DOWN:

            speed -= SPEED_STEP

        elif key == Keyboard.LEFT:

            steering -= ANGLE_STEP
            steering_key_pressed = True

        elif key == Keyboard.RIGHT:

            steering += ANGLE_STEP
            steering_key_pressed = True

        # -------- COMANDOS CIL --------

        elif key == ord('A'):

            current_command = "left"
            print("COMMAND = LEFT")

        elif key == ord('W'):

            current_command = "straight"
            print("COMMAND = STRAIGHT")

        elif key == ord('D'):

            current_command = "right"
            print("COMMAND = RIGHT")

        elif key == ord('S'):

            current_command = "follow"
            print("COMMAND = FOLLOW")

        elif key == ord(' '):

             now = time.time()

             if now - last_enter_time > 0.5:

                recording = not recording

                last_enter_time = now

                if recording:
                    print("=== RECORDING STARTED ===")
                else:
                    print("=== RECORDING STOPPED ===")

        key = keyboard.getKey()

    # ---------------------------------
    # AUTOCENTRADO DEL VOLANTE
    # ---------------------------------

    if not steering_key_pressed:

        steering *= 0.92

        if abs(steering) < 0.01:
            steering = 0.0

    # ---------------------------------
    # LIMITES
    # ---------------------------------

    speed = max(
        min(speed, MAX_SPEED),
        MIN_SPEED
    )

    steering = max(
        min(steering, MAX_ANGLE),
        -MAX_ANGLE
    )

    # ---------------------------------
    # APLICAR CONTROL
    # ---------------------------------

    driver.setCruisingSpeed(speed)
    driver.setSteeringAngle(steering)

    # ---------------------------------
    # CAPTURA AUTOMATICA
    # ---------------------------------

    if recording:

     step_counter += 1

     if step_counter % CAPTURE_EVERY_N_STEPS == 0:

        image = get_image()

        save_sample(
                image,
                steering,
                speed,
                current_command
            )

        if sample_id % 100 == 0:

                print(f"Samples: {sample_id}")