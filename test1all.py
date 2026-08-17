import cv2
import json
import os
import numpy as np
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = r"D:\Traffic Ai\Dataset_1\dataset-1"

MODEL_PATH = "yolov8m.pt"

ROI_FOLDER = os.path.join(DATASET_PATH, "roi")

CONFIDENCE = 0.30

ROADS = [
    "road1",
    "road2",
    "road3",
    "road4"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# =========================================================
# VEHICLE CLASSES
# =========================================================

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# =========================================================
# CREATE ROI FOLDER
# =========================================================

os.makedirs(ROI_FOLDER, exist_ok=True)


# =========================================================
# LOAD YOLO
# =========================================================

print("\n========================================")
print("Loading YOLOv8m...")
print("========================================")

model = YOLO(MODEL_PATH)

print("YOLOv8m loaded successfully.")


# =========================================================
# GET IMAGES FROM ROAD
# =========================================================

def get_images(road_folder):

    images = []

    for file in os.listdir(road_folder):

        if file.lower().endswith(IMAGE_EXTENSIONS):

            images.append(file)

    # Sort naturally by image number
    images.sort(
        key=lambda x: int(
            ''.join(
                filter(
                    str.isdigit,
                    os.path.splitext(x)[0]
                )
            )
        )
        if any(
            c.isdigit()
            for c in os.path.splitext(x)[0]
        )
        else 0
    )

    return images


# =========================================================
# ROI SELECTION
# =========================================================

def select_roi(image, road_name):

    points = []

    window_name = f"Select ROI - {road_name}"

    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(points) < 4:

                points.append([x, y])

                print(
                    f"{road_name} - "
                    f"Point {len(points)}: "
                    f"({x}, {y})"
                )

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print("\n========================================")
    print(f"ROI SELECTION: {road_name}")
    print("========================================")
    print("Click 4 points:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print()
    print("S = Save ROI")
    print("R = Reset")
    print("ESC = Exit")
    print("========================================")

    while True:

        display = image.copy()

        # ---------------------------------------------
        # Draw points
        # ---------------------------------------------

        for i, point in enumerate(points):

            x, y = point

            cv2.circle(
                display,
                (x, y),
                7,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ---------------------------------------------
        # Draw lines
        # ---------------------------------------------

        if len(points) >= 2:

            for i in range(len(points) - 1):

                cv2.line(
                    display,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    (0, 255, 0),
                    2
                )

        # ---------------------------------------------
        # Close polygon
        # ---------------------------------------------

        if len(points) == 4:

            cv2.line(
                display,
                tuple(points[3]),
                tuple(points[0]),
                (0, 255, 0),
                2
            )

            overlay = display.copy()

            polygon = np.array(
                points,
                dtype=np.int32
            )

            cv2.fillPoly(
                overlay,
                [polygon],
                (0, 255, 0)
            )

            display = cv2.addWeighted(
                overlay,
                0.20,
                display,
                0.80,
                0
            )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(1) & 0xFF

        # ---------------------------------------------
        # SAVE
        # ---------------------------------------------

        if key == ord("s"):

            if len(points) != 4:

                print(
                    "ERROR: Select exactly "
                    "4 points."
                )

                continue

            cv2.destroyWindow(
                window_name
            )

            return points

        # ---------------------------------------------
        # RESET
        # ---------------------------------------------

        elif key == ord("r"):

            points.clear()

            print(
                f"{road_name} ROI reset."
            )

        # ---------------------------------------------
        # ESC
        # ---------------------------------------------

        elif key == 27:

            cv2.destroyWindow(
                window_name
            )

            return None


# =========================================================
# SAVE ROI
# =========================================================

def save_roi(road_name, points):

    roi_file = os.path.join(
        ROI_FOLDER,
        f"{road_name}_roi.json"
    )

    with open(
        roi_file,
        "w"
    ) as file:

        json.dump(
            {
                "road": road_name,
                "points": points
            },
            file,
            indent=4
        )

    print(
        f"{road_name} ROI saved:"
    )

    print(
        roi_file
    )


# =========================================================
# LOAD ROI
# =========================================================

def load_roi(road_name):

    roi_file = os.path.join(
        ROI_FOLDER,
        f"{road_name}_roi.json"
    )

    if not os.path.exists(roi_file):

        return None

    with open(
        roi_file,
        "r"
    ) as file:

        data = json.load(file)

    return data["points"]


# =========================================================
# VEHICLE DETECTION
# =========================================================

def detect_vehicles(
    image,
    points,
    road_name,
    image_name
):

    polygon = np.array(
        points,
        dtype=np.int32
    )

    # ---------------------------------------------
    # Create ROI mask
    # ---------------------------------------------

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    # ---------------------------------------------
    # Apply ROI
    # ---------------------------------------------

    roi_image = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    # ---------------------------------------------
    # YOLO detection
    # ---------------------------------------------

    results = model(
        roi_image,
        conf=CONFIDENCE,
        verbose=False
    )

    output = image.copy()

    # ---------------------------------------------
    # Draw ROI
    # ---------------------------------------------

    cv2.polylines(
        output,
        [polygon],
        True,
        (0, 255, 255),
        3
    )

    vehicle_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    total_vehicles = 0

    # ---------------------------------------------
    # Process detections
    # ---------------------------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            # Ignore non-vehicles
            if class_id not in VEHICLE_CLASSES:

                continue

            vehicle_name = VEHICLE_CLASSES[
                class_id
            ]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # -----------------------------------------
            # Vehicle center
            # -----------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            # -----------------------------------------
            # Check center inside ROI
            # -----------------------------------------

            inside = cv2.pointPolygonTest(
                polygon,
                (center_x, center_y),
                False
            )

            if inside < 0:

                continue

            # -----------------------------------------
            # Count
            # -----------------------------------------

            total_vehicles += 1

            vehicle_counts[
                vehicle_name
            ] += 1

            # -----------------------------------------
            # Bounding box
            # -----------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # -----------------------------------------
            # Label
            # -----------------------------------------

            label = (
                f"{vehicle_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                output,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # =================================================
    # DISPLAY INFORMATION
    # =================================================

    cv2.rectangle(
        output,
        (10, 10),
        (300, 180),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        road_name,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        image_name,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"TOTAL: {total_vehicles}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Car: {vehicle_counts['car']}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Bike: {vehicle_counts['motorcycle']}",
        (120, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Bus: {vehicle_counts['bus']}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        f"Truck: {vehicle_counts['truck']}",
        (120, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    return output, total_vehicles, vehicle_counts


# =========================================================
# MAIN
# =========================================================

print("\n")
print("============================================")
print("     TRAFFIC VEHICLE ROI SYSTEM")
print("============================================")
print()
print("Road order:")
print("ROAD 1 → ROAD 2 → ROAD 3 → ROAD 4")
print("then repeat")
print()
print("ESC = Stop")
print("============================================")


# =========================================================
# STEP 1
# SELECT ROI FOR ALL FOUR ROADS
# =========================================================

road_rois = {}

for road_name in ROADS:

    road_folder = os.path.join(
        DATASET_PATH,
        road_name
    )

    if not os.path.exists(road_folder):

        print(
            f"\nERROR: {road_folder} "
            "does not exist."
        )

        exit()

    images = get_images(
        road_folder
    )

    if len(images) == 0:

        print(
            f"\nERROR: No images found "
            f"in {road_name}"
        )

        exit()

    # ---------------------------------------------
    # First image
    # ---------------------------------------------

    first_image_name = images[0]

    first_image_path = os.path.join(
        road_folder,
        first_image_name
    )

    first_image = cv2.imread(
        first_image_path
    )

    if first_image is None:

        print(
            f"ERROR loading "
            f"{first_image_path}"
        )

        exit()

    # ---------------------------------------------
    # Check existing ROI
    # ---------------------------------------------

    existing_roi = load_roi(
        road_name
    )

    if existing_roi is not None:

        print(
            f"\nExisting ROI found for "
            f"{road_name}."
        )

        print(
            "Using saved ROI."
        )

        road_rois[
            road_name
        ] = existing_roi

    else:

        # Select new ROI
        selected_points = select_roi(
            first_image,
            road_name
        )

        if selected_points is None:

            print(
                "\nProgram stopped."
            )

            exit()

        save_roi(
            road_name,
            selected_points
        )

        road_rois[
            road_name
        ] = selected_points


# =========================================================
# ALL ROIs READY
# =========================================================

print("\n")
print("============================================")
print("ALL FOUR ROIs READY")
print("============================================")

for road_name in ROADS:

    print(
        f"{road_name}: READY"
    )


print("\nStarting vehicle detection...")
print("Press ESC anytime to stop.")


# =========================================================
# FIND MAXIMUM NUMBER OF IMAGES
# =========================================================

road_images = {}

maximum_images = 0

for road_name in ROADS:

    road_folder = os.path.join(
        DATASET_PATH,
        road_name
    )

    images = get_images(
        road_folder
    )

    road_images[
        road_name
    ] = images

    if len(images) > maximum_images:

        maximum_images = len(images)


# =========================================================
# MAIN LOOP
# =========================================================

stop_program = False

for image_index in range(
    maximum_images
):

    if stop_program:
        break

    # ---------------------------------------------
    # ROAD 1 → ROAD 2 → ROAD 3 → ROAD 4
    # ---------------------------------------------

    for road_name in ROADS:

        if image_index >= len(
            road_images[road_name]
        ):

            continue

        image_name = road_images[
            road_name
        ][image_index]

        image_path = os.path.join(
            DATASET_PATH,
            road_name,
            image_name
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"Could not load: "
                f"{image_path}"
            )

            continue

        print("\n--------------------------------------------")

        print(
            f"Processing: "
            f"{road_name} → "
            f"{image_name}"
        )

        # -----------------------------------------
        # Detect
        # -----------------------------------------

        output, total, counts = detect_vehicles(
            image,
            road_rois[road_name],
            road_name,
            image_name
        )

        # -----------------------------------------
        # Print result
        # -----------------------------------------

        print(
            f"Car        : {counts['car']}"
        )

        print(
            f"Motorcycle : {counts['motorcycle']}"
        )

        print(
            f"Bus        : {counts['bus']}"
        )

        print(
            f"Truck      : {counts['truck']}"
        )

        print(
            f"TOTAL      : {total}"
        )

        # -----------------------------------------
        # Display
        # -----------------------------------------

        cv2.imshow(
            "Traffic Vehicle Detection",
            output
        )

        # -----------------------------------------
        # Wait
        #
        # Press ESC = stop
        # Any other key = next image
        # -----------------------------------------

        key = cv2.waitKey(0) & 0xFF

        if key == 27:

            print(
                "\nESC pressed."
            )

            stop_program = True

            break


# =========================================================
# END
# =========================================================

cv2.destroyAllWindows()

print("\n")
print("============================================")
print("PROGRAM STOPPED")
print("============================================")