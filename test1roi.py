import cv2
import json
import numpy as np
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = "road1 image 1.jpg"
ROI_FILE = "roi.json"
MODEL_PATH = "yolov8m.pt"

CONFIDENCE = 0.30


# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# =========================================================
# STEP 1: LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("\nImage loaded successfully.")


# =========================================================
# STEP 2: SELECT ROI
# =========================================================

points = []


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append([x, y])

            print(
                f"Point {len(points)}: "
                f"({x}, {y})"
            )


cv2.namedWindow("Select ROI")

cv2.setMouseCallback(
    "Select ROI",
    mouse_callback
)


print("\n======================================")
print("          ROI SELECTION")
print("======================================")
print("Click 4 points:")
print("1. Top-left")
print("2. Top-right")
print("3. Bottom-right")
print("4. Bottom-left")
print()
print("S = Save ROI and start YOLO detection")
print("R = Reset points")
print("ESC = Exit")
print("======================================\n")


while True:

    display = image.copy()

    # -------------------------------------
    # Draw points
    # -------------------------------------

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


    # -------------------------------------
    # Draw lines
    # -------------------------------------

    if len(points) >= 2:

        for i in range(len(points) - 1):

            cv2.line(
                display,
                tuple(points[i]),
                tuple(points[i + 1]),
                (0, 255, 0),
                2
            )


    # -------------------------------------
    # Complete ROI
    # -------------------------------------

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
        "Select ROI",
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # =================================================
    # SAVE ROI
    # =================================================

    if key == ord("s"):

        if len(points) != 4:

            print(
                "\nERROR: Select exactly "
                "4 points first."
            )

        else:

            # Save ROI
            with open(ROI_FILE, "w") as f:

                json.dump(
                    {
                        "points": points
                    },
                    f,
                    indent=4
                )


            print("\n======================================")
            print("ROI SAVED")
            print("======================================")
            print("Points:", points)
            print("File:", ROI_FILE)


            cv2.destroyAllWindows()

            break


    # =================================================
    # RESET
    # =================================================

    elif key == ord("r"):

        points.clear()

        print("ROI reset.")


    # =================================================
    # EXIT
    # =================================================

    elif key == 27:

        print("Cancelled.")

        cv2.destroyAllWindows()

        exit()


# =========================================================
# STEP 3: CREATE ROI MASK
# =========================================================

polygon = np.array(
    points,
    dtype=np.int32
)


mask = np.zeros(
    image.shape[:2],
    dtype=np.uint8
)


cv2.fillPoly(
    mask,
    [polygon],
    255
)


# =========================================================
# STEP 4: APPLY ROI
# =========================================================

roi_image = cv2.bitwise_and(
    image,
    image,
    mask=mask
)


# =========================================================
# STEP 5: LOAD YOLOv8
# =========================================================

print("\nLoading YOLOv8m...")

model = YOLO(
    MODEL_PATH
)

print("YOLOv8m loaded successfully.")


# =========================================================
# STEP 6: RUN DETECTION
# =========================================================

print("\nRunning vehicle detection...")

results = model(
    roi_image,
    conf=CONFIDENCE,
    verbose=False
)


# =========================================================
# STEP 7: DRAW RESULTS
# =========================================================

output = image.copy()


# Draw ROI boundary

cv2.polylines(
    output,
    [polygon],
    True,
    (0, 255, 255),
    3
)


total_vehicles = 0


vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}


# =========================================================
# STEP 8: PROCESS DETECTIONS
# =========================================================

for result in results:

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )


        # -----------------------------------------
        # Ignore non-vehicle classes
        # -----------------------------------------

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
        # Make sure detection is actually inside ROI
        # -----------------------------------------

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )


        inside = cv2.pointPolygonTest(
            polygon,
            (center_x, center_y),
            False
        )


        if inside < 0:

            continue


        # -----------------------------------------
        # Count vehicle
        # -----------------------------------------

        total_vehicles += 1

        vehicle_counts[
            vehicle_name
        ] += 1


        # -----------------------------------------
        # Draw bounding box
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
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


# =========================================================
# STEP 9: DISPLAY COUNTS
# =========================================================

cv2.rectangle(
    output,
    (10, 10),
    (250, 175),
    (0, 0, 0),
    -1
)


cv2.putText(
    output,
    f"TOTAL: {total_vehicles}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2
)


y = 75


for vehicle, count in vehicle_counts.items():

    cv2.putText(
        output,
        f"{vehicle}: {count}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y += 25


# =========================================================
# STEP 10: SAVE RESULT
# =========================================================

cv2.imwrite(
    "vehicle_detection_result.jpg",
    output
)


# =========================================================
# STEP 11: PRINT RESULT
# =========================================================

print("\n======================================")
print("       VEHICLE DETECTION RESULT")
print("======================================")

print(
    f"Cars        : {vehicle_counts['car']}"
)

print(
    f"Motorcycles : {vehicle_counts['motorcycle']}"
)

print(
    f"Buses       : {vehicle_counts['bus']}"
)

print(
    f"Trucks      : {vehicle_counts['truck']}"
)

print("--------------------------------------")

print(
    f"TOTAL       : {total_vehicles}"
)

print("--------------------------------------")

print(
    "Result saved as:"
)

print(
    "vehicle_detection_result.jpg"
)

print("======================================")


# =========================================================
# STEP 12: SHOW RESULT
# =========================================================

cv2.imshow(
    "Vehicle Detection - ROI",
    output
)

print("\nPress any key to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()