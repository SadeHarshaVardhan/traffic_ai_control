import cv2
import json
import os
import time
import numpy as np
from ultralytics import YOLO


# =========================================================
# PATHS
# =========================================================

DATASET_PATH = r"D:\Traffic Ai\Dataset_1\dataset-1"

MODEL_PATH = r"D:\Traffic Ai\yolov8m.pt"

ROI_FOLDER = os.path.join(
    DATASET_PATH,
    "roi"
)


# =========================================================
# TRAFFIC SIGNAL TIMES
# =========================================================

ROAD_TIMES = {
    "road1": 30,
    "road2": 30,
    "road3": 45,
    "road4": 60
}


ROADS = [
    "road1",
    "road2",
    "road3",
    "road4"
]


# =========================================================
# YOLO SETTINGS
# =========================================================

CONFIDENCE = 0.30


# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print()
print("==============================================")
print("              TRAFFIC AI")
print("       VEHICLE COUNTING SYSTEM")
print("==============================================")
print()

print("Loading model...")
print(
    f"Model: {MODEL_PATH}"
)

if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: YOLO model not found!")
    print(
        f"Check this path:\n{MODEL_PATH}"
    )
    exit()


model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# =========================================================
# GET IMAGES
# =========================================================

def get_images(road_folder):

    images = []

    for file in os.listdir(road_folder):

        if file.lower().endswith(
            IMAGE_EXTENSIONS
        ):

            images.append(file)


    # Sort by filename
    images.sort()


    return images


# =========================================================
# LOAD ROI
# =========================================================

def load_roi(road_name):

    roi_file = os.path.join(
        ROI_FOLDER,
        f"{road_name}_roi.json"
    )


    if not os.path.exists(roi_file):

        print()
        print(
            f"ERROR: ROI not found for "
            f"{road_name}"
        )

        print(
            f"Expected file:"
        )

        print(
            roi_file
        )

        exit()


    with open(
        roi_file,
        "r"
    ) as file:

        data = json.load(file)


    return np.array(
        data["points"],
        dtype=np.int32
    )


# =========================================================
# DETECT VEHICLES INSIDE ROI
# =========================================================

def detect_vehicles(
    image,
    polygon
):

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
    # Keep only ROI
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


    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }


    total = 0


    # ---------------------------------------------
    # Process detections
    # ---------------------------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )


            # Ignore people and other objects
            if class_id not in VEHICLE_CLASSES:

                continue


            vehicle_name = (
                VEHICLE_CLASSES[
                    class_id
                ]
            )


            # -----------------------------------------
            # Bounding box
            # -----------------------------------------

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
                (
                    center_x,
                    center_y
                ),
                False
            )


            if inside < 0:

                continue


            # -----------------------------------------
            # Count vehicle
            # -----------------------------------------

            total += 1

            counts[
                vehicle_name
            ] += 1


    return total, counts


# =========================================================
# LOAD ALL FOUR ROIs
# =========================================================

print()
print("==============================================")
print("              LOADING ROIs")
print("==============================================")


road_rois = {}


for road in ROADS:

    road_rois[
        road
    ] = load_roi(
        road
    )

    print(
        f"{road.upper()} ROI loaded."
    )


# =========================================================
# LOAD ROAD IMAGES
# =========================================================

print()
print("==============================================")
print("            LOADING ROAD IMAGES")
print("==============================================")


road_images = {}


for road in ROADS:

    road_folder = os.path.join(
        DATASET_PATH,
        road
    )


    if not os.path.exists(
        road_folder
    ):

        print()
        print(
            f"ERROR: Road folder not found:"
        )

        print(
            road_folder
        )

        exit()


    images = get_images(
        road_folder
    )


    if len(images) == 0:

        print(
            f"{road.upper()}: "
            "No images found."
        )

        exit()


    road_images[
        road
    ] = images


    print(
        f"{road.upper()}: "
        f"{len(images)} images"
    )


# =========================================================
# SYSTEM READY
# =========================================================

print()
print("==============================================")
print("           TRAFFIC AI READY")
print("==============================================")

print()

for road in ROADS:

    print(
        f"{road.upper()} "
        f"Signal = "
        f"{ROAD_TIMES[road]} seconds"
    )


print()
print("ESC = STOP")
print()


# =========================================================
# IMAGE INDEX
# =========================================================

image_index = 0


# =========================================================
# MAIN TRAFFIC SIGNAL LOOP
# =========================================================

while True:


    # =====================================================
    # PROCESS ROAD 1 → ROAD 2 → ROAD 3 → ROAD 4
    # =====================================================

    for road in ROADS:


        # -------------------------------------------------
        # Check image availability
        # -------------------------------------------------

        if image_index >= len(
            road_images[road]
        ):

            continue


        image_name = road_images[
            road
        ][image_index]


        image_path = os.path.join(
            DATASET_PATH,
            road,
            image_name
        )


        # -------------------------------------------------
        # Load image
        # -------------------------------------------------

        image = cv2.imread(
            image_path
        )


        if image is None:

            print(
                f"Could not load:"
            )

            print(
                image_path
            )

            continue


        # =================================================
        # START SIGNAL TIMER
        # =================================================

        signal_duration = ROAD_TIMES[
            road
        ]


        signal_start = time.time()


        print()
        print()
        print("==============================================")
        print(
            f"🟢 {road.upper()} "
            f"SIGNAL START"
        )

        print(
            f"📸 IMAGE: "
            f"{image_name}"
        )

        print(
            f"⏱ SIGNAL TIME: "
            f"{signal_duration} seconds"
        )

        print("==============================================")


        # =================================================
        # YOLO PROCESSING
        # =================================================

        processing_start = time.time()


        total, counts = detect_vehicles(
            image,
            road_rois[road]
        )


        processing_time = (
            time.time()
            - processing_start
        )


        # =================================================
        # CALCULATE REMAINING TIME
        # =================================================

        elapsed = (
            time.time()
            - signal_start
        )


        remaining = max(
            0,
            signal_duration - elapsed
        )


        # =================================================
        # DISPLAY RESULT
        # =================================================

        print()
        print("----------------------------------------------")

        print(
            f"📸 {road.upper()} "
            f"IMAGE CAPTURED"
        )


        print(
            f"🚗 Cars        : "
            f"{counts['car']}"
        )


        print(
            f"🏍 Motorcycles : "
            f"{counts['motorcycle']}"
        )


        print(
            f"🚌 Buses       : "
            f"{counts['bus']}"
        )


        print(
            f"🚚 Trucks      : "
            f"{counts['truck']}"
        )


        print("----------------------------------------------")


        print(
            f"🚦 TOTAL VEHICLES: "
            f"{total}"
        )


        print(
            f"⚡ Processing Time: "
            f"{processing_time:.2f}s"
        )


        print("----------------------------------------------")


        # =================================================
        # LIVE SIGNAL COUNTDOWN
        # =================================================

        while True:

            elapsed = (
                time.time()
                - signal_start
            )


            remaining = (
                signal_duration
                - elapsed
            )


            # ---------------------------------------------
            # Signal finished
            # ---------------------------------------------

            if remaining <= 0:

                break


            # ---------------------------------------------
            # Terminal countdown
            # ---------------------------------------------

            print(
                f"\r🚦 "
                f"{road.upper()} "
                f"TIME REMAINING: "
                f"{remaining:05.1f}s",
                end="",
                flush=True
            )


            # ---------------------------------------------
            # ESC check
            # ---------------------------------------------

            key = (
                cv2.waitKey(50)
                & 0xFF
            )


            if key == 27:

                print()
                print()
                print(
                    "ESC pressed."
                )

                cv2.destroyAllWindows()

                exit()


        # =================================================
        # SIGNAL FINISHED
        # =================================================

        print()
        print()
        print("----------------------------------------------")

        print(
            f"🔴 {road.upper()} "
            f"SIGNAL TIME OVER"
        )

        print("----------------------------------------------")


    # =====================================================
    # NEXT IMAGE
    # =====================================================

    image_index += 1


    # =====================================================
    # CHECK DATASET END
    # =====================================================

    maximum_images = max(
        len(
            road_images[road]
        )
        for road in ROADS
    )


    if image_index >= maximum_images:

        print()
        print("==============================================")
        print(
            "ALL IMAGES PROCESSED"
        )
        print("==============================================")

        break


# =========================================================
# END
# =========================================================

cv2.destroyAllWindows()

print()
print("==============================================")
print("          TRAFFIC AI STOPPED")
print("==============================================")