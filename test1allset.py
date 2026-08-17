import cv2
import json
import os
import time
import numpy as np
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = r"D:\Traffic Ai\Dataset_1\dataset-1"

MODEL_PATH = "yolov8m.pt"

ROI_FOLDER = os.path.join(
    DATASET_PATH,
    "roi"
)

CONFIDENCE = 0.30


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

os.makedirs(
    ROI_FOLDER,
    exist_ok=True
)


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("==============================================")
print("             TRAFFIC AI")
print("       VEHICLE COUNTING SYSTEM")
print("==============================================")
print()

print("Loading YOLOv8m...")

model = YOLO(MODEL_PATH)

print("YOLOv8m loaded successfully.")


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

    # Natural image-number sorting
    def image_number(filename):

        name = os.path.splitext(filename)[0]

        numbers = ''.join(
            filter(
                str.isdigit,
                name
            )
        )

        if numbers:
            return int(numbers)

        return 0

    images.sort(
        key=image_number
    )

    return images


# =========================================================
# SELECT ROI
# =========================================================

def select_roi(
    image,
    road_name
):

    points = []

    window_name = (
        f"ROI - {road_name}"
    )


    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(points) < 4:

                points.append(
                    [x, y]
                )

                print(
                    f"{road_name} "
                    f"Point {len(points)}: "
                    f"({x}, {y})"
                )


    cv2.namedWindow(
        window_name
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    print()
    print("----------------------------------------------")
    print(
        f"SELECT ROI FOR "
        f"{road_name.upper()}"
    )
    print("----------------------------------------------")

    print("Click:")
    print("1 → Top-left")
    print("2 → Top-right")
    print("3 → Bottom-right")
    print("4 → Bottom-left")

    print()
    print("S → Save")
    print("R → Reset")
    print("ESC → Exit")


    while True:

        display = image.copy()


        # -----------------------------------------
        # Points
        # -----------------------------------------

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


        # -----------------------------------------
        # Lines
        # -----------------------------------------

        if len(points) >= 2:

            for i in range(
                len(points) - 1
            ):

                cv2.line(
                    display,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    (0, 255, 0),
                    2
                )


        # -----------------------------------------
        # Complete polygon
        # -----------------------------------------

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


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        # SAVE

        if key == ord("s"):

            if len(points) != 4:

                print(
                    "Select exactly 4 points."
                )

                continue


            cv2.destroyWindow(
                window_name
            )

            return points


        # RESET

        elif key == ord("r"):

            points.clear()

            print(
                "ROI reset."
            )


        # ESC

        elif key == 27:

            cv2.destroyWindow(
                window_name
            )

            return None


# =========================================================
# SAVE ROI
# =========================================================

def save_roi(
    road_name,
    points
):

    filename = os.path.join(
        ROI_FOLDER,
        f"{road_name}_roi.json"
    )


    with open(
        filename,
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


# =========================================================
# LOAD ROI
# =========================================================

def load_roi(
    road_name
):

    filename = os.path.join(
        ROI_FOLDER,
        f"{road_name}_roi.json"
    )


    if not os.path.exists(
        filename
    ):

        return None


    with open(
        filename,
        "r"
    ) as file:

        data = json.load(file)


    return data["points"]


# =========================================================
# DETECT VEHICLES
# =========================================================

def detect_vehicles(
    image,
    points
):

    polygon = np.array(
        points,
        dtype=np.int32
    )


    # -----------------------------------------
    # ROI mask
    # -----------------------------------------

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )


    cv2.fillPoly(
        mask,
        [polygon],
        255
    )


    # -----------------------------------------
    # ROI image
    # -----------------------------------------

    roi_image = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )


    # -----------------------------------------
    # YOLO
    # -----------------------------------------

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


    # -----------------------------------------
    # Process boxes
    # -----------------------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )


            if class_id not in VEHICLE_CLASSES:

                continue


            vehicle_name = (
                VEHICLE_CLASSES[
                    class_id
                ]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Center point

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # Check center inside ROI

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


            total += 1

            counts[
                vehicle_name
            ] += 1


    return total, counts


# =========================================================
# PREPARE ROADS
# =========================================================

road_rois = {}

road_images = {}


print()
print("==============================================")
print("             PREPARING ROADS")
print("==============================================")


for road in ROADS:

    folder = os.path.join(
        DATASET_PATH,
        road
    )


    if not os.path.exists(folder):

        print(
            f"ERROR: {folder}"
        )

        exit()


    images = get_images(
        folder
    )


    if not images:

        print(
            f"No images in {road}"
        )

        exit()


    road_images[
        road
    ] = images


    # First image

    first_image_path = os.path.join(
        folder,
        images[0]
    )


    first_image = cv2.imread(
        first_image_path
    )


    # Existing ROI

    roi = load_roi(
        road
    )


    if roi is not None:

        print(
            f"{road}: ROI loaded."
        )

        road_rois[
            road
        ] = roi


    else:

        print(
            f"{road}: Select ROI."
        )


        roi = select_roi(
            first_image,
            road
        )


        if roi is None:

            print(
                "Program stopped."
            )

            exit()


        save_roi(
            road,
            roi
        )


        road_rois[
            road
        ] = roi


# =========================================================
# READY
# =========================================================

print()
print("==============================================")
print("             TRAFFIC AI READY")
print("==============================================")

for road in ROADS:

    print(
        f"{road.upper()} : "
        f"{ROAD_TIMES[road]} seconds"
    )


print()
print("Press ESC to stop.")
print()


# =========================================================
# IMAGE INDEX
# =========================================================

image_index = 0


# =========================================================
# MAIN SIGNAL LOOP
# =========================================================

while True:

    for road in ROADS:

        # -----------------------------------------
        # Check images
        # -----------------------------------------

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


        # -----------------------------------------
        # LOAD IMAGE
        # -----------------------------------------

        image = cv2.imread(
            image_path
        )


        if image is None:

            continue


        # =================================================
        # SIGNAL START
        # =================================================

        signal_time = ROAD_TIMES[
            road
        ]


        signal_start = time.time()


        print()
        print()
        print("==============================================")
        print(
            f"🚦 {road.upper()} "
            f"GREEN SIGNAL"
        )
        print(
            f"Signal Time: "
            f"{signal_time} seconds"
        )
        print(
            f"Image: "
            f"{image_name}"
        )
        print("==============================================")


        # =================================================
        # CAPTURE + DETECT
        # =================================================

        capture_start = time.time()


        total, counts = detect_vehicles(
            image,
            road_rois[road]
        )


        processing_time = (
            time.time()
            - capture_start
        )


        # =================================================
        # REMAINING TIME
        # =================================================

        elapsed = (
            time.time()
            - signal_start
        )


        remaining = max(
            0,
            signal_time - elapsed
        )


        # =================================================
        # RESULT
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


        print(
            f"⏱ Remaining Signal Time: "
            f"{remaining:.2f}s"
        )


        print("----------------------------------------------")


        # =================================================
        # COUNTDOWN
        # =================================================

        while True:

            elapsed = (
                time.time()
                - signal_start
            )


            remaining = (
                signal_time
                - elapsed
            )


            if remaining <= 0:

                break


            # -----------------------------------------
            # Terminal countdown
            # -----------------------------------------

            print(
                f"\r{road.upper()} "
                f"TIME REMAINING: "
                f"{remaining:05.1f}s",
                end="",
                flush=True
            )


            # -----------------------------------------
            # ESC
            # -----------------------------------------

            key = (
                cv2.waitKey(50)
                & 0xFF
            )


            if key == 27:

                cv2.destroyAllWindows()

                print()
                print()
                print(
                    "ESC PRESSED."
                )

                exit()


        print()


        # =================================================
        # SIGNAL OVER
        # =================================================

        print()
        print(
            f"🔴 {road.upper()} "
            f"SIGNAL TIME OVER"
        )


        # =================================================
        # NEXT ROAD
        # =================================================

    # ---------------------------------------------
    # NEXT IMAGE SET
    # ---------------------------------------------

    image_index += 1


    # ---------------------------------------------
    # Restart dataset when finished
    # ---------------------------------------------

    maximum_images = max(
        len(road_images[road])
        for road in ROADS
    )


    if image_index >= maximum_images:

        print()
        print(
            "All images completed."
        )

        break


# =========================================================
# END
# =========================================================

cv2.destroyAllWindows()

print()
print("==============================================")
print("          TRAFFIC AI FINISHED")
print("==============================================")