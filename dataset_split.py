import os
import shutil

# ==============================
# GIVE YOUR FOLDER PATH HERE
# ==============================
source_folder = r"D:\Traffic Ai\archive\IITM-HeTra_v2\Dataset-1\images"

# Output folder
output_folder = os.path.join(source_folder, "Divided")

# Image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

# ==============================
# GET ALL IMAGES
# ==============================
images = [
    file for file in os.listdir(source_folder)
    if file.lower().endswith(image_extensions)
]

# Sort images for consistent distribution
images.sort()

print("Total images found:", len(images))

# ==============================
# CREATE 4 × 4 FOLDERS
# ==============================
folders = []

for i in range(1, 5):

    # dataset-1, dataset-2, dataset-3, dataset-4
    dataset_folder = os.path.join(
        output_folder,
        f"dataset-{i}"
    )

    for j in range(1, 5):

        # road1, road2, road3, road4
        road_folder = os.path.join(
            dataset_folder,
            f"road{j}"
        )

        os.makedirs(road_folder, exist_ok=True)

        folders.append((road_folder, j))

# ==============================
# DISTRIBUTE AND RENAME IMAGES
# ==============================

for index, image in enumerate(images):

    # Select one of the 16 folders
    folder_index = index % 16

    destination_folder, road_number = folders[folder_index]

    # Get original extension
    extension = os.path.splitext(image)[1]

    # Image number inside the road folder
    image_number = index // 16 + 1

    # New image name
    new_name = f"road{road_number} image {image_number}{extension}"

    # Source and destination
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(
        destination_folder,
        new_name
    )

    # Copy image
    shutil.copy2(source_path, destination_path)

print("\nDone!")
print("Images have been divided into 4 datasets × 4 roads.")
print("Images have also been renamed.")