from rembg import remove, new_session
import cv2
import numpy as np
import os

session = new_session('u2net')

background_dir = "data/backgrounds"
image_dir = "data/images"
output_dir = "data/outputs"
os.makedirs(output_dir, exist_ok=True)

backgrounds = []
background_files = []
for file in sorted(os.listdir(background_dir)):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        bg = cv2.imread(os.path.join(background_dir, file))
        if bg is not None:
            backgrounds.append(bg)
            background_files.append(file)

def replace_background(image_path, bg_index, output_path="output.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return
    if bg_index < 0 or bg_index >= len(backgrounds):
        print("Invalid background index")
        return
    background = backgrounds[bg_index]
    fg_removed_bytes = remove(image, session=session)
    fg_rgba = cv2.imdecode(np.frombuffer(fg_removed_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    alpha = fg_rgba[:, :, 3]
    alpha_3channel = cv2.merge([alpha, alpha, alpha])
    bg_resized = cv2.resize(background, (image.shape[1], image.shape[0]))
    foreground = image.astype(float) * (alpha_3channel.astype(float) / 255.0)
    background = bg_resized.astype(float) * (1 - alpha_3channel.astype(float) / 255.0)
    result = cv2.add(foreground, background).astype(np.uint8)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")

for file in sorted(os.listdir(image_dir)):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(image_dir, file)
        print(f"\nImage: {file}")
        for i, bg_file in enumerate(background_files):
            print(f"{i}: {bg_file}")
        try:
            choice = int(input("Choose background index: "))
        except:
            print("Invalid input, skipping")
            continue
        output_path = os.path.join(output_dir, f"out_{file}")
        replace_background(input_path, choice, output_path)
