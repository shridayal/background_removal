from rembg import remove, new_session
import cv2
import numpy as np
import onnxruntime # Import onnxruntime

# Load models
session = new_session('u2net')

# Read images
image = cv2.imread('/content/WhatsApp Image 2025-09-13 at 22.44.41.jpeg')
background = cv2.imread('/content/plain-pastel-color-84co1sdwelht9w24.jpg')

# Check if images were loaded successfully
if image is None:
    print("Error: Could not load 'person.jpg'. Please ensure the file exists in the correct directory.")
elif background is None:
    print("Error: Could not load 'new_background.jpg'. Please ensure the file exists in the correct directory.")
else:
    # Remove background
    fg_removed = remove(image, session=session)

    # Convert to RGBA and extract alpha
    fg_rgba = cv2.cvtColor(fg_removed, cv2.COLOR_BGR2BGRA)
    alpha = fg_rgba[:, :, 3]
    alpha_3channel = cv2.merge([alpha, alpha, alpha])

    # Resize background
    bg_resized = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Composite
    foreground = image * (alpha_3channel / 255.0)
    background = bg_resized * (1 - alpha_3channel / 255.0)
    result = foreground + background

    cv2.imwrite('output.jpg', result)
    print("Background removed and image saved as 'output.jpg'")