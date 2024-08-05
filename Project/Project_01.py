import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "Resources/12 (1).jpg"

# Read the image
image = cv2.imread(image_path)

def detect_faces(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Training Face Data using Cascade Classifier
    trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_img)

    print(face_coordinates)

    for coordinate in face_coordinates:
        (x, y, w, h) = coordinate
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    return image

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

print("------------------------------------------------------------")

print("FACE DETECTION USING HAAR CASCADE CLASSIFIER")

# Detect faces
x = detect_faces(image)

print("------------------------------------------------------------")

# Display the image with detected faces
plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
plt.show()