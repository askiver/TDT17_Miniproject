import cv2
import matplotlib.pyplot as plt

# define image path
image_path = 'data/train_and_val/images/combined_image_194_png.rf.9aa3244e3a3508fb7adbd0920de3eed3.jpg'
image = cv2.imread(image_path)
image_height, image_width, image_channels = image.shape

# See how the different channels look
blue, green, red = cv2.split(image)

for i, channel in enumerate([blue, green, red]):
    cv2.imwrite(f'color_image_{i}.jpg', channel)


# Visualize each channel
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(red, cmap='Reds'), plt.title('Red Channel')
plt.subplot(1, 3, 2), plt.imshow(green, cmap='Greens'), plt.title('Green Channel')
plt.subplot(1, 3, 3), plt.imshow(blue, cmap='Blues'), plt.title('Blue Channel')
plt.show()

print(f'Image channels: {image_channels}')

box_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
with open(box_path, 'r') as file:
    labels = file.read().splitlines()

for label in labels:
    class_id, x_center, y_center, width, height = map(float, label.strip().split())

    # Convert normalized values to pixel coordinates
    x_center_pixel = int(x_center * image_width)
    y_center_pixel = int(y_center * image_height)
    box_width_pixel = int(width * image_width)
    box_height_pixel = int(height * image_height)

    x_min = int(x_center_pixel - box_width_pixel / 2)
    y_min = int(y_center_pixel - box_height_pixel / 2)
    x_max = int(x_center_pixel + box_width_pixel / 2)
    y_max = int(y_center_pixel + box_height_pixel / 2)

    # Draw bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Add class ID text near the box
    cv2.putText(image, f'Class {int(class_id)}', (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display the image
cv2.imshow('Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('box_image.jpg', image)