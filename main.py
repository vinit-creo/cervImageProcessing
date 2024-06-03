import cv2
from matplotlib import pyplot as plt



img = cv2.imread("/Users/vinit/Downloads/experiment_media/image_2.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(grad_x, grad_y)
direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)
magnitude = cv2.convertScaleAbs(magnitude)
cv2.imshow('Gradient Magnitude', magnitude)
cv2.imshow('Gradient Direction', direction)


plt.subplot(1, 1, 1)
plt.imshow(img_gray)
plt.show()
