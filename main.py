import cv2
from matplotlib import pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

img = cv2.imread("assets/image1.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = cv2.magnitude(grad_x, grad_y)
magnitude = cv2.convertScaleAbs(magnitude)

transform = transforms.Compose([
    transforms.ToTensor()
])

tensor = transform(img_gray) # tensors are printed here
demo_array = np.moveaxis(tensor.numpy() *255, 0,-1)
print(Image.fromarray(demo_array.astype(np.uint8)))
plt.subplot(1, 1, 1)
plt.imshow(img_gray)
plt.show()
