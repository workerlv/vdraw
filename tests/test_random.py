import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vdraw import draw, find
import cv2

image = cv2.imread("/workspace/tests/original_image.jpg")
mask = cv2.imread("/workspace/tests/segmentation_mask_target.png", cv2.IMREAD_GRAYSCALE)

draw.overlay_binary_segmentation_mask(
    image, mask, save_image_in_path="tests/results/overlayed_image.jpg"
)

find.bounding_boxes_of_binary_segmentation_mask(
    mask, save_image_in_path="tests/results/bbox_image.jpg"
)
