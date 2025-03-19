from pathlib import Path
import numpy as np
import cv2


def segmentation_mask_bboxes(
    mask: np.ndarray, preview_image: bool = False, save_image_in_path: str = None
):
    """
    Finds bounding boxes of segmentation masks and optionally displays or saves the image with drawn boxes.

    :param mask: Segmentation mask as a NumPy array (grayscale or binary).
    :param preview_image: If True, displays the image with bounding boxes.
    :param save_image_in_path: If provided, saves the image with bounding boxes at the given path.
    :return: List of bounding boxes [(x, y, w, h), ...].
    """
    # Convert mask to binary
    binary_mask = (mask > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get bounding boxes
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Create an image to display bounding boxes
    mask_with_boxes = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Preview the image if enabled
    if preview_image:
        cv2.imshow("Bounding Boxes", mask_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save the image if a path is provided
    if save_image_in_path:
        save_image_in_path = Path(save_image_in_path)
        if not save_image_in_path.parent.exists():
            raise ValueError(
                f"Parent directory {save_image_in_path.parent} does not exist"
            )
        cv2.imwrite(save_image_in_path, mask_with_boxes)

    return bounding_boxes


# TODO: add params and typings
def largest_combined_bbox(mask: np.ndarray):
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    min_x, min_y, max_x, max_y = mask.shape[1], mask.shape[0], 0, 0

    for x, y, w, h in bounding_boxes:
        if w < 100 and h < 100:
            continue
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    width = max_x - min_x
    height = max_y - min_y

    return [min_x, min_y, width, height]
