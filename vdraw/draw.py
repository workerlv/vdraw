from pathlib import Path
import numpy as np
import cv2


def overlay_binary_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color=(0, 255, 0),
    alpha=0.6,
    preview_image=False,
    save_image_in_path=None,
):
    """
    Overlays a segmentation mask on an RGB image with a given color and transparency.

    :param image: RGB image as a NumPy array.
    :param mask: Segmentation mask as a NumPy array (grayscale or binary, same spatial size as image).
    :param color: RGB color for the overlay (default is green).
    :param alpha: Transparency level of the overlay (default is 0.6).
    :param preview_image: Whether to display the overlayed image in a window (default is False).
    :param save_image_in_path: Path to save the overlayed image (default is None).
    :return: Image with overlayed mask.
    """
    # Check if image and mask are valid
    if image is None or mask is None:
        raise ValueError("Image or mask is None")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same spatial dimensions")

    # Ensure the mask is binary (0 and 1)
    mask = (mask > 0).astype(np.uint8)

    # Create a color mask
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[mask == 1] = color

    # Blend the images
    overlayed_image = cv2.addWeighted(color_mask, alpha, image, 1 - alpha, 0)

    if preview_image:
        cv2.namedWindow("Overlayed Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Overlayed Image", overlayed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_image_in_path:
        save_image_in_path = Path(save_image_in_path)
        if not save_image_in_path.parent.exists():
            raise ValueError(
                f"Parent directory {save_image_in_path.parent} does not exist"
            )
        cv2.imwrite(save_image_in_path, overlayed_image)

    return overlayed_image
