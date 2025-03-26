import numpy as np


def combined_crop(
    img: np.ndarray,
    mask: np.ndarray,
    bounding_box: list[int],
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    bb_x1, bb_y1, bb_width, bb_height = bounding_box

    bb_center_x = bb_x1 + bb_width // 2
    bb_center_y = bb_y1 + bb_height // 2

    aspect_ratio = crop_width / crop_height

    if crop_width > bb_width:
        crop_height = int(crop_width / aspect_ratio)
    elif crop_height > bb_height:
        crop_width = int(crop_height * aspect_ratio)

    if crop_width > img.shape[1]:
        crop_width = img.shape[1]
        crop_height = int(crop_width / aspect_ratio)

    if crop_height > img.shape[0]:
        crop_height = img.shape[0]
        crop_width = int(crop_height * aspect_ratio)

    x1 = bb_center_x - crop_width // 2
    y1 = bb_center_y - crop_height // 2

    x1 = max(0, min(x1, img.shape[1] - crop_width))
    y1 = max(0, min(y1, img.shape[0] - crop_height))

    cropped_img = img[y1 : y1 + crop_height, x1 : x1 + crop_width]
    cropped_mask = mask[y1 : y1 + crop_height, x1 : x1 + crop_width]

    return cropped_img, cropped_mask
