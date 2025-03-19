def combine_overlapping_bboxes(bboxes: list[list[int]]) -> list[list[int]]:
    if not bboxes:
        return []

    local_bboxes = bboxes.copy()

    # Helper function to check overlap
    def is_overlapping(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Check if the bounding boxes overlap
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    # Helper function to merge two bounding boxes
    def merge_bboxes(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        new_x = min(x1, x2)
        new_y = min(y1, y2)
        new_w = max(x1 + w1, x2 + w2) - new_x
        new_h = max(y1 + h1, y2 + h2) - new_y

        return [new_x, new_y, new_w, new_h]

    merged = True
    while merged:
        merged = False
        new_bboxes = []
        while local_bboxes:
            current = local_bboxes.pop(0)
            has_merged = False
            for i in range(len(local_bboxes)):
                if is_overlapping(current, local_bboxes[i]):
                    # Merge and break to restart the process
                    current = merge_bboxes(current, local_bboxes.pop(i))
                    has_merged = True
                    merged = True
                    break
            new_bboxes.append(current)

        local_bboxes = new_bboxes

    clean_small_bboxes = [
        bbox for bbox in local_bboxes if bbox[2] > 50 and bbox[3] > 50
    ]

    return clean_small_bboxes
