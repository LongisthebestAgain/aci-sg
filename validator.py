def do_bboxes_intercept(bbox1, bbox2):
    """
    Determine if two bounding boxes intercept.

    Parameters:
    bbox1, bbox2: List or tuple of four elements [x1, y1, x2, y2]
                  where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
    bool: True if the bounding boxes intercept, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Check if one rectangle is on left side of the other
    if x1_max < x2_min or x2_max < x1_min:
        return False

    # Check if one rectangle is above the other
    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True

