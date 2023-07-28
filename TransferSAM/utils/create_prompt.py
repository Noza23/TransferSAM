import numpy as np
def identify_box(batched_segmentation, prob: float=0.2, ROI=False):
    """
    Identify the box prompts given batch of segmentations.
    
    Parameters:
        batched_segmentation (numpy.ndarray): Batch of segmentation of shape BxHxW.
        prob (float): Probability of sampling a completly random box. Defaults to 0.2.
        ROI (bool): Flag to indicate whether prompt is sampled for ROI training. Defaults to False.

    Returns:
        (numpy.ndarray): Batch of box prompts.
    """
    if ROI:
        box = np.array([0, 100, 512, 450])
        return np.tile(box, (len(batched_segmentation), 1))
    boxes = []
    for segmentation in batched_segmentation:
        # Get bounding box size
        H, W = segmentation.shape
        # If Blank slice: mark with [-1, -1, -1, -1]
        if np.count_nonzero(segmentation) == 0:
            box = [-1, -1, -1, -1]
        else:
            # If non-blank slice
            y_indices, x_indices = np.where(segmentation > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_delta, y_delta = x_max - x_min, y_max - y_min
            # Add perturbation to bounding box coordinates
            x_min = np.round(max(0, x_min - min(np.abs(np.random.normal(scale=x_delta * 0.1)), 20)), 3)
            x_max = np.round(min(W, x_max + min(np.abs(np.random.normal(scale=x_delta * 0.1)), 20)), 3)
            y_min = np.round(max(0, y_min - min(np.abs(np.random.normal(scale=y_delta * 0.1)), 20)), 3)
            y_max = np.round(min(H, y_max + min(np.abs(np.random.normal(scale=y_delta * 0.1)), 20)), 3)
            box = [x_min, y_min, x_max, y_max]
        boxes.append(box)
    boxes = np.array(boxes)

    blanks, _ = np.where(boxes == -1)
    non_blanks, _ = np.where(boxes != -1)
    for blank in blanks:
        # with probability prob sample completly random box for exploration
        choice = np.random.binomial(1, 1 - prob)
        if choice:
            # Randomly choose 2 positive boxes and caluclate average
            id_1, id_2 = np.random.choice(np.unique(non_blanks), 2, replace=False)
            boxes[blank] = (boxes[id_1] + boxes[id_2]) / 2
        else:
            # random box
            x_delta = np.random.randint(40, 250)
            y_delta = np.random.randint(40, 200)
            max_pos_x, max_pos_y = W - x_delta, H - y_delta
            rand_x = np.random.randint(0, max_pos_x + 1)
            rand_y = np.random.randint(0, max_pos_y + 1)
            # Random box
            boxes[blank] = [rand_x, rand_y, rand_x + x_delta, rand_y + y_delta]
    return boxes

def draw_boxes(seg):
    """
    Identify multiple instances on a single segmentation and generate multiple prompts accordingly.

    Parameters:
        seg (numpy.ndarray): segmentation of shape HxW.
    
    Returns:
        (tuple): A tuple containing seperation point between multiple instances and associated box prompts.
        If only one instance is present separation point is None.
    """
    _, x = np.where(seg > 0)
    # Blank
    if len(x) == 0:
        # Random box
        H, W = seg.shape
        x_delta = np.random.randint(40, 250)
        y_delta = np.random.randint(40, 200)
        max_pos_x, max_pos_y = W - x_delta, H - y_delta
        rand_x = np.random.randint(0, max_pos_x + 1)
        rand_y = np.random.randint(0, max_pos_y + 1)
        box = np.array([rand_x, rand_y, rand_x + x_delta, rand_y + y_delta]).reshape((1, 4))
        return None, box
    x = np.unique(x)
    # Check for multiple instances
    if x.max() - x.min() < 100:
        return None, identify_box(seg[None, :, :])
    diffs = np.abs(np.diff(x))
    if diffs.max() >= 40:
        # Two instances
        i = np.where(diffs  == diffs.max())[0][0]
        sep_point = int(((x[i] + x[i+1]) / 2))
        box_1 = identify_box(seg[None, :, 0:sep_point])
        box_2 = identify_box(seg[None, :, sep_point:])
        box_2[0, 0] += sep_point
        box_2[0, 2] += sep_point
        return sep_point, np.vstack([box_1, box_2])
    else:
        return None, identify_box(seg[None, :, :])
