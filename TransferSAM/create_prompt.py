import numpy as np
def identify_box(batched_segmentation, prob: float=0.2):
    # Takes batched segmentation slices of shape BxHxW
    # prob: probability to sample completly random box
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
            # add perturbation to bounding box coordinates
            x_delta, y_delta = x_max - x_min, y_max - y_min
            # Taken from official SAM Paper
            x_min = np.round(max(0, x_min - min(np.abs(np.random.normal(scale=x_delta * 0.1)), 20)), 3)
            x_max = np.round(min(W, x_max + min(np.abs(np.random.normal(scale=x_delta * 0.1)), 20)), 3)
            y_min = np.round(max(0, y_min - min(np.abs(np.random.normal(scale=y_delta * 0.1)), 20)), 3)
            y_max = np.round(min(H, y_max + min(np.abs(np.random.normal(scale=y_delta * 0.1)), 20)), 3)
            box = [x_min, y_min, x_max, y_max]
        boxes.append(box)
        
    # Convert list of boxes to np.array
    boxes = np.array(boxes)
    
    # Identify blanks
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
            # Position
            max_pos_x, max_pos_y = W - x_delta, H - y_delta
            rand_x = np.random.randint(0, max_pos_x + 1)
            rand_y = np.random.randint(0, max_pos_y + 1)
            # Random box
            boxes[blank] = [rand_x, rand_y, rand_x + x_delta, rand_y + y_delta]
    return boxes

