from matplotlib import pyplot as plt
import numpy as np

# Taken from SAM demo
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    return None

# Taken from SAM demo
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Taken from SAM demo
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Transform bbox returned by the SAM auto segmentation
def transform_bbox(bbox):
    # bbox returned by auto segmentation has shape XYWH
    x0, y0 = bbox[0], bbox[1]
    x1, y1 = x0 + bbox[2], y0 + bbox[3]
    return np.array([x0, y0, x1, y1])

def visualize3d_scan(image, segmentation, step_size=20):
    # Takes 3d image RGB numpy array of shape SxCxHxW
    # And segmentation: numpy array of shape SxHxW
    image = image.transpose((0, 2, 3, 1))
    indices = np.arange(image.shape[0], step=step_size)
    for index, img, s in zip(indices, image[indices, :, :], segmentation[indices, :, :]):
        # Create the figure and subplots
        fig, axes = plt.subplots(1, 2)
        # Plot data in the first subplot
        axes[0].imshow(img)
        axes[0].set_title('CEP scan: ' + str(index))
        # Remove axis
        axes[0].axis("off")
        # Plot data in the second subplot
        axes[1].imshow(s)
        axes[1].set_title('Segmentation')
        axes[1].axis("off")
        # Adjust spacing betweensubplots
        plt.tight_layout()
        # Display the plot
        plt.show()

def visualize_segment(image, segment, box=[0, 0, 0, 0], slice:int=20):
    # Visualize single slice
    # Takes 3d image RGB numpy array of shape SxCxHxW
    # And segmentation: numpy array of shape SxHxW
    # Box to draw
    # Slice number
    image = image.transpose((0, 2, 3, 1))
    #segment = segment.unsqueeze(3)
    plt.figure(figsize = (10,10))
    plt.imshow(image[slice, :, :])
    plt.axis("off")
    plt.imshow(segment[slice, : ,:], alpha=0.5)
    show_box(box, plt.gca())
    plt.show()