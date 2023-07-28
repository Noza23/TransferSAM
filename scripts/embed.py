import numpy as np
import nibabel as nib
import torch
import cv2
from torchvision import transforms


def embed_image(
    image: np.array,
    segmentation: np.array,
    model,
    transform,
    device: torch.device,
    balance_seg: bool=True,
    max_size: int=400,
    batch_size: int=1
) -> np.array:
    """
    Generate embeddings for .nii images
    
    Parameters:
        image (numpy.ndarray): Image to embedd.
        segmentation (numpy.ndarray): Segmentation, needed to balance blank/non-blank ratio.
        model (torch.nn.Module): SAM.
        transform (ResizeLongestSide): Transform to reshape images into the shape of model input.
        device (torch.device): Device.
        balance_seg (bool): Flag, whether to balance blank/non-blank 50:50. Defaults to True.
        max_size (int): Maximum number of slices to embedd. Positive samples are prioritised. Defaults to 400.
        batch_size (int): How many Slices to embedd in a single forward-pass. 5 Slices already fill 15GB GPU memory. Defaults to 1.
    
    Returns:
        (numpy.ndarray): Batch of Embeddings of shape Bx256x64x64
    """

    transform_resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1")

    image_scaled = cv2.convertScaleAbs(image)
    segmentation = cv2.convertScaleAbs(segmentation)

    # Rescale if not Sx512x512
    if image_scaled.shape[1:] != (512, 512):
        image_scaled = (np.vstack([transform_resize(img) for img in image_scaled]) * 255).astype("uint8") 
        print(f"Image has been rescaled to {image_scaled.shape}")
    if segmentation.shape[1:] != (512, 512):
        segmentation = (np.vstack([transform_resize(se) for se in segmentation]) * 255).astype("uint8")
        print(f"Segmentation has been rescaled to {segmentation.shape}")

    # identify positive segmentations
    where_pos, _, _ = np.where(segmentation > 0)
    start, stop, last = np.min(where_pos), np.max(where_pos), segmentation.shape[0]
    n_positive = stop - start + 1

    if n_positive > max_size:
        indices = np.arange(start, start + max_size)
    else:
        indices_negative = np.concatenate((np.arange(start), np.arange(stop + 1, last)))
        if balance_seg:
            # Balance Negative samples
            sampled = np.random.choice(
              indices_negative,
              size=min(n_positive, min(max_size, last)-n_positive),
              replace=False
            )
        else:
            # Don't balance just fill max_size randomly
            sampled = np.random.choice(
              indices_negative,
              size=min(max_size, last)-n_positive,
              replace=False
            )

        # Combine positive and negative segmentations.
        indices = np.sort(np.concatenate((np.arange(start, stop + 1), sampled)))
        is_blank = ~np.isin(indices, np.unique(where_pos))

    embeddings = []
    if model.device != device:
        model.to(device)

    # Split job into batches
    batches = np.array_split(indices, np.ceil(len(indices) / batch_size))
    for i, batch in enumerate(batches):
        if len(batch) == 1:
            # convert slice to RGB
            image_rgb = cv2.cvtColor(image_scaled[batch], cv2.COLOR_GRAY2RGB)
            # Reshape to 1 x 3 x 1024 x 1024 as model expects
            # Expects a numpy array with shape HxWxC in uint8 format.
            image_resized = transform.apply_image(image_rgb)
            # Convert to torch and move to device
            input_image_torch = torch.as_tensor(image_resized, device=device)
            # Reshape to BxCxHxW
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        else:
            # convert slices to RGB
            image_rgb = [cv2.cvtColor(sl, cv2.COLOR_GRAY2RGB) for sl in image_scaled[batch]]
            # Reshape slices: list containing 1 x 1024 x 1024 x 3
            image_resized = [transform.apply_image(sl).transpose(2, 0, 1)[None, :, :, :] for sl in image_rgb]
            # Convert slices to torch and move to device: B x 1024 x 1024 x 3
            input_image_torch = torch.as_tensor(np.vstack(image_resized), device=device).contiguous()
        # Preprocess: Normalize pixel values and pad to a square input.
        input_image_preporcessed = model.preprocess(input_image_torch)
        
        # Compute Embedding without grad
        with torch.no_grad():
            input_image_embedded = model.image_encoder(input_image_preporcessed)
            embedding = input_image_embedded.cpu().numpy()
        embeddings.append(embedding)
        if ((i+1) * batch_size) % 40 == 0: 
          print(f"{(i + 1) * batch_size} slices have been embedded")
        input_image_embedded = None
        torch.cuda.empty_cache()
    return np.vstack(embeddings), indices, is_blank


if __name__ == "__main__":
    import sys
    import os
    import argparse
    import gc
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    os.chdir(parent_dir)
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    
    # Set Random-Seed for reproducibility    
    np.random.seed(42)
    # Set-up parser
    parser = argparse.ArgumentParser(description="Embedding .nii.gz images")
    # Add arguments
    parser.add_argument("-i", type=str, help="Path to the dataset directory", required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to the SAM model checkpoint.", required=True)
    parser.add_argument("--max_size", type=int, help="Maximum Number of slices to embedd from a single .nii", required=True)
    parser.add_argument("--batch_size", type=int, help="Number of slices to embedd in a single forward-pass Note: 5 Slices already fill 15GB memory", required=True)
    parser.add_argument("--case_start", type=int, help="Case to start from", required=True)
    parser.add_argument("--case_end", type=int, help="Case to end at [excluded]", required=True)

    # Collect arguments
    args = parser.parse_args()
    path = args.i
    checkpoint = args.checkpoint
    max_size = args.max_size
    batch_size = args.batch_size
    case_start = args.case_start
    case_end = args.case_end
    
    print("####### arguments #######")
    print(f"path: {path}\ncheckpoint: {checkpoint}\nmax_size: {max_size}\nbatch_size: {batch_size}\ncase_start: {case_start}\ncase_end: {case_end}\n")
    print("#######   device  #######")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Computations are done on {device}")
    datapoints = sorted([f for f in os.listdir(path) if not f.startswith('.')])
    print("#######   Cases  #######")
    print(f"From Total of {len(datapoints)} cases we process starting from {case_start} including {case_end - 1}\n")
    datapoints = datapoints[case_start:case_end]
    print(f"Number of cases: {len(datapoints)}")

    # Model
    sam_base = sam_model_registry["vit_b"](checkpoint)
    sam_transform = ResizeLongestSide(sam_base.image_encoder.img_size)
    
    # Embedding Loop
    for datapoint in datapoints:
        if "embedding.npz" in os.listdir(os.path.join(path, datapoint)):
            print(f"Embedding for {datapoint} already exists")
            continue
        print(f"Starting Embedding {datapoint}.")
        img_path = os.path.join(path, datapoint, "imaging.nii.gz")
        seg_path = os.path.join(path, datapoint, "segmentation.nii.gz")
        image = nib.load(img_path).get_fdata() # Sx512x512
        segmentation = nib.load(seg_path).get_fdata() # Sx512x512
        embeddings, indices, is_blank = embed_image(
            image,
            segmentation=segmentation,
            model=sam_base,
            transform=sam_transform,
            device=device,
            balance_seg=True,
            max_size=max_size,
            batch_size=batch_size
        )
        # Save in npz compressed format
        np.savez_compressed(
            os.path.join(path, datapoint, 'embedding.npz'),
            embeddings=embeddings,
            indices=indices,
            is_blank=is_blank
        )
        print(f"Embeddings size {embeddings.shape} succesfully saved for {datapoint} in {os.path.join(path, datapoint)}")
        embeddings, indices, is_blank, image, segmentation = None, None, None, None, None
        gc.collect()
        torch.cuda.empty_cache()