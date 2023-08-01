import numpy as np
import torch
from torchvision import transforms
import cv2
from scipy.ndimage import label
import TransferSAM.utils.visualize_kits as vis
from .utils.create_prompt import draw_boxes
from .utils.transforms import ResizeLongestSide
from matplotlib import pyplot as plt

class Predictor:
    """
    Predictor class, that generates predictions for .nii images.

    Attributes:
        sam_base (torch.nn.Module): SAM Base model for generating Embeddings.
        decoder_ROI (torch.nn.Module): Fine-Tuned decoder for ROI predictions.
        decoder_tumor (torch.nn.Module): Fine-Tuned decoder for tumor predictions.
        decoder_cyst (torch.nn.Module): Fine-Tuned decoder for cyst predictions.
        threshold (float): Threshold in the prediction of ROI. Defaults to 0.
        seed (int): Seed for reproducible predictions. Defaults to 42.
        device (str): String name of the desired Device. Defaults to cpu.

    Methods:
        __init__: Initializes a Predictor instance.
        generate_embedding: Generates Embeddings for a single slice of shape HxW.
        predict_ROI: Predicts ROI.
        postprocess_ROI: Post-Processes ROI to remove noisy pixels.
        predict_Stage2: Predicts Tumor and Cyst and returns final single prediction mask.
        predict_complete: Combines whole predicting pipeline.
        predict_case: Predicts a complete case of shape SxHxW.
        visualize_result: Visualizes Imagging with final predictions.
    """
    def __init__(
            self,
            sam_base,
            decoder_ROI,
            decoder_tumor=None,
            decoder_cyst=None,
            threshold: float=0,
            seed: int=42,
            device=None
        ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.sam_base = sam_base.to(self.device)
        self.decoder_ROI = decoder_ROI.to(self.device)
        self.decoder_tumor = decoder_tumor.to(self.device)
        self.decoder_cyst = decoder_cyst.to(self.device)
        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
        self.sam_transform = ResizeLongestSide(sam_base.image_encoder.img_size)
        # Fix-Box for ROI prediction
        self.box_roi = np.array([0, 100, 512, 450])
        # threshold for sigmoid scores
        self.threshold = threshold
        # Setting seed for reproduciblity
        self.seed = seed

    def generate_embedding(self, slc):
        """
        Generate Embedding for a single slice.
        
        Parameters:
            slc (numpy.ndarray): Slice of shape HxW

        Returns:
            (torch.Tensor): Embedding of shape: 1x256x64x64
        """
        np.random.seed(self.seed)
        slc = cv2.convertScaleAbs(slc)
        if slc.shape != (512, 512):
            slc = self.transform(slc).numpy()
            print(f"Image has been rescaled to {slc.shape}")
        # Convert to RGB
        slc = cv2.cvtColor(slc, cv2.COLOR_GRAY2RGB)
        # Reshape to 1024x1024 model input size
        slc = self.sam_transform.apply_image(slc)
        slc_torch = torch.as_tensor(slc, device=self.device)
        # Permute to shape BxCxHxW
        slc_torch = slc_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # Preprocess: Normalize pixel values and pad to a square input
        slc_torch = self.sam_base.preprocess(slc_torch)
        # Embedd
        with torch.no_grad():
            slc_embedded = self.sam_base.image_encoder(slc_torch)
            embedding = slc_embedded
        return embedding
    
    def predict_ROI(self, embd):
        """
        Generate ROI prediction.
        
        Parameters:
            embd (torch.Tensor): Embedding of shape: 1x256x64x64.

        Returns:
            (numpy.ndarray, torch.Tensor): Bool-mask: 512x512 and Tensor of sigmoid scores: 1x1x512x512.
        """
        np.random.seed(self.seed)
        image_pe = self.sam_base.prompt_encoder.get_dense_pe()
        box_reshaped = self.sam_transform.apply_boxes(
            self.box_roi,
            original_size=(512, 512)
        )
        # Prompt Embeddings
        sparse_embeddings, dense_embeddings = self.sam_base.prompt_encoder(
            points=None,
            boxes=torch.as_tensor(box_reshaped, device=self.device), # Bx4
            masks=None
        )
        masks, _ = self.decoder_ROI(
            image_embeddings=embd, # 1x256x64x64
            image_pe=image_pe, # 1x256x64x64
            sparse_prompt_embeddings=sparse_embeddings, # 1x2x256
            dense_prompt_embeddings=dense_embeddings, # 1x256x64x64
            multimask_output=False
        )
        mask_pred = self.sam_base.postprocess_masks(
            masks,
            input_size=(1024, 1024),
            original_size=(512, 512)
        )
        mask_pred = mask_pred.detach().cpu()
        binary_mask = torch.squeeze(mask_pred > self.threshold).numpy()
        return binary_mask, mask_pred
    
    def postprocess_ROI(self, roi, min_component_size=5):
        """
        Postprocesses ROI prediction.
        Parameters:
            roi (numpy.ndarray): Bool-mask: 512x512.
            min_component_size (int): Minimum size of cluster to keep.

        Returns:
            (numpy.ndarray): Bool-mask where clusters with smaller than min_component_size are removed.
        """
        # Filter out noise pixels with connected component analysis
        labels, num_labels = label(roi) # Identify clusters
        component_sizes = [(labels == label_id).sum() for label_id in range(num_labels + 1)] # Calculate cluster sizes
        filtered_labels = [label_id for label_id, size in enumerate(component_sizes) if size >= min_component_size] # Filter out all clusters smaller min_size
        # selected_components = np.argsort(np.array(component_sizes)[filtered_labels])[::-1][1:min(3, len(component_sizes))] # Select at most 2 components excluded background class
        if len(filtered_labels) == 0:
          return np.zeros((512, 512), dtype=np.uint8)
        return np.isin(labels, np.array(filtered_labels)[1:]) # exclude background class
        # return np.isin(labels, np.array(filtered_labels)[selected_components])
    
    def predict_Stage2(self, embed, ROI):
        """
        Predict Tumor and Cyst and generate final prediction mask.

        Parameters:
            embed (torch.Tensor): Embedding of shape: 1x256x64x64.
            ROI (numpy.ndarray): Bool-mask of ROI prediction: 512x512.
        Returns:
            (numpy.ndarray, torch.Tensor): Box prompts: (NumberOfInstance)x4, Final segmentation mask: 512x512.
        """
        np.random.seed(self.seed)
        image_pe = self.sam_base.prompt_encoder.get_dense_pe()
        sep_point, boxes = draw_boxes(ROI)
        ROI = torch.tensor(ROI)
        boxes_reshaped = self.sam_transform.apply_boxes(
            boxes,
            original_size=(512, 512)
        )
        # Prompt Embeddings
        sparse_embeddings, dense_embeddings = self.sam_base.prompt_encoder(
            points=None,
            boxes=torch.as_tensor(boxes_reshaped, device=self.device), # Bx4
            masks=None
        )
        if sep_point is not None:
            print("There are 2 Kidneys in the slice")
            embed = torch.tile(embed, (2, 1, 1, 1))
        else:
            print("There is 1 Kidney in the slice")

        # Tumor prediction
        masks_tumor, _ = self.decoder_tumor(
            image_embeddings=embed, # Ix256x64x64
            image_pe=image_pe, # 1x256x64x64
            sparse_prompt_embeddings=sparse_embeddings, # Ix2x256
            dense_prompt_embeddings=dense_embeddings, # Ix256x64x64
            multimask_output=False
        )
        # Cyst prediction
        masks_cyst, _ = self.decoder_cyst(
            image_embeddings=embed, # Ix256x64x64
            image_pe=image_pe, # 1x256x64x64
            sparse_prompt_embeddings=sparse_embeddings, # Ix2x256
            dense_prompt_embeddings=dense_embeddings, # Ix256x64x64
            multimask_output=False
        )
        masks_tumor_pred = self.sam_base.postprocess_masks(
            masks_tumor,
            input_size=(1024, 1024),
            original_size=(512, 512)
        )
        masks_cyst_pred = self.sam_base.postprocess_masks(
            masks_cyst,
            input_size=(1024, 1024),
            original_size=(512, 512)
        )
        # Combined
        masks_tumor_pred = masks_tumor_pred.detach().cpu()
        masks_cyst_pred = masks_cyst_pred.detach().cpu()
        if sep_point is None:
            roi_mask = ROI[None, None, :, :].to(dtype=torch.float32)
        else:
            roi_mask_left, roi_mask_right = ROI.clone(), ROI.clone()
            roi_mask_left[:, sep_point:] = 0.
            roi_mask_right[:, 0:sep_point] = 0.
            roi_mask = torch.stack([roi_mask_left, roi_mask_right]).unsqueeze(1)
        roi_mask[torch.where(roi_mask > 0)] = 1e-9
        roi_mask[torch.where(roi_mask < 0)] = -1e-9
        _, prediction_mask = torch.max(torch.stack([torch.zeros_like(masks_tumor_pred), roi_mask, masks_tumor_pred, masks_cyst_pred], dim=1), dim=1)
        prediction_mask = torch.sum(prediction_mask, dim=0)
        return boxes, torch.squeeze(prediction_mask).to(dtype=torch.uint8)
    
    def predict_complete(self, slc, i: int=0):
        """
        Given a slice of shape HxW runs complete prediction loop.

        Parameters:
            slc (numpy.ndarray): Slice of shape HxW.
            i (numpy.ndarray): Indicator for slice.
        Returns:
            (torch.Tensor): Final segmentation mask of shape 512x512.
        """
        if len(slc.shape) != 2:
            raise ValueError(f"Slice should have a shape of HxW")
        embed = self.generate_embedding(slc)
        roi, _ = self.predict_ROI(embed)
        # If all background dont do second step:
        if np.all(~roi):
            print(f"Slice {i} was predicted as Blank")
            return roi.astype("uint8")
        roi = self.postprocess_ROI(roi, min_component_size=5)
        if np.all(~roi):
            print(f"Slice {i + 1} was predicted as Blank")
            return roi.astype("uint8")
        _, stage2_pred = self.predict_Stage2(embed, roi)
        return stage2_pred

    def visualize_result(self, image_slice, stage2_pred, boxes):
        """
        Visualize Final result.

        Parameters:
            image_slice (numpy.ndarray): Single slice of shape HxW
            stage2_pred (torch.Tensor): Final segmentation mask of shape 512x512.
        Returns:
            (None)
        """
        if image_slice.shape != (512, 512):
            image_slice = self.transform(image_slice).numpy()
            print(f"Image has been rescaled to {image_slice.shape}")
        plt.imshow(image_slice)
        # visualize boxes
        [vis.show_box(box, plt.gca()) for box in boxes]
        vis.show_mask(stage2_pred[None, None, :, :], plt.gca())
    
    def predict_case(self, imagging):
        """
        Predicts a complete case of shape SxHxW.

        Parameters:
            imagging (numpy.ndarray): Imagging of shape SxHxW
        Returns:
            (numpy.ndarray): Batch of Final segmentation masks of shape Bx512x512.
        """
        prediction_masks = [self.predict_complete(slc, i) for i, slc in enumerate(imagging)]
        return np.stack(prediction_masks)

