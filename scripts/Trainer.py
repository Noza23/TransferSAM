import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the sys.path list to access segment_anything directory
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.chdir(parent_dir)
import gc
import re
import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from TransferSAM.KiTS_dataset import KiTSdata, ContinueTrainingSampler
import matplotlib.pyplot as plt
from TransferSAM.utils.create_prompt import  identify_box, draw_boxes
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from monai.losses import DiceLoss, GeneralizedDiceFocalLoss, DiceCELoss
import logging
import pandas as pd
import pickle

class CosineScheduler:
    """
    Class to generate cosine LR schedule with warm-up.

    Attributes:
        base_lr (float): Learning Rate after warm-up steps.
        max_update (int): Maximum number of updates.
        final_lr (float): Lower-bound of LR to reach.
        warmup_steps (int): Number of warm-up steps.
        warmup_begin_lr (float): Starting LR for warm-up steps.

    Methods:
        __init__: Initializes a new CosineScheduler instance.
        get_warmup_lr: Return warm-up LR.
        __call__: Call Protocol for each schedualer step.
    """
    def __init__(
        self,
        base_lr,
        max_update,
        final_lr=0,
        warmup_steps=0,
        warmup_begin_lr=0
    ):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * (
                1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)
            ) / 2
        return self.base_lr

class Trainer:
    """
    Trainer Class, main object in training process.

    Attributes:
        model (torch.nn.Module): Model architecture.
        train_data (torch.utils.data.DataLoader): Training DataLoader.
        valid_data (torch.utils.data.DataLoader): Validation DataLoader.
        n_epoch (int): Number of epochs to train for.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss (torch.nn.Module): Loss Function.
        extra_loss (torch.nn.Module): Extra Loss Function to compute. (Just for inference no grad computations.)
        device (torch.device): Device.
        batch_acc (int): Number of batches to accumulate in each optimization step.
        save_every (int): Save checkpoint in every int steps.
        validate_every (int): Validate model in every int steps.
        lr_schedule (dict): Dictionary of configurations from config.yaml file.
        transform (ResizeLongestSide): Transformation necessary for SAM.
        LOG_PATH (str): Output path for checkpoints and log files.
        logger (logging.Logger): training Logger.
        writter (torch.utils.tensorboard.SummaryWritter): Logging Loss scalars.
        tensorboard_steps (dict): Dictionary to track tensorboard steps.

    Methods:
        __init__: Initializes a new Trainer instance.
        _save_lrs: Save LR-schedule plot on disk.
        _generate_lr_shedule: Generates LR schedule based on configuration from config.yaml.
        _restructure_gt: Splits multiple-instances in one segmentation into multiple segmentations.
        _predict: Generates predictions for batch of datapoints.
        _run_batch: Runs a single Batch.
        _run_epoch: Runs a single training epoch. Possible to continue training from certain point.
        _save_checkpoint: Saves a model checkpoint and optimizer state.
        train: Starts training. Main Function of the training loop.
        continue_training: Continues training form a checkpoint.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        n_epoch: int, 
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        device: torch.device,
        batch_acc: int,
        save_every: int,
        validate_every: int,
        lr_schedule: dict,
        save_path: str,
    ):
        
        self.device = device
        self.model = model.to(device)
        self.transform  = ResizeLongestSide(model.image_encoder.img_size)
        self.train_data = train_data
        self.valid_data = valid_data
        self.n_epoch = n_epoch
        self.optimizer = optimizer
        self.loss = loss
        self.batch_acc = batch_acc
        self.save_every = save_every
        self.validate_every = validate_every
        self.lr_schedule = lr_schedule
        self.LOG_PATH = save_path
        self.tensorboard_steps = {
            "Avg batch Loss/train": 1,
            "Avg batch Loss/valid": 1,
            "Avg Epoch Loss/valid": 1,
            "Extra Loss": 1
        }
        if loss._get_name() == 'DiceLoss':
            self.extra_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.extra_loss = DiceLoss(reduction='mean', sigmoid=True)
             
        # Configure logger
        logger_train = logging.getLogger("train_log")
        logger_train.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(save_path, "training.log"))
        handler2.setFormatter(formatter)
        logger_train.addHandler(handler1)
        logger_train.addHandler(handler2)
        self.logger = logger_train
        
        # Tensorboard  SummaryWriter.
        self.writer = SummaryWriter(os.path.join(save_path, "logs"))

        self.logger.info("Trainer instance has been configured and has following attributes:")
        for arg, value in self.__dict__.items():
          if arg=="model":
            self.logger.info(f"{arg}: SAM_base")
          else:
            self.logger.info(f"{arg}: {value}")
        
        # Generate Learning rates
        self.lrs = self._generate_lr_shedule(lr_schedule)
        # Save learning rate shedule plot
        self._save_lrs()
        self.logger.info(f"LR shedule plot has been saved in {save_path}")

    def _save_lrs(self):
        """
        Save LR-schedule plot on disk.
        Returns:
            (None)
        """
        plt.plot(range(1, len(self.lrs) + 1), self.lrs)
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.savefig(os.path.join(self.LOG_PATH, 'LR_schedule.jpeg'))
        plt.clf()

    def _generate_lr_shedule(self, config: dict):
        """
        Generates LR schedule based on configuration from config.yaml.
        
        Parameters:
        config (dict): Configuration dict set in config.yaml.

        Returns:
            (list) : List of LRs.
        """
        for key, value in config.items():
            total_updates = self.n_epoch  * int(len(self.train_data.dataset) / self.batch_acc)
            if key:
                scheduler = CosineScheduler(
                    base_lr=value['warmup_lr'],
                    max_update=total_updates,
                    final_lr=value["min_lr"],
                    warmup_begin_lr=value['warmup_lr']/200,
                    warmup_steps=value['warmup_steps']
                )
                lrs = [scheduler(t) for t in range(total_updates + 100)]
            else:
                initial_lr = self.optimizer.param_groups[0]["lr"]
                lrs = [initial_lr for t in range(total_updates + 100)]
            return lrs

    def _restructure_gt(self, seg: torch.Tensor, sep_points: list):
        """
        Splits multiple-instances in one segmentation into multiple segmentations.
        Used in ["kidney", "tumor", "cyst"] training.
        
        Parameters:
        seg (torch.Tensor): Batch of segmentations of shape BxHxW.
        sep_points (list): List of points indicating how many instances present in each segmentation.

        Returns:
            (torch.Tensor) : Restructured batched segmentation masks of shape BxHxW.
        """
        i = 0
        for point in sep_points:
            if point is not None:
                seg[i][:, point:] = 0
                seg[i + 1][:, 0:point] = 0
                i += 2
            else:
                i += 1
        return seg

    def _predict(self, embeddings: torch.Tensor, gt_masks: torch.Tensor, train: bool=True):
        """
        Generates predictions for batch of datapoints.
        
        Parameters:
        embeddings (torch.Tensor): Batch of embeddings of shape 1xBx256x64x64.
        gt_masks (torch.Tensor): Batch of ground truth masks of shape 1xBx512x512.
        
        Returns:
            embeddings (torch.Tensor): Restructured batched embeddings of shape 1xBx256x64x64.
            gt_masks (torch.Tensor): Restructured batched GT masks of shape 1xBx512x512.
            masks_pred (torch.Tensor): Mask Predictions of shape Bx1x512x512
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        if self.train_data.dataset.instance_type == "ROI":
            # For ROI training generate fix sized box prompts.
            box_prompts = identify_box(gt_masks.squeeze(), prob=0.2, ROI=True)
        else:
            # Generate Box Prompts based on segmentationsand split multiple instances into multiple embedding-segmentaion pair.
            boxes = [draw_boxes(s)[1] for s in gt_masks.squeeze()]
            sep_points = [draw_boxes(s)[0] for s in gt_masks.squeeze()]
            ids = []
            _ = [ids.extend(len(b) * [i]) for i, b in enumerate(boxes)]
            box_prompts = np.vstack(boxes)
            embeddings = embeddings[0][ids][None, :, :, :, :]
            gt_masks = gt_masks[0][ids]
            sep_counter = sum([point is not None for point in sep_points])
            # Quick check.
            self.logger.info(f"ids have length: {len(ids)}, | {len(boxes) + sep_counter}")
            gt_masks = self._restructure_gt(gt_masks.clone(), sep_points)[None, :, :, :]
            predictions = {"kidney": 1, "tumor": 2, "cyst": 3}
            # Identify desired segements and remove the rest.
            gt_masks[torch.where(gt_masks != predictions[self.train_data.dataset.instance_type])] = 0
            # Select only positive samples
            slcs, _, _ = torch.where(gt_masks[0] == predictions[self.train_data.dataset.instance_type])
            gt_masks = gt_masks[0][slcs.unique()][None, :, :, :]
            embeddings = embeddings[0][slcs.unique()][None, :, :, :, :]
            box_prompts = box_prompts[slcs.unique()]
            # Turn segmentation into [0, 1] encoding.
            gt_masks.clip_(max=1)
            self.logger.info(f"gt_masks adjusted: {gt_masks.unique()}")
            self.logger.info(f"Embeddings have been reshaped to {embeddings.shape} according to boxes shape: {box_prompts.shape}")
            self.logger.info(f"gt_masks have been reshaped aswell: {gt_masks.shape}")
        
        # Reshape box to model input resolution[1024x1024].
        box_prompts_reshaped = self.transform.apply_boxes(
            box_prompts,
            original_size=(512, 512)
        )
        # Convert to torch and move to the device.
        box_prompts_torch = torch.as_tensor(
            box_prompts_reshaped,
            dtype=torch.float,
            device=self.device
        )
        # Encode prompts: without grad.
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_prompts_torch, # Bx4
                masks=None # Bx1x256x256
            )
        # Decode masks:
        np.random.seed(42) # We fix random positional encoding with a random seed.
        if train:
            masks, _ = self.model.mask_decoder(
                image_embeddings=embeddings.squeeze(), # Bx256x64x64
                image_pe=self.model.prompt_encoder.get_dense_pe(), # 1x256x64x64
                sparse_prompt_embeddings=sparse_embeddings, # Bx2x256
                dense_prompt_embeddings=dense_embeddings, # Bx256x64x64
                multimask_output=False
            )
        else:
            # Decode masks with no_grad
            with torch.no_grad():
                masks, _ = self.model.mask_decoder(
                    image_embeddings=embeddings.squeeze(), # Bx256x64x64
                    image_pe=self.model.prompt_encoder.get_dense_pe(), # 1x256x64x64
                    sparse_prompt_embeddings=sparse_embeddings, # Bx2x256
                    dense_prompt_embeddings=dense_embeddings, # Bx256x64x64
                    multimask_output=False
                )
        # Reshape the masks back to original resolution[512, 512]. # Bx1x512x512
        masks_pred = self.model.postprocess_masks(
            masks,
            input_size=(1024, 1024),
            original_size=(512, 512)
        )
        # Clean memory
        masks, sparse_embeddings, dense_embeddings = None, None, None
        torch.cuda.empty_cache()
        gc.collect()
        return embeddings, gt_masks, masks_pred

    def _run_batch(self, embeddings, gt_masks):
        """
        Runs a single Batch.
        
        Parameters:
        embeddings (torch.Tensor): Batch of embeddings of shape 1xBx256x64x64.
        gt_masks (torch.Tensor): Batch of ground truth masks of shape 1xBx512x512.
        
        Returns:
            (float): Average Batch Loss.
        """
        self.logger.info(f"Embeddings and gt_masks have shapes: {embeddings.shape}; {gt_masks.shape}")
        self.logger.info("Predicting...")
        embeddings, gt_masks, masks_pred = self._predict(embeddings, gt_masks, train=True)
        # Move gt_masks to device.
        gt_masks = gt_masks.to(device=device, dtype=torch.long).squeeze()
        
        self.logger.info("Computing Losses...")
        if self.loss._get_name() == "CrossEntropyLoss":
            # CrossEntropyLoss expects [BxHxW]
            loss = self.loss(masks_pred, gt_masks)
            loss_add = self.extra_loss(masks_pred, gt_masks)
        else:
            loss = self.loss(masks_pred, gt_masks.unsqueeze(1))
            loss_add = self.extra_loss(masks_pred, gt_masks.unsqueeze(1))
        self.logger.info(f"{self.extra_loss._get_name()}: {loss_add}")
        self.writer.add_scalar("Extra training Loss", loss_add, self.tensorboard_steps["Extra Loss"])
        self.tensorboard_steps["Extra Loss"] +=1
        
        # Backward pass
        self.logger.info("Computing Gradients...")
        loss.backward()
        # Clear memory
        embeddings, gt_masks, masks_pred = None, None, None
        self.logger.info("Clearning memory...")
        torch.cuda.empty_cache()
        gc.collect()
        return loss.item()
    
    def _validate(self):
        """
        Runs a complete Validation.
        Returns:
            (float): Average Validation Loss over all validation batches.
        """
        self.logger.info(f"Starting Validation run.")
        total_loss = 0.
        batch_loss = 0.
        # Drop last uncomplete batch in case of batch_accumulation.
        last_batch = int(len(self.valid_data) - len(self.valid_data) % self.batch_acc)
        for i, data in enumerate(self.valid_data):
            embeddings, gt_masks = data
            embeddings, gt_masks, masks_pred = self._predict(embeddings, gt_masks, train=False)
            gt_masks = gt_masks.to(device=device, dtype=torch.long).squeeze()
            
            if self.loss._get_name() == "CrossEntropyLoss":
                # CrossEntropyLoss expects [BxHxW]
                loss = self.loss(masks_pred, gt_masks)
            else:
                loss = self.loss(masks_pred, gt_masks.unsqueeze(1))
            total_loss += loss.item()
            batch_loss += loss.item()
            
            # Clear memory
            embeddings, gt_masks, masks_pred = None, None, None
            torch.cuda.empty_cache()
            gc.collect()
            
            # Accumualate batches in case of a batch_accumulation to keep consistancy with training loop.
            if (i + 1) % self.batch_acc == 0:
                batch_id = int((i + 1) / self.batch_acc)
                # Log meta_data
                self.logger.info(f"{batch_id}. [{self.device}]"
                    f"Valid_Batch: {batch_id} | "
                    f"Size: {self.valid_data.dataset.batch_size * self.batch_acc} | "
                    f"avg Loss: {batch_loss / self.batch_acc}")
                # Add Loss to tensorboard
                self.writer.add_scalar(f"Avg batch {self.loss._get_name()} Loss/valid", batch_loss / self.batch_acc, self.tensorboard_steps["Avg batch Loss/valid"])
                # Increment step
                self.tensorboard_steps["Avg batch Loss/valid"] += 1
                # restart total loss
                batch_loss = 0.
            
            # Drop_last uncomplete batch in batch_accumulation.
            if (i + 1) == last_batch:
                return total_loss / last_batch

    def _run_epoch(self, epoch, start_index=0):
        """
        Runs a single training epoch. Possible to continue training from certain point.
        
        Parameters:
            epoch (int): Indicator of current epoch. Starting from 0. 
            start_index (int): Indicator for current batch.
        
        Returns:
            (None)
        """
        self.optimizer.zero_grad()
        batch_loss = 0.
        # In case of batch_accumulation we drop the last uncomplete batch.
        last_batch = len(self.train_data.dataset) - len(self.train_data.dataset) % self.batch_acc
        print(f"start_index: {start_index}")
        print(f"Iterations left: {self.train_data.__len__()}")
        for i, data in enumerate(self.train_data, start=start_index):
            self.logger.info(f"Starting Batch {i + 1}")
            embeddings, gt_masks = data
            batch_loss += self._run_batch(embeddings, gt_masks)
            
            # Batch accumulation.
            if (i + 1) % self.batch_acc == 0:
                self.logger.info(f"Accumulating Gradients of last {self.batch_acc} mini_batches...")
                batch_id = int((i + 1) / self.batch_acc)
                # Updating Learning Rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lrs[(epoch * last_batch) + batch_id - 1]
                self.logger.info(f"LR has been updated to {self.lrs[(epoch * last_batch) + batch_id - 1]}")
                self.writer.add_scalar("LR schedule", self.lrs[batch_id - 1], self.tensorboard_steps["Avg batch Loss/train"])
                self.logger.info("Updating parameters...")
                self.optimizer.step()
                self.logger.info("Zeroing gradients...")
                self.optimizer.zero_grad()
                self.logger.info("Optimization step completed.")
            
                # Log meta_data
                self.logger.info(f"{batch_id}. [{self.device}] Epoch {epoch} | "
                    f"Batch: {batch_id} | "
                    f"Size: {self.train_data.dataset.batch_size * self.batch_acc} | "
                    f"avg Loss: {batch_loss / self.batch_acc}")
                self.writer.add_scalar(f"Avg batch {self.loss._get_name()} Loss/train", batch_loss / self.batch_acc, self.tensorboard_steps["Avg batch Loss/train"])
                self.tensorboard_steps["Avg batch Loss/train"] += 1

                # Reset batch_loss
                batch_loss = 0.

                if batch_id % self.save_every == 0:
                    self._save_checkpoint(epoch, batch_id=batch_id)
                if batch_id % self.validate_every == 0:
                    self._validate()
            
            # Drop_last uncomplete batch in batch_accumulation.
            if (i + 1) == last_batch:
                break

    def _save_checkpoint(self, epoch_id, batch_id="done"):
        """
        Saves a model checkpoint and optimizer state.
        
        Parameters:
            epoch_id (int): Indicator of current epoch.
            batch_id (str): Indicator of current batch.
        
        Returns:
            (None)
        """
        checkpoint = self.model.state_dict()
        PATH = os.path.join(self.LOG_PATH, "checkpoints", "checkpoint_" + str(epoch_id) + "_" + str(batch_id) + ".pth")
        PATH_OPT = os.path.join(self.LOG_PATH, "checkpoints", "optimizer_" + str(epoch_id) + "_" + str(batch_id) + ".pkl")
        # Save Model checkpoint.
        torch.save(checkpoint, PATH)
        # Save Optimizer state.
        with open(PATH_OPT, "wb") as f:
            pickle.dump(self.optimizer, f)
        if batch_id == "done":
            self.logger.info(f"Epoch {epoch_id} finished | Training checkpoint saved at {PATH}")
        else:
            self.logger.info(f"Epoch {epoch_id} | Batch {batch_id} | Training checkpoint saved at {PATH}")


    def train(self, max_epochs: int, seed: int):
        """
        Starts training. Main Function of the training loop.        
        
        Parameters:
            max_epochs (int): Number of epochs to train for.
            seed (int): Random seed for reproducibility.
        
        Returns:
            (None)
        """
        # Create directory for checkpoints.
        os.makedirs(os.path.join(self.LOG_PATH, "checkpoints"))

        for epoch in range(self.n_epoch - max_epochs, max_epochs):
            self.logger.info(f"[Training] Starting Epoch {epoch}")
            self._run_epoch(epoch)
            self.logger.info(f"[Training] Epoch N.{epoch} has been finished.")
            
            # Save after every epoch
            self._save_checkpoint(epoch)
            # Validate after every epoch
            total_loss = self._validate()
            
            self.logger.info(f"Avg Epoch Loss/valid: {total_loss}")
            self.writer.add_scalar(f"Avg Epoch {self.loss._get_name()} Loss/valid", total_loss, self.tensorboard_steps["Avg Epoch Loss/valid"])
            self.tensorboard_steps["Avg Epoch Loss/valid"] += 1

            # After each Eppoch regenerate training_set: reshuffle data.
            self.logger.info(f"Starting regenerating training plan for the next epoch.")
            # Setting seed for reproducibility.
            np.random.seed(seed + epoch + 10)
            self.train_data.dataset._regenerate_training_plan()
            self.train_data.dataset.training_plan.to_csv(os.path.join(self.LOG_PATH, f"training_plan_epoch{epoch + 1}.csv"), index=False)
            self.logger.info(f"Training plan has been regenerated and saved in {self.LOG_PATH}")
        
        self.writer.close()
        self.logger.info(f"Training has been succesfully finished.")
    
    def continue_training(self, epoch: int, batch: int, seed: int):
        """
        Continues training form a checkpoint.
        
        Parameters:
            epoch (int): Indicator of epoch to continue from.
            batch (int): Indicator of batch to continue from.
            seed (int): Random seed for reproducibility

        Returns:
            (None)
        """

        self.logger.info(f"Continuing training from | epoch: {epoch}, batch: {batch} |")
        self._run_epoch(epoch, start_index=batch) # Run from checkpoint batch
        self.logger.info(f"[Training] Epoch N.{epoch} has been finished.")
        # Save after every epoch
        self._save_checkpoint(epoch)
        # Validate after every epoch
        total_loss = self._validate()
        
        self.logger.info(f"Avg Epoch Loss/valid: {total_loss}")
        self.writer.add_scalar(f"Avg Epoch {self.loss._get_name()} Loss/valid", total_loss, self.tensorboard_steps["Avg Epoch Loss/valid"])
        self.tensorboard_steps["Avg Epoch Loss/valid"] += 1
        
        # After each epoch regenerate training_set
        self.logger.info(f"Starting regenerating training plan for the next epoch.")
        # Setting seed for reproducibility.
        np.random.seed(seed + epoch + 10)
        self.train_data.dataset._regenerate_training_plan()
        self.train_data.dataset.training_plan.to_csv(os.path.join(self.LOG_PATH, f"training_plan_epoch{epoch + 1}"), index=False)
        self.logger.info(f"Training plan has been regenerated and saved in {self.LOG_PATH}")
        
        self.train(self.n_epoch - (epoch + 1))
        
        self.writer.close()
        self.logger.info(f"Epoch has been succesfully finished and saved.")
        
def prepare_objects(
    model_checkpoint,
    instance_type,
    batch_size,
    case_sample,
    valid_prop,
    data_path,
    seed,
    optim_config,
    logger_setup,
    ct,
    retrain_decoder,
    device
):
    """
    Prepeare all necessary objects for training.

    Parameters:
        model_checkpoint (str): Path to the model checkpoint.
        retrain_decoder (bool): Flag, whether to retrain decoder from scratch.
        device (torch.cuda.device): Device.
        seed (int): Random Seed for reproducibility.
    Returns:
        DataLoaders, Model, Optimizer and Loss.
    """
    # Setting Random Seed for reproduciblity.
    np.random.seed(seed)
    logger_setup.info(f"Preparing training data...")
    train_set = KiTSdata(
        instance_type,
        case_sample=case_sample,
        batch_size=batch_size,
        device=device,
        path=data_path,
        train=True,
        valid_prop=valid_prop,
        logger=logger_setup
    )
    logger_setup.info(f"Training Data has been prepeared.")
    logger_setup.info(f"Training set has {len(train_set)} batches")
    logger_setup.info(f"Preparing validation data...")
    valid_set = KiTSdata(
        instance_type,
        case_sample=case_sample,
        batch_size=batch_size,
        device=device,
        path=data_path,
        train=False,
        valid_prop=valid_prop,
        logger=logger_setup
    )
    logger_setup.info(f"Validation Data has been prepeared.")
    logger_setup.info(f"Validation set has {len(valid_set)} batches")

    if ct:
        # Create CustomSampler to start sampling from where it stopped and use training_plan from that epoch.
        epoch, batch = re.search(r"checkpoint_(\d+)_(\d+)\.pth", training_config["model"]).groups()
        sampler = ContinueTrainingSampler(train_set, int(batch), train_set.__len__())
        if int(epoch) > 0:
            train_set.training_plan = pd.read_csv(f"{training_config['save_path']}/training_plan_epoch{epoch}.csv")
    else:
        sampler = None

    # We dont set batch_size in DataLaoder since the batches are already configured in Dataset class [training_plan].
    logger_setup.info(f"Preparing DataLoader with CustomSampler: {sampler}")
    train_loader = DataLoader(train_set, sampler=sampler)
    valid_loader = DataLoader(valid_set)
    
    # Configuring Model
    logger_setup.info("Reading SAM base model checkpoint from the registry")
    if retrain_decoder:
        state_dict_encoder = {}
        model = sam_model_registry["vit_b"]()
        # Loading Prompt Encoder paramters
        with open(model_checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))
        [state_dict_encoder.update({k[15:]:v}) for k, v in state_dict.items() if k.startswith("prompt_encoder")]
        model.prompt_encoder.load_state_dict(state_dict_encoder, strict=True)
    else:
        logger_setup.info("SAM Model Decoder checkpoint has been picked")
        model = sam_model_registry["vit_b"](model_checkpoint)
    
    # Configuring Optimizer
    logger_setup.info(f"Initializing AdamW optimizer and {optim_config['loss']} Loss")
    optimizer = torch.optim.AdamW(
        params=model.mask_decoder.parameters(),
        lr=optim_config["initial_lr"],
        betas=tuple(optim_config["betas"]),
        weight_decay=optim_config["weight_decay"]
    )
    
    # Configuring Loss
    if optim_config["loss"] == "Dice":
        loss = DiceLoss(sigmoid=True, reduction='mean', squared_pred=True)
    elif optim_config["loss"] == "CrossEntropy":
        loss = torch.nn.BCEWithLogitsLoss()
    elif optim_config["loss"] == "GeneralizedDiceFocalLoss":
        loss = GeneralizedDiceFocalLoss(sigmoid=True)
    elif optim_config["loss"] == "DiceCELoss":
        loss = DiceCELoss(sigmoid=True)
    return train_loader, valid_loader, model, optimizer, loss

def main(
    device,
    training_config,
    data_config,
    optim_config,
    lr_schedule,
    logger_setup,
    ct
):
    """
    Main Function of the Training Loop.
    Prepeares all necessary objects and runs training loop.

    Parameters:
        Takes parameters from the config.yaml file.
        device (torch.cuda.device): Device.
        training_config (dict): Training Configuration from config.yaml.
        data_config (dict): Data Configuration from config.yaml.
        optim_config (dict): Optimization Configuration from config.yaml.
        lr_schedule (dict): LR Configuration from config.yaml.
        logger_setup (logging.logger): Logger for Setup process logs.
        ct (bool): Flag, that indicates whether to continue training or start from the beginning.
    Returns:
        (None)
    """
    train_loader, valid_loader, model, optimizer, loss = prepare_objects(
        training_config["model"],
        data_config["instance_type"],
        training_config["batch_size"],
        training_config["case_sample"],
        data_config["valid_prop"],
        data_config["data_path"],
        training_config["seed"],
        optim_config,
        logger_setup,
        ct,
        training_config["retrain_decoder"],
        device
    )
    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        training_config["n_epoch"],
        optimizer,
        loss,
        device,
        training_config["batch_acc"],
        training_config["save_every"],
        training_config["validate_every"],
        lr_schedule,
        training_config["save_path"]
    )
    if ct:
        # Continue Training
        epoch, batch = re.search(r"checkpoint_(\d+)_(\d+)\.pth", training_config["model"]).groups()
        trainer.tensorboard_steps = {
            "Avg batch Loss/train": int(epoch) * 1000 + int(batch),
            "Avg batch Loss/valid": 1 + int(epoch),
            "Avg Epoch Loss/valid": 1 + int(epoch),
            "Extra Loss": int(epoch) * 1000 + int(batch) * trainer.batch_acc
        }
        PATH_OPT = os.path.join(training_config["save_path"], "checkpoints", "optimizer_" + str(epoch) + "_" + str(batch) + ".pkl")
        with open(PATH_OPT, "rb") as f:
            optimizer = pickle.load(f)
        trainer.optimizer = optimizer
        trainer.continue_training(int(epoch), int(batch), int(training_config["seed"]))
    else:
        trainer.train(training_config["n_epoch"], int(training_config["seed"]))


if __name__ == "__main__":
    import argparse
    import logging
    import yaml
    # Create Parser
    parser = argparse.ArgumentParser(description='Training Loop')
    parser.add_argument("--config_file", type=str, help="filled-out config.yaml or configuration.yaml from experiment", required=True)
    parser.add_argument("--device", type=str, help="device identifier [cpu, cuda:0, cuda:1]", required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Possibility to continue training from the checkpoint")
    args = parser.parse_args()
    
    # Read the config.yaml file
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    data_config = config["data_config"]
    training_config = config["training_config"]
    optimizer_config = config["optimizer_config"]
    lr_schedule = config["lr_schedule"]

    if not os.path.exists(training_config["save_path"]):
        os.makedirs(training_config["save_path"])
    if optimizer_config["loss"] not in ["Dice", "CrossEntropy", "GeneralizedDiceFocalLoss", "DiceCELoss"]:
        print(f"{optimizer_config['loss']} is not supported choose either 'Dice', 'CrossEntropy', 'GeneralizedDiceFocalLoss' or 'DiceCELoss'")
        sys.exit()
    
    # Checking if the save_path have already been used to avoid overwritting important data.
    fn = os.path.join(training_config["save_path"], "setup.log")
    if os.path.exists(fn) and args.checkpoint is None:
      # Job has been already been started
      print(f"Directory: [{training_config['save_path']}] has already been used | " 
        f"Please clean the directory or choose a new one and restart the job afterwards."
      )
      sys.exit()

    # Configure logger for Setup
    logger_setup = logging.getLogger("setup_log")
    logger_setup.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(training_config["save_path"], "setup.log"))
    handler2.setFormatter(formatter)
    logger_setup.addHandler(handler1)
    logger_setup.addHandler(handler2)

    logger_setup.info("Starting setup....")
    device = torch.device(args.device)
    print(torch.cuda.get_device_name())
    logger_setup.info(f"Device is set to {device.type}")

    if args.checkpoint is not None:
        # This runs when continuing training of an experiment.
        training_config["model"] = args.checkpoint
        logger_setup.info("Model has been set to the provided checkpoint.")
    # Saving a copy of configuration in the save_path.
    with open(os.path.join(training_config["save_path"], "configuration.yaml"), "w") as file:
        yaml.dump(config, file)
    logger_setup.info(f"config.yaml has been copied to {training_config['save_path']}")
    continue_training = args.checkpoint != None
    logger_setup.info(f"Continue Training mode: {continue_training}")

    # Main function call
    main(device, training_config, data_config, optimizer_config, lr_schedule, logger_setup, ct=continue_training)
