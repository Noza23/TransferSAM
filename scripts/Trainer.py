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
from utils.KiTS_dataset import KiTSdata, ContinueTrainingSampler
import matplotlib.pyplot as plt
from utils.create_prompt import  identify_box
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from monai.losses import DiceLoss, GeneralizedDiceFocalLoss, DiceCELoss, FocalLoss
import logging


class Trainer:
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
        save_every: int, # batch
        validate_every: int, # batch
        lr_schedule: dict,
        save_path: str,
    ) -> None:
        
        self.device = device
        self.model = model.to(device)
        self.transform  = ResizeLongestSide(model.image_encoder.img_size)
        self.transform_maskprompts = ResizeLongestSide(target_length=256)
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
            if train_data.dataset.instance_type in ["full", "Stage2"]:
                self.extra_loss = torch.nn.CrossEntropyLoss()
            else:
                self.extra_loss = torch.nn.BCEWithLogitsLoss()
        else:
            if train_data.dataset.instance_type in ["full", "Stage2"]:
                self.extra_loss = DiceLoss(reduction='mean', softmax=True, to_onehot_y=True, squared_pred=True)
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
        
        self.writer = SummaryWriter(os.path.join(save_path, "logs"))
        
        self.logger.info("Trainer instance has been configured and has following attributes:")
        for arg, value in self.__dict__.items():
          if arg=="model":
            self.logger.info(f"{arg}: SAM_base")
          else:
            self.logger.info(f"{arg}: {value}")
        
        # generate Learning rates
        self.lrs = self._generate_lr_shedule(lr_schedule)
        # Save learning rate shedule plot
        self._save_lrs()
        self.logger.info(f"LR shedule plot has been saved in {save_path}")

    def _save_lrs(self):
        plt.plot(range(1, len(self.lrs) + 1), self.lrs)
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.savefig(os.path.join(self.LOG_PATH, 'LR_schedule.jpeg'))
        plt.clf()
    
    def _cosine_annealing_with_wave(self, batch, num_batches, interval, max_lr, min_lr):
        wave_height = max_lr - ((batch // interval) * (max_lr - min_lr) / (num_batches // interval))
        wave_length = num_batches // interval
        wave_value = (math.cos(math.pi * (batch % wave_length) / wave_length) + 1) / 2 * wave_height + min_lr
        return wave_value
    
    def _generate_lr_shedule(self, config: dict):
        for key, value in config.items():
            if key:
                constant_n = value["constant_lr"]
                gamma = value["gamma"]
                lr_init = self.optimizer.param_groups[0]["lr"]
                # Total number of batches
                n_batches = int(len(self.train_data.dataset) / self.batch_acc)
                lrs =  []
                # Warm-up steps
                if value['warmup_lr'] is not None:
                    lrs.extend(value['warmup_steps'] * [value['warmup_lr']])
                    n_batches = n_batches - value['warmup_steps']
                if gamma is not None:
                    steps = math.ceil((n_batches * self.n_epoch) / constant_n)
                    for step in range(steps):
                        lrs.extend(constant_n * [lr_init * gamma**step])
                    return lrs
                for epoch in range(self.n_epoch):
                    # Constant part
                    lrs.extend(constant_n * [lr_init / (epoch + 1)])
                    # Cosine part
                    for batch in range(n_batches - constant_n):
                        lr = self._cosine_annealing_with_wave(
                            batch,
                            n_batches - constant_n,
                            max(int((n_batches - constant_n) / 100), 1),
                            lr_init / (epoch + 1),
                            value["min_lr"]
                        )
                        lrs.append(lr)
                return lrs

    def _predict(self, embeddings, gt_masks, train=True):
        if self.train_data.dataset.instance_type in ["full", "Stage2"]:
            multimask_output = True
        else:
            multimask_output = False
        self.logger.info(f"multimask_output was set to {multimask_output}")
        if self.train_data.dataset.instance_type == "Stage2":
            mask_prompts = self.transform_maskprompts.apply_image_torch(gt_masks.permute(1, 0, 2, 3).to(device=self.device, dtype=torch.float32))
        else:
            mask_prompts = None
        ### Predict either in train or eval mode ###
        if train:
            self.model.train()
        else:
            self.model.eval()
        # Generate Box prompts.
        box_prompts = identify_box(gt_masks.squeeze(), prob=0.2)
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
                masks=mask_prompts # Bx1x256x256
            )
        # Decode masks:
        np.random.seed(42)
        if train:
            masks, _ = self.model.mask_decoder(
                image_embeddings=embeddings.squeeze(), # Bx256x64x64
                image_pe=self.model.prompt_encoder.get_dense_pe(), # 1x256x64x64
                sparse_prompt_embeddings=sparse_embeddings, # Bx2x256
                dense_prompt_embeddings=dense_embeddings, # Bx256x64x64
                multimask_output=multimask_output
            )
        else:
            # Decode masks with no_grad
            with torch.no_grad():
                masks, _ = self.model.mask_decoder(
                    image_embeddings=embeddings.squeeze(), # Bx256x64x64
                    image_pe=self.model.prompt_encoder.get_dense_pe(), # 1x256x64x64
                    sparse_prompt_embeddings=sparse_embeddings, # Bx2x256
                    dense_prompt_embeddings=dense_embeddings, # Bx256x64x64
                    multimask_output=multimask_output
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
        print(f"Predicted MAsks: {masks_pred.shape, masks_pred.dtype, masks_pred.device}")
        return masks_pred


    def _run_batch(self, embeddings, gt_masks):
        ### Run single batch. ###
        # embeddings: 1xBx256x64x64
        self.logger.info("     Predicting...")
        self.logger.info(f"Embeddings and gt_masks have shapes: {embeddings.squeeze().shape}; {gt_masks.squeeze().shape}")
        self.logger.info(f"Picture contains following classes: {gt_masks.unique()}")
        masks_pred = self._predict(embeddings, gt_masks, train=True)
        # Move gt_masks to the same device.
        gt_masks = gt_masks.to(device=device, dtype=torch.long).squeeze()
        # Compute Loss and gradients
        self.logger.info("     Computing Losses...")
        # CrossEntropyLoss expects [BxHxW]
        if self.loss._get_name() == "CrossEntropyLoss":
            loss = self.loss(masks_pred, gt_masks)
        else:
            print(f"Losses: gt_masks: {gt_masks.unique()}, masks_pred: {masks_pred.unique()}")
            loss = self.loss(masks_pred, gt_masks.unsqueeze(1))

        if self.extra_loss._get_name() == "CrossEntropyLoss":
            loss_add = self.extra_loss(masks_pred, gt_masks)
        else:
            loss_add = self.extra_loss(masks_pred, gt_masks.unsqueeze(1))

        self.writer.add_scalar("Extra training Loss", loss_add, self.tensorboard_steps["Extra Loss"])
        self.tensorboard_steps["Extra Loss"] +=1

        self.logger.info(f"     Mini_batch avg.Loss: {loss.item()} ...")
        # Backward pass
        self.logger.info("     Computing Gradients...")
        loss.backward()
        # Clear memory
        embeddings, gt_masks, masks_pred = None, None, None
        self.logger.info("     Clearning memory...")
        torch.cuda.empty_cache()
        gc.collect()
        return loss.item()
    
    def _validate(self):
        self.logger.info(f"Starting Validation run")
        total_loss = 0.
        batch_loss = 0.
        last_batch = int(len(self.valid_data) - len(self.valid_data) % self.batch_acc)
        for i, data in enumerate(self.valid_data):
            embeddings, gt_masks = data
            masks_pred = self._predict(embeddings, gt_masks, train=False)
            gt_masks = gt_masks.to(device=device, dtype=torch.long).squeeze()
            if self.loss._get_name() == "CrossEntropyLoss":
                loss = self.loss(masks_pred, gt_masks)
            else:
                loss = self.loss(masks_pred, gt_masks.unsqueeze(1))
            total_loss += loss.item()
            batch_loss += loss.item()
            # Clear memory
            embeddings, gt_masks, masks_pred = None, None, None
            torch.cuda.empty_cache()
            gc.collect()
            # Accumualate loss
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
            # Drop_last uncomplete batch in batch_accumulation
            if (i + 1) == last_batch:
                return total_loss / last_batch

    def _run_epoch(self, epoch, start_index=0):
        self.optimizer.zero_grad()
        batch_loss = 0.
        last_batch = len(self.train_data.dataset) - len(self.train_data.dataset) % self.batch_acc
        # To allow continuing training from checkpoint with use start_index argument
        print(f"start_index: {start_index}")
        print(f"Iterations left: {self.train_data.__len__()}")
        for i, data in enumerate(self.train_data, start=start_index):
            self.logger.info(f"Starting mini_batch {i + 1}")
            # Embeddings have been moved to device in __getitem__.
            embeddings, gt_masks = data
            batch_loss += self._run_batch(embeddings, gt_masks)
            # Batch accumulation.
            if (i + 1) % self.batch_acc == 0:
                self.logger.info(f"Accumulating Gradients of last {self.batch_acc} mini_batches...")
                batch_id = int((i + 1) / self.batch_acc)
                # Updating Learning Rate
                if self.lrs is not None:
                    self.logger.info("Updating Learning Rate...")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lrs[(epoch * last_batch) + batch_id - 1]
                    self.logger.info(f"LR has been updated to {self.lrs[(epoch * last_batch) + batch_id - 1]}")
                    # Logging LR
                    self.writer.add_scalar("LR schedule", self.lrs[batch_id - 1], self.tensorboard_steps["Avg batch Loss/train"])
                self.logger.info("     Monitoring Gradients...")
                self._monitor_grads(self.model.mask_decoder.named_parameters(), id=batch_id)
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
                # Log loss to tensorboard
                self.writer.add_scalar(f"Avg batch {self.loss._get_name()} Loss/train", batch_loss / self.batch_acc, self.tensorboard_steps["Avg batch Loss/train"])
                # Increment step
                self.tensorboard_steps["Avg batch Loss/train"] += 1

                # Reset batch_loss
                batch_loss = 0.
                if batch_id % 50 == 0:
                    plt.savefig(os.path.join(self.LOG_PATH, "gradients", f"gradients_{batch_id}.jpeg"))
                    self.logger.info("Gradient Monitoring Plot has been saved.")
                    plt.clf()
                # Save every n-th batch
                if batch_id % self.save_every == 0:
                    self._save_checkpoint(epoch, batch_id=batch_id)
                # validate every m-th batch
                if batch_id % self.validate_every == 0:
                    self._validate()
            # Drop_last uncomplete batch in batch_accumulation
            if (i + 1) == last_batch:
                break

    def _save_checkpoint(self, epoch_id, batch_id="done"):
        checkpoint = self.model.state_dict()
        PATH = os.path.join(self.LOG_PATH, "checkpoints", "checkpoint_" + str(epoch_id) + "_" + str(batch_id) + ".pth")
        torch.save(checkpoint, PATH)
        if batch_id == "done":
            self.logger.info(f"Epoch {epoch_id} finished | Training checkpoint saved at {PATH}")
        else:
            self.logger.info(f"Epoch {epoch_id} | Batch {batch_id} | Training checkpoint saved at {PATH}")

    def _monitor_grads(self, params, id):
        layers, norms = [], []
        for name, param in params:
            if (param.requires_grad and param.grad is not None) and ("bias" not in name):
                layers.append(name)
                norms.append(torch.linalg.norm(param.grad.cpu()).item())
        plt.plot(layers, norms, alpha=min(0.01 + id / 100, 1), color="blue")
        plt.xlim(xmin=0, xmax=len(layers))
        plt.xticks(rotation=90)
        plt.xlabel("Layers")
        plt.title("Monitoring gradient L2-norm")
        plt.ylabel("L2 Norm")
        plt.grid(True)

    def train(self, max_epochs: int):
        os.makedirs(os.path.join(self.LOG_PATH, "checkpoints"))
        os.makedirs(os.path.join(self.LOG_PATH, "gradients"))
        for epoch in range(self.n_epoch - max_epochs, max_epochs):
            self.logger.info(f"[Training] Starting Epoch {epoch}")
            plt.figure(figsize=(15, 15))
            self._run_epoch(epoch)
            self.logger.info(f"[Training] Epoch N.{epoch} has been finished.")
            # Save after every epoch
            self._save_checkpoint(epoch)
            # Validate after every epoch
            total_loss = self._validate()
            self.logger.info(f"Avg Epoch Loss/valid: {total_loss}")
            # Add loss to tensorboard
            self.writer.add_scalar(f"Avg Epoch {self.loss._get_name()} Loss/valid", total_loss, self.tensorboard_steps["Avg Epoch Loss/valid"])
            # Increment step
            self.tensorboard_steps["Avg Epoch Loss/valid"] += 1
            # After each epoch regenerate training_set
            self.logger.info(f"Starting regenerating training plan for the next epoch.")
            np.random.seed(105 + epoch)
            self.train_data.dataset._regenerate_training_plan()
            self.train_data.dataset.training_plan.to_csv(os.path.join(self.LOG_PATH, f"training_plan_epoch{epoch + 1}"), index=False)
            self.logger.info(f"Training plan has been regenerated and saved in {self.LOG_PATH}")
        self.writer.close()
        plt.clf()
        self.logger.info(f"Training has been succesfully finished.")
    
    def continue_training(self, epoch, batch):
        self.logger.info(f"Continuing training from | epoch: {epoch}, batch: {batch} |")
        plt.figure(figsize=(15, 15))
        self._run_epoch(epoch, start_index=batch) # Run from checkpoint batch
        self.logger.info(f"[Training] Epoch N.{epoch} has been finished.")
        # Save after every epoch
        self._save_checkpoint(epoch)
        # Validate after every epoch
        total_loss = self._validate()
        self.logger.info(f"Avg Epoch Loss/valid: {total_loss}")
        # Add loss to tensorboard
        self.writer.add_scalar(f"Avg Epoch {self.loss._get_name()} Loss/valid", total_loss, self.tensorboard_steps["Avg Epoch Loss/valid"])
        # Increment step
        self.tensorboard_steps["Avg Epoch Loss/valid"] += 1
        # After each epoch regenerate training_set
        self.logger.info(f"Starting regenerating training plan for the next epoch.")
        np.random.seed(105 + epoch)
        self.train_data.dataset._regenerate_training_plan()
        self.train_data.dataset.training_plan.to_csv(os.path.join(self.LOG_PATH, f"training_plan_epoch{epoch + 1}"), index=False)
        self.logger.info(f"Training plan has been regenerated and saved in {self.LOG_PATH}")
        # Continue training from checkpoint epoch
        self.train(self.n_epoch - (epoch + 1))
        self.writer.close()
        plt.clf()
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
    logger_setup.info(f"Setting random.seed to {seed} before generating Datasets for reproducibility.")
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
        # Create CustomSampler to start sampling from where it stopped
        _, batch = re.search(r"checkpoint_(\d+)_(\d+)\.pth", training_config["model"]).groups()
        sampler = ContinueTrainingSampler(train_set, int(batch), train_set.__len__())
    else:
        sampler = None

    # We dont set batch_size in DataLaoder since the batches are already configured in Dataset class [training_plan]
    logger_setup.info(f"Preparing DataLoader with CustomSampler: {sampler}")
    train_loader = DataLoader(train_set, sampler=sampler)
    valid_loader = DataLoader(valid_set)
    logger_setup.info("Reading SAM base model checkpoint from the registry")

    if retrain_decoder and not ct:
        state_dict_encoder = {}
        if instance_type in ["full", "Stage2"]:
            model = sam_model_registry["custom"](n_classes=4)
        else:
            model = sam_model_registry["vit_b"]()
        # Loading Prompt Encoder paramters
        with open(model_checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device("cpu"))
        [state_dict_encoder.update({k[15:]:v}) for k, v in state_dict.items() if k.startswith("prompt_encoder")]
        model.prompt_encoder.load_state_dict(state_dict_encoder, strict=True)
    elif retrain_decoder and ct:
        model = sam_model_registry["custom"](model_checkpoint)
    elif (not retrain_decoder) and (instance_type in ["full", "Stage2"]):
        model = sam_model_registry["custom"](model_checkpoint, n_classes=4)
    else:
        model = sam_model_registry["vit_b"](model_checkpoint)

    logger_setup.info(f"Initializing AdamW optimizer and {optim_config['loss']} Loss")
    optimizer = torch.optim.AdamW(
        params=model.mask_decoder.parameters(),
        lr=optim_config["initial_lr"],
        betas=tuple(optim_config["betas"]),
        weight_decay=optim_config["weight_decay"]
    )

    if instance_type in ["full", "Stage2"]:
        # Multi-class Losses
        if optim_config["loss"] == "Dice":
            loss = DiceLoss(reduction='mean', softmax=True, to_onehot_y=True, squared_pred=True, include_background=False)
        elif optim_config["loss"] == "CrossEntropy":
            loss = torch.nn.CrossEntropyLoss()
        elif optim_config["loss"] == "FocalLoss":
            loss = FocalLoss(to_onehot_y=True, gamma=2, use_softmax=True, include_background=False) 
        elif optim_config["loss"] == "GeneralizedDiceFocalLoss":
            loss = GeneralizedDiceFocalLoss(to_onehot_y=True, gamma=2, softmax=True, include_background=False)
        elif optim_config["loss"] == "DiceCELoss":
            loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, ce_weight=torch.tensor([0, 0.021, 0.075, 0.90]).to(device=device))
            print(f"Loss was selected: {loss}")
    else:
        if optim_config["loss"] == "Dice":
            loss = DiceLoss(sigmoid=True, reduction='mean', squared_pred=True)
        elif optim_config["loss"] == "CrossEntropy":
            loss = torch.nn.BCEWithLogitsLoss()
        elif optim_config["loss"] == "GeneralizedDiceFocalLoss":
            loss = GeneralizedDiceFocalLoss(gamma=3.5, sigmoid=True)  
        elif optim_config["loss"] == "DiceCELoss":
            loss = DiceCELoss(sigmoid=True, squared_pred=True)
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
        epoch, batch = re.search(r"checkpoint_(\d+)_(\d+)\.pth", training_config["model"]).groups()
        trainer.tensorboard_steps = {
            "Avg batch Loss/train": int(epoch) * 1000 + int(batch),
            "Avg batch Loss/valid": 1 + int(epoch),
            "Avg Epoch Loss/valid": 1 + int(epoch),
            "Extra Loss": int(epoch) * 1000 + int(batch) * trainer.batch_acc
        }
        trainer.continue_training(int(epoch), int(batch))
    else:
        trainer.train(training_config["n_epoch"])



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
    
    # Read the config file
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    data_config = config["data_config"]
    training_config = config["training_config"]
    optimizer_config = config["optimizer_config"]
    lr_schedule = config["lr_schedule"]

    if not os.path.exists(training_config["save_path"]):
        os.makedirs(training_config["save_path"])
    if optimizer_config["loss"] not in ["Dice", "CrossEntropy", "GeneralizedDiceFocalLoss", "FocalLoss", "DiceCELoss"]:
        print(f"{optimizer_config['loss']} is not supported choose either 'Dice' or 'CrossEntropy'")
        sys.exit()
    
    # Checking if the save_path have already been used to avoid overwritten important data.
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
        training_config["model"] = args.checkpoint
        logger_setup.info("Model has been set to the provided checkpoint.")

    # Saving a copy of configuration
    with open(os.path.join(training_config["save_path"], "configuration.yaml"), "w") as file:
        yaml.dump(config, file)
    logger_setup.info(f"config.yaml has been copied to {training_config['save_path']}")
    continue_training = args.checkpoint != None
    logger_setup.info(f"Continue Training mode: {continue_training}")
    main(device, training_config, data_config, optimizer_config, lr_schedule, logger_setup, ct=continue_training)
