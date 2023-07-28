def prepare_objects(
    model_checkpoint,
    instance_type,
    batch_size,
    case_sample,
    valid_prop,
    data_path,
    save_path,
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
        epoch, batch = re.search(r"checkpoint_(\d+)_(\d+)\.pth", model_checkpoint).groups()
        sampler = ContinueTrainingSampler(train_set, int(batch), train_set.__len__())
        if int(epoch) > 0:
            train_set.training_plan = pd.read_csv(f"{save_path}/training_plan_epoch{epoch}.csv")
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
        training_config['save_path'],
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
    import os
    import sys
    import argparse
    import logging
    import yaml
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory to the sys.path list to access segment_anything directory
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    os.chdir(parent_dir)
    from segment_anything import sam_model_registry
    from TransferSAM.KiTS_dataset import KiTSdata, ContinueTrainingSampler
    from TransferSAM.Trainer import Trainer

    
    from monai.losses import DiceLoss, GeneralizedDiceFocalLoss, DiceCELoss
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import pandas as pd
    import pickle
    import re
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