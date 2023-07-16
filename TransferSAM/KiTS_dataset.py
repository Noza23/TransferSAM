import os
import numpy as np
import nibabel as nib
import torch
import cv2
import logging
from torch.utils.data import Dataset, Sampler
import pandas as pd
from torchvision import transforms

# Creating custom Dataset: Simple (map-style dataset), 
class KiTSdata(Dataset):
    def __init__(
        self,
        instance_type: str,
        case_sample: int, # How many cases to sample randomly in each __getitem__ step.
        batch_size: int, # How many slices should be sampled randomly from the sampled cases in each __getitem__ step.
        device: torch.device,
        path: str,
        train: bool=True,
        valid_prop: float=0.1,
        logger=None
    ):
        valid_instances = {"kidney", "cyst", "tumor", "full", "ROI", "Stage2"}
        if instance_type not in valid_instances:
            raise ValueError(f"instance_type must be one of: {valid_instances}")
        if batch_size < 2:
            raise ValueError(f"batch_size must be at least 2")
        
        self.instance_type = instance_type
        self.path = path
        self.train = train
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.log_path = os.path.dirname(self.logger.handlers[1].baseFilename)

        # Keep only cases where instance_type is present and embedding is avaliable if full segmentation take all instances
        if instance_type in ["full", "ROI", "Stage2"]:
            self.cases = sorted([f for f in os.listdir(self.path) if not f.startswith('.')])
        else:
            self.cases = self._identify_cases()
    
        # Train-Validation split is based on number of cases without shuffle.
        train_count = np.floor(len(self.cases) * (1 - valid_prop)).astype(int)
        if train:
            self.cases = self.cases[0:train_count]
            self.logger.info(f"Training set consists of {len(self.cases)} cases")
        else:
            self.cases = self.cases[train_count:]
            self.logger.info(f"Validation set consists of {len(self.cases)} cases")

        # Create DataFrame of datapoints (slices) with identifer from which case it is.
        self.datapoints = self._create_datapoint_df()
        self.case_sample = case_sample
        self.batch_size = batch_size
        # Generate training plan in batches
        self.training_plan = self._generate_training_plan(data=self.datapoints.copy())
        if train:
            self.training_plan.to_csv(os.path.join(self.log_path, "training_plan.csv"), index=False)
            self.logger.info(f"Training plan has been generated and saved in {self.log_path} directory.")
        else:
            self.training_plan.to_csv(os.path.join(self.log_path, "validation_set.csv"), index=False)
            self.logger.info(f"Validation set has been generated and saved in {self.log_path} directory.")

        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def _identify_cases(self):
        self.logger.info("Identifying cases...")
        # Returns cases where the "self.instance_type" instance is present and embedding is avaliable
        cases = sorted([f for f in os.listdir(self.path) if not f.startswith('.')])
        cases_relevant = []
        for case in cases:
            case_path = os.path.join(self.path, case)
            case_instance_path = case_path + "/instances/"
            embedding_avaliable = "embedding.npz" in os.listdir(case_path)
            instances_avaliable = os.path.exists(case_instance_path)

            if embedding_avaliable and instances_avaliable:
                bool_value = np.any([inst.startswith(self.instance_type) for inst in os.listdir(case_instance_path)])
            else:
                continue
            if bool_value:
                cases_relevant.append(case)
        self.logger.info(f"There are {len(cases_relevant)} relevant cases for {self.instance_type} instance.")
        return cases_relevant
    

    def _create_datapoint_df(self):
        self.logger.info("Generating annotation file...")
        # Creates Annotation file

        slices_total = []
        embedding_ids = []
        cases_total = []
        is_blank_total = []
        for case in self.cases:
            embedding_path = os.path.join(self.path, case, "embedding.npz")
            slices = np.load(embedding_path)["indices"].astype(np.int16)
            embedding_id = np.arange(slices.shape[0], dtype=np.int16)
            cases = np.repeat(case, slices.shape[0])
            is_blank = np.load(embedding_path)["is_blank"].astype(np.bool)
            slices_total.extend(slices)
            embedding_ids.extend(embedding_id)
            cases_total.extend(cases)
            is_blank_total.extend(is_blank)

        return pd.DataFrame(
            {
                "case": cases_total,
                "slice": slices_total,
                "embedding_id": embedding_ids,
                "is_blank": is_blank_total
            }
        )

    def _generate_training_plan(self, data):
        if self.train:
            self.logger.info("### Generating training plan... ###")
        else:
            self.logger.info("### Generating validation set... ###")

        # Generates training_plan based on case_sample and batch_size.
        training_plan = pd.DataFrame(
            columns=["case", "slice", "embedding_id", "is_blank", "batch"],
            dtype=np.int16
        )
        batch_i = 1
        # While there are enough blanks and non-blanks
        while np.all(data["is_blank"].value_counts()  >= int(self.batch_size / 2)):
            # Cases still avaliable
            cases_avaliable = pd.Series(data["case"].unique())
            n_cases_avaliable = cases_avaliable.shape[0]
            # if not enough cases avaliable break
            if n_cases_avaliable < self.case_sample:
              break

            # sample cases
            cases_sampled = cases_avaliable.sample(self.case_sample)
            self.logger.info(f"Initializing Batch {batch_i}: from total of {data.shape[0]} slices")
            
            # Sample according to distribution of blanks and non-blanks in dataset
            n_blanks = int(self.batch_size * data["is_blank"].value_counts()[True] / data.shape[0])
            dps = self._sample_stratified(data, cases_sampled, int(self.batch_size - n_blanks), blank=False)
            dps_blank = self._sample_stratified(data, cases_sampled, n_blanks, blank=True)

            datapoints_sampled  = pd.concat([dps, dps_blank], ignore_index=False)
            # Append training_plan with next batch
            datapoints_sampled["batch"] = batch_i
            training_plan = pd.concat([training_plan, datapoints_sampled], ignore_index=True)
            # Drop already sampled slices
            data.drop(index=datapoints_sampled.index, inplace=True)
            batch_i += 1
        return training_plan
    
    def _sample_stratified(self, data, cases_sampled, batch_size: int, blank: bool):
        # Filter avaliable data
        cases_avaliable = pd.Series(data["case"].unique())
        df = data[(data["case"].isin(cases_sampled)) & (data["is_blank"] == blank)]

        # If df has not enough datapoints to fill the batch take all 
        if df.shape[0] < batch_size:
            result = df.copy()
        else:
            # Else sample batch_size stratified by case
            probabs = self._calculate_probabs(df) + 0.0001 # for stability
            result = df.sample(batch_size, weights=probabs)

        # If df could not fill batch_size we sample another case until the batch is complete.
        while result.shape[0] < batch_size:
            # Sample additional case
            case_additional = cases_avaliable[~cases_avaliable.isin(cases_sampled)].sample(1).values.item()
            self.logger.info(f"Additonal case borrowed: {case_additional}")
                             
            # List avaliable slices associated to the case sampled.
            df_add = data[(data["case"] == case_additional) & (data["is_blank"] == blank)]
                        
            # Sample [stratified]: min(avaliable slices, batch.size - number of alredy sampled slices)
            if df_add.shape[0] < (batch_size - result.shape[0]):
                    result_add = df_add.copy()
            else:
                # Calculate the ratio of occurances of each case to allow stratified sampling
                probabs = self._calculate_probabs(df_add) + 0.0001 # for stability
                result_add = df_add.sample(batch_size - result.shape[0], weights=probabs)

            # Append new slices
            result = pd.concat([result, result_add], ignore_index=False)
        return result 
        
    def _calculate_probabs(self, df:pd.DataFrame):
        probs = df['case'].map((1 - df["case"].value_counts() / df.shape[0]) / df["case"].value_counts())
        return probs

    def _regenerate_training_plan(self):
        self.training_plan = self._generate_training_plan(self.datapoints.copy())

    # lenght protocol
    def __len__(self):
        # Max number of batch in training_plan
        batch_n = self.training_plan["batch"].max()
        return batch_n
    
    # getitem protocol
    def __getitem__(self, idx):
        batch_id = idx + 1
        if self.instance_type == "Stage2":
            batch_data = self.training_plan[(self.training_plan["batch"] == batch_id) & (self.training_plan["is_blank"] == 0)].sort_values("slice")
        else:
            batch_data = self.training_plan[self.training_plan["batch"] == batch_id].sort_values("slice")
        cases = batch_data["case"].unique()
        # According to training plan read slices and and gt_masks
        embeddings = []
        gt_masks = []
        for case in cases:
            slice_ids = batch_data[batch_data["case"] == case]["slice"].tolist()
            embedding_ids = batch_data[batch_data["case"] == case]["embedding_id"].tolist()
            # Read segmentations of instances for selected slices.
            instances = torch.from_numpy(self._read_instances(case, slice_ids=slice_ids))
            # Reading Embeddings
            embedding_path = os.path.join(self.path, case, "embedding.npz")
            # Memory efficient load of large compressed np arrays
            embedding = np.load(embedding_path, mmap_mode="r")
            # Keep only embeddings for sampled slices and convert to torch
            embeddings_sampled = embedding["embeddings"][embedding_ids]
            embedding.close()
            embeddings.append(embeddings_sampled)
            gt_masks.append(instances.squeeze(1))
        embeddings_torch = torch.as_tensor(np.vstack(embeddings), dtype=torch.float, device=self.device)
        gt_masks_torch =  torch.vstack(gt_masks) # We will move gt_masks later to the device
        if embeddings_torch.shape[0] != gt_masks_torch.shape[0]:
            raise ValueError(f"Embeddings and gt_masks must have same shape.")
        return embeddings_torch, gt_masks_torch 

    def _read_instances(self, case, slice_ids):
        # If full segmentation
        if self.instance_type in ["full", "ROI", "Stage2"]:
            seg_path = os.path.join(self.path, case, "segmentation.nii.gz")
            segment_final = self._read_single_instance(seg_path, slice_ids)
            # Return numpy array shape: Sx1xHxW
            return segment_final[:, None, :, :]

        # Read instances Helper-Function
        instances_path = os.path.join(self.path, case, "instances")
        # Sorted list of all avaliable instances
        instances = sorted(os.listdir(instances_path))
        # Get path of selected type of instances
        instance_files = [inst for inst in instances if inst.startswith(self.instance_type)]
        n_instances = sum([1 if inst.endswith("annotation-1.nii.gz") else 0 for inst in instance_files])

        # Case: if no instance avaliable
        if n_instances == 0:
            self.logger.info(f"No {self.instance_type} instance present for {case}.")
            return np.array([0])
        
        # We will always take the first annotation
        # Case: only one instance avaliable:
        if n_instances == 1:
            # Read the first annotation
            seg_path = os.path.join(instances_path, instance_files[0])
            segment_final = self._read_single_instance(seg_path, slice_ids)
            # Return numpy array shape: Sx1xHxW
            return segment_final[:, None, :, :]
    
        # Case: more than one instance avaliable:
        elif n_instances > 1:
            # Use only first annotations
            files = [inst for inst in instance_files if inst.endswith("annotation-1.nii.gz")]
            # Read segments
            segments = []
            for file in files:
                seg_path = os.path.join(instances_path, file)
                segment = self._read_single_instance(seg_path, slice_ids)
                segments.append(segment)
            return np.sum(segments, axis=0, dtype=np.uint8)[:, None, :, :]
        
    def _read_single_instance(self, seg_path, slice_ids):
        # Reads selected slices from a single segmentation file and scales array in range [0, 255].
        segment_file = nib.load(seg_path, mmap="r")
        segment = segment_file.get_fdata()[slice_ids]
        segment_file.uncache()
        # transform to uint8
        segment = cv2.convertScaleAbs(segment)
        # If resolution not 512x512 then resize
        if segment.shape[1:] != (512, 512):
            segment = (np.vstack([self.transform(se) for se in segment]) * 255).astype("uint8")
        if self.instance_type not in ["full", "Stage2"]:
            # if not multiclass segmentation turn to binary
            segment = np.clip(segment, None, 1)
        return segment

# Continue Training sampler
class ContinueTrainingSampler(Sampler):
    def __init__(self, data_source, start_index, end_index):
        self.data_source = data_source
        self.start_index = start_index
        self.end_index = end_index

    def __iter__(self):
        indices = range(self.start_index, self.end_index)
        return iter(indices)

    def __len__(self):
        return self.end_index - self.start_index
