from .KiTS_dataset import KiTSdata, ContinueTrainingSampler
from  .Predictor import Predictor
from  .Trainer import Trainer, CosineScheduler
import pickle

class KiTSAM():
    """
    Class to combine 3 Models into a single KiTSAM instance.

    Attributes:
        ROI (str): Path to roi model checkpoint.
        tumor (str): Path to tumor model checkpoint.
        cyst (str): Path to cyst model checkpoint.

    Methods:
        __init__: Initializes a new KiTSAM instance.
    """
    def __init__(
        self,
        sam_base: str,
        roi: str,
        tumor: str,
        cyst: str
    ):
        self.sam_base = sam_model_registry["vit_b"](sam_base)
        self.model_roi = sam_model_registry["vit_b"](roi).mask_decoder
        self.tumor_decoder = sam_model_registry["vit_b"](tumor).mask_decoder
        self.cyst_decoder = sam_model_registry["vit_b"](cyst).mask_decoder


def save_KiTSAM(kitsam, path:str) -> None:
    with open(path, 'wb') as file:
        pickle.dump(kitsam, file)
    print(f"KiTSAM.pkl has benn saved in {path} directory.")

def load_KiTSAM(path:str):
    with open(path, 'rb') as file:
        kitsam = pickle.load(file)
    return kitsam
