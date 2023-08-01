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

if __name__ == "__main__":
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    sys.path.append(parent_dir)
    from segment_anything import sam_model_registry    
    import argparse
    from TransferSAM import save_KiTSAM

    # Create Parser
    parser = argparse.ArgumentParser(description='Combine Models')
    parser.add_argument("--sam_base", type=str, help="Path to SAM_base checkpoint for image encoder", required=True)
    parser.add_argument("--roi_model", type=str, help="Path to ROI model checkpoint", required=True)
    parser.add_argument("--tumor_model", type=str, help="Path to tumor model checkpoint", required=True)
    parser.add_argument("--cyst_model", type=str, help="Path to cyst model checkpoint", required=True)

    args = parser.parse_args()

    kitsam = KiTSAM(
        sam_base=args.sam_base,
        roi=args.roi_model,
        tumor=args.tumor_model,
        cyst=args.cyst_model
    )
    save_KiTSAM(kitsam, './models/KiTSAM.pkl')