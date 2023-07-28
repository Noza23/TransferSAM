if __name__ == "__main__":
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    os.chdir(parent_dir)
    import argparse
    from torchvision import transforms
    from segment_anything import sam_model_registry
    import nibabel as nib
    from TransferSAM.Predictor import Predictor
    import pickle
    
    reshape = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    # Create Parser
    parser = argparse.ArgumentParser(description='Predictor')
    parser.add_argument("--casesdir", type=str, help="Directory where the test cases are saved.", required=True)
    parser.add_argument("--case_start", type=int, help="Case to start from", required=True)
    parser.add_argument("--case_end", type=int, help="Case to end at [excluded]", required=True)
    parser.add_argument("--output_path", type=str, help="Output Directory", required=True)
    parser.add_argument("--device", type=str, help="device identifier [cpu, cuda:0, cuda:1]", required=True)
    args = parser.parse_args()
    case_dir = args.casesdir
    case_start = args.case_start
    case_end = args.case_end
    PRED_PATH = args.output_path

    cases = sorted([f for f in os.listdir(case_dir) if not f.startswith('.')])
    relevant_cases = cases[case_start:case_end]

    with open('./models/KiTSAM.pkl', 'rb') as file:
        kitsam = pickle.load(file)

    predictor = Predictor(kitsam.model_roi, kitsam.tumor_decoder, kitsam.cyst_decoder, threshold=0, seed=42, device=args.device)

    for cs in relevant_cases:
        case_path = os.path.join(case_dir, cs)
        immaging_nifti = nib.load(case_path)
        imagging = immaging_nifti.get_fdata()
        print(f"Starting predicting: {cs}")
        # Predict
        prediction = predictor.predict_case(imagging)
        # Prediction always return Sx512x512
        if imagging.shape[1:] != (512, 512):
            prediction = reshape(prediction).numpy().astype("uint8")
            print(f"Prediction has been reshaped to {imagging.shape[1:0]}")
        # Save Prediction
        prediction_nifti = nib.Nifti1Image(prediction, affine=immaging_nifti.affine)
        nib.save(prediction_nifti, os.path.join(PRED_PATH, cs))
        print(f"Prediction for {cs} has benn sucesfully saved in {PRED_PATH}")

