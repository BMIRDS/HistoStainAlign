# General imports
import os
import sys
from pathlib import Path
import h5py
import pandas as pd
import argparse

# Torch imports
import torch

# Internal imports (Gigapath)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../pdl1_project/prov-gigapath'))

from gigapath.pipeline import run_inference_with_slide_encoder
import gigapath.slide_encoder as slide_encoder

def read_h5_file(file_path, device): 
    """
    Returns features and coords given a path for a .h5 file
    Args:
        file_path (str): Path to the h5 file
        device (torch.device): Device to store the tensors
    Returns:
        features_tensor (torch.Tensor): Tensor of features
        coords_tensor (torch.Tensor): Tensor of coordinates
    """   
    
    with h5py.File(file_path, 'r') as h5_file:    
        coords, features = None, None
        def get_features_and_coords(name, obj):
            nonlocal coords, features 
            if isinstance(obj, h5py.Dataset):
                if 'coords' in name:
                    coords = obj[:]
                if 'features' in name:
                    features = obj[:]
        h5_file.visititems(get_features_and_coords)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
        return features_tensor, coords_tensor

def get_model(model_path):
    """
    Returns adjusted model state_dict given a model path
    Args:
        model_path (str): Path to the model
    Returns:
        new_checkpoint (dict): the new model state_dict
    """
    
    checkpoint = torch.load(model_path)
    # Adjust the state_dict keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # If the checkpoint itself is the state_dict
    adjusted_state_dict = {}
    for key, value in state_dict.items():
        # Remove the 'module.slide_encoder.' prefix
        new_key = key.replace('module.slide_encoder.', '')
        adjusted_state_dict[new_key] = value
    # Create a new checkpoint with only the 'model' key
    new_checkpoint = {'model': adjusted_state_dict}
    return new_checkpoint

def get_HE_slides_pdl1(args, df_info):
    """
    Returns a list of all paths to tile embeds of PD-L1 H&E slides
    Args:
        args (argparse.Namespace): Arguments
        df_info (pd.DataFrame): Dataframe with slide information
    Returns:
        HandE_slides_tile_embeds (list): List of paths to tile embeds
        for all PD-L1 H&E slides in the dataframe
    """

    HandE_slides_tile_embeds = []
    data_dir = Path(args.tile_embed_dir)
    for index, row in df_info.iterrows():
        he_slide = row['slide_id'].split(".")[0]
        he_path = data_dir / f"{he_slide}.h5"
        if he_path.exists():
            HandE_slides_tile_embeds.append(he_path)
        else:
            print(f"[INFO] {he_path} does not exist")
    return HandE_slides_tile_embeds

def get_HE_slides_ki67(args, df_info):
    """
    Returns a list of all paths to tile embeds of Ki-67 H&E slides
    Args:
        args (argparse.Namespace): Arguments
        df_info (pd.DataFrame): Dataframe with slide information
    Returns:
        HandE_slides_tile_embeds (list): List of paths to tile embeds
        for all Ki-67 H&E slides in the dataframe
    """

    HandE_slides_tile_embeds = []
    data_dir = Path(args.tile_embed_dir)
    for index, row in df_info.iterrows():
        he_slide = row['slide_id'].split(".")[0]
        he_path = data_dir / f"{he_slide}.h5"

        if he_path.exists():
            HandE_slides_tile_embeds.append(he_path)
        else:
            print(f"[INFO] {he_path} does not exist")

    return HandE_slides_tile_embeds

def get_HE_slides_p53(args, df_info):
    """
    Returns a list of all paths to tile embeds of p53 H&E slides
    Args:
        args (argparse.Namespace): Arguments
        df_info (pd.DataFrame): Dataframe with slide information
    Returns:
        HandE_slides_tile_embeds (list): List of paths to tile embeds
        for all p53 H&E slides in the dataframe
    """

    HandE_slides_tile_embeds = []
    data_dir = Path(args.tile_embed_dir)
    for index, row in df_info.iterrows():
        he_slide = row['slide_id'].split(".")[0]
        he_path = data_dir / f"{he_slide}.h5"
        if he_path.exists():
            HandE_slides_tile_embeds.append(he_path)
        else:
            print(f"[INFO] {he_path} does not exist")
    return HandE_slides_tile_embeds

def process_and_save_slide_embeds(HE_slides, model, output_dir, fold, model_path, device, split):
    """
    Processes and saves slide embeddings for a list of H&E slides
    Args:
        HE_slides (list): List of paths to H&E slides
        model (torch.nn.Module): Slide encoder model
        output_dir (str): Directory to save slide embeddings
        fold (int): Fold number
        model_path (str): Path to the model (slide encoder)
        device (torch.device): Device to store the tensors
        split (str): Train, Val or Test
    Returns: None
    """
    for he_slide in HE_slides:
        he_embedding, he_coords = read_h5_file(he_slide, device)
        slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=model, 
                                                        tile_embeds = he_embedding, coords = he_coords)['last_layer_embed']

        output_path = output_dir / f"{fold}" / model_path.stem / split / f"{he_slide.stem}.pt"
        os.makedirs(output_path.parent, exist_ok=True)

        # save slide embeddings
        torch.save(slide_embeds, output_path)
        print(f"[INFO] Saved slide embeddings to: {output_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate slide embeddings for PD-L1 H&E slides')
    parser.add_argument('--folds', type=list, default=[0, 1, 2, 3, 4], help='List of folds to generate slide embeddings for')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory consisting of model checkpoints')
    parser.add_argument('--dataset_csv', type=str, help='Path to the train csv file')
    parser.add_argument('--output_dir', type=str, help='Directory to save slide embeddings')
    parser.add_argument('--study', type=str, choices=['pdl1', 'p53', 'ki67'], help='Study to generate slide embeddings for')
    parser.add_argument('--model_name', type=str, help='Name of TANGLE model checkpoint')
    parser.add_argument('--tile_embed_dir', type=str, help='Directory where tile embeddings are stored')
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in args.folds:
        # obtaining paths to all models in the fold
        model_paths = [Path(f"{args.checkpoint_dir}/fold_{fold}/{args.model_name}"), Path('slide_encoder.pth')]
        print(f"[INFO] Models: {model_paths}")

        # obtaining paths to all PD-L1 H&E slides
        df_info_train = pd.read_csv(f"{args.dataset_csv}/train_{fold}.csv")
        df_info_val = pd.read_csv(f"{args.dataset_csv}/val_{fold}.csv")
        df_info_test = pd.read_csv(f"{args.dataset_csv}/test_{fold}.csv")

        if args.study == 'pdl1':
            HE_slides_train = get_HE_slides_pdl1(args, df_info_train)
            HE_slides_val = get_HE_slides_pdl1(args, df_info_val)
            HE_slides_test = get_HE_slides_pdl1(args, df_info_test)
        elif args.study == 'p53':
            HE_slides_train = get_HE_slides_p53(args, df_info_train)
            HE_slides_val = get_HE_slides_p53(args,df_info_val)
            HE_slides_test = get_HE_slides_p53(args, df_info_test)
        elif args.study == 'ki67':
            HE_slides_train = get_HE_slides_ki67(args, df_info_train)
            HE_slides_val = get_HE_slides_ki67(args, df_info_val)
            HE_slides_test = get_HE_slides_ki67(args, df_info_test)

        for model_path in model_paths:
            new_checkpoint = get_model(model_path)
            if model_path.name != 'slide_encoder.pth':
                model_save_path = model_path.parent / f"{model_path.stem}_adjusted.pt"
                torch.save(new_checkpoint, model_save_path)
                print(f"[INFO] Adjusted Model saved at: {model_save_path}")
                model = slide_encoder.create_model(str(model_save_path), "gigapath_slide_enc12l768d", 1536, global_pool=True)
            else:
                # Loading the original slide encoder model
                model = slide_encoder.create_model(str(model_path), "gigapath_slide_enc12l768d", 1536, global_pool=True)

            # saving the slide embeddings
            process_and_save_slide_embeds(HE_slides_train, model, Path(args.output_dir), fold, model_path, DEVICE, 'train')
            process_and_save_slide_embeds(HE_slides_val, model, Path(args.output_dir), fold, model_path, DEVICE, 'val')
            process_and_save_slide_embeds(HE_slides_test, model, Path(args.output_dir), fold, model_path, DEVICE, 'test')
