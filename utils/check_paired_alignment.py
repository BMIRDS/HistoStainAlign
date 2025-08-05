import torch
import pandas as pd
import os
import ast
import sys
import h5py
import argparse
from pathlib import Path
import random
import numpy as np
import scipy.stats
from scipy.stats import wilcoxon

# fix random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)

# Internal imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../pdl1_project/prov-gigapath'))

from gigapath.pipeline import run_inference_with_slide_encoder
import gigapath.slide_encoder as slide_encoder

def mean_confidence_interval(data, confidence=0.95):
    """For a given list of data, returns mean and confidence interval"""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

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

def load_paired_embeddings(model, matching_pairs, device, data_dir):
    """
    Returns list of extracted features from H&E and their paired IHC slides.
    Args:
        model (torch.nn.Module): Model to extract embeddings
        matching_pairs (list): List of matching pairs of H&E and IHC slides
        device (torch.device): Device to store the tensors
        data_dir (str): Path to the directory containing the tile embeddings
    Returns:
        zip(features_HandE, features_IHC): List of extracted features from H&E and their paired IHC slides
    """
    model.eval()
    features_HandE, features_IHC = [], []
    for pair in matching_pairs:
        print(f"Processing pair: {pair}")
        # Load HandE embeddings
        he_embedding, he_coords = read_h5_file(f"{data_dir}/{pair[0]}.h5", device)
        HE_embedding = run_inference_with_slide_encoder(slide_encoder_model=model,
                                                        tile_embeds = he_embedding, coords = he_coords)['last_layer_embed']
        features_HandE.append(HE_embedding)

        # Load IHC embeddings
        ihc_embedding, ihc_coords = read_h5_file(f"{data_dir}/{pair[1]}.h5", device)
        IHC_embedding = run_inference_with_slide_encoder(slide_encoder_model=model,
                                                        tile_embeds = ihc_embedding, coords = ihc_coords)['last_layer_embed']
        features_IHC.append(IHC_embedding)

    return features_HandE, features_IHC

def get_matching_pairs_p53(df_info):
    """Returns list of matching pairs of p53 H&E and IHC slides"""
    HE_slides, IHC_slides = [], []
    for index, row in df_info.iterrows():
        matching_p53_list = ast.literal_eval(row['Matching p53'])
        if matching_p53_list:
            IHC_slides.append(matching_p53_list[0].split(".")[0])
            HE_slides.append(row['svs_file'].split(".")[0])
    return list(zip(HE_slides, IHC_slides))

def get_matching_pairs_pdl1(df_info, data_dir):
    """Returns list of matching pairs of pdl1 H&E and IHC slides"""
    HE_slides, IHC_slides = [], []
    for index, row in df_info.iterrows():
        he_slide, ihc_slide = row['he_slide_name'].split(".")[0], row['ihc_slide_name'].split(".")[0]

        # check if tile embeddings are present
        he_path = Path(data_dir) / f"{he_slide}.h5"
        ihc_path = Path(data_dir) / f"{ihc_slide}.h5"

        if not he_path.exists() or not ihc_path.exists():
            print(f"[INFO] Missing tile embeddings for slide {he_slide} or {ihc_slide}. Skipping...")
            continue

        HE_slides.append(he_slide)
        IHC_slides.append(ihc_slide)

    return list(zip(HE_slides, IHC_slides))

def get_matching_pairs_ki67(df_info, data_dir):
    """Returns list of matching pairs of ki67 H&E and IHC slides"""
    HE_slides, IHC_slides = [], []
    for index, row in df_info.iterrows():
        he_slide, ihc_slide = row['HE_Slide'].split(".")[0], row['IHC_Slide'].split(".")[0]

        # check if tile embeddings are present
        he_path = data_dir / f"{he_slide}.h5"
        ihc_path = data_dir / f"{ihc_slide}.h5"

        if not he_path.exists() or not ihc_path.exists():
            print(f"[INFO] Missing tile embeddings for slide {he_slide} or {ihc_slide}. Skipping...")
            continue

        HE_slides.append(he_slide)
        IHC_slides.append(ihc_slide)
    return list(zip(HE_slides, IHC_slides))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check paired alignment')
    parser.add_argument('--study', type=str, help='Study to check paired alignment')
    parser.add_argument('--checkpoint_dir_base', type=str, help='Path to the checkpoint directory')
    parser.add_argument('--checkpoint_dir_new', type=str, help='Path to the checkpoint directory')
    parser.add_argument('--model', type=str, help='Model to load')
    parser.add_argument('--dataset_csv', type=str, help='Path to the dataset csv')
    parser.add_argument('--data_dir', type=str, help='Path to the tile embeddings directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--info_csv', type=str, help='Path to the info csv')
    args = parser.parse_args()

    # Global variables
    DF_INFO = pd.read_csv(args.info_csv)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_paired_base = []
    all_shuffled_base = []
    all_diff_base = []
    all_paired_new = []
    all_shuffled_new = []
    all_diff_new = []

    for fold in range(args.folds):
        # Load models
        base_model = slide_encoder.create_model(
            f"{args.checkpoint_dir_base}/fold_{fold}/{args.model}",
            "gigapath_slide_enc12l768d", 1536, global_pool=True
        )
        new_model = slide_encoder.create_model(
            f"{args.checkpoint_dir_new}/fold_{fold}/{args.model}",
            "gigapath_slide_enc12l768d", 1536, global_pool=True
        )

        # Get matching pairs from the test set
        if args.study == "p53":
            df_info_test = pd.read_csv(f"{args.dataset_csv}/test_{fold}.csv")
            df_info_test = df_info_test.rename(columns={'slide_id': 'svs_file'})
            df_info_test = pd.merge(df_info_test, DF_INFO[['svs_file', 'Matching p53']], on='svs_file', how='left')
            HE_IHC_pairs_test = get_matching_pairs_p53(df_info_test)
        elif args.study == "pdl1":
            df_info_test = pd.read_csv(f"{args.dataset_csv}/test_{fold}.csv")
            df_info_test = df_info_test.rename(columns={'slide_id': 'he_slide_name'})
            df_info_test = pd.merge(df_info_test, DF_INFO[['he_slide_name', 'ihc_slide_name']], on='he_slide_name', how='left')
            HE_IHC_pairs_test = get_matching_pairs_pdl1(df_info_test, args.data_dir)
        elif args.study == "ki67":
            df_info_test = pd.read_csv(f"{args.dataset_csv}/test_{fold}.csv")
            df_info_test = df_info_test.rename(columns={'slide_id': 'HE_Slide'})
            df_info_test = pd.merge(df_info_test, DF_INFO[['HE_Slide', 'IHC_Slide']], on='HE_Slide', how='left')
            HE_IHC_pairs_test = get_matching_pairs_ki67(df_info_test, Path(args.data_dir))

        # Load embeddings
        features_HandE_base, features_IHC_base = load_paired_embeddings(
            base_model, HE_IHC_pairs_test, DEVICE, args.data_dir
        )
        features_HandE_new, features_IHC_new = load_paired_embeddings(
            new_model, HE_IHC_pairs_test, DEVICE, args.data_dir
        )

        # Calculate per-sample differences
        for i in range(len(HE_IHC_pairs_test)):
            # Base model calculations
            he_base = features_HandE_base[i]
            true_ihc_base = features_IHC_base[i]

            paired_sim_base = torch.nn.functional.cosine_similarity(
                he_base, true_ihc_base, dim=1
            ).item()
            all_paired_base.append(paired_sim_base)

            shuffled_sims_base = []
            for j in range(len(features_IHC_base)):
                if j != i:
                    shuffled_sim = torch.nn.functional.cosine_similarity(
                        he_base, features_IHC_base[j], dim=1
                    ).item()
                    shuffled_sims_base.append(shuffled_sim)

            avg_shuffled_base = np.mean(shuffled_sims_base) if shuffled_sims_base else 0
            all_shuffled_base.append(avg_shuffled_base)
            all_diff_base.append(paired_sim_base - avg_shuffled_base)

            # New model calculations
            he_new = features_HandE_new[i]
            true_ihc_new = features_IHC_new[i]

            paired_sim_new = torch.nn.functional.cosine_similarity(
                he_new, true_ihc_new, dim=1
            ).item()
            all_paired_new.append(paired_sim_new)

            shuffled_sims_new = []
            for j in range(len(features_IHC_new)):
                if j != i:
                    shuffled_sim = torch.nn.functional.cosine_similarity(
                        he_new, features_IHC_new[j], dim=1
                    ).item()
                    shuffled_sims_new.append(shuffled_sim)
            avg_shuffled_new = np.mean(shuffled_sims_new) if shuffled_sims_new else 0
            all_shuffled_new.append(avg_shuffled_new)
            all_diff_new.append(paired_sim_new - avg_shuffled_new)

    print(f"[INFO] Fold-wise results:")

    # Reporting results
    print("\n=== Baseline Model ===")
    print(f"Paired similarities: {mean_confidence_interval(all_paired_base)}")
    print(f"Shuffled similarities: {mean_confidence_interval(all_shuffled_base)}")
    print(f"Alignment scores (paired - shuffled): {mean_confidence_interval(all_diff_base)}")

    print("\n=== New Model ===")
    print(f"Paired similarities: {mean_confidence_interval(all_paired_new)}")
    print(f"Shuffled similarities: {mean_confidence_interval(all_shuffled_new)}")
    print(f"Alignment scores (paired - shuffled): {mean_confidence_interval(all_diff_new)}")

    # Statistical comparison
    print("\n=== Statistical Comparison ===")
    stat, p_value = wilcoxon(all_diff_base, all_diff_new, alternative='less')
    print(f"Wilcoxon signed-rank test: statistic={stat:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference: New model has better alignment (p < 0.05)")
    else:
        print("No significant difference between models")