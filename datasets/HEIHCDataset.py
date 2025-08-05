# Standard library imports
import h5py

# Torch Imports
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def read_h5_file(file_path): 
    """
    Reads a h5 files for a given path
    Args:
        file_path (str): Path to the h5 file
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
        features_tensor = torch.tensor(features, dtype=torch.float32)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        return features_tensor, coords_tensor

class HEIHCDataset(Dataset):
    """Custom Dataset to handle paired H&E and IHC embeddings"""

    def __init__(self, tile_embeds_dir, he_ihc_pairs, n_tokens, df_info, study):
        """
        Args:
            data_dir (Path): Path to the slide embeddings directory.
            he_ihc_pairs (list of tuples): List of (H&E filename, IHC filename) pairs.
        """
        self.tile_embeds_dir = tile_embeds_dir
        self.he_ihc_pairs = he_ihc_pairs
        self.n_tokens = n_tokens
        self.df_info = df_info
        self.study = study

    def __len__(self):
        return len(self.he_ihc_pairs)

    def __getitem__(self, idx):
        he_filename, ihc_filename = self.he_ihc_pairs[idx]

        # specific handling for different stains to get the label
        if self.study == "p53":
            label = self.df_info[self.df_info['svs_file'] == f"{he_filename}.svs"]['label'].values[0]
        elif self.study == "pdl1":
            label = self.df_info[self.df_info['he_slide_name'] == f"{he_filename}.svs"]['label'].values[0]
        elif self.study == "ki67":
            label = self.df_info[self.df_info['HE_Slide'] == f"{he_filename}.svs"]['label'].values[0]
            
        # Load H&E and IHC embeddings from .pt files
        he_path = self.tile_embeds_dir / f"{he_filename}.h5"
        ihc_path = self.tile_embeds_dir / f"{ihc_filename}.h5"
        he_embedding, he_coords = read_h5_file(he_path)
        ihc_embedding, ihc_coords = read_h5_file(ihc_path)

        # Randomly sample n_tokens from the H&E embeddings
        HandE_patch_indices = torch.randint(0, he_embedding.size(0), 
                                            (self.n_tokens,)).tolist() if he_embedding.shape[0] < self.n_tokens else torch.randperm(
                                                he_embedding.size(0))[:self.n_tokens].tolist()   

        he_embedding = he_embedding[HandE_patch_indices]
        he_coords = he_coords[HandE_patch_indices] 

        # Randomly sample n_tokens from the IHC embeddings
        IHCPatchIndices = torch.randint(0, ihc_embedding.size(0),
                                        (self.n_tokens,)).tolist() if ihc_embedding.shape[0] < self.n_tokens else torch.randperm(
                                            ihc_embedding.size(0))[:self.n_tokens].tolist()
        ihc_embedding = ihc_embedding[IHCPatchIndices]
        ihc_coords = ihc_coords[IHCPatchIndices]

        return he_embedding, ihc_embedding, he_coords, ihc_coords, label
    