# Standard library imports
import os
import sys

# Third party imports
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py

# prov-gigapath repo
sys.path.append(os.path.join(os.path.dirname(__file__), '../prov-gigapath'))
from gigapath.pipeline import run_inference_with_tile_encoder, load_tile_slide_encoder

def save_to_h5(file_path, embeddings, coordinates):
    # create file_path if it doesn't exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('features', data=embeddings.numpy())
        f.create_dataset('coords', data=coordinates.numpy())

def extract_tile_embeddings(slide_path: Path, tile_encoder):
    extensions = ["*.jpeg", "*.jpg", "*.JPEG", "*.JPG"]
    image_paths = [str(p) for ext in extensions for p in slide_path.rglob(ext)]
    tile_encoder_outputs = run_inference_with_tile_encoder(image_paths, tile_encoder)
    return tile_encoder_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract embeddings from tiles.')
    parser.add_argument('--patch_dir', type=str, required=True, help='Path to the patches.')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to save the H5 files. Recommend: data/<project_name>_embeddings')
    args = parser.parse_args()
    
    # derived parameters
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_paths = list(Path(args.patch_dir).iterdir())

    tile_encoder, _ = load_tile_slide_encoder(global_pool=True)
        
    for slide_path in tqdm(slide_paths, desc="Processing slides"):
        print(f"Processing slide: {slide_path}")
        outputs = extract_tile_embeddings(slide_path, tile_encoder)
        out_file = out_dir / f"{slide_path.stem}.h5"

        if out_file.exists():
            print(f"Skipping {slide_path.stem} as it already exists.")
            continue

        embeddings, coords = outputs['tile_embeds'], outputs['coords']
        save_to_h5(out_file, embeddings, coords)

        # read the saved file
        with h5py.File(out_file, 'r') as f:
            assert f['features'].shape == embeddings.shape
            assert f['coords'].shape == coords.shape
