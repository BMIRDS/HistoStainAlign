# HistoStainAlign

![Framework](support/framework.png)

**HistoStainAlign** is the official repository for the paper:  
[Cross-Modality Learning for Predicting IHC Biomarkers from H&E-Stained Whole-Slide Images](https://www.arxiv.org/abs/2506.15853).

HistoStainAlign enables cross-modality prediction of immunohistochemistry (IHC) biomarkers from hematoxylin and eosin (H&E) stained whole-slide images (WSIs), leveraging deep learning and domain adaptation.

---

## Features

- **Tile Embedding Extraction**: Process WSIs to extract tile-level embeddings using Gigapath.
- **Flexible Training**: Train models with customizable classification heads.
- **Slide-Level Embedding Generation**: Aggregate tile embeddings for slide-level analysis.
- **Evaluation Tools**: Includes linear probe scripts for benchmarking embeddings.

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/BMIRDS/HistoStainAlign.git
    cd HistoStainAlign
    ```

2. **Install dependencies**
    - All Python dependencies are listed in `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
    - Additionally, install [Gigapath](https://github.com/prov-gigapath/prov-gigapath) and its dependencies:
        ```bash
        git clone https://github.com/prov-gigapath/prov-gigapath.git
        cd prov-gigapath
        pip install -r requirements.txt
        ```

---

## Usage

The workflow consists of four main steps:

1. **Extract tile embeddings**
    - `00_extract_tile_embeds.py`: Use Gigapath’s tile encoder to generate embeddings.
2. **Train the model**
    - `01_train_model_with_classification_head.py`: Train using the HistoStainAlign framework.
3. **Generate slide-level embeddings**
    - `02_generate_slide_embeddings.py`: Aggregate tile embeddings for each WSI.
4. **Evaluate with linear probe**
    - `03_run_linear_probe.py`: Assess slide embeddings using linear probing.

Refer to comments within each script for detailed usage and parameter options.

---

## Data

To use this repository, you will need access to appropriately formatted WSIs and IHC biomarker labels. Data preparation steps can be adapted from the scripts provided.

---

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.

---

## Citation

If you use this code or its ideas in your research, please cite:

```bibtex
@article{HistoStainAlign2024,
  title={Cross-Modality Learning for Predicting IHC Biomarkers from H&E-Stained Whole-Slide Images},
  author={Your Authors},
  journal={arXiv preprint arXiv:2506.15853},
  year={2024}
}
```