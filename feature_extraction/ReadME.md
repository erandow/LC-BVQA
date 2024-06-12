# Fine-Tune the Feature Extractor (FE) Model

To start extracting features using the FE model, run the following command. Make sure you follow the configuration steps beforehand:

```bash
python .\train_FE_model.py --dataset_path "path_to_dataset_images" --csv_path "path_to_mos_scores_csv"
```

# `Configuration Steps:`
## 1. `config.yaml`

This file contains all the hyperparameters and configurations needed for training. You can customize the fine-tuning process within this file.

| Hyperparameter                | Value      |
|-------------------------------|------------|
| `learning_rate`               | 0.001      |
| `batch_size`                  | 32         |
| `epochs`                      | 50         |
| `train_size`                  | 0.75       |
| `image_name_column_keyword`   | "image"    |
| `model.label_column_keyword`  | "label"    |

## 2. Preparing the Dataset

### `--dataset_path`

All images for training must be placed in a single folder. Sub-folders are not supported.

### `--csv_path`

The image scores should be saved in a CSV file. This file must have two columns: one for the image names and another for the labels.

- The key of each column can be changed from the `config.yaml` file.