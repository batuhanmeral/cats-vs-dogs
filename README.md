# Cats vs Dogs: End-to-End Image Classification with PyTorch

A learning-oriented, end-to-end binary image classifier (cat vs dog) built in a single Jupyter notebook, [`cats_vs_dogs.ipynb`](cats_vs_dogs.ipynb). Every section starts with a Turkish explanation of what the step does and why, and the code comments are in Turkish too.

The notebook covers:

1. **Data cleaning**: detects corrupt and non-JPEG files with PIL (`verify()` + `convert("RGB")`) and excludes them without touching the raw data.
2. **Stratified split**: 70 / 15 / 15 train/val/test, copied into `data/{train,val,test}/{cat,dog}`. `SUBSET_SIZE` controls how many images are used.
3. **EDA**: sample images, image size distribution, class balance.
4. **Data pipeline**: `ImageFolder` + `DataLoader` + `torchvision.transforms.v2` with ImageNet normalization.
5. **Model 1**: CNN from scratch (4 × Conv-BN-ReLU-MaxPool → AdaptiveAvgPool → Linear) with `BCEWithLogitsLoss`, a hand-written training loop, checkpointing, and early stopping.
6. **Model 2**: the same CNN with data augmentation and dropout.
7. **Model 3**: transfer learning with MobileNetV2: a frozen backbone with a new head first, then fine-tuning of the last blocks with a lower learning rate. BatchNorm layers stay in eval mode throughout.
8. **Evaluation**: a single pass over the test set, a comparison table, a confusion matrix, a classification report, and the misclassified images.
9. **Inference**: `predict_image(path)` plus saving and reloading the model as a `state_dict`.
10. **Conclusion**.

## Results

Run with `SUBSET_SIZE = 6000` (4,200 train / 900 val / 900 test images) and `seed = 42` on a local NVIDIA RTX 3060. Expect small differences on other hardware such as a Colab T4.

| Model | Input | Val accuracy | Test accuracy | Trainable params | Epochs |
|---|---|---|---|---|---|
| Model 1: CNN from scratch | 128×128 | 86.9% | 84.1% | 389,153 | 30 |
| Model 2: CNN + augmentation + dropout | 128×128 | 86.2% | 87.2% | 389,153 | 40 |
| Model 3: MobileNetV2 transfer learning | 224×224 | 99.0% | **98.4%** | 1,207,361 (of 2.2M) | 5 + 3 |

The best model was selected by validation accuracy, not test accuracy.

## Dataset

[Kaggle: Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) (~25,000 JPEGs).

Place the `PetImages` folder next to the notebook:

```
cats-vs-dogs/
├── cats_vs_dogs.ipynb
└── PetImages/
    ├── Cat/
    └── Dog/
```

If `PetImages/` is not found, the notebook downloads the dataset automatically with `kagglehub`. To download it manually:

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('bhavikjikadara/dog-and-cat-classification-dataset'))"
```

The command prints the download path. Copy (or symlink) the `PetImages` folder from that path into the project root. If `kagglehub` asks for credentials, run `kagglehub.login()` or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

## Running

**Google Colab (recommended):** upload or open `cats_vs_dogs.ipynb`, select *Runtime → Change runtime type → T4 GPU*, then *Run all*. All dependencies are preinstalled on Colab.

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # for a CUDA build of PyTorch, see https://pytorch.org/get-started/locally/
jupyter notebook cats_vs_dogs.ipynb
```

Set `SUBSET_SIZE = None` in the settings cell to train on the full dataset.

Generated artifacts (`data/`, `models/*.pth`) are git-ignored.

## License

[MIT](LICENSE). The dataset is distributed separately on Kaggle under its own terms.
