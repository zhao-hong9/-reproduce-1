# SS-MixNet

SS-MixNet is a TensorFlow/Keras implementation for hyperspectral image classification. The model combines spectral and spatial mixer blocks with depthwise attention to classify remote-sensing pixels and generate prediction maps.

## Project Files

- `Main_SS_MixNet.py` - trains SS-MixNet, evaluates the test set, displays prediction maps, and exports MATLAB `.mat` results.
- `Predicted.py` - loads trained SS-MixNet weights and generates prediction maps.

## Requirements

Install the Python packages used by the scripts:

```bash
pip install numpy scipy scikit-learn tensorflow keras matplotlib tqdm
```

This project also imports a local `utils.py` module:

```python
from utils import *
```

Make sure `utils.py` is available in the project directory or on your Python path. It should provide functions such as:

- `loadData`
- `img_display`
- `applyPCA`
- `get_img_indexes`
- `splitTrainTestSet`
- `createImageCubes`
- `predict_by_batching`
- `get_class_map`

## Dataset

The dataset is selected in both scripts with:

```python
DATASET = 'Pingan'  ## Pingan, Tangdaowan, Qingyun
```

Supported dataset names in the current code are:

- `Pingan`
- `Tangdaowan`
- `Qingyun`

The actual dataset loading logic is expected to be implemented in `utils.py` through `loadData(DATASET)`. The loaded values should be:

```python
data, gt, class_name = loadData(DATASET)
```

Where:

- `data` is the hyperspectral image cube.
- `gt` is the ground-truth label map.
- `class_name` is the list of class names.

## Training

Run:

```bash
python Main_SS_MixNet.py
```

The training script performs the following steps:

1. Loads the selected dataset.
2. Displays the ground-truth map.
3. Applies PCA with 15 components.
4. Splits labeled pixels into train, validation, and test sets.
5. Creates image cubes with a `9 x 9` spatial window.
6. Builds and trains SS-MixNet for 100 epochs.
7. Saves the best model weights.
8. Evaluates Overall Accuracy, Average Accuracy, and Kappa.
9. Generates prediction maps.
10. Saves the predicted class map as a MATLAB `.mat` file.

The model checkpoint is saved as:

```text
Pingan_SS_MixNet.h5
```

The filename changes according to the value of `DATASET`.

## Prediction

After training, run:

```bash
python Predicted.py
```

`Predicted.py` loads:

```text
{DATASET}_SS_MixNet.h5
```

Then it generates:

- Full predicted class map
- Masked predicted class map
- MATLAB output file

The output path is:

```text
Matlab_Outputs/{DATASET}/SS_MixNet.mat
```

Create the output folder before running prediction if it does not already exist.

## Model Overview

The SS-MixNet architecture includes:

- Two 3D convolution layers for local spectral-spatial feature extraction.
- A spectral mixer block for spectral feature interaction.
- A spatial mixer block for spatial feature interaction.
- Feature fusion by concatenation.
- Depthwise attention for feature recalibration.
- Global average pooling and a dense softmax classifier.

## Notes

- `Main_SS_MixNet.py` currently calls:

```python
model.load_weights(f"{DATASET}_SS_MixNet_{i}.h5")
```

However, `i` is not defined in the script. If you want to load the checkpoint saved during training, change it to:

```python
model.load_weights(f"{DATASET}_SS_MixNet.h5")
```

- The repository currently does not include `utils.py`, dataset files, or `Matlab_Outputs/` folders. Add them before running the scripts.

## Citation

If this code is based on a paper or thesis, add the citation information here.
