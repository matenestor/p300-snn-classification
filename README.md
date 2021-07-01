### Description

Four kinds of neural networks to classify target and non-target numbers from recorder P300 brain waves. Networks are both artificial and spiking (3rd generation of neural networks). Target number is a chosen number from range 1 to 9 by examined participant. Non-target is every other number from the range. The numbers were shown to a participant, who then focused on the number, which created event-related potential, that the networks are classifying.

Dataset is available at:
https://doi.org/10.7910/DVN/G9RRLN

### Usage

Open jupyter notebook server with input `jupyter notebook` into terminal from a directory with `p300-snn-classification.ipynb` file. Or run the file `p300-snn-classification.py` directly from a terminal.

#### Parameters

You can adjust simulation parameters for testing.

- `SPLITS` -- amount of cross-validation splits
- `SPLIT_SIZE` -- percentage for train-test dataset split
- `BATCH` -- size of batches for training and testing
- `EPOCHS` -- number of epochs to run training
- `AVERAGE_SAMPLES` -- `True` for averaging random brainwave signal samples among people. `False` for averaging single brainwave signal of one person (smoothening).
- `AVERAGING_AMOUNT` -- list with amount of samples to be averaged
- `TRAIN_DATA_AMOUNT` -- amount of new training dataset size (applied only when is `AVERAGE_SAMPLES` True)

### Python requirements

- numpy==1.19
- pandas==1.2.5
- sicpy==1.7.0
- scikit-learn==0.24.2
- matplotlib==3.4.1
- notebook==6.3.0
- nengo==3.1.0
- nengo-dl==3.4.1
- tensorflow==2.5.0
