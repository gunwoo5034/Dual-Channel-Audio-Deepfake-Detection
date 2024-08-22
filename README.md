# Dual-Channel-Audio-Deepfake-Detection

This code is modified from [MarHershey's Repository](https://github.com/MarkHershey/AudioDeepFakeDetection.git). 

## Set up Environment

```bash
# Python virtual enviroment
conda create -n AudioDetection python=3.8

# Install required
pip install -r requirements.txt
```


## Set up Dataset

You may download the datasets from following URLs:

- ASVspoof 2019 : [Link](https://www.asvspoof.org/index2019.html)
- FakeAVCeleb : [Link](https://sites.google.com/view/fakeavcelebdash-lab/)
- SPC(Self Colleted) : Not published

After downloading dataset, you may set them under `Data/train` and `Data/test`. Dataset folder should look like:

```
Data
├── train
│   ├── real
│   │   └── real wavs
│   └── fake
│       └── fake wavs
└── test
    ├── real
    │   └── real wavs
    └── fake
        └── fake wavs
```


## Training

Use the [`train.py`](train.py) to train the model.

```
usage: train.py [--train_dir TRAIN_DIR] [--test_dir TEST_DIR] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                [--seed SEED] [--feature_classname {wave,lfcc,mfcc}] [--model_classname {WaveRNN, TSSD, RawNet, MLP, ShallowCNN, LCNN}]
                [--device DEVICE] [--deterministic] [--restore] [--eval_only] [--channel {True, False}] [--model_len {1,3,6,10,30}] 
                [--amount_of_use AMOUNT_OF_USE]

optional parser argument:
--train_dir, --train     Directory containing train data. (default: 'data/train')
--test_dir , --test      Directory containing test data. (default: 'data/test')
--batch_size             Batch_size. (default: 64)
--epochs                 Number of maximum epochs to train. (default: 50)
--seed                   Random seed. (default: 1234)
--feature_classname      Feature classname. (default: 'lfcc')
--modle_classname        Model classname. (default: 'ShallowCNN')
--device                 Device to use. (default: 'cuda' if possible)
--deterministic          Whether to use deterministic training (default: True)
--restore                Whether to restore from checkpoint
--eval_only              Whether to evaluate only
--channel                Original sound or D/R 2ch(True: Dual channel, False: single channel)
--model_len              length of model in train(30/10/6/3/1) (default: 6)
--amount_of_use          Only some of the sample are used (default: 160)
```

### Label
You should prepare label.txt for label of your dataset. or you can modify `load_directory_split_train_test` in `Dataloader.py` fitting on your datasets. 


Example:

To train the model `ShallowCNN` with `lfcc` feature for dual channel,
you can run the following command:
```
python train.py --train_dir data/train --batch_size 64 --epoch 50 --feature_classname lfcc --model_classname ShallowCNN --channel True
```

Please use inline environment variable `CUDA_VISIBLE_DEVICES` to specify the GPU deivce(s) to use. For example:
```
CUDA_VISIBLE_DEVICE=0 python train.py
```


## Evaluation


After Training, the best model and best predictions will be saved in th `saved` directory. Go to the directory `saved` to see the evaluation results

To evaluate on the test set using trained model, you can run the following command:
```
python train.py --test_dir data/test --feature_classname lfcc --model_classname ShallowCNN --restore --eval_only --channel True
```

Run the following command to compute the evaluation results based on saved predictions and labels:
```
python metrics.py
```


