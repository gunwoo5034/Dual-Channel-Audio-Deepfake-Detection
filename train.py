import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from puts import printc, timestamp_seconds
from torch.utils.data import ConcatDataset
from torchinfo import summary

from DataLoader import lfcc, load_directory_split_train_test, mfcc
from models.cnn import ShallowCNN
from models.lstm import SimpleLSTM, WaveLSTM
from models.mlp import MLP
from models.rnn import WaveRNN
from models.tssd import TSSD
from models.rawnet2 import RawNet
from trainer import ModelTrainer
from utils import set_seed_all
from models.cnn import LCNN
warnings.filterwarnings("ignore")
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)


# all feature classnames
FEATURE_CLASSNAMES: Tuple[str] = ("wave", "lfcc", "mfcc")
# all model classnames
MODEL_CLASSNAMES: Tuple[str] = (
    "MLP",
    "WaveRNN",
    "WaveLSTM",
    "SimpleLSTM",
    "ShallowCNN",
    "TSSD",
    "RawNet",
    "LCNN"
)



def init_logger(log_file: Union[Path, str]) -> None:
    # create file handler
    fh = logging.FileHandler(log_file)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # clear handlers
    LOGGER.handlers = []
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)
    return None


def train(
    exp_name : str,
    train_dir: Union[Path, str],
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    batch_size: int = 32,
    save_dir: Union[str, Path] = None,
    train_size: float = 0.8,
    feature_classname: str = "wave",
    model_classname: str = "SimpleLSTM",
    amount_to_use: int = 160,
    checkpoint=None,
    channel :bool = False,
    model_len : int = 6,
) -> None:
    if model_len == 6:
        time = 481
        linear_input = 7424 
    elif model_len == 10:
        time = 801
        linear_input = 12544
    else:
        time = 2401
        linear_input = 38144

    input_channel = 2 if channel else 1
    d_args = {"nb_samp": 16000 * model_len * input_channel,
            "first_conv": 1024 ,  # no. of filter coefficients 
            "in_channels": 1,
            "filts": [20, [20, 20], [20, 128], [128, 128]], # no. of filters channel in residual blocks
            "blocks": [2, 4],
            "nb_fc_node": 1024,
            "gru_node": 1024,
            "nb_gru_layer": 3,
            "nb_classes": 2
    }
    
    KWARGS_MAP: Dict[str, dict] = {
    "SimpleLSTM": {
        "lfcc": {"feat_dim": 40, "time_dim": time *input_channel, "mid_dim": 30, "out_dim": 1}, #6초 (40,1443) 10초(40,2403)
        "mfcc": {"feat_dim": 40, "time_dim": time *input_channel, "mid_dim": 30, "out_dim": 1},
    },
    "ShallowCNN": {
        "lfcc": {"in_features": input_channel , "out_dim": 1 , "linear_input": linear_input},
        "mfcc": {"in_features": input_channel , "out_dim": 1 , "linear_input": linear_input},
    },
    "MLP": {
        "lfcc": {"in_dim": 40 * time *input_channel, "out_dim": 1},
        "mfcc": {"in_dim": 40 * time *input_channel, "out_dim": 1},
    },
    "TSSD": {
        "wave": {"in_dim": 16000 * model_len , "in_channels": input_channel},
    },
    "WaveRNN": {
        "wave": {"num_frames": 10, "input_length": 16000 * model_len , "hidden_size": 500},
    },
    "WaveLSTM": {
        "wave": {
            "num_frames": 20,
            "input_len": 16000 * model_len * input_channel,
            "hidden_dim": 50,
            "out_dim": 1,
        }
    },
    "RawNet" : {
        "wave" : {"d_args":d_args}
    },
    "LCNN" : {
        "lfcc" : {"in_channels" : input_channel,},
        "mfcc" : {"in_channels" : input_channel,}
    }
}
    feature_classname = feature_classname.lower()
    assert feature_classname in FEATURE_CLASSNAMES
    assert model_classname in MODEL_CLASSNAMES

    # get feature transformation function
    feature_fn = None if feature_classname == "wave" else eval(feature_classname)
    assert feature_fn in (None, lfcc, mfcc)
    # get model constructor
    Model = eval(model_classname)
    assert Model in (SimpleLSTM, ShallowCNN, WaveLSTM, MLP, TSSD, WaveRNN,RawNet ,LCNN )

    model_kwargs: dict = KWARGS_MAP.get(model_classname).get(feature_classname)
    if model_kwargs is None:
        raise ValueError(
            f"model_kwargs not found for {model_classname} and {feature_classname}"
        )
    model_kwargs.update({"device": device})

    LOGGER.info(f"Training model: {model_classname}")
    LOGGER.info(f"Input feature : {feature_classname}")
    LOGGER.info(f"Model kwargs  : {json.dumps(model_kwargs, indent=2)}")

    ###########################################################################

    train_dir = Path(train_dir)
    assert train_dir.is_dir()


    LOGGER.info("Loading data...")

    real_dataset_train, real_dataset_test, fake_dataset_train, fake_dataset_test = load_directory_split_train_test(
        path=train_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        train_size=train_size,
        use_double_delta=False,
        phone_call=False,
        pad=True,
        label=1,
        amount_to_use=None,
        channel = channel,
        model_len = model_len,
        model_classname= model_classname,
        txt_file_paths= f"/home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/train/ASVspoof2019_LA_train/ASVspoof2019.LA.cm.train.trn.txt"
    )

    
    dataset_train, dataset_val = None, None
    dataset_train = ConcatDataset([real_dataset_train, fake_dataset_train])
    dataset_val = ConcatDataset([real_dataset_test, fake_dataset_test])
    pos_weight = len(real_dataset_train) / len(fake_dataset_train)


    ###########################################################################

    LOGGER.info(f"Training model on {len(dataset_train)} audio files.")
    LOGGER.info(f"Testing model on  {len(dataset_val)} audio files.")
    LOGGER.info(f"Train/val ratio: {len(dataset_train) / len(dataset_val)}")
    LOGGER.info(f"Real/Fake ratio in training: {round(pos_weight, 3)} (pos_weight)")

    pos_weight = torch.Tensor([pos_weight]).to(device)

    model = Model(**model_kwargs).to(device)

    
    #######################################################################
    ModelTrainer(
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        lr=0.0001,
        optimizer_kwargs={"weight_decay": 0.0001},
    ).train(
        exp_name = exp_name,
        model=model,
        dataset_train=dataset_train,
        dataset_test=dataset_val,
        save_dir=save_dir,
        pos_weight=pos_weight,
        checkpoint=checkpoint,
        channel = channel,
        model_classname= model_classname,
    )
  



def eval_only(
    exp_name : str,
    test_dir: Union[Path, str],
    amount_to_use: int = None,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available else "cpu",
    batch_size: int = 32,
    save_dir: Union[str, Path] = None,
    test_size: float = 1,
    feature_classname: str = "wave",
    model_classname: str = "SimpleLSTM",
    in_distribution: bool = True,
    checkpoint=None,
    channel :bool = False,
    model_len : int = 6,
    eval_only : bool = True
    
) -> None:
    if model_len == 6:
        time = 481
        linear_input = 7424 #원래 7424
    elif model_len == 10:
        time = 801
        linear_input = 12544
    else:
        time = 2401
        linear_input = 38144

    input_channel = 2 if channel else 1
    d_args = {"nb_samp": 16000 * model_len * input_channel,
            "first_conv": 1024 ,  # no. of filter coefficients 
            "in_channels": 1,
            "filts": [20, [20, 20], [20, 128], [128, 128]], # no. of filters channel in residual blocks
            "blocks": [2, 4],
            "nb_fc_node": 1024,
            "gru_node": 1024,
            "nb_gru_layer": 3,
            "nb_classes": 2
    }
    
    KWARGS_MAP: Dict[str, dict] = {
    "SimpleLSTM": {
        "lfcc": {"feat_dim": 40, "time_dim": time *input_channel, "mid_dim": 30, "out_dim": 1}, #6초 (40,1443) 10초(40,2403)
        "mfcc": {"feat_dim": 40, "time_dim": time *input_channel, "mid_dim": 30, "out_dim": 1},
    },
    "ShallowCNN": {
        "lfcc": {"in_features": input_channel, "out_dim": 1 , "linear_input": linear_input},#TODO 다시바꾸기 in_features = in_features
        "mfcc": {"in_features": input_channel, "out_dim": 1 , "linear_input": linear_input},
    },
    "MLP": {
        "lfcc": {"in_dim": 40 * time *input_channel, "out_dim": 1},
        "mfcc": {"in_dim": 40 * time *input_channel, "out_dim": 1},
    },
    "TSSD": {
        "wave": {"in_dim": 16000 * model_len , "in_channels": input_channel},
    },
    "WaveRNN": {
        "wave": {"num_frames": 10, "input_length": 16000 * model_len , "hidden_size": 500},
    },
    "WaveLSTM": {
        "wave": {
            "num_frames": 20,
            "input_len": 16000 * model_len * input_channel,
            "hidden_dim": 50,
            "out_dim": 1,
        }
    },
    "Multimodal" : {
        "wave" : {"in_channels": 5* model_len, "in_channels_wave": input_channel ,"in_dim": 16000* model_len, "audio_model": "TSSD"},
        "mfcc" : {"in_channels": 5* model_len, "input_shape": 40 * time * input_channel , "audio_model": "MLP"},
        "lfcc" : {"in_channels": 5* model_len, "input_shape": 40 * time * input_channel , "aduio_model": "MLP"}
    },
    "RawNet" : {
        "wave" : {"d_args":d_args}
    },
    "LCNN" : {
        "lfcc" : {"in_channels" : input_channel,},
        "mfcc" : {"in_channels" : input_channel,}
    }
}
    feature_classname = feature_classname.lower()
    assert feature_classname in FEATURE_CLASSNAMES
    assert model_classname in MODEL_CLASSNAMES

    # get feature transformation function
    feature_fn = None if feature_classname == "wave" else eval(feature_classname)
    assert feature_fn in (None, lfcc, mfcc)
    # get model constructor
    Model = eval(model_classname)
    assert Model in (SimpleLSTM, ShallowCNN, WaveLSTM, MLP, TSSD, WaveRNN,RawNet , LCNN)

    model_kwargs: dict = KWARGS_MAP.get(model_classname).get(feature_classname)
    if model_kwargs is None:
        raise ValueError(
            f"model_kwargs not found for {model_classname} and {feature_classname}"
        )
    model_kwargs.update({"device": device})

    LOGGER.info(f"Evaluating model: {model_classname}")
    LOGGER.info(f"Input feature : {feature_classname}")
    LOGGER.info(f"Model kwargs  : {json.dumps(model_kwargs, indent=2)}")

    ###########################################################################

    test_dir = Path(test_dir)
    assert test_dir.is_dir()

 
    LOGGER.info("Loading data...")

    real_dataset_test, _ ,fake_dataset_test, _= load_directory_split_train_test(
        path=test_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        train_size=test_size,
        use_double_delta=False,
        phone_call=False,
        pad=True,
        label=1,
        amount_to_use=None,
        channel= channel,
        eval_only= eval_only,
        model_len = model_len,
        model_classname = model_classname,
        txt_file_paths= f"/home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof/train/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    )


    dataset_test = ConcatDataset([real_dataset_test, fake_dataset_test])

    ###########################################################################

    LOGGER.info(f"Testing model on  {len(dataset_test)} audio files.")


    model = Model(**model_kwargs).to(device)


    #######################################################################

    ModelTrainer(batch_size=batch_size, epochs=epochs, device=device).eval(
        exp_name = exp_name,
        model=model,
        dataset_test=dataset_test,
        save_dir=save_dir,
        checkpoint=checkpoint,
        channel= channel,
        model_classname= model_classname
    )


def experiment(
    name: str,
    train_dir: str,
    test_dir:str,
    epochs: int,
    batch_size: int,
    feature_classname: str,
    model_classname: str,
    device: str,
    seed: Optional[int] = None,
    restore: bool = False,
    evaluate_only: bool = False,
    channel : bool = False,
    model_len: int= 6,
    **kwargs,
):
    root_save_dir = Path("saved")
    save_dir = root_save_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / f"{timestamp_seconds()}.log"
    restore_path = save_dir / "best.pt"
    if restore and restore_path.is_file():
        LOGGER.info(f"Restoring from {restore_path}")
        ckpt = torch.load(restore_path, map_location=lambda storage, loc: storage)
    else:
        ckpt = None

    init_logger(log_file)
    if seed is not None:
        set_seed_all(seed)

    LOGGER.info(f"Batch size: {batch_size}, seed: {seed}, epochs: {epochs}")

    if evaluate_only:
        eval_only(
            exp_name = name,
            test_dir=test_dir,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            save_dir=save_dir,
            feature_classname=feature_classname,
            model_classname=model_classname,
            checkpoint=ckpt,
            channel = channel,
            model_len = model_len,
            eval_only = evaluate_only
        )
    else:
        train(
            exp_name = name,
            train_dir=train_dir,
            epochs=epochs,
            device=device,
            batch_size=batch_size,
            save_dir=save_dir,
            feature_classname=feature_classname,
            model_classname=model_classname,
            checkpoint=ckpt,
            channel = channel,
            model_len = model_len,
        )



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        "--train",
        help="Directory containing real data. (default: 'data/real')",
        type=str,
        #default = "data_asvspoof_DR/train/ASVspoof2019_LA_train",
        default = f"/home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/train/ASVspoof2019_LA_train"
    )
    
    parser.add_argument(
        "--test_dir",
        "--test",
        help="Directory containing real data. (default: 'data/real')",
        type=str,
        #default="data_asvspoof_DR/train/ASVspoof2019_LA_eval"
        default = f"/home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/train/ASVspoof2019_LA_eval"
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size. (default: 256)",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--epochs",
        help="Number of maximum epochs to train. (default: 20)",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        help="Random seed. (default: 42)",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--feature_classname",
        help="Feature classname. (default: 'lfcc')",
        choices=FEATURE_CLASSNAMES,
        type=str,
        default="mfcc",
    )
    parser.add_argument(
        "--model_classname",
        help="Model classname. (default: 'ShallowCNN')",
        choices=MODEL_CLASSNAMES,
        type=str,
        default="ShallowCNN",
    )
    parser.add_argument(
        "--device",
        help="Device to use. (default: 'cuda' if possible)",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--deterministic",
        help="Whether to use deterministic training (reproducible results).",
        action="store_true",
        default = True,
    )
    parser.add_argument(
        "--restore",
        help="Whether to restore from checkpoint.",
        action="store_true",

    )
    parser.add_argument(
        "--eval_only",
        help="Whether to evaluate only.",
        action="store_true",
        #default=True
    )
    parser.add_argument(
        "--channel",
        help = "original sound or D/R 2ch(True =2ch, False = 1ch)",
        type = bool,
        default = True
    )
    parser.add_argument(
        "--model_len",
        help = "length of model in train (30/10/6)",
        type = int,
        default = 6,
    )
    parser.add_argument(
        "--amount_of_use",
        help= "Only some of the samples are used",
        type = int,
        default= 160
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()


    channel = "2ch" if args.channel else "1ch"
    model_len = str(args.model_len)
    exp_name = f"{args.model_classname}_{args.feature_classname}_{model_len}_{channel}"
    try:
        printc(f">>>>> Starting experiment: {exp_name}")
        experiment(
            name=exp_name,
            train_dir=args.train_dir,
            test_dir = args.test_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            feature_classname=args.feature_classname,
            model_classname=args.model_classname,
            amount_of_use = args.amount_of_use,
            device=args.device,
            seed=args.seed if args.deterministic else None,
            restore=args.restore,
            evaluate_only=args.eval_only,
            channel = args.channel,
            model_len = args.model_len,   
        )
        printc(f">>>>> Experiment Done: {exp_name}\n\n")
    except Exception as e:
        printc(f">>>>> Experiment Failed: {exp_name}\n\n", color="red")
        LOGGER.exception(e)


if __name__ == "__main__":
    main()
 