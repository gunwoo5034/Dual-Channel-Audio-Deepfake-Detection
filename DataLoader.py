import functools
import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import random
from torchaudio.transforms import Resample
from torchvision import transforms
import numpy as np
import torch
import torchaudio
import re
import PIL

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchaudio.functional import apply_codec

from module.lfcc import LFCC
from utils import find_wav_files

LOGGER = logging.getLogger(__name__)


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


  
class AudioDataset(Dataset):
    """Torch dataset to load data from a provided directory.

    Args:
        directory_or_path_list: Path to the directory containing wav files to load. Or a list of paths.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        directory_or_path_list: Union[Union[str, Path], List[Union[str, Path]]],
        sample_rate: int = 16000,
        amount: Optional[int] = None,
        normalize: bool = True,
        trim: bool = False,
        phone_call: bool = False,
        channel : bool = False,
        eval_only: bool = False,
    ) -> None:
        super().__init__()

        self.trim = trim
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.phone_call = phone_call
        self.channel = channel
        self.eval_only = eval_only
        if isinstance(directory_or_path_list, list):
            paths = directory_or_path_list
        elif isinstance(directory_or_path_list, Path) or isinstance(
            directory_or_path_list, str
        ):
            directory = Path(directory_or_path_list)
            if not directory.exists():
                raise IOError(f"Directory does not exists: {self.directory}")

            paths = find_wav_files(directory)
            if paths is None:
                raise IOError(f"Directory did not contain wav files: {self.directory}")
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!"
            )

        if amount is not None:
            paths = paths[:amount*3]
        

        if channel: 
            paths_D = []
            paths_R = []
            for path in paths:
                if  str(path).endswith('_d.wav'):
                    paths_D.append(path)
                elif str(path).endswith('_r.wav'):
                    paths_R.append(path)
                # paths_R = [path for path in paths if str(path).endswith('_r.wav') and "ff4" not in str(path)]
                # paths = [path for path in paths if str(path).endswith('_d.wav') and "ff4" not in str(path) ]
            paths_D = sorted(paths_D)
            paths_R = sorted(paths_R)
            self._paths = paths_D
            self._paths_R = paths_R
        else:
            if self.eval_only:
                paths = [path for path in paths if '_d' not in str(path)[-10:] and "_r" not in str(path)[-10:] and "ff4" not in str(path)]
                self._paths = paths
            else:
                paths = [path for path in paths if not str(path).endswith(('_d.wav', '_r.wav')) and "ff4" not in str(path)]
                self._paths = paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:


        if self.channel:
            path_R = self._paths_R[index]
            waveform_R, sample_rate_R = torchaudio.load(str(path_R), normalize = self.normalize)
            
            if sample_rate_R != self.sample_rate:
                waveform_R, sample_rate_R = torchaudio.sox_effects.apply_effects_file(
                    path_R, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
                )
            if self.trim:
                (
                    waveform_trimmed,
                    sample_rate_trimmed,
                ) = torchaudio.sox_effects.apply_effects_tensor(
                    waveform_R, sample_rate_R, SOX_SILENCE
                )

                if waveform_trimmed.size()[1] > 0:
                    waveform_R = waveform_trimmed
                    sample_rate_R = sample_rate_trimmed
            scaled_waveform_R = min_max_scaling(waveform_R)
        path = self._paths[index]
        waveform, sample_rate = torchaudio.load(str(path), normalize=self.normalize)

        # resamplling
        if sample_rate != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )

        if self.trim:
            (
                waveform_trimmed,
                sample_rate_trimmed,
            ) = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, SOX_SILENCE
            )

            if waveform_trimmed.size()[1] > 0:
                waveform = waveform_trimmed
                sample_rate = sample_rate_trimmed
                
            waveform = apply_codec(waveform, sample_rate, format="gsm")
        waveform = torch.mean(waveform, axis=0)
        audio_path = str(path)
        scaled_waveform = min_max_scaling(waveform)
        
        if self.channel:
            return scaled_waveform , scaled_waveform_R,0, sample_rate, str(audio_path)
        else:
            return scaled_waveform,0,0, sample_rate, str(audio_path)

    def __len__(self) -> int:
        return len(self._paths)


class PadDataset(Dataset):
    def __init__(self, dataset: Dataset, feature_fn = None ,select =5 ,cut: int = 16000 , label=None , vid_fps =25, channel = False , model_len = 6): #cut은 몇초 모델을 사용할것인지
        self.dataset = dataset
        self.cut = cut * model_len  # max 4 sec (ASVSpoof default)
        self.label = label
        self.channel = channel
        self.model_len = model_len
        self.fps = int(vid_fps/select)
        self.select = select
        self.feature_fn = feature_fn
        #self.crop = crop

    def __getitem__(self, index):
        
        if self.channel :
            waveform_D, waveform_R, image_files,sample_rate, audio_path = self.dataset[index]
            waveform_D = waveform_D.squeeze(0)
            waveform_R = waveform_R.squeeze(0)
            waveform_len_D = waveform_D.shape[0]
            waveform_len_R = waveform_R.shape[0]
            if waveform_len_D != waveform_len_R:
                waveform_D = waveform_D[:waveform_len_R]
            
            num_repeats = int(self.cut/waveform_len_R) +1
            padded_waveform_R = torch.tile(waveform_R,(1,num_repeats))
            padded_waveform_D = torch.tile(waveform_D,(1,num_repeats))
            
            padded_waveform_R = padded_waveform_R.squeeze(0)
            padded_waveform_D = padded_waveform_D.squeeze(0)
            
            waveform_len_R = padded_waveform_R.shape[0]
            max_point = waveform_len_R - self.cut
            start_point = int(random.uniform(0,max_point))
            start_point_image = int(start_point / (len(padded_waveform_D)/ len(image_files))) if image_files != 0 else start_point
            padded_waveform_D = padded_waveform_D[start_point:start_point + self.cut]
            padded_waveform_R = padded_waveform_R[start_point:start_point + self.cut]
            #return padded_waveform_R[start_point:start_point + self.cut], padded_waveform_D[start_point:start_point + self.cut],sample_rate ,str(audio_path) , self.label
        else:
            waveform,_,image_files, sample_rate, audio_path = self.dataset[index]
            waveform = waveform.squeeze(0)
            waveform_len = waveform.shape[0]

            num_repeats = int(self.cut / waveform_len) + 1
            padded_waveform = torch.tile(waveform, (1, num_repeats))
            padded_waveform = padded_waveform.squeeze(0)
            waveform_len = padded_waveform.shape[0]
            max_point = waveform_len - self.cut
            start_point = int(random.uniform(0,max_point))
            start_point_image = int(start_point / (len(padded_waveform)/ len(image_files))) if image_files != 0 else start_point
            padded_waveform =padded_waveform[start_point:start_point + self.cut]
            #return padded_waveform[start_point:start_point + self.cut], 0, sample_rate, str(audio_path), self.label
        if image_files != 0:
            
            images = []
            indices = torch.arange(start_point_image, len(image_files), self.select).tolist()
            selected_image_path = [image_files[i] for i in indices]
            selected_image_path = selected_image_path[:self.model_len * self.fps] #TODO model_len맞나?
            
            if self.feature_fn is None:
                for img_path in selected_image_path:
                    img = PIL.Image.open(img_path)
                    img = data_transforms(img)
                    images.append(img)
                images = torch.stack(images)
                if images.size(0)<=self.model_len*self.fps:
                    selected_image = torch.tile(images,(2,1,1,1))
                    selected_image = selected_image[:self.model_len * self.fps,]    
                    
                    
                    
                if self.channel:
                    return  padded_waveform_D , padded_waveform_R, sample_rate, str(audio_path), self.label, selected_image
                else:
                    return  padded_waveform , torch.tensor(0)  ,  sample_rate, str(audio_path), self.label, selected_image
            else:
                if self.channel:
                    return padded_waveform_D , padded_waveform_R, sample_rate, str(audio_path), self.label, selected_image_path
                else:
                    return padded_waveform , torch.tensor(0) ,sample_rate, str(audio_path), self.label, selected_image_path
        else:
            if self.channel:
                return padded_waveform_D, padded_waveform_R, sample_rate ,str(audio_path), self.label, torch.tensor(0) 
            else:
                return padded_waveform, torch.tensor(0) , sample_rate, str(audio_path), self.label, torch.tensor(0)
    def __len__(self):
        return len(self.dataset)


class TransformDataset(Dataset):
    """A generic transformation dataset.

    Takes another dataset as input, which provides the base input.
    When retrieving an item from the dataset, the provided transformation gets applied.

    Args:
        dataset: A dataset which return a (waveform, sample_rate)-pair.
        transformation: The torchaudio transformation to use.
        needs_sample_rate: Does the transformation need the sampling rate?
        transform_kwargs: Kwargs for the transformation.
    """

    def __init__(
        self,
        dataset: Dataset,
        transformation: Callable,
        needs_sample_rate: bool = False,
        transform_kwargs: dict = {},
        channel: bool = False,
        fps: int = 5
    ) -> None:
        super().__init__()
        self._dataset = dataset

        self._transform_constructor = transformation
        self._needs_sample_rate = needs_sample_rate
        self._transform_kwargs = transform_kwargs
        self._transform = None
        self._channel = channel
        self. fps = fps

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, waveform_R ,sample_rate, audio_path, label ,selected_image_path = self._dataset[index]
        images = []
        if selected_image_path:
            for img_path in selected_image_path:
                img = PIL.Image.open(img_path)
                img = data_transforms(img)
                images.append(img)
            cut = int(self.fps * (len(waveform) / sample_rate))
            images = torch.stack(images)
            if images.size(0)<=cut:
                selected_image = torch.tile(images,(2,1,1,1))
                selected_image = selected_image[:cut]    
        else:
            selected_image = torch.tensor(0)
            
        if self._transform is None:
            if self._needs_sample_rate:
                self._transform = self._transform_constructor(
                    sample_rate, **self._transform_kwargs
                )
            else:
                self._transform = self._transform_constructor(**self._transform_kwargs)

        if self._channel:
            return self._transform(waveform), self._transform(waveform_R), sample_rate, str(audio_path) ,label , selected_image
        else:
            return self._transform(waveform), torch.tensor(0), sample_rate, str(audio_path), label ,selected_image


class DoubleDeltaTransform(torch.nn.Module):
    """A transformation to compute delta and double delta features.

    Args:
        win_length (int): The window length to use for computing deltas (Default: 5).
        mode (str): Mode parameter passed to padding (Default: replicate).
    """

    def __init__(self, win_length: int = 5, mode: str = "replicate") -> None:
        super().__init__()
        self.win_length = win_length
        self.mode = mode

        self._delta = torchaudio.transforms.ComputeDeltas(
            win_length=self.win_length, mode=self.mode
        )

    def forward(self, X):
        """
        Args:
             specgram (Tensor): Tensor of audio of dimension (..., freq, time).
        Returns:
            Tensor: specgram, deltas and double deltas of size (..., 3*freq, time).
        """
        delta = self._delta(X)
        double_delta = self._delta(delta)

        return torch.hstack((X, delta, double_delta))


# =====================================================================
# Helper functions.
# =====================================================================


def _build_preprocessing(
    directory_or_audiodataset: Union[Union[str, Path], AudioDataset],
    transform: torch.nn.Module,
    audiokwargs: dict = {},
    transformkwargs: dict = {},
    channel: bool = False,
    fps = 5
) -> TransformDataset:
    """Generic function template for building preprocessing functions."""
    if isinstance(directory_or_audiodataset, AudioDataset) or isinstance(
        directory_or_audiodataset, PadDataset
    ):
        return TransformDataset(
            dataset=directory_or_audiodataset,
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
            channel= channel,
            fps = fps
        )
    elif isinstance(directory_or_audiodataset, str) or isinstance(
        directory_or_audiodataset, Path
    ):
        return TransformDataset(
            dataset=AudioDataset(directory=directory_or_audiodataset, **audiokwargs),
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
            channel = channel,
            fps = fps
        )
    else:
        raise TypeError("Unsupported type for directory_or_audiodataset!")


mfcc = functools.partial(_build_preprocessing, transform=torchaudio.transforms.MFCC)
lfcc = functools.partial(_build_preprocessing, transform=LFCC)


def double_delta(dataset: Dataset, delta_kwargs: dict = {} , channel: bool = False) -> TransformDataset:
    return TransformDataset(
        dataset=dataset,
        transformation=DoubleDeltaTransform,
        transform_kwargs=delta_kwargs,
        channel = channel
    )

def natural_sort_key(s):
    """주어진 문자열에 대한 자연 정렬 키를 생성하는 함수"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# image_datasets = datasets.ImageFolder(image_test_data_dir, transform=data_transforms)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=1)
])
def min_max_scaling(waveform):
    min_val = waveform.min()
    max_val = waveform.max()
    scaled_waveform = (waveform - min_val) / (max_val - min_val)
    return scaled_waveform


def load_directory_split_train_test(
    path: Union[Path, str],
    feature_fn: Callable,
    feature_kwargs: dict,
    train_size: float,
    use_double_delta: bool = True,
    phone_call: bool = False,
    pad: bool = False,
    label: Optional[int] = None,
    amount_to_use: Optional[int] = None,
    channel : bool = False,
    model_len : int = 6,
    eval_only : bool = False,
    model_classname: str = "ShallowCNN",
    txt_file_paths:str = "a.txt"
) -> Tuple[TransformDataset, TransformDataset]:
    """Load all wav files from directory, apply the feature transformation
    and split into test/train.

    Args:
        path (Union[Path, str]): Path to directory.
        feature_fn (Callable): This is assumed to be mfcc or lfcc function.
        feature_fn (dict): Kwargs for the feature_fn.
        test_size (float): Ratio of train/test split.
        use_double_delta (bool): Additionally calculate delta and double delta features (Default True)?
        amount_to_use (Optional[int]): If supplied, limit data.
    """
    fake_set = set()
    real_set = set()

    # spoof와 bonafide를 분류하여 fake_set과 real_set에 추가
    if os.path.isdir(txt_file_paths):
        for txt_file in os.listdir(txt_file_paths):
            with open(os.path.join(txt_file_paths, txt_file)) as file:
                for line in file:
                    parts = line.split()
                    if parts[-1] == "spoof":
                        fake_set.add(parts[1])
                    elif parts[-1] == "bonafide":
                        real_set.add(parts[1])
    else:
        with open(txt_file_paths) as file:
            for line in file:
                parts = line.split()
                if parts[-1] == "spoof":
                    fake_set.add(parts[1])
                elif parts[-1] == "bonafide":
                    real_set.add(parts[1])

    # set으로 바뀐 fake_list와 real_list
    fake_paths = set()
    real_paths = set()
    # 파일들을 찾아가면서 fake_set과 real_set에 속하는 것들만 걸러냄
    
    for path in find_wav_files(path):
        filename = str(path).split("/")[-1]
        filename = os.path.splitext(filename)[0]
        if channel:
            filename = filename[:-2]
        if filename in real_set:
            real_paths.add(path)
        elif filename in fake_set:
            fake_paths.add(path)
    real_paths = sorted(list(real_paths))
    fake_paths = sorted(list(fake_paths))
    if real_paths is None or fake_paths is None:
        raise IOError(f"Could not load files from {path}!")

    if amount_to_use is not None:
        real_paths = real_paths[:amount_to_use]
        fake_paths = fake_paths[:amount_to_use]
    real_train_size = int(train_size * len(real_paths))
    real_train_paths = real_paths[:real_train_size]
    real_test_paths  = real_paths[real_train_size:]
    
    fake_train_size = int(train_size * len(fake_paths))
    fake_train_paths = fake_paths[:fake_train_size]
    fake_test_paths  = fake_paths[fake_train_size:]
    
    real_train_dataset = AudioDataset(real_train_paths, phone_call = phone_call, channel = channel, eval_only= eval_only)
    if pad:
        real_train_dataset = PadDataset(real_train_dataset, label=1,channel = channel,model_len= model_len)
        
    real_test_dataset = AudioDataset(real_test_paths, phone_call = phone_call, channel = channel, eval_only= eval_only)
    if pad:
        real_test_dataset = PadDataset(real_test_dataset, label=1,channel = channel,model_len= model_len)
        
    fake_train_dataset = AudioDataset(fake_train_paths, phone_call = phone_call, channel = channel, eval_only= eval_only)
    if pad:
        fake_train_dataset = PadDataset(fake_train_dataset, label=0,channel = channel,model_len= model_len)
        
    fake_test_dataset = AudioDataset(fake_test_paths, phone_call = phone_call, channel = channel, eval_only= eval_only)
    if pad:
        fake_test_dataset = PadDataset(fake_test_dataset, label=0,channel = channel,model_len= model_len)
    

    LOGGER.info(f"Loading data from {path}...!")
    


    if feature_fn is None:
        return real_train_dataset, real_test_dataset , fake_train_dataset, fake_test_dataset

    real_dataset_train = feature_fn(
        directory_or_audiodataset=real_train_dataset,
        transformkwargs=feature_kwargs,
        channel = channel,

    )

    real_dataset_test = feature_fn(
        directory_or_audiodataset=real_test_dataset,
        transformkwargs=feature_kwargs,
        channel = channel,

    )
    fake_dataset_train = feature_fn(
        directory_or_audiodataset=fake_train_dataset,
        transformkwargs=feature_kwargs,
        channel = channel,

    )

    fake_dataset_test = feature_fn(
        directory_or_audiodataset=fake_test_dataset,
        transformkwargs=feature_kwargs,
        channel = channel,

    )
    if use_double_delta:
        real_dataset_train = double_delta(real_dataset_train ,channel= channel)
        real_dataset_test = double_delta(real_dataset_test , channel= channel)
        fake_dataset_train = double_delta(fake_dataset_train ,channel= channel)
        fake_dataset_test = double_delta(fake_dataset_test , channel= channel)

    return real_dataset_train, real_dataset_test , fake_dataset_train , fake_dataset_test

