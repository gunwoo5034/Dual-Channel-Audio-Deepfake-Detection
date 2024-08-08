









# echo "MLP-mfcc-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname MLP --model_len 6 

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname MLP --model_len 6 --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake


echo "MLP-mfcc-6-2ch"

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname MLP --model_len 6 --channel True

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname MLP --model_len 6 --channel True --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake








# echo "MLP-lfcc-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname MLP --model_len 6 

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname MLP --model_len 6 --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake


echo "MLP-lfcc-6-2ch"

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname MLP --model_len 6 --channel True

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname MLP --model_len 6 --channel True --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake








echo "ShallowCNN-mfcc-6-1ch"

#CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname ShallowCNN --model_len 6 

#CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname ShallowCNN --model_len 6 --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake

echo "ShallowCNN-mfcc-6-2ch"

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname ShallowCNN --model_len 6 --channel True

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname ShallowCNN --model_len 6 --channel True --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake









# echo "ShallowCNN-lfcc-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname ShallowCNN --model_len 6 --real_dir /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof/train/LA

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname ShallowCNN --model_len 6 --restore --eval_only --test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00

# echo "ShallowCNN-lfcc-6-2ch"

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname ShallowCNN --model_len 6 --channel True

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname ShallowCNN --model_len 6 --channel True --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00






echo "WaveRNN-wave-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname WaveRNN --model_len 6 

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname WaveRNN --model_len 6 --restore --eval_only ##--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake

# echo "WaveRNN-wave-6-2ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname WaveRNN --model_len 6 --channel True

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname WaveRNN --model_len 6 --channel True --restore --eval_only ##--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake





















echo "WaveLSTM-wave-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname WaveLSTM --model_len 6 

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname WaveLSTM --model_len 6 --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake


echo "WaveLSTM-wave-6-2ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname WaveLSTM --model_len 6 --channel True

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --channel True --model_classname WaveLSTM --model_len 6 --restore --eval_only --test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00

















echo "TSSD-wave-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname TSSD --model_len 6 

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname TSSD --model_len 6 --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake


echo "TSSD-wave-6-2ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname TSSD --model_len 6 --channel True

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname TSSD --model_len 6 --channel True --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake



echo "SimpleLSTM-mfcc-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname SimpleLSTM --model_len 6 

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname SimpleLSTM --model_len 6 --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake

echo "SimpleLSTM-mfcc-6-2ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname SimpleLSTM --model_len 6 --channel True

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --channel True --model_classname SimpleLSTM --model_len 6 --restore --eval_only --test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00








echo "SimpleLSTM-lfcc-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname SimpleLSTM --model_len 6 

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname SimpleLSTM --model_len 6 --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/real --test_dir_fake /home/gunwoo/kunwoolee/DEEPFAKE_project/Data_split/splilt_6/test/fake

# echo "SimpleLSTM-lfcc-6-2ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname SimpleLSTM --model_len 6 --channel True

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --channel True --model_classname SimpleLSTM --model_len 6 --restore --eval_only --test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00



# echo "RawNet-wave-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname RawNet --model_len 6 --real_dir /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof/train/LA

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname RawNet --model_len 6 --restore --eval_only --test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof/evaluation/ASVspoof2021_DF_eval_00

# echo "RawNet-wave-6-2ch"

# python train.py --feature_classname wave --model_classname RawNet --model_len 6 --channel True --real_dir /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/train

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname wave --model_classname RawNet --model_len 6 --channel True --restore --eval_only --test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00



# echo "LCNN-lfcc-6-1ch"

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname LCNN --model_len 6 --real_dir /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof/train/LA

# CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname LCNN --model_len 6 --restore --eval_only --test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof/evaluation/ASVspoof2021_DF_eval_00

echo "LCNN-lfcc-6-2ch"

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname LCNN --model_len 6 --channel True #--real_dir /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/train

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname lfcc --model_classname LCNN --model_len 6 --channel True --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00

echo "LCNN-mfcc-6-2ch"

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname LCNN --model_len 6 --channel True #--real_dir /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/train

CUDA_VISIBLE_DEVICES=1 python train.py --feature_classname mfcc --model_classname LCNN --model_len 6 --channel True --restore --eval_only #--test_dir_real /home/gunwoo/kunwoolee/DEEPFAKE_project/data_asvspoof_DR/ASVspoof2021_DF_eval_00