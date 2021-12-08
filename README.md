# BO Term Project

## Training

```bash
$ python train.py --batch_size 256 --tgt_dir {{DATA_ROOT}}
```

## Inference / Test

```bash
$ python test.py --src_dir {{DIR_TO_SOURCE_FOLDER}} --tgt_dir {{DATA_ROOT} --tgt_dir {{NAME_OF_TGT_FOLDER}} --model_name {{CHECKPOINT_FILE_NAME}} --batch_size 1}
```

## Dataset / Checkpoint
Example dataset and checkpoint file are located in data_root folder. 