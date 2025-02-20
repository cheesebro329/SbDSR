### Settings & Arguments

- audios are all preprocessed and stored in seperated folders for each augmentation
- `lang_3`: ISO 639-3 language code, `["spa", "ita", "tam"]`
- `lang_2`: ISO 639-1 language code, `["es", "it", "ta"]`
- `exp_name`: fine-tuning data setting
  - `none`: no fine-tuning; used for inference only
  - `original`: fine-tune with FLEURS, no augmentation
  - `speed`, ..., `speaker-prosody`: add one of the augmentation

### Fine-tuning
```
python mms_experiment.py --exp_name {exp_name} --language {lang_3}
```
- input: language and selected augmentation
- output: an experiment directory named `{lang_2}_{exp_name}_mms`

### Inference
```
python mms_inference.py {exp_dir_name} {lang_3} {ckpt_stepnum}
```
- required: csv files for inference dataset, formatted as `[split, path to wav, severity, text]`
- input: experiment directory's name, corresponding language code, and selected checkpoint step number
- output: a file named `mms/{exp_dir_name}.txt`, storing reference-hypothesis pairs

### Scoring
```
python geter.py mms/{exp_dir_name}.txt {path_to_csv}
```
- input: output from inference, and path to inference dataset's csv
- output: averaged CER and WER for all utterances, and for each severity level
