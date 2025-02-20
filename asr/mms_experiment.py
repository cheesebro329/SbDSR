# ref: https://huggingface.co/blog/mms_adapters
import numpy as np
import librosa
import os, argparse, glob
from mms_utils import *
from whispernorm import BasicTextNormalizer
from indicnlp.normalize.indic_normalize import TamilNormalizer
from transformers import Wav2Vec2ForCTC, AutoProcessor, Trainer, TrainingArguments
from evaluate import load
from safetensors.torch import save_file as safe_save_file
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE

"""
Dataset settings
"""
def pair_wav_and_text(wav_dir, text_file):
    wav_files = glob.glob(wav_dir + '/*.wav')
    with open(text_file, 'r') as f:
        texts = f.readlines()
    ret = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0].split('_')[-1]
        text = texts[int(wav_id)].strip()
        ret.append([wav_file, text])
    return ret

def prepare_dataset(pairs, processor, normalizer):
    li = []
    for i, line in enumerate(pairs):
        d = {}
        audio_array, sampling_rate = librosa.load(line[0], sr=16000)
        d["input_values"] = processor(audio_array, sampling_rate=sampling_rate).input_values[0]
        d["input_length"] = len(d["input_values"])
        d["labels"] = processor(text=normalizer.normalize(line[1])).input_ids
        li.append(d)
    return li

"""
Pretrained model settings
"""
def get_parser():
    parser = argparse.ArgumentParser(description="Fine-tune a speech-to-text model.")
    parser.add_argument("--language", type=str, default="spa", help="language code ISO 639-3")
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    return parser


"""
Argument parsing
"""
parser = get_parser()
args = parser.parse_args()

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id, target_lang=args.language)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
model = Wav2Vec2ForCTC.from_pretrained(model_id, target_lang=args.language, ignore_mismatched_sizes=True)
processor.tokenizer.set_target_lang(args.language)
model.load_adapter(args.language)

model.init_adapter_layers()
model.freeze_base_model()
adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True

wer_metric, cer_metric = load("wer"), load("cer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

if args.language == "ita": lang = 'it'
elif args.language == "spa": lang = 'es'
elif args.language == "tam": lang = 'ta'
else: raise ValueError("Language not supported")

training_args = TrainingArguments(
    output_dir=f"{args.exp_dir}/{lang}_{args.exp_name}",
    group_by_length=True,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=args.max_epoch,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=200,
    eval_steps=100,
    logging_steps=100,
    learning_rate=5e-4,
    warmup_steps=100,
    save_total_limit=2,
    push_to_hub=False,
)

"""
Train
"""
print("Start training...")
basedir = "" # path to stored FLERUS data, with subdirectores named original, speed, tempo, ...
train_csv = pair_wav_and_text(f"{basedir}/original/{lang}/train", f"{basedir}/text/{lang}_train.txt")
aug_train_csv = pair_wav_and_text(f"{basedir}/{args.exp_name}/{lang}/train", f"{basedir}/text/{lang}_train.txt")
valid_csv = pair_wav_and_text(f"{basedir}/original/{lang}/validation", f"{basedir}/text/{lang}_validation.txt")
if args.exp_name != "original": train_csv += aug_train_csv

normalizer = TamilNormalizer().normalize if args.language == "tam" else BasicTextNormalizer()
train_set = prepare_dataset(train_csv, processor, normalizer)
valid_set = prepare_dataset(valid_csv, processor, normalizer)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=processor.feature_extractor,
)
trainer.train()

"""
Save adapter
"""
adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(args.language)
adapter_file = os.path.join(training_args.output_dir, adapter_file)

safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})
