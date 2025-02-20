# ref: https://huggingface.co/blog/mms_adapters
import torch
import librosa
import csv, sys
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, AutoProcessor
from transformers import Wav2Vec2Processor

"""
Settings parser args
"""
csv_dir = "" # path to the csv file, format = split, path to wav, severity, text
exp_dir = "" # path to the experiment directory
exp_subdir = sys.argv[1]
lang = sys.argv[2]
num = sys.argv[3]
finetune_model = f"{exp_dir}{exp_subdir}/checkpoint-{num}" # resume from checkpoint; note: link adapter into the folder
model_id = "facebook/mms-1b-all"
if "none" in finetune_model: finetune_model = model_id
model = Wav2Vec2ForCTC.from_pretrained(finetune_model, target_lang=lang, ignore_mismatched_sizes=True).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(model_id)
processor.tokenizer.set_target_lang(lang)

"""
Setting dataset
"""
def read_csv_to_list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        ret = []
        for row in reader:
            if row[0]=='test': ret.append(row)
        return ret

"""
Main: collect performace
"""
csv_files = {"spa": "pc-gita.csv", "ita": "easycall.csv", "tam": "ssnce.csv"}
sentences_csv_files = {"spa": "pc-gita_sentence.csv", "ita": "easycall_sentence.csv", "tam": "ssnce_sentence.csv", }
test_set = read_csv_to_list(csv_dir + csv_files[lang]) # or sentences_csv_files

print(f"Start inferencing {exp_subdir}...")
with torch.no_grad():
    for i, row in tqdm(enumerate(test_set)):
        ref = row[3]
        np_audio = librosa.load(row[1], sr=16000)[0]
        speech = processor(np_audio, sampling_rate=16000, return_tensors="pt", padding=True)
        outputs = model(speech.input_values.to("cuda")).logits
        ids = torch.argmax(outputs, dim=-1)[0]
        hyp = processor.decode(ids)
        # save inference results
        with open(f"mms/{exp_subdir}.txt", "a") as f:
            f.write(f"{i}: {ref}\n")
            f.write(f"{i}: {hyp}\n")
