import jiwer
import sys
import csv
from whispernorm import BasicTextNormalizer
from indicnlp.normalize.indic_normalize import TamilNormalizer

# usage: calculate error rate from output of mms_inference.py
# pass file name as the first argument
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

# csv file name as the second argument to get severity level
# csv format: split, path to wav, severity, text
with open(sys.argv[2], 'r') as f:
    reader = csv.reader(f)
    severity = []
    for row in reader:
        if row[0]=='test': severity.append(row[2])

# whisper normalizer doesn't work for Tamil
normalizer = TamilNormalizer().normalize if "ta" in sys.argv[1] else BasicTextNormalizer()

n = len(lines)//2
error_rates = {'all': {'cer': [], 'wer': []}}
for i in range(n):
    ref = lines[i*2].split(": ")[1].replace('_', ' ').strip()
    hyp = lines[i*2+1].split(": ")[1].strip()
    ref = normalizer(ref)
    hyp = normalizer(hyp)
    cer, wer = jiwer.cer(ref, hyp), jiwer.wer(ref, hyp)
    error_rates['all']['cer'].append(cer)
    error_rates['all']['wer'].append(wer)
    # severity level
    if severity[i] not in error_rates:
        error_rates[severity[i]] = {'cer': [], 'wer': []}
    error_rates[severity[i]]['cer'].append(cer)
    error_rates[severity[i]]['wer'].append(wer)


for key in error_rates:
    error_rates[key]['cer'] = sum(error_rates[key]['cer']) / len(error_rates[key]['cer']) *100
    error_rates[key]['wer'] = sum(error_rates[key]['wer']) / len(error_rates[key]['wer']) *100

print(f"CER, WER of {sys.argv[1]}:")
error_rates = dict(sorted(error_rates.items()))
for key in error_rates:
    print(f"{key}: {error_rates[key]['cer']:.1f} / {error_rates[key]['wer']:.1f}")
