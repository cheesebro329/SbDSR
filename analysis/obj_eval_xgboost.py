import sys
import re

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns

### ------------------ DATA LOADING ------------------ ###
lang = sys.argv[1]
mat = sys.argv[2]
lang_dict = {"es": "spanish", "it": "italian", "ta": "tamil"}

def split_fn(df, lang):
    """Extracts language-specific data based on filename."""
    df["lang"] = df["filename"].apply(lambda x: x.split("_")[0])
    return df[df["lang"] == lang]

df_dys = pd.read_csv(f"disvoice_features/csv/{lang_dict[lang]}_data_prosody_disvoice.csv", index_col=False)
df_fleurs = pd.read_csv(f"disvoice_features/csv/fleurs_{lang}_data_prosody_disvoice.csv", index_col=False)
df_speed = split_fn(pd.read_csv(f"disvoice_features/csv/speed_data_prosody_disvoice.csv", index_col=False), lang)
df_tempo = split_fn(pd.read_csv(f"disvoice_features/csv/tempo_data_prosody_disvoice.csv", index_col=False), lang)
df_speaker = split_fn(pd.read_csv(f"disvoice_features/csv/speaker_data_prosody_disvoice.csv", index_col=False), lang)
df_twopass = split_fn(pd.read_csv(f"disvoice_features/csv/twopass_data_prosody_disvoice.csv", index_col=False), lang)

def extract_es_speaker(filename):
    match = re.match(r"([A-Z]+(?:\d{4})?)", filename)
    return match.group(1) if match else filename 

def extract_ta_speaker(filename):
    if "C" in filename:
        return filename.split("_")[0]
    else:
        match = re.match(r"([A-Z]+(?:\d{4})?)", filename)
        return match.group(1) if match else filename 

if lang == "es":
    df_dys["speaker"] = df_dys["filename"].apply(extract_es_speaker)
    df_dys["severity"] = df_dys["speaker"].apply(lambda x: "C" if "C" in x else "D")

if lang == "it":
    df_dys['material'] = df_dys['filename'].apply(lambda x: 'word' if len(x.split("_")) == 3 else 'sent')
    df_dys["speaker"] = df_dys["filename"].apply(lambda x: x.split("_")[0])
    df_dys["severity"] = df_dys["filename"].apply(lambda x: "C" if "mc" in x or "fc" in x else "D")

if lang == "ta":
    df_dys["speaker"] = df_dys["filename"].apply(extract_ta_speaker)
    df_dys["severity"] = df_dys["speaker"].apply(lambda x: "C" if "C" in x else "D")


# Data Mapping
df_dys["severity"] = df_dys["severity"].map({"C": 0, "D": 1})
# df_dys = df_dys[df_dys["material"] == mat]


### ------------------ Feature LOADING ------------------ ###
# Ensure all datasets have the same features
common_columns = set(df_fleurs.columns) & set(df_dys.columns) & set(df_speed.columns) & set(df_tempo.columns) & set(df_speaker.columns) & set(df_twopass.columns)
common_columns = list(set(common_columns) - {"language","filename","filepath","duration","split"})

datasets = {name: df[common_columns].fillna(0) for name, df in zip(
    ["df_fleurs", "df_dys", "df_speed", "df_tempo", "df_speaker", "df_twopass"],
    [df_fleurs, df_dys, df_speed, df_tempo, df_speaker, df_twopass]
)}

### ------------------ Train-Test Split ------------------ ###
df_train = datasets["df_dys"]
X = df_train.drop(columns=["severity"], errors='ignore')
y = df_train["severity"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

### ------------------ Classification ------------------ ###
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'scale_pos_weight': [np.sum(y_train == 0) / np.sum(y_train == 1)]  # Adjust for class imbalance
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost Classifier with Grid Search
gs_model = GridSearchCV(
    XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42),
    param_grid,
    cv=cv,
    scoring='f1',
    verbose=1,
    n_jobs=-1,
)

# Train with cross-validation
gs_model.fit(X_train_scaled, y_train)

# Best model from Grid Search
best_model = gs_model.best_estimator_
print("Best Parameters:", gs_model.best_params_)

# Test performance
y_val_pred = best_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)
print(f"Test Accuracy: {val_accuracy:.4f}, Test F1: {val_f1:.4f}")
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(cm_normalized, display_labels=["Healthy", "Dysarthria"])

fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap="Blues", ax=ax)
plt.title(f"{lang.upper()} - {mat} Normalized Confusion Matrix ({mat})")
plt.savefig(f"results/{lang}_{mat}_dys_normalized_confusion_matrix.pdf")
plt.close()


### ------------------ Classification of Synthesized Datasets ------------------ ###
classification_results = []
for synth_name in ["df_fleurs", "df_speed", "df_tempo", "df_speaker", "df_twopass"]:
    df_test = datasets[synth_name]
    X_test = df_test.drop(columns=["severity"], errors='ignore')
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = best_model.predict(X_test_scaled)
    
    healthy_count = np.sum(y_pred == 0)
    dys_count = np.sum(y_pred == 1)
    
    classification_results.append({
        "Dataset": synth_name,
        "Total Samples": len(y_pred),
        "Classified as Healthy": healthy_count,
        "Classified as Dysarthria": dys_count,
        "Healthy %": round(healthy_count / len(y_pred) * 100, 2),
        "Dysarthria %": round(dys_count / len(y_pred) * 100, 2)
    })

# Save and display classification results
df_classification_results = pd.DataFrame(classification_results)
print(df_classification_results)
df_classification_results.to_csv(f"results/{lang}_{mat}_synthesized_classification_results.csv", index=False)

plt.figure(figsize=(8, 5))
colorblind_palette = sns.color_palette("colorblind", 2)

plt.bar(
    df_classification_results["Dataset"],
    df_classification_results["Healthy %"],
    color=colorblind_palette[1],
    label="Healthy"
)

plt.bar(
    df_classification_results["Dataset"],
    df_classification_results["Dysarthria %"],
    bottom=df_classification_results["Healthy %"],
    color=colorblind_palette[0],
    label="Dysarthria"
)

plt.xlabel("Synthesized Dataset")
plt.ylabel("Classification Percentage")
plt.title(f"{lang.upper()} - {mat} Classification of Synthesized Datasets")
plt.legend()
plt.xticks(rotation=30)
plt.ylim(0, 100)
plt.savefig(f"results/{lang}_{mat}_synthesized_classification_barplot.pdf")
plt.show()

