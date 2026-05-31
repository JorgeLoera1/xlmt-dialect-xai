# ============================================================
# CLASIFICACIÓN DE DIALECTOS EN ESPAÑOL CON EXPLICABILIDAD
# Modelo: bertin-project/bertin-roberta-base-spanish + LoRA
# Metodología: EDA → Fine-tuning → XAI (LIME, SHAP, IG, Attn)
# Autor: Jorge Antonio Loera Grande
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings, re, os, json, time, pickle
warnings.filterwarnings("ignore")

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import evaluate
from lime.lime_text import LimeTextExplainer
import shap

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
print(f"PyTorch: {torch.__version__}")

MODEL_NAME      = "bertin-project/bertin-roberta-base-spanish"
MAX_LENGTH      = 128
DATA_PATH       = "data/Datos_dialectos.parquet"
SAVE_DIR        = "model_outputs/bertin_lora"
CHECKPOINT_DIR  = os.path.join(SAVE_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(SAVE_DIR, "modelo_final")
TOKENIZER_DIR   = os.path.join(SAVE_DIR, "tokenizer")
METADATA_PATH   = os.path.join(SAVE_DIR, "metadata.json")
MERGED_DIR      = os.path.join(SAVE_DIR, "modelo_merged")
FIG_EDA_DIR     = "results/figures/eda"
FIG_TRAIN_DIR   = "results/figures/training"
os.makedirs(SAVE_DIR,      exist_ok=True)
os.makedirs(FIG_EDA_DIR,   exist_ok=True)
os.makedirs(FIG_TRAIN_DIR, exist_ok=True)

print(f"Modelo base: {MODEL_NAME}")
print(f"Directorio de guardado: {SAVE_DIR}")

# ============================================================
# FUNCIONES DE LIMPIEZA
# ============================================================
STOPWORDS_ES    = set(stopwords.words('spanish'))
TOKENS_ANONIMOS = {
    'usr', '_usr', '__usr', 'user', '_user',
    'url', '_url', '__url', 'link', '_link',
    'http', 'https', 'rt'
}

def limpiar_texto_eda(texto):
    # elimina stopwords para que WordCloud y XAI muestren solo léxico dialectal relevante
    texto = str(texto).lower().strip()
    texto = re.sub(r'_+usr\w*', ' ', texto)
    texto = re.sub(r'_+url\w*', ' ', texto)
    texto = re.sub(r'\busr\b',  ' ', texto)
    texto = re.sub(r'\burl\b',  ' ', texto)
    texto = re.sub(r'\brt\b',   ' ', texto)
    texto = re.sub(r'http\S+|www\.\S+', ' ', texto)
    texto = re.sub(r'[@#]\w+', ' ', texto)
    texto = re.sub(r'\b\d+\b', ' ', texto)
    texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ]', ' ', texto)
    palabras = texto.split()
    palabras = [p for p in palabras
                if p not in STOPWORDS_ES
                and p not in TOKENS_ANONIMOS
                and len(p) > 2]
    return ' '.join(palabras).strip()

def limpiar_texto_modelo(texto):
    # conserva stopwords y puntuación para mantener el contexto semántico que el transformer aprovecha
    texto = str(texto).strip()
    texto = re.sub(r'_+usr\w*', '', texto)
    texto = re.sub(r'_+url\w*', '', texto)
    texto = re.sub(r'\busr\b',  '', texto)
    texto = re.sub(r'\burl\b',  '', texto)
    texto = re.sub(r'\brt\b',   '', texto)
    texto = re.sub(r'http\S+|www\.\S+', '', texto)
    texto = re.sub(r'[@#]\w+', '', texto)
    texto = re.sub(r'[^\w\sáéíóúüñÁÉÍÓÚÜÑ¿?¡!.,;:]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

print("Funciones de limpieza definidas")

# ============================================================
# CARGA DE DATOS
# ============================================================
df = pd.read_parquet(DATA_PATH)
df['texto'] = df['texto'].astype(str)
df['pais']  = df['pais'].astype(str)

assert "texto" in df.columns and "pais" in df.columns, \
    "El parquet debe tener columnas 'texto' y 'pais'"

print(f"Dataset cargado: {df.shape[0]} muestras, {df['pais'].nunique()} países")

# ============================================================
# EDA
# ============================================================
print("-"*60)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("-"*60)

dist = df['pais'].value_counts()
print("\nDistribución por país:")
print(dist)

df['num_palabras'] = df['texto'].apply(lambda x: len(str(x).split()))
df['num_chars']    = df['texto'].apply(lambda x: len(str(x)))
print(f"\nEstadísticas de texto:")
print(df[['num_palabras', 'num_chars']].describe())

# Gráfica 1: Barras
fig, ax = plt.subplots(figsize=(14, 5))
dist.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title("Distribución de muestras por país", fontsize=14, fontweight='bold')
ax.set_xlabel("País"); ax.set_ylabel("Cantidad")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_EDA_DIR, "eda_distribucion_barras.png"), dpi=150, bbox_inches='tight')
plt.close()

# Gráfica 2: Pie
fig, ax = plt.subplots(figsize=(10, 8))
ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%',
       startangle=90, textprops={'fontsize': 9})
ax.set_title("Proporción por clase", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_EDA_DIR, "eda_distribucion_pie.png"), dpi=150, bbox_inches='tight')
plt.close()

# Gráfica 3: Boxplot longitud
fig, ax = plt.subplots(figsize=(14, 5))
df.boxplot(column='num_palabras', by='pais', ax=ax, rot=45)
ax.set_title("Distribución de longitud (palabras) por país", fontsize=13, fontweight='bold')
ax.set_xlabel("País"); ax.set_ylabel("Número de palabras")
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(FIG_EDA_DIR, "eda_longitud.png"), dpi=150, bbox_inches='tight')
plt.close()

# Nubes de palabras por país
print("\n--- Generando nube de palabras por país ---")
for pais in dist.index.tolist():
    textos_raw   = " ".join(df[df['pais'] == pais]['texto'].astype(str))
    textos_clean = limpiar_texto_eda(textos_raw)
    if not textos_clean.strip():
        print(f"   {pais}: sin texto después de limpieza, se omite")
        continue
    fig, ax = plt.subplots(figsize=(10, 5))
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='Blues', max_words=80).generate(textos_clean)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Palabras más frecuentes — {pais}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    nombre_archivo = os.path.join(FIG_EDA_DIR, f"eda_wordcloud_{pais.lower().replace(' ', '_')}.png")
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   {pais} -> {nombre_archivo}")

# ============================================================
# PREPROCESAMIENTO Y SPLITS
# ============================================================
df['texto_eda']    = df['texto'].apply(limpiar_texto_eda)
df['texto_modelo'] = df['texto'].apply(limpiar_texto_modelo)

print("\nEjemplo de limpieza:")
print(f"  Original:     {df['texto'].iloc[0][:100]}")
print(f"  Para modelo:  {df['texto_modelo'].iloc[0][:100]}")
print(f"  Para EDA/XAI: {df['texto_eda'].iloc[0][:100]}")

paises_unicos = sorted(df['pais'].unique())
label2id = {p: i for i, p in enumerate(paises_unicos)}
id2label = {i: p for p, i in label2id.items()}
df['label'] = df['pais'].map(label2id)
NUM_LABELS = len(paises_unicos)

print(f"\n{NUM_LABELS} clases identificadas:")
for k, v in label2id.items():
    print(f"   {v}: {k}")

X     = df['texto_modelo'].to_numpy()
X_eda = df['texto_eda'].to_numpy()
y     = df['label'].to_numpy()

X_train, X_temp, y_train, y_temp, Xeda_train, Xeda_temp = train_test_split(
    X, y, X_eda, test_size=0.20, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test, Xeda_val, Xeda_test = train_test_split(
    X_temp, y_temp, Xeda_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"\nSplits: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

np.save(os.path.join(SAVE_DIR, "X_test.npy"),    X_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"),    y_test)
np.save(os.path.join(SAVE_DIR, "Xeda_test.npy"), Xeda_test)
print(f"Splits de test guardados en {SAVE_DIR}/")

# ============================================================
# TOKENIZACIÓN
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"Tokenizador guardado en: {TOKENIZER_DIR}")

class DialectDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_len):
        self.textos    = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.textos[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # RoBERTa no usa token_type_ids → NO se incluyen
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.etiquetas[idx], dtype=torch.long)
        }

train_dataset = DialectDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset   = DialectDataset(X_val,   y_val,   tokenizer, MAX_LENGTH)
test_dataset  = DialectDataset(X_test,  y_test,  tokenizer, MAX_LENGTH)
print(f"Datasets creados | max_length={MAX_LENGTH}")

# ============================================================
# MODELO BERTIN + LoRA (HIPERPARÁMETROS MEJORADOS)
# Baseline monolingüe: mide cuánto aporta el preentrenamiento
# multilingüe de XLM-T sobre texto exclusivamente español.
# ============================================================
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# r=16 (vs r=64 de XLM-T): BERTIN es monolingüe, requiere menor adaptación dialectal
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,               # rango bajo suficiente dado que el dominio (español) ya está en el preentrenamiento
    lora_alpha=32,      # alpha = 2×r es la convención estándar
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"],
    modules_to_save=["classifier"]
)

model = get_peft_model(base_model, lora_config)
# Bertin no usa gradient_checkpointing → enable_input_require_grads() no es necesario aquí
model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nBertin + LoRA listo")
print(f"   Parámetros totales:     {total_params:,}")
print(f"   Parámetros entrenables: {trainable_params:,} ({100*(trainable_params/total_params):.2f}%)")
print(f"   LoRA r={lora_config.r}, alpha={lora_config.lora_alpha}")

# ============================================================
# ENTRENAMIENTO (HIPERPARÁMETROS MEJORADOS)
# ============================================================
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)['accuracy']
    f1  = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']
    return {"accuracy": acc, "f1_macro": f1}

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=8,                  # más épocas, early stopping detiene si no mejora
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,                  # RoBERTa monolingüe es más sensible a LR altas que XLM-T multilingüe
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\n--- Iniciando entrenamiento ---")
train_result = trainer.train()
print("--- Entrenamiento completado ---")

# ============================================================
# GUARDADO DEL MODELO
# ============================================================
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Adaptadores LoRA guardados en: {FINAL_MODEL_DIR}")

merged_model = model.merge_and_unload()   # fusiona adaptadores LoRA en el modelo base para carga estándar HF
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Modelo merged guardado en: {MERGED_DIR}")

metadata = {
    "model_name":    MODEL_NAME,
    "num_labels":    NUM_LABELS,
    "label2id":      label2id,
    "id2label":      id2label,
    "max_length":    MAX_LENGTH,
    "lora_r":        lora_config.r,
    "lora_alpha":    lora_config.lora_alpha,
    "learning_rate": training_args.learning_rate,
    "epochs":        training_args.num_train_epochs,
    "train_samples": len(X_train),
    "val_samples":   len(X_val),
    "test_samples":  len(X_test),
    "train_runtime": train_result.metrics.get("train_runtime", None),
}
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"Metadata guardada en: {METADATA_PATH}")

# ============================================================
# EVALUACIÓN EN TEST
# ============================================================
print("\n" + "-"*60)
print("EVALUACIÓN EN CONJUNTO DE TEST")
print("-"*60)

preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=-1)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=paises_unicos))

acc_test = accuracy_score(y_test, y_pred)
f1_test  = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy en test: {acc_test:.4f}")
print(f"F1-Macro en test: {f1_test:.4f}")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
metadata["test_accuracy"] = round(acc_test, 4)
metadata["test_f1_macro"] = round(f1_test, 4)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

np.save(os.path.join(SAVE_DIR, "y_pred_test.npy"), y_pred)
np.save(os.path.join(SAVE_DIR, "y_proba_test.npy"), preds_output.predictions)
print(f"Predicciones guardadas en {SAVE_DIR}/")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(max(8, NUM_LABELS), max(6, NUM_LABELS - 2)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=paises_unicos, yticklabels=paises_unicos)
plt.title("Matriz de Confusión — Test Set", fontsize=14, fontweight='bold')
plt.ylabel("Etiqueta Real"); plt.xlabel("Predicción")
plt.tight_layout()
plt.savefig(os.path.join(FIG_TRAIN_DIR, "matriz_confusion_bertin.png"), dpi=150, bbox_inches='tight')
plt.close()

# Curva de aprendizaje
history   = trainer.state.log_history
eval_data = [(h['epoch'], h['eval_f1_macro'], h['eval_loss'])
             for h in history if 'eval_f1_macro' in h]

if eval_data:
    epochs_e, f1s, losses = zip(*eval_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_e, f1s, 'b-o', linewidth=2, markersize=6)
    ax1.set_title("F1-Macro por Época (Validación)")
    ax1.set_xlabel("Época"); ax1.set_ylabel("F1-Macro")
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs_e, losses, 'r-o', linewidth=2, markersize=6)
    ax2.set_title("Loss por Época (Validación)")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_TRAIN_DIR, "curva_aprendizaje_bertin.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO — BERTIN + LoRA")
print("="*60)
print(f"""
Modelo:        {MODEL_NAME}
F1-Macro test: {f1_test:.4f}
Accuracy test: {acc_test:.4f}
Modelo guardado en: {SAVE_DIR}/

Para análisis XAI, ejecutar:
  python src/xai/xai_methods.py
""")