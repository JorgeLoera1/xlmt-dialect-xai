# ============================================================
# CLASIFICACIÓN DE DIALECTOS — SIN FINE-TUNING
# Modelo: cardiffnlp/twitter-xlm-roberta-base (pesos congelados)
# Enfoque: Extracción de embeddings [CLS] + clasificador ligero
# Referencia: Experimento_16Marzo.py (con LoRA fine-tuning)
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
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from scipy.stats import spearmanr
from lime.lime_text import LimeTextExplainer

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
print(f"PyTorch: {torch.__version__}")

MODEL_NAME    = "cardiffnlp/twitter-xlm-roberta-base"
MAX_LENGTH    = 128
BATCH_SIZE    = 64
DATA_PATH     = "data/Datos_dialectos.parquet"
SAVE_DIR      = "model_outputs/xlmt_frozen"
FIG_TRAIN_DIR = "results/figures/training"

os.makedirs(SAVE_DIR,      exist_ok=True)
os.makedirs(FIG_TRAIN_DIR, exist_ok=True)
print(f"Modelo base (congelado): {MODEL_NAME}")
print(f"Directorio de guardado:  {SAVE_DIR}")

# ============================================================
# FUNCIONES DE LIMPIEZA (idénticas al experimento de referencia)
# Mismo preprocesamiento garantiza que las diferencias en F1 se
# atribuyen únicamente al fine-tuning, no al texto de entrada.
# ============================================================
STOPWORDS_ES    = set(stopwords.words('spanish'))
TOKENS_ANONIMOS = {
    'usr', '_usr', '__usr', 'user', '_user',
    'url', '_url', '__url', 'link', '_link',
    'http', 'https', 'rt'
}

def limpiar_texto_eda(texto):
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
# CARGA DE DATOS Y SPLITS
# ============================================================
df = pd.read_parquet(DATA_PATH)
df['texto'] = df['texto'].astype(str)
df['pais']  = df['pais'].astype(str)
print(f"Dataset cargado: {df.shape[0]} muestras, {df['pais'].nunique()} países")

df['texto_eda']    = df['texto'].apply(limpiar_texto_eda)
df['texto_modelo'] = df['texto'].apply(limpiar_texto_modelo)

paises_unicos = sorted(df['pais'].unique())
label2id = {p: i for i, p in enumerate(paises_unicos)}
id2label = {i: p for p, i in label2id.items()}
df['label'] = df['pais'].map(label2id)
NUM_LABELS = len(paises_unicos)

print(f"\n{NUM_LABELS} clases: {paises_unicos}")

X     = df['texto_modelo'].to_numpy()
X_eda = df['texto_eda'].to_numpy()
y     = df['label'].to_numpy()

X_train, X_temp, y_train, y_temp, Xeda_train, Xeda_temp = train_test_split(
    X, y, X_eda, test_size=0.20, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test, Xeda_val, Xeda_test = train_test_split(
    X_temp, y_temp, Xeda_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"\nSplits: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

ref_dir = "model_outputs/xlmt_lora_v2"
if os.path.exists(os.path.join(ref_dir, "y_test.npy")):
    ref_y_test = np.load(os.path.join(ref_dir, "y_test.npy"))
    # la misma semilla garantiza splits idénticos, pero se verifica explícitamente para asegurar comparación justa
    assert np.array_equal(y_test, ref_y_test), "Los splits de test no coinciden con el experimento de referencia"
    print("Split de test verificado: coincide con experimento de referencia")
else:
    print("Advertencia: experimento de referencia no encontrado, verificación omitida")

# ============================================================
# CARGA DEL MODELO CONGELADO
# ============================================================
print(f"\nCargando {MODEL_NAME} (pesos congelados)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder   = AutoModel.from_pretrained(MODEL_NAME)

# Congelar TODOS los parámetros del transformer
for param in encoder.parameters():
    param.requires_grad = False

encoder.to(device)
encoder.eval()

total_params = sum(p.numel() for p in encoder.parameters())
print(f"Modelo cargado | Parámetros: {total_params:,} | Todos congelados")

# ============================================================
# EXTRACCIÓN DE EMBEDDINGS [CLS]
# ============================================================
def extraer_embeddings(textos, batch_size=BATCH_SIZE, desc=""):
    """Extrae el vector [CLS] de cada texto usando el encoder congelado."""
    embeddings = []
    n_batches  = (len(textos) + batch_size - 1) // batch_size

    t0 = time.time()
    with torch.no_grad():
        for i in range(n_batches):
            batch = textos[i * batch_size : (i + 1) * batch_size].tolist()
            enc   = tokenizer(
                batch,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            enc.pop("token_type_ids", None)

            outputs = encoder(**enc)
            cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()   # token [CLS] = representación global de la secuencia (convenio BERT/RoBERTa)
            embeddings.append(cls_vec)

            if (i + 1) % 100 == 0 or (i + 1) == n_batches:
                elapsed = time.time() - t0
                print(f"  {desc} batch {i+1}/{n_batches} — {elapsed:.0f}s transcurridos")

    return np.vstack(embeddings)


# la extracción toma ~30 min; se cachea en disco para no repetir si el script se interrumpe
emb_train_path = os.path.join(SAVE_DIR, "emb_train.npy")
emb_val_path   = os.path.join(SAVE_DIR, "emb_val.npy")
emb_test_path  = os.path.join(SAVE_DIR, "emb_test.npy")

if os.path.exists(emb_train_path):
    print("\nCargando embeddings desde disco...")
    E_train = np.load(emb_train_path)
    E_val   = np.load(emb_val_path)
    E_test  = np.load(emb_test_path)
    print(f"  Train: {E_train.shape} | Val: {E_val.shape} | Test: {E_test.shape}")
else:
    print("\nExtrayendo embeddings [CLS] — esto tomará varios minutos...")
    t_total = time.time()

    print("  Extrayendo train...")
    E_train = extraer_embeddings(X_train, desc="Train")
    np.save(emb_train_path, E_train)
    print(f"  Train guardado: {E_train.shape}")

    print("  Extrayendo val...")
    E_val = extraer_embeddings(X_val, desc="Val")
    np.save(emb_val_path, E_val)
    print(f"  Val guardado: {E_val.shape}")

    print("  Extrayendo test...")
    E_test = extraer_embeddings(X_test, desc="Test")
    np.save(emb_test_path, E_test)
    print(f"  Test guardado: {E_test.shape}")

    print(f"\nExtracción total: {(time.time()-t_total)/60:.1f} min")

# ============================================================
# CLASIFICADOR LIGERO — Regresión Logística
# ============================================================
print("\n" + "="*60)
print("ENTRENAMIENTO — Regresión Logística sobre embeddings [CLS]")
print("="*60)

t0 = time.time()
clf = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='saga',             # estocástico, eficiente para datasets grandes con multinomial
    multi_class='multinomial',
    n_jobs=-1,
    random_state=SEED,
    verbose=1
)
clf.fit(E_train, y_train)
t_clf = time.time() - t0
print(f"\nRegresión Logística entrenada en {t_clf:.1f}s")

# Guardar clasificador
clf_path = os.path.join(SAVE_DIR, "clasificador_logreg.pkl")
with open(clf_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Clasificador guardado en: {clf_path}")

# ============================================================
# EVALUACIÓN EN VALIDACIÓN
# ============================================================
y_pred_val = clf.predict(E_val)
acc_val = accuracy_score(y_val, y_pred_val)
f1_val  = f1_score(y_val, y_pred_val, average='macro')
print(f"\nValidación → Accuracy: {acc_val:.4f} | F1-Macro: {f1_val:.4f}")

# ============================================================
# EVALUACIÓN EN TEST
# ============================================================
print("\n" + "-"*60)
print("EVALUACIÓN EN CONJUNTO DE TEST")
print("-"*60)

y_pred_test  = clf.predict(E_test)
y_proba_test = clf.predict_proba(E_test)

acc_test = accuracy_score(y_test, y_pred_test)
f1_test  = f1_score(y_test, y_pred_test, average='macro')

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_test, target_names=paises_unicos))
print(f"Accuracy en test: {acc_test:.4f}")
print(f"F1-Macro en test: {f1_test:.4f}")

np.save(os.path.join(SAVE_DIR, "y_pred_test.npy"),  y_pred_test)
np.save(os.path.join(SAVE_DIR, "y_proba_test.npy"), y_proba_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"),       y_test)
np.save(os.path.join(SAVE_DIR, "X_test.npy"),       X_test)

ref_pred_path = os.path.join(ref_dir, "y_pred_test.npy")
if os.path.exists(ref_pred_path):
    ref_y_pred = np.load(ref_pred_path)
    ref_y_test_cmp = np.load(os.path.join(ref_dir, "y_test.npy"))
    ref_acc    = accuracy_score(ref_y_test_cmp, ref_y_pred)
    ref_f1     = f1_score(ref_y_test_cmp, ref_y_pred, average='macro')

    print("\n" + "="*60)
    print("COMPARACIÓN: Sin fine-tuning vs Con fine-tuning (LoRA)")
    print("="*60)
    print(f"{'Métrica':<20} {'Sin FT (LogReg)':<20} {'Con FT (LoRA)':<20} {'Diferencia'}")
    print(f"{'-'*80}")
    print(f"{'Accuracy':<20} {acc_test:.4f}{'':>14} {ref_acc:.4f}{'':>14} {ref_acc - acc_test:+.4f}")
    print(f"{'F1-Macro':<20} {f1_test:.4f}{'':>14} {ref_f1:.4f}{'':>14} {ref_f1 - f1_test:+.4f}")
else:
    print("\nComparación con LoRA omitida (ejecutar train_xlmt_v2.py primero)")

# ============================================================
# VISUALIZACIONES
# ============================================================

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(max(8, NUM_LABELS), max(6, NUM_LABELS - 2)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=paises_unicos, yticklabels=paises_unicos)
plt.title("Matriz de Confusión — Sin Fine-tuning (LogReg sobre [CLS])",
          fontsize=14, fontweight='bold')
plt.ylabel("Etiqueta Real")
plt.xlabel("Predicción")
plt.tight_layout()
plt.savefig(os.path.join(FIG_TRAIN_DIR, "matriz_confusion_baseline.png"), dpi=150, bbox_inches='tight')
plt.close()

# Comparación de F1 por clase
report_ft  = classification_report(ref_y_test, ref_y_pred,
                                    target_names=paises_unicos, output_dict=True)
report_sft = classification_report(y_test, y_pred_test,
                                    target_names=paises_unicos, output_dict=True)

f1_ft_por_clase  = [report_ft[p]['f1-score']  for p in paises_unicos]
f1_sft_por_clase = [report_sft[p]['f1-score'] for p in paises_unicos]

x   = np.arange(len(paises_unicos))
ancho = 0.35
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - ancho/2, f1_ft_por_clase,  ancho, label='Con fine-tuning (LoRA)', color='steelblue',  alpha=0.85)
ax.bar(x + ancho/2, f1_sft_por_clase, ancho, label='Sin fine-tuning (LogReg)', color='darkorange', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(paises_unicos, rotation=45, ha='right')
ax.set_ylabel("F1-Score por clase")
ax.set_title("F1 por País: Sin Fine-tuning vs Con Fine-tuning", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_TRAIN_DIR, "comparacion_f1_por_pais.png"), dpi=150, bbox_inches='tight')
plt.close()

# Resumen de métricas globales
metricas_labels = ['Accuracy', 'F1-Macro']
valores_ft  = [ref_acc, ref_f1]
valores_sft = [acc_test, f1_test]

x2 = np.arange(len(metricas_labels))
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x2 - 0.2, valores_ft,  0.35, label='Con fine-tuning (LoRA)', color='steelblue',  alpha=0.85)
ax.bar(x2 + 0.2, valores_sft, 0.35, label='Sin fine-tuning (LogReg)', color='darkorange', alpha=0.85)
for i, (vft, vsft) in enumerate(zip(valores_ft, valores_sft)):
    ax.text(i - 0.2, vft  + 0.005, f"{vft:.3f}",  ha='center', fontsize=10, fontweight='bold')
    ax.text(i + 0.2, vsft + 0.005, f"{vsft:.3f}", ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x2)
ax.set_xticklabels(metricas_labels)
ax.set_ylabel("Valor")
ax.set_ylim(0, 0.7)
ax.set_title("Métricas globales: Sin vs Con Fine-tuning", fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_TRAIN_DIR, "comparacion_global.png"), dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualizaciones guardadas")

# ============================================================
# XAI — LIME (agnóstico al modelo)
# ============================================================
print("\n" + "="*60)
print("EXPLICABILIDAD — LIME (sobre clasificador frozen)")
print("="*60)

def predecir_proba_frozen(textos):
    """Pipeline completo: texto → embedding [CLS] → probabilidad del clasificador."""
    if isinstance(textos, str):
        textos = [textos]
    textos_arr = np.array(textos)
    embs = extraer_embeddings(textos_arr, batch_size=32, desc="")
    return clf.predict_proba(embs)

def predecir_clase_frozen(texto):
    probs = predecir_proba_frozen([texto])[0]
    return id2label[np.argmax(probs)], probs

lime_explainer = LimeTextExplainer(
    class_names=paises_unicos,
    split_expression=r'\W+',
    random_state=SEED
)

lime_resultados = []
for idx in range(5):
    texto_idx  = X_test[idx]
    label_real = id2label[y_test[idx]]
    clase_pred, probs = predecir_clase_frozen(texto_idx)

    print(f"\nEjemplo {idx+1}:")
    print(f"  Texto: '{texto_idx[:100]}...'")
    print(f"  Real: {label_real} | Predicción: {clase_pred}")

    exp = lime_explainer.explain_instance(
        texto_idx, predecir_proba_frozen,
        num_features=15,
        num_samples=300,
        top_labels=3
    )
    importancias = exp.as_list(label=label2id[clase_pred])

    fig = exp.as_pyplot_figure(label=label2id[clase_pred])
    plt.title(f"LIME (Sin FT) — Ejemplo {idx+1} (Pred: {clase_pred})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"lime_ejemplo_{idx+1}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    lime_resultados.append({
        'idx': idx, 'texto': texto_idx,
        'real': label_real, 'pred': clase_pred,
        'importancias': importancias
    })
    print(f"  Top palabras: {[f'{w}({s:.3f})' for w, s in importancias[:5]]}")

lime_path = os.path.join(SAVE_DIR, "lime_explicaciones.json")
with open(lime_path, "w", encoding="utf-8") as f:
    json.dump(lime_resultados, f, ensure_ascii=False, indent=2)
print(f"\nExplicaciones LIME guardadas en: {lime_path}")

# ============================================================
# CUANTIFICACIÓN DE EXPLICABILIDAD
# ============================================================
print("\n" + "="*60)
print("CUANTIFICACIÓN DE EXPLICABILIDAD")
print("="*60)

def calcular_faithfulness(texto, importancias, top_k=5):
    clase_pred, probs_orig = predecir_clase_frozen(texto)
    conf_orig = probs_orig.max()
    clase_idx = int(np.argmax(probs_orig))
    palabras_pos = [w for w, s in importancias if s > 0][:top_k]
    if not palabras_pos:
        return 0.0, conf_orig
    texto_ocluido = texto
    for palabra in palabras_pos:
        texto_ocluido = re.sub(r'\b' + re.escape(palabra) + r'\b', '', texto_ocluido)
    texto_ocluido = re.sub(r'\s+', ' ', texto_ocluido).strip()
    probs_ocluido = predecir_proba_frozen([texto_ocluido])[0]
    conf_ocluida  = probs_ocluido[clase_idx]
    faithfulness  = float(conf_orig - conf_ocluida)
    return max(0.0, min(1.0, faithfulness)), float(conf_orig)

def calcular_complexity(importancias):
    if not importancias:
        return float('nan')
    scores = np.abs([s for _, s in importancias])
    if scores.sum() == 0:
        return float('nan')
    probs       = scores / scores.sum()
    entropy     = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0

def calcular_stability(texto, importancias_orig, n_perturbaciones=5, noise_level=0.1):
    correlaciones   = []
    palabras_orig   = [w for w, _ in importancias_orig]
    scores_orig_map = dict(importancias_orig)
    for _ in range(n_perturbaciones):
        palabras   = texto.split()
        n_drop     = max(1, int(len(palabras) * noise_level))
        idx_drop   = np.random.choice(len(palabras), n_drop, replace=False)
        texto_pert = ' '.join([p for i, p in enumerate(palabras) if i not in idx_drop])
        try:
            exp_pert  = lime_explainer.explain_instance(
                texto_pert, predecir_proba_frozen,
                num_features=len(importancias_orig),
                num_samples=150
            )
            clase_idx_pert    = int(np.argmax(predecir_proba_frozen([texto_pert])[0]))
            importancias_pert = dict(exp_pert.as_list(label=clase_idx_pert))
            common = set(palabras_orig) & set(importancias_pert.keys())
            if len(common) < 3:
                continue
            scores_a = [scores_orig_map[w]   for w in common]
            scores_b = [importancias_pert[w] for w in common]
            rho, _ = spearmanr(scores_a, scores_b)
            if not np.isnan(rho):
                correlaciones.append(rho)
        except Exception:
            pass
    return float(np.mean(correlaciones)) if correlaciones else 0.0

N_EVAL = min(30, len(X_test))
print(f"Calculando métricas XAI en {N_EVAL} muestras del test...")

resultados_xai = []
for idx in range(N_EVAL):
    texto      = X_test[idx]
    real       = id2label[y_test[idx]]
    clase_pred, probs = predecir_clase_frozen(texto)
    clase_idx  = int(np.argmax(probs))

    exp          = lime_explainer.explain_instance(
        texto, predecir_proba_frozen, num_features=15, num_samples=200
    )
    importancias = exp.as_list(label=clase_idx)

    faith = calcular_faithfulness(texto, importancias)[0]
    compl = calcular_complexity(importancias)
    comp  = calcular_faithfulness(texto, importancias)[0]   # comprehensiveness = faithfulness top-5
    stab  = calcular_stability(texto, importancias)

    resultados_xai.append({
        'idx':               idx,
        'texto':             texto[:80],
        'real':              real,
        'pred':              clase_pred,
        'correcto':          real == clase_pred,
        'faithfulness':      faith,
        'complexity':        compl,
        'comprehensiveness': comp,
        'stability':         stab,
    })

    if idx % 5 == 0:
        print(f"  [{idx+1}/{N_EVAL}] faith={faith:.3f} | compl={compl:.3f} "
              f"| comp={comp:.3f} | stab={stab:.3f}")

df_xai = pd.DataFrame(resultados_xai)
resumen = df_xai[['faithfulness', 'complexity', 'comprehensiveness', 'stability']].describe()
print("\nResumen métricas XAI:")
print(resumen)

df_xai.to_csv(os.path.join(SAVE_DIR, "xai_metricas_por_muestra.csv"), index=False)
resumen.to_csv(os.path.join(SAVE_DIR, "xai_metricas_resumen.csv"))

# Visualización métricas XAI
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metricas = ['faithfulness', 'complexity', 'comprehensiveness', 'stability']
colores  = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
titulos  = ['Faithfulness\n(↑ mejor)', 'Complexity\n(↓ mejor)',
            'Comprehensiveness\n(↑ mejor)', 'Stability\n(↑ mejor)']

for i, (met, col, tit) in enumerate(zip(metricas, colores, titulos)):
    correct_vals   = df_xai[df_xai['correcto'] == True][met].dropna()
    incorrect_vals = df_xai[df_xai['correcto'] == False][met].dropna()
    axes[i].hist(correct_vals,   bins=10, alpha=0.7, color=col,
                 label='Correcto',   edgecolor='black')
    axes[i].hist(incorrect_vals, bins=10, alpha=0.7, color='salmon',
                 label='Incorrecto', edgecolor='black')
    axes[i].set_title(tit, fontsize=12, fontweight='bold')
    axes[i].set_xlabel(met)
    axes[i].set_ylabel("Frecuencia")
    axes[i].legend(fontsize=9)
    if len(correct_vals) > 0:
        axes[i].axvline(correct_vals.mean(), color=col, linestyle='--', linewidth=2)

plt.suptitle("Métricas XAI (Sin Fine-tuning): Predicciones Correctas vs Incorrectas",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_TRAIN_DIR, "xai_metricas_baseline.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# METADATA Y RESUMEN FINAL
# ============================================================
metadata = {
    "model_name":         MODEL_NAME,
    "fine_tuning":        False,
    "classifier":         "LogisticRegression(C=1.0, solver=saga, multinomial)",
    "embedding":          "CLS token (768 dims)",
    "num_labels":         NUM_LABELS,
    "label2id":           label2id,
    "id2label":           id2label,
    "max_length":         MAX_LENGTH,
    "train_samples":      len(X_train),
    "val_samples":        len(X_val),
    "test_samples":       len(X_test),
    "embedding_time_s":   None,   # se puede actualizar manualmente si se desea
    "clf_train_time_s":   round(t_clf, 2),
    "val_accuracy":       round(acc_val, 4),
    "val_f1_macro":       round(f1_val, 4),
    "test_accuracy":      round(acc_test, 4),
    "test_f1_macro":      round(f1_test, 4),
    "ref_test_accuracy":  round(ref_acc, 4),
    "ref_test_f1_macro":  round(ref_f1, 4),
}

metadata_path = os.path.join(SAVE_DIR, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"\nMetadata guardada en: {metadata_path}")

print("\n" + "="*60)
print("PIPELINE SIN FINE-TUNING — COMPLETADO")
print("="*60)
print(f"""
Modelo base (congelado): {MODEL_NAME}
Clasificador:            Regresión Logística sobre [CLS]

Resultados en test:
  Accuracy  — Sin FT: {acc_test:.4f}  |  Con FT (LoRA): {ref_acc:.4f}  |  Δ {ref_acc-acc_test:+.4f}
  F1-Macro  — Sin FT: {f1_test:.4f}  |  Con FT (LoRA): {ref_f1:.4f}  |  Δ {ref_f1-f1_test:+.4f}

Archivos en {SAVE_DIR}/:
  metadata.json
  emb_train.npy | emb_val.npy | emb_test.npy
  clasificador_logreg.pkl
  y_pred_test.npy | y_proba_test.npy
  lime_explicaciones.json
  xai_metricas_por_muestra.csv
""")
