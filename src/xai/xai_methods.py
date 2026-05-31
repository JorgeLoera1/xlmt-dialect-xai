# ============================================================
# XAI STANDALONE — Carga modelo merged y ejecuta explicabilidad
# Modelo: cardiffnlp/twitter-xlm-roberta-base (XLM-T) merged
# Métodos: LIME, SHAP, Integrated Gradients, Attention Rollout
#          + Métricas: Faithfulness, Comprehensiveness,
#                      Complexity, Stability
# Fixes aplicados:
#   - LIME KeyError: usa available_labels() como fallback
#   - IG delta: n_steps 50 → 300 para mejor convergencia
#   - Attention Rollout: attn_implementation="eager" explícito
# Autor: Jorge Antonio Loera Grande
# ============================================================

import os, re, json, time, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
from scipy.stats import spearmanr
from collections import Counter
import shap

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ============================================================
# RUTAS — ajusta SAVE_DIR si es necesario
# ============================================================
SAVE_DIR   = "model_outputs/xlmt_lora_v2"
MERGED_DIR = os.path.join(SAVE_DIR, "modelo_merged")
META_PATH  = os.path.join(SAVE_DIR, "metadata.json")
MAX_LENGTH = 128

FIG_XAI_DIR     = "results/figures/xai"
FIG_XAI_ROBUSTO = os.path.join(FIG_XAI_DIR, "robusto")
FIG_XAI_PAISES  = os.path.join(FIG_XAI_DIR, "por_pais")
os.makedirs(FIG_XAI_DIR,     exist_ok=True)
os.makedirs(FIG_XAI_ROBUSTO, exist_ok=True)
os.makedirs(FIG_XAI_PAISES,  exist_ok=True)

# Verificar archivos requeridos
for ruta in [MERGED_DIR, META_PATH,
             os.path.join(SAVE_DIR, "X_test.npy"),
             os.path.join(SAVE_DIR, "y_test.npy"),
             os.path.join(SAVE_DIR, "Xeda_test.npy")]:
    assert os.path.exists(ruta), f"No encontrado: {ruta}"
print("Todos los archivos requeridos encontrados")

# ============================================================
# CARGAR METADATA, MODELO Y SPLITS
# ============================================================
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

id2label      = {int(k): v for k, v in metadata["id2label"].items()}
label2id      = metadata["label2id"]
paises_unicos = sorted(label2id.keys())
NUM_LABELS    = metadata["num_labels"]
MAX_LENGTH    = metadata.get("max_length", 128)

print(f"Metadata cargada: {NUM_LABELS} clases, max_length={MAX_LENGTH}")

print("\nCargando tokenizer y modelo merged...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)

# ── FIX Attention Rollout: cargar con attn_implementation="eager" ──
# Evita el warning de SDPA que no soporta output_attentions=True
xai_model = AutoModelForSequenceClassification.from_pretrained(
    MERGED_DIR,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    attn_implementation="eager"   # ← FIX: fuerza implementación manual
)
xai_model.to(device)
xai_model.eval()
print(f"Modelo merged cargado desde: {MERGED_DIR}")

X_test    = np.load(os.path.join(SAVE_DIR, "X_test.npy"),    allow_pickle=True)
y_test    = np.load(os.path.join(SAVE_DIR, "y_test.npy"))
Xeda_test = np.load(os.path.join(SAVE_DIR, "Xeda_test.npy"), allow_pickle=True)
print(f"Test set cargado: {len(X_test)} muestras")

# ============================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================
def predecir_proba(textos):
    if isinstance(textos, str):
        textos = [textos]
    encodings = tokenizer(
        list(textos),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    encodings.pop("token_type_ids", None)
    with torch.no_grad():
        logits = xai_model(**encodings).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()

def predecir_clase(texto):
    probs = predecir_proba([texto])[0]
    return id2label[np.argmax(probs)], probs

# Prueba rápida
clase, probs = predecir_clase(X_test[0])
print(f"\nPrueba rápida:")
print(f"  Texto:      '{X_test[0][:80]}...'")
print(f"  Predicción: {clase} | Real: {id2label[y_test[0]]}")

# ============================================================
# LIME EXPLAINER (instancia global, usada en métricas también)
# ============================================================
lime_explainer = LimeTextExplainer(
    class_names=paises_unicos,
    split_expression=r'\W+',   # separa por no-palabras; cubre tildes, emojis y puntuación en tweets
    random_state=SEED
)

def explicar_con_lime(texto, num_features=15, num_samples=500):
    # top_labels=NUM_LABELS garantiza que todas las clases están en available_labels(); sin esto LIME lanza KeyError
    exp = lime_explainer.explain_instance(
        texto, predecir_proba,
        num_features=num_features,
        num_samples=num_samples,
        top_labels=NUM_LABELS   # ← todas las clases siempre disponibles
    )
    return exp

def get_importancias_seguro(exp, clase_idx):
    # fallback seguro: si clase_idx no está en available_labels(), usa el label disponible con mayor prob
    disponibles = exp.available_labels()
    if clase_idx in disponibles:
        return exp.as_list(label=clase_idx), clase_idx
    else:
        # Fallback: usar el label disponible con mayor score
        fallback = disponibles[0]
        print(f"    [LIME fallback] label {clase_idx} no disponible, "
              f"usando {fallback} ({id2label[fallback]})")
        return exp.as_list(label=fallback), fallback

# ============================================================
# XAI 1: LIME
# ============================================================
print("\n" + "-"*60)
print("EXPLICABILIDAD — LIME")
print("-"*60)

lime_resultados = []
for idx in range(5):
    texto_idx  = X_test[idx]
    label_real = id2label[y_test[idx]]
    clase_pred, probs = predecir_clase(texto_idx)
    clase_idx = int(np.argmax(probs))

    print(f"\nEjemplo {idx+1}:")
    print(f"  Texto: '{texto_idx[:100]}...'")
    print(f"  Real: {label_real} | Predicción: {clase_pred}")

    exp = explicar_con_lime(texto_idx)
    importancias, label_usado = get_importancias_seguro(exp, clase_idx)

    fig = exp.as_pyplot_figure(label=label_usado)
    plt.title(f"LIME — Ejemplo {idx+1} (Pred: {id2label[label_usado]})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_XAI_DIR, f"lime_ejemplo_{idx+1}.png"), dpi=150, bbox_inches='tight')
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
# XAI 2: SHAP
# ============================================================
print("\n" + "-"*60)
print("EXPLICABILIDAD — SHAP")
print("-"*60)

shap_explainer = shap.Explainer(
    predecir_proba,
    masker=shap.maskers.Text(tokenizer=r'\W+'),
    output_names=paises_unicos
)

N_SHAP   = 20
idx_shap = np.random.choice(len(X_test), N_SHAP, replace=False)
textos_shap = X_test[idx_shap].tolist()
np.save(os.path.join(SAVE_DIR, "shap_idx.npy"), idx_shap)

print(f"Calculando SHAP values para {N_SHAP} muestras...")
t0 = time.time()
shap_values = shap_explainer(textos_shap)
print(f"SHAP calculado en {time.time()-t0:.1f}s")

shap_path = os.path.join(SAVE_DIR, "shap_values.pkl")
with open(shap_path, "wb") as f:
    pickle.dump(shap_values, f)
print(f"SHAP values guardados en: {shap_path}")

# Gráficas globales por país (top 3)
for clase_idx, clase_nombre in enumerate(paises_unicos[:3]):
    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_values[:, :, clase_idx].mean(0), max_display=15, show=False)
    plt.title(f"SHAP — Importancia global para '{clase_nombre}'", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_XAI_DIR, f"shap_global_{clase_nombre}.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_shap_texto(shap_vals_ejemplo, idx_ejemplo=0, top_n=15):
    tokens = shap_vals_ejemplo.data
    values = shap_vals_ejemplo.values

    for clase_idx, clase_nombre in enumerate(paises_unicos[:3]):
        scores     = values[:, clase_idx]
        top_idx    = np.argsort(np.abs(scores))[-top_n:][::-1]
        top_tokens = [tokens[i] for i in top_idx]
        top_scores = [scores[i] for i in top_idx]
        colors     = ['#d73027' if s > 0 else '#4575b4' for s in top_scores]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(top_tokens)), top_scores, color=colors, edgecolor='black')
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"SHAP local — Ejemplo {idx_ejemplo+1} | Clase: {clase_nombre}",
                     fontsize=12, fontweight='bold')
        rojo = mpatches.Patch(color='#d73027', label='Favorece esta clase')
        azul = mpatches.Patch(color='#4575b4', label='En contra de esta clase')
        ax.legend(handles=[rojo, azul])
        plt.tight_layout()
        fname = os.path.join(FIG_XAI_DIR, f"shap_local_ejemplo{idx_ejemplo+1}_{clase_nombre}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {fname}")

print("\nVisualizando SHAP local para los primeros 3 ejemplos:")
for i in range(3):
    print(f"\n  Ejemplo {i+1}:")
    plot_shap_texto(shap_values[i], idx_ejemplo=i)

# ============================================================
# XAI 3: INTEGRATED GRADIENTS
# Fixes aplicados:
#   - int(label_idx) para evitar AssertionError de numpy.int64
#   - n_steps=300 (antes 50) para mejor convergencia del delta
#   - baseline con pad_token_id correcto
# ============================================================
print("\n" + "-"*60)
print("EXPLICABILIDAD — Integrated Gradients")
print("-"*60)

from captum.attr import LayerIntegratedGradients

def forward_for_captum(input_ids, attention_mask):
    return xai_model(input_ids=input_ids, attention_mask=attention_mask).logits

def explicar_ig(texto, label_idx=None):
    encoding = tokenizer(
        texto, max_length=MAX_LENGTH, padding='max_length',
        truncation=True, return_tensors='pt'
    ).to(device)
    encoding.pop("token_type_ids", None)

    input_ids      = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    if label_idx is None:
        with torch.no_grad():
            label_idx = int(
                forward_for_captum(input_ids, attention_mask).argmax().item()
            )

    # FIX 1: convertir a int nativo — Captum no acepta numpy.int64
    label_idx = int(label_idx)

    # FIX 2: baseline con pad_token_id (no zeros que es <unk>)
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

    lig = LayerIntegratedGradients(
        forward_for_captum,
        xai_model.roberta.embeddings.word_embeddings
    )

    # FIX 3: n_steps=300 para mejor convergencia del delta
    # FIX OOM: internal_batch_size=8 → procesa los 300 pasos en
    # mini-lotes de 8, evita el CUDA out of memory en 6GB VRAM
    attributions, delta = lig.attribute(
        input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=label_idx,
        return_convergence_delta=True,
        n_steps=300,
        internal_batch_size=8   # ← FIX OOM: mini-lotes dentro de IG
    )

    attributions_sum = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())

    tokens_especiales = {'<pad>', '<s>', '</s>', '▁', 'Ġ', '<unk>'}
    valid = [(t.lstrip('▁').lstrip('Ġ'), float(a))
             for t, a in zip(tokens, attributions_sum)
             if t not in tokens_especiales and t.strip() != '']

    return valid, id2label[label_idx], delta.item()

def plot_ig(token_attributions, clase_pred, titulo=""):
    tokens, attrs = zip(*token_attributions[:20])
    attrs      = np.array(attrs)
    attrs_norm = (attrs - attrs.min()) / (attrs.max() - attrs.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(max(10, len(tokens)*0.5), 3))
    colors = ['#d73027' if a > 0.5 else '#4575b4' for a in attrs_norm]
    ax.bar(range(len(tokens)), attrs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Atribución IG")
    ax.set_title(f"Integrated Gradients — Clase: {clase_pred} {titulo}", fontsize=12)
    rojo = mpatches.Patch(color='#d73027', label='Alta importancia')
    azul = mpatches.Patch(color='#4575b4', label='Baja importancia')
    ax.legend(handles=[rojo, azul])
    plt.tight_layout()
    return fig

ig_resultados = []
for idx in range(5):
    texto_idx      = X_test[idx]
    label_real     = id2label[y_test[idx]]
    label_pred_idx = int(np.argmax(predecir_proba([texto_idx])[0]))

    token_attrs, clase_pred, delta = explicar_ig(texto_idx, label_pred_idx)

    # Interpretar delta
    calidad = "OK" if abs(delta) < 0.05 else \
              "aceptable" if abs(delta) < 0.5 else "mejorable"
    print(f"Ejemplo {idx+1} | delta={delta:.4f} [{calidad}] | Pred: {clase_pred}")

    fig = plot_ig(token_attrs, clase_pred, f"— Ejemplo {idx+1} (Real: {label_real})")
    plt.savefig(os.path.join(FIG_XAI_DIR, f"ig_ejemplo_{idx+1}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    ig_resultados.append({
        'idx': idx, 'texto': texto_idx,
        'real': label_real, 'pred': clase_pred,
        'delta': delta,
        'atribuciones': [(t, float(a)) for t, a in token_attrs]
    })

ig_path = os.path.join(SAVE_DIR, "ig_atribuciones.json")
with open(ig_path, "w", encoding="utf-8") as f:
    json.dump(ig_resultados, f, ensure_ascii=False, indent=2)
print(f"Atribuciones IG guardadas en: {ig_path}")

# ============================================================
# XAI 4: ATTENTION ROLLOUT
# Fix: attn_implementation="eager" ya aplicado al cargar modelo
# ============================================================
print("\n" + "-"*60)
print("EXPLICABILIDAD — Attention Rollout")
print("-"*60)

TOKENS_FILTRO = {'<pad>', '<s>', '</s>', '▁', 'Ġ', '<unk>'}  # ▁/Ġ son prefijos de subword (SentencePiece/BPE), no forman parte de la palabra visible

def get_attention_rollout(texto):
    encoding = tokenizer(
        texto, max_length=MAX_LENGTH, padding='max_length',
        truncation=True, return_tensors='pt'
    ).to(device)

    input_ids      = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = xai_model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True   # funciona sin warning gracias a eager
        )

    attn_matrices = [a.squeeze(0).mean(0).cpu().numpy() for a in outputs.attentions]

    rollout = np.eye(attn_matrices[0].shape[0])
    for attn in attn_matrices:
        attn_res = attn + np.eye(attn.shape[0])
        row_sums = attn_res.sum(axis=-1, keepdims=True)
        attn_res = attn_res / (row_sums + 1e-10)
        rollout  = rollout @ attn_res

    cls_rollout = rollout[0]
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().cpu())

    valid = []
    for t, s in zip(tokens, cls_rollout):
        if t in TOKENS_FILTRO:
            continue
        t_display = t.lstrip('▁').lstrip('Ġ')
        if t_display.strip() == '':
            continue
        valid.append((t_display, float(s)))
    return valid

def plot_attention_heatmap(token_scores, titulo=""):
    tokens, scores = zip(*token_scores[:25])
    scores = np.array(scores).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(max(12, len(tokens)*0.55), 2.5))
    im = ax.imshow(scores, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.4, shrink=0.5)
    ax.set_title(f"Attention Rollout {titulo}", fontsize=12)
    plt.tight_layout()
    return fig

attn_resultados = []
for idx in range(5):
    rollout_scores = get_attention_rollout(X_test[idx])
    clase_pred, _  = predecir_clase(X_test[idx])
    label_real     = id2label[y_test[idx]]

    fig = plot_attention_heatmap(
        rollout_scores, f"— Ejemplo {idx+1} (Pred: {clase_pred})")
    plt.savefig(os.path.join(FIG_XAI_DIR, f"attention_rollout_{idx+1}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ejemplo {idx+1} guardado | Real: {label_real} | Pred: {clase_pred}")

    attn_resultados.append({
        'idx': idx, 'texto': X_test[idx],
        'real': label_real, 'pred': clase_pred,
        'scores': rollout_scores
    })

attn_path = os.path.join(SAVE_DIR, "attention_rollout.json")
with open(attn_path, "w", encoding="utf-8") as f:
    json.dump(attn_resultados, f, ensure_ascii=False, indent=2)
print(f"Attention rollout guardado en: {attn_path}")

# ============================================================
# CUANTIFICACIÓN DE EXPLICABILIDAD
# Métricas: Faithfulness, Comprehensiveness, Complexity, Stability
# FIX KeyError: get_importancias_seguro() en todos los calls de LIME
# ============================================================
print("\n" + "="*60)
print("CUANTIFICACIÓN DE EXPLICABILIDAD")
print("="*60)

def calcular_faithfulness(texto, importancias_lime, top_k=5):
    clase_pred, probs_orig = predecir_clase(texto)
    conf_orig  = probs_orig.max()
    clase_idx  = int(np.argmax(probs_orig))

    palabras_pos = [w for w, s in importancias_lime if s > 0][:top_k]
    if not palabras_pos:
        return 0.0, float(conf_orig)

    texto_ocluido = texto
    for palabra in palabras_pos:
        texto_ocluido = re.sub(
            r'\b' + re.escape(palabra) + r'\b', '', texto_ocluido)
    texto_ocluido = re.sub(r'\s+', ' ', texto_ocluido).strip()

    probs_ocluido = predecir_proba([texto_ocluido])[0]
    conf_ocluida  = probs_ocluido[clase_idx]
    return max(0.0, min(1.0, float(conf_orig - conf_ocluida))), float(conf_orig)

def calcular_complexity(importancias):
    if not importancias:
        return float('nan')
    scores = np.abs([s for _, s in importancias])
    if scores.sum() == 0:
        return float('nan')
    probs       = scores / scores.sum()
    entropy     = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0   # 1.0 = importancias uniformes (explicación difusa), ~0 = una sola palabra domina

def calcular_stability(texto, importancias_orig, n_perturbaciones=5, noise_level=0.1):
    correlaciones   = []
    palabras_orig   = [w for w, _ in importancias_orig]
    scores_orig_map = dict(importancias_orig)

    for _ in range(n_perturbaciones):
        palabras = texto.split()
        n_drop   = max(1, int(len(palabras) * noise_level))
        idx_drop = np.random.choice(len(palabras), n_drop, replace=False)
        texto_pert = ' '.join([p for i, p in enumerate(palabras)
                               if i not in idx_drop])
        try:
            exp_pert = lime_explainer.explain_instance(
                texto_pert, predecir_proba,
                num_features=len(importancias_orig),
                num_samples=200,
                top_labels=NUM_LABELS   # FIX: todas las clases disponibles
            )
            clase_idx_pert = int(np.argmax(predecir_proba([texto_pert])[0]))
            # FIX: usar get_importancias_seguro también en stability
            imp_pert, _ = get_importancias_seguro(exp_pert, clase_idx_pert)
            importancias_pert = dict(imp_pert)

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

# --- Calcular métricas ---
N_EVAL = min(30, len(X_test))   # 30 muestras: balance entre representatividad estadística y costo de LIME
print(f"Calculando métricas en {N_EVAL} muestras...\n")

resultados_xai = []
for idx in range(N_EVAL):
    texto      = X_test[idx]
    real       = id2label[y_test[idx]]
    clase_pred, probs = predecir_clase(texto)
    clase_idx  = int(np.argmax(probs))

    exp = lime_explainer.explain_instance(
        texto, predecir_proba,
        num_features=15,
        num_samples=300,
        top_labels=NUM_LABELS    # FIX: todas las clases, elimina KeyError definitivamente
    )
    # FIX KeyError: usar fallback seguro
    importancias, _ = get_importancias_seguro(exp, clase_idx)

    faith = calcular_faithfulness(texto, importancias)[0]
    compl = calcular_complexity(importancias)
    stab  = calcular_stability(texto, importancias)

    resultados_xai.append({
        'idx':          idx,
        'texto':        texto[:80],
        'real':         real,
        'pred':         clase_pred,
        'correcto':     real == clase_pred,
        'faithfulness': faith,
        'complexity':   compl,
        'stability':    stab,
    })

    if idx % 5 == 0:
        print(f"  [{idx+1}/{N_EVAL}] faith={faith:.3f} | compl={compl:.3f} | stab={stab:.3f}")

df_xai = pd.DataFrame(resultados_xai)
print("\nMétricas calculadas")

resumen = df_xai[['faithfulness', 'complexity', 'stability']].describe()
print("\nResumen de métricas XAI:")
print(resumen)

df_xai.to_csv(os.path.join(SAVE_DIR, "xai_metricas_por_muestra.csv"), index=False)
resumen.to_csv(os.path.join(SAVE_DIR, "xai_metricas_resumen.csv"))
print(f"Guardado en {SAVE_DIR}/")

# ============================================================
# VISUALIZACIONES DE MÉTRICAS XAI
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metricas = ['faithfulness', 'complexity', 'stability']
colores  = ['#2196F3', '#FF9800', '#9C27B0']
titulos  = ['Faithfulness\n(↑ mejor)', 'Complexity\n(↓ mejor)', 'Stability\n(↑ mejor)']

for i, (met, col, tit) in enumerate(zip(metricas, colores, titulos)):
    correct_vals   = df_xai[df_xai['correcto'] == True][met].dropna()
    incorrect_vals = df_xai[df_xai['correcto'] == False][met].dropna()
    axes[i].hist(correct_vals,   bins=10, alpha=0.7, color=col,
                 label='Correcto',   edgecolor='black')
    axes[i].hist(incorrect_vals, bins=10, alpha=0.7, color='salmon',
                 label='Incorrecto', edgecolor='black')
    axes[i].set_title(tit, fontsize=12, fontweight='bold')
    axes[i].set_xlabel(met); axes[i].set_ylabel("Frecuencia")
    axes[i].legend(fontsize=9)
    if len(correct_vals) > 0:
        axes[i].axvline(correct_vals.mean(), color=col,
                        linestyle='--', linewidth=2)

plt.suptitle("Métricas XAI: Predicciones Correctas vs Incorrectas",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_XAI_DIR, "xai_metricas_comparacion.png"), dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
for pais in paises_unicos:
    subset = df_xai[df_xai['real'] == pais]
    if len(subset) > 0:
        plt.scatter(subset['faithfulness'], subset['stability'],
                    label=pais, s=80, alpha=0.7)
plt.xlabel("Faithfulness", fontsize=12)
plt.ylabel("Stability", fontsize=12)
plt.title("Faithfulness vs Stability por País", fontsize=13)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_XAI_DIR, "xai_scatter_metricas.png"), dpi=150, bbox_inches='tight')
plt.close()

metricas_por_pais = df_xai.groupby('real')[
    ['faithfulness', 'complexity', 'stability']
].mean()
plt.figure(figsize=(10, max(5, len(paises_unicos)*0.4)))
sns.heatmap(metricas_por_pais, annot=True, fmt='.3f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': 'Valor métrica'})
plt.title("Métricas XAI promedio por país", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_XAI_DIR, "xai_heatmap_por_pais.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Gráficas de métricas guardadas")

# ============================================================
# ANÁLISIS GLOBAL: Palabras clave por país (LIME agregado)
# ============================================================
print("\n" + "="*60)
print("ANÁLISIS GLOBAL: Palabras más importantes por país")
print("="*60)

palabras_por_pais = {pais: Counter() for pais in paises_unicos}
N_POR_PAIS = 100

for pais in paises_unicos:
    indices_pais = [i for i, y in enumerate(y_test) if id2label[y] == pais][:N_POR_PAIS]
    for idx in indices_pais:
        texto     = Xeda_test[idx]
        clase_idx = label2id[pais]
        try:
            exp = lime_explainer.explain_instance(
                texto, predecir_proba,
                num_features=10,
                num_samples=200,
                top_labels=NUM_LABELS   # FIX: todas las clases disponibles
            )
            # FIX: también aquí usar fallback seguro
            imp, _ = get_importancias_seguro(exp, clase_idx)
            for word, score in imp:
                if score > 0:
                    palabras_por_pais[pais][word] += score
        except Exception:
            pass
    print(f"  {pais}: {len(indices_pais)} muestras procesadas")

palabras_path = os.path.join(SAVE_DIR, "palabras_clave_por_pais.json")
with open(palabras_path, "w", encoding="utf-8") as f:
    json.dump(
        {p: palabras_por_pais[p].most_common(20) for p in paises_unicos},
        f, ensure_ascii=False, indent=2
    )
print(f"Palabras clave guardadas en: {palabras_path}")

n_cols   = min(3, len(paises_unicos))
n_rows   = (len(paises_unicos) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
axes_flat = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

for i, pais in enumerate(paises_unicos):
    if i >= len(axes_flat):
        break
    top_words = palabras_por_pais[pais].most_common(10)
    if top_words:
        words, scores = zip(*top_words)
        axes_flat[i].barh(list(words), list(scores),
                          color='steelblue', edgecolor='black')
        axes_flat[i].set_title(f"Top palabras — {pais}",
                               fontsize=11, fontweight='bold')
        axes_flat[i].invert_yaxis()

for j in range(len(paises_unicos), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Palabras más discriminativas por país (LIME agregado)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_XAI_DIR, "xai_palabras_por_pais.png"), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# ANÁLISIS POR CONFIANZA: alta vs baja probabilidad por clase
# Para cada país: N ejemplos con mayor probabilidad predicha
# y N con menor. Se comparan las métricas XAI entre grupos.
# ============================================================
print("\n" + "="*60)
print("ANÁLISIS POR CONFIANZA: Alta vs Baja probabilidad por clase")
print("="*60)

y_proba_test = np.load(os.path.join(SAVE_DIR, "y_proba_test.npy"))
N_POR_CLASE  = 10  # ejemplos por grupo (alta / baja) por país

resultados_confianza = []

for pais in paises_unicos:
    clase_idx     = label2id[pais]
    indices_pais  = np.where(y_test == clase_idx)[0]

    if len(indices_pais) < 2 * N_POR_CLASE:
        print(f"  {pais}: pocas muestras ({len(indices_pais)}), omitido")
        continue

    probs_clase      = y_proba_test[indices_pais, clase_idx]
    orden            = np.argsort(probs_clase)[::-1]
    indices_ordenados = indices_pais[orden]

    grupos = {
        "alta": indices_ordenados[:N_POR_CLASE],
        "baja": indices_ordenados[-N_POR_CLASE:]
    }

    for grupo, idxs in grupos.items():
        for test_idx in idxs:
            texto     = X_test[test_idx]
            prob_pred = float(y_proba_test[test_idx, clase_idx])

            try:
                exp = lime_explainer.explain_instance(
                    texto, predecir_proba,
                    num_features=15, num_samples=200,
                    top_labels=NUM_LABELS
                )
                importancias, _ = get_importancias_seguro(exp, clase_idx)

                faith = calcular_faithfulness(texto, importancias)[0]
                compl = calcular_complexity(importancias)
                stab  = calcular_stability(texto, importancias, n_perturbaciones=3)
            except Exception as e:
                print(f"    Error en {pais}/{grupo}/{test_idx}: {e}")
                faith, compl, stab = float('nan'), float('nan'), float('nan')

            resultados_confianza.append({
                'pais':         pais,
                'grupo':        grupo,
                'test_idx':     int(test_idx),
                'prob_clase':   prob_pred,
                'faithfulness': faith,
                'complexity':   compl,
                'stability':    stab,
            })

    print(f"  {pais}: alta ({N_POR_CLASE}) y baja ({N_POR_CLASE}) procesados")

df_conf = pd.DataFrame(resultados_confianza)
df_conf.to_csv(os.path.join(SAVE_DIR, "xai_confianza_por_clase.csv"), index=False)
print(f"\nResultados guardados en xai_confianza_por_clase.csv")

# Resumen numérico alta vs baja
resumen_conf = df_conf.groupby('grupo')[['faithfulness', 'complexity', 'stability']].mean()
print("\nPromedio por grupo:")
print(resumen_conf.to_string())

# Heatmap: métricas promedio por país y grupo
for metrica in ['faithfulness', 'complexity', 'stability']:
    pivot = df_conf.pivot_table(
        index='pais', columns='grupo', values=metrica, aggfunc='mean'
    )
    # Reordenar columnas: alta primero
    cols = [c for c in ['alta', 'baja'] if c in pivot.columns]
    pivot = pivot[cols]

    plt.figure(figsize=(5, max(4, len(pivot) * 0.45)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu',
                linewidths=0.5, cbar_kws={'label': metrica})
    plt.title(f"{metrica.capitalize()} por país: Alta vs Baja confianza",
              fontsize=12, fontweight='bold')
    plt.xlabel("Grupo de confianza")
    plt.ylabel("País")
    plt.tight_layout()
    fname = os.path.join(FIG_XAI_DIR, f"xai_confianza_{metrica}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {fname}")

# Barras comparativas alta vs baja (promedio global)
metricas_conf = ['faithfulness', 'complexity', 'stability']
fig, axes = plt.subplots(1, len(metricas_conf), figsize=(14, 5))

for ax, met in zip(axes, metricas_conf):
    data_alta = df_conf[df_conf['grupo'] == 'alta'][met].dropna()
    data_baja = df_conf[df_conf['grupo'] == 'baja'][met].dropna()

    medias = [data_alta.mean(), data_baja.mean()]
    stds   = [data_alta.std(),  data_baja.std()]
    colores_bar = ['#2196F3', '#FF7043']

    bars = ax.bar(['Alta confianza', 'Baja confianza'], medias, yerr=stds,
                  color=colores_bar, alpha=0.85, edgecolor='black', capsize=5)
    for bar, val in zip(bars, medias):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_title(met.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel("Valor promedio")
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("Métricas XAI: Alta vs Baja confianza de predicción",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_XAI_DIR, "xai_confianza_barras.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: xai_confianza_barras.png")

# ============================================================
# HEATMAP ROBUSTO: 30 muestras aleatorias por país
# ============================================================
print("\n" + "="*60)
print("HEATMAP ROBUSTO — 30 muestras aleatorias por país")
print("="*60)

ROBUSTO_DIR        = FIG_XAI_ROBUSTO
N_POR_PAIS_ROBUSTO = 30

resultados_robusto = []

for pais in paises_unicos:
    clase_idx_pais = label2id[pais]
    indices_pais   = np.where(y_test == clase_idx_pais)[0]

    n_disponibles = len(indices_pais)
    n_muestras    = min(N_POR_PAIS_ROBUSTO, n_disponibles)

    if n_muestras == 0:
        print(f"  {pais}: sin muestras, omitido")
        continue

    indices_sel = np.random.choice(indices_pais, n_muestras, replace=False)
    print(f"  {pais}: {n_muestras}/{n_disponibles} muestras seleccionadas")

    for idx in indices_sel:
        texto      = X_test[idx]
        clase_pred, probs = predecir_clase(texto)
        clase_idx  = int(np.argmax(probs))

        exp = lime_explainer.explain_instance(
            texto, predecir_proba,
            num_features=15,
            num_samples=300,
            top_labels=NUM_LABELS
        )
        importancias, _ = get_importancias_seguro(exp, clase_idx)

        faith = calcular_faithfulness(texto, importancias)[0]
        compl = calcular_complexity(importancias)
        stab  = calcular_stability(texto, importancias)

        resultados_robusto.append({
            'idx':          int(idx),
            'pais':         pais,
            'pred':         clase_pred,
            'correcto':     pais == clase_pred,
            'faithfulness': faith,
            'complexity':   compl,
            'stability':    stab,
        })

df_robusto = pd.DataFrame(resultados_robusto)
df_robusto.to_csv(os.path.join(ROBUSTO_DIR, "xai_metricas_robusto.csv"), index=False)

metricas_por_pais_robusto = df_robusto.groupby('pais')[
    ['faithfulness', 'complexity', 'stability']
].mean()

plt.figure(figsize=(10, max(5, len(paises_unicos) * 0.4)))
sns.heatmap(metricas_por_pais_robusto, annot=True, fmt='.3f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': 'Valor métrica'})
plt.title(f"Métricas XAI promedio por país\n({N_POR_PAIS_ROBUSTO} muestras aleatorias por país)",
          fontsize=13, fontweight='bold')
plt.tight_layout()
heatmap_path = os.path.join(ROBUSTO_DIR, "xai_heatmap_por_pais_robusto.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nHeatmap robusto guardado en: {heatmap_path}")

resumen_robusto = df_robusto[['faithfulness', 'complexity', 'stability']].describe()
print("\nResumen métricas (robusto):")
print(resumen_robusto.to_string())

# ============================================================
# ANÁLISIS POR MUESTRA: 5 muestras aleatorias por país
# LIME + SHAP + IG + Attention Rollout + Métricas XAI
# Salida: ./xai_muestras_por_pais/{pais}/muestra_N_*.png
# ============================================================
print("\n" + "="*60)
print("ANÁLISIS POR MUESTRA — 5 muestras aleatorias por país")
print("="*60)

MUESTRAS_DIR    = FIG_XAI_PAISES
N_MUESTRAS_PAIS = 5

resultados_muestras = []

for pais in paises_unicos:
    clase_idx_pais = label2id[pais]
    indices_pais   = np.where(y_test == clase_idx_pais)[0]

    n_disponibles = len(indices_pais)
    n_sel         = min(N_MUESTRAS_PAIS, n_disponibles)

    if n_sel == 0:
        print(f"  {pais}: sin muestras, omitido")
        continue

    # Carpeta por país (espacios → guión bajo para el sistema de archivos)
    pais_dir = os.path.join(MUESTRAS_DIR, pais.replace(" ", "_"))
    os.makedirs(pais_dir, exist_ok=True)

    indices_sel = np.random.choice(indices_pais, n_sel, replace=False)
    print(f"\n{'─'*50}")
    print(f"País: {pais}  ({n_sel}/{n_disponibles} muestras)")

    for muestra_num, idx in enumerate(indices_sel, 1):
        texto      = X_test[idx]
        label_real = id2label[y_test[idx]]
        clase_pred, probs = predecir_clase(texto)
        clase_idx  = int(np.argmax(probs))
        prefijo    = os.path.join(pais_dir, f"muestra_{muestra_num}")

        print(f"  [{muestra_num}] '{texto[:70]}...'")
        print(f"       Real: {label_real} | Pred: {clase_pred}")

        # ── LIME ──────────────────────────────────────────────
        exp_lime = explicar_con_lime(texto)
        importancias, label_usado = get_importancias_seguro(exp_lime, clase_idx)
        fig = exp_lime.as_pyplot_figure(label=label_usado)
        plt.title(f"LIME — {pais} | Muestra {muestra_num} (Pred: {id2label[label_usado]})",
                  fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{prefijo}_lime.png", dpi=150, bbox_inches='tight')
        plt.close()

        # ── SHAP ──────────────────────────────────────────────
        shap_vals_muestra = shap_explainer([texto])
        shap_ejemplo      = shap_vals_muestra[0]
        tokens_shap       = shap_ejemplo.data
        values_shap       = shap_ejemplo.values          # (n_palabras, n_clases)

        scores_shap = values_shap[:, clase_idx]
        top_idx     = np.argsort(np.abs(scores_shap))[-15:][::-1]
        top_tokens  = [tokens_shap[i] for i in top_idx]
        top_scores  = [scores_shap[i] for i in top_idx]
        colores_shap = ['#d73027' if s > 0 else '#4575b4' for s in top_scores]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(top_tokens)), top_scores,
                color=colores_shap, edgecolor='black')
        ax.set_yticks(range(len(top_tokens)))
        ax.set_yticklabels(top_tokens, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"SHAP — {pais} | Muestra {muestra_num} | Clase: {clase_pred}",
                     fontsize=11, fontweight='bold')
        rojo = mpatches.Patch(color='#d73027', label='Favorece esta clase')
        azul = mpatches.Patch(color='#4575b4', label='En contra de esta clase')
        ax.legend(handles=[rojo, azul])
        plt.tight_layout()
        plt.savefig(f"{prefijo}_shap.png", dpi=150, bbox_inches='tight')
        plt.close()

        # ── Integrated Gradients ──────────────────────────────
        token_attrs, clase_ig, delta = explicar_ig(texto, clase_idx)
        calidad = "OK" if abs(delta) < 0.05 else \
                  "aceptable" if abs(delta) < 0.5 else "mejorable"
        print(f"       IG delta={delta:.4f} [{calidad}]")
        if token_attrs:
            fig = plot_ig(token_attrs, clase_ig,
                          f"— {pais} | Muestra {muestra_num} (Real: {label_real})")
            plt.savefig(f"{prefijo}_ig.png", dpi=150, bbox_inches='tight')
            plt.close()

        # ── Attention Rollout ─────────────────────────────────
        rollout_scores = get_attention_rollout(texto)
        fig = plot_attention_heatmap(
            rollout_scores,
            f"— {pais} | Muestra {muestra_num} (Pred: {clase_pred})"
        )
        plt.savefig(f"{prefijo}_attention.png", dpi=150, bbox_inches='tight')
        plt.close()

        # ── Métricas XAI ──────────────────────────────────────
        faith = calcular_faithfulness(texto, importancias)[0]
        compl = calcular_complexity(importancias)
        stab  = calcular_stability(texto, importancias)
        print(f"       faith={faith:.3f} | compl={compl:.3f} | stab={stab:.3f}")

        resultados_muestras.append({
            'pais':         pais,
            'muestra':      muestra_num,
            'idx':          int(idx),
            'texto':        texto[:100],
            'real':         label_real,
            'pred':         clase_pred,
            'correcto':     label_real == clase_pred,
            'delta_ig':     float(delta),
            'faithfulness': faith,
            'complexity':   compl,
            'stability':    stab,
        })

# Guardar CSV con todas las métricas
df_muestras = pd.DataFrame(resultados_muestras)
df_muestras.to_csv(os.path.join(MUESTRAS_DIR, "xai_metricas_muestras.csv"), index=False)

# Heatmap de métricas promedio por país (sobre estas 5 muestras)
metricas_muestras_pais = df_muestras.groupby('pais')[
    ['faithfulness', 'complexity', 'stability']
].mean()

plt.figure(figsize=(10, max(5, len(paises_unicos) * 0.4)))
sns.heatmap(metricas_muestras_pais, annot=True, fmt='.3f', cmap='YlGnBu',
            linewidths=0.5, cbar_kws={'label': 'Valor métrica'})
plt.title(f"Métricas XAI por país\n({N_MUESTRAS_PAIS} muestras aleatorias por país)",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MUESTRAS_DIR, "xai_heatmap_muestras_pais.png"),
            dpi=150, bbox_inches='tight')
plt.close()

print(f"\nTodo guardado en: {MUESTRAS_DIR}/")
print(f"  Subcarpetas: una por país, 4 gráficas por muestra (lime/shap/ig/attention)")
print(f"  xai_metricas_muestras.csv — métricas por muestra")
print(f"  xai_heatmap_muestras_pais.png — heatmap resumen")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*60)
print("XAI COMPLETO FINALIZADO")
print("="*60)
print(f"""
Métricas XAI (3, sin redundancia):
  - Faithfulness  (↑ mejor): caída de confianza al ocultar palabras clave
  - Complexity    (↓ mejor): entropía de la distribución de importancias
  - Stability     (↑ mejor): correlación Spearman bajo perturbaciones

Archivos generados en {SAVE_DIR}/:
  lime_explicaciones.json | shap_values.pkl
  ig_atribuciones.json    | attention_rollout.json
  xai_metricas_por_muestra.csv | xai_metricas_resumen.csv
  palabras_clave_por_pais.json
  xai_confianza_por_clase.csv

Gráficas:
  lime_ejemplo_*.png
  shap_global_*.png / shap_local_*.png
  ig_ejemplo_*.png
  attention_rollout_*.png
  xai_metricas_comparacion.png   (3 métricas: correctas vs incorrectas)
  xai_scatter_metricas.png       (faithfulness vs stability por país)
  xai_heatmap_por_pais.png       (3 métricas promedio por país)
  xai_palabras_por_pais.png      (top palabras discriminativas por país)
  xai_confianza_faithfulness.png (heatmap alta vs baja confianza)
  xai_confianza_complexity.png
  xai_confianza_stability.png
  xai_confianza_barras.png       (comparación global alta vs baja)
""")
