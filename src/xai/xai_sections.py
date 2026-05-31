# ============================================================
# XAI NUEVAS SECCIONES — Solo heatmap robusto y análisis
# por muestra (5 x país, 4 métodos + métricas XAI)
# Requiere que model_outputs/xlmt_lora_v2/ exista con el modelo
# y los splits ya guardados por xai_methods.py
# Autor: Jorge Antonio Loera Grande
# ============================================================

import os, re, json, time
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
from captum.attr import LayerIntegratedGradients
import shap

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ============================================================
# RUTAS
# ============================================================
SAVE_DIR   = "model_outputs/xlmt_lora_v2"
MERGED_DIR = os.path.join(SAVE_DIR, "modelo_merged")
META_PATH  = os.path.join(SAVE_DIR, "metadata.json")

for ruta in [MERGED_DIR, META_PATH,
             os.path.join(SAVE_DIR, "X_test.npy"),
             os.path.join(SAVE_DIR, "y_test.npy")]:
    assert os.path.exists(ruta), f"No encontrado: {ruta}"
print("Archivos requeridos encontrados")

# ============================================================
# METADATA, MODELO Y DATOS
# ============================================================
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

id2label      = {int(k): v for k, v in metadata["id2label"].items()}
label2id      = metadata["label2id"]
paises_unicos = sorted(label2id.keys())
NUM_LABELS    = metadata["num_labels"]
MAX_LENGTH    = metadata.get("max_length", 128)
print(f"Metadata: {NUM_LABELS} clases, max_length={MAX_LENGTH}")

print("\nCargando tokenizer y modelo merged...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
xai_model = AutoModelForSequenceClassification.from_pretrained(
    MERGED_DIR,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    attn_implementation="eager"   # SDPA no soporta output_attentions=True; eager sí, necesario para Rollout
)
xai_model.to(device)
xai_model.eval()
print(f"Modelo cargado desde: {MERGED_DIR}")

X_test = np.load(os.path.join(SAVE_DIR, "X_test.npy"), allow_pickle=True)
y_test = np.load(os.path.join(SAVE_DIR, "y_test.npy"))
print(f"Test set: {len(X_test)} muestras")

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
clase, _ = predecir_clase(X_test[0])
print(f"Prueba rápida — Pred: {clase} | Real: {id2label[y_test[0]]}")

# ============================================================
# LIME
# ============================================================
lime_explainer = LimeTextExplainer(
    class_names=paises_unicos,
    split_expression=r'\W+',
    random_state=SEED
)

def explicar_con_lime(texto, num_features=15, num_samples=500):
    return lime_explainer.explain_instance(
        texto, predecir_proba,
        num_features=num_features,
        num_samples=num_samples,
        top_labels=NUM_LABELS
    )

def get_importancias_seguro(exp, clase_idx):
    disponibles = exp.available_labels()
    if clase_idx in disponibles:
        return exp.as_list(label=clase_idx), clase_idx
    fallback = disponibles[0]
    print(f"    [LIME fallback] label {clase_idx} → {fallback} ({id2label[fallback]})")
    return exp.as_list(label=fallback), fallback

# ============================================================
# SHAP
# ============================================================
shap_explainer = shap.Explainer(
    predecir_proba,
    masker=shap.maskers.Text(tokenizer=r'\W+'),
    output_names=paises_unicos
)

# ============================================================
# INTEGRATED GRADIENTS
# ============================================================
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
            label_idx = int(forward_for_captum(input_ids, attention_mask).argmax().item())
    label_idx = int(label_idx)

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    lig = LayerIntegratedGradients(
        forward_for_captum,
        xai_model.roberta.embeddings.word_embeddings
    )
    attributions, delta = lig.attribute(
        input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=label_idx,
        return_convergence_delta=True,
        n_steps=300,
        internal_batch_size=8
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

# ============================================================
# ATTENTION ROLLOUT
# ============================================================
TOKENS_FILTRO = {'<pad>', '<s>', '</s>', '▁', 'Ġ', '<unk>'}

def get_attention_rollout(texto):
    encoding = tokenizer(
        texto, max_length=MAX_LENGTH, padding='max_length',
        truncation=True, return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = xai_model.roberta(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            output_attentions=True
        )

    attn_matrices = [a.squeeze(0).mean(0).cpu().numpy() for a in outputs.attentions]
    rollout = np.eye(attn_matrices[0].shape[0])
    for attn in attn_matrices:
        attn_res = attn + np.eye(attn.shape[0])
        attn_res = attn_res / (attn_res.sum(axis=-1, keepdims=True) + 1e-10)
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

# ============================================================
# MÉTRICAS XAI
# ============================================================
def calcular_faithfulness(texto, importancias_lime, top_k=5):
    clase_pred, probs_orig = predecir_clase(texto)
    conf_orig = probs_orig.max()
    clase_idx = int(np.argmax(probs_orig))

    palabras_pos = [w for w, s in importancias_lime if s > 0][:top_k]
    if not palabras_pos:
        return 0.0, float(conf_orig)

    texto_ocluido = texto
    for palabra in palabras_pos:
        texto_ocluido = re.sub(r'\b' + re.escape(palabra) + r'\b', '', texto_ocluido)
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
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0

def calcular_stability(texto, importancias_orig, n_perturbaciones=5, noise_level=0.1):
    correlaciones   = []
    palabras_orig   = [w for w, _ in importancias_orig]
    scores_orig_map = dict(importancias_orig)

    for _ in range(n_perturbaciones):
        palabras = texto.split()
        n_drop   = max(1, int(len(palabras) * noise_level))
        idx_drop = np.random.choice(len(palabras), n_drop, replace=False)
        texto_pert = ' '.join([p for i, p in enumerate(palabras) if i not in idx_drop])
        try:
            exp_pert = lime_explainer.explain_instance(
                texto_pert, predecir_proba,
                num_features=len(importancias_orig),
                num_samples=200,
                top_labels=NUM_LABELS
            )
            clase_idx_pert = int(np.argmax(predecir_proba([texto_pert])[0]))
            imp_pert, _    = get_importancias_seguro(exp_pert, clase_idx_pert)
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

# ============================================================
# SECCIÓN 1: HEATMAP ROBUSTO — 30 muestras aleatorias por país
# ============================================================
print("\n" + "="*60)
print("HEATMAP ROBUSTO — 30 muestras aleatorias por país")
print("="*60)

ROBUSTO_DIR        = "results/figures/xai/robusto"
N_POR_PAIS_ROBUSTO = 30   # 30 muestras aleatorias: suficiente para estimar la distribución de métricas por país
os.makedirs(ROBUSTO_DIR, exist_ok=True)

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
# SECCIÓN 2: ANÁLISIS POR MUESTRA — 5 x país, 4 métodos
# Salida: ./xai_muestras_por_pais/{pais}/muestra_N_*.png
# ============================================================
print("\n" + "="*60)
print("ANÁLISIS POR MUESTRA — 5 muestras aleatorias por país")
print("="*60)

MUESTRAS_DIR    = "results/figures/xai/por_pais"
N_MUESTRAS_PAIS = 5   # 5 muestras por país: suficiente para inspección cualitativa sin saturar el análisis
os.makedirs(MUESTRAS_DIR, exist_ok=True)

resultados_muestras = []

for pais in paises_unicos:
    clase_idx_pais = label2id[pais]
    indices_pais   = np.where(y_test == clase_idx_pais)[0]

    n_disponibles = len(indices_pais)
    n_sel         = min(N_MUESTRAS_PAIS, n_disponibles)

    if n_sel == 0:
        print(f"  {pais}: sin muestras, omitido")
        continue

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

        scores_shap  = values_shap[:, clase_idx]
        top_idx      = np.argsort(np.abs(scores_shap))[-15:][::-1]
        top_tokens   = [tokens_shap[i] for i in top_idx]
        top_scores   = [scores_shap[i] for i in top_idx]
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

# Guardar CSV y heatmap resumen
df_muestras = pd.DataFrame(resultados_muestras)
df_muestras.to_csv(os.path.join(MUESTRAS_DIR, "xai_metricas_muestras.csv"), index=False)

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

print(f"\n{'='*60}")
print("FINALIZADO")
print(f"{'='*60}")
print(f"""
Carpeta 1 — {ROBUSTO_DIR}/
  xai_metricas_robusto.csv
  xai_heatmap_por_pais_robusto.png

Carpeta 2 — {MUESTRAS_DIR}/
  {{pais}}/muestra_N_lime.png
  {{pais}}/muestra_N_shap.png
  {{pais}}/muestra_N_ig.png
  {{pais}}/muestra_N_attention.png
  xai_metricas_muestras.csv
  xai_heatmap_muestras_pais.png
""")
