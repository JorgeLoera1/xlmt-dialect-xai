# ============================================================
# XAI CONFIANZA EXPLICACIONES
# Para cada país: 10 tweets con mayor probabilidad predicha
# y 10 con menor. Aplica los 4 métodos XAI a cada uno:
#   LIME / SHAP / Integrated Gradients / Attention Rollout
# Salida: ./xai_confianza_explicaciones/{pais}/{alta|baja}/
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
OUT_DIR    = "results/figures/xai"
N_POR_CLASE = 10   # 10 tweets por grupo (alta/baja confianza) por país: costo total O(21×2×10×4 métodos)

for ruta in [MERGED_DIR, META_PATH,
             os.path.join(SAVE_DIR, "X_test.npy"),
             os.path.join(SAVE_DIR, "y_test.npy"),
             os.path.join(SAVE_DIR, "y_proba_test.npy")]:
    assert os.path.exists(ruta), f"No encontrado: {ruta}"
print("Archivos requeridos encontrados\n")

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

print("Cargando tokenizer y modelo merged...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MERGED_DIR,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    attn_implementation="eager"   # SDPA no expone output_attentions; eager sí, necesario para Rollout
)
model.to(device)
model.eval()
print(f"Modelo cargado: {MERGED_DIR}\n")

X_test       = np.load(os.path.join(SAVE_DIR, "X_test.npy"),       allow_pickle=True)
y_test       = np.load(os.path.join(SAVE_DIR, "y_test.npy"))
y_proba_test = np.load(os.path.join(SAVE_DIR, "y_proba_test.npy"))
print(f"Test set: {len(X_test)} muestras\n")

# ============================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================
def predecir_proba(textos):
    if isinstance(textos, str):
        textos = [textos]
    enc = tokenizer(
        list(textos), max_length=MAX_LENGTH,
        padding='max_length', truncation=True, return_tensors='pt'
    ).to(device)
    enc.pop("token_type_ids", None)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()

def predecir_clase(texto):
    probs = predecir_proba([texto])[0]
    return id2label[np.argmax(probs)], probs

# ============================================================
# LIME
# ============================================================
lime_explainer = LimeTextExplainer(
    class_names=paises_unicos,
    split_expression=r'\W+',
    random_state=SEED
)

def explicar_lime(texto, num_features=15, num_samples=500):
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

def guardar_lime(exp, label_usado, titulo, ruta):
    fig = exp.as_pyplot_figure(label=label_usado)
    plt.title(titulo, fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# SHAP
# ============================================================
print("Inicializando SHAP explainer...")
shap_explainer = shap.Explainer(
    predecir_proba,
    masker=shap.maskers.Text(tokenizer=r'\W+'),
    output_names=paises_unicos
)

def guardar_shap(texto, clase_idx, titulo, ruta):
    shap_vals = shap_explainer([texto])[0]
    tokens    = shap_vals.data
    scores    = shap_vals.values[:, clase_idx]

    top_idx    = np.argsort(np.abs(scores))[-15:][::-1]
    top_tokens = [tokens[i] for i in top_idx]
    top_scores = [scores[i] for i in top_idx]
    colores    = ['#d73027' if s > 0 else '#4575b4' for s in top_scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(top_tokens)), top_scores, color=colores, edgecolor='black')
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels(top_tokens, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    rojo = mpatches.Patch(color='#d73027', label='Favorece la clase')
    azul = mpatches.Patch(color='#4575b4', label='En contra de la clase')
    ax.legend(handles=[rojo, azul])
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# INTEGRATED GRADIENTS
# ============================================================
def forward_captum(input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

def explicar_ig(texto, label_idx):
    enc = tokenizer(
        texto, max_length=MAX_LENGTH, padding='max_length',
        truncation=True, return_tensors='pt'
    ).to(device)
    enc.pop("token_type_ids", None)
    input_ids      = enc['input_ids']
    attention_mask = enc['attention_mask']
    label_idx      = int(label_idx)   # Captum rechaza numpy.int64; necesita int nativo de Python
    baseline_ids   = torch.full_like(input_ids, tokenizer.pad_token_id)   # baseline neutro: pad ≠ <unk> (token 0), evita atribución espuria

    lig = LayerIntegratedGradients(
        forward_captum, model.roberta.embeddings.word_embeddings
    )
    attributions, delta = lig.attribute(
        input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=label_idx,
        return_convergence_delta=True,
        n_steps=300,           # más pasos → delta más pequeño (mejor aproximación de Riemann)
        internal_batch_size=8  # procesa los 300 pasos en mini-lotes de 8 para evitar OOM en 6GB VRAM
    )
    attr_sum = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()
    tokens   = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())

    especiales = {'<pad>', '<s>', '</s>', '▁', 'Ġ', '<unk>'}
    valid = [(t.lstrip('▁').lstrip('Ġ'), float(a))
             for t, a in zip(tokens, attr_sum)
             if t not in especiales and t.strip() != '']
    return valid, delta.item()

def guardar_ig(token_attrs, clase_pred, titulo, ruta):
    if not token_attrs:
        return
    tokens, attrs = zip(*token_attrs[:20])
    attrs      = np.array(attrs)
    attrs_norm = (attrs - attrs.min()) / (attrs.max() - attrs.min() + 1e-8)
    colores    = ['#d73027' if a > 0.5 else '#4575b4' for a in attrs_norm]

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.5), 3))
    ax.bar(range(len(tokens)), attrs, color=colores, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Atribución IG")
    ax.set_title(titulo, fontsize=11)
    rojo = mpatches.Patch(color='#d73027', label='Alta importancia')
    azul = mpatches.Patch(color='#4575b4', label='Baja importancia')
    ax.legend(handles=[rojo, azul])
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# ATTENTION ROLLOUT
# ============================================================
TOKENS_FILTRO = {'<pad>', '<s>', '</s>', '▁', 'Ġ', '<unk>'}

def explicar_attention_rollout(texto):
    enc = tokenizer(
        texto, max_length=MAX_LENGTH, padding='max_length',
        truncation=True, return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model.roberta(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
            output_attentions=True
        )
    attn_matrices = [a.squeeze(0).mean(0).cpu().numpy() for a in outputs.attentions]
    rollout = np.eye(attn_matrices[0].shape[0])
    for attn in attn_matrices:
        attn_res = attn + np.eye(attn.shape[0])
        attn_res = attn_res / (attn_res.sum(axis=-1, keepdims=True) + 1e-10)
        rollout  = rollout @ attn_res

    cls_rollout = rollout[0]
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'].squeeze().cpu())
    valid = []
    for t, s in zip(tokens, cls_rollout):
        if t in TOKENS_FILTRO:
            continue
        t_display = t.lstrip('▁').lstrip('Ġ')
        if t_display.strip():
            valid.append((t_display, float(s)))
    return valid

def guardar_attention(token_scores, titulo, ruta):
    if not token_scores:
        return
    tokens, scores = zip(*token_scores[:25])
    scores = np.array(scores).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.55), 2.5))
    im = ax.imshow(scores, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.4, shrink=0.5)
    ax.set_title(titulo, fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# BUCLE PRINCIPAL
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)
resumen_rows = []
t_total = time.time()

for pais in paises_unicos:
    clase_idx    = label2id[pais]
    indices_pais = np.where(y_test == clase_idx)[0]

    if len(indices_pais) < 2 * N_POR_CLASE:
        print(f"\n{pais}: pocas muestras ({len(indices_pais)}), omitido")
        continue

    probs_pais      = y_proba_test[indices_pais, clase_idx]
    orden           = np.argsort(probs_pais)[::-1]
    indices_ordenados = indices_pais[orden]

    grupos = {
        "alta": indices_ordenados[:N_POR_CLASE],
        "baja": indices_ordenados[-N_POR_CLASE:]
    }

    print(f"\n{'='*60}")
    print(f"País: {pais.upper()}  —  alta confianza | baja confianza")
    print(f"{'='*60}")

    for grupo, idxs in grupos.items():
        grupo_dir = os.path.join(OUT_DIR, pais, grupo)
        os.makedirs(grupo_dir, exist_ok=True)

        for num, test_idx in enumerate(idxs, 1):
            texto      = X_test[test_idx]
            label_real = id2label[y_test[test_idx]]
            clase_pred, probs = predecir_clase(texto)
            pred_idx   = int(np.argmax(probs))
            prob_clase = float(y_proba_test[test_idx, clase_idx])
            prefijo    = os.path.join(grupo_dir, f"tweet_{num:02d}")

            print(f"\n  [{grupo}] Tweet {num:02d} | prob={prob_clase:.4f}")
            print(f"    Real: {label_real} | Pred: {clase_pred}")
            print(f"    '{texto[:80]}...'")

            # ── LIME ──────────────────────────────────────────
            try:
                exp_lime = explicar_lime(texto)
                importancias, label_usado = get_importancias_seguro(exp_lime, pred_idx)
                titulo_lime = (f"LIME — {pais.upper()} | {grupo} confianza | "
                               f"Tweet {num} | Pred: {id2label[label_usado]} "
                               f"(prob={prob_clase:.3f})")
                guardar_lime(exp_lime, label_usado, titulo_lime, f"{prefijo}_lime.png")
                print(f"    LIME OK — top: {[w for w, _ in importancias[:3]]}")
            except Exception as e:
                print(f"    LIME ERROR: {e}")
                importancias = []

            # ── SHAP ──────────────────────────────────────────
            try:
                titulo_shap = (f"SHAP — {pais.upper()} | {grupo} confianza | "
                               f"Tweet {num} | Pred: {clase_pred} "
                               f"(prob={prob_clase:.3f})")
                guardar_shap(texto, pred_idx, titulo_shap, f"{prefijo}_shap.png")
                print(f"    SHAP OK")
            except Exception as e:
                print(f"    SHAP ERROR: {e}")

            # ── Integrated Gradients ───────────────────────────
            try:
                token_attrs, delta = explicar_ig(texto, pred_idx)
                calidad = "OK" if abs(delta) < 0.05 else \
                          "aceptable" if abs(delta) < 0.5 else "mejorable"
                titulo_ig = (f"IG — {pais.upper()} | {grupo} confianza | "
                             f"Tweet {num} | Clase: {clase_pred} "
                             f"[delta={delta:.3f} {calidad}]")
                guardar_ig(token_attrs, clase_pred, titulo_ig, f"{prefijo}_ig.png")
                print(f"    IG OK — delta={delta:.4f} [{calidad}]")
            except Exception as e:
                print(f"    IG ERROR: {e}")
                token_attrs, delta = [], float('nan')

            # ── Attention Rollout ──────────────────────────────
            try:
                rollout = explicar_attention_rollout(texto)
                titulo_ar = (f"Attention Rollout — {pais.upper()} | {grupo} | "
                             f"Tweet {num} | Pred: {clase_pred} "
                             f"(prob={prob_clase:.3f})")
                guardar_attention(rollout, titulo_ar, f"{prefijo}_attention.png")
                print(f"    Attention Rollout OK")
            except Exception as e:
                print(f"    Attention Rollout ERROR: {e}")

            resumen_rows.append({
                'pais':       pais,
                'grupo':      grupo,
                'tweet_num':  num,
                'test_idx':   int(test_idx),
                'texto':      texto[:120],
                'real':       label_real,
                'pred':       clase_pred,
                'correcto':   label_real == clase_pred,
                'prob_clase': round(prob_clase, 4),
            })

# ============================================================
# CSV RESUMEN
# ============================================================
df_resumen = pd.DataFrame(resumen_rows)
csv_path = os.path.join(OUT_DIR, "resumen_tweets.csv")
df_resumen.to_csv(csv_path, index=False)

# ============================================================
# HEATMAP: accuracy por país y grupo
# ============================================================
if not df_resumen.empty:
    pivot_acc = df_resumen.pivot_table(
        index='pais', columns='grupo', values='correcto', aggfunc='mean'
    )
    cols = [c for c in ['alta', 'baja'] if c in pivot_acc.columns]
    pivot_acc = pivot_acc[cols]

    plt.figure(figsize=(5, max(5, len(pivot_acc) * 0.45)))
    sns.heatmap(pivot_acc, annot=True, fmt='.2f', cmap='YlGnBu',
                linewidths=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Accuracy'})
    plt.title("Accuracy por país: Alta vs Baja confianza\n(10 tweets por grupo)",
              fontsize=12, fontweight='bold')
    plt.xlabel("Grupo")
    plt.ylabel("País")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "heatmap_accuracy.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# RESUMEN FINAL
# ============================================================
elapsed = time.time() - t_total
print(f"\n{'='*60}")
print("FINALIZADO")
print(f"{'='*60}")
print(f"Tiempo total: {elapsed/60:.1f} minutos")
print(f"""
Carpeta de salida: {OUT_DIR}/

Estructura:
  {{pais}}/
    alta/   tweet_01_lime.png  tweet_01_shap.png
            tweet_01_ig.png    tweet_01_attention.png
            ...                (hasta tweet_10)
    baja/   tweet_01_lime.png  ...

  resumen_tweets.csv      — texto, real, pred, prob por tweet
  heatmap_accuracy.png    — accuracy alta vs baja por país

Total de tweets analizados: {len(df_resumen)}
  ({N_POR_CLASE} alta + {N_POR_CLASE} baja) × {len(paises_unicos)} países
  = {2 * N_POR_CLASE * len(paises_unicos)} gráficas × 4 métodos
  = {2 * N_POR_CLASE * len(paises_unicos) * 4} imágenes en total
""")
