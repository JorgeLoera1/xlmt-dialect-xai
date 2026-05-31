# ============================================================
# XAI POR DIALECTO — Top-50 ejemplos correctos por clase
# Modelo: XLM-T + LoRA (merged)
# Métodos: LIME + SHAP
# Salida: palabras más discriminativas por dialecto (top-20)
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
from collections import defaultdict
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
os.makedirs(OUT_DIR, exist_ok=True)

for ruta in [MERGED_DIR, META_PATH,
             os.path.join(SAVE_DIR, "X_test.npy"),
             os.path.join(SAVE_DIR, "y_test.npy"),
             os.path.join(SAVE_DIR, "y_pred_test.npy"),
             os.path.join(SAVE_DIR, "y_proba_test.npy")]:
    assert os.path.exists(ruta), f"No encontrado: {ruta}"
print("Archivos requeridos verificados")

# ============================================================
# CARGAR METADATA, MODELO Y DATOS
# ============================================================
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

id2label      = {int(k): v for k, v in metadata["id2label"].items()}
label2id      = metadata["label2id"]
paises_unicos = sorted(label2id.keys())
NUM_LABELS    = metadata["num_labels"]
MAX_LENGTH    = metadata.get("max_length", 128)

print(f"Clases: {NUM_LABELS} | max_length: {MAX_LENGTH}")

print("\nCargando tokenizer y modelo merged...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
xai_model = AutoModelForSequenceClassification.from_pretrained(
    MERGED_DIR,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    attn_implementation="eager"
)
xai_model.to(device)
xai_model.eval()
print("Modelo cargado")

X_test        = np.load(os.path.join(SAVE_DIR, "X_test.npy"),        allow_pickle=True)
y_test        = np.load(os.path.join(SAVE_DIR, "y_test.npy"))
y_pred_test   = np.load(os.path.join(SAVE_DIR, "y_pred_test.npy"))
y_proba_test  = np.load(os.path.join(SAVE_DIR, "y_proba_test.npy"))
print(f"Test set: {len(X_test)} muestras")

# ============================================================
# FUNCIÓN DE PREDICCIÓN
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

# ============================================================
# SELECCIÓN: top-50 correctamente clasificados por clase
# Solo ejemplos correctos: las predicciones erróneas mezclan
# señales de múltiples dialectos y contaminan las atribuciones.
# ============================================================
N_POR_CLASE = 50
TOP_PALABRAS = 20

seleccion = {}   # pais -> {'indices': [...], 'probs': [...]}

print("\nSeleccionando top-50 correctamente clasificados por clase...")
for pais in paises_unicos:
    clase_idx = label2id[pais]

    # Solo ejemplos donde etiqueta real == pais Y predicción == pais (correctos)
    mask_correctos = (y_test == clase_idx) & (y_pred_test == clase_idx)
    indices_correctos = np.where(mask_correctos)[0]

    if len(indices_correctos) == 0:
        print(f"  {pais}: sin ejemplos correctamente clasificados, omitido")
        continue

    # Ordenar por P(clase) descendente
    probs_clase = y_proba_test[indices_correctos, clase_idx]
    orden = np.argsort(probs_clase)[::-1]
    top_idx = indices_correctos[orden[:N_POR_CLASE]]
    top_probs = probs_clase[orden[:N_POR_CLASE]]

    seleccion[pais] = {
        "indices": top_idx,
        "probs":   top_probs,
        "n":       len(top_idx)
    }
    print(f"  {pais}: {len(top_idx)} ejemplos | P media = {top_probs.mean():.3f}")

paises_validos = sorted(seleccion.keys())
print(f"\nPaíses con ejemplos suficientes: {len(paises_validos)}/{NUM_LABELS}")

# ============================================================
# LIME — Agregación de importancias por clase
# ============================================================
print("\n" + "="*60)
print("LIME — Agregación por dialecto")
print("="*60)

lime_explainer = LimeTextExplainer(
    class_names=paises_unicos,
    split_expression=r'\W+',
    random_state=SEED
)

lime_agregado = {}   # pais -> {palabra: score_acumulado}

for pais in paises_validos:
    clase_idx = label2id[pais]
    indices   = seleccion[pais]["indices"]
    acum      = defaultdict(float)
    n_ok      = 0

    print(f"\n  [{pais}] Procesando {len(indices)} ejemplos con LIME...")
    t0 = time.time()

    for i, test_idx in enumerate(indices):
        texto = X_test[test_idx]
        try:
            exp = lime_explainer.explain_instance(
                texto, predecir_proba,
                num_features=TOP_PALABRAS,
                num_samples=200,
                top_labels=NUM_LABELS
            )
            disponibles = exp.available_labels()
            label_usado = clase_idx if clase_idx in disponibles else disponibles[0]
            importancias = exp.as_list(label=label_usado)

            for palabra, score in importancias:
                if score > 0:          # solo contribuciones positivas hacia esta clase
                    acum[palabra] += score
            n_ok += 1
        except Exception as e:
            print(f"    Error en {test_idx}: {e}")

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(indices)} — {time.time()-t0:.0f}s")

    # Normalizar por número de ejemplos procesados
    if n_ok > 0:
        lime_agregado[pais] = {
            w: s / n_ok for w, s in sorted(acum.items(), key=lambda x: -x[1])   # promedio por muestra para comparar entre países con distinto n disponible
        }
    print(f"    Completado: {n_ok} ejemplos exitosos | {len(acum)} palabras únicas")

# Guardar resultados LIME
lime_out = {
    pais: dict(list(scores.items())[:TOP_PALABRAS])
    for pais, scores in lime_agregado.items()
}
lime_json_path = os.path.join(OUT_DIR, "lime_palabras_por_dialecto.json")
with open(lime_json_path, "w", encoding="utf-8") as f:
    json.dump(lime_out, f, ensure_ascii=False, indent=2)
print(f"\nResultados LIME guardados: {lime_json_path}")

# Guardar CSV
lime_rows = []
for pais, scores in lime_agregado.items():
    for rank, (palabra, score) in enumerate(list(scores.items())[:TOP_PALABRAS], 1):
        lime_rows.append({"pais": pais, "rank": rank, "palabra": palabra, "score_lime": score})
pd.DataFrame(lime_rows).to_csv(os.path.join(OUT_DIR, "lime_palabras_por_dialecto.csv"), index=False)

# ============================================================
# LIME — Gráficas de barras por país
# ============================================================
print("\nGenerando gráficas LIME por país...")
for pais, scores in lime_agregado.items():
    top = list(scores.items())[:TOP_PALABRAS]
    if not top:
        continue
    palabras, vals = zip(*top)

    fig, ax = plt.subplots(figsize=(9, 6))
    colores = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(palabras)))
    bars = ax.barh(range(len(palabras)), vals, color=colores, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(palabras)))
    ax.set_yticklabels(palabras, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Score LIME promedio (contribución positiva)", fontsize=10)
    ax.set_title(f"LIME — Top {TOP_PALABRAS} palabras | {pais.upper()}", fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va='center', fontsize=8)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"lime_{pais}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Gráficas por país guardadas en {OUT_DIR}/")

# ============================================================
# LIME — Heatmap global: países × palabras más frecuentes
# ============================================================
print("Generando heatmap LIME global...")

# Recopilar las top-10 palabras de cada país
palabras_globales = []
for pais in paises_validos:
    if pais in lime_agregado:
        palabras_globales += list(lime_agregado[pais].keys())[:10]
palabras_globales = list(dict.fromkeys(palabras_globales))   # deduplicar preservando orden

heatmap_data = pd.DataFrame(index=paises_validos, columns=palabras_globales, dtype=float).fillna(0.0)
for pais in paises_validos:
    if pais in lime_agregado:
        for palabra, score in lime_agregado[pais].items():
            if palabra in heatmap_data.columns:
                heatmap_data.loc[pais, palabra] = score

fig, ax = plt.subplots(figsize=(max(18, len(palabras_globales)*0.55),
                                max(8,  len(paises_validos)*0.45)))
sns.heatmap(heatmap_data.astype(float), cmap='YlOrRd', linewidths=0.3,
            linecolor='lightgray', ax=ax, cbar_kws={'label': 'Score LIME promedio'})
ax.set_title("Heatmap LIME — Palabras discriminativas por dialecto", fontsize=14, fontweight='bold')
ax.set_xlabel("Palabra", fontsize=11)
ax.set_ylabel("País", fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lime_heatmap_global.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Heatmap LIME guardado")

# ============================================================
# SHAP — Agregación de importancias por clase
# ============================================================
print("\n" + "="*60)
print("SHAP — Agregación por dialecto")
print("="*60)

shap_explainer = shap.Explainer(
    predecir_proba,
    masker=shap.maskers.Text(tokenizer=r'\W+'),
    output_names=paises_unicos
)

shap_agregado = {}   # pais -> {palabra: score_acumulado}

for pais in paises_validos:
    clase_idx = label2id[pais]
    indices   = seleccion[pais]["indices"]
    acum      = defaultdict(float)
    conteo    = defaultdict(int)
    n_ok      = 0

    print(f"\n  [{pais}] Procesando {len(indices)} ejemplos con SHAP...")
    t0 = time.time()

    for i, test_idx in enumerate(indices):
        texto = X_test[test_idx]
        try:
            sv = shap_explainer([texto])
            tokens = sv.data[0]
            scores = sv.values[0][:, clase_idx]   # SHAP values para esta clase

            for token, score in zip(tokens, scores):
                token_clean = token.strip()
                if token_clean and score > 0:
                    acum[token_clean]   += score
                    conteo[token_clean] += 1
            n_ok += 1
        except Exception as e:
            print(f"    Error en {test_idx}: {e}")

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(indices)} — {time.time()-t0:.0f}s")

    # Score promedio por palabra
    if n_ok > 0:
        shap_agregado[pais] = {
            w: acum[w] / conteo[w]
            for w in sorted(acum, key=lambda x: -acum[x])
            if len(w) > 1
        }
    print(f"    Completado: {n_ok} ejemplos | {len(acum)} tokens únicos")

# Guardar resultados SHAP
shap_out = {
    pais: dict(list(scores.items())[:TOP_PALABRAS])
    for pais, scores in shap_agregado.items()
}
shap_json_path = os.path.join(OUT_DIR, "shap_palabras_por_dialecto.json")
with open(shap_json_path, "w", encoding="utf-8") as f:
    json.dump(shap_out, f, ensure_ascii=False, indent=2)
print(f"\nResultados SHAP guardados: {shap_json_path}")

# Guardar CSV
shap_rows = []
for pais, scores in shap_agregado.items():
    for rank, (palabra, score) in enumerate(list(scores.items())[:TOP_PALABRAS], 1):
        shap_rows.append({"pais": pais, "rank": rank, "palabra": palabra, "score_shap": score})
pd.DataFrame(shap_rows).to_csv(os.path.join(OUT_DIR, "shap_palabras_por_dialecto.csv"), index=False)

# ============================================================
# SHAP — Gráficas de barras por país
# ============================================================
print("\nGenerando gráficas SHAP por país...")
for pais, scores in shap_agregado.items():
    top = list(scores.items())[:TOP_PALABRAS]
    if not top:
        continue
    palabras, vals = zip(*top)

    fig, ax = plt.subplots(figsize=(9, 6))
    colores = plt.cm.Blues(np.linspace(0.4, 0.9, len(palabras)))
    bars = ax.barh(range(len(palabras)), vals, color=colores, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(palabras)))
    ax.set_yticklabels(palabras, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value promedio", fontsize=10)
    ax.set_title(f"SHAP — Top {TOP_PALABRAS} palabras | {pais.upper()}", fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2,
                f"{val:.5f}", va='center', fontsize=8)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"shap_{pais}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

print(f"  Gráficas SHAP por país guardadas en {OUT_DIR}/")

# ============================================================
# SHAP — Heatmap global
# ============================================================
print("Generando heatmap SHAP global...")

palabras_shap_global = []
for pais in paises_validos:
    if pais in shap_agregado:
        palabras_shap_global += list(shap_agregado[pais].keys())[:10]
palabras_shap_global = list(dict.fromkeys(palabras_shap_global))

heatmap_shap = pd.DataFrame(index=paises_validos, columns=palabras_shap_global, dtype=float).fillna(0.0)
for pais in paises_validos:
    if pais in shap_agregado:
        for palabra, score in shap_agregado[pais].items():
            if palabra in heatmap_shap.columns:
                heatmap_shap.loc[pais, palabra] = score

fig, ax = plt.subplots(figsize=(max(18, len(palabras_shap_global)*0.55),
                                max(8,  len(paises_validos)*0.45)))
sns.heatmap(heatmap_shap.astype(float), cmap='Blues', linewidths=0.3,
            linecolor='lightgray', ax=ax, cbar_kws={'label': 'SHAP value promedio'})
ax.set_title("Heatmap SHAP — Palabras discriminativas por dialecto", fontsize=14, fontweight='bold')
ax.set_xlabel("Palabra / Token", fontsize=11)
ax.set_ylabel("País", fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_heatmap_global.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Heatmap SHAP guardado")

# ============================================================
# COMPARACIÓN LIME vs SHAP — palabras en común por país
# ============================================================
# Jaccard entre LIME y SHAP: si ambos métodos coinciden en las top palabras, la señal es robusta
print("\nCalculando coincidencias LIME vs SHAP por país...")
coincidencias = []
for pais in paises_validos:
    top_lime = set(list(lime_agregado.get(pais, {}).keys())[:TOP_PALABRAS])
    top_shap = set(list(shap_agregado.get(pais, {}).keys())[:TOP_PALABRAS])
    comun = top_lime & top_shap
    coincidencias.append({
        "pais":          pais,
        "n_lime":        len(top_lime),
        "n_shap":        len(top_shap),
        "n_comun":       len(comun),
        "jaccard":       len(comun) / len(top_lime | top_shap) if (top_lime | top_shap) else 0.0,
        "palabras_comun": ", ".join(sorted(comun))
    })

df_coincidencias = pd.DataFrame(coincidencias)
df_coincidencias.to_csv(os.path.join(OUT_DIR, "lime_shap_coincidencias.csv"), index=False)

print("\nCoincidencia LIME vs SHAP por país:")
print(df_coincidencias[["pais","n_comun","jaccard"]].to_string(index=False))

# Gráfica de Jaccard por país
fig, ax = plt.subplots(figsize=(12, 5))
colores_j = ['#2ecc71' if j >= 0.3 else '#e67e22' if j >= 0.15 else '#e74c3c'
             for j in df_coincidencias['jaccard']]
ax.bar(df_coincidencias['pais'], df_coincidencias['jaccard'],
       color=colores_j, edgecolor='black', linewidth=0.5)
ax.axhline(df_coincidencias['jaccard'].mean(), color='navy', linestyle='--',
           linewidth=1.5, label=f"Media = {df_coincidencias['jaccard'].mean():.3f}")
ax.set_xlabel("País", fontsize=11)
ax.set_ylabel("Índice de Jaccard (LIME ∩ SHAP)", fontsize=11)
ax.set_title(f"Coincidencia entre top-{TOP_PALABRAS} palabras LIME y SHAP por dialecto",
             fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lime_shap_jaccard.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Gráfica Jaccard guardada")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*60)
print("COMPLETADO — Archivos generados en:", OUT_DIR)
print("="*60)
archivos = [
    "lime_palabras_por_dialecto.json",
    "lime_palabras_por_dialecto.csv",
    "lime_heatmap_global.png",
    "shap_palabras_por_dialecto.json",
    "shap_palabras_por_dialecto.csv",
    "shap_heatmap_global.png",
    "lime_shap_coincidencias.csv",
    "lime_shap_jaccard.png",
    f"lime_{{pais}}.png  ×{len(paises_validos)}",
    f"shap_{{pais}}.png  ×{len(paises_validos)}",
]
for a in archivos:
    print(f"  {a}")
