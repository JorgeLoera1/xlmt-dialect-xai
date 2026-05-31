# Clasificación de Variedad Dialectal del Español en Twitter con Explicabilidad Post-Hoc

Reporte Técnico — Estancia de Investigación MCE 2026  
**Autor:** Jorge Antonio Loera Grande

---

## Descripción

Este proyecto implementa un sistema de clasificación automática de **21 variedades dialectales del español** en tweets, combinando fine-tuning eficiente con **LoRA** sobre el modelo `cardiffnlp/twitter-xlm-roberta-base` (XLM-T) y cuatro métodos de **explicabilidad post-hoc** (LIME, SHAP, Integrated Gradients y Attention Rollout).

Hasta donde se tiene conocimiento, es la primera investigación que aplica métodos XAI post-hoc cuantitativos a la clasificación de 21 variedades dialectales del español en Twitter.

---

## Resultados principales

| Modelo | F1-Macro | Accuracy |
|--------|----------|----------|
| Azar aleatorio (21 clases) | 0.0476 | — |
| XLM-T congelado + Reg. Logística | 0.1895 | 0.1916 |
| BERTIN + LoRA | 0.3207 | 0.3212 |
| XLM-T + LoRA v1 (5 épocas) | 0.3857 | 0.3861 |
| **XLM-T + LoRA v2 (10 épocas)** | **0.3923** | **0.3954** |

### Métricas XAI (promedio global)

| Grupo de confianza | Faithfulness | Complexity | Stability |
|--------------------|-------------|------------|-----------|
| Alta confianza (p ≥ 0.5) | 0.822 | 0.630 | +0.674 |
| Baja confianza (p < 0.5) | 0.089 | 0.867 | −0.247 |

> Las explicaciones XAI son informativas únicamente cuando la confianza del modelo es alta (p ≥ 0.5).

---

## Países cubiertos (21)

`ar` Argentina · `bo` Bolivia · `cl` Chile · `co` Colombia · `cr` Costa Rica · `cu` Cuba · `do` Rep. Dominicana · `ec` Ecuador · `es` España · `gq` Guinea Ecuatorial · `gt` Guatemala · `hn` Honduras · `mx` México · `ni` Nicaragua · `pa` Panamá · `pe` Perú · `pr` Puerto Rico · `py` Paraguay · `sv` El Salvador · `uy` Uruguay · `ve` Venezuela

---

## Estructura del repositorio

```
xlmt-dialect-xai/
├── src/
│   ├── train/
│   │   ├── train_xlmt_v2.py       # Entrenamiento final (XLM-T + LoRA, 10 épocas)
│   │   ├── train_xlmt_v1.py       # Versión inicial (5 épocas); corregido bug de doble encoding UTF-8 en regex
│   │   ├── train_bertin.py        # Baseline monolingüe (BERTIN + LoRA)
│   │   └── train_baseline.py      # XLM-T congelado + Regresión Logística
│   └── xai/
│       ├── xai_methods.py         # LIME, SHAP, Integrated Gradients, Attention Rollout
│       ├── xai_metrics.py         # Faithfulness, Complexity, Stability por grupo de confianza
│       ├── xai_words.py           # Palabras discriminativas por dialecto
│       └── xai_sections.py        # Heatmap robusto y análisis por muestra (5 × país)
├── results/figures/
│   ├── training/                  # Curvas de aprendizaje, matrices de confusión
│   ├── eda/                       # Distribuciones y wordclouds por país
│   └── xai/
│       ├── confianza/             # Explicaciones por grupo: alta (p≥0.5) y baja (p<0.5)
│       │   └── {pais}/alta|baja/  # 4 métodos × tweet por grupo
│       ├── por_pais/              # 5 muestras aleatorias × 21 países × 4 métodos
│       │   └── {pais}/            # lime / shap / ig / attention por muestra
│       └── robusto/               # Heatmap de métricas XAI (30 muestras × país)
├── data/
│   └── README.md                  # Instrucciones para descargar el dataset
├── models/
│   └── README.md                  # Links a modelos en HuggingFace Hub
├── requirements.txt
└── .gitignore
```

---

## Instalación

```bash
git clone https://github.com/jorgeloera/xlmt-dialect-xai.git
cd xlmt-dialect-xai
pip install -r requirements.txt
```

> Requiere Python 3.11 y GPU con al menos 6 GB VRAM (probado en RTX 3050).

---

## Uso

### 1. Descargar el dataset

El dataset está disponible en HuggingFace: https://huggingface.co/datasets/JorgeLoera/spanish-dialect-tweets

Ver instrucciones en [`data/README.md`](data/README.md).

### 2. Entrenar el modelo final

```bash
python src/train/train_xlmt_v2.py
```

### 3. Reproducir el análisis XAI

```bash
python src/xai/xai_methods.py   # LIME, SHAP, IG, Attention Rollout + métricas por muestra
python src/xai/xai_metrics.py   # Faithfulness, Complexity, Stability por grupo de confianza
python src/xai/xai_words.py     # top palabras discriminativas por dialecto
python src/xai/xai_sections.py  # heatmap robusto (30 muestras × país)
```

Los resultados se guardan en `results/figures/xai/`.

---

## Modelo pre-entrenado

El modelo final está disponible en HuggingFace Hub. Ver [`models/README.md`](models/README.md).

---

## Metodología

### Fine-tuning con LoRA

| Hiperparámetro | Valor |
|----------------|-------|
| Modelo base | `cardiffnlp/twitter-xlm-roberta-base` |
| Rango (r) | 64 |
| Alpha (α) | 128 |
| Dropout | 0.1 |
| Épocas | 10 |
| Batch efectivo | 128 |
| Learning rate | 1×10⁻⁴ |
| Parámetros entrenables | ~2% del total |

### Métodos XAI implementados

| Método | Nivel de atribución |
|--------|---------------------|
| LIME | Palabra |
| SHAP | Palabra |
| Integrated Gradients | Subtoken |
| Attention Rollout | Subtoken |

### Métricas de evaluación XAI

Se evaluaron cuatro métricas candidatas: Faithfulness, Comprehensiveness, Complexity y Stability. **Comprehensiveness fue excluida** del análisis final por redundancia: su correlación con Faithfulness fue r ≈ 1.0, aportando información duplicada sin valor explicativo adicional.

### Tipos de señales aprendidas por el modelo

- **Tipo A — Geográficas:** topónimos y gentilicios (`malabo`, `bolivia`, `ecuador`)
- **Tipo B — Políticas y culturales:** referencias a figuras nacionales (`amlo`, `capriles`, `bukele`)
- **Tipo C — Léxico dialectal genuino:** palabras propias de cada variedad (`weon`, `mae`, `catracha`)

---

