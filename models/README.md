# Modelos entrenados

Los pesos de los modelos no están incluidos en este repositorio por su tamaño. Están disponibles en HuggingFace Hub:

| Modelo | Descripción | F1-Macro | Link |
|--------|-------------|----------|------|
| XLM-T + LoRA v2 | Modelo final (10 épocas) | 0.3923 | _[link pendiente]_ |
| XLM-T + LoRA v1 | Versión inicial (5 épocas) | 0.3857 | _[link pendiente]_ |
| BERTIN + LoRA   | Baseline monolingüe       | 0.3207 | _[link pendiente]_ |
| XLM-T congelado | Sin fine-tuning + LogReg  | 0.1895 | _[link pendiente]_ |

## Cargar el modelo final

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("jorgeloera/xlmt-dialect-xai")
tokenizer = AutoTokenizer.from_pretrained("jorgeloera/xlmt-dialect-xai")
```
