# Dataset

El dataset utilizado en este proyecto es `Datos_dialectos.parquet` (~77 MB, 912,681 tweets en español de 21 países hispanohablantes).

Por su tamaño no está incluido en este repositorio. Puedes descargarlo desde:

- **HuggingFace Datasets:** https://huggingface.co/datasets/JorgeLoera/spanish-dialect-tweets

## Formato

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `texto` | str | Texto del tweet (preprocesado) |
| `pais`  | str | Código de país ISO (ar, bo, cl, ...) |

## Países cubiertos (21)

`ar` Argentina · `bo` Bolivia · `cl` Chile · `co` Colombia · `cr` Costa Rica · `cu` Cuba · `do` Rep. Dominicana · `ec` Ecuador · `es` España · `gq` Guinea Ecuatorial · `gt` Guatemala · `hn` Honduras · `mx` México · `ni` Nicaragua · `pa` Panamá · `pe` Perú · `pr` Puerto Rico · `py` Paraguay · `sv` El Salvador · `uy` Uruguay · `ve` Venezuela
