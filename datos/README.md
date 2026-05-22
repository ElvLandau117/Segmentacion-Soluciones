# datos/

Carpeta requerida por la rúbrica de despliegue de la actividad de
Coursera / Universidad de los Andes.

## ¿Por qué esta carpeta está vacía en GitHub?

El **MaIA Scoliosis Dataset** es **propiedad de la Universidad de los
Andes** y no se redistribuye con este repositorio. Está listado en
`.gitignore` para evitar commits accidentales.

## Sobre el dataset

| Característica | Valor |
|---|---|
| Nombre | MaIA Scoliosis Dataset |
| Propietario | Universidad de los Andes — Maestría en IA |
| Modalidad | Radiografías AP (anteroposterior) de columna completa |
| Total de imágenes | 250 |
| Casos Normal | 71 |
| Casos Scoliosis | 179 |
| Resolución típica | 259×971 a 381×1074 px (sin estandarizar) |
| Etiquetas | Máscaras de segmentación binaria + multiclase (24 vértebras C3 a L5) |
| Splits usados | 70% train (174) / 15% val (38) / 15% test (38), seed=42 — ver `data_splits.json` |

## Estructura esperada del dataset (cuando se descarga)

```
MaIA_Scoliosis_Dataset/
├── Normal/
│   ├── N_1.jpg
│   ├── N_2.jpg
│   └── ...                      # 71 archivos
├── Scoliosis/
│   ├── S_21.jpg
│   ├── S_22.jpg
│   └── ...                      # 179 archivos
├── LabelBinaryJPG/
│   ├── Label_N_1.jpg            # 1 = spine, 0 = background
│   ├── Label_S_21.jpg
│   └── ...                      # 250 archivos (uno por imagen)
├── LabelMultiClass_Color_JPG/   # vértebras coloreadas (visualización)
├── LabelMultiClass_Gray_JPG/    # vértebras en grises (intermedio)
├── LabelMultiClass_ID_PNG/      # IDs 0-35 (ground truth de entrenamiento)
├── RadiographMetrics/           # Cobb angle ground truth + metadata
├── dataset_index.csv            # tabla maestra con metadata por imagen
└── labels_dictionary.json       # mapping ID → nombre de clase
```

## Cómo obtener el dataset

El dataset NO es público. Para acceso académico, contactar:

- **Profesor titular** del curso de Maestría en IA, U. Andes.
- **Líder del proyecto** Elvis Hernández — `elvis.hernandez@en-firme.com`.

Una vez obtenido, descomprimir en la raíz del repo:
```bash
unzip MaIA_Scoliosis_Dataset.zip -d ./
```

La carpeta resultante `MaIA_Scoliosis_Dataset/` está en `.gitignore` —
no se subirá al repo accidentalmente.

## Splits reproducibles

El archivo `data_splits.json` en la raíz del repo SÍ está commiteado.
Contiene los nombres exactos de los archivos de cada split (train / val /
test) con seed=42, para que cualquier evaluador pueda reproducir los
splits sin re-shuffle.

## Más información

- `spine_segmentation/data/dataset.py` — clase `SpineMulticlassDataset`.
- `spine_segmentation/data/splits.py` — generación reproducible de splits.
- `spine_segmentation/data/class_mapping.py` — 3 esquemas de clases
  (vertebrae_24, full_36, regional_5).
- `notebooks/01_EDA.ipynb` — análisis exploratorio del dataset.
- `README.md` raíz sección 1 — descripción del problema clínico.
