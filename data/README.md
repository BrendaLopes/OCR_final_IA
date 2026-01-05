# Datasets

Este repositorio NO incluye datasets grandes (por tamaño).
El dataset manuscrito está en Google Drive y debe descargarse localmente.

## Dataset manuscrito (Drive)

Estructuras soportadas:

### A) Dataset principal por clases
- `mayusculas/A/*.png`
- `mayusculas/B/*.png`
- ...
- `minusculas/a/*.png`
- ...
- `numeros/0/*.png`
- ...

Ejemplos de nombres: `A_Adrian_Garcia.png`, `6_Brenda_Lopes.jpg`.

### B) "Datasets OCR otros años"
- `Datasets OCR otros años/u01204/Mayúsculas/*.png`
- `Datasets OCR otros años/u01204/Minúsculas/*.png`
- `Datasets OCR otros años/u01204/Números/*.png`

En algunos casos, los archivos se llaman directamente `a.png`, `b.png`, etc.

## Cómo colocarlo en el proyecto

1. Descarga desde Drive y coloca todo en:

`data/raw/handwritten/`

Ejemplo:

```
data/raw/handwritten/
  mayusculas/
  minusculas/
  numeros/
  Datasets OCR otros años/
```

2. Genera el dataset procesado:

```bash
python scripts/make_handwritten_dataset.py \
  --input data/raw/handwritten \
  --output data/processed/handwritten_chars \
  --copy-originals
```

Salida:

- `data/processed/handwritten_chars/images/` (normalizadas 32×32)
- `data/processed/handwritten_chars/labels.csv`

---

## Commit de documentación

```bash
git add data/README.md
git commit -m "docs: add dataset instructions"
git push
```
