
## 2026-01-07 — MVP manuscrito (A–Z + 0–9)

### Objetivo
Reducir el problema de 62 clases (A–Z, a–z, 0–9) a un MVP más robusto de 36 clases (A–Z, 0–9) para:
- disminuir ambigüedades (p.ej. i/I/1/l, o/O/0, s/S/5)
- garantizar un OCR funcional como base y ampliar después con minúsculas.

### Prueba / Acción
1) Se generó un dataset filtrado (AZ09) a partir del dataset procesado completo:
- Origen: `data/processed/handwritten_chars/labels.csv` (columnas: filename,label,source_path)
- Destino: `data/processed/handwritten_AZ09/`
  - `images/*.png`
  - `labels.csv` (columnas: filename,label)
- Filtro aplicado: etiquetas que cumplan `^[A-Z0-9]$`.

Resultado del filtrado:
- filas: 1364
- clases únicas: 36

2) Entrenamiento CNN (desde cero) para clasificación de caracteres manuscritos AZ09:
Comando:
`python3 scripts/train_handwritten_cnn.py --data-dir data/processed/handwritten_AZ09 --model-dir models/handwritten_char_cnn_AZ09 --out-dir outputs/AZ09 --epochs 20 --batch-size 64`

Arquitectura (SimpleCNN):
- Entrada: 1×32×32 (escala de grises, tinta invertida)
- 3 bloques Conv(3×3)+ReLU+MaxPool(2)
- Flatten → Linear(128*4*4→256) + Dropout(0.25) → Linear(256→36)

Aumentación (solo train):
- rotación aleatoria (-10° a 10°)
- traslación (-2 a 2 píxeles)
- erosión/dilatación ligera

### Resultados
- Device: mps (Mac)
- Mejor accuracy validación: 0.7766
- Archivos generados:
  - `models/handwritten_char_cnn_AZ09/model.pt` (checkpoint: model_state, classes, epoch, val_acc)
  - `models/handwritten_char_cnn_AZ09/classes.json`
  - `outputs/AZ09/train_log.csv`
  - `outputs/AZ09/confusion_matrix.png`

### Observaciones / Decisiones
- El modelo mejora respecto al entrenamiento con 62 clases (problema más difícil).
- Se detectaron previamente ejemplos “rotos” por preprocesado (manchas negras, recortes agresivos) y confusiones por mayúsc/minúsc.
- Siguiente paso: QA automático (hard examples) con el modelo AZ09 para localizar:
  1) errores por mala calidad de imagen procesada
  2) confusiones típicas entre clases similares
  3) criterios para filtrar/ajustar el preprocesado en `make_handwritten_dataset.py`.
