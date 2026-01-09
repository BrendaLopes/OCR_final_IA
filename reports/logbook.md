##reports/logbook.md

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

  [2026-01-08] Experimento AZ09 v1

Objetivo: MVP de reconocimiento manuscrito limitado a A–Z + 0–9 (36 clases).

Dataset: data/processed/handwritten_AZ09, 1364 imágenes, 36 clases.

Modelo: CNN simple 3 bloques conv + FC (SimpleCNN).

Entrenamiento: 20 epochs, batch 64, Adam lr=1e-3, augment suave (rotación ±10°, traslación ±2px, erode/dilate).

Resultado: best val acc = 0.7766, train acc final ~0.88.

Diagnóstico (QA): muchos hard examples muestran fallo de preprocesado (recortes negros/barras/manchas), causando predicciones muy confiadas erróneas.

Decisión: mejorar normalización (umbral robusto + connected components + QC) y reconstruir dataset limpio.

[2026-01-08] QA y análisis de errores

Herramienta: scripts/qa_hard_examples.py --only-wrong --k 30 --open

Observación: aparecen imágenes normalizadas casi negras (posible sombra/fondo capturado).

Pares confusos detectados: 0/O, Q/O, N/M, V/W, 6/U/G, 1/I.

Acción siguiente: patch en make_handwritten_dataset.py para seleccionar componente conectado del carácter y filtrar outliers.

[2026-01-08] Mejora de preprocesado (dataset manuscrito)

Problema detectado: muchas imágenes normalizadas salen como bloques negros/barras → el recorte capturaba sombra/borde como “tinta”.

Cambio aplicado en make_handwritten_dataset.py:

binarizado robusto (Otsu en dos polaridades, elección por ratio de tinta),

selección de componente conectado “más probable” (descarta fondo gigante),

control de calidad (descarta outliers de tinta/bbox).

Objetivo: asegurar que cada muestra 32×32 contenga realmente el carácter, reduciendo ruido y mejorando la precisión del clasificador.

Fecha/Hora: 08/01 17:20

Prueba: Reconstrucción del dataset manuscrito con nuevo preprocesado (binarizado + recorte por componente + padding a cuadrado + resize 32×32 + guardado de rechazados).

Comando: python3 scripts/make_handwritten_dataset.py --input ... --output ... --copy-originals --save-rejected

Resultado esperado: eliminación de recortes defectuosos (“barras negras”), y dataset normalizado estable.

Siguiente acción: revisión visual (contact sheet) + reentrenamiento CNN


“Tras aplicar una etapa de control de calidad (QC) con --reject-on-qc, se eliminaron 90 muestras corruptas del conjunto original. El modelo CNN alcanzó un 69.9 % de precisión en validación con 62 clases (dígitos, mayúsculas y minúsculas). Las confusiones más frecuentes corresponden a caracteres visualmente similares (p.ej. 1/I/l, O/0, Z/2). El resultado cumple los requisitos del proyecto y demuestra una generalización adecuada con un dataset limitado.”