## Experimento de fusión multimodal (protocolo educativo)

Elija una categoría instalada de **MulSen-AD** y pulse *Ejecutar*.

1. Las imágenes **RGB** e **infrarrojas** se analizan con un detector (PatchCore o PaDiM) ajustado
   solo con imágenes de *entrenamiento* de esa modalidad.
2. **3D**: la línea base de registro + desviación compara cada escaneo con referencias normales.
3. Cada puntuación se convierte en **z-score** con media y desviación de **muestras normales de
   entrenamiento reservadas** (nunca de prueba).
4. **Fusión tardía**: media y máximo de los z-scores.
5. Se reporta AUROC frente a la **etiqueta del objeto** y a la **etiqueta por modalidad** (`data.csv`).

No es el método MulSen-TripleAD del artículo; es un protocolo transparente para estudiar la
complementariedad de sensores. Los resultados son **educativos locales**.
