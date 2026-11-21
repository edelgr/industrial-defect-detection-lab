## ¿Qué es un benchmark industrial?

Un benchmark fija **cinco cosas** para que los resultados sean comparables:

1. **Datos** — productos, sensores y condiciones.
2. **Particiones** — qué imágenes se usan para entrenar, validar (p. ej. elegir umbrales) y probar.
   Los datos de prueba nunca deben influir en el entrenamiento ni en los umbrales.
3. **Anotaciones** — etiquetas, máscaras de píxel, máscaras por región, etiquetas por punto o por modalidad.
4. **Métricas** — AUROC de imagen, AUROC de píxel / PRO / sPRO, SegF1.
5. **Protocolo** — cómo se calculan y reportan (p. ej. el servidor privado de MVTec AD 2).

### Por qué el protocolo cambia los resultados publicados
- Elegir el umbral con etiquetas de prueba infla F1, precisión y exhaustividad.
- El AUROC de píxel está dominado por los píxeles normales; PRO y sPRO ponderan cada región por igual.
- Redimensionar imágenes grandes a 256 px puede borrar defectos diminutos.

### Cuatro tipos de resultados — nunca los mezcle
| Categoría | Significado en IDD Lab |
|---|---|
| **Reportado por autores** | Citado de un artículo |
| **Benchmark reproducido** | Protocolo oficial completo re-ejecutado |
| **Educativo local** | Experimento local con subconjunto o protocolo no oficial |
| **Demostración sintética** | Calculado sobre datos sintéticos |
