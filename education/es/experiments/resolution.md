## ¿Por qué importa la resolución de la imagen?

**Intuición.** Una imagen de sheet_metal mide 4224 × 1056 píxeles —cuatro veces más ancha que alta— y un
defecto típico ocupa alrededor del 0,25 % de ella (mediana de las 90 máscaras públicas); algunos, solo
≈ 0,01 %. Antes de que PatchCore vea nada, la imagen se redimensiona a la entrada de la red:

| Variante | Entrada de la red | Reducción (x / y) | Geometría |
|---|---|---|---|
| Cuadrada (tamaño base del artículo) | 256 × 256 | 16,5 × / 4,1 × | **aplastada**: el detalle horizontal se comprime 4× más que el vertical |
| Aspecto conservado | 1024 × 256 | 4,1 × / 4,1 × | sin distorsión |

PatchCore compara *rasgos de parches* de la imagen de prueba con un banco de memoria de parches normales.
A 256 × 256 cada celda de `layer2` cubre ≈ 132 × 33 píxeles originales: un rayón fino se convierte en una
señal menor que una celda y se diluye. Aplastar la imagen también cambia la *forma* de la textura normal
(marcas de cepillado, bordes) respecto a los patrones de ImageNet que aprendió la red, así que incluso la
noción de «parche normal» queda distorsionada. Con el aspecto conservado cada celda cubre ≈ 33 × 33 píxeles:
4× más parches y más detalle, pero también más memoria (banco mayor) y más latencia.

**Qué observar.**
1. *Lo que ve la red*: compare las dos entradas redimensionadas y busque el defecto en cada una.
2. Los *mapas de calor* usan la misma escala (mapa ÷ umbral calibrado): los colores son comparables.
3. *Máscara predicha frente a la referencia*: verde = píxeles de defecto encontrados, rojo = falsas alarmas,
   azul = píxeles de defecto no detectados.
4. *Zoom*: la vista más informativa para defectos pequeños.
5. *Métricas almacenadas*: AU-PRO@0.05 mide qué fracción de **cada** región defectuosa se encuentra
   permitiendo solo un 5 % de píxeles falsos positivos; es la métrica de localización del benchmark MVTec AD 2.

**Interpretación (en los resultados verificados).** El AUROC de imagen cambia poco y las métricas de
localización (AU-PRO, F1 de píxel) cambian mucho. Una explicación plausible (hipótesis para discutir, no algo
que este experimento demuestre): la puntuación de imagen solo necesita que *un* parche destaque, mientras que
la localización necesita la *extensión y la posición* de defectos pequeños, que se conservan mejor a mayor
resolución. La resolución es un compromiso coste/precisión: el artículo de MVTec AD 2 pide a los
participantes del leaderboard declarar el tamaño de imagen usado.

**Limitaciones.** Una categoría, un backbone, una semilla; `test_public` tiene solo 15 escenas defectuosas y
4 normales, cada una con 6 iluminaciones, así que diferencias de pocos puntos de AUROC están dentro del ruido.
Resultados educativos locales, no puntuaciones oficiales de MVTec AD 2.

**Lectura complementaria.** Vo, Pham, Pham y Nguyen, *IEEE Access* 14 (2026), doi:10.1109/ACCESS.2026.3735579:
análisis de la resolución, la estructura espacial, la iluminación y la sensibilidad a la semilla en la
localización de anomalías sin entrenamiento sobre MVTec AD 2 (Atlas → MVTec AD 2 → referencias complementarias).
