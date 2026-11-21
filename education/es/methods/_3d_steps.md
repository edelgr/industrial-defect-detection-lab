## Cómo funciona la inspección 3D (línea base educativa)

1. **Nube normal.** Uno o más escaneos sin defectos sirven de referencia; se submuestrean en vóxeles y se estiman normales.
2. **Nube de prueba.** El escaneo a inspeccionar, en su propio sistema de coordenadas.
3. **Alineación.** *Gruesa*: descriptores FPFH + RANSAC. *Fina*: ICP punto-a-plano. Revise el
   **fitness** y el **RMSE**: si la alineación falla, las desviaciones no tienen sentido.
4. **Comparación geométrica.** Para cada punto alineado, el punto de referencia más cercano y su normal.
5. **Desviación.** Distancia punto-a-superficie `|n·(p − q)|` en las unidades del archivo. Puntuación = percentil 99.
6. **Visualización 3D coloreada** y comparación con la verdad de terreno.

**Limitaciones.** Objetos simétricos pueden alinearse invertidos; la variación normal de forma
aparece como anomalía. Los métodos publicados (Reg3D-AD, M3DM, MulSen-TripleAD) añaden
características aprendidas y bancos de memoria.
