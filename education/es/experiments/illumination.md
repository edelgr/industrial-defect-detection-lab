## ¿Puede un modelo soportar cambios de iluminación?

**Intuición.** Los modelos no supervisados aprenden lo «normal» a partir de imágenes de entrenamiento tomadas
con una sola configuración de luz. En MVTec AD 2 todas las imágenes de entrenamiento y validación son
`regular`; `test_public` vuelve a fotografiar las *mismas escenas* (las mismas piezas físicas) como `regular`,
`overexposed` (sobreexpuesta), `underexposed` (subexpuesta) y con cambios de luz (`shift_1…3`). Un modelo
robusto debería dar a cada escena el mismo veredicto con cualquier iluminación. Un modelo que aprendió el
*brillo* de las imágenes normales en lugar de su *estructura* genera falsas alarmas en piezas normales bajo
una luz no vista, o no detecta defectos cuando baja el contraste.

**Qué observar: modo Básico.**
1. Elija una **escena** (◀ ▶ para recorrerlas). Cada **tarjeta** es una condición de iluminación de la misma
   pieza: la imagen (con el ground truth contorneado), su mapa de anomalía, una barra con la puntuación frente
   al umbral, la decisión OK/NOK y su resultado: **TP** (defecto detectado), **TN** (pieza buena aceptada),
   **FP** (falsa alarma) o **FN** (defecto no detectado). El umbral es el de validación, que no usa etiquetas de test.
2. Pulse *Calcular mapas de anomalía* (EN VIVO, si está disponible): todos los mapas usan la misma escala de
   color, así se ve hacia dónde se desplaza la evidencia cuando solo cambia la luz. Haga doble clic en una
   imagen o pulse ⤢ para ampliar imagen y mapa lado a lado.
3. El **gráfico** muestra la puntuación de esta escena en cada condición, con las zonas OK y NOK; un ◆ marca
   dónde cambia la decisión. *¿Qué ha pasado?* explica el resultado con palabras sencillas.

**Qué observar: modo Avanzado.**
1. El **gráfico de todas las escenas** tiene una línea por escena: haga clic en una línea para seleccionar esa
   escena (las demás se atenúan) y en un punto para seleccionar una condición. ✗ marca los errores de la escena
   seleccionada y ◆ sus cruces del umbral. Las líneas solo unen la misma escena: las condiciones son
   categorías, no un continuo.
2. *Escenas × iluminación*: cada decisión como TP/TN/FP/FN, con errores, cambios de decisión, rango de
   puntuaciones y distancia mínima al umbral por escena. *AUROC por iluminación*: AUROC por condición, agrupado
   y por escena, con intervalos del 95 % por bootstrap de clústeres. *Estadística*: el análisis por escenas y
   sus limitaciones. *Referencias*: los artículos del dataset y del método.

**¿Almacenado o en vivo?** Las puntuaciones, decisiones, AUROC e intervalos proceden siempre de las puntuaciones
por imagen **almacenadas** de la ejecución verificada. La inferencia EN VIVO solo añade los mapas de anomalía;
cada mapa muestra su puntuación en vivo junto a la almacenada.

**Por qué escenas y no imágenes.** sheet_metal tiene 114 imágenes de prueba pero solo 19 escenas (4 normales y
15 anómalas) × 6 condiciones; fruit_jelly, 80 imágenes = 20 escenas × 4. Las variantes de una escena están muy
correlacionadas: **no** son observaciones independientes. Por eso IDD Lab
* informa el AUROC por condición (una imagen por escena, independientes dentro de la condición),
* informa un AUROC a nivel de escena (una puntuación agregada por escena),
* calcula intervalos de confianza por **bootstrap de clústeres**: remuestrea escenas completas (estratificado
  por clase) y mantiene juntas todas sus variantes de iluminación. Al lado se muestra el intervalo ingenuo a
  nivel de imagen: finge que hay 114 (u 80) imágenes independientes y suele ser más estrecho. Sin embargo, con
  solo 4–5 escenas normales incluso el bootstrap por clústeres es poco fiable (muy pocos clústeres) y a veces
  resulta más estrecho que el ingenuo (p. ej., PaDiM en fruit_jelly): lea ambos solo como indicaciones aproximadas.

**Limitaciones.** Con 4–5 escenas normales todos los intervalos son anchos y discretos: quitar o duplicar una
sola escena normal puede mover el AUROC hasta ≈ 0,2–0,25 (con 4 escenas normales, cada una aporta la cuarta parte de todas las comparaciones normal–anómala). Las diferencias por condición son exploratorias y no deben
leerse como un ranking de iluminaciones. Con menos de 3 escenas por clase IDD Lab no informa ningún intervalo.
Resultados educativos locales sobre `test_public`; la evaluación oficial de robustez usa la partición privada
`test_private_mixed` en el servidor de MVTec.

**Lectura complementaria.** Vo, Pham, Pham y Nguyen, *IEEE Access* 14 (2026), doi:10.1109/ACCESS.2026.3735579:
análisis de la resolución, la estructura espacial, la iluminación y la sensibilidad a la semilla en la
localización de anomalías sin entrenamiento sobre MVTec AD 2 (Atlas → MVTec AD 2 → referencias complementarias).
