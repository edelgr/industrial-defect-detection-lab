## Comprender las métricas de evaluación

### Ranking frente a decisión
Un detector de anomalías produce una **puntuación**, no un veredicto. Se pueden hacer dos preguntas distintas:

* **Ranking**: *¿las piezas defectuosas reciben puntuaciones más altas que las buenas?* Se responde con
  métricas sin umbral (AUROC, AP, AU-PRO).
* **Decisión**: *con el umbral que realmente usaremos en la línea, ¿cuántas piezas se aceptan o rechazan
  correctamente?* Se responde con VP/FP/VN/FN, precisión, exhaustividad (recall), F1 y las tasas de falsa
  aceptación y falso rechazo. Dependen por completo de **un** umbral elegido.

Un modelo puede ordenar bien y aun así decidir mal si el umbral está en el lugar equivocado.

### Las métricas, de forma intuitiva
* **AUROC de imagen**: tome al azar una imagen defectuosa y una buena; el AUROC es la probabilidad de que la
  defectuosa reciba la puntuación mayor (0,5 = azar; 1,0 = ranking perfecto). Ignora *cuánto* se separan las
  puntuaciones y dónde está cualquier umbral.
* **AUROC de píxel**: la misma pregunta para píxeles (un píxel de defecto frente a uno normal). En MVTec AD 2
  los defectos son diminutos (≈ 0,25–0,4 % de la imagen), así que dominan los píxeles normales: un mapa puede
  alcanzar un AUROC de píxel alto y aun así no detectar defectos pequeños completos.
* **AU-PRO@0.05**: para cada *región* de defecto conexa se mide la fracción de sus píxeles por encima del umbral
  (solapamiento por región, PRO) y se promedia entre regiones, de modo que un rayón diminuto cuenta tanto como
  una abolladura grande. Se recorre el umbral mientras la tasa de falsos positivos en píxeles normales sea
  ≤ 5 %, se toma el área bajo esa curva y se normaliza a 0–1. Es la métrica principal de localización de
  MVTec AD 2 (el artículo reduce el límite clásico de 0,3 a 0,05 para tolerar menos píxeles falsos positivos).
* **F1**: media armónica de la precisión (de las piezas rechazadas, ¿cuántas eran defectuosas?) y la
  exhaustividad (de las defectuosas, ¿cuántas se rechazaron?). F1 vale 0 si no se rechaza nada.
* **Falsos positivos (FP)**: piezas buenas rechazadas, con coste de chatarra y reinspección (tasa de falso
  rechazo, FRR). **Falsos negativos (FN)**: piezas defectuosas aceptadas, que llegan al cliente (tasa de falsa
  aceptación, FAR). Mover el umbral cambia unos por otros.

### De dónde sale el umbral
Un protocolo válido fija el umbral **sin etiquetas de test**. IDD Lab usa el percentil 99 de las puntuaciones
de las imágenes `validation/good` (piezas normales nunca usadas para el ajuste). La regla base del artículo de
MVTec AD 2 usa la media + 3 desviaciones típicas de las puntuaciones por píxel de las imágenes de validación sin
defectos, y el *máximo* del mapa de anomalía como puntuación de imagen (Sec. IV-B2). Se muestran ambas.

Si las normales de validación puntúan más bajo que las normales de test (otra iluminación, pocas imágenes,
variación natural), un umbral por percentil puede quedar **por encima de todas las imágenes de test**: no se
rechaza nada, exhaustividad = 0, F1 = 0, aunque el ranking (AUROC) sea razonable. El ejemplo real de PaDiM ·
fruit_jelly reproduce exactamente esta situación: ábralo con el botón y compare los histogramas gris
(validación), azul (test normal) y rojo (test anómalo) con el umbral verde.

### La exploración retrospectiva no es evaluación
El deslizador permite mover el umbral sobre las puntuaciones *almacenadas* del test para ver el compromiso
precisión/exhaustividad. En cuanto lo mueve, el umbral se ha elegido **mirando las etiquetas de test**: la
franja naranja marca esos números como exploración retrospectiva. Son optimistas y nunca deben informarse como
resultados (el artículo de MVTec AD 2 considera inapropiado ajustar con datos de test en detección no
supervisada; sus etiquetas privadas lo hacen imposible en el benchmark oficial). Los números válidos son siempre los de la columna verde.
