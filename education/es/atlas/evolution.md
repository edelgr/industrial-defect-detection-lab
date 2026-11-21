### Cómo evolucionaron los benchmarks de detección de anomalías industriales

**2019 — MVTec AD: el protocolo no supervisado.** Las fábricas rara vez tienen suficientes
ejemplos de cada defecto. MVTec AD fijó un protocolo realista: *entrenar solo con imágenes sin
defectos* y detectar y localizar defectos nunca vistos. Para 2022 muchas categorías superaban el
99 % de AUROC de imagen.

**2022 — MVTec LOCO AD: lógica, no solo textura.** Una caja de desayuno con *dos* naranjas en vez
de una no tiene ningún rayón. LOCO añadió anomalías *lógicas* (piezas faltantes, sobrantes o mal
colocadas) y la métrica sPRO.

**2022 — VisA: diversidad.** Doce productos con múltiples instancias, PCB complejas y objetos no
alineados; sus particiones few-shot impulsaron la investigación zero/few-shot.

**2023 — Real3D-AD: solo geometría.** Nubes de puntos de 360° con solo cuatro prototipos normales
por clase: sin color ni textura, y el registro se vuelve esencial.

**2025/26 — MVTec AD 2: realismo y evaluación honesta.** Cambios de iluminación entre
entrenamiento y prueba, objetos transparentes, defectos diminutos en imágenes enormes y
**etiquetas de prueba privadas** evaluadas solo en el servidor de MVTec.

**2025 — MulSen-AD y Real-IAD D³: más de un sensor.** Algunos defectos están *dentro* de la pieza
y solo los ve la termografía infrarroja; otros son geométricos y solo los ve un escáner 3D.

**2025/26 — 3D-ADAM: fabricación aditiva.** Las piezas impresas en 3D tienen líneas de capa
*normales*, lo que dificulta definir qué es una anomalía.

> **Por qué importa:** un método excelente en un benchmark puede fallar en otro, porque cada uno
> prueba un supuesto distinto — alineación, localidad, iluminación, modalidad.
