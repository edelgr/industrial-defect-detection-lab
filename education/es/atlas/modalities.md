## Modalidades de sensado

| Modalidad | Qué se mide físicamente | Defectos que revela | Precauciones |
|---|---|---|---|
| **Cámara RGB / gris** | Luz reflejada por la superficie | Rayones, manchas, decoloración, piezas faltantes | La iluminación altera la apariencia; materiales brillantes o transparentes son difíciles |
| **Infrarrojo — termografía lock-in** (MulSen-AD) | Respuesta térmica a un calentamiento modulado periódicamente | Grietas sub-superficiales, huecos, delaminación, roturas internas | Las imágenes suelen ser *visualizaciones normalizadas*, **no temperaturas calibradas** |
| **Escaneo 3D láser / luz estructurada** | Geometría: coordenadas 3D de la superficie (mm) | Abolladuras, protuberancias, deformación, material faltante | Requiere registro; unidades, ruido y oclusiones importan |
| **Pseudo-3D por estéreo fotométrico** (Real-IAD D³) | Normales/sombreado a partir de varias direcciones de luz | Relieve fino invisible en una sola foto | No es profundidad métrica |
| **Imágenes de profundidad (xyz TIFF)** | Coordenadas 3D por píxel alineadas con la imagen | Defectos geométricos con alineación a la imagen | Valores faltantes, ruido del sensor |

## Tipos de anomalía

- **Estructural / superficial** — cambio local de apariencia: rayón, grieta, mancha.
- **Lógica** — cada parte se ve bien, pero la *configuración* es incorrecta (MVTec LOCO AD).
- **Geométrica** — la forma se desvía: abolladuras, protuberancias (Real3D-AD, MulSen-AD 3D).
- **Interna** — defectos bajo la superficie, visibles en termografía pero no en cámaras (MulSen-AD).

Un método diseñado para un tipo puede ser ciego a otro: PatchCore destaca en anomalías
estructurales pero falla en lógicas; los métodos solo RGB no ven defectos internos.
