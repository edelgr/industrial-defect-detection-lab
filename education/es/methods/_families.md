## Familias de detección de defectos y anomalías industriales

Elija un método a la izquierda. Los puntos llenos (●) son ejecutables en esta máquina; los vacíos
(○) se explican pero no se ejecutan en IDD Lab v0.1.

| Grupo | Idea | Datos de entrenamiento | Salida típica |
|---|---|---|---|
| **A — Visión clásica** | Operadores diseñados a mano | Ninguno o una referencia | Máscaras / mapas |
| **B — Supervisado** | Aprender de defectos etiquetados | Muchos defectos etiquetados | Clases, cajas, máscaras |
| **C — Reconstrucción** | Redibujar imágenes normales; lo mal reconstruido es anómalo | Imágenes normales | Mapas de residuo |
| **D — Distribuciones y bancos de memoria** | Características pre-entrenadas + estadística o vecinos | Imágenes normales | Mapas de distancia |
| **E — Maestro–estudiante** | El estudiante imita al maestro; el desacuerdo es anomalía | Imágenes normales | Mapas de desacuerdo |
| **F — Modelos fundacionales** | Características genéricas, zero/few-shot | 0–k imágenes normales | Mapas de similitud |
| **G — 3D y multimodal** | Geometría, registro, varios sensores y fusión | Escaneos normales | Puntuaciones por punto o modalidad |

**Supervisado vs. detección de anomalías.** Los modelos supervisados reconocen las clases de
defecto con las que se entrenaron; los detectores de anomalías encuentran lo que difiere de lo
normal, incluso defectos nunca vistos.
