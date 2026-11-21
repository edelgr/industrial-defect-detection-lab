### Reading the comparison: why MVTec AD 2, MulSen-AD and Real-IAD D³ are different problems

- **MVTec AD 2** stays with RGB images but makes them *realistic*: lighting changes, transparent
  and reflective objects, high normal variability and tiny defects on very large images. Its
  research question is **robustness**, and its private server enforces honest evaluation.
- **MulSen-AD** asks whether **several physical sensors** are needed at the *object* level: some
  defects are visible only to infrared (internal) or only to 3D (geometric). Its question is
  **sensor complementarity** — and labels are given per modality.
- **Real-IAD D³** asks how **2D, pseudo-3D (photometric stereo) and 3D** complement each other at
  very high resolution for small electronic parts. Its question is **fine-grained multimodal
  representation**, at the cost of a very large, gated dataset.

Earlier datasets (MVTec AD, VisA) mostly test *whether* a method can model normal appearance;
newer ones test *under which conditions and with which sensors* it still works.
