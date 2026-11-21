## Multimodal fusion experiment (educational protocol)

Select an installed **MulSen-AD** category on the left and press *Run*.

1. **RGB** and **infrared** images are each analysed by an image detector (PatchCore or PaDiM)
   fitted on that modality's *training* images only.
2. **3D**: the registration + deviation baseline compares each scan with merged normal reference
   scans.
3. Every modality's raw score is converted to a **z-score** using the mean and standard deviation
   measured on **held-out normal training samples** (never test data).
4. **Late fusion**: mean and maximum of the z-scores.
5. AUROC is reported against the **object label** and against the **per-modality label** from
   MulSen-AD's `data.csv`.

This is *not* the MulSen-TripleAD method of the paper; it is a transparent protocol for studying
sensor complementarity. Results are **local educational** results.
