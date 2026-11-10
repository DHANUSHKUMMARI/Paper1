# Datasets

The datasets are **not stored in this repository**:

- **Size:** the medium profile is roughly 60–100 GB. GitHub rejects files over 100 MB and recommends keeping repositories small; Git LFS storage would exceed the project budget.
- **Licences:** GenImage is released for non-commercial research and SID-Set (CC-BY-4.0) requires attribution; redistributing copies here could breach their terms.

Instead, the data is rebuilt on Google Drive from the original sources, and this folder keeps the small files needed to reproduce it exactly (`manifests/`: file lists, labels, splits, shard checksums).

| Dataset | Source | Drive folder | Licence | Role |
|---|---|---|---|---|
| GenImage (Validation + SD v1.4) | Kaggle mirror `vtphatt2/GenImage-*` (original: github.com/weili-thu/genimage) | `01_GenImage` | Non-commercial research | Cross-generator detection |
| DiffSeg30k | Hugging Face `Chaos2629/Diffseg30k` | `02_DiffSeg30k` | Apache-2.0 | Local AI edits + masks |
| SID-Set (capped) | Hugging Face `saberzl/SID_Set` | `03_SID-Set` | CC-BY-4.0 | Real / synthetic / tampered |
| BR-Gen | Hugging Face `clpbc/BR-Gen` | `04_BR-Gen` | See dataset card | Cross-dataset localization |
| Chameleon | Manual (AIDE repository links) | `05_Chameleon` | See source | Hard test set |
| MMTD-Set | Hugging Face `zhipeixu/MMTD-Set-34k` | `06_MMTD-Set` | See dataset card | Reference explanations |

Downloads run **once**, from `notebooks/00_setup_and_data.ipynb` (or `notebooks/tools/Paper1_Dataset_Downloader_v3_medium.ipynb`). Later runs read Drive only. Cite each dataset's original paper.
