<!--
Exported from the Claude doc "Paper 1 - Research Specification" (revision 49) on 13 Nov 2026.
The Claude doc is the master copy. If it changes, re-export it and replace this file; do not edit here.
-->

# Paper 1 — Research Specification: Abstract, Keywords, Architecture & Methods

Nov 10, 2026 · @Dhanush

**Direction (confirmed 10 Nov 2026):** Paper 1 is a digital image forensics paper on AI-generated and AI-edited images, with a focus on diffusion models. This spec is the single source of truth for the scope, problem, objectives, method, evaluation and target journals; hand it to Claude Code together with the Implementation Brief. Numbers in the abstract are placeholders until experiments run.

## 1. Research direction and scope

Paper 1 **detects, localizes and explains images that are fully generated or locally edited by modern generative models, with a focus on diffusion models**. It builds a forensic detector and an explanation module; it does not build or train generative models.

| In scope | Out of scope (future work or Paper 2) |
| --- | --- |
| Fully AI-generated images (diffusion models, plus GAN for generality) | Video and audio deepfakes |
| Images with AI-edited regions (diffusion inpainting and editing of real photos and, in DiffSeg30k, of AI-generated base images) | Live identity verification (KYC), liveness and camera-injection attacks |
| Pixel-level localization of edited regions | Classic copy-move and splicing made without generative models |
| Evidence-grounded text explanations with measured hallucination | Designing watermarks or provenance standards (C2PA); detection is treated as complementary to them |
| Robustness to compression, resizing, blur and noise | Training or fine-tuning large multimodal LLMs |
| An insurance-claim-style case study (protocol P7) | Attribution of the exact generator as a main task |

**Terminology for the paper.** Say “AI-generated and AI-edited images” or “diffusion-generated and diffusion-edited images”. “Deepfake” may be used in its broad sense, which covers both face forgery and whole-image synthesis ([DF40](https://ar5iv.labs.arxiv.org/html/2406.13495)). Avoid “research on diffusion models” and “a diffusion-based approach”: the method does not use diffusion.

**Why this matters now**

- **Regulation:** India’s 2026 IT Rules require large platforms to deploy technical measures, including automated tools, to verify users’ declarations about synthetic content ([Mondaq](https://webiis10.mondaq.com/india/social-media/1746136/indias-2026-it-rules-on-synthetic-media-a-structural-shift-in-intermediary-liability)). The EU AI Act Article 50, in force since 2 August 2026, expects detection of both marked and unmarked synthetic content ([Mondaq](https://www.mondaq.com/new-technology/1843358/eu-ai-act-article-50)).
- **Provenance is not enough:** messaging and social platforms strip C2PA credentials on upload, and most AI-generated content carries none, so classifiers remain the only forensic tool in those cases ([SoftwareSeni](https://softwareseni.com/durable-content-credentials-how-provenance-survives-metadata-stripping)).
- **Industry:** insurers report AI-added “damage” in claim photos and few feel prepared; commercial responses highlight suspicious regions for investigators ([SAS](https://www.sas.com/en_us/news/press-releases/2026/may/synthetic-images-ai-insurance-fraud.html)).

## 2. Problem statement

Diffusion models can now produce fully synthetic images, and edit small regions of real photographs, with a realism that defeats both human inspection and many existing detectors. Current forensic methods have three linked weaknesses. (1) **Poor generalization:** detectors trained on one generator lose much of their accuracy on unseen generators and on compressed real-world images. (2) **Little localization of AI edits:** most AI-generated image detectors classify whole images and cannot show which region was altered, although partial diffusion-based edits are a common form of manipulation. (3) **Untrustworthy explanations:** explainable systems based on multimodal large language models require costly fine-tuning, and their explanations are rarely checked against the evidence the detector actually used, producing fluent but unverified, sometimes hallucinated, justifications with no standard way to measure their faithfulness.

There is therefore a need for a lightweight forensic framework that detects and localizes AI-generated and AI-edited content, generalizes beyond its training generators, and produces explanations that are grounded in, and verifiable against, measurable detector evidence.

## 3. Research objectives

**Objective 1 — Generalizable detection and localization.** To design and develop a lightweight dual-branch network that fuses frozen foundation-model (CLIP) semantics with frequency and noise-residual forensic traces to classify images as real, fully generated or locally edited and to localize edited regions at pixel level, with strong generalization to unseen generators and datasets and robustness to common image degradations.

**Objective 2 — Grounded, hallucination-aware explanation and its evaluation.** To develop an evidence-grounded explanation module that converts detector attributions and predicted masks into an ID-tagged evidence record and generates forensic reports with a frozen LLM that may cite only that evidence, enforced by a rule-based verifier; and to establish a quantitative protocol for the faithfulness and hallucination of forensic explanations.

| Objective | Measured by | Protocols (section 10) | Contributions (section 11) |
| --- | --- | --- | --- |
| O1 | Cross-generator accuracy and AP; pixel F1 and IoU of edit masks; accuracy under degradations; Chameleon accuracy | P1–P5, P7 | C1, C4 |
| O2 | Claim-grounding rate, hallucination rate, counterfactual consistency, energy pointing game, human study | P6, P7 | C2, C3 |

**Research questions**

- RQ1 (O1): Does fusing CLIP semantics with frequency and noise traces improve detection and edit localization on unseen generators and editors?
- RQ2 (O1): How much accuracy is kept under JPEG compression, resizing, blur and noise?
- RQ3 (O2): Does restricting the LLM to verified detector evidence reduce hallucinated claims compared with a multimodal LLM that sees the raw image?
- RQ4 (O2): Do the attribution maps and cited regions agree with ground-truth edit masks, and does removing a cited region lower the detector’s confidence?

## 4. Working title

**EviGround: Evidence-Grounded, Hallucination-Aware Explainable Detection and Localization of AI-Generated and AI-Edited Images**

*EviGround* is a placeholder framework name; keep it short, pronounceable and unused (check Google Scholar before finalizing). A shorter alternative for journals with title limits: *Evidence-Grounded Explainable Forensics for AI-Generated and AI-Edited Images*.

## 5. Rough abstract (\~250 words)

Diffusion models can now generate fully synthetic images and edit small regions of real photographs with a realism that defeats human inspection and many existing detectors. Recent explainable forensic systems built on multimodal large language models (MLLMs) add natural-language explanations, but they require costly fine-tuning, and their explanations are rarely checked against the evidence the detector actually used, leaving them prone to hallucination. We propose EviGround, a lightweight framework that (i) classifies images as real, fully generated or locally edited and localizes edited regions with a dual-branch network fusing frozen CLIP semantics with frequency and noise-residual forensic traces; (ii) converts attribution maps and predicted masks into a structured, identifier-tagged evidence record; and (iii) prompts a frozen open-source LLM that never sees the image and may cite only this evidence, with a rule-based verifier that rejects unsupported claims. We further introduce an evaluation protocol that scores attribution maps against ground-truth edit masks and measures the claim-grounding rate, hallucination rate and counterfactual consistency of the generated reports. Experiments on GenImage, DiffSeg30k, SID-Set, BR-Gen and Chameleon, and an insurance-claim case study of diffusion-inpainted damage, show that EviGround reaches \[XX\]% mean accuracy on unseen generators and \[XX\] pixel-level F1 for edit localization, reduces the explanation hallucination rate from \[XX\]% (ungrounded MLLM) to \[XX\]%, and remains robust to JPEG compression and resizing, while using \[XX\]× fewer trainable parameters than fine-tuned MLLM forensic models. Code and evaluation scripts will be publicly released.

*Replace every \[XX\] with measured results; if a claim is not supported by your experiments, remove it rather than soften it.*

## 6. Keywords

AI-generated image detection · Image forgery localization · Diffusion models · Explainable artificial intelligence (XAI) · Large language models · Hallucination · Explanation faithfulness · Digital image forensics

Most Q2 journals allow 4–8 keywords; drop from the end if the limit is lower. Avoid repeating exact title words where the journal asks for complementary keywords.

## 7. Architecture

```text
EviGround architecture, modules M1-M7 (text version of Fig. 1; the drawn figure is in the Claude doc)

                      Input image (native resolution; 256x256 crops / tiles)
                 ______________________|_______________________
                |                      |                       |
        M1 RGB view            M1 Noise residual R      M1 Frequency map F
        224x224 for CLIP       SRM high-pass filters    block DCT log-magnitude
                |                      |_______________________|
                v                                  v
   M2 Semantic branch                    M3 Forensic-trace branch
   CLIP ViT-L/14, frozen, LoRA r=8       EfficientNet-B0 on [R, F]
   -> patch tokens S                     -> trace tokens T
                |__________________________________|
                                 v
                 M4 Cross-attention fusion + learned gate g
                 fused tokens Z = gated mix of S and T
                 ________________|_________________
                |                                  |
     Classification head                  Localization head
     real / fully generated /             FPN decoder -> predicted edit mask
     locally edited
                |__________________________________|
                                 v
        M5 Evidence record (JSON, every item ID-tagged)      <- only this reaches the LLM
        regions E1, E2 ... (box, area, mask confidence, anomalies); global items G1, G2 ...
                                 |
   M7 Agentic tools  <- - ->  M6 Frozen LLM  ------>  Verifier (IDs, numbers)
   (optional)                 sees evidence only         | rejected claim -> retry (max 2), then drop
                                                         v accepted claims
                              Grounded forensic report: verdict + cited claims + mask overlay + heatmap
```

The image flows down through two branches into one fused representation; the LLM at the bottom never sees pixels, only the ID-tagged evidence record (highlighted), and every claim it makes passes the verifier before reaching the report. Use this as Fig. 1 of the paper; redraw it in your figure tool with the same layout.

| Module | Role | Trained or frozen |
| --- | --- | --- |
| M1 Forensic input views | RGB, SRM noise residual and DCT frequency views | Fixed (no parameters) |
| M2 Semantic branch | CLIP ViT-L/14 patch tokens | Frozen backbone + trained LoRA |
| M3 Forensic-trace branch | EfficientNet-B0 on noise + frequency views | Trained |
| M4 Fusion + heads | Cross-attention, gate, 3-class head, mask decoder | Trained |
| M5 Evidence extraction | Grad-CAM++ + mask regions → ID-tagged JSON | Fixed procedure |
| M6 LLM + verifier | Writes the report from evidence; rule-based checks | Frozen LLM, fixed rules |
| M7 Agentic tools (optional) | Extra evidence on request | Fixed tools |

## 8. Methods

EviGround has six modules (M1–M6) plus an optional agentic extension (M7). Only M2's LoRA adapters, M3, M4 and the heads are trained; CLIP and the LLM stay frozen.

### M1 — Forensic input views

From each RGB image I, take a random 256×256 crop in training and 256×256 tiles at test time, always at native resolution: images are never padded or resized before the forensic views are computed. Build three views:

- **RGB view** I, resized to 224×224 for the semantic branch only.
- **Noise-residual view** R = SRM(I): fixed high-pass SRM filters expose noise inconsistencies left by diffusion up-sampling and inpainting boundaries.
- **Frequency view** F: block-wise DCT log-magnitude of the grayscale image.

```latex
R = \mathrm{SRM}(I), \qquad F = \log\left(1 + \left|\mathrm{DCT}_{8\times 8}(I_{\mathrm{gray}})\right|\right)
```

Forensic views are computed at full crop resolution: never down-sample them, or the generator traces disappear.

**Why 256×256, and the crop rules.** GenImage real images come from ImageNet and are often smaller than 512 pixels on one side, while its Stable Diffusion images are exactly 512×512. A 512 crop would force padding on real images only, which a detector can learn as a shortcut. A 256 crop fits almost every image without padding; the few training images with a side under 256 are skipped and counted. For edited images the crop is drawn so that it contains both edited and unedited pixels (edited share between 1% and 90%, up to 10 tries).

**Format and size harmonization (training data).** GenImage stores real images as JPEG and generated images as PNG, and the two classes differ in size; detectors trained on the raw files learn these differences instead of generator traces ([Grommelt et al., 2024](https://arxiv.org/abs/2403.17608)). The rule: within each source, every class must end with the same compression history, set by the most heavily compressed class of that source. Classes that are lossless or less compressed are JPEG-compressed once, as whole images before cropping, to that class's measured quality and chroma subsampling; a class already at that compression is left alone. Compressing an image at a clearly lower quality than its first compression is acceptable, because the stronger compression dominates. What must be avoided is one class compressed twice at similar quality while another is compressed once. The random 256 crop comes after this step.

| Source | What the probe found (13 Nov 2026) | Harmonization |
| --- | --- | --- |
| GenImage | Real: JPEG, quality 96 in about two thirds, the rest spread. Generated: PNG | Compress generated images with quality and subsampling sampled from the real images' measured distribution; real images untouched (as Grommelt et al.) |
| SID-Set | Real: JPEG at many qualities. Synthetic: PNG. Tampered: JPEG quality 75 with 4:2:0 subsampling (a few PNG) | Compress real, synthetic and PNG tampered images to quality 75, 4:2:0; skip and count real images already below quality 75 |
| DiffSeg30k and the COCO real negatives | All PNG after preparation; one pipeline for both | Compress both with the GenImage real distribution |

Checkpoint CP1 verifies on the harmonized data that file format, JPEG quality and crop size no longer predict the label, and reports pixel-level 8×8 blockiness per class and source as a diagnostic of the compression history.

**Test-time rule.** Tile the image with 256×256 tiles (stride 256, last tile aligned to the border, at most 9 tiles on a regular grid); an image smaller than 256 is processed once at its own size. Stitch the tile masks into the full mask. Image-level scores: P(real) and P(fully generated) are the mean over tiles; P(edited) is the maximum over tiles. The image is called edited if P(edited) ≥ 0.5 (threshold fixed on validation data), otherwise the larger of the other two decides.

### M2 — Semantic branch (frozen foundation model)

CLIP ViT-L/14 image encoder (`openai/clip-vit-large-patch14`, loaded as `CLIPVisionModel` from Hugging Face `transformers`), frozen, with LoRA adapters (rank 8) on the query and value projections of the last 4 blocks. Output: one class token and 256 patch tokens on a 16×16 grid (width 1024), projected to d = 512. Foundation features are what make detectors generalize to unseen generators (UnivFD, NS-Net); LoRA adds forgery sensitivity at a small parameter cost. `transformers` is used instead of `open_clip` because its CLIP exposes separate query and value layers that the `peft` LoRA library can target directly.

### M3 — Forensic-trace branch

A lightweight CNN (EfficientNet-B0, ImageNet-initialized, first layer adapted to 4 input channels) takes the concatenated \[R, F\] views (3 noise-residual channels + 1 frequency channel) and outputs feature maps at strides 4, 8 and 16. The stride-16 map is projected to width d and flattened into trace tokens T. For a 256×256 crop this is a 16×16 grid, so each trace token covers the same image area as one CLIP patch token.

### M4 — Cross-attention fusion and heads

Bidirectional cross-attention lets each branch query the other; a learned gate g weighs their contribution (g is reported later as evidence):

```latex
\tilde{S} = S + \mathrm{CA}(S, T, T), \quad \tilde{T} = T + \mathrm{CA}(T, S, S), \quad Z = g \odot \tilde{S} + (1-g) \odot \tilde{T}
```

- **Classification head:** attention pooling over the fused tokens Z (one weight α per token, summing to 1), joined with CLIP's class token → 3-way softmax over {real, fully generated, locally edited}.
- **Localization head:** a lightweight FPN decoder on Z and T̃ → edit-probability mask M̂ ∈ \[0,1\]^(H×W).

Mask targets: the mask marks only locally edited pixels. It is all zeros for real images and for fully generated images, and the ground-truth mask for edited images (DiffSeg30k, SID-Set tampered). About half of DiffSeg30k's base images are themselves AI-generated; their unedited background is still labelled 0, which is why a fully generated image cannot be labelled all ones. Whether an image is fully generated is decided by the classification head, not by the mask.

**Training loss**

```latex
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}(\hat{y}, y) + \lambda_{\mathrm{seg}}\big(\mathcal{L}_{\mathrm{BCE}}(\hat{M}, M) + \mathcal{L}_{\mathrm{Dice}}(\hat{M}, M)\big) + \lambda_{\mathrm{att}}\,\mathcal{L}_{\mathrm{att}}
```

The optional attention-alignment term pushes the model's own attention-pooling weights α to sit inside the true edited region, so the decision is right for the right reasons. Apply it to edited images only, with the mask M down-sampled to the 16×16 token grid. It is applied to α and not to the Grad-CAM++ map of M5, because a loss on Grad-CAM++ needs second-order gradients, which are slow and unstable in mixed precision:

```latex
\mathcal{L}_{\mathrm{att}} = 1 - \sum_{p} \alpha_p\, M_p, \qquad \sum_{p} \alpha_p = 1
```

Start with λ\_seg = 1.0 and λ\_att = 0.1; tune on the validation split. Report results with and without L\_att as an ablation.

### M5 — Evidence extraction

1. Compute the Grad-CAM++ map A for the predicted class on the 16×16 grid of fused tokens Z, up-sampled to the image size.
2. Threshold M̂ at τ = 0.5 and take connected components as candidate regions r₁…rₖ (ignore regions under 0.5% of the image).
3. For each region compute: bounding box, area %, mean mask confidence, attribution mass inside the region, frequency anomaly (high-frequency DCT energy in the region relative to the rest of the image) and noise anomaly (residual variance in the region relative to the background).
4. Add global evidence: class probabilities, branch gate g (semantic vs trace reliance) and image-level frequency and noise scores.
5. Give every item an ID (E1, E2 … for regions; G1, G2 … for global evidence) and serialize to JSON.

Images predicted real or fully generated have no mask regions. For a fully generated verdict, add up to 3 attribution regions (A1, A2 …) from the top 10% of the Grad-CAM++ map, so the report can still point to where the detector looked; the faithfulness metrics EPG and CC are computed on edited images only.

```json
{
  "prediction": {"real": 0.04, "fully_generated": 0.07, "locally_edited": 0.89},
  "global": [
    {"id": "G1", "type": "branch_reliance", "semantic": 0.38, "trace": 0.62},
    {"id": "G2", "type": "image_freq_anomaly", "value": 0.21}
  ],
  "regions": [
    {"id": "E1", "bbox": [312, 140, 96, 120], "area_pct": 4.1, "mask_conf": 0.93,
     "attr_mass": 0.71, "freq_anomaly": 2.8, "noise_anomaly": 3.4}
  ]
}
```

### M6 — Grounded LLM reasoner and verifier

- **Model:** an open instruction-tuned LLM of about 7–8B parameters (e.g., Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct), frozen, temperature 0. It receives **only the JSON evidence record, never the image**. This is the key design choice: the model cannot describe artifacts it was not given.
- **Prompt contract:** every claim must cite at least one evidence ID; output follows a fixed JSON schema {verdict, confidence, claims\[{text, evidence\_ids}\], limitations}.
- **Verifier (rule-based):** checks that (a) every cited ID exists, (b) numbers quoted in a claim match the evidence within tolerance, (c) region claims cite regions with mask\_conf ≥ τ. Failing claims are returned to the LLM with the error (at most 2 retries), then dropped. Log every rejection, as these counts feed the hallucination metrics.
- **Output:** a short forensic report a non-expert can read, plus the mask overlay and attribution heatmap.

### M7 — Agentic extension (optional, ablation only)

The LLM may call three tools before writing the report: re-run the detector on a zoomed crop of a region, compute an error-level-analysis map of a region, and compute a region's frequency spectrum. Tool outputs are appended to the evidence record with new IDs. Report it as an ablation (with vs. without tools) so the core contribution stays simple.

## 9. Implementation and training details

| Item | Setting |
| --- | --- |
| Software | PyTorch 2.x, transformers (CLIP and the LLM), timm (EfficientNet-B0), peft (LoRA), bitsandbytes (4-bit LLM), pytorch-grad-cam (Grad-CAM++) |
| Hardware | Colab Pro: L4 (22.5 GB) for training and evaluation, T4 (15 GB) for debugging and LLM reports (4-bit, about 6 GB) |
| Two trained models | **Detection-protocol model:** GenImage SD v1.4 only (real + fully generated), the training set used by published GenImage baselines. Storage limit: a fixed random subset of 40,000 real + 40,000 generated SD v1.4 training images is used (seed 42) out of about 162,000 of each; say so in the paper, since published baselines train on the full set. **Unified model:** GenImage SD v1.4 + DiffSeg30k (images edited only by the 6 training editors) + about 4,850 real COCO val2017 photos prepared like DiffSeg30k's base images (shorter side to 1024, centre crop), because the public DiffSeg30k release contains no real images; the 150 photos reserved for Claims-Edit are excluded + SID-Set train subset (10k per class). Same architecture and settings |
| Validation (model selection) | A fixed, stratified 10% held out from each training source. No test set is ever used for model selection or threshold tuning |
| Class and source balance | Weighted sampler, 1 : 1 : 1 over real / fully generated / edited; within a class, SID-Set supplies one third of the samples in every class, so its quality-75 compression cannot point to a class (real: GenImage, SID-Set, COCO negatives one third each; generated: GenImage two thirds; edited: DiffSeg30k two thirds) |
| Epoch definition | Unified: 24,000 sampled crops per epoch (8,000 per class), 20 epochs = 15,000 steps. Detection-protocol: 16,000 crops per epoch, 15 epochs = 7,500 steps |
| Optimizer | AdamW; lr 1e-4 (fusion + heads + trace branch), 5e-5 (LoRA); weight decay 0.05; cosine schedule, 1-epoch warm-up |
| Batch / stopping | 32; early stopping on validation AUC (patience 4) |
| Augmentation | After the harmonization of section 8 (M1): random JPEG (quality 60–100, p = 0.5, same for all classes), resize 0.5–1.5× before cropping (p = 0.3), Gaussian blur (σ 0–2, p = 0.2), Gaussian noise, horizontal flip; all applied *before* computing SRM/DCT views |
| Seeds | 3 runs (42, 123, 2026) of each model, report mean ± std |
| Release | Code, trained weights, LLM prompt and verifier rules on GitHub with a Zenodo DOI |

These are starting values, not tuned results; record every change you make so the final paper's settings match what you ran.

### 9.1 Verification checkpoints

Eleven checkpoints (CP0–CP10) let you verify the implementation as it happens; the day for each is in Paper 1 — 15-Day Execution Plan (EviGround). After each one, Claude Code adds a short entry to `docs/CHECKPOINTS.md`: what was run, the exact numbers, the run folder, and PASS or FAIL against the criteria below. Do not start the next stage until the current checkpoint passes. Targets marked *(ours)* are working thresholds set for this project, not values from the literature.

| CP · day | What you check | Pass criteria | If it fails |
| --- | --- | --- | --- |
| CP0 · Day 1 — Environment | Fresh Colab runtime: mount Drive, clone with the secret, run `pytest`; CI on GitHub | All tests pass in Colab and CI; `git grep` finds no tokens; no dataset, checkpoint or secret file is tracked | Fix before any data work |
| CP1 · Day 2 — Data integrity and leakage | Leakage report, manifests, 4×4 sample grid per class | 0 duplicate image hashes across train/val/test; class counts equal the manifests; a classifier using only file format, JPEG quality and crop size (original image size is reported separately as a diagnostic) scores ≤ 60% real-vs-generated on the harmonized training data; pixel-level 8×8 blockiness is reported per class, and a clear class difference means stop and ask *(ours)* | Harmonization (section 8, M1) is already in the pipeline; if the check still fails, find the leaking attribute from the classifier's feature importances, remove it and re-check |
| CP2 · Day 4 — Model build | Unit tests, model summary, 64-image overfit log, kill-and-resume log | Output shapes as in section 8; frozen CLIP weights unchanged after a step; overfit ≥ 99% train accuracy within 300 steps; resumed run continues at the same step with loss within 1% | Debug; do not launch full training |
| CP3 · Day 5 — Training health | `metrics.csv`, loss and AUC curves, `best.pt`, `compute.json` | No NaN or divergence; validation AUC rises then plateaus; in-distribution SD v1.4 accuracy ≥ 95% *(ours; the GenImage ResNet-50 baseline reaches 99.9%)* | Check learning rates, label and mask alignment, augmentation order |
| CP4 · Day 6 — Generalization gate | Seed 42 of both models: P1 for the detection-protocol model; P2 and P3 for the unified model; each next to a CLIP linear probe trained on the same data | P1 mean accuracy over 8 generators above the linear probe and above the GenImage ResNet-50 average of 72.1%; P2 binary mIoU ≥ 0.80 on the held-out editors (ours; the DiffSeg30k paper's SegFormer reaches 0.951 on this pair, so any gap is reported, not hidden*)*; red flag: ≥ 98% on every unseen generator suggests leakage | One fix only (stronger JPEG/resize augmentation, LoRA on layers 16–23, or higher λ\_seg), re-run once; if it still fails, narrow the claims and continue |
| CP5 · Day 7 — Evidence faithfulness | 20 evidence overlays, randomization sanity test, EPG, deletion/insertion curves | On edited images EPG is clearly above the mask's area fraction (the random baseline); trained vs randomized-weight maps SSIM < 0.5; deletion AUC below random-order deletion | Use the supervised mask-decoder output as the evidence map and report Grad-CAM++ as secondary |
| CP6 · Day 8 — Grounded explanations | 30 reports read beside their evidence JSON and image; verifier logs; code path of M6 | 100% of claims in verified reports cite an existing evidence ID; 0 unsupported claims in the 30 read; pre-verifier rejection rate logged (this is HR); the LLM input never contains image pixels or paths; MockLLM tests pass | Tighten the prompt and output schema; reports still failing after 2 retries fall back to a template report, and their count is reported |
| CP7 · Day 10 — Full benchmark | All result CSVs for P1–P6 and baselines | Mean ± std over 3 seeds and bootstrap 95% CIs everywhere; at least two baselines reproduce their published numbers within about 3 points; EviGround's HR below the ungrounded MLLM's; every table cell traces to a CSV row | Re-run a mis-reproduced baseline once with its authors' settings; otherwise quote its published numbers and say so |
| CP8 · Day 11 — Claims-Edit case study | Claims-Edit manifest, 30 pairs checked by eye, P7 table, 6 example reports | Inpainting model not among training editors (asserted in the log); ≤ 10% of pairs dropped; no tuning on this set | Swap the inpainting model and rebuild; never tune on the case-study set |
| CP9 · Day 14 — Ablations, statistics, human study | Ablation table, significance tests, efficiency table, rating sheets | Each of the 6 ablations compared with the full model by a paired test; Fleiss' κ reported with its interpretation (≥ 0.40 = moderate agreement); efficiency table complete | Report honestly where a component does not help; a low κ becomes a stated limitation |
| CP10 · Day 15 — Manuscript integrity | 15 numbers picked at random from the manuscript, traced through `docs/RESULTS.md` | All 15 match a CSV and a run folder exactly; no number typed by hand; release `v1.0` tagged; data and code availability statement written | Do not submit until every mismatch is fixed |

GenImage reference numbers: [GenImage repository, Table 3](https://github.com/weili-thu/genimage).

## 10. Evaluation protocol

**Experimental protocols**

| Protocol | Model (section 9) | Test on | Metrics |
| --- | --- | --- | --- |
| P1 Cross-generator detection | Detection-protocol model (GenImage SD v1.4 only); unified model as a second, clearly labelled row | GenImage validation, all 8 generators; raw files and the JPEG- and size-harmonized version | ACC, AP, mean over generators |
| P2 Edit localization, unseen editors | Unified model | DiffSeg30k validation images containing an edit by a held-out editor (Glide, Kolors) | Pixel F1, IoU, AUC; ACC, mIoU and boundary F1 as defined in the DiffSeg30k paper |
| P3 Cross-dataset localization | Unified model | BR-Gen (never used in training) | Image ACC / AUC, pixel F1, IoU |
| P3b In-domain three-class benchmark | Unified model | SID-Set validation split (not cross-dataset: a disjoint SID-Set train subset is in the training data) | 3-class ACC, per-class F1; pixel F1 and IoU on tampered images |
| P4 Hard real-world test | Detection-protocol model; unified model as a second row | Chameleon | ACC, real recall, fake recall |
| P5 Robustness | Both models | Fixed subset: 500 real + 500 generated per GenImage generator and 1,000 DiffSeg30k held-out images, under JPEG 95→50, resize 0.5–1.5×, blur σ 1–3, noise | Accuracy / F1 vs. severity curves |
| P6 Explanation quality | Unified model | 1,500 images, fixed seed: 400 edited each from DiffSeg30k held-out, SID-Set validation and BR-Gen; 150 real; 150 fully generated | EPG, deletion/insertion AUC, GR, HR-E, HR-GT, CC, human study |
| P7 Insurance-claim case study | Unified model (no retraining) | Claims-Edit set (below) | ACC, pixel F1, HR-GT, CC, example reports |

**Fair-comparison rules (state them in the paper).**

- *Why two models.* Published GenImage results train on SD v1.4 only. A model that also saw DiffSeg30k and SID-Set cannot be compared with them, so P1 and P4 use the detection-protocol model for the headline numbers.
- *Label every baseline row* as one of: **same protocol** (published numbers, same training set), **zero-shot** (authors' released weights, different training data), or **same training data** (our own ablations and the CLIP linear probe).
- *Editor hold-out in DiffSeg30k.* Each image carries up to three edits by different models. Training images are those whose every edit comes from the 6 training editors (SD 2, SD 3.5, SDXL, FLUX.1, HunyuanDiT 1.1, Kandinsky 2.2). Test images contain at least one Glide or Kolors edit; pixel metrics are reported on all edited pixels and on held-out-editor pixels only. Glide + Kolors is one of the three hold-out pairs in the DiffSeg30k paper and the only one whose models appear nowhere else in our training data (FLUX generates SID-Set's synthetic images; SD 2 is close to SD v1.4).
- *Dataset bias.* GenImage results are reported on the raw files, for comparison with the literature, and on the harmonized version, which is the honest measure of generator traces.

**Claims-Edit case study (industrial relevance, about 2–3 compute units).** Build about 300 test images: 150 real photos of everyday objects and vehicles from public, licence-compatible sources (e.g., COCO categories car, couch, chair, dining table, laptop) and 150 edited copies where a diffusion inpainting model from an editor family the unified model never trains on (Kolors inpainting, since Kolors is held out in P2; if its weights cannot be loaded, BrushNet or PowerPaint) adds realistic damage — cracks, dents, stains, broken glass — inside a random object mask, saving each mask as ground truth. Do not use SDXL, FLUX, SD 3.5, SD 2, HunyuanDiT or Kandinsky 2.2 here: they are DiffSeg30k training editors. Report it as a case study, not a benchmark claim, and release the generation script and masks.

**Explanation faithfulness and hallucination metrics (the core of the paper)**

- **Energy pointing game (EPG):** share of attribution energy inside the true edited region.

```latex
\mathrm{EPG} = \frac{\sum_p A_p M_p}{\sum_p A_p}
```

- **Deletion / insertion AUC:** remove or add pixels in attribution order and track the detector score (RISE protocol).
- **Sanity check:** randomize model weights layer by layer; a faithful map must change (Adebayo et al.).
- **Claim-grounding rate (GR)** and **two hallucination rates (HR-E, HR-GT), all measured on the LLM's first output, before the verifier removes anything,** so the gain from grounding is visible:

```latex
\mathrm{GR} = \frac{\#\,\text{claims passing all verifier rules}}{\#\,\text{claims}}, \quad \text{HR-E} = 1 - \mathrm{GR}, \quad \text{HR-GT} = \frac{\#\,\text{checkable claims contradicting the ground truth}}{\#\,\text{checkable claims}}
```

A claim is one entry of the report's `claims` list. **HR-E** (against the evidence) counts claims that cite a missing ID, misquote a number or cite a low-confidence region; it exists only for evidence-grounded systems. **HR-GT** (against the ground truth) is defined for every system, including an MLLM that sees the image, and answers RQ3. Checkable claims are the verdict claim (wrong if it differs from the true class) and region claims (wrong if less than 10% of the claimed box lies inside the ground-truth mask). Our region claims take their box from the cited evidence ID; the ungrounded MLLM is prompted with the same report schema and must give a box for each region claim. Claims about a property that has no ground truth (for example a frequency anomaly) are excluded from HR-GT.

- **Counterfactual consistency (CC):** blank out each region the report cites; the detector's edited-class probability must drop by at least δ (e.g., 0.2). CC is the share of cited regions where it does.
- **Human study:** 200 reports, 3 raters, 1–5 Likert scores for correctness, usefulness and clarity; report Fleiss' κ. Raters give informed consent, see reports in random order without knowing which system wrote them, and no personal data is collected; say so in the paper's ethics statement.

**Baselines** (confirm public code before committing)

- Detection (P1, P4): CNNSpot, UnivFD, NPR, DIRE, FatFormer, AIDE. Use their published GenImage SD v1.4 numbers (same protocol); run released weights only where no such number exists, and label that row zero-shot
- Localization (P2, P3): the DiffSeg30k paper's SegFormer and DeepLabv3+ results for the same held-out pair (trained on DiffSeg30k only); TruFor and NFA-ViT with released weights (zero-shot); our semantic-only and trace-only variants (same training data)
- Explanation (P6): an ungrounded MLLM (Qwen2-VL-7B-Instruct, 4-bit) that sees the raw image and writes a report in the same schema; a fixed-template report built from the evidence record without any LLM
- Large explainable models: FakeShield and SIDA need more GPU memory than this project has, so compare with their published SID-Set numbers and parameter counts only; REVEAL and FORGE are compared through their published numbers and discussed in related work unless code and weights that fit a 22 GB GPU are released

**Ablations**

1. Remove the frequency view (F); remove the noise view (R)
2. Semantic branch only vs. trace branch only vs. fused
3. With vs. without L\_att
4. Grad-CAM++ vs. the model's own attention-pooling map α as the attribution source
5. Grounded LLM vs. LLM shown the image vs. no verifier vs. fixed template without an LLM (needs no retraining)
6. With vs. without M7 agentic tools

**Statistics and efficiency:** mean ± std over 3 seeds; McNemar's test for accuracy and bootstrap 95% CIs for F1 against the best baseline; trainable and total parameters, FLOPs, latency per image and GPU memory against FakeShield/SIDA-scale models.

## 11. Novelty and contributions

| Approach | Where the explanation comes from | Trains or fine-tunes a large (M)LLM? | Explanation checked against detector evidence? | Localizes AI edits? |
| --- | --- | --- | --- | --- |
| FakeShield (ICLR 2025) | Fine-tuned multimodal LLM looking at the image | Yes | No: text compared with GPT-4o reference descriptions | Yes |
| SIDA (CVPR 2025) | Fine-tuned large multimodal model (13B) looking at the image | Yes | No: explanations are not verified against detector evidence | Yes |
| REVEAL (2026) | MLLM reasoning over chains of evidence from expert models | Yes: expert-grounded reinforcement learning | Evidence-grounded training | No: labels and text only |
| FORGE (EMNLP 2026 Findings) | MLLM with an added forensic vision encoder; region-level explanations of faces | Yes: trains adapters in two stages | Region-specific evaluation and a human study | Face regions |
| ACM MM 2026 Explainable Deepfake Detection Challenge (top entries) | DINOv3 or similar detector + Qwen3-VL explainer | Yes: preference learning with evidence-grounded rewards | Scored by an LLM judge | No |
| TriDF (CVPR 2026) | Benchmark, not a detector: measures MLLM hallucination | n/a | Measures it, does not fix it | n/a |
| **EviGround (ours)** | **Frozen LLM that never sees the image; cites only ID-tagged detector evidence** | **No** | **Yes: rule-based verifier + GR / HR / CC against masks** | **Yes, with fully synthetic detection in one model** |

**Four differentiators to state in the Introduction and defend in the experiments**

1. **Training-free explanation:** no (M)LLM fine-tuning or reinforcement learning; the LLM is frozen and swappable.
2. **Deterministic verification:** a rule-based verifier, not an LLM judge, decides whether each claim is supported.
3. **One lightweight model** for fully generated detection and AI-edit localization, at Colab-scale compute.
4. **Faithfulness measured against ground truth:** energy pointing game and counterfactual consistency use real edit masks, not reference text.

**Contributions (use as the numbered list in your Introduction)**

1. A lightweight dual-branch detector-localizer that handles real, fully generated and locally edited images in one model.
2. An evidence-grounded LLM reasoning module with a rule-based verifier, requiring no (M)LLM fine-tuning.
3. A faithfulness and hallucination evaluation protocol for forensic explanations, using edit masks as ground truth.
4. Cross-generator, cross-dataset, robustness and ablation experiments on five public benchmarks plus an insurance-claim case study, with released code.

Every contribution is tested by a protocol in section 10: C1 by P1–P5, C2 and C3 by P6 and ablation 5, C4 by the full experiment set including P7. Re-check arXiv and Google Scholar monthly for new evidence-grounded forensic explainers, and add them here before submission.

**Positioning (keep the claims honest).** The novelty of the paper is the evidence-grounded explanation module and its evaluation protocol (C2, C3). The detector (C1) is claimed to be lightweight, unified and competitive, not state of the art: strong specialised models exist for each task (for example SegFormer trained on DiffSeg30k reaches 0.951 mIoU on unseen editors). Report every gap openly; reviewers accept a competitive detector with a new, well-tested explanation method far more readily than an unsupported state-of-the-art claim.

## 12. Target journals

Quartiles differ by index and subject category, and they change every year. Before submitting, confirm the journal’s current Scopus (SJR) and Web of Science (JCR) quartile and which one your institution counts.

| Order | Journal (all Springer, hybrid: subscription route has no APC) | Quartile in the CS-IT sheet (SJR 2025) | Why it fits |
| --- | --- | --- | --- |
| 1 | Multimedia Systems | Q1, SJR 0.669, SCIE | Detection + localization + explanation as one multimedia forensics system |
| 2 | Pattern Analysis and Applications | Q2, SJR 0.662, SCIE | Pattern analysis methods with industrial case studies; suits the Claims-Edit study |
| 3 | The Visual Computer | Q2, SJR 0.599, SCIE | Image analysis; published an image-forgery survey in 2025. Desk rejections are reported |
| 4 | Applied Intelligence | Q2, SJR 0.889, SCIE | Applied AI; frame the paper around the LLM, verifier and trustworthy explanation |
| 5 | International Journal of Multimedia Information Retrieval | Q1, SJR 1.071, SCIE | Multimedia analysis; weakest scope fit of the five because the journal centres on retrieval |

**Submission strategy:** write the manuscript in the Springer Nature LaTeX template (`sn-jnl`), which all five journals accept, and check each journal's reference style and length limit before submitting. Submit to Multimedia Systems first; after a rejection, revise using the reviews and move down the list. Cite 2–3 recent papers from the journal you submit to. The earlier Elsevier targets (JISA, JVCIR) were dropped because they are not on the no-APC list.

## 13. Pre-implementation review (11 Nov 2026)

The specification and reference list were checked against the datasets' own papers before coding. These points were wrong or unsafe and are now fixed above; the Implementation Brief (v1.2) matches.

| # | Problem found | Fix | Section |
| --- | --- | --- | --- |
| 1 | A 512 crop with padding would pad GenImage real images only (ImageNet images are smaller than SD images): a learnable shortcut | 256 crops, no padding, tiled inference | 8 (M1) |
| 2 | GenImage real images are JPEG and generated ones PNG; detectors learn the format | Harmonization for all training data; results on raw and harmonized test files | 8 (M1), 10 |
| 3 | P1 compared a model trained on three datasets with baselines trained on SD v1.4 only | Separate detection-protocol model for P1 and P4 | 9, 10 |
| 4 | SID-Set validation was listed as cross-dataset although SID-Set train data is used | Moved to an in-domain benchmark (P3b); BR-Gen is the cross-dataset test | 10 |
| 5 | Held-out editors FLUX and HunyuanDiT: FLUX also generates SID-Set's synthetic images | Hold out Glide and Kolors; image-level hold-out rule for multi-edit images | 10 |
| 6 | Claims-Edit named SDXL, FLUX or Kandinsky 2.2 inpainting, all DiffSeg30k training editors | Kolors inpainting (held out), else BrushNet or PowerPaint | 10 |
| 7 | Mask target of all ones for fully generated images contradicts DiffSeg30k, where AI-generated backgrounds are labelled 0 | Mask marks edited pixels only | 8 (M4) |
| 8 | The alignment loss on Grad-CAM++ needs second-order gradients | Loss on the model's attention-pooling weights; Grad-CAM++ stays as the post-hoc map | 8 (M4, M5) |
| 9 | Hallucination rate was not defined for a model that cites no evidence IDs | Two rates: HR-E (against evidence) and HR-GT (against ground truth, all systems) | 10 |
| 10 | LoRA on `open_clip` attention layers is not directly supported by `peft` | CLIP loaded through `transformers` | 8 (M2), 9 |
| 11 | Validation data doubled as test data; epoch length undefined | 10% held out per training source; fixed samples per epoch | 9 |
| 12 | Target journals still listed Elsevier titles | Five Springer no-APC journals | 12 |
| 13 | Reference list lacked the works the method itself relies on | 19 references added (list v3, 83 in total) | References |

**Added 11 Nov 2026, while reviewing the Milestone 2 plan.** Item 2 above first said to crop and then re-compress every image. That would leave real JPEGs compressed twice and generated PNGs once. Section 8 (M1) now compresses only lossless-format images, once, before cropping, at the measured quality of the JPEG images in the same source.

**Added 12 Nov 2026, data storage.** The first download failed: the SD v1.4 archive (89.8 GB) cannot be unpacked on Colab's disk, DiffSeg30k is two zip files (not the format the first downloader assumed), and Drive had about 110 GB free. Rules from now on: keep source archives on Drive and unpack them on Colab's local disk each session; never unpack the whole SD v1.4 archive, read the 40,000 + 40,000 subset straight from it; BR-Gen (68.5 GB, test only) is not needed for CP1 and is handled before P3.

**Added 13 Nov 2026, after the data probe.** Three findings changed the data rules. (1) SID-Set's tampered images are almost all JPEG quality 75, so the harmonization rule now equalizes each source to its most compressed class (section 8, M1), with dropping SID-Set tampered images from training as the agreed fallback if CP1 still fails. (2) The public DiffSeg30k release has no real images, so matched real COCO negatives are added. (3) The shortcut check uses the crop the model sees, not the original image size.

## Sources

- [Insurers grapple with AI-generated images (SAS, May 2026)](https://www.sas.com/en_us/news/press-releases/2026/may/synthetic-images-ai-insurance-fraud.html)
- [India’s 2026 IT Rules on synthetic media (Mondaq)](https://webiis10.mondaq.com/india/social-media/1746136/indias-2026-it-rules-on-synthetic-media-a-structural-shift-in-intermediary-liability)
- [EU AI Act Article 50 (Mondaq)](https://www.mondaq.com/new-technology/1843358/eu-ai-act-article-50)
- [Durable Content Credentials and metadata stripping (SoftwareSeni)](https://softwareseni.com/durable-content-credentials-how-provenance-survives-metadata-stripping)
- [DF40: definition of deepfake in the broad sense](https://ar5iv.labs.arxiv.org/html/2406.13495)
- [FORGE (EMNLP 2026 Findings)](https://aclanthology.org/2026.findings-emnlp.2054/)
- [REVEAL (2026)](https://www.alphaxiv.org/abs/2511.23158)
- [Explainable Deepfake Detection Challenge, ACM MM 2026](https://www.alphaxiv.org/abs/2607.21007)
- [TriDF (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Jiang-Lin_TriDF_Evaluating_Perception_Detection_and_Hallucination_for_Interpretable_DeepFake_Detection_CVPR_2026_paper.pdf)
- [DiffSeg30k paper: editors, base images, hold-out results](https://arxiv.org/html/2511.19111v2)
- [Fake or JPEG? Biases in GenImage (Grommelt et al., ECCV 2024 Workshops)](https://arxiv.org/abs/2403.17608)
- [SID-Set dataset card](https://huggingface.co/datasets/saberzl/SID_Set) · [SIDA paper](https://arxiv.org/pdf/2412.04292)
- [BR-Gen and NFA-ViT paper](https://arxiv.org/html/2504.11922v4)
- Journal quartiles: the CS-IT subscription-route (no-APC) journal sheet prepared 27 October 2026; confirm each journal's current indexing before submission
- [The Visual Computer (Springer)](https://springer.com/journal/371) · [Applied Intelligence (Springer)](https://link.springer.com/journal/10489)
