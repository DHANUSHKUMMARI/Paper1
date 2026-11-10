# Verification checkpoints (CP0–CP10)

Criteria are defined in section 9.1 of the Research Specification. The day for each checkpoint is in the 15-day execution plan.

Claude Code: after each checkpoint, fill in its entry below. Record what was run, the exact numbers, the run folder or commit, and PASS or FAIL against the criteria. Do not start the next stage while a checkpoint is FAIL. Dhanush verifies each entry and initials it.

| CP | Day | Stage | Status | Verified by |
|----|-----|-------|--------|-------------|
| CP0 | 1 | Environment | pending | |
| CP1 | 2 | Data integrity and leakage | pending | |
| CP2 | 4 | Model build | pending | |
| CP3 | 5 | Training health | pending | |
| CP4 | 6 | Generalization gate | pending | |
| CP5 | 7 | Evidence faithfulness | pending | |
| CP6 | 8 | Grounded explanations | pending | |
| CP7 | 10 | Full benchmark | pending | |
| CP8 | 11 | Claims-Edit case study | pending | |
| CP9 | 14 | Ablations, statistics, human study | pending | |
| CP10 | 15 | Manuscript integrity | pending | |

## Entry template (copy once per checkpoint)

### CPn — <stage> — <date>

- **Run / commit:** `runs/...` · `git <short sha>`
- **What was run:** commands or notebook cells
- **Numbers:** each pass criterion with its measured value
- **Result:** PASS | FAIL
- **If FAIL:** fix applied, re-run result, or fallback taken (and why)
- **Verified by Dhanush:** ☐
