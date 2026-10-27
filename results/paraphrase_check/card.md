# Reworded-copy check (2026-10-27 02:42:36)

Embedder `embed-gemma:300m` at http://127.0.0.1:52625 (tag `flm0.9.43-embedgemma300m`), k = 5. Calibration: lm-sys/llm-decontaminator@931834aa0990d9558a06c5d34daf4ad3852a72ae, HumanEval prompts against their GPT-4 rewordings.

Self-check: AUC 0.9994, top-1 retrieval 0.9726, reworded median cosine 0.7225 against unrelated 0.078, same text twice 0.99978. t_pos (5th percentile of reworded pairs) = 0.5694.

| Test set | Problems | t_neg | Cut-off | Recall on rewordings | Problems with a flagged row |
| --- | --- | --- | --- | --- | --- |
| humaneval+ | 164 | 0.7353 | 0.7353 | 43.8% | 3 |
| mbpp+ | 378 | 0.8625 | 0.8625 | 2.1% | 0 |
| gate | 173 | 0.8351 | 0.8351 | 5.5% | 5 |
| mmlu-pro | 100 | 0.536 | 0.5694 | 94.5% | 5 |

t_neg counts easy and hard versions of one Codeforces problem as one problem (problems with a version: gate 12).

Test problems with a flagged training row (report EvalPlus with and without these):
- humaneval+: HumanEval/25, HumanEval/51, HumanEval/59
- gate: cc/test/46, cc/test/63, cc/test/75, cc/test/87, cc/valid/15
- mmlu-pro: mmlupro/12178, mmlupro/7710, mmlupro/7756, mmlupro/8974, mmlupro/9314

| Training source | Rows checked | Rows flagged |
| --- | --- | --- |
| mix_first_2k | 2000 | 10 |
| ocr_first | 1000 | 3 |
| ocr_first_b | 505 | 2 |
| self_oci_r1 | 512 | 5 |
| loop/banaxi_v4pro | 915 | 2 |
| loop/banaxi_v4pro_nopy_12k | 686 | 2 |
| loop/glm53_all_12k | 28070 | 0 |
| loop/glm53_all_nopy_12k | 24689 | 0 |
| loop/math_v4_aops | 9800 | 31 |
| loop/openswe_v4flash_12k | 26266 | 0 |
| loop/tachibana4_v4pro | 4561 | 0 |
| loop/tachibana4_v4pro_nopy_12k | 8072 | 0 |

27 row keys blocked for the loop. Embedding speed: 2.355 texts/s (8 texts in 3.4 s). Lowest free RAM seen: 12.44 GB. Took 0.3 min.
