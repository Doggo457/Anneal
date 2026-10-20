# Reworded-copy check (2026-10-20 13:38:28)

Embedder `embed-gemma:300m` at http://127.0.0.1:52625 (tag `flm0.9.43-embedgemma300m`), k = 5. Calibration: lm-sys/llm-decontaminator@931834aa0990d9558a06c5d34daf4ad3852a72ae, HumanEval prompts against their GPT-4 rewordings.

Self-check: AUC 0.9994, top-1 retrieval 0.9726, reworded median cosine 0.7225 against unrelated 0.078, same text twice 0.99978. t_pos (5th percentile of reworded pairs) = 0.5694.

| Test set | Problems | t_neg | Cut-off | Recall on rewordings | Problems with a flagged row |
| --- | --- | --- | --- | --- | --- |
| humaneval+ | 164 | 0.7353 | 0.7353 | 43.8% | 3 |
| mbpp+ | 378 | 0.8625 | 0.8625 | 2.1% | 0 |
| gate-cc | 195 | 0.8631 | 0.8631 | 2.1% | 3 |
| gate-oci | 300 | 0.8165 | 0.8165 | 8.2% | 16 |
| mmlu-pro | 100 | 0.536 | 0.5694 | 94.5% | 5 |

Test problems with a flagged training row (report EvalPlus with and without these):
- humaneval+: HumanEval/25, HumanEval/51, HumanEval/59
- gate-cc: cc/test/63, cc/test/75, cc/valid/15
- gate-oci: oci/0126fce66dadb1edee99c5c4b2aa3261, oci/28c329d1621eddf83c0f3ba69323d71a, oci/2dee880a2d6407b7be6db1fff0bd3604, oci/519d9c421c764dc6dcdc85b1a18b6238, oci/5dade6f55cd12befce35663decd09aba, oci/6ecef340a5c56d91c85bc28b8b0f3c6b, oci/727a188eb4b00e0fa89c9562e9fd7c78, oci/ae703f01ebdb9aa92886ec5b55d25450, oci/b037ee92f282ad19a2759221a8c78771, oci/b1819e4241cc6033310c851b93b3bf33, oci/c5654306060e5962369c153723f40d75, oci/e29f241b1fabafb75018920e0f575602, oci/ec527aff9774629a8c826fb7ddb69a60, oci/ee3fcdebfbf9a0182e4c1ad57364f8c5, oci/f4afb8075d11a3f647e91b8cbc619caf, oci/fddbacb171630cb36aba671de8d0850b
- mmlu-pro: mmlupro/12178, mmlupro/7710, mmlupro/7756, mmlupro/8974, mmlupro/9314

| Training source | Rows checked | Rows flagged |
| --- | --- | --- |
| mix_first_2k | 2000 | 30 |
| ocr_first | 1000 | 1 |
| ocr_first_b | 505 | 2 |
| self_oci_r1 | 512 | 27 |
| loop/banaxi_v4pro | 915 | 2 |
| loop/agent_sessions | 4 | 0 |
| loop/math_v4_aops | 9800 | 31 |
| loop/tachibana4_v4pro | 4561 | 0 |

47 row keys blocked for the loop. Embedding speed: 0.661 texts/s (14784 texts in 22358.3 s). Lowest free RAM seen: 7.01 GB. Took 372.9 min.

Training speed A/B (results\first-run\train\metrics.jsonl): 99.15 tok/s over 12 steps before, 99.42 tok/s over 63 steps during (-0.27% slower).
