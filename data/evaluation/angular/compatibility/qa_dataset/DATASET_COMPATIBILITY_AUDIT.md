# Dataset Compatibility Audit

- Dataset: `data/evaluation/angular/qa_dataset.jsonl`
- Fingerprint: `cea4e11d87a353b084b73f95b231bf7c41a1a2c651312295ffe5bcb2a0bd03b6`
- Questions: 140
- Evidence items / sentences: 233 / 349
- Schema-valid questions: 140
- Compatible questions: 92
- Gate: **FAIL**

## Provenance resolution

- Documents: {'resolved': 233, 'unresolved': 0, 'ambiguous': 0, 'invalid': 0}
- Sections: {'resolved': 216, 'unresolved': 13, 'ambiguous': 4, 'invalid': 0}
- Sentences: {'exact': 205, 'normalized_exact': 93, 'unresolved': 51, 'ambiguous': 0, 'invalid': 0}

## Evidence-to-chunk mapping

| Strategy | Compatible questions | Evidence mapped | Evidence unmapped |
| --- | ---: | ---: | ---: |
| fixed_size | 92 | 190 | 43 |
| structure_aware | 92 | 190 | 43 |
| prompt_based | 92 | 190 | 43 |

## Gate reasons

- one or more section paths do not resolve uniquely
- one or more evidence sentences do not resolve uniquely
- one or more evidence items do not map to every chunk strategy

## Failure root causes

- ambiguous_section_mapping: 4
- corpus_version_mismatch: 43
- evidence_text_normalization_issue: 8
- section_path_mismatch: 13

## Compatibility by evidence scope

- cross_document: 4 / 12
- cross_section: 12 / 29
- single_section: 76 / 99

This is an offline compatibility result, not a retrieval or answer benchmark.
