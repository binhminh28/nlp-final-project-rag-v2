# Dataset Compatibility Audit

- Dataset: `data/evaluation/angular/qa_dataset.jsonl`
- Fingerprint: `9866799ea8f87c6a7c118cbaf0d8c298524757bd753db5d640e1a32234206d74`
- Questions: 140
- Evidence items / sentences: 233 / 353
- Schema-valid questions: 140
- Compatible questions: 140
- Gate: **PASS**

## Provenance resolution

- Documents: {'resolved': 233, 'unresolved': 0, 'ambiguous': 0, 'invalid': 0}
- Sections: {'resolved': 233, 'unresolved': 0, 'ambiguous': 0, 'invalid': 0}
- Sentences: {'exact': 272, 'normalized_exact': 81, 'unresolved': 0, 'ambiguous': 0, 'invalid': 0}

## Evidence-to-chunk mapping

| Strategy | Compatible questions | Evidence mapped | Evidence unmapped |
| --- | ---: | ---: | ---: |
| fixed_size | 140 | 233 | 0 |
| structure_aware | 140 | 233 | 0 |
| prompt_based | 140 | 233 | 0 |

## Gate reasons

- all strict dataset, provenance, and chunk coverage checks passed

## Failure root causes


## Compatibility by evidence scope

- cross_document: 12 / 12
- cross_section: 29 / 29
- single_section: 99 / 99

This is an offline compatibility result, not a retrieval or answer benchmark.
