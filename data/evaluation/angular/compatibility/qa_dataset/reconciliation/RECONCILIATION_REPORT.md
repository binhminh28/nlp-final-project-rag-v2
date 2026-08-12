# Dataset–Corpus Reconciliation Report

## Executive summary

- Questions reviewed: 46
- Evidence items affected: 55
- Evidence items unmapped to chunks: 43
- Unresolved evidence sentence occurrences: 51
- Compatibility failure rows: 65
- High-confidence proposals: 25
- Medium-confidence proposals: 28
- No-safe-correction evidence cases: 9
- Question semantic reviews required: 22

The exact source revision used to author the dataset is not recorded. The clustering of retired headings, punctuation drift, paraphrases, and absent source-exact statements is consistent with authorship against another documentation snapshot and/or non-source-exact curation.

## Resolved pipeline finding

One section resolver defect was found and fixed with a regression test: an exact full path was previously pooled with suffix matches. Exact full paths now win before suffix search. This legitimately changed compatibility from 92 to 94 questions and reduced the current failure queue from 68 to 65 rows.

## Root-cause classification

| Classification | Questions | Evidence items |
| --- | ---: | ---: |
| ambiguous_source_content | 1 | 1 |
| dataset_evidence_not_source_exact | 7 | 7 |
| dataset_evidence_paraphrase | 24 | 30 |
| dataset_from_different_corpus_version | 10 | 11 |
| dataset_section_path_error | 5 | 6 |

## Systematic patterns

Failure rows by compatibility root cause:

- ambiguous_section_mapping: 1 rows, 1 evidence items, 1 questions
- corpus_version_mismatch: 43 rows, 36 evidence items, 30 questions
- evidence_text_normalization_issue: 8 rows, 8 evidence items, 8 questions
- section_path_mismatch: 13 rows, 13 evidence items, 11 questions

Most affected documents:

- `angular:guide/i18n/manage-marked-text.md`: 4 affected evidence items
- `angular:guide/routing/loading-strategies.md`: 4 affected evidence items
- `angular:guide/forms/signals/validation.md`: 3 affected evidence items
- `angular:guide/forms/typed-forms.md`: 3 affected evidence items
- `angular:guide/hydration.md`: 3 affected evidence items
- `angular:guide/i18n/prepare.md`: 3 affected evidence items
- `angular:guide/routing/define-routes.md`: 3 affected evidence items
- `angular:guide/templates/ng-template.md`: 3 affected evidence items
- `angular:guide/testing/component-harnesses-overview.md`: 3 affected evidence items
- `angular:guide/forms/signals/comparison.md`: 2 affected evidence items

Repeated authored section paths:

- `Lazy loading`: 4 affected evidence items
- `Use a custom ID`: 4 affected evidence items

## Projected compatibility

- Current: 94 / 140
- PROJECTED — NOT ACTUAL GATE RESULT after all high-confidence proposals: 110 / 140
- Still requiring human review: 30

## Incompatible cross-document questions

- `q_hard_004`: 1 failing evidence item(s); documents resolved 2/2; evidence: q_hard_004_e01@angular:guide/incremental-hydration.md=resolved, q_hard_004_e02@angular:guide/hydration.md=resolved, q_hard_004_e03@angular:guide/hydration.md=failed; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
- `q_hard_019`: 1 failing evidence item(s); documents resolved 3/3; evidence: q_hard_019_e01@angular:guide/templates/ng-content.md=resolved, q_hard_019_e02@angular:guide/templates/ng-template.md=failed, q_hard_019_e03@angular:guide/templates/ng-container.md=resolved; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
- `q_hard_024`: 1 failing evidence item(s); documents resolved 2/2; evidence: q_hard_024_e01@angular:guide/testing/component-harnesses-overview.md=failed, q_hard_024_e02@angular:guide/testing/creating-component-harnesses.md=resolved; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
- `q_hard_029`: 2 failing evidence item(s); documents resolved 2/2; evidence: q_hard_029_e01@angular:guide/routing/testing.md=failed, q_hard_029_e02@angular:guide/routing/define-routes.md=failed; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
- `q_hard_031`: 1 failing evidence item(s); documents resolved 2/2; evidence: q_hard_031_e01@angular:guide/forms/signals/validation.md=failed, q_hard_031_e02@angular:guide/forms/form-validation.md=resolved; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
- `q_hard_036`: 1 failing evidence item(s); documents resolved 2/2; evidence: q_hard_036_e01@angular:guide/testing/component-harnesses-overview.md=failed, q_hard_036_e02@angular:guide/testing/using-component-harnesses.md=resolved; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
- `q_hard_039`: 1 failing evidence item(s); documents resolved 2/2; evidence: q_hard_039_e01@angular:ecosystem/service-workers/communications.md=resolved, q_hard_039_e02@angular:ecosystem/service-workers/communications.md=failed, q_hard_039_e03@angular:ecosystem/service-workers/overview.md=resolved; action: Review the ranked renamed/relocated canonical section path.
- `q_hard_040`: 1 failing evidence item(s); documents resolved 2/2; evidence: q_hard_040_e01@angular:guide/i18n/manage-marked-text.md=failed, q_hard_040_e02@angular:guide/i18n/merge.md=resolved, q_hard_040_e03@angular:guide/i18n/merge.md=resolved; action: Review the ranked renamed/relocated canonical section path.

## Per-question review queue

### q_easy_011

What syntax does Angular use to add event listeners to elements in templates?

- `q_easy_011_e01` — dataset_evidence_not_source_exact (high); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Replace authored punctuation/formatting drift with the exact canonical source text.
  - Canonical document: `angular:introduction/essentials/templates.md`
  - Sentence 0: authored `Angular lets you add event listeners to an element in your template with parentheses.`; top source candidate `Angular lets you add event listeners to an element in your template with parentheses:` (score 0.893529, block 20, section `['Templates', 'Handling user interaction']`)

### q_easy_019

How can you generate a code coverage report when running Angular unit tests?

- `q_easy_019_e01` — dataset_evidence_not_source_exact (high); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Replace authored punctuation/formatting drift with the exact canonical source text.
  - Canonical document: `angular:guide/testing/code-coverage.md`
  - Sentence 0: authored `To generate a coverage report, add the `--coverage` flag to the `ng test` command.`; top source candidate `To generate a coverage report, add the `--coverage` flag to the `ng test` command:` (score 0.892949, block 9, section `['Code coverage', 'Generating a report']`)

### q_easy_023

Why does a FormControl inferred from a string initial value normally include null in its type?

- `q_easy_023_e01` — dataset_evidence_not_source_exact (high); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Replace authored punctuation/formatting drift with the exact canonical source text.
  - Canonical document: `angular:guide/forms/typed-forms.md`
  - Sentence 0: authored `You might wonder: why does the type of this control include `null`? This is because the control can become `null` at any time, by calling reset.`; top source candidate `You might wonder: why does the type of this control include `null`? This is because the control can become `null` at any time, by calling reset:` (score 0.896071, block 22, section `['Typed Forms', '`FormControl`: Getting Started', 'Nullability']`)

### q_easy_053

What risk arises when source text is changed after assigning it a custom translation ID?

- `q_easy_053_e01` — dataset_section_path_error (high); failures: section_resolution; answer support: support_unclear; action: Review the ranked renamed/relocated canonical section path.
  - Canonical document: `angular:guide/i18n/manage-marked-text.md`
  - Authored path: `['Use a custom ID']`; top current path: `['Manage marked text with custom IDs']` (path score 0.361837, evidence hits 2)

### q_hard_035

How does Angular's default HttpClient XSRF defense use the browser same-origin model to distinguish a legitimate mutating request from a forged cross-site request?

- `q_hard_035_e03` — dataset_evidence_not_source_exact (high); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Replace authored punctuation/formatting drift with the exact canonical source text.
  - Canonical document: `angular:guide/security.md`
  - Sentence 1: authored `By default, an interceptor sends this header on all mutating requests such as `POST` to relative and same origin URLs, but not on `GET` or `HEAD` requests.`; top source candidate `By default, an interceptor sends this header on all mutating requests (such as `POST`) to relative and same origin URLs, but not on `GET` or `HEAD` requests.` (score 0.896333, block 92, section `['Security', 'HTTP-level vulnerabilities', '`HttpClient` XSRF/CSRF security']`)

### q_hard_040

How do custom translation IDs and locale-specific builds solve different i18n maintenance problems, and what risks must be managed for each?

- `q_hard_040_e01` — dataset_section_path_error (high); failures: section_resolution; answer support: support_unclear; action: Review the ranked renamed/relocated canonical section path.
  - Canonical document: `angular:guide/i18n/manage-marked-text.md`
  - Authored path: `['Use a custom ID']`; top current path: `['Manage marked text with custom IDs']` (path score 0.361837, evidence hits 2)

### q_medium_014

How do Signal Forms and Reactive Forms differ in how validation rules are organized?

- `q_medium_014_e01` — dataset_evidence_not_source_exact (high); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Replace authored punctuation/formatting drift with the exact canonical source text.
  - Canonical document: `angular:guide/forms/signals/comparison.md`
  - Sentence 0: authored `Signal Forms uses a schema function where you bind validators to field paths.`; top source candidate `Signal Forms uses a schema function where you bind validators to field paths:` (score 0.892857, block 24, section `['Comparison with other form approaches', 'Understanding the differences', 'How validation works']`)

### q_medium_020

What two pieces are required for nested route content to render and update inside a parent component?

- `q_medium_020_e01` — dataset_evidence_not_source_exact (high); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Replace authored punctuation/formatting drift with the exact canonical source text.
  - Canonical document: `angular:guide/routing/define-routes.md`
  - Sentence 0: authored `You can add child routes to any route definition with the `children` property.`; top source candidate `You can add child routes to any route definition with the `children` property:` (score 0.892763, block 81, section `['Define routes', 'Nested Routes']`)
- `q_medium_020_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/routing/define-routes.md`
  - Sentence 0: authored `To display child routes, the parent component includes its own `<router-outlet>`.`; top source candidate `To display child routes, the parent component (`Product` in the example above) includes its own `<router-outlet>`.` (score 0.749444, block 85, section `['Define routes', 'Nested Routes']`)

### q_medium_022

How do functional and DI-based HTTP interceptors differ according to Angular's recommendation?

- `q_medium_022_e02` — dataset_evidence_not_source_exact (high); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Replace authored punctuation/formatting drift with the exact canonical source text.
  - Canonical document: `angular:guide/http/interceptors.md`
  - Sentence 1: authored `A DI-based interceptor is an injectable class which implements the `HttpInterceptor` interface.`; top source candidate `A DI-based interceptor is an injectable class which implements the `HttpInterceptor` interface:` (score 0.894086, block 66, section `['Interceptors', 'DI-based interceptors']`)

### q_medium_034

What is both the benefit and the risk of using a custom translation ID?

- `q_medium_034_e01` — dataset_section_path_error (high); failures: section_resolution; answer support: support_unclear; action: Review the ranked renamed/relocated canonical section path.
  - Canonical document: `angular:guide/i18n/manage-marked-text.md`
  - Authored path: `['Use a custom ID']`; top current path: `['Manage marked text with custom IDs']` (path score 0.361837, evidence hits 2)
- `q_medium_034_e02` — dataset_section_path_error (high); failures: section_resolution; answer support: support_unclear; action: Review the ranked renamed/relocated canonical section path.
  - Canonical document: `angular:guide/i18n/manage-marked-text.md`
  - Authored path: `['Use a custom ID']`; top current path: `['Manage marked text with custom IDs']` (path score 0.361837, evidence hits 1)

### q_hard_004

Which DOM-related constraints from full hydration also apply to incremental hydration, and why can native DOM manipulation violate them?

- `q_hard_004_e03` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/hydration.md`
  - Sentence 1: authored `This mismatch will result in hydration failure and throw a DOM mismatch error.`; top source candidate `This mismatch will result in hydration failure and throw a DOM mismatch error ([see below](#errors)).` (score 0.835714, block 37, section `['Hydration', 'Constraints', 'Direct DOM Manipulation']`)

### q_hard_019

How do ng-content, ng-template, and ng-container differ in their roles even though none of them corresponds to a normal rendered wrapper element?

- `q_hard_019_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/templates/ng-template.md`
  - Sentence 0: authored `The `<ng-template>` element lets you declare a template fragment – a section of content that you can dynamically or programmatically render.`; top source candidate `Inspired by the [native `<template>` element](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/template), the `<ng-template>` element lets you declare a **template fragment** – a section of content that you can dynamically or programmatically render.` (score 0.882112, block 1, section `['Create template fragments with ng-template']`)

### q_hard_024

Why should a component harness for projected content expose a scoped HarnessLoader instead of forcing users to inspect the projected DOM directly?

- `q_hard_024_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/testing/component-harnesses-overview.md`
  - Sentence 0: authored `They make tests less brittle by insulating themselves against implementation details of a component, such as its DOM structure.`; top source candidate `- They make tests less brittle by insulating themselves against implementation details of a component, such as its DOM structure` (score 0.997826, block 3, section `['Component harnesses overview']`)

### q_hard_029

Why does testing a nested route require more assertions than testing a single routed component?

- `q_hard_029_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/routing/testing.md`
  - Sentence 1: authored `You need to verify that: The parent component renders properly. The child component renders within it. Ensure that both components can access their respective route data.`; top source candidate `1. The parent component renders properly.
2. The child component renders within it.
3. Ensure that both components can access their respective route data.` (score 0.919115, block 24, section `['Testing routing and navigation', 'Testing scenarios', 'Nested routes']`)
- `q_hard_029_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0], evidence_sentence_resolution[1]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/routing/define-routes.md`
  - Sentence 0: authored `You can add child routes to any route definition with the `children` property.`; top source candidate `You can add child routes to any route definition with the `children` property:` (score 0.892763, block 81, section `['Define routes', 'Nested Routes']`)
  - Sentence 1: authored `To display child routes, the parent component includes its own `<router-outlet>`.`; top source candidate `To display child routes, the parent component (`Product` in the example above) includes its own `<router-outlet>`.` (score 0.749444, block 85, section `['Define routes', 'Nested Routes']`)

### q_hard_031

How does validation timing differ between Signal Forms and an async validator configured with updateOn: 'blur' in traditional Angular forms?

- `q_hard_031_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0], evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/forms/signals/validation.md`
  - Sentence 0: authored `Synchronous validation - All synchronous validation rules run when value changes.`; top source candidate `1. **Synchronous validation** - All synchronous validation rules run when value changes` (score 0.996584, block 19, section `['Validation', 'Validation basics', 'Validation timing']`)
  - Sentence 1: authored `Asynchronous validation - Asynchronous validation rules run only after all synchronous validation rules pass.`; top source candidate `2. **Asynchronous validation** - Asynchronous validation rules run only after all synchronous validation rules pass` (score 0.997465, block 19, section `['Validation', 'Validation basics', 'Validation timing']`)

### q_hard_036

Why can the same Angular component harness implementation be useful for both unit and end-to-end testing, and what limitation still exists across environments?

- `q_hard_036_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0], evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/testing/component-harnesses-overview.md`
  - Sentence 0: authored `They make tests less brittle by insulating themselves against implementation details of a component, such as its DOM structure.`; top source candidate `- They make tests less brittle by insulating themselves against implementation details of a component, such as its DOM structure` (score 0.997826, block 3, section `['Component harnesses overview']`)
  - Sentence 1: authored `They can be used across multiple testing environments.`; top source candidate `- They can be used across multiple testing environments` (score 0.99486, block 3, section `['Component harnesses overview']`)

### q_hard_039

Why is reloading the page generally safer than calling activateUpdate() when a new Angular service-worker version is ready?

- `q_hard_039_e02` — dataset_section_path_error (medium); failures: section_resolution; answer support: support_unclear; action: Review the ranked renamed/relocated canonical section path.
  - Canonical document: `angular:ecosystem/service-workers/communications.md`
  - Authored path: `['Updating to the latest version', 'Safety of updating without reloading']`; top current path: `['Communicating with the Service Worker', '`SwUpdate` service', 'Updating to the latest version']` (path score 1.0, evidence hits 2)

### q_easy_005

What rendering modes can be configured for Angular server routes?

- `q_easy_005_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0], evidence_sentence_resolution[1], evidence_sentence_resolution[2]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/ssr.md`
  - Sentence 0: authored `Server (SSR) — Renders the application on the server for each request, sending a fully populated HTML page to the browser.`; top source candidate `| **Server (SSR)**    | Renders the application on the server for each request, sending a fully populated HTML page to the browser. |` (score 0.886694, block 20, section `['Server and hybrid rendering', 'Server routing', 'Rendering modes']`)
  - Sentence 1: authored `Client (CSR) — Renders the application in the browser. This is the default Angular behavior.`; top source candidate `| **Client (CSR)**    | Renders the application in the browser. This is the default Angular behavior.                               |` (score 0.882447, block 20, section `['Server and hybrid rendering', 'Server routing', 'Rendering modes']`)
  - Sentence 2: authored `Prerender (SSG) — Prerenders the application at build time, generating static HTML files for each route.`; top source candidate `| **Prerender (SSG)** | Prerenders the application at build time, generating static HTML files for each route.                      |` (score 0.884434, block 20, section `['Server and hybrid rendering', 'Server routing', 'Rendering modes']`)

### q_easy_020

What is a component harness in Angular testing?

- `q_easy_020_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/testing/component-harnesses-overview.md`
  - Sentence 0: authored `A component harness is a class that allows tests to interact with components the way an end user does via a supported API.`; top source candidate `A <strong>component harness</strong> is a class that allows tests to interact with components the way an end user does via a supported API.` (score 0.849593, block 1, section `['Component harnesses overview']`)

### q_easy_028

What do loadComponent and loadChildren return through their loader functions?

- `q_easy_028_e01` — dataset_from_different_corpus_version (none); failures: section_resolution; answer support: support_unclear; action: Confirm the source snapshot or manually locate the retired section.
  - Canonical document: `angular:guide/routing/loading-strategies.md`
  - Authored path: `['Lazy loading']`; top current path: `['Route Loading Strategies']` (path score 0.384444, evidence hits 1)

### q_easy_029

Why can lazy loading routes improve an Angular application's initial load speed?

- `q_easy_029_e01` — dataset_from_different_corpus_version (none); failures: section_resolution; answer support: support_unclear; action: Confirm the source snapshot or manually locate the retired section.
  - Canonical document: `angular:guide/routing/loading-strategies.md`
  - Authored path: `['Lazy loading']`; top current path: `['Route Loading Strategies']` (path score 0.384444, evidence hits 1)

### q_easy_033

What does RouterOutlet mark in an Angular application?

- `q_easy_033_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/routing/router-reference.md`
  - Sentence 0: authored ``RouterOutlet` | The directive (`<router-outlet>`) that marks where the router displays a view.`; top source candidate `| `RouterOutlet`        | The directive \(`<router-outlet>`\) that marks where the router displays a view.                                                                                                                                                          |` (score 0.882447, block 8, section `['Router reference', 'Router terminology']`)

### q_easy_050

What is used to mark text strings in Angular component code for translation?

- `q_easy_050_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/i18n/prepare.md`
  - Sentence 0: authored `Use the `$localize` tagged message string to mark text strings in component code.`; top source candidate `- Use the `$localize` tagged message string to mark text strings in component code` (score 0.996497, block 2, section `['Prepare component for translation']`)

### q_easy_051

Which extract-i18n option sets the translation output file format?

- `q_easy_051_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/i18n/translation-files.md`
  - Sentence 0: authored ``--format` | Set the format of the output file`; top source candidate `| `--format`      | Set the format of the output file    |` (score 0.976087, block 11, section `['Work with translation files', 'Extract the source language file']`)

### q_hard_001

How do CSR, SSG, and SSR differ in when HTML is generated, and what happens after hydration for SSG and SSR applications?

- `q_hard_001_e03` — dataset_from_different_corpus_version (none); failures: evidence_sentence_resolution[0], evidence_sentence_resolution[1]; answer support: support_unclear; action: Identify the dataset's source revision or re-author evidence from the canonical corpus.
  - Canonical document: `angular:guide/routing/rendering-strategies.md`
  - Sentence 0: authored `SSR generates HTML on the server for the initial request for a route.`; top source candidate `**SSR generates HTML on the server for the initial request for a route**, providing dynamic content with good SEO.` (score 0.708397, block 26, section `['Rendering strategies in Angular', 'Server-Side Rendering (SSR)']`)
  - Sentence 1: authored `Once the client renders the page, Angular hydrates the app and it then runs entirely in the browser like a traditional SPA.`; top source candidate `Once the client renders the page, Angular [hydrates](/guide/hydration#what-is-hydration) the app and it then runs entirely in the browser like a traditional SPA - subsequent navigation, route changes, and API calls all happen client-side without additional server rendering.` (score 0.637897, block 27, section `['Rendering strategies in Angular', 'Server-Side Rendering (SSR)']`)

### q_hard_007

Why can SSR handle personalized dynamic pages that build-time prerendering cannot, and what infrastructure cost does SSR introduce?

- `q_hard_007_e01` — dataset_section_path_error (medium); failures: section_resolution; answer support: support_unclear; action: Review the ranked renamed/relocated canonical section path.
  - Canonical document: `angular:guide/ssr.md`
  - Authored path: `['Server-side rendering']`; top current path: `['Server and hybrid rendering', 'Server routing', 'Rendering modes', 'Choosing a rendering mode', 'Server-side rendering (SSR)']` (path score 0.88125, evidence hits 1)

### q_hard_008

Why is ngSkipHydration considered a temporary workaround rather than a preferred fix for components that manipulate the DOM directly?

- `q_hard_008_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/hydration.md`
  - Sentence 0: authored `This mismatch will result in hydration failure and throw a DOM mismatch error.`; top source candidate `This mismatch will result in hydration failure and throw a DOM mismatch error ([see below](#errors)).` (score 0.835714, block 37, section `['Hydration', 'Constraints', 'Direct DOM Manipulation']`)

### q_hard_011

How can linkedSignal preserve a manually selected value when its source options change, instead of always resetting to the first new option?

- `q_hard_011_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/signals/linked-signal.md`
  - Sentence 1: authored `To accomplish this, you can create a `linkedSignal` with a separate source and computation.`; top source candidate `To accomplish this, you can create a `linkedSignal` with a separate _source_ and _computation_:` (score 0.89382, block 11, section `['Dependent state with `linkedSignal`', 'Accounting for previous state']`)

### q_hard_012

Why is resource chain preferable to directly reading an upstream resource value in downstream params when the downstream resource performs asynchronous work?

- `q_hard_012_e01` — dataset_from_different_corpus_version (none); failures: evidence_sentence_resolution[0]; answer support: support_unclear; action: Identify the dataset's source revision or re-author evidence from the canonical corpus.
  - Canonical document: `angular:guide/signals/resource.md`
  - Sentence 0: authored ``chain(userResource)` reads the value of `userResource` and automatically propagates its status to `companyResource`.`; top source candidate `Here `companyResource` depends on the user's `companyId`, which is only known once `userResource` has loaded. `chain(userResource)` reads the value of `userResource` and automatically propagates its status to `companyResource`:` (score 0.593893, block 42, section `['Async reactivity with resources', 'Chaining resources']`)

### q_hard_014

What happens to an in-flight Resource load when its params change, and how can a loader cooperate with that cancellation?

- `q_hard_014_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/signals/resource.md`
  - Sentence 1: authored `The native `fetch` function accepts an `AbortSignal`.`; top source candidate `For example, the native `fetch` function accepts an `AbortSignal`:` (score 0.781926, block 21, section `['Async reactivity with resources', 'Resource loaders', 'Aborting requests']`)

### q_hard_020

Why does projected content still use the parent's change-detection and dependency-injection context even though it appears visually inside the receiving component?

- `q_hard_020_e01` — dataset_from_different_corpus_version (none); failures: evidence_sentence_resolution[1]; answer support: support_unclear; action: Identify the dataset's source revision or re-author evidence from the canonical corpus.
  - Canonical document: `angular:guide/components/content-projection.md`
  - Sentence 1: authored `Angular tracks it as part of the parent's view.`; top source candidate `Angular tracks it as part of the parent's view, which has a couple of side effects worth knowing about.` (score 0.578, block 39, section `['Content projection with ng-content', 'Caveats', "Projected content lives in the parent's view"]`)

### q_hard_021

How does Angular transform structural-directive shorthand such as *select into ng-template form, and why can only one shorthand structural directive appear on one element?

- `q_hard_021_e02` — dataset_from_different_corpus_version (none); failures: evidence_sentence_resolution[2]; answer support: support_unclear; action: Identify the dataset's source revision or re-author evidence from the canonical corpus.
  - Canonical document: `angular:guide/directives/structural-directives.md`
  - Sentence 2: authored ``<ng-container>` can be used to create wrapper layers when multiple structural directives need to be applied.`; top source candidate `Multiple directives would require multiple nested `<ng-template>`, and it's unclear which directive should be first. `<ng-container>` can be used to create wrapper layers when multiple structural directives need to be applied around the same physical DOM element or component, which allows the user to define the nested structure.` (score 0.446824, block 23, section `['Structural directives', 'One structural directive per element']`)

### q_hard_022

When an ng-template fragment is rendered somewhere else, which injection context does it normally use, and how can NgTemplateOutlet change that behavior?

- `q_hard_022_e02` — dataset_from_different_corpus_version (none); failures: section_resolution; answer support: support_unclear; action: Confirm the source snapshot or manually locate the retired section.
  - Canonical document: `angular:guide/templates/ng-template.md`
  - Authored path: `['Providing injectors to template fragments', "Inheriting the outlet's injector"]`; top current path: `['Create template fragments with ng-template', 'Providing injectors to template fragments']` (path score 1.0, evidence hits 1)

### q_hard_027

Why can deeply nested lazy routes hurt performance even though lazy loading improves the initial load?

- `q_hard_027_e01` — dataset_from_different_corpus_version (none); failures: evidence_sentence_resolution[1], section_resolution; answer support: support_likely_unchanged; action: Confirm the source snapshot or manually locate the retired section.
  - Canonical document: `angular:guide/routing/loading-strategies.md`
  - Authored path: `['Lazy loading']`; top current path: `['Route Loading Strategies']` (path score 0.384444, evidence hits 1)
  - Sentence 1: authored `These portions of your code compile into separate JavaScript chunks that the router requests only when the user visits the corresponding route.`; top source candidate `These portions of your code compile into separate JavaScript "chunks" that the router requests only when the user visits the corresponding route.` (score 0.896181, block 14, section `['Route Loading Strategies', 'Lazily loaded components and routes']`)

### q_hard_032

Describe the complete state transition of a traditional asynchronous form validator from invocation until the final validity result becomes available.

- `q_hard_032_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/forms/form-validation.md`
  - Sentence 1: authored `The `isRoleTaken()` method dispatches an HTTP request that checks if the role is available.`; top source candidate `The `isRoleTaken()` method dispatches an HTTP request that checks if the role is available, and returns `Observable<boolean>` as the result.` (score 0.715111, block 97, section `['Validating form input', 'Creating asynchronous validators', 'Implementing a custom async validator']`)

### q_hard_037

Why should a harness author expose task-oriented methods such as toggle() or isOpen() instead of exposing internal TestElement objects?

- `q_hard_037_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/testing/creating-component-harnesses.md`
  - Sentence 0: authored ``TestElement` is an abstraction designed to work across different test environments.`; top source candidate ``TestElement` is an abstraction designed to work across different test environments (Unit tests, WebDriver, etc).` (score 0.763511, block 23, section `['Creating harnesses for your components', 'Working with `TestElement` instances']`)
- `q_hard_037_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/testing/creating-component-harnesses.md`
  - Sentence 0: authored `Do not expose `TestElement` instances to harness users unless it's an element the component consumer defines directly.`; top source candidate `Do not expose `TestElement` instances to harness users unless it's an element the component consumer defines directly, such as the component's host element.` (score 0.765616, block 25, section `['Creating harnesses for your components', 'Working with `TestElement` instances']`)

### q_hard_038

Why does testing an overlay component require a different harness loader from testing a child inside the fixture root?

- `q_hard_038_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/testing/using-component-harnesses.md`
  - Sentence 1: authored `Code that displays a floating element or pop-up often attaches DOM elements directly to the document body.`; top source candidate `For example, code that displays a floating element or pop-up often attaches DOM elements directly to the document body, such as the `Overlay` service in Angular CDK.` (score 0.70737, block 15, section `['Using component harnesses in tests', 'Test harness environments and loaders', 'Using the loader from `TestbedHarnessEnvironment` for unit tests']`)
- `q_hard_038_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/testing/using-component-harnesses.md`
  - Sentence 1: authored `The dialog is appended to `document.body`, outside of the fixture's root element, so we use `rootLoader` in this case.`; top source candidate `// The dialog is appended to `document.body`, outside of the fixture's root element,` (score 0.723367, block 30, section `['Using component harnesses in tests', 'Using a harness loader']`)

### q_medium_002

Why can direct DOM manipulation cause Angular hydration to fail?

- `q_medium_002_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/hydration.md`
  - Sentence 1: authored `This mismatch will result in hydration failure and throw a DOM mismatch error.`; top source candidate `This mismatch will result in hydration failure and throw a DOM mismatch error ([see below](#errors)).` (score 0.835714, block 37, section `['Hydration', 'Constraints', 'Direct DOM Manipulation']`)

### q_medium_006

In Signal Forms, in what order do synchronous validation, asynchronous validation, and field-state updates occur?

- `q_medium_006_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0], evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/forms/signals/validation.md`
  - Sentence 0: authored `Synchronous validation - All synchronous validation rules run when value changes.`; top source candidate `1. **Synchronous validation** - All synchronous validation rules run when value changes` (score 0.996584, block 19, section `['Validation', 'Validation basics', 'Validation timing']`)
  - Sentence 1: authored `Asynchronous validation - Asynchronous validation rules run only after all synchronous validation rules pass.`; top source candidate `2. **Asynchronous validation** - Asynchronous validation rules run only after all synchronous validation rules pass` (score 0.997465, block 19, section `['Validation', 'Validation basics', 'Validation timing']`)
- `q_medium_006_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/forms/signals/validation.md`
  - Sentence 0: authored `Field state updates - The `valid()`, `invalid()`, `errors()`, and `pending()` signals update.`; top source candidate `3. **Field state updates** - The `valid()`, `invalid()`, `errors()`, and `pending()` signals update` (score 0.996746, block 19, section `['Validation', 'Validation basics', 'Validation timing']`)

### q_medium_012

How does the nonNullable option change both the type-related expectation and reset behavior of a FormControl?

- `q_medium_012_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/forms/typed-forms.md`
  - Sentence 0: authored `This is because the control can become `null` at any time, by calling reset.`; top source candidate `This is because the control can become `null` at any time, by calling reset:` (score 0.892568, block 22, section `['Typed Forms', '`FormControl`: Getting Started', 'Nullability']`)
- `q_medium_012_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/forms/typed-forms.md`
  - Sentence 1: authored `This will cause the control to reset to its initial value, instead of `null`.`; top source candidate `This will cause the control to reset to its initial value, instead of `null`:` (score 0.892667, block 24, section `['Typed Forms', '`FormControl`: Getting Started', 'Nullability']`)

### q_medium_013

How do Signal Forms and Reactive Forms differ in where they store form data?

- `q_medium_013_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/forms/signals/comparison.md`
  - Sentence 1: authored `You access values through the form hierarchy.`; top source candidate `You access values through the form hierarchy:` (score 0.887778, block 16, section `['Comparison with other form approaches', 'Understanding the differences', 'Where your form data lives']`)

### q_medium_018

What is the main performance benefit and the main future cost of lazy-loading Angular routes?

- `q_medium_018_e01` — dataset_from_different_corpus_version (none); failures: section_resolution; answer support: support_unclear; action: Confirm the source snapshot or manually locate the retired section.
  - Canonical document: `angular:guide/routing/loading-strategies.md`
  - Authored path: `['Lazy loading']`; top current path: `['Route Loading Strategies']` (path score 0.384444, evidence hits 1)

### q_medium_026

When would keepalive and request priority serve different purposes in HttpClient fetch requests?

- `q_medium_026_e01` — dataset_from_different_corpus_version (none); failures: section_resolution; answer support: support_unclear; action: Confirm the source snapshot or manually locate the retired section.
  - Canonical document: `angular:guide/http/making-requests.md`
  - Authored path: `['Advanced fetch options', 'Keep-alive connections']`; top current path: `['Making HTTP requests', 'Advanced fetch options']` (path score 1.0, evidence hits 2)
- `q_medium_026_e02` — dataset_from_different_corpus_version (none); failures: evidence_sentence_resolution[1], section_resolution; answer support: support_unclear; action: Confirm the source snapshot or manually locate the retired section.
  - Canonical document: `angular:guide/http/making-requests.md`
  - Authored path: `['Advanced fetch options', 'Request priority for Core Web Vitals']`; top current path: `['Making HTTP requests', 'Advanced fetch options']` (path score 1.0, evidence hits 1)
  - Sentence 1: authored `Use `priority: 'high'` for requests that affect Largest Contentful Paint (LCP).`; top source candidate `TIP: Use `priority: 'high'` for requests that affect Largest Contentful Paint (LCP) and `priority: 'low'` for requests that don't impact initial user experience.` (score 0.581966, block 86, section `['Making HTTP requests', 'Advanced fetch options', 'Fetch options', 'Request priority for Core Web Vitals']`)

### q_medium_032

How are internationalization and localization distinguished in Angular's i18n documentation?

- `q_medium_032_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[1]; answer support: support_unclear; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/i18n/overview.md`
  - Sentence 1: authored `The localization process includes extracting text for translation into different languages and formatting data for a specific locale.`; top source candidate `- Extract text for translation into different languages
- Format data for a specific locale` (score 0.683387, block 2, section `['Angular Internationalization (i18n)']`)

### q_medium_033

How does Angular mark translatable text differently in template content, template attributes, and component code?

- `q_medium_033_e01` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0], evidence_sentence_resolution[1]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/i18n/prepare.md`
  - Sentence 0: authored `Use the `i18n` attribute to mark text in component templates.`; top source candidate `- Use the `i18n` attribute to mark text in component templates` (score 0.995299, block 2, section `['Prepare component for translation']`)
  - Sentence 1: authored `Use the `i18n-` attribute to mark attribute text strings in component templates.`; top source candidate `- Use the `i18n-` attribute to mark attribute text strings in component templates` (score 0.996452, block 2, section `['Prepare component for translation']`)
- `q_medium_033_e02` — dataset_evidence_paraphrase (medium); failures: evidence_sentence_resolution[0]; answer support: support_likely_unchanged; action: Compare the authored statement with the lexical source candidate and verify semantics manually.
  - Canonical document: `angular:guide/i18n/prepare.md`
  - Sentence 0: authored `Use the `$localize` tagged message string to mark text strings in component code.`; top source candidate `- Use the `$localize` tagged message string to mark text strings in component code` (score 0.996497, block 2, section `['Prepare component for translation']`)

### q_hard_023

What are the two primary ways to render a TemplateRef, and how do they differ in where the rendering API is invoked?

- `q_hard_023_e02` — ambiguous_source_content (none); failures: section_resolution; answer support: support_unclear; action: Review every duplicate canonical section and select the intended full path.
  - Canonical document: `angular:guide/templates/ng-template.md`
  - Authored path: `['Using NgTemplateOutlet']`; top current path: `['Create template fragments with ng-template', 'Rendering a template fragment', 'Using `NgTemplateOutlet`']` (path score 1.0, evidence hits 1)

## Safety

Every proposal requires human review and has `auto_apply: false`. No dataset, compatibility gate, retrieval behavior, or benchmark artifact was modified by this reconciliation run.

BENCHMARK GATE: FAIL
