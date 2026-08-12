# Canonical Benchmark V2 Freeze Validation

## A. Canonical identity

- `dataset_fingerprint`: `9866799ea8f87c6a7c118cbaf0d8c298524757bd753db5d640e1a32234206d74`
- `generation_fingerprint`: `c4f4768ec9b80361dfd0a1e252f74ff348aa4e4c953bcca02761ba345f38b301`
- `evaluation_fingerprint`: `c6867bffbd9775d3ef9b4ce666ae09f1995a6ccb7a7ef14858bbcdb736c1fa55`
- `retrieval_benchmark_fingerprint`: `9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7`
- `answer_benchmark_fingerprint`: `375983ff4b3c4e84b303d7298c4dd93b44782430bbc1fd6dff41db6f3b60af23`

Artifact roots: `data/benchmark/angular/canonical_v2/` and the exact retrieval-fingerprint directory under `data/retrieval/angular/canonical_production_v2/`.

## B. Completeness

| Strategy | Dataset Qs | Requested | Answers | Eval rows | Retrieval rows | Duplicate IDs | Missing IDs | Invalid rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_size | 140 | 140 | 140 | 140 | 140 | 0 | 0 | 0 |
| structure_aware | 140 | 140 | 140 | 140 | 140 | 0 | 0 | 0 |
| prompt_based | 140 | 140 | 140 | 140 | 140 | 0 | 0 | 0 |

Total: 140 unique questions, 420 answers, 420 evaluation rows, and 420 canonical-protocol retrieval rows. PASS.

## C. Paired alignment

All three strategies exactly match the canonical 140-ID universe. Question, gold answer, document, difficulty, question type, and available evidence/retrieval metadata have zero mismatches. PASS.

## D. Generation integrity

420/420 answers are non-empty successful results with `finish_reason=stop`; 0 length completions, failures, retries, or integrity defects. Provider diagnostics reconcile as 416 calls plus 4 same-fingerprint `fixed_size` cache hits. PASS.

## E. Answer metric reproduction

| Strategy | Precision | Recall | Token F1 | Exact | Containment |
|---|---:|---:|---:|---:|---:|
| fixed_size | 0.2858 | 0.7420 | 0.3830 | 0.0071 | 0.0143 |
| structure_aware | 0.2798 | 0.7398 | 0.3766 | 0.0071 | 0.0071 |
| prompt_based | 0.2707 | 0.7555 | 0.3721 | 0.0000 | 0.0143 |

Every per-query score was recomputed from generated and gold text, then macro-aggregated and matched to stored raw aggregates and four-decimal published values. PASS.

## F. Retrieval metric reproduction

| Strategy | Hit@1 | Hit@5 | Hit@10 | MRR | Recall@10 | Evidence coverage | All evidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_size | 0.8071 | 0.9643 | 0.9643 | 0.8732 | 0.9429 | 0.6905 | 0.5714 |
| structure_aware | 0.7786 | 0.9786 | 1.0000 | 0.8692 | 0.9857 | 0.7476 | 0.6357 |
| prompt_based | 0.7571 | 0.9786 | 0.9857 | 0.8508 | 0.9786 | 0.7320 | 0.6143 |

Recomputed offline from committed `same_token_budget` rows with candidate k=50, 2,048-token budget, and n=140 per strategy. PASS.

## G. Stratified reproduction

### Difficulty

| Difficulty | n | fixed_size | structure_aware | prompt_based |
|---|---:|---:|---:|---:|
| easy | 60 | 0.3892 | 0.3831 | 0.3723 |
| medium | 40 | 0.3863 | 0.3862 | 0.3939 |
| hard | 40 | 0.3702 | 0.3571 | 0.3501 |

### Question type

| Question type | n | fixed_size | structure_aware | prompt_based |
|---|---:|---:|---:|---:|
| behavior | 22 | 0.3823 | 0.3770 | 0.3682 |
| cause_effect | 2 | 0.2993 | 0.2976 | 0.3085 |
| comparison | 26 | 0.3333 | 0.3493 | 0.3400 |
| definition | 15 | 0.3152 | 0.3085 | 0.2558 |
| fact | 21 | 0.4836 | 0.4564 | 0.5103 |
| list | 4 | 0.6402 | 0.6101 | 0.4812 |
| mechanism | 7 | 0.3882 | 0.3765 | 0.3547 |
| procedure | 11 | 0.3672 | 0.3756 | 0.3842 |
| security_mechanism | 1 | 0.4649 | 0.4521 | 0.4961 |
| sequence | 3 | 0.4080 | 0.3608 | 0.2929 |
| syntax | 1 | 0.3000 | 0.3077 | 0.3200 |
| synthesis | 2 | 0.3460 | 0.3038 | 0.3202 |
| tradeoff | 10 | 0.3758 | 0.3839 | 0.3888 |
| why | 15 | 0.3532 | 0.3361 | 0.3419 |

Membership and every published Token F1 stratum reproduce. This is a reproducibility check only; no inferential claim is made for small strata. PASS.

## H. Paired comparison validation

| Pair | Left wins | Ties | Left losses | Mean delta | Positive sum | Negative sum | Mean win | Mean loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_size vs structure_aware | 75 | 3 | 62 | +0.006382 | 5.527960 | -4.634520 | 0.073706 | -0.074750 |
| fixed_size vs prompt_based | 81 | 3 | 56 | +0.010847 | 6.155605 | -4.637015 | 0.075995 | -0.082804 |
| structure_aware vs prompt_based | 60 | 3 | 77 | +0.004465 | 5.814541 | -5.189390 | 0.096909 | -0.067395 |

The structure-aware comparison is correctly oriented: 60 wins, 3 exact raw-score ties, and 77 losses, while the positive deltas sum to 5.814541 and negative deltas sum to -5.189390. Its larger winning magnitudes yield mean delta +0.004465; this is not a sign, tie, orientation, or formatting bug.

## I. Metric semantics

- Normalization: Unicode NFKC, casefold, whitespace split/rejoin; punctuation retained; no article removal, stemming, or lemmatization.
- Tokenization: Unicode regex word runs or individual non-word/non-whitespace symbols.
- Overlap: Counter/multiset intersection; precision=overlap/prediction tokens; recall=overlap/gold tokens; harmonic F1.
- Aggregation: per-question scores followed by unweighted macro mean; raw floats retained.
- Empty Cases: both token lists => PRF 1; exactly one empty => PRF 0.
- Exact Match: equality after NFKC/casefold/whitespace normalization.
- Containment: gold token sequence occurs contiguously in generated-answer token sequence.
- Ties: exact raw-float equality (delta == 0); no rounded tie threshold.
- Rounding: round to four decimals for published display checks only.

## J. Deterministic spot-check appendix

Selection is algorithmic: top five minimum advantages over both alternatives for each strategy, then the five smallest three-way score ranges. Repeated IDs are retained when selection criteria overlap.

### q_easy_021 — strongest_fixed_size_advantage

- Difficulty/type: `easy` / `fact`
- Question: Since which Angular version are reactive forms strictly typed by default?
- Gold snippet: Reactive forms are strictly typed by default starting with Angular 14.
- `fixed_size` F1 `1.000000000000`; answer: Reactive forms are strictly typed by default starting with Angular 14.; first chunk `angular:guide/forms/typed-forms.md::fixed::000000`; evidence coverage `1.0`; chunk snippet: Typed Forms  As of Angular 14, reactive forms are strictly typed by default.  As background for this guide, you should already be familiar with [Angular Reactive Forms](guide/forms
- `structure_aware` F1 `0.750000000000`; answer: Reactive forms have been strictly typed by default since Angular 14.; first chunk `angular:guide/forms/typed-forms.md::structure::000000`; evidence coverage `1.0`; chunk snippet: Typed Forms  As of Angular 14, reactive forms are strictly typed by default.  As background for this guide, you should already be familiar with [Angular Reactive Forms](guide/forms
- `prompt_based` F1 `0.750000000000`; answer: Reactive forms have been strictly typed by default since Angular 14.; first chunk `angular:guide/forms/typed-forms.md::prompt::000000`; evidence coverage `1.0`; chunk snippet: Typed Forms  As of Angular 14, reactive forms are strictly typed by default.  As background for this guide, you should already be familiar with [Angular Reactive Forms](guide/forms

### q_easy_036 — strongest_fixed_size_advantage

- Difficulty/type: `easy` / `fact`
- Question: When do decorator-based view query results become available?
- Gold snippet: View query results become available in `ngAfterViewInit`.
- `fixed_size` F1 `0.500000000000`; answer: Decorator-based view query results become available in the ngAfterViewInit lifecycle method. Before that point the value is undefined.; first chunk `angular:guide/components/queries.md::fixed::000004`; evidence coverage `0.0`; chunk snippet: : While the Angular team recommends using the signal-based query function for new projects, the original decorator-based query APIs remain fully supported.  You can alternatively d
- `structure_aware` F1 `0.235294117647`; answer: Decorator-based view query results (e.g., @ViewChild/@ViewChildren) become available in the ngAfterViewInit lifecycle method — before that they are undefined. If you set @ViewChild(..., {static: true}), the result is available earlier, in ngOnInit.; first chunk `angular:guide/components/queries.md::structure::000010`; evidence coverage `0.0`; chunk snippet: View queries {#decorator-view-queries}  You can query for a single result with the `@ViewChild` decorator.  @Component({   selector: 'custom-card-header',   /*...*/ }) export class
- `prompt_based` F1 `0.275862068966`; answer: Decorator-based view query results become available in the ngAfterViewInit lifecycle method. Before that point the value is undefined. (If you set @ViewChild/@ContentChild with { static: true }, the result is available earlier — in ngOnInit.); first chunk `angular:guide/components/queries.md::prompt::000006`; evidence coverage `0.0`; chunk snippet:   Decorator-based queries  TIP: While the Angular team recommends using the signal-based query function for new projects, the original decorator-based query APIs remain fully suppo

### q_easy_010 — strongest_fixed_size_advantage

- Difficulty/type: `easy` / `definition`
- Question: What is a binding in an Angular template?
- Gold snippet: A binding is a dynamic connection between a component's template and its data that keeps the rendered template updated as the component data changes.
- `fixed_size` F1 `0.442307692308`; answer: A binding in an Angular template is a dynamic connection between a component's template and its data. It lets the template read values (for example via text interpolation {{ ... }}) or set element/component/directive properties (for example with [property]="..."), so when the com; first chunk `angular:guide/templates/overview.md::fixed::000000`; evidence coverage `0.0`; chunk snippet: Template syntax  In Angular, a template is a chunk of HTML. Use special syntax within a template to leverage many of Angular's features.  TIP: Check out Angular's [Essentials](esse
- `structure_aware` F1 `0.218181818182`; answer: A binding in an Angular template is a dynamic connection between a component’s template and its data that keeps the rendered DOM automatically up-to-date when the data changes.   Common template binding forms (from the supplied context): - Interpolation (double curly braces) for ; first chunk `angular:guide/templates/binding.md::structure::000000`; evidence coverage `0.0`; chunk snippet: Binding dynamic text, properties and attributes  In Angular, a **binding** creates a dynamic connection between a component's template and its data. This connection ensures that ch
- `prompt_based` F1 `0.263736263736`; answer: A binding in an Angular template is a dynamic connection between the template and a component's data that keeps the rendered view in sync with the underlying data. When the bound data changes, Angular updates the DOM automatically.  Examples from the provided context: - Text inte; first chunk `angular:tutorials/learn-angular/steps/6-property-binding/README.md::prompt::000002`; evidence coverage `0.0`; chunk snippet:   Property binding is one of Angular's many powerful features. If you'd like to learn more checkout [the Angular documentation](guide/templates/binding#css-class-and-style-property

### q_easy_020 — strongest_fixed_size_advantage

- Difficulty/type: `easy` / `definition`
- Question: What is a component harness in Angular testing?
- Gold snippet: A component harness is a class that allows tests to interact with components the way an end user does through a supported API.
- `fixed_size` F1 `0.381818181818`; answer: A component harness is a class that lets tests interact with an Angular component the way an end user does via a supported API. Harnesses encapsulate component interactions, making tests less brittle (by insulating them from implementation details like DOM structure), more readab; first chunk `angular:guide/testing/component-harnesses-overview.md::fixed::000001`; evidence coverage `1.0`; chunk snippet: es)                                | | Component harness authors   | Developers who maintain some reusable Angular components and want to create a test harness for its users to use
- `structure_aware` F1 `0.232044198895`; answer: A component harness is a class that lets tests interact with an Angular component the way an end user does via a supported API. Harnesses can be created for any component (from small widgets to full pages) and provide a stable, high-level API that hides implementation details.  B; first chunk `angular:guide/testing/using-component-harnesses.md::structure::000000`; evidence coverage `1.0`; chunk snippet: Using component harnesses in tests
- `prompt_based` F1 `0.229508196721`; answer: A component harness is a class that lets tests interact with a component the way an end user does via a supported API. Harnesses can be created for any component (from small widgets to full pages) and are used by test code to perform actions and read state without depending on im; first chunk `angular:guide/testing/component-harnesses-overview.md::prompt::000000`; evidence coverage `1.0`; chunk snippet: Component harnesses overview  A <strong>component harness</strong> is a class that allows tests to interact with components the way an end user does via a supported API. You can cr

### q_easy_012 — strongest_fixed_size_advantage

- Difficulty/type: `easy` / `list`
- Question: What are the four main parts of an Angular component?
- Gold snippet: The four main parts are a `@Component` decorator, an HTML template, a CSS selector, and a TypeScript class containing the component's behavior.
- `fixed_size` F1 `0.646153846154`; answer: The four main parts are: 1. The @Component decorator (configuration) 2. An HTML template 3. A CSS selector 4. A TypeScript class (component behavior); first chunk `angular:introduction/essentials/components.md::fixed::000000`; evidence coverage `0.6`; chunk snippet: Components  The fundamental building block for creating applications in Angular.  Components are the main building blocks of Angular applications. Each component represents a part 
- `structure_aware` F1 `0.505050505051`; answer: The four main parts are: 1. A @Component decorator (metadata/configuration for Angular)   2. An HTML template (what renders into the DOM)   3. A CSS selector (how the component is used in HTML)   4. A TypeScript class (the component's behaviors, e.g., handling input or fetching d; first chunk `angular:introduction/essentials/components.md::structure::000000`; evidence coverage `0.6`; chunk snippet: Components  The fundamental building block for creating applications in Angular.  Components are the main building blocks of Angular applications. Each component represents a part 
- `prompt_based` F1 `0.522727272727`; answer: According to the provided context, the four main parts are: - A @Component decorator (configuration/metadata) - An HTML template (what renders into the DOM) - A CSS selector (how the component is used in HTML) - A TypeScript class (the component's behaviors); first chunk `angular:tutorials/first-app/steps/02-Home/README.md::prompt::000001`; evidence coverage `0.6`; chunk snippet:   Conceptual preview of Angular components  Angular apps are built around components, which are Angular's building blocks. Components contain the code, HTML layout, and CSS style i

### q_easy_052 — strongest_structure_aware_advantage

- Difficulty/type: `easy` / `fact`
- Question: What file extension does Angular use for XLIFF 2 translation files?
- Gold snippet: XLIFF 2 translation files use the `.xlf` extension.
- `fixed_size` F1 `0.214285714286`; answer: The supplied context does not state what file extension Angular uses for XLIFF 2 translation files. The documents only show examples using XLIFF (version 1.2) with the .xlf extension (e.g., messages.xlf and messages.fr.xlf). If you need the XLIFF 2 extension, I don't have that in; first chunk `angular:guide/i18n/example.md::fixed::000000`; evidence coverage `0.0`; chunk snippet: Example Angular Internationalization application  `fr-CA` and `en-US` example  The following tabs display the example application and the associated translation files.  Referenced 
- `structure_aware` F1 `0.720000000000`; answer: Angular uses the .xlf file extension for XLIFF 2 translation files.; first chunk `angular:guide/i18n/example.md::structure::000000`; evidence coverage `0.0`; chunk snippet: Example Angular Internationalization application
- `prompt_based` F1 `0.202898550725`; answer: The provided context does not state the file extension used specifically for XLIFF 2. The examples show the .xlf extension (e.g., messages.fr.xlf) and also mention .xtb as an alternative format, but no explicit mention of the XLIFF 2 file extension is given.; first chunk `angular:guide/i18n/example.md::prompt::000000`; evidence coverage `0.0`; chunk snippet: Example Angular Internationalization application  `fr-CA` and `en-US` example  The following tabs display the example application and the associated translation files.  Referenced 

### q_easy_001 — strongest_structure_aware_advantage

- Difficulty/type: `easy` / `definition`
- Question: What is hydration in Angular?
- Gold snippet: Hydration is the process that restores a server-side rendered application on the client, including reusing server-rendered DOM structures, persisting application state, and transferring application data already retrieved by the server.
- `fixed_size` F1 `0.410958904110`; answer: Hydration is the process that restores a server-side rendered (SSR) Angular application on the client. It reuses the server-rendered DOM structures, preserves application state and server-fetched data, and performs other steps so the client-side app can continue from the server-r; first chunk `angular:guide/hydration.md::fixed::000000`; evidence coverage `1.0`; chunk snippet: Hydration  What is hydration  Hydration is the process that restores the server-side rendered application on the client. This includes things like reusing the server rendered DOM s
- `structure_aware` F1 `0.742268041237`; answer: Hydration in Angular is the process that restores a server-side rendered (SSR) application on the client. This includes reusing the server-rendered DOM structures, persisting application state, transferring data already retrieved by the server, and other steps needed to make the ; first chunk `angular:guide/hydration.md::structure::000003`; evidence coverage `1.0`; chunk snippet: How do you enable hydration in Angular  Hydration can be enabled for server-side rendered (SSR) applications only. Follow the [Angular SSR Guide](guide/ssr) to enable server-side r
- `prompt_based` F1 `0.354066985646`; answer: Hydration in Angular is the process that restores a server-side rendered (SSR) application on the client by matching and reusing the server-rendered DOM and application state instead of tearing down and re-rendering it. This includes reusing DOM structures, persisting application; first chunk `angular:guide/hydration.md::prompt::000000`; evidence coverage `1.0`; chunk snippet: Hydration  What is hydration  Hydration is the process that restores the server-side rendered application on the client. This includes things like reusing the server rendered DOM s

### q_easy_040 — strongest_structure_aware_advantage

- Difficulty/type: `easy` / `behavior`
- Question: Do static query results update after initialization?
- Gold snippet: No. Static query results do not update after initialization.
- `fixed_size` F1 `0.758620689655`; answer: No. As stated in the provided context, static query results do not update after initialization.; first chunk `angular:guide/components/queries.md::fixed::000006`; evidence coverage `1.0`; chunk snippet:  options work the same way as signal-based queries except where described below.  Static queries  `@ViewChild` and `@ContentChild` decorators accept the `static` option.  @Componen
- `structure_aware` F1 `1.000000000000`; answer: No. Static query results do not update after initialization.; first chunk `angular:guide/components/queries.md::structure::000013`; evidence coverage `1.0`; chunk snippet: Static queries  `@ViewChild` and `@ContentChild` decorators accept the `static` option.  @Component({   selector: 'custom-card',   template: '<custom-card-header>Visit sunny Califo
- `prompt_based` F1 `0.431372549020`; answer: No. Static query results do not update after initialization. Setting {static: true} makes the result available earlier (e.g., in ngOnInit) but the returned value is not updated afterward.; first chunk `angular:guide/components/queries.md::prompt::000008`; evidence coverage `1.0`; chunk snippet:   Decorator-based query options  All query decorators accept an options object as a second parameter. These options work the same way as signal-based queries except where described

### q_easy_039 — strongest_structure_aware_advantage

- Difficulty/type: `easy` / `behavior`
- Question: What effect does setting static: true have on a ViewChild or ContentChild query?
- Gold snippet: It makes the result available earlier in `ngOnInit`, based on the guarantee that the target is always present and not conditionally rendered.
- `fixed_size` F1 `0.426966292135`; answer: Setting static: true tells Angular the queried ViewChild or ContentChild is always present (not conditionally rendered). That makes the query result available earlier (in ngOnInit), but the static query result does not update after initialization. The static option is only suppor; first chunk `angular:guide/components/queries.md::fixed::000006`; evidence coverage `0.0`; chunk snippet:  options work the same way as signal-based queries except where described below.  Static queries  `@ViewChild` and `@ContentChild` decorators accept the `static` option.  @Componen
- `structure_aware` F1 `0.567164179104`; answer: Setting static: true tells Angular the query target is always present (not conditionally rendered). That makes the query result available earlier—usable in ngOnInit—and the static query result does not update after initialization.; first chunk `angular:guide/components/queries.md::structure::000013`; evidence coverage `0.0`; chunk snippet: Static queries  `@ViewChild` and `@ContentChild` decorators accept the `static` option.  @Component({   selector: 'custom-card',   template: '<custom-card-header>Visit sunny Califo
- `prompt_based` F1 `0.425531914894`; answer: Setting static: true tells Angular the queried target is always present (not conditionally rendered). As a result the query result is resolved earlier — it is available in ngOnInit — and that resolved value does not update after initialization. (Note: the static option applies to; first chunk `angular:guide/components/queries.md::prompt::000007`; evidence coverage `0.0`; chunk snippet:   Content queries {#decorator-content-queries}  You can query for a single result with the `@ContentChild` decorator.  @Component({   selector: 'custom-toggle',   /*...*/ }) export

### q_hard_029 — strongest_structure_aware_advantage

- Difficulty/type: `hard` / `why`
- Question: Why does testing a nested route require more assertions than testing a single routed component?
- Gold snippet: A nested route contains multiple route layers, so a useful test needs to verify that the parent renders, that the child renders within the parent outlet, and that each layer can access its route data. The production routing model likewise requires child routes in configuration an
- `fixed_size` F1 `0.411764705882`; answer: The supplied context shows that nested routes involve multiple layers (a parent component with a router-outlet and one or more child components). So testing a nested route requires more assertions because you must verify all of the following, not just a single component:  - The p; first chunk `angular:guide/routing/testing.md::fixed::000002`; evidence coverage `0.6`; chunk snippet: () ? true : router.parseUrl('/login'); };  Router outlets  Router outlet tests are more of an integration test since you're essentially testing the integration between the [`Router
- `structure_aware` F1 `0.540145985401`; answer: Because a nested route renders more than one component and tests the integration between them. You must assert that the parent rendered, that the child rendered inside the parent (e.g., via the router-outlet), and that each layer receives its route data/state. In short, nested ro; first chunk `angular:guide/routing/testing.md::structure::000006`; evidence coverage `0.6`; chunk snippet: Nested routes  Testing nested routes ensures that both the parent and child components render correctly when navigating to nested URLs. This is important because nested routes invo
- `prompt_based` F1 `0.378109452736`; answer: Because a nested route renders multiple components and involves the router inserting a child into a parent outlet, you have to verify more things than for a single routed component. Specifically (from the context):  - You must assert the parent component renders correctly. - You ; first chunk `angular:guide/routing/testing.md::prompt::000004`; evidence coverage `0.6`; chunk snippet:   Nested routes  Testing nested routes ensures that both the parent and child components render correctly when navigating to nested URLs. This is important because nested routes in

### q_easy_047 — strongest_prompt_based_advantage

- Difficulty/type: `easy` / `fact`
- Question: Which Angular directive can dynamically render a template fragment at an ng-container location?
- Gold snippet: `NgTemplateOutlet` can dynamically render a template fragment at the `<ng-container>` location.
- `fixed_size` F1 `0.153846153846`; answer: The NgTemplateOutlet directive (used as *ngTemplateOutlet or [ngTemplateOutlet]) from @angular/common.; first chunk `angular:guide/templates/ng-template.md::fixed::000005`; evidence coverage `1.0`; chunk snippet:  Injects `ViewContainerRef` and programmatically renders the injected `TemplateRef`  Angular supports a special convenience syntax for structural directives. If you apply the direc
- `structure_aware` F1 `0.193548387097`; answer: The NgTemplateOutlet directive (from @angular/common).; first chunk `angular:guide/templates/ng-container.md::structure::000003`; evidence coverage `1.0`; chunk snippet: Rendering template fragments  You can use Angular's built-in `NgTemplateOutlet` directive to dynamically render a template fragment to the location of the `<ng-container>`.  @Compo
- `prompt_based` F1 `0.595744680851`; answer: The NgTemplateOutlet directive (used as *ngTemplateOutlet or [ngTemplateOutlet]) can dynamically render a TemplateRef at an <ng-container> location.; first chunk `angular:guide/templates/ng-template.md::prompt::000003`; evidence coverage `1.0`; chunk snippet:   Rendering a template fragment  Once you have a reference to a template fragment's `TemplateRef` object, you can render a fragment in one of two ways: in your template with the `N

### q_easy_046 — strongest_prompt_based_advantage

- Difficulty/type: `easy` / `fact`
- Question: Which Angular directive can dynamically render a component at an ng-container location?
- Gold snippet: `NgComponentOutlet` can dynamically render a component at the `<ng-container>` location.
- `fixed_size` F1 `0.260869565217`; answer: The NgComponentOutlet directive.; first chunk `angular:guide/templates/ng-container.md::fixed::000000`; evidence coverage `1.0`; chunk snippet: Grouping elements with ng-container  `<ng-container>` is a special element in Angular that groups multiple elements together or marks a location in a template without rendering a r
- `structure_aware` F1 `0.260869565217`; answer: The NgComponentOutlet directive.; first chunk `angular:guide/templates/ng-container.md::structure::000001`; evidence coverage `1.0`; chunk snippet: Using `<ng-container>` to display dynamic contents  `<ng-container>` can act as a placeholder for rendering dynamic content.
- `prompt_based` F1 `0.619047619048`; answer: The NgComponentOutlet directive. It is a structural directive that dynamically renders a component at the <ng-container> location.; first chunk `angular:guide/templates/ng-container.md::prompt::000000`; evidence coverage `1.0`; chunk snippet: Grouping elements with ng-container  `<ng-container>` is a special element in Angular that groups multiple elements together or marks a location in a template without rendering a r

### q_easy_025 — strongest_prompt_based_advantage

- Difficulty/type: `easy` / `fact`
- Question: What is the source of truth for Signal Forms according to Angular's forms comparison?
- Gold snippet: The source of truth is a user-defined writable signal model.
- `fixed_size` F1 `0.520000000000`; answer: The source of truth is a user-defined writable signal model — Signal Forms store data in a writable signal (e.g., calling the form signal to get current values).; first chunk `angular:guide/forms/signals/comparison.md::fixed::000000`; evidence coverage `1.0`; chunk snippet: Comparison with other form approaches  Angular provides three approaches to building forms: Signal Forms, Reactive Forms, and Template-driven Forms. Each has distinct patterns for 
- `structure_aware` F1 `0.541666666667`; answer: The source of truth for Signal Forms is a user-defined writable signal model — i.e., your form data lives in a writable signal (the form model).; first chunk `angular:guide/forms/signals/comparison.md::structure::000001`; evidence coverage `1.0`; chunk snippet: Quick comparison  | Feature          | Signal Forms                       | Reactive Forms                        | Template-driven Forms   | | ---------------- | -----------------
- `prompt_based` F1 `0.722222222222`; answer: The source of truth is a user-defined writable signal model — Signal Forms store form data in a writable signal.; first chunk `angular:guide/forms/signals/overview.md::prompt::000000`; evidence coverage `1.0`; chunk snippet: Forms with Angular Signals  Signal Forms is a library that allows you to manage form state in Angular applications by building on the reactive foundation of signals. With automatic

### q_medium_023 — strongest_prompt_based_advantage

- Difficulty/type: `medium` / `why`
- Question: Why must an HTTP interceptor clone a request or response when it wants to change most of its properties?
- Gold snippet: HttpRequest and HttpResponse are mostly immutable, so an interceptor cannot directly mutate them. Instead, it calls clone() and specifies the changed properties on the new instance.
- `fixed_size` F1 `0.365217391304`; answer: Because HttpRequest and HttpResponse instances are immutable: most of their properties cannot be changed in place. Interceptors must call .clone() and supply the modified properties to produce a new instance with the desired changes. This preserves immutability (making intercepto; first chunk `angular:guide/http/interceptors.md::fixed::000001`; evidence coverage `0.5`; chunk snippet: Interceptors` feature:  bootstrapApplication(App, {   providers: [provideHttpClient(withInterceptors([loggingInterceptor, cachingInterceptor]))], });  The interceptors you configur
- `structure_aware` F1 `0.341085271318`; answer: Because HttpRequest and HttpResponse objects are immutable: interceptors cannot change their properties directly. To apply mutations you must create a modified copy using .clone(), specifying the changed properties. This preserves idempotence (so the same request can safely pass ; first chunk `angular:guide/http/interceptors.md::structure::000005`; evidence coverage `0.5`; chunk snippet: Modifying requests  Most aspects of `HttpRequest` and `HttpResponse` instances are _immutable_, and interceptors cannot directly modify them. Instead, interceptors apply mutations 
- `prompt_based` F1 `0.485436893204`; answer: Because most properties of HttpRequest and HttpResponse are immutable, interceptors cannot change them in place. To apply mutations you must create a new instance via .clone() and specify the changed properties. This preserves immutability (making interceptors idempotent across r; first chunk `angular:guide/http/interceptors.md::prompt::000002`; evidence coverage `0.5`; chunk snippet:   Modifying requests  Most aspects of `HttpRequest` and `HttpResponse` instances are _immutable_, and interceptors cannot directly modify them. Instead, interceptors apply mutation

### q_medium_032 — strongest_prompt_based_advantage

- Difficulty/type: `medium` / `comparison`
- Question: How are internationalization and localization distinguished in Angular's i18n documentation?
- Gold snippet: Internationalization is the process of designing and preparing a project for use in different locales, whereas localization is the process of building locale-specific versions, including translating text and formatting locale-specific data.
- `fixed_size` F1 `0.520325203252`; answer: According to the Angular i18n docs:  - Internationalization (i18n) is the process of designing and preparing your project so it can be used in different locales worldwide. - Localization is the process of building localized versions of the project for specific locales. That inclu; first chunk `angular:guide/i18n/overview.md::fixed::000000`; evidence coverage `1.0`; chunk snippet: Angular Internationalization (i18n)  _Internationalization_, sometimes referenced as i18n, is the process of designing and preparing your project for use in different locales aroun
- `structure_aware` F1 `0.526315789474`; answer: According to the Angular i18n docs:  - Internationalization (i18n) is the process of designing and preparing your project so it can be used in different locales around the world. - Localization is the process of building locale-specific versions of the project — including extract; first chunk `angular:guide/i18n/overview.md::structure::000000`; evidence coverage `1.0`; chunk snippet: Angular Internationalization (i18n)  _Internationalization_, sometimes referenced as i18n, is the process of designing and preparing your project for use in different locales aroun
- `prompt_based` F1 `0.634615384615`; answer: According to the Angular i18n docs:  - Internationalization (i18n) is the process of designing and preparing your project for use in different locales. - Localization is the process of building versions of your project for different locales; it includes extracting text for transl; first chunk `angular:guide/i18n/overview.md::prompt::000000`; evidence coverage `1.0`; chunk snippet: Angular Internationalization (i18n)  _Internationalization_, sometimes referenced as i18n, is the process of designing and preparing your project for use in different locales aroun

### q_easy_038 — exact_or_near_tie

- Difficulty/type: `easy` / `fact`
- Question: What object does @ViewChildren create to hold query results?
- Gold snippet: `@ViewChildren` creates a `QueryList` containing the query results.
- `fixed_size` F1 `0.692307692308`; answer: @ViewChildren creates a QueryList object to hold the query results.; first chunk `angular:guide/components/queries.md::fixed::000001`; evidence coverage `1.0`; chunk snippet: ().map((action) => action.text)); }  `viewChildren` creates a signal with an `Array` of the query results.  **Queries never pierce through component boundaries.** View queries can 
- `structure_aware` F1 `0.692307692308`; answer: @ViewChildren creates a QueryList object to hold the query results.; first chunk `angular:guide/components/queries.md::structure::000014`; evidence coverage `1.0`; chunk snippet: Using QueryList  `@ViewChildren` and `@ContentChildren` both provide a `QueryList` object that contains a list of results.  `QueryList` offers a number of convenience APIs for work
- `prompt_based` F1 `0.692307692308`; answer: @ViewChildren creates a QueryList object to hold the query results.; first chunk `angular:guide/components/queries.md::prompt::000001`; evidence coverage `1.0`; chunk snippet:   View queries  View queries retrieve results from the elements in the component's _view_ — the elements defined in the component's own template. You can query for a single result 

### q_easy_045 — exact_or_near_tie

- Difficulty/type: `easy` / `definition`
- Question: What is <ng-container> in Angular?
- Gold snippet: `<ng-container>` is a special element that groups elements or marks a template location without rendering a real DOM element.
- `fixed_size` F1 `0.251308900524`; answer: <ng-container> is a special Angular template element that groups or marks a location in a template without creating a real DOM element. Key points from the documentation:  - It does not render an actual element in the DOM — its children are rendered directly in its place. - You c; first chunk `angular:guide/templates/ng-container.md::fixed::000000`; evidence coverage `1.0`; chunk snippet: Grouping elements with ng-container  `<ng-container>` is a special element in Angular that groups multiple elements together or marks a location in a template without rendering a r
- `structure_aware` F1 `0.254143646409`; answer: <ng-container> is a special Angular template element that groups or marks a location in a template without producing a real DOM element. It lets you:  - Group multiple elements together (e.g., inside a section) while not adding an extra wrapper element in the rendered DOM. - Act ; first chunk `angular:guide/templates/ng-container.md::structure::000000`; evidence coverage `1.0`; chunk snippet: Grouping elements with ng-container  `<ng-container>` is a special element in Angular that groups multiple elements together or marks a location in a template without rendering a r
- `prompt_based` F1 `0.259887005650`; answer: <ng-container> is a special Angular template element that groups or marks a location in a template without rendering an actual DOM element. You can:  - wrap multiple elements so they are not surrounded by an extra DOM node (e.g., inside a section). - apply directives (including s; first chunk `angular:guide/templates/ng-container.md::prompt::000000`; evidence coverage `1.0`; chunk snippet: Grouping elements with ng-container  `<ng-container>` is a special element in Angular that groups multiple elements together or marks a location in a template without rendering a r

### q_easy_044 — exact_or_near_tie

- Difficulty/type: `easy` / `behavior`
- Question: What can ViewContainerRef.createComponent() be used for?
- Gold snippet: It can be used to dynamically create and render a component.
- `fixed_size` F1 `0.143790849673`; answer: ViewContainerRef.createComponent() is used to dynamically create and render a component as part of an existing view container. In particular:  - It creates a component and automatically inserts its host element and host view into the container’s view hierarchy at the container’s ; first chunk `angular:guide/components/programmatic-rendering.md::fixed::000002`; evidence coverage `1.0`; chunk snippet: THEME_DATA', {   factory: () => 'light', });  @Component({   selector: 'themed-panel',   template: `<div [class]="theme">...</div>`, }) export class ThemedPanel {   theme = inject(
- `structure_aware` F1 `0.133333333333`; answer: You can use ViewContainerRef.createComponent() to dynamically create and render a component and insert its host element and view into the container’s view hierarchy at the container’s location. Typical uses:  - Add a component at runtime so it becomes part of the container’s logi; first chunk `angular:guide/components/programmatic-rendering.md::structure::000006`; evidence coverage `1.0`; chunk snippet: Using ViewContainerRef  A **view container** is a node in Angular's component tree that can contain content. Any component or directive can inject `ViewContainerRef` to get a refer
- `prompt_based` F1 `0.137500000000`; answer: ViewContainerRef.createComponent() is used to dynamically create and render a component at the location of a view container (the place in the DOM of the component or directive that injected the ViewContainerRef). Specifically:  - It instantiates the component and inserts its host; first chunk `angular:guide/components/programmatic-rendering.md::prompt::000004`; evidence coverage `1.0`; chunk snippet:   Using ViewContainerRef  A **view container** is a node in Angular's component tree that can contain content. Any component or directive can inject `ViewContainerRef` to get a ref

### q_hard_032 — exact_or_near_tie

- Difficulty/type: `hard` / `sequence`
- Question: Describe the complete state transition of a traditional asynchronous form validator from invocation until the final validity result becomes available.
- Gold snippet: The async validator begins an asynchronous operation such as an HTTP request and the control becomes pending. The validator eventually produces null for valid input or ValidationErrors for invalid input. When its observable completes, pending becomes false and Angular updates the
- `fixed_size` F1 `0.149068322981`; answer: Using only the supplied context, the end-to-end state transition for a traditional async validator is:  1. Invocation - The async validator function/resource is invoked whenever the field value changes (all validators run on every change; no short‑circuiting). - The validator mus; first chunk `angular:guide/forms/signals/validation.md::fixed::000001`; evidence coverage `0.5072727272727272`; chunk snippet: `) complete immediately. Asynchronous validation rules (like `validateHttp()`) may take time and set the `pending()` signal to `true` while executing.  All validation rules run on 
- `structure_aware` F1 `0.139860139860`; answer: Using the supplied context, the complete state transition for a traditional asynchronous form validator is:  1. Invocation (user changes the field value)    - Validation rules automatically run on every value change (unless the field is hidden/disabled or updateOn delays checks).; first chunk `angular:guide/forms/signals/validation.md::structure::000025`; evidence coverage `0.0`; chunk snippet: Pending state  While async validation runs, the field's `pending()` signal returns `true`. Use this to show loading indicators:  @if (form.username().pending()) {   <span class="sp
- `prompt_based` F1 `0.135714285714`; answer: The lifecycle from the moment a control's validators are invoked until the final validity result is available proceeds as follows:  1. Value change / invocation    - The user changes the control value (or the control is otherwise updated). All validators are run on each change.  ; first chunk `angular:guide/forms/form-validation.md::prompt::000009`; evidence coverage `0.0`; chunk snippet:   Creating asynchronous validators  Asynchronous validators implement the `AsyncValidatorFn` and `AsyncValidator` interfaces. These are very similar to their synchronous counterpar

### q_hard_003 — exact_or_near_tie

- Difficulty/type: `hard` / `mechanism`
- Question: How does incremental hydration combine @defer, hydration triggers, and Event Replay to keep deferred server-rendered content responsive to early user interactions?
- Gold snippet: A hydrate trigger on an @defer block causes Angular to render the block's main content during SSR while leaving it dehydrated on the client until the trigger fires. User events that occur before hydration are queued, and Event Replay replays them after hydration. Incremental hydr
- `fixed_size` F1 `0.291262135922`; answer: Incremental hydration ties the three features together so server-rendered, deferred sections stay interactive to early user actions without paying the cost of hydrating everything up front:  - @defer + hydrate triggers: you mark parts of the UI with @defer and add one or more hyd; first chunk `angular:guide/incremental-hydration.md::fixed::000000`; evidence coverage `0.75`; chunk snippet: Incremental Hydration  **Incremental hydration** is an advanced type of [hydration](guide/hydration) that can leave sections of your application dehydrated and _incrementally_ trig
- `structure_aware` F1 `0.301886792453`; answer: Incremental hydration uses three features together to keep server-rendered, deferred content responsive to early user interactions:  - @defer blocks mark parts of the UI as deferrable so they can be left dehydrated on the client to reduce initial bundle size and cost. - Hydrate t; first chunk `angular:guide/incremental-hydration.md::structure::000003`; evidence coverage `0.75`; chunk snippet: How does incremental hydration work?  Incremental hydration builds on top of full-application [hydration](guide/hydration), [deferrable views](/guide/templates/defer), and [event r
- `prompt_based` F1 `0.285714285714`; answer: Incremental hydration uses three pieces together:  - @defer blocks mark content as deferrable and, during the initial server render, can render the full template (not just the placeholder) while keeping that content “dehydrated” on the client so its JS and listeners are not yet a; first chunk `angular:guide/incremental-hydration.md::prompt::000002`; evidence coverage `0.75`; chunk snippet:   How does incremental hydration work?  Incremental hydration builds on top of full-application [hydration](guide/hydration), [deferrable views](/guide/templates/defer), and [event

All sampled scores, question/answer bindings, gold records, strategy-specific retrieval rows, and chunk strategy labels match. PASS.

## K. Immutable artifact inventory

The freeze manifest records SHA-256 and byte size for each authoritative file. Inventory count: 34.

- `FINAL_BENCHMARK_HANDOFF.md` — `b3a6626a01567a4180ab77d0f194b7c8f3c69e053f237b1e2cb50eacef0dc6eb` (12780 bytes)
- `FINAL_BENCHMARK_REPORT.md` — `5e884cc3d6bada96743da4ac66ca53edec8194870145a6e74ba0a5c1372695e5` (10232 bytes)
- `configs/generation_gpt5mini_v2.json` — `0305de52f013fbb0216a9ec6b245977e04cd4242e33f03d66488a2db7850a727` (1053 bytes)
- `data/benchmark/angular/canonical_v2/evaluation/evaluations.jsonl` — `c87f6202ee289567d1be686b5960de5d419fb916e030c39682ee2655dfa2ea37` (1203645 bytes)
- `data/benchmark/angular/canonical_v2/evaluation/manifest.json` — `6c49599dcc4eca8549c016b2ebb4235244947ffe45d83a0bc33159a8edd2b017` (54018 bytes)
- `data/benchmark/angular/canonical_v2/evaluation/paired.jsonl` — `6b308e6a6c94cdc99985e43bce49889c89a25d1bf0246f07905038af34f7f73f` (265562 bytes)
- `data/benchmark/angular/canonical_v2/evaluation/stats.json` — `10738162027de15374a93695f62e00dbadfee1e4802ca8409e965dafd52196e3` (169 bytes)
- `data/benchmark/angular/canonical_v2/evaluation/summary.json` — `6f570c9a7050c44fa58d8cbd940cb03406b31bdc6a1ae628a458d63a461f1cbb` (92231 bytes)
- `data/benchmark/angular/canonical_v2/generation/fixed_size/answers.jsonl` — `5646ea7cff302ccce7f0e0ef9e424cacfdf4c5a988ca1f538b011939fe9f4894` (275451 bytes)
- `data/benchmark/angular/canonical_v2/generation/fixed_size/failures.jsonl` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 bytes)
- `data/benchmark/angular/canonical_v2/generation/fixed_size/manifest.json` — `c79703f5381327493edc773b2b9448a53d6b6c0467801ac8ae6810274c714f74` (36219 bytes)
- `data/benchmark/angular/canonical_v2/generation/fixed_size/stats.json` — `5cd8c55befb1c22c81cc579a34876780ffbf5f51a9cb252abb8c2d904bff770c` (264 bytes)
- `data/benchmark/angular/canonical_v2/generation/fixed_size.provider_diagnostics.jsonl` — `9e7c2339d39b66a5acb26e7186ac9eb05748472964cd878a34c84b990552b8a2` (138851 bytes)
- `data/benchmark/angular/canonical_v2/generation/prompt_based/answers.jsonl` — `426d7c584540ec65582fe43a29c08c2b4e3e7cba06a63fb096bfddd59640bf2f` (280415 bytes)
- `data/benchmark/angular/canonical_v2/generation/prompt_based/failures.jsonl` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 bytes)
- `data/benchmark/angular/canonical_v2/generation/prompt_based/manifest.json` — `f6fad815541abfddd32ac8b63dbc6026d51e1ce99977d0aa06370e95be63c345` (36219 bytes)
- `data/benchmark/angular/canonical_v2/generation/prompt_based/stats.json` — `d2de663e8f851036013c4cfbc15d8d1b835e063a51ab9d32e5e63f53ea5c1491` (266 bytes)
- `data/benchmark/angular/canonical_v2/generation/prompt_based.provider_diagnostics.jsonl` — `da98d5f4a8760cc0b4e580b9013439847fbba89ce7519774c3a6633a39d09232` (142823 bytes)
- `data/benchmark/angular/canonical_v2/generation/structure_aware/answers.jsonl` — `9f83cb0ebe2b1a893285aa6a1d75d9fcad3505f12b904e16e23bacf8742c7d10` (277208 bytes)
- `data/benchmark/angular/canonical_v2/generation/structure_aware/failures.jsonl` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 bytes)
- `data/benchmark/angular/canonical_v2/generation/structure_aware/manifest.json` — `35964d8524e0caace498cc7254142d0d4f9b8d0e1233ad27bb5bb4ff83373b63` (36219 bytes)
- `data/benchmark/angular/canonical_v2/generation/structure_aware/stats.json` — `c4e6a57fdbef74a501c88d291022796023f163ad5b90db0fffff16f71d08d4c5` (266 bytes)
- `data/benchmark/angular/canonical_v2/generation/structure_aware.provider_diagnostics.jsonl` — `1b38da9482e2345f3bf4ba1a10949b1b69fefce27c6a5f61149d3e71abb9a8cf` (142817 bytes)
- `data/benchmark/angular/canonical_v2/generation_health.json` — `70ccd59f9202cd564ea3015c58a1c3351975ab977e6d986c2bb8f4dff4ef4ab2` (5014 bytes)
- `data/benchmark/angular/canonical_v2/inputs/fixed_size.generation_inputs.jsonl` — `f9ec48f62182068355775bfd8afd19002b317b02a0db2f22a970fa8b80d37a41` (1828352 bytes)
- `data/benchmark/angular/canonical_v2/inputs/manifest.json` — `b500b3b9b2b1077b7c1271250c2ea42339502f1af9958c690d00d8fa5e1c4af8` (24506 bytes)
- `data/benchmark/angular/canonical_v2/inputs/prompt_based.generation_inputs.jsonl` — `15a5e5c593de15421c4db297a4e78144ca356e6ed15cad70f9baef07f40ea04d` (2016249 bytes)
- `data/benchmark/angular/canonical_v2/inputs/stats.json` — `cc1f9bb3531e83411a5d035926a86e8c4bcf50f0493669e711b81259b8685e22` (171 bytes)
- `data/benchmark/angular/canonical_v2/inputs/structure_aware.generation_inputs.jsonl` — `2b8adc399c2e6411ffd86aa131d982191fc77e4164f8b8149ddee5e2978eb76d` (2337080 bytes)
- `data/evaluation/angular/qa_dataset.jsonl` — `2621d3685b58439629a0dfe1cb95fd236714f42d598d5bd00ef9dbabc003cef9` (162302 bytes)
- `data/retrieval/angular/canonical_production_v2/9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7/aggregate.json` — `6ad86c73da4061572d78a8f73e73250271221d6a674ff078ae6865979224620d` (4666 bytes)
- `data/retrieval/angular/canonical_production_v2/9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7/manifest.json` — `11e075bca02e6a708fd0e92dabb0d83cf43a558054cc2d5b284a8550f7c28cdd` (2779 bytes)
- `data/retrieval/angular/canonical_production_v2/9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7/per_query.jsonl` — `b60f0ec229f3cc9fb8b505596e2899fa9b99569b30a2a34e2bf11e3489ce1154` (27235900 bytes)
- `data/retrieval/angular/canonical_production_v2/9dab4015ca1ae4c4abda04ccc5809a5e030c793fcc7feff932d04ecfb116a6b7/stats.json` — `75616219b514a0657c1542f6ed3b3b3db117ff36af47ba7c41495f0ac290e050` (218 bytes)

Nearby historical/alternate artifacts explicitly excluded:

- generation/fixed_size/manifest.inconsistent-overlap-audit.json (diagnostic, not referenced by commit manifest)
- retrieval sibling d50392b7.../failure.json (failed noncanonical run)
- same_top_k rows in canonical retrieval file (committed secondary protocol, excluded from same_token_budget answer comparison)

## L. Test results

Focused freeze, answer-evaluation, and evidence-evaluation tests: 51 passed in 0.74s. Full repository suite: 333 passed in 2.71s. Baseline 329 tests plus 4 new focused freeze tests; zero regressions.

## M. Freeze policy and decision

Downstream work may read these artifacts and write derived outputs only under canonical_v2/statistical_analysis; canonical inputs, retrieval, answers, evaluation, and manifests are immutable.

All blocking gates pass. Canonical artifact hashes were captured before publication; only the validation report, validator/test code, and freeze declaration are new.

**CANONICAL PRODUCTION BENCHMARK V2: FROZEN FOR STATISTICAL ANALYSIS**
