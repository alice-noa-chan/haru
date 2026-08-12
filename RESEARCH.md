# Haru Research Notes

Haru uses a custom Causal Folded Recurrent Decoder (CFRD). The design combines
recurrent parameter sharing, local causal attention, compressed summary memory,
and supervised early exits.

## Related architecture research

- [TinyStories](https://arxiv.org/abs/2305.07759)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [Landmark Attention](https://arxiv.org/abs/2305.16300)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
- [DeepLoop](https://arxiv.org/abs/2607.13491)

This list is a starting point rather than a complete literature review. A
novelty claim about the exact combination used by Haru requires a separate,
systematic search.

## Relation-binding diagnosis

The released 6.8M checkpoint was tested with paired prompts that keep the
candidate answers fixed while swapping only the decisive entity-to-attribute
or entity-to-role assignment. This is stricter than ordinary next-token loss:
both members of a pair must be correct, and the preferred candidate must flip.

The main observations were:

- The independent 200-pair V2 evaluation produced 0/200 strict pairs and
  0/200 preference flips. This indicates that the model often ignored the
  changed binding rather than merely choosing the wrong surface form.
- A narrow relation-training set appeared to score 14/25 on its nearby test,
  but fell to 6/200 on V2. That gap is consistent with template or lexical
  memorization, not a general relation skill.
- Factorially varying templates, relation vocabulary, and cues improved strict
  accuracy in two CPU replications: 2/40 to 12/40 (`p=0.0129`) and 0/40 to
  9/40 (`p=0.00391`). The gain therefore survived a seed change.
- Those gains were limited to location and state bindings. Negation, speaker,
  and transfer-role subsets remained at 0/24, showing that attribute lookup
  and directional role binding are not the same capability.
- Adding fixed name pairs did not replicate: an initial 17/40 result became
  5/40 on the second seed, below the 9/40 factorial baseline. Stable identities
  made memorization easier without reliably teaching variable roles.

These results do not identify a single cause in isolation. Together they point
to three interacting bottlenecks: insufficient independent model capacity,
information compression before distant evidence can be rebound, and a plain
language-model objective that does not directly punish identity shortcuts.

## Next-training design

The `haru-v2-binding` profile maps each diagnosed bottleneck to a separable
change:

1. Three physical decoder cells reduce forced parameter sharing across six
   recurrent passes.
2. A final non-recurrent block performs full-context causal attention with its
   own attention and SwiGLU parameters. It provides a direct path for exact
   token relationships after the folded stack, without replacing the existing
   local and summary-memory paths.
3. The tokenizer grows from 8,192 to 12,000 entries, while the complete model
   remains within the intended compact range at 11,634,459 parameters.
4. Every training micro-batch contains balanced counterfactual pairs with newly
   permuted entity identities. Template, lexical payload, and entity choices
   are sampled independently.
5. Candidate continuations are compared only over their answer-token spans.
   The margin loss must prefer candidate A under prompt A and candidate B under
   the counterfactual prompt, directly penalizing a fixed-name preference.
6. Validation reserves different names and phrasings. Best-checkpoint selection
   combines held-out language-model loss with held-out relation loss, while
   strict pair accuracy remains the primary binding diagnostic.

The synthetic relation objective has weight 0.20 and remains auxiliary to the
ordinary next-token objective. It is intended to teach a reusable binding
operation, not to replace broad text training with a small synthetic corpus.

## Required validation and limits

The implementation tests causality, serialization, gradient flow, deterministic
sampler resume, split leakage, and real SentencePiece candidate encoding. These
are code-level checks; they do not demonstrate that an untrained configuration
has become a better language model.

After full training, evaluation should report at least:

- language-model validation loss at recurrent depths 2, 4, and 6;
- decision accuracy and strict pair accuracy overall and per relation family;
- preference-flip rate, mean signed margin, and multiple training seeds;
- ablations for the third cell, binding block, vocabulary size, and relation
  loss rather than attributing a combined gain to one component;
- free-generation quality and memorization checks outside synthetic templates.

This design targets entity and role hallucinations caused by weak contextual
binding. It cannot guarantee factual truth, recover knowledge absent from the
training data, or turn Haru into an instruction-following assistant. Those
claims require separate data, retrieval, calibration, and safety evaluation.
