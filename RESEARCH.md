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

## Parameter count is the wrong axis to match on alone

A folded architecture decouples parameters from compute, so "matched
parameters" and "matched compute" name two different experiments. Measured on a
512-token forward with `torch.utils.flop_counter`:

| Model | Parameters | GFLOPs | Blocks applied |
|---|---:|---:|---:|
| Haru v1.1 CFRD (depth 6) | 11,634,459 | 17.31 | 7 |
| Dense 4 layers x FFN 1184 | 11,669,377 | 11.95 | 4 |
| Dense 7 layers x FFN 1184 | 16,943,233 | 17.34 | 7 |

Three cells applied six times, plus the binding block, give CFRD seven block
applications from four blocks' worth of parameters. Holding parameters fixed
therefore hands CFRD about 1.45x the forward FLOPs, and a win under that
condition alone would not establish an architectural advantage.

Three further properties of the comparison matter:

- A dense decoder cannot match parameters, sequential depth, and FFN ratio at
  once. At seven layers the parameter budget forces FFN 512, or 1.33x d_model,
  which handicaps the control instead of isolating the architecture. Both arms
  therefore keep FFN near 3x d_model, CFRD's own cell proportion, and each arm
  concedes exactly one axis.
- The learning rate is CFRD's tuned value, shared by every arm. CFRD's residual
  gates start at sigmoid(-1) and its updates are scaled by 1/sqrt(recurrences),
  so this setting is more likely to suit CFRD than the controls. A baseline win
  under a CFRD-tuned schedule is therefore stronger evidence than the reverse.
- CFRD's `ModelOutput.loss` is the deep-supervision total, `final_loss + 0.15 x
  mean(auxiliary exits)`, while the dense baseline has one exit and so reports
  plain cross-entropy. The two fields are not comparable, and every comparison
  must read `final_loss`. On shifted targets at initialization CFRD's
  `final_loss` is 9.5698 and the baseline's is 9.4677, both near the uniform
  reference of ln(12000) = 9.3927; CFRD's `loss` for the same batch is 11.0091,
  which is 1.15x its exit mean and not a property of the architecture at all.

The v1.1 depth sweep should be read against these compute figures rather than
against parameter count: depth 2, 4, and 6 cost 10.01, 13.66, and 17.31 GFLOPs
and reach perplexity 19.24, 8.35, and 6.82. A parameter-matched dense model at
11.95 GFLOPs sits between depth 2 and depth 4.

## Direction test: CFRD against matched dense baselines

A CPU-scale replication of `compare_architectures.py`, three seeds per arm,
400 steps, 819,200 tokens per arm per seed, 8,192-entry tokenizer, relation
objective at weight 0.20 in every arm. The CFRD arm keeps the release
proportions at reduced size: d_model 192, 3 cells, 4 recurrences, binding block
enabled, 3.34M parameters.

| Arm | Parameters | GFLOPs | Validation loss | Paired delta | Strict pairs |
|---|---:|---:|---:|---:|---:|
| CFRD | 3,339,673 | 1.93 | 4.4414 (sd 0.0158) | reference | 0.233 |
| Dense, parameter-matched | 3,364,801 | 1.72 | 4.3988 (sd 0.0148) | -0.0426 | 0.350 |
| Dense, compute-matched | 3,809,089 | 1.95 | 4.3974 (sd 0.0211) | -0.0441 | 0.347 |

CFRD lost both comparisons. The paired delta was negative on every seed and
exceeded the spread of the paired differences, so it is not seed noise at this
scale. It also lost the parameter-matched comparison to an arm spending 11%
fewer FLOPs.

The binding result matters more, because binding is what the architecture was
built for. Strict pair accuracy has a chance level of 0.25, since both
directions of a pair must flip:

| Arm | Seed 1337 | Seed 1338 | Seed 1339 | Mean | Mean margin |
|---|---:|---:|---:|---:|---:|
| CFRD | 0.24 | 0.07 | 0.39 | 0.233 | +0.56 |
| Dense, parameter-matched | 0.35 | 0.40 | 0.30 | 0.350 | +1.17 |
| Dense, compute-matched | 0.40 | 0.32 | 0.32 | 0.347 | +1.29 |

CFRD's mean sits at chance. Both dense arms sit clearly above it with roughly
twice the decision margin. The spread is the sharper finding: CFRD ranges from
0.07 to 0.39 across seeds while the dense arms stay inside 0.30 to 0.40. A
single CFRD seed could report either a strong binding result or a total failure,
which is consistent with the earlier diagnosis that a narrow relation-training
set scored 14/25 nearby and 6/200 on V2, and with the fixed-name experiment
that produced 17/40 and then 5/40 on a second seed.

Limits of this test, in the order they would change the conclusion:

- The budget is roughly 900x smaller than a release run (819K tokens against
  753.7M). Folded architectures are argued to need depth-in-time that only
  appears with training, so this measures early optimization, not converged
  quality.
- The scale is 3.34M parameters at d_model 192 with 4 recurrences, not
  11.6M at 384 with 6. Parameter sharing may pay off only where independent
  capacity is the binding constraint.
- One tokenizer (8,192) and one corpus slice.
- The learning rate is the project's own CFRD-tuned 4e-4, shared by all arms,
  so the schedule favors CFRD if it favors anything.

None of these limits explain a below-chance binding mean or a 5.6x spread
across seeds. The conclusion this test supports is narrow and sufficient to
redirect v1.2: at compact scale, folding is not what produces Haru's binding
behaviour, and the components v1.1 shipped together have to be separated
before any further capacity is spent on the fold.

## v1.2 hypothesis, and its refutation

This section states the hypothesis that motivated `cell_attention = "full"`, and
the ablation result that refuted it. Both are kept: the reasoning was sound and
the prediction was wrong, and deleting the prediction would leave the next
reader free to propose it again.

### The hypothesis

The direction test rules out one explanation and points at another. Folding is
not what produces Haru's binding behaviour at compact scale: a plain dense
decoder holding the same parameters, and spending 11% fewer FLOPs, beat CFRD on
both language-model loss and strict pair accuracy. So the question is not how
much more recurrence to add, but which part of CFRD was ever doing the work.

The remaining structural difference between CFRD and those baselines is not the
fold at all. It is what a cell is allowed to see. A CFRD cell reads a 64-token
chunk and reaches past it only through four compressed summary slots; every
dense baseline reads the full context directly. Three independent observations
line up behind that being the binding constraint:

- v1.1's binding gain, 0/100 to 21/100, arrived together with the binding
  block, which is one ordinary full-context attention layer appended after the
  recurrent stack.
- README already lists "summary memory can lose names, quotations, and exact
  event details" as a known limitation. That is a description of the failure
  mode strict pair accuracy measures.
- The relation families that improved in v1.1 were location and ownership,
  which a single attribute lookup can answer. Speaker attribution and transfer,
  which need an exact token relationship recovered at distance, stayed at 0/20.

v1.2 therefore keeps the fold and changes what the fold operates on.
`cell_attention = "full"` gives every recurrence unrestricted causal attention
and drops summary memory, which exists only to cross a chunk boundary that no
longer exists. At release shape this is cheaper on both axes, 10,944,393
parameters and 16.03 GFLOPs against 11,634,459 and 17.31, so it is not buying
quality with compute.

### What each result means, decided before the run

`compare_architectures.py --ablate` runs CFRD, both matched dense baselines,
and three single-field variants under shared seeds. The reading is fixed in
advance so it cannot be chosen afterwards:

| Observation | Conclusion |
|---|---|
| `cfrd-no-binding-block` collapses binding, `cfrd-full-attention` recovers it | Full-context attention was the mechanism. v1.2 adopts full cells and drops summary memory. |
| `cfrd-full-attention` still loses to the dense baselines | The fold itself is the cost, not the attention range. Fold no further; treat CFRD as a compute-sharing trick with a quality price and say so. |
| `cfrd-unfolded` beats `cfrd` by more than the seed spread | Parameter sharing is the liability. Its saving has to be priced against that gap rather than assumed free. |
| Every CFRD arm stays inside the seed spread of the others | No component is carrying the result. The 21/100 in v1.1 is then attributable to the relation objective and the larger vocabulary, both of which apply to a dense model equally. |

The last row is the one that would end the architecture, and it is a live
possibility rather than a formality: CFRD's strict pair accuracy already ranges
0.07 to 0.39 across three seeds while both dense arms stay inside 0.30 to 0.40.

No v1.2 checkpoint should be trained until this table exists. v1.1 was released
having changed four things at once, which is why nothing in it can be
attributed, and repeating that with a fifth change would be the same mistake at
greater cost.

### The result: row two fired

Three seeds, 300 steps, 614,400 tokens per arm per seed, relation objective at
weight 0.20 in every arm. These numbers are not comparable to the 400-step table
above; only rows within this table may be compared to each other.

| Arm | Parameters | GFLOPs | Validation loss | Paired delta | Strict pairs | Margin |
|---|---:|---:|---:|---:|---:|---:|
| CFRD | 3,339,673 | 1.93 | 4.7273 | reference | 0.110 | +0.16 |
| Dense, parameter-matched | 3,364,801 | 1.72 | 4.6557 | -0.0715 | 0.203 | +0.53 |
| Dense, compute-matched | 3,809,089 | 1.95 | 4.6518 | -0.0754 | 0.343 | +0.83 |
| `cfrd-no-binding-block` | 2,945,687 | 1.73 | 4.7920 | +0.0647 | 0.043 | +0.03 |
| `cfrd-unfolded` | 3,791,327 | 1.93 | 4.6958 | -0.0315 | 0.043 | +0.04 |
| `cfrd-full-attention` | 3,166,665 | 1.82 | 4.7911 | +0.0638 | 0.050 | +0.09 |

Every paired delta exceeded its own spread across the three seeds.

**The hypothesis is refuted.** Row one required two things: that removing the
binding block collapses binding, and that full-context cells recover it. Only
the first happened. Removing the block took strict pairs from 0.110 to 0.043
and the decision margin from +0.16 to +0.03, so the block is load-bearing
inside CFRD. But `cfrd-full-attention` reached 0.050, worse than CFRD itself,
and lost to CFRD on language-model loss by +0.0638. Giving every cell the full
context did not reproduce what the binding block does. Whatever the block
contributes, it is not simply attention range.

**Row two fired.** `cfrd-full-attention` lost to both dense baselines on both
metrics. So did every other CFRD arm. The best CFRD configuration tested,
`cfrd-unfolded` at 4.6958, still lost to the parameter-matched dense baseline
at 4.6557 while carrying 13% more parameters, and its strict pairs sat at 0.043.
The conclusion fixed in advance stands: the fold itself is the cost, not the
attention range.

**Row three fired, and narrows the diagnosis.** `cfrd-unfolded` beat CFRD on
language-model loss by 0.0315, exceeding the spread, so parameter sharing does
carry a measurable quality price. But it did not improve binding at all, 0.043
against 0.110. Sharing costs language modelling; it is not what breaks binding.

One reading is not available. At this 300-step budget only the compute-matched
dense arm is clearly above the 0.25 chance level, and even the
parameter-matched dense arm sits below it at 0.203. Binding has barely begun to
emerge in most arms, so the binding column separates CFRD from dense but cannot
support fine distinctions among the CFRD variants, several of which are pinned
near zero. The language-model column is the reliable one here: its standard
deviations run 0.004 to 0.028 and every delta is consistent in sign across all
three seeds.

Two caveats belong with the refutation. `cfrd-full-attention` carries 5% fewer
parameters than CFRD because dropping summary memory frees more than full
attention costs, so a small part of its deficit is capacity rather than
structure. And the budget is roughly 1,200x smaller than a release run, so this
tests early optimization rather than converged quality, which is precisely the
regime where a folded model is most often argued to be at a disadvantage.

### Where this leaves v1.2, before the release-scale run

No configuration of CFRD tested here reaches a plain dense decoder on either
metric, at matched parameters or matched compute. The architecture is not
paying for itself at compact scale under this budget, and three separate
attempts to locate the component that would justify it, the binding block,
parameter sharing, and attention range, each failed to close the gap.

That leaves two honest options, and no third:

1. Establish the regime where folding pays before spending more on it. The
   claim that recurrent depth needs training time to become useful is testable:
   run the same table at release scale and a release budget on a GPU. Until
   that exists, "CFRD is better for compact Korean models" is unsupported.
2. Accept the measurement and make the dense configuration the release path,
   keeping the parts that are separable from the fold and that a dense model
   can use unchanged: the relation objective, the Korean surface features, and
   the tokenizer.

What is no longer defensible is shipping a v1.2 that adds a sixth change to the
fold and reports the combined result, which is the path v1.0 and v1.1 both
took.

## Release-scale result: the compact-scale conclusion does not hold

Option 1 was run. Three seeds, 6,000 steps, 49,152,000 tokens per arm per seed,
at the release architecture on an a GPU: d_model 384, 3 cells, 6 recurrences,
binding block enabled, the 12,000-entry tokenizer, relation objective at weight
0.20 in every arm.

| Arm | Parameters | GFLOPs | Validation loss | Paired delta | Strict pairs | Margin |
|---|---:|---:|---:|---:|---:|---:|
| CFRD | 11,634,459 | 17.31 | **3.3765** (sd 0.0131) | reference | **0.413** | +2.46 |
| Dense, parameter-matched | 11,669,377 | 11.95 | 3.4377 (sd 0.0034) | +0.0612 | 0.260 | +2.46 |
| Dense, compute-matched | 16,943,233 | 17.34 | 3.3977 (sd 0.0169) | +0.0212 | 0.320 | +2.46 |

A positive delta means CFRD won. Both deltas are positive on all three seeds
and exceed the spread of the paired differences.

**CFRD wins both comparisons at release scale.** The compute-matched result is
the stronger one: that arm carries 45.6% more parameters at the same FLOPs and
still loses by 0.0212 nats. On held-out binding CFRD reaches 0.413 strict pairs
with every seed above the 0.25 chance level (0.46, 0.42, 0.36), while the
parameter-matched dense arm straddles chance at 0.260 (0.24, 0.32, 0.22).

This directly reverses the compact-scale table. Both results are real at their
own scale, and the difference between them is the scale and budget: 11.6M
parameters and 49.2M tokens per arm here, against 3.3M parameters and 0.6M
tokens there. That is the limit the compact-scale section listed first, and it
turned out to be the one that mattered. The compact-scale sections are kept
unchanged rather than rewritten, because the finding they report is accurate
where it was measured and the reversal is the more useful record.

### What this result does not yet establish

Three confounds sit between "CFRD won this table" and "folding is why", and
none of them is addressed by the run above.

- **Deep supervision is not ablated.** CFRD optimizes `final_loss + 0.15 x
  mean(auxiliary exits)` at depths 2 and 4, so it receives gradient signal the
  dense arms never get. A dense decoder can carry auxiliary heads at
  intermediate layers just as easily. Part of the margin may be deep
  supervision rather than the fold, and `aux_exit_loss_weight = 0` measures it.
- **The learning rate favors CFRD by construction.** Every arm shares 4e-4,
  which is the project's own CFRD-tuned value, with warmup and no decay. This
  was recorded earlier as a reason a baseline win would be the stronger
  evidence; the same reasoning now weakens a CFRD win. A short sweep per
  architecture is needed before the margin can be attributed to structure.
- **The budget is still 15x short of a release run**, 49.2M tokens per arm
  against 753.7M for the v1.1 checkpoint. The direction of the scale effect is
  now known, but not where it saturates.

No ablation arms were run at this scale, so which component produces the win is
unmeasured. At compact scale removing the binding block cost the most, but that
table's conclusions did not survive the change of scale and there is no reason
to assume its component ranking did either.

### Where this leaves v1.2

v1.2 is a CFRD model. Option 2, making the dense configuration the release
path, is withdrawn: it rested on the compact-scale table, which the release
scale contradicts.

The order of work is fixed by what is unmeasured rather than by what is
promising:

1. Rerun this table with `--ablate` at release scale, to find which component
   produces the win rather than crediting the architecture as a whole.
2. Ablate deep supervision, by setting `aux_exit_loss_weight = 0`, and add
   auxiliary heads to the dense baseline. This is the confound most likely to
   explain the margin without involving the fold at all.
3. Sweep the learning rate per architecture, so the shared CFRD-tuned value
   stops being a thumb on the scale.
4. Only then extend to a release budget and train a v1.2 checkpoint.

The rule that produced v1.0 and v1.1's unattributable results still applies: a
release may change one thing at a time, and this table is the first evidence
Haru has ever had that the architecture does anything at all.

## Required validation and limits

The implementation tests causality, serialization, gradient flow, deterministic
sampler resume, split leakage, and real SentencePiece candidate encoding. These
are code-level checks; they do not demonstrate that an untrained configuration
has become a better language model.

After full training, evaluation should report at least:

- language-model validation loss at recurrent depths 2, 4, and 6;
- decision accuracy and strict pair accuracy overall and per relation family;
- preference-flip rate, mean signed margin, and multiple training seeds;
- both matched dense baselines, trained on identical windows under an
  identical schedule, with the relation objective enabled in every arm so the
  architecture and the objective stay separable;
- ablations for the third cell, binding block, vocabulary size, and relation
  loss rather than attributing a combined gain to one component;
- free-generation quality and memorization checks outside synthetic templates.

This design targets entity and role hallucinations caused by weak contextual
binding. It cannot guarantee factual truth, recover knowledge absent from the
training data, or turn Haru into an instruction-following assistant. Those
claims require separate data, retrieval, calibration, and safety evaluation.
