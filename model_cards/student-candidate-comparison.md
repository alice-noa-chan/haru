## Released models and student candidates

Two students were compared at exactly **13,688,705 parameters** and
**500,039,680 completed distillation tokens** each. Both are publicly available.

| Model | Role | Repository |
|---|---|---|
| Gated8-13m | Selected student, non-IT | [haru_3-student-base](https://huggingface.co/alice-noa-chan/haru_3-student-base) |
| Gated8-13m IT | IT version of the selected student | [haru_3-student-chat](https://huggingface.co/alice-noa-chan/haru_3-student-chat) |
| PairShare8-r48 | Unselected experimental student, non-IT | [haru_3-student-pairshare-base](https://huggingface.co/alice-noa-chan/haru_3-student-pairshare-base) |
| Teacher base | Frozen non-IT teacher | [haru_3-teacher-base](https://huggingface.co/alice-noa-chan/haru_3-teacher-base) |
| Teacher IT | Frozen IT teacher | [haru_3-teacher-chat](https://huggingface.co/alice-noa-chan/haru_3-teacher-chat) |

PairShare shares each SwiGLU FFN across two adjacent layers and adds a rank-48
linear residual adapter per layer. Its attention layers and KV caches remain
independent. Gated8 uses independent FFNs with width 512; PairShare uses width 960.

| Candidate | Validation story BPC (lower is better) | Training-host CPU generation, 4 threads |
|---|---:|---:|
| PairShare8-r48 | 0.945834 | 93.28 tokens/s |
| Gated8-13m | 0.949001 | 110.63 tokens/s |

The paired document-bootstrap 95% interval for PairShare minus Gated validation
BPC was **[-0.009002, 0.003038]**, including zero. The predeclared selection rule
therefore chose the faster CPU model, **Gated8-13m**. Test scores did not drive
selection. This is one training seed; the bootstrap does not measure variation
between training seeds. PairShare is preserved for architecture experiments;
**no PairShare IT checkpoint was trained**.

See the [Haru collection](https://huggingface.co/collections/alice-noa-chan/haru-6abb683d0a8b9677162d7704),
[training instructions](https://github.com/alice-noa-chan/haru/blob/main/TRAINING.md),
and [completed evaluation](https://github.com/alice-noa-chan/haru/blob/main/results/student_final_evaluation.json).
