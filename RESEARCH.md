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
