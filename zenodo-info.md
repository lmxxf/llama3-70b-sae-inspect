# Zenodo 发布信息

## Basic Information

**Resource type:** Preprint

**Title:** Sparse Feature Analysis of Deep Layer Expansion: A Mechanistic Interpretation via SAE

**Publication date:** 2026-01-31

**Version:** v2 (2026-02-01)

**Authors/Creators:**
1. Jin, Yanyan (lmxxf@hotmail.com, ORCID: 0009-0008-0169-0409)
2. Zhao, Lei (ORCID: 0009-0008-9765-6837)

**Description:**
```
Zhao (2026) demonstrated that expert-level prompts induce "Deep Layer Expansion"—a 60-100% increase in Effective Intrinsic Dimension (EID) at deep layers. However, EID is a global metric that does not reveal which semantic features are activated. In this paper, we apply Sparse Autoencoder (SAE) analysis to decompose the activation differences between prompt styles. Using Goodfire's Llama-3.3-70B SAE (Layer 50, 65,536 features), we find that: (1) "Explain to a novice" activates 17% more features than "explain to an expert" (132.4 vs 113.1 on average); (2) 369 features are exclusively activated by novice prompts vs 208 for expert prompts; (3) 10 features show perfect separation (100% activation in one condition, 0% in the other); (4) These features are prompt-driven, not topic-driven—100% activation rate across all 50 topics within their respective conditions. We propose the "Mode Switch" hypothesis: LLMs contain dedicated features for toggling between teaching and expert communication modes. These findings provide mechanistic evidence that prompt-induced EID differences reflect distinct sparse feature activation patterns, not merely statistical noise.
```

**Keywords:**
- Sparse Autoencoder
- SAE
- Interpretability
- Prompt Engineering
- Feature Activation
- Llama
- Deep Layer Expansion
- LLM
- Mode Switch
- Neural Signature

## Related Works

| Relation | Identifier |
|----------|------------|
| Is supplement to | https://zenodo.org/records/18410085 |
| Is supplement to | https://github.com/lmxxf/llama3-70b-sae-inspect |

## Version History

- **v1 (2026-01-31):** Initial release - activation count and exclusive feature analysis
- **v2 (2026-02-01):** Added feature semantic analysis (Section 3.5), proposed "Mode Switch" hypothesis

## Zenodo Record

- **v1:** https://zenodo.org/records/18441075
- **v2:** https://zenodo.org/records/18448092

## Other Fields

其他字段留空。
