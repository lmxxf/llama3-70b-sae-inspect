# Zenodo 发布信息

## Basic Information

**Resource type:** Preprint

**Title:** Sparse Feature Analysis of Deep Layer Expansion: A Mechanistic Interpretation via SAE

**Publication date:** 2026-01-31

**Version:** v3 (2026-02-02)

**Authors/Creators:**
1. Jin, Yanyan (lmxxf@hotmail.com, ORCID: 0009-0008-0169-0409)
2. Zhao, Lei (ORCID: 0009-0008-9765-6837)

**Description:**
```
Zhao (2026) demonstrated that expert-level prompts induce "Deep Layer Expansion"—a 60-100% increase in Effective Intrinsic Dimension (EID) at deep layers. However, EID is a global metric that does not reveal which semantic features are activated. In this paper, we apply Sparse Autoencoder (SAE) analysis to decompose the activation differences between prompt styles. Using Goodfire's Llama-3.3-70B SAE (Layer 50, 65,536 features), we find that: (1) "Explain to a novice" activates 17% more features than "explain to an expert" (132.4 vs 113.1 on average); (2) 369 features are exclusively activated by novice prompts vs 208 for expert prompts; (3) 10 features show perfect separation between Novice vs Expert conditions; (4) Through AutoInterp analysis (6 conditions × 50 topics = 300 samples), we discover these features exhibit semantic subdivision—encoding distinct dimensions such as "expert identity," "serious attitude," "depth requirement," and "technical analysis"; (5) UMAP visualization confirms that 6 prompt conditions form distinct clusters in both raw activation space and SAE feature space, with SAE acting as a semantic denoiser that merges noise-only conditions (standard/padding/spaces) while preserving semantic distinctions (novice/expert/guru). These findings suggest prompt effects are compositional, with different elements triggering different feature subsets.
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
- Semantic Subdivision
- Neural Signature
- AutoInterp
- UMAP
- Semantic Denoising

## Related Works

| Relation | Identifier |
|----------|------------|
| Is supplement to | https://zenodo.org/records/18410085 |
| Is supplement to | https://github.com/lmxxf/llama3-70b-sae-inspect |

## Version History

- **v1 (2026-01-31):** Initial release - activation count and exclusive feature analysis
- **v2 (2026-02-01):** Added AutoInterp analysis (Section 3.5), discovered semantic subdivision structure
- **v3 (2026-02-02):** Added UMAP visualization (Section 4.5), showing spatial separation and SAE denoising effect

## Zenodo Record

- **v1:** https://zenodo.org/records/18441075
- **v2:** https://zenodo.org/records/18449508
- **v3:** https://zenodo.org/records/18457748

## Other Fields

其他字段留空。
