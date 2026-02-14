# Zenodo 发布信息

## Basic Information

**Resource type:** Preprint

**Title:** Sparse Feature Analysis of Deep Layer Expansion: Cognitive Geometry and Steering Anatomy of Persona Prompts

**Publication date:** 2026-01-31

**Version:** v4 (2026-02-15)

**Authors/Creators:**
1. Jin, Yanyan (lmxxf@hotmail.com, ORCID: 0009-0008-0169-0409)
2. Zhao, Lei (ORCID: 0009-0008-9765-6837)

**Description:**
```
Zhao (2026) demonstrated that expert-level prompts induce "Deep Layer Expansion"—a 60–100% increase in Effective Intrinsic Dimension (EID) at deep layers. This paper introduces two orthogonal measures—SAF (Sparse Active Features) and EID—to systematically analyze Layer 50 activations across 16 persona prompt conditions and 100 technical topics using Goodfire's Llama-3.3-70B SAE (65,536 features).

Key findings: (1) SAF and EID are orthogonal cognitive measures—expert achieves the highest dimensionality (EID rank 3/16) with the fewest features (SAF rank 13/16), while socratic activates the most features (SAF rank 1/16) but achieves only moderate dimensionality (EID rank 6/16); (2) Any persona prompt yields nEID 1.52–2.18, indicating role assignment alone is sufficient for >50% dimensional expansion; (3) Persona prompts are geometrically approximable as one-dimensional steering vectors (66–82% variance explained by a single direction); (4) Different personas' steering directions form a multi-dimensional "cognitive dimension space"—expert-guru-debugger share a professional depth direction (cosine 0.54–0.65), socratic is nearly orthogonal to all others, and novice-expert cosine is only 0.46 (novice is not an "inverse expert" but an independent dimension).
```

**Keywords:**
- Sparse Autoencoder
- SAE
- Interpretability
- Persona Prompts
- Steering Vector
- Cognitive Geometry
- Feature Activation
- Llama
- Deep Layer Expansion
- LLM
- Effective Intrinsic Dimension
- EID
- Residual Stream
- Multi-dimensional Manifold

## Related Works

| Relation | Identifier |
|----------|------------|
| Is supplement to | https://zenodo.org/records/18410085 |
| Is supplement to | https://github.com/lmxxf/llama3-70b-sae-inspect |

## Version History

- **v1 (2026-01-31):** Initial release - activation count and exclusive feature analysis
- **v2 (2026-02-01):** Added AutoInterp analysis (Section 3.5), discovered semantic subdivision structure
- **v3 (2026-02-02):** Added UMAP visualization (Section 4.5), showing spatial separation and SAE denoising effect
- **v4 (2026-02-15):** Major upgrade — added Persona experiment (16 conditions × 100 topics), EID analysis, Steering experiment, revealing multi-dimensional manifold structure of the cognitive dimension space

## Zenodo Record

- **v1:** https://zenodo.org/records/18441075
- **v2:** https://zenodo.org/records/18449508
- **v3:** https://zenodo.org/records/18457875
- **v4:** https://zenodo.org/records/18643112

## Other Fields

其他字段留空。
