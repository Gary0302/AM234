# DeepSEA Ablation Study — Attention Architecture for Chromatin Feature Prediction

**Course**: AM234  
**Task**: Predict 919 binary chromatin accessibility features from 1000 bp DNA windows (DeepSEA dataset, hg19).  
**Input**: `[batch, 4, 1000]` one-hot encoded DNA sequence  
**Output**: 919 logits (one per chromatin track — DNase, TF binding, histone marks)  
**Primary metric**: AUROC averaged across all 919 tracks

---

## Key Finding

The dominant factor driving AUROC is **how the model embeds each DNA position before attention**, not the number of attention layers.

| Model | Embedding | Attention | Test AUROC |
|-------|-----------|-----------|------------|
| No-context baseline | Raw 4-dim one-hot | 1 layer (intra-window) | 0.705 |
| Small-window control | Raw 4-dim one-hot | 1 layer (intra-window) | 0.702 |
| Causal 5-mer + single attention | Causal conv k=5 | 1 layer (intra-window) | **0.779** |
| Causal 5-mer + dual attention | Causal conv k=5 | 2 layers (intra + inter) | **0.787** |
| No-context + dual attention | Raw 4-dim one-hot | 2 layers (intra + inter) | 0.699 |

**Adding a second attention layer only helps when the embedding is already rich (CausalConv).  
Without CausalConv, adding more attention layers makes things slightly worse.**

For full architecture details, ablation design, and attention math see [MODELS.md](MODELS.md).

---

## What Is CausalConv?

Each DNA position is embedded using a 1D convolution over the 5 bases ending at that position (k=5, left-only padding). This gives each position a **5-mer context** — 4⁵ = 1024 possible values instead of just 4. The "causal" part means the model only looks backward (no lookahead), which is appropriate for a scanning-style model.

## What Is Dual Attention?

- **Level 1 (intra-window)**: The 1000 bp sequence is split into 50 non-overlapping 20 bp windows. Self-attention runs independently inside each window. This captures short-range motif patterns (TF binding sites are typically 6–20 bp).
- **Level 2 (inter-window)**: After pooling each window to one vector, a second self-attention runs across all 50 windows. This captures long-range co-occurrence of motifs across the full 1000 bp region.

---

## Functional Fine-Mapping Integration

The models can score any SNP by comparing model output between the reference and alternative allele. These **ISM (In Silico Mutagenesis) scores** can serve as functional priors for Bayesian fine-mapping (e.g., SuSiE).

### Quick start

```bash
# On the H400 server — see DATA.md for server access details
cd /home/gary/snowball/experiments/exp82_genomics_ordersnn

# Create a SNP list (TSV: chrom, pos_1based, ref, alt, name)
cat > my_snps.tsv << EOF
chr1    109817590   G   T   rs12740374
chr19   45411941    T   C   rs429358
EOF

# Run ISM scoring
MODEL=CausalConvDualAttn \
CKPT=runs/spike_attn_v7/best.pt \
SNP_FILE=my_snps.tsv \
OUT_DIR=/tmp/ism_output \
DEVICE=cuda:0 \
python3 scripts/ism_ablation.py
```

**Outputs** (in `OUT_DIR/`):
- `ref_matrix.npy` — shape `[n_SNPs, 919]`, reference allele probabilities
- `alt_matrix.npy` — shape `[n_SNPs, 919]`, alternative allele probabilities
- `delta_matrix.npy` — `alt - ref`, signed per-track effect
- `ism_scores.tsv` — one row per SNP: `sum_abs_delta` and `max_abs_delta` (collapse scores for use as SuSiE prior)

### Using ISM scores as SuSiE priors

Pass `sum_abs_delta` (or `max_abs_delta`) as the `pi` prior vector in SuSiE. SNPs with higher ISM scores get a higher prior probability of being causal. See `scripts/susie_ism_pipeline.py` for a full worked example on 5 lipid GWAS loci (SORT1, PCSK9, HMGCR, LDLR, APOE).

---

## Files in This Repo

| File | Description |
|------|-------------|
| `README.md` | This file — overview and quick start |
| `MODELS.md` | Architecture details, ablation design, attention math |
| `DATA.md` | Server paths, checkpoint locations, GWAS data, ISM usage |
| `models/spike_attn_v5v6.py` | All model class definitions (PyTorch) |
| `scripts/ism_ablation.py` | ISM scoring pipeline — input SNP list, output probability matrices |
| `scripts/susie_ism_pipeline.py` | End-to-end SuSiE fine-mapping with ISM priors + 1000G LD |
