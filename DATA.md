# Data & Checkpoints

All data and model weights live on the H400 GPU server.  
Contact Gary Yang (yanggary2388@gmail.com) for server access.

---

## Model Checkpoints

Path prefix: `/home/gary/snowball/experiments/exp82_genomics_ordersnn/runs/`

| Descriptive name | Folder | Test AUROC |
|-----------------|--------|------------|
| SmallWindow (no context, 1-layer attn) | `spike_attn_v6_b/best.pt` | 0.705 |
| CausalConvSingle (5-mer, 1-layer attn) | `spike_attn_v6/best.pt` | 0.779 |
| SmallWindowDual (no context, 2-layer attn) | `spike_attn_v6_b2/best.pt` | 0.699 |
| **CausalConvDual (5-mer, 2-layer attn)** | `spike_attn_v7/best.pt` | **0.787** |

Use `CausalConvDual` for ISM scoring — it gives the strongest functional signals.

---

## Reference Genome (hg19)

Available chromosomes (covers all 5 GWAS loci used in this study):

```
/home/gary/snowball/experiments/exp82_genomics_ordersnn/data/hg19/
  chr1.fa    chr1.fa.fai
  chr5.fa    chr5.fa.fai
  chr19.fa   chr19.fa.fai
```

If your SNPs are on other chromosomes, download the missing ones:
```bash
# Example for chr22
wget -P data/hg19 \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr22.fa.gz
gunzip data/hg19/chr22.fa.gz
samtools faidx data/hg19/chr22.fa
```

---

## GWAS Summary Statistics

GLGC 2013 LDL-C (Willer et al.), pre-extracted for 5 lipid loci (±300 kb):

```
/home/gary/snowball/experiments/exp82_genomics_ordersnn/data/gwas/loci/
  SORT1_LDL.tsv    PCSK9_LDL.tsv    HMGCR_LDL.tsv
  LDLR_LDL.tsv     APOE_LDL.tsv
```

Columns: `rsid  chr  pos  A1  A2  beta  se  z  N  pval`

---

## Running ISM on Your SNP List

### 1. Prepare your SNP file (TSV, no header)

```
chr1    109817590   G   T   rs12740374
chr1     55505647   G   T   rs11591147
chr19    45411941   T   C   rs429358
```

Columns: `chrom  pos(1-based)  ref_allele  alt_allele  [optional_name]`

### 2. Run scoring

```bash
cd /home/gary/snowball/experiments/exp82_genomics_ordersnn

MODEL=CausalConvDual \
CKPT=runs/spike_attn_v7/best.pt \
SNP_FILE=/path/to/your_snps.tsv \
OUT_DIR=/path/to/output \
DEVICE=cuda:0 \
python3 scripts/ism_ablation.py
```

To compare across all 4 models, change `MODEL` and `CKPT` to each checkpoint in turn.

### 3. Output files

| File | Shape | Description |
|------|-------|-------------|
| `ref_matrix.npy` | `[n_SNPs, 919]` | Predicted probabilities for reference allele |
| `alt_matrix.npy` | `[n_SNPs, 919]` | Predicted probabilities for alternative allele |
| `delta_matrix.npy` | `[n_SNPs, 919]` | `alt − ref` (signed per-track effect) |
| `ism_scores.tsv` | `n_SNPs rows` | `sum_abs_delta`, `max_abs_delta` per SNP |

### 4. Using ISM scores as SuSiE priors

`sum_abs_delta` (or `max_abs_delta`) from `ism_scores.tsv` can be used directly as the functional prior `pi` in SuSiE:

```python
import numpy as np, pandas as pd

scores = pd.read_csv("ism_scores.tsv", sep="\t")
# Build pi vector: uniform baseline, boosted at scored positions
pi = np.ones(n_snps_in_locus) / n_snps_in_locus
for snp_idx, score in zip(scores["locus_index"], scores["sum_abs_delta"]):
    pi[snp_idx] += 10 * score
pi /= pi.sum()

# Pass pi to SuSiE as the prior inclusion probability
```

A full worked example covering the 5 GLGC loci (1000G LD + SuSiE-RSS) is in `scripts/susie_ism_pipeline.py`.

---

## Pre-computed ISM Scores

ISM scores for 7 known causal lipid variants, all 4 models (stored on H400):

```
/tmp/ism_v6_b/ism_scores.tsv     SmallWindow
/tmp/ism_v6/ism_scores.tsv       CausalConvSingle
/tmp/ism_v6_b2/ism_scores.tsv    SmallWindowDual
/tmp/ism_v7/ism_scores.tsv       CausalConvDual
```
