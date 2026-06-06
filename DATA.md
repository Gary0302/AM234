# Data & Checkpoints

Model weights live on the H400 GPU server.  
All fine-mapping data (GWAS, ISM scores, LD matrices) is bundled in this repo under `data/`.

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

Available chromosomes on the H400 server (covers all 5 GWAS loci):

```
/home/gary/snowball/experiments/exp82_genomics_ordersnn/data/hg19/
  chr1.fa    chr1.fa.fai
  chr5.fa    chr5.fa.fai
  chr19.fa   chr19.fa.fai
```

If your SNPs are on other chromosomes:
```bash
wget -P data/hg19 \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr22.fa.gz
gunzip data/hg19/chr22.fa.gz && samtools faidx data/hg19/chr22.fa
```

---

## Fine-Mapping Data (in this repo)

Everything needed to run `scripts/susie_ism_pipeline.py` is in `data/`:

```
data/
  gwas/
    SORT1_LDL.tsv     PCSK9_LDL.tsv     HMGCR_LDL.tsv
    LDLR_LDL.tsv      APOE_LDL.tsv
  ism/
    ism_scores.tsv    (4033 SNPs × 5 loci, CausalConvDual / v7 model)
  ld/
    SORT1_snps.tsv    SORT1_ld.npy    (1590 SNPs)
    PCSK9_snps.tsv    PCSK9_ld.npy    (2233 SNPs)
    HMGCR_snps.tsv    HMGCR_ld.npy    (1409 SNPs)
    LDLR_snps.tsv     LDLR_ld.npy     (1694 SNPs)
    APOE_snps.tsv     APOE_ld.npy     (1970 SNPs)
```

### GWAS files

GLGC 2013 LDL-C (Willer et al.), ±300 kb windows around each locus.  
Columns: `rsid  chr  pos  A1  A2  beta  se  z  N  pval`

### ISM scores (`data/ism/ism_scores.tsv`)

Columns: `chrom  pos  ref  alt  name  sum_abs_delta  max_abs_delta`

The `name` field encodes locus membership: **`{rsid}_locus{i}_{j}`**
- `i` = locus index: 0=SORT1, 1=PCSK9, 2=HMGCR, 3=LDLR, 4=APOE
- `j` = SNP index within that locus (matches row order in the GWAS TSV)

Use this to map ISM scores back to the correct SNP in the LD panel:

```python
import numpy as np, pandas as pd

ism   = pd.read_csv("data/ism/ism_scores.tsv", sep="\t")
snps  = pd.read_csv("data/ld/SORT1_snps.tsv",  sep="\t")   # 1000G panel
LD    = np.load("data/ld/SORT1_ld.npy")                    # shape [p, p]

# Build pi prior: uniform baseline boosted by ISM score
p  = len(snps)
pi = np.ones(p) / p
ism_by_rsid = {r["name"].split("_")[0].lower(): float(r["sum_abs_delta"])
               for _, r in ism.iterrows()
               if r["sum_abs_delta"] not in ("NA", "")}

for i, row in snps.iterrows():
    score = ism_by_rsid.get(row["rsid"].lower())
    if score is not None:
        pi[i] += 10 * score
pi /= pi.sum()
```

### LD matrices (`data/ld/`)

Pre-computed from 1000 Genomes Phase 3 (EUR superpopulation, n=503).  
`{LOCUS}_snps.tsv` — SNP panel: `rsid  pos  ref  alt`  
`{LOCUS}_ld.npy`   — Pearson correlation matrix, float32, shape `[p, p]`

---

## Running the Pipeline

```bash
cd /path/to/AM234
python3 scripts/susie_ism_pipeline.py
```

No internet access or server access required — all data is in `data/`.  
Results are written to `data/susie_results.tsv`.

---

## Running ISM on Your Own SNP List

### 1. Prepare your SNP file (TSV, no header)

```
chr1    109817590   G   T   rs12740374
chr1     55505647   G   T   rs11591147
chr19    45411941   T   C   rs429358
```

Columns: `chrom  pos(1-based)  ref_allele  alt_allele  [optional_name]`

### 2. Run scoring (on H400 server)

```bash
cd /home/gary/snowball/experiments/exp82_genomics_ordersnn

MODEL=v7 \
CKPT=runs/spike_attn_v7/best.pt \
SNP_FILE=/path/to/your_snps.tsv \
HG19_DIR=data/hg19 \
OUT_DIR=/tmp/ism_output \
DEVICE=cuda:0 \
python3 scripts/ism_ablation.py
```

### 3. Output files

| File | Shape | Description |
|------|-------|-------------|
| `ref_matrix.npy` | `[n_SNPs, 919]` | Reference allele predicted probabilities |
| `alt_matrix.npy` | `[n_SNPs, 919]` | Alternative allele predicted probabilities |
| `delta_matrix.npy` | `[n_SNPs, 919]` | `alt − ref` signed per-track effect |
| `ism_scores.tsv` | `n_SNPs rows` | `sum_abs_delta`, `max_abs_delta` per SNP |
