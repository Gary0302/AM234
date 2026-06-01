# Data & Checkpoints

## Model Checkpoints (on H400 GPU server)

All model weights live at:
```
/home/gary/snowball/experiments/exp82_genomics_ordersnn/runs/
  spike_attn_v6_b/best.pt   — ablation: no CausalConv, 1-layer attn   (test AUROC=0.705)
  spike_attn_v6/best.pt     — CausalConv k=5, 1-layer attn             (test AUROC=0.779)
  spike_attn_v6_b2/best.pt  — no CausalConv, 2-layer attn             (test AUROC=0.699)
  spike_attn_v7/best.pt     — CausalConv k=5, 2-layer attn             (test AUROC=0.787)
```

## Reference Genome

hg19 chromosomes used for ISM (chr1, chr5, chr19 only — covers all 5 GWAS loci):
```
/home/gary/snowball/experiments/exp82_genomics_ordersnn/data/hg19/
```

## GWAS Summary Statistics

GLGC 2013 LDL-C (Willer et al.) pre-extracted for 5 loci:
```
/home/gary/snowball/experiments/exp82_genomics_ordersnn/data/gwas/loci/
  SORT1_LDL.tsv   PCSK9_LDL.tsv   HMGCR_LDL.tsv   LDLR_LDL.tsv   APOE_LDL.tsv
```

## ISM Scores (pre-computed)

ISM scoring for 7 known causal SNPs, all 4 models:
```
/tmp/ism_v6_b/ism_scores.tsv
/tmp/ism_v6/ism_scores.tsv
/tmp/ism_v6_b2/ism_scores.tsv
/tmp/ism_v7/ism_scores.tsv
```

## Running ISM on new SNP list

```bash
cd /home/gary/snowball/experiments/exp82_genomics_ordersnn
MODEL=v7 CKPT=runs/spike_attn_v7/best.pt \
    SNP_FILE=your_snps.tsv OUT_DIR=/tmp/ism_output DEVICE=cuda:0 \
    python3 ism_ablation.py
```

SNP file format (TSV): `chrom  pos(1-based)  ref  alt  [name]`
Outputs: `ref_matrix.npy`, `alt_matrix.npy`, `delta_matrix.npy`, `ism_scores.tsv`
