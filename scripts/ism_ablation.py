"""ISM scoring for fine-mapping priors.

For each SNP, runs model(ref_seq) and model(alt_seq), collects per-track
probability vectors, and computes a single-score summary.

Usage:
    MODEL=v7  CKPT=runs/spike_attn_v7/best.pt  SNP_FILE=snps.tsv  python3 ism_ablation.py

SNP_FILE format (TSV, no header required; # lines are comments):
    chrom  pos(1-based)  ref  alt  [rsid_or_name]

Outputs written to OUT_DIR (default: runs/ism_<MODEL>/):
    ref_matrix.npy     float32 [n_SNPs × 919]
    alt_matrix.npy     float32 [n_SNPs × 919]
    delta_matrix.npy   float32 [n_SNPs × 919]  (alt - ref)
    ism_scores.tsv     chrom, pos, ref, alt, name, sum_abs_delta, max_abs_delta

To download missing chromosomes (hg19):
    wget -P data/hg19 https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chrN.fa.gz
    gunzip data/hg19/chrN.fa.gz && samtools faidx data/hg19/chrN.fa
"""
import os, sys, csv, pathlib
import torch
import torch.nn.functional as F
import numpy as np
from pyfaidx import Fasta

BASE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from models.spike_attn_v5v6 import MODEL_REGISTRY

# ── Config from env ──────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get('MODEL', 'v7')
CKPT       = pathlib.Path(os.environ.get('CKPT', f'runs/spike_attn_{MODEL_NAME}/best.pt'))
SNP_FILE   = pathlib.Path(os.environ.get('SNP_FILE', ''))
HG19_DIR   = pathlib.Path(os.environ.get('HG19_DIR', str(BASE_DIR / 'data' / 'hg19')))
OUT_DIR    = pathlib.Path(os.environ.get('OUT_DIR', f'runs/ism_{MODEL_NAME}'))
DEVICE     = os.environ.get('DEVICE', 'cuda:0')
FLANK      = 500     # → 1000 bp window
N_TRACKS   = 919

NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def one_hot(seq: str) -> torch.Tensor:
    idx = torch.tensor([NUC_TO_IDX.get(c.upper(), 0) for c in seq], dtype=torch.long)
    return F.one_hot(idx, num_classes=4).float().t()   # [4, 1000]


def load_snps(path: pathlib.Path):
    """Parse TSV: chrom, pos(1-based), ref, alt, [name].  # = comment."""
    snps = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2].upper(), parts[3].upper()
            name = parts[4] if len(parts) > 4 else f'{chrom}:{pos}:{ref}>{alt}'
            snps.append((chrom, pos, ref, alt, name))
    return snps


def load_model(model_name, ckpt_path, device):
    assert model_name in MODEL_REGISTRY, f'MODEL must be one of {list(MODEL_REGISTRY)}'
    model = MODEL_REGISTRY[model_name]().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # checkpoint may be wrapped as {'model': ...} or bare state_dict
    sd = state.get('model', state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    if not SNP_FILE or not SNP_FILE.exists():
        print(f'ERROR: SNP_FILE not set or does not exist. Set SNP_FILE=path/to/snps.tsv')
        sys.exit(1)
    if not CKPT.exists():
        print(f'ERROR: checkpoint not found: {CKPT}')
        sys.exit(1)

    device = torch.device(DEVICE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'[ism_ablation] model={MODEL_NAME}  ckpt={CKPT}  device={device}')
    model = load_model(MODEL_NAME, CKPT, device)
    pc = sum(p.numel() for p in model.parameters())
    print(f'  params={pc/1e6:.3f}M')

    snps = load_snps(SNP_FILE)
    print(f'  {len(snps)} SNPs from {SNP_FILE}')

    # Pre-load required chromosomes
    needed_chroms = sorted({s[0] for s in snps})
    fa = {}
    for c in needed_chroms:
        fa_path = HG19_DIR / f'{c}.fa'
        if not fa_path.exists():
            print(f'  ERROR: {fa_path} not found.')
            print(f'  Download: wget -P {HG19_DIR} https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/{c}.fa.gz')
            print(f'            gunzip {HG19_DIR}/{c}.fa.gz && samtools faidx {HG19_DIR}/{c}.fa')
            sys.exit(1)
        fa[c] = Fasta(str(fa_path))
        print(f'  loaded {c}')

    ref_mat  = np.zeros((len(snps), N_TRACKS), dtype=np.float32)
    alt_mat  = np.zeros((len(snps), N_TRACKS), dtype=np.float32)
    rows = []

    with torch.no_grad():
        for i, (chrom, pos, ref, alt, name) in enumerate(snps):
            start_0 = pos - 1 - FLANK
            end_0   = pos - 1 + FLANK
            if start_0 < 0:
                print(f'  WARN: {name} too close to chromosome start, skipping')
                rows.append({'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt, 'name': name,
                             'sum_abs_delta': 'NA', 'max_abs_delta': 'NA', 'warn': 'near_start'})
                continue

            seq = str(fa[chrom][chrom][start_0:end_0]).upper()
            if len(seq) != 1000:
                print(f'  WARN: {name} window length={len(seq)}, skipping')
                continue

            # Check ref allele
            hg19_base = seq[FLANK]
            if hg19_base != ref:
                print(f'  WARN: {name}: expected ref={ref}, hg19 has {hg19_base} — using hg19 base as ref')
                ref = hg19_base

            seq_r = list(seq); seq_r[FLANK] = ref
            seq_a = list(seq); seq_a[FLANK] = alt
            x_ref = one_hot(''.join(seq_r)).unsqueeze(0).to(device)   # [1, 4, 1000]
            x_alt = one_hot(''.join(seq_a)).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits_ref = model(x_ref)   # [1, 919]
                logits_alt = model(x_alt)
            p_ref = torch.sigmoid(logits_ref.float()).squeeze(0).cpu().numpy()
            p_alt = torch.sigmoid(logits_alt.float()).squeeze(0).cpu().numpy()

            ref_mat[i] = p_ref
            alt_mat[i] = p_alt
            delta = p_alt - p_ref
            sum_d = float(np.abs(delta).sum())
            max_d = float(np.abs(delta).max())

            rows.append({'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt, 'name': name,
                         'sum_abs_delta': f'{sum_d:.6f}', 'max_abs_delta': f'{max_d:.6f}', 'warn': ''})
            if (i + 1) % 10 == 0 or i == 0:
                print(f'  [{i+1}/{len(snps)}] {name}  sum|Δ|={sum_d:.4f}  max|Δ|={max_d:.4f}')

    delta_mat = alt_mat - ref_mat

    np.save(OUT_DIR / 'ref_matrix.npy',   ref_mat)
    np.save(OUT_DIR / 'alt_matrix.npy',   alt_mat)
    np.save(OUT_DIR / 'delta_matrix.npy', delta_mat)

    with open(OUT_DIR / 'ism_scores.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['chrom','pos','ref','alt','name',
                                           'sum_abs_delta','max_abs_delta','warn'],
                           delimiter='\t')
        w.writeheader()
        w.writerows(rows)

    print(f'\nOutputs written to {OUT_DIR}/')
    print(f'  ref_matrix.npy   {ref_mat.shape}')
    print(f'  alt_matrix.npy   {alt_mat.shape}')
    print(f'  delta_matrix.npy {delta_mat.shape}')
    print(f'  ism_scores.tsv   ({len(rows)} rows)')


if __name__ == '__main__':
    main()
