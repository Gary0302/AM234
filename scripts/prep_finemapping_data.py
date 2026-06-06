"""Generate all data needed for fine-mapping without any additional downloads.

Outputs (written to /home/gary/AM234/data/):
  gwas/                    - 5 GWAS loci TSVs (SORT1, PCSK9, HMGCR, LDLR, APOE)
  ism/ism_scores.tsv       - ISM scores for every SNP in the 5 loci (v7 model)
  ld/<LOCUS>_snps.tsv      - 1000G EUR SNP panel per locus (rsid, pos, ref, alt)
  ld/<LOCUS>_ld.npy        - LD matrix (p x p, float32) matching the SNP panel

Run from the AM234 directory:
    cd /home/gary/AM234
    python3 scripts/prep_finemapping_data.py
"""
import sys, pathlib, urllib.request, csv, os, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cyvcf2
from pyfaidx import Fasta

# ── Paths ──────────────────────────────────────────────────────────────────────
EXP_DIR  = pathlib.Path('/home/gary/snowball/experiments/exp82_genomics_ordersnn')
REPO_DIR = pathlib.Path('/home/gary/AM234')

GWAS_SRC = EXP_DIR / 'data' / 'gwas' / 'loci'
HG19_DIR = EXP_DIR / 'data' / 'hg19'
CKPT_V7  = EXP_DIR / 'runs' / 'spike_attn_v7' / 'best.pt'

OUT_GWAS = REPO_DIR / 'data' / 'gwas'
OUT_ISM  = REPO_DIR / 'data' / 'ism'
OUT_LD   = REPO_DIR / 'data' / 'ld'

for d in [OUT_GWAS, OUT_ISM, OUT_LD]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EXP_DIR / 'models'))
sys.path.insert(0, str(EXP_DIR))

LOCI = [
    {"name": "SORT1",  "chrom": "chr1",  "vcf_chrom": "1",  "center": 109817590,
     "causal": {"rs12740374": 109817590}},
    {"name": "PCSK9",  "chrom": "chr1",  "vcf_chrom": "1",  "center":  55505647,
     "causal": {"rs11591147":  55505647}},
    {"name": "HMGCR",  "chrom": "chr5",  "vcf_chrom": "5",  "center":  74651084,
     "causal": {"rs17238484": 74651084, "rs12916": 74656539}},
    {"name": "LDLR",   "chrom": "chr19", "vcf_chrom": "19", "center":  11202306,
     "causal": {"rs6511720":  11202306}},
    {"name": "APOE",   "chrom": "chr19", "vcf_chrom": "19", "center":  45411941,
     "causal": {"rs429358": 45411941, "rs7412": 45412079}},
]
WINDOW = 300_000
VCF_TPL = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
           "ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")

FLANK     = 500
N_TRACKS  = 919
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# ── Step 1: Copy GWAS files ────────────────────────────────────────────────────
def copy_gwas():
    print("\n[1/3] Copying GWAS loci files...", flush=True)
    for locus in LOCI:
        src = GWAS_SRC / f"{locus['name']}_LDL.tsv"
        dst = OUT_GWAS / f"{locus['name']}_LDL.tsv"
        shutil.copy2(src, dst)
        df = pd.read_csv(dst, sep='\t')
        print(f"  {locus['name']}: {len(df)} SNPs → {dst}", flush=True)


# ── Step 2: ISM scoring ────────────────────────────────────────────────────────
def one_hot(seq):
    idx = torch.tensor([NUC_TO_IDX.get(c.upper(), 0) for c in seq], dtype=torch.long)
    return F.one_hot(idx, num_classes=4).float().t()  # [4, 1000]


def build_snp_input():
    """Collect all SNPs across 5 loci into one input list with locus-encoded names."""
    snps = []
    for locus_idx, locus in enumerate(LOCI):
        gwas = pd.read_csv(GWAS_SRC / f"{locus['name']}_LDL.tsv", sep='\t')
        for snp_idx, row in enumerate(gwas.itertuples(index=False)):
            chrom = row.chr if hasattr(row, 'chr') else locus['chrom']
            pos   = int(row.pos)
            # A2 = non-effect allele, treat as ref; A1 = effect allele, treat as alt
            ref   = str(row.A2).upper()
            alt   = str(row.A1).upper()
            rsid  = str(row.rsid)
            # name encodes locus_idx and snp_idx so collaborator can map back
            name  = f"{rsid}_locus{locus_idx}_{snp_idx}"
            snps.append((chrom, pos, ref, alt, name))
    print(f"  Total SNPs to score: {len(snps)}", flush=True)
    return snps


def run_ism():
    print("\n[2/3] Running ISM scoring (v7 model)...", flush=True)

    from models.spike_attn_v5v6 import MODEL_REGISTRY
    device = torch.device('cuda:0')

    model_name = 'v7'
    assert model_name in MODEL_REGISTRY
    model = MODEL_REGISTRY[model_name]().to(device)
    state = torch.load(CKPT_V7, map_location=device, weights_only=False)
    sd = state.get('model', state) if isinstance(state, dict) else state
    model.load_state_dict(sd)
    model.eval()
    print(f"  Loaded v7 checkpoint ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)",
          flush=True)

    snps = build_snp_input()

    # Pre-load chromosomes
    needed_chroms = sorted({s[0] for s in snps})
    fa = {}
    for c in needed_chroms:
        fa_path = HG19_DIR / f'{c}.fa'
        fa[c] = Fasta(str(fa_path))
        print(f"  loaded genome: {c}", flush=True)

    rows = []
    with torch.no_grad():
        for i, (chrom, pos, ref, alt, name) in enumerate(snps):
            start_0 = pos - 1 - FLANK
            end_0   = pos - 1 + FLANK
            if start_0 < 0:
                rows.append({'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt,
                             'name': name, 'sum_abs_delta': 'NA', 'max_abs_delta': 'NA'})
                continue

            seq = str(fa[chrom][chrom][start_0:end_0]).upper()
            if len(seq) != 1000:
                rows.append({'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt,
                             'name': name, 'sum_abs_delta': 'NA', 'max_abs_delta': 'NA'})
                continue

            hg19_base = seq[FLANK]
            if hg19_base != ref:
                ref = hg19_base  # use hg19 as ref

            seq_r = list(seq); seq_r[FLANK] = ref
            seq_a = list(seq); seq_a[FLANK] = alt
            x_ref = one_hot(''.join(seq_r)).unsqueeze(0).to(device)
            x_alt = one_hot(''.join(seq_a)).unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                lr = model(x_ref)
                la = model(x_alt)
            p_ref = torch.sigmoid(lr.float()).squeeze(0).cpu().numpy()
            p_alt = torch.sigmoid(la.float()).squeeze(0).cpu().numpy()
            delta = p_alt - p_ref
            sum_d = float(np.abs(delta).sum())
            max_d = float(np.abs(delta).max())

            rows.append({'chrom': chrom, 'pos': pos, 'ref': ref, 'alt': alt,
                         'name': name, 'sum_abs_delta': f'{sum_d:.6f}',
                         'max_abs_delta': f'{max_d:.6f}'})

            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(snps)}] {name}  sum|Δ|={sum_d:.4f}", flush=True)

    out_path = OUT_ISM / 'ism_scores.tsv'
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh,
                           fieldnames=['chrom', 'pos', 'ref', 'alt', 'name',
                                       'sum_abs_delta', 'max_abs_delta'],
                           delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    print(f"  ISM scores → {out_path}  ({len(rows)} rows)", flush=True)

    # Also write to /tmp/ism_v7/ so existing susie_ism_pipeline.py still works
    tmp_dir = pathlib.Path('/tmp/ism_v7')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, tmp_dir / 'ism_scores.tsv')
    print(f"  Also copied → {tmp_dir}/ism_scores.tsv", flush=True)


# ── Step 3: 1000G LD matrices ──────────────────────────────────────────────────
def get_eur_samples():
    url = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
           "integrated_call_samples_v3.20130502.ALL.panel")
    print("  Fetching EUR sample list...", flush=True)
    resp = urllib.request.urlopen(url, timeout=60)
    samples = set()
    for line in resp.read().decode().splitlines()[1:]:
        p = line.split('\t')
        if len(p) >= 3 and p[2] == 'EUR':
            samples.add(p[0])
    print(f"  {len(samples)} EUR samples", flush=True)
    return samples


def fetch_ld_and_save(locus, eur_samples):
    chrom = locus['vcf_chrom']
    lo    = locus['center'] - WINDOW
    hi    = locus['center'] + WINDOW
    url   = VCF_TPL.format(c=chrom)

    vcf      = cyvcf2.VCF(url)
    all_samp = list(vcf.samples)
    eur_idx  = np.array([i for i, s in enumerate(all_samp) if s in eur_samples])
    print(f"  {locus['name']}: {len(eur_idx)} EUR samples", flush=True)

    pos_list, rsid_list, ref_list, alt_list, geno_list = [], [], [], [], []
    for v in vcf(f"{chrom}:{lo}-{hi}"):
        if v.var_type != 'snp' or ',' in ','.join(v.ALT):
            continue
        gt = v.gt_types[eur_idx].astype(float)
        gt[gt == 3] = np.nan
        maf = np.nanmean(gt) / 2
        if maf < 0.01 or maf > 0.99:
            continue
        pos_list.append(v.POS)
        rsid_list.append(v.ID or f"{chrom}:{v.POS}")
        ref_list.append(v.REF)
        alt_list.append(v.ALT[0])
        geno_list.append(gt)

    G = np.vstack(geno_list).T.astype(float)
    cmeans = np.nanmean(G, axis=0)
    for j in range(G.shape[1]):
        mask = np.isnan(G[:, j])
        if mask.any():
            G[mask, j] = cmeans[j]
    mu  = G.mean(0); sig = G.std(0); sig[sig == 0] = 1
    Gs  = (G - mu) / sig
    LD  = np.clip(Gs.T @ Gs / Gs.shape[0], -1, 1).astype(np.float32)

    snp_df = pd.DataFrame({'rsid': rsid_list, 'pos': pos_list,
                            'ref':  ref_list,  'alt': alt_list})
    print(f"  {locus['name']}: {len(snp_df)} SNPs × LD {LD.shape}", flush=True)

    snp_path = OUT_LD / f"{locus['name']}_snps.tsv"
    ld_path  = OUT_LD / f"{locus['name']}_ld.npy"
    snp_df.to_csv(snp_path, sep='\t', index=False)
    np.save(ld_path, LD)
    print(f"  Saved → {snp_path}  {ld_path}", flush=True)


def compute_ld():
    print("\n[3/3] Fetching 1000G EUR LD matrices...", flush=True)
    eur_samples = get_eur_samples()
    for locus in LOCI:
        try:
            fetch_ld_and_save(locus, eur_samples)
        except Exception as e:
            print(f"  ERROR {locus['name']}: {e}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    copy_gwas()
    run_ism()
    compute_ld()
    print("\nDone. All data written to /home/gary/AM234/data/", flush=True)
