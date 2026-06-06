"""SuSiE fine-mapping pipeline with ISM functional priors.

Loads pre-computed data from data/ in this repo — no internet access required.

For each of 5 lipid loci, runs SuSiE-RSS and compares PIP of known causal
SNPs with and without ISM functional priors.

Usage:
    cd /path/to/AM234
    python3 scripts/susie_ism_pipeline.py
"""
import pathlib
import numpy as np
import pandas as pd

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
GWAS_DIR = REPO_DIR / 'data' / 'gwas'
ISM_FILE = REPO_DIR / 'data' / 'ism' / 'ism_scores.tsv'
LD_DIR   = REPO_DIR / 'data' / 'ld'
OUT_FILE = REPO_DIR / 'data' / 'susie_results.tsv'

LOCI = [
    {"name": "SORT1",  "causal": {"rs12740374": 109817590}},
    {"name": "PCSK9",  "causal": {"rs11591147":  55505647}},
    {"name": "HMGCR",  "causal": {"rs17238484": 74651084, "rs12916": 74656539}},
    {"name": "LDLR",   "causal": {"rs6511720":  11202306}},
    {"name": "APOE",   "causal": {"rs429358": 45411941, "rs7412": 45412079}},
]


# ── SuSiE-RSS ────────────────────────────────────────────────────────────────
def susie_rss(z, R, N, L=5, prior_var=0.1, pi0=None, max_iter=100, tol=1e-3):
    """Sum of Single Effects RSS.  Returns pip [p], alpha [L,p]."""
    p = len(z)
    if pi0 is None:
        pi0 = np.ones(p) / p
    pi0 = pi0 / pi0.sum()

    alpha = np.tile(pi0, (L, 1))
    mu    = np.zeros((L, p))
    diag_R = np.clip(np.diag(R), 1e-6, None)

    for _ in range(max_iter):
        alpha_old = alpha.copy()
        for l in range(L):
            z_res = z - sum(R @ (alpha[k] * mu[k]) for k in range(L) if k != l)
            s2    = N * prior_var * diag_R
            lbf   = 0.5 * (np.log(1.0 / (1.0 + s2)) + z_res**2 * s2 / (1.0 + s2))
            log_w = lbf + np.log(pi0 + 1e-300)
            log_w -= log_w.max()
            w = np.exp(log_w)
            alpha[l]  = w / w.sum()
            sigma2_l  = prior_var * diag_R / (1.0 + s2)
            mu[l]     = sigma2_l * N * z_res
        if np.abs(alpha - alpha_old).max() < tol:
            break

    pip = 1.0 - np.prod(1.0 - alpha, axis=0)
    return pip, alpha


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_precomputed(locus):
    """Load pre-computed 1000G EUR SNP panel and LD matrix for a locus."""
    snp_df = pd.read_csv(LD_DIR / f"{locus['name']}_snps.tsv", sep='\t')
    LD     = np.load(LD_DIR / f"{locus['name']}_ld.npy")
    snp_df['rsid_l'] = snp_df['rsid'].str.lower()
    return snp_df, LD


def match_gwas(locus, snp_df):
    """Match GWAS z-scores to the 1000G SNP panel by rsid or position."""
    gwas = pd.read_csv(GWAS_DIR / f"{locus['name']}_LDL.tsv", sep='\t')
    gwas['rsid_l'] = gwas['rsid'].str.lower()
    z_by_rsid  = dict(zip(gwas['rsid_l'], gwas['z']))
    z_by_pos   = dict(zip(gwas['pos'],    gwas['z']))
    n_by_rsid  = dict(zip(gwas['rsid_l'], gwas['N']))
    n_by_pos   = dict(zip(gwas['pos'],    gwas['N']))

    z_vals, n_vals, keep_idx = [], [], []
    for i, row in snp_df.iterrows():
        z = z_by_rsid.get(row['rsid_l']) or z_by_pos.get(row['pos'])
        n = n_by_rsid.get(row['rsid_l']) or n_by_pos.get(row['pos'])
        if z is not None and np.isfinite(float(z)):
            z_vals.append(float(z))
            n_vals.append(float(n or 89138))
            keep_idx.append(i)

    sub = snp_df.loc[keep_idx].copy().reset_index(drop=True)
    sub['z'] = z_vals
    sub['N'] = n_vals
    print(f"  {len(sub)} SNPs with GWAS z-scores", flush=True)
    return sub


def build_ism_prior(sub_df, ism_df):
    """Build ISM prior vector over all SNPs in sub_df.

    ism_df name column format: "{rsid}_locus{i}_{j}"
    Match by rsid (first token) with pos fallback.
    """
    p  = len(sub_df)
    pi = np.ones(p) / p

    ism_by_rsid = {}
    ism_by_pos  = {}
    for _, r in ism_df.iterrows():
        try:
            score = float(r['sum_abs_delta'])
        except (ValueError, TypeError):
            continue
        rsid = r['name'].split('_')[0].lower()
        ism_by_rsid[rsid] = score
        ism_by_pos[int(r['pos'])] = score

    for i, row in sub_df.iterrows():
        score = ism_by_rsid.get(row['rsid_l']) or ism_by_pos.get(int(row['pos']))
        if score is not None:
            pi[i] += 10 * score

    return pi / pi.sum()


def find_causal_idx(sub_df, causal_dict):
    """causal_dict: {rsid: hg19_pos}. Match by position."""
    pos_to_rsid = {pos: rsid for rsid, pos in causal_dict.items()}
    idx = {}
    for i, row in sub_df.iterrows():
        if row['pos'] in pos_to_rsid:
            idx[pos_to_rsid[row['pos']]] = i
    return idx


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ism_df = pd.read_csv(ISM_FILE, sep='\t')
    print(f"Loaded ISM scores: {len(ism_df)} SNPs", flush=True)

    all_rows = []

    for locus in LOCI:
        print(f"\n{'='*60}\nLocus: {locus['name']}", flush=True)

        snp_df, LD = load_precomputed(locus)
        sub = match_gwas(locus, snp_df)
        if len(sub) < 50:
            print(f"  Too few matched SNPs ({len(sub)}), skipping", flush=True)
            continue

        # Subset LD to matched SNPs
        orig_pos_map = {row['pos']: i for i, row in snp_df.iterrows()}
        sub_orig_idx = np.array([orig_pos_map[p] for p in sub['pos'] if p in orig_pos_map])
        if len(sub_orig_idx) != len(sub):
            mask = [p in orig_pos_map for p in sub['pos']]
            sub = sub[mask].reset_index(drop=True)
            sub_orig_idx = np.array([orig_pos_map[p] for p in sub['pos']])

        LD_sub = LD[np.ix_(sub_orig_idx, sub_orig_idx)]
        z_vals = sub['z'].values.astype(float)
        N_val  = float(sub['N'].median())

        causal_idx = find_causal_idx(sub, locus['causal'])
        print(f"  Causal SNPs in panel: {list(causal_idx.keys())}", flush=True)

        # Baseline SuSiE
        pip_base, _ = susie_rss(z_vals, LD_sub, N_val)
        row_base = {'locus': locus['name'], 'model': 'baseline', 'n_snps': len(sub)}
        for rsid, idx in causal_idx.items():
            row_base[rsid] = round(float(pip_base[idx]), 4)
        all_rows.append(row_base)

        # SuSiE with ISM prior
        pi = build_ism_prior(sub, ism_df)
        pip_ism, _ = susie_rss(z_vals, LD_sub, N_val, pi0=pi)
        row_ism = {'locus': locus['name'], 'model': 'ISM_v7', 'n_snps': len(sub)}
        for rsid, idx in causal_idx.items():
            row_ism[rsid] = round(float(pip_ism[idx]), 4)
        if causal_idx:
            causal_pips = [row_ism.get(r, float('nan')) for r in causal_idx]
            print(f"  ISM_v7: causal PIPs = {causal_pips}", flush=True)
        all_rows.append(row_ism)

    results = pd.DataFrame(all_rows)
    print(f"\n{'='*60}")
    print(results.to_string(index=False))
    results.to_csv(OUT_FILE, sep='\t', index=False)
    print(f"\nSaved → {OUT_FILE}", flush=True)


if __name__ == '__main__':
    main()
