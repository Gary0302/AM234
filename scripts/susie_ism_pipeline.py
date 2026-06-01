"""Full SuSiE fine-mapping pipeline with ISM functional priors.

For each of 5 lipid loci, runs SuSiE-RSS (own implementation) and compares
PIP of known causal SNPs with and without ISM functional priors.

Usage:
    python3 susie_ism_pipeline.py
"""
import sys, pathlib, urllib.request
import numpy as np
import pandas as pd

BASE     = pathlib.Path('/home/gary/snowball/experiments/exp82_genomics_ordersnn')
LOCI_DIR = BASE / 'data' / 'gwas' / 'loci'
ISM_DIR  = pathlib.Path('/tmp')
OUT_FILE = BASE / 'data' / 'gwas' / 'susie_results.tsv'

LOCI = [
    {"name": "SORT1",  "vcf_chrom": "1",  "center": 109817590,
     "causal": {"rs12740374": 109817590}},
    {"name": "PCSK9",  "vcf_chrom": "1",  "center":  55505647,
     "causal": {"rs11591147":  55505647}},
    {"name": "HMGCR",  "vcf_chrom": "5",  "center":  74651084,
     "causal": {"rs17238484": 74651084, "rs12916": 74656539}},
    {"name": "LDLR",   "vcf_chrom": "19", "center":  11202306,
     "causal": {"rs6511720":  11202306}},
    {"name": "APOE",   "vcf_chrom": "19", "center":  45411941,
     "causal": {"rs429358": 45411941, "rs7412": 45412079}},
]
WINDOW = 300_000
VCF_TPL = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
           "ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")


# ── Self-contained SuSiE-RSS ──────────────────────────────────────────────────
def susie_rss(z, R, N, L=5, prior_var=0.1, pi0=None, max_iter=100, tol=1e-3):
    """Sum of Single Effects RSS.  Returns pip [p], alpha [L,p]."""
    p = len(z)
    if pi0 is None:
        pi0 = np.ones(p) / p
    pi0 = pi0 / pi0.sum()

    alpha = np.tile(pi0, (L, 1))  # [L, p]
    mu    = np.zeros((L, p))

    diag_R = np.clip(np.diag(R), 1e-6, None)

    for _ in range(max_iter):
        alpha_old = alpha.copy()
        for l in range(L):
            # residual z excluding effect l
            z_res = z - sum(R @ (alpha[k] * mu[k]) for k in range(L) if k != l)
            s2   = N * prior_var * diag_R
            lbf  = 0.5 * (np.log(1.0 / (1.0 + s2)) + z_res**2 * s2 / (1.0 + s2))
            log_w = lbf + np.log(pi0 + 1e-300)
            log_w -= log_w.max()
            w = np.exp(log_w)
            alpha[l] = w / w.sum()
            sigma2_l = prior_var * diag_R / (1.0 + s2)
            mu[l]    = sigma2_l * N * z_res
        if np.abs(alpha - alpha_old).max() < tol:
            break

    pip = 1.0 - np.prod(1.0 - alpha, axis=0)
    return pip, alpha


# ── Step 1: EUR sample list ───────────────────────────────────────────────────
def get_eur_samples():
    url = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
           "integrated_call_samples_v3.20130502.ALL.panel")
    print("[step1] Fetching EUR sample list...", flush=True)
    resp = urllib.request.urlopen(url, timeout=30)
    samples = set()
    for line in resp.read().decode().splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 3 and p[2] == "EUR":
            samples.add(p[0])
    print(f"  {len(samples)} EUR samples", flush=True)
    return samples


# ── Step 2: Genotypes → LD ────────────────────────────────────────────────────
def fetch_ld(locus, eur_samples):
    import cyvcf2
    chrom  = locus["vcf_chrom"]
    lo, hi = locus["center"] - WINDOW, locus["center"] + WINDOW
    url    = VCF_TPL.format(c=chrom)

    vcf      = cyvcf2.VCF(url)
    all_samp = list(vcf.samples)
    eur_idx  = np.array([i for i, s in enumerate(all_samp) if s in eur_samples])
    print(f"  {len(eur_idx)} EUR samples in VCF", flush=True)

    pos_list, rsid_list, ref_list, alt_list, geno_list = [], [], [], [], []
    for v in vcf(f"{chrom}:{lo}-{hi}"):
        if v.var_type != "snp" or "," in ",".join(v.ALT):
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

    if not geno_list:
        raise RuntimeError("No SNPs in region")

    G = np.vstack(geno_list).T.astype(float)   # [n_eur, n_snps]
    # mean-impute
    cmeans = np.nanmean(G, axis=0)
    for j in range(G.shape[1]):
        mask = np.isnan(G[:, j])
        if mask.any():
            G[mask, j] = cmeans[j]
    mu  = G.mean(0); sig = G.std(0); sig[sig == 0] = 1
    Gs  = (G - mu) / sig
    LD  = np.clip(Gs.T @ Gs / Gs.shape[0], -1, 1)

    snp_df = pd.DataFrame({"rsid": rsid_list, "pos": pos_list,
                            "ref": ref_list,  "alt": alt_list})
    snp_df["rsid_l"] = snp_df["rsid"].str.lower()
    print(f"  LD panel: {len(snp_df)} SNPs × {len(eur_idx)} EUR", flush=True)
    return snp_df, LD


# ── Step 3: Match GWAS z-scores ───────────────────────────────────────────────
def match_gwas(locus, snp_df):
    gwas = pd.read_csv(LOCI_DIR / f"{locus['name']}_LDL.tsv", sep="\t")
    gwas["rsid_l"] = gwas["rsid"].str.lower()
    gwas_pos = dict(zip(gwas["pos"], gwas["z"]))
    gwas_n   = dict(zip(gwas["pos"], gwas["N"]))
    gwas_rid = dict(zip(gwas["rsid_l"], gwas["z"]))
    gwas_rid_n = dict(zip(gwas["rsid_l"], gwas["N"]))

    z_vals, n_vals, keep_idx = [], [], []
    for i, row in snp_df.iterrows():
        z = gwas_rid.get(row["rsid_l"]) or gwas_pos.get(row["pos"])
        n = gwas_rid_n.get(row["rsid_l"]) or gwas_n.get(row["pos"])
        if z is not None and np.isfinite(z):
            z_vals.append(float(z)); n_vals.append(float(n or 89138))
            keep_idx.append(i)

    sub = snp_df.loc[keep_idx].copy().reset_index(drop=True)
    sub["z"] = z_vals
    sub["N"] = n_vals
    print(f"  {len(sub)} SNPs with GWAS z-scores", flush=True)
    return sub


# ── Step 4: ISM prior vector ──────────────────────────────────────────────────
def build_ism_prior(model_name, sub_df, ism_df, causal_dict):
    """Map ISM sum|delta| to locus SNPs by position; normalise to valid prior."""
    p  = len(sub_df)
    pi = np.ones(p) / p   # base: uniform

    # Build position → ISM score from causal_dict (rsid→pos) + ism_df scores
    ism_by_rsid = {}
    for _, r in ism_df.iterrows():
        rid = r["name"].split("_")[0].lower()
        try:
            ism_by_rsid[rid] = float(r["sum_abs_delta"])
        except Exception:
            pass

    # Map through causal_dict: pos → score
    pos_to_score = {}
    for rsid, pos in causal_dict.items():
        score = ism_by_rsid.get(rsid.lower())
        if score is not None:
            pos_to_score[pos] = score

    for i, row in sub_df.iterrows():
        score = pos_to_score.get(row["pos"])
        if score is not None:
            pi[i] += 10 * score   # boost causal SNPs by ISM score

    return pi / pi.sum()


# ── Step 5: Find causal SNP indices ──────────────────────────────────────────
def find_causal_idx(sub_df, causal_dict):
    """causal_dict: {rsid: hg19_pos}.  Match by position (1000G IDs are None)."""
    pos_to_rsid = {pos: rsid for rsid, pos in causal_dict.items()}
    idx = {}
    for i, row in sub_df.iterrows():
        p = row["pos"]
        if p in pos_to_rsid:
            idx[pos_to_rsid[p]] = i
    return idx


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    eur_samples = get_eur_samples()

    models_ism = {}
    for m in ["v6_b", "v6", "v6_b2", "v7"]:
        p = ISM_DIR / f"ism_{m}" / "ism_scores.tsv"
        if p.exists():
            models_ism[m] = pd.read_csv(p, sep="\t")

    all_rows = []

    for locus in LOCI:
        print(f"\n{'='*60}\nLocus: {locus['name']}", flush=True)

        try:
            snp_df, LD = fetch_ld(locus, eur_samples)
        except Exception as e:
            print(f"  VCF error: {e}", flush=True); continue

        sub = match_gwas(locus, snp_df)
        if len(sub) < 50:
            print(f"  Too few SNPs ({len(sub)})", flush=True); continue

        # LD subset to matched SNPs (position match)
        orig_idx_map = {row["pos"]: i for i, row in snp_df.iterrows()}
        sub_orig_idx = np.array([orig_idx_map[pos] for pos in sub["pos"] if pos in orig_idx_map])
        if len(sub_orig_idx) != len(sub):
            # Keep only positions that exist
            mask = [pos in orig_idx_map for pos in sub["pos"]]
            sub = sub[mask].reset_index(drop=True)
            sub_orig_idx = np.array([orig_idx_map[pos] for pos in sub["pos"]])

        LD_sub = LD[np.ix_(sub_orig_idx, sub_orig_idx)]
        z_vals = sub["z"].values.astype(float)
        N_val  = float(sub["N"].median())

        causal_idx = find_causal_idx(sub, locus["causal"])
        print(f"  Causal SNPs in panel: {list(causal_idx.keys())}", flush=True)
        if not causal_idx:
            # show nearby rsids for debugging
            causal_l = [r.lower() for r in locus["causal"]]
            close = [r for r in sub["rsid"].str.lower() if any(c[:6] in r for c in causal_l)]
            print(f"  (close rsids: {close[:5]})", flush=True)

        # Run SuSiE baseline
        print("  SuSiE baseline...", flush=True)
        pip_base, _ = susie_rss(z_vals, LD_sub, N_val)
        row_base = {"locus": locus["name"], "model": "baseline", "n_snps": len(sub)}
        for rsid, idx in causal_idx.items():
            row_base[rsid] = round(float(pip_base[idx]), 4)
        all_rows.append(row_base)

        # Run SuSiE with ISM priors
        for model_name, ism_df in models_ism.items():
            pi = build_ism_prior(model_name, sub, ism_df, locus["causal"])
            pip, _ = susie_rss(z_vals, LD_sub, N_val, pi0=pi)
            row = {"locus": locus["name"], "model": model_name, "n_snps": len(sub)}
            for rsid, idx in causal_idx.items():
                row[rsid] = round(float(pip[idx]), 4)
            all_rows.append(row)
            if causal_idx:
                causal_pip = [row.get(r, float("nan")) for r in causal_idx]
                print(f"  {model_name}: causal PIPs = {causal_pip}", flush=True)

    results = pd.DataFrame(all_rows)
    print(f"\n{'='*60}")
    print(results.to_string(index=False))
    results.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"\nSaved → {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
