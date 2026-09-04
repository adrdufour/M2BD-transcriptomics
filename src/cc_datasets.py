"""
Deterministic generator for the three CC datasets.

    bulk_cortex.h5ad      developmental bulk cortex + a case/control contrast
    fetal_cortex_sc.h5ad  mid-gestation cortex nuclei, raw counts, no QC applied
    cortex_visium.h5ad     one Visium section of fetal cortex, unannotated

All three are built from ONE shared cell-type expression profile matrix e(g, k),
so that:
  * the single-cell object is a valid reference for deconvolving the bulk object
  * the bulk signal really is  sum_k p_k * e(g, k)  (the "bulk equation", Q5)
  * cases differ from controls BOTH in composition (fewer neurons, more glia and
    microglia) AND in regulation (excitatory neurons down-regulate the synaptic
    module) -- so the correct answer to Q5a is "both", with evidence.

Seeded; running it twice gives byte-identical objects.

Usage:  python cc_datasets.py OUTDIR
"""

import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

SEED = 20260904

# --------------------------------------------------------------------------
# 1.  Genes
# --------------------------------------------------------------------------

MARKERS = {
    # progenitors
    "RG":         ["VIM", "SOX2", "PAX6", "HES1", "HES5", "SLC1A3", "NES", "GLI3", "TTYH1"],
    "oRG":        ["HOPX", "PTPRZ1", "FAM107A", "TNC", "MOXD1", "LIFR", "SLC1A3", "VIM"],
    "IPC":        ["EOMES", "PPP1R17", "NEUROG1", "NEUROG2", "PENK", "ELAVL4"],
    # neurons
    "ExN_deep":   ["BCL11B", "FEZF2", "TBR1", "CRYM", "SOX5", "NEUROD2", "NEUROD6", "SYN1", "SNAP25"],
    "ExN_upper":  ["SATB2", "CUX2", "RORB", "POU3F2", "NEUROD2", "NEUROD6", "SYN1", "SNAP25"],
    "IN_MGE":     ["DLX1", "DLX2", "GAD1", "GAD2", "LHX6", "SST", "SNAP25"],
    "IN_CGE":     ["DLX1", "DLX2", "GAD1", "GAD2", "SP8", "NR2F2", "SNAP25"],
    "CR":         ["RELN", "TP73", "LHX1", "TBR1"],
    # glia
    "Astro":      ["AQP4", "ALDH1L1", "GFAP", "S100B", "SLC1A3", "GJA1", "SPARCL1"],
    "OPC":        ["OLIG1", "OLIG2", "PDGFRA", "CSPG4", "SOX10"],
    "Micro":      ["AIF1", "C1QA", "C1QB", "C1QC", "P2RY12", "CX3CR1", "TYROBP", "TREM2", "CSF1R"],
    "Endo":       ["CLDN5", "PECAM1", "FLT1", "RGS5"],
}

CELL_TYPES = list(MARKERS)

# the two modules the volcano plot is built around
SYNAPTIC = [
    "SYN1", "SYN2", "SNAP25", "SYT1", "SYP", "DLG4", "GRIN1", "GRIN2B", "GRIA1",
    "GRIA2", "NRXN1", "NRXN3", "NLGN1", "NLGN3", "STXBP1", "RAB3A", "CAMK2A",
    "CAMK2B", "GABRB3", "GABRA1", "SCN2A", "CACNA1E", "KCNQ2", "SHANK2",
    "SHANK3", "DLGAP1", "HOMER1", "PCLO", "BSN", "CPLX1", "CPLX2", "VAMP2",
    "STX1A", "SV2A", "NEFL", "NEFM", "MAP2", "TUBB3", "RBFOX3", "ATP2B2",
]
IMMUNE_GLIAL = [
    "C1QA", "C1QB", "C1QC", "AIF1", "TYROBP", "TREM2", "CSF1R", "CD68",
    "P2RY12", "CX3CR1", "ITGAM", "PTPRC", "IFITM2", "IFITM3", "B2M", "HLA-A",
    "HLA-B", "HLA-DRA", "CD74", "GFAP", "AQP4", "S100B", "VIM", "CHI3L1",
    "SERPINA3", "CRYAB", "MT2A", "MT1X", "SOX9", "ID3", "NFKBIA", "TNFRSF1A",
    "IL33", "CCL3", "CCL4", "SPP1", "APOE", "CLU", "A2M", "TIMP1",
]
MT_GENES = ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP6", "MT-CO3",
            "MT-ND3", "MT-ND4", "MT-ND5", "MT-CYB", "MT-ND4L", "MT-ATP8", "MT-ND6"]

N_FILLER = 2600
FILLER = [f"AC{i:06d}.1" if i % 3 == 0 else f"GENE{i:05d}" for i in range(N_FILLER)]

# A co-expression module is hundreds of genes, not the forty that have famous
# names. The named genes above are the ones the volcano plot labels; these are
# the full modules that actually move together.
_perm = np.random.default_rng(777).permutation(N_FILLER)
SYNAPTIC_MODULE = SYNAPTIC + [FILLER[i] for i in _perm[:230]]
IMMUNE_MODULE = IMMUNE_GLIAL + [FILLER[i] for i in _perm[230:460]]


def gene_universe():
    named = []
    for v in MARKERS.values():
        named += v
    named += SYNAPTIC + IMMUNE_GLIAL + MT_GENES
    named = sorted(set(named))
    return named + FILLER


# --------------------------------------------------------------------------
# 2.  Shared cell-type expression profiles  e(g, k)
# --------------------------------------------------------------------------

def build_profiles(genes, rng):
    """Mean expression per gene per cell type, on a linear (CPM-like) scale."""
    G, K = len(genes), len(CELL_TYPES)
    idx = {g: i for i, g in enumerate(genes)}

    # housekeeping baseline shared by all cell types: log-normal
    base = np.exp(rng.normal(1.2, 1.1, size=G))
    E = np.tile(base[:, None], (1, K))

    # cell-type-specific noise so that types are not identical
    # Most genes are somewhat cell-type-preferential, not only the markers.
    # This is what makes a shift in composition move hundreds of genes in the
    # bulk contrast -- the whole point of Q5.
    E *= np.exp(rng.normal(0, 0.80, size=(G, K)))

    for k, ct in enumerate(CELL_TYPES):
        for g in MARKERS[ct]:
            E[idx[g], k] *= 45.0

    # module-level structure
    neuronal = [CELL_TYPES.index(c) for c in ["ExN_deep", "ExN_upper", "IN_MGE", "IN_CGE", "CR"]]
    glial = [CELL_TYPES.index(c) for c in ["Astro", "Micro", "OPC"]]
    for g in SYNAPTIC_MODULE:
        E[idx[g], :] *= 0.35
        E[idx[g], neuronal] *= 22.0
    for g in IMMUNE_MODULE:
        E[idx[g], :] *= 0.5
        E[idx[g], glial] *= 14.0
    # microglia carry the immune half of the module hardest
    mic = CELL_TYPES.index("Micro")
    for g in IMMUNE_GLIAL[:20]:
        E[idx[g], mic] *= 4.0

    # Mitochondrial transcripts: set the MT fraction per cell type explicitly,
    # genuinely higher in mature neurons because they are metabolically
    # demanding. This is what makes a blanket 5% MT cut-off delete neurons (Q3b).
    mt_i = np.array([idx[g] for g in MT_GENES])
    target = np.full(K, 0.030)
    for c in ["ExN_deep", "ExN_upper", "IN_MGE", "IN_CGE", "CR"]:
        target[CELL_TYPES.index(c)] = 0.075
    target[CELL_TYPES.index("Micro")] = 0.018
    target[CELL_TYPES.index("Endo")] = 0.022
    for k in range(K):
        other = E[:, k].sum() - E[mt_i, k].sum()
        E[mt_i, k] *= (target[k] * other) / ((1 - target[k]) * E[mt_i, k].sum())

    return E, idx


# --------------------------------------------------------------------------
# 3.  Composition as a function of developmental age
# --------------------------------------------------------------------------

def proportions(age_pcw, rng, case=False):
    """Cell-type proportions at a given post-conceptional week."""
    t = np.clip((age_pcw - 8) / 32.0, 0, 1)          # 0 at 8 pcw, 1 at 40 pcw
    p = {
        "RG":        0.26 * (1 - t) ** 1.4 + 0.02,
        "oRG":       0.13 * np.exp(-((t - 0.35) ** 2) / 0.06) + 0.01,
        "IPC":       0.14 * (1 - t) ** 1.1 + 0.01,
        "ExN_deep":  0.06 + 0.16 * t,
        "ExN_upper": 0.02 + 0.28 * t ** 1.3,
        "IN_MGE":    0.03 + 0.05 * t,
        "IN_CGE":    0.02 + 0.04 * t,
        "CR":        0.02 * (1 - t) + 0.002,
        "Astro":     0.01 + 0.10 * t ** 1.8,
        "OPC":       0.005 + 0.05 * t ** 2.0,
        "Micro":     0.02 + 0.04 * t,
        "Endo":      0.02 + 0.01 * t,
    }
    v = np.array([p[c] for c in CELL_TYPES], float)

    if case:
        # composition difference in cases: fewer excitatory neurons,
        # more microglia and astrocytes.  This is HALF of the Q2 signal.
        v[CELL_TYPES.index("ExN_upper")] *= 0.74
        v[CELL_TYPES.index("ExN_deep")] *= 0.86
        v[CELL_TYPES.index("Micro")] *= 1.70
        v[CELL_TYPES.index("Astro")] *= 1.35

    v *= np.exp(rng.normal(0, 0.12, size=v.size))     # donor-to-donor variability
    return v / v.sum()


# --------------------------------------------------------------------------
# 4.  Bulk
# --------------------------------------------------------------------------

def make_bulk(genes, E, idx, rng):
    rows = []
    # (a) developmental series, controls only, 8 -> 40 pcw
    ages = np.round(np.linspace(8, 39, 24) + rng.normal(0, 0.4, 24), 1)
    for i, a in enumerate(ages):
        rows.append(dict(sample_id=f"DEV{i+1:02d}", cohort="development",
                         diagnosis="control", age_pcw=float(a)))
    # (b) case/control contrast, age-matched, late fetal (20 vs 20)
    for i in range(40):
        a = float(np.round(rng.uniform(30, 38), 1))
        rows.append(dict(sample_id=f"CC{i+1:02d}", cohort="contrast",
                         diagnosis="case" if i % 2 == 0 else "control", age_pcw=a))

    obs = pd.DataFrame(rows).set_index("sample_id")
    n = obs.shape[0]
    obs["region"] = rng.choice(["PFC", "V1", "TC"], n, p=[0.5, 0.25, 0.25])
    obs["sex"] = rng.choice(["M", "F"], n)
    obs["RIN"] = np.round(rng.uniform(5.8, 9.2, n), 1)
    obs["PMI_h"] = np.round(rng.uniform(4, 28, n), 1)
    # batch is balanced across diagnosis on purpose: the honest teaching case
    obs["batch"] = np.where(np.arange(n) % 3 == 0, "B1", np.where(np.arange(n) % 3 == 1, "B2", "B3"))

    G = len(genes)
    P = np.zeros((n, len(CELL_TYPES)))
    mu = np.zeros((n, G))

    # regulatory effect: in cases, excitatory neurons themselves down-regulate
    # the synaptic module.  This is the OTHER half of the Q2 signal.
    reg = np.ones((G, len(CELL_TYPES)))
    exn = [CELL_TYPES.index(c) for c in ["ExN_deep", "ExN_upper"]]
    for g in SYNAPTIC_MODULE:
        reg[idx[g], exn] = 0.78

    for i, (sid, r) in enumerate(obs.iterrows()):
        is_case = r["diagnosis"] == "case"
        p = proportions(r["age_pcw"], rng, case=is_case)
        P[i] = p
        Ei = E * reg if is_case else E
        mu[i] = Ei @ p

    # Region and batch effects. Batch is deliberately the second-largest source
    # of variance -- it is what PC2 mostly is (Q1b) -- but it is balanced across
    # diagnosis, so it is a nuisance and not a confounder.
    reg_eff = {r: np.exp(rng.normal(0, 0.05, G)) for r in obs["region"].unique()}
    bat_eff = {b: np.exp(rng.normal(0, 0.148, G)) for b in obs["batch"].unique()}
    for i, (sid, r) in enumerate(obs.iterrows()):
        mu[i] *= reg_eff[r["region"]] * bat_eff[r["batch"]]

    # library size + negative binomial
    lib = rng.uniform(2.2e7, 4.5e7, n)
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    disp = 1.0 / np.clip(rng.gamma(6.0, 1 / 200, G), 5e-4, None)
    counts = rng.negative_binomial(
        np.clip(disp[None, :], 1, 1e4),
        np.clip(disp[None, :] / (disp[None, :] + mu), 1e-9, 1 - 1e-9),
    ).astype(np.float32)

    A = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs,
                   var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    A.obsm["true_proportions"] = pd.DataFrame(P, index=obs.index, columns=CELL_TYPES)
    A.uns["about"] = ("Simulated developmental bulk cortex for the M2BD CC. "
                      "obsm['true_proportions'] is the ground truth used to build the data; "
                      "do not show it to students before Figure 5.")
    return A


# --------------------------------------------------------------------------
# 5.  Single cell
# --------------------------------------------------------------------------

SC_MIX = {"RG": 0.17, "oRG": 0.11, "IPC": 0.12, "ExN_deep": 0.15, "ExN_upper": 0.13,
          "IN_MGE": 0.07, "IN_CGE": 0.05, "CR": 0.02, "Astro": 0.06, "OPC": 0.03,
          "Micro": 0.06, "Endo": 0.03}


def make_sc(genes, E, idx, rng, n_cells=8000):
    G = len(genes)
    p = np.array([SC_MIX[c] for c in CELL_TYPES])
    p = p / p.sum()
    lab = rng.choice(len(CELL_TYPES), n_cells, p=p)

    # library sizes: a healthy log-normal core ...
    depth = np.exp(rng.normal(np.log(4200), 0.32, n_cells))
    kind = np.array(["cell"] * n_cells, dtype=object)

    # ... plus the three things QC is for (Q3)
    n_amb = int(0.06 * n_cells)          # ambient / empty droplets
    n_dbl = int(0.045 * n_cells)         # doublets
    n_dam = int(0.05 * n_cells)          # damaged, high MT
    order = rng.permutation(n_cells)
    i_amb, i_dbl, i_dam = order[:n_amb], order[n_amb:n_amb + n_dbl], order[n_amb + n_dbl:n_amb + n_dbl + n_dam]
    depth[i_amb] = np.exp(rng.normal(np.log(320), 0.35, n_amb))
    depth[i_dbl] = np.exp(rng.normal(np.log(14500), 0.22, n_dbl))
    kind[i_amb], kind[i_dbl], kind[i_dam] = "ambient", "doublet", "damaged"

    prof = E / E.sum(0, keepdims=True)          # G x K, columns sum to 1
    ambient_prof = prof @ p                     # soup

    P = prof[:, lab].T.copy()                   # cells x genes
    # doublets are a mixture of two types
    second = rng.choice(len(CELL_TYPES), i_dbl.size, p=p)
    P[i_dbl] = 0.55 * P[i_dbl] + 0.45 * prof[:, second].T
    # every droplet carries some soup
    P = 0.90 * P + 0.10 * ambient_prof[None, :]
    P[i_amb] = 0.15 * P[i_amb] + 0.85 * ambient_prof[None, :]

    # damaged cells: cytoplasmic RNA lost, mitochondrial fraction inflated
    mt = np.array([g in set(MT_GENES) for g in genes])
    P[np.ix_(i_dam, np.where(mt)[0])] *= 12.0
    P /= P.sum(1, keepdims=True)

    X = rng.poisson(P * depth[:, None]).astype(np.float32)
    # 3'-capture dropout on top of the Poisson sampling
    X *= (rng.random(X.shape) > 0.22)

    obs = pd.DataFrame(
        {"true_cell_type": [CELL_TYPES[i] for i in lab],
         "droplet_kind": kind,
         "donor": rng.choice(["D1", "D2", "D3"], n_cells),
         "chemistry": "3'v3"},
        index=pd.Index([f"CELL_{i:05d}" for i in range(n_cells)], name="barcode"),
    )
    var = pd.DataFrame({"mt": mt}, index=pd.Index(genes, name="gene"))
    A = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
    A.uns["about"] = ("Simulated mid-gestation human cortex nuclei for the M2BD CC. "
                      "RAW counts, no QC applied. obs['true_cell_type'] and "
                      "obs['droplet_kind'] are ground truth -- instructor use only.")
    return A


# --------------------------------------------------------------------------
# 6.  Visium
# --------------------------------------------------------------------------

# ventricular surface -> pial surface
COMPARTMENTS = ["VZ", "SVZ", "IZ", "SP", "CP", "MZ"]
COMP_MIX = {
    "VZ":  {"RG": 0.62, "IPC": 0.12, "oRG": 0.08, "Endo": 0.05, "Micro": 0.04, "Astro": 0.03, "ExN_deep": 0.03, "OPC": 0.03},
    "SVZ": {"oRG": 0.34, "IPC": 0.30, "RG": 0.14, "Micro": 0.06, "Astro": 0.05, "OPC": 0.05, "ExN_deep": 0.04, "Endo": 0.02},
    "IZ":  {"ExN_deep": 0.30, "ExN_upper": 0.22, "IN_MGE": 0.12, "OPC": 0.11, "IPC": 0.09, "Astro": 0.07, "Micro": 0.05, "Endo": 0.04},
    "SP":  {"ExN_deep": 0.44, "IN_CGE": 0.14, "IN_MGE": 0.12, "Astro": 0.10, "ExN_upper": 0.08, "Micro": 0.06, "Endo": 0.06},
    "CP":  {"ExN_upper": 0.45, "ExN_deep": 0.26, "IN_MGE": 0.08, "IN_CGE": 0.07, "Astro": 0.06, "Micro": 0.04, "Endo": 0.04},
    "MZ":  {"CR": 0.40, "ExN_upper": 0.20, "Astro": 0.14, "IN_CGE": 0.10, "Endo": 0.08, "Micro": 0.08},
}
# thickness of each compartment, ventricle -> pia
COMP_WIDTH = np.array([0.13, 0.19, 0.20, 0.10, 0.30, 0.08])
# The numbers the students see on Figure 6. Derived from a seeded shuffle rather
# than written out, so that this file is not an answer key to Q6a.
GROUP_FOR_COMPARTMENT = dict(zip(
    COMPARTMENTS, 1 + np.random.default_rng(4919).permutation(len(COMPARTMENTS))))


def make_visium(genes, E, idx, rng, n_rows=44, n_cols=34):
    G = len(genes)
    xs, ys, comp, frac = [], [], [], []
    edges = np.concatenate([[0], np.cumsum(COMP_WIDTH)])
    edges = edges / edges[-1]

    for r in range(n_rows):
        for c in range(n_cols):
            x = c * 100.0 + (50.0 if r % 2 else 0.0)
            y = r * 86.6
            u = r / (n_rows - 1)                      # 0 = ventricle, 1 = pia
            u_j = np.clip(u + rng.normal(0, 0.012), 0, 0.999)
            k = int(np.searchsorted(edges, u_j, side="right") - 1)
            xs.append(x); ys.append(y); comp.append(COMPARTMENTS[k]); frac.append(u)

    n = len(xs)
    P = np.zeros((n, len(CELL_TYPES)))
    for i, cp in enumerate(comp):
        v = np.array([COMP_MIX[cp].get(ct, 0.0) for ct in CELL_TYPES])
        v = v + 0.01
        v *= np.exp(rng.normal(0, 0.22, v.size))
        P[i] = v / v.sum()

    mu = (E @ P.T).T
    depth = np.exp(rng.normal(np.log(6500), 0.30, n))
    mu = mu / mu.sum(1, keepdims=True) * depth[:, None]
    X = rng.poisson(mu).astype(np.float32)

    obs = pd.DataFrame(
        {"true_compartment": comp,
         "spot_group": [GROUP_FOR_COMPARTMENT[c] for c in comp],
         "radial_fraction": np.round(frac, 3),
         "n_cells_estimate": rng.integers(2, 11, n)},
        index=pd.Index([f"SPOT_{i:04d}" for i in range(n)], name="spot"),
    )
    A = ad.AnnData(X=sparse.csr_matrix(X), obs=obs,
                   var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    A.obsm["spatial"] = np.column_stack([np.array(xs), np.array(ys)]).astype(np.float32)
    A.obsm["true_proportions"] = pd.DataFrame(P, index=obs.index, columns=CELL_TYPES)
    A.uns["spot_diameter_um"] = 55
    A.uns["about"] = ("Simulated Visium section of fetal cortex for the M2BD CC. "
                      "obs['spot_group'] is what students see on Figure 6; "
                      "obs['true_compartment'] is the answer -- instructor use only.")
    return A


# --------------------------------------------------------------------------

# Columns that answer a graded question outright. The student build drops them:
# obs['true_cell_type'] is Q4, obs['true_compartment'] is Q6a, and
# obsm['true_proportions'] is Q5. The instructor build keeps them, and is what
# the expected-output report is computed from.
TRUTH = {
    "bulk": {"obs": [], "obsm": ["true_proportions"]},
    "sc": {"obs": ["true_cell_type", "droplet_kind"], "obsm": []},
    "visium": {"obs": ["true_compartment", "radial_fraction"], "obsm": ["true_proportions"]},
}


def strip_truth(adata, which):
    a = adata.copy()
    for c in TRUTH[which]["obs"]:
        if c in a.obs:
            del a.obs[c]
    for k in TRUTH[which]["obsm"]:
        if k in a.obsm:
            del a.obsm[k]
    a.uns["about"] = a.uns.get("about", "").split("obs[")[0].strip()
    return a


def build_all(outdir=None, include_truth=False):
    """Build the three datasets.

    include_truth=False is the student build: the columns that answer Q4, Q5
    and Q6 in one line are removed. This is the default, because this function
    is what the notebook calls when it cannot reach the repository.
    """
    rng = np.random.default_rng(SEED)
    genes = gene_universe()
    E, idx = build_profiles(genes, rng)
    out = {"bulk": make_bulk(genes, E, idx, rng),
           "sc": make_sc(genes, E, idx, rng),
           "visium": make_visium(genes, E, idx, rng)}
    if not include_truth:
        out = {k: strip_truth(v, k) for k, v in out.items()}
    if outdir:
        import os
        names = {"bulk": "bulk_cortex.h5ad", "sc": "fetal_cortex_sc.h5ad",
                 "visium": "cortex_visium.h5ad"}
        for k, a in out.items():
            a.write_h5ad(os.path.join(outdir, names[k]), compression="gzip")
    return out["bulk"], out["sc"], out["visium"]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    truth = "--with-truth" in sys.argv
    b, s, v = build_all(out, include_truth=truth)
    print("instructor build" if truth else "student build")
    print("bulk   ", b.shape, list(b.obsm))
    print("sc     ", s.shape, list(s.obs.columns))
    print("visium ", v.shape, list(v.obs.columns))
