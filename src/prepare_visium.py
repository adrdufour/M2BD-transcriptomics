"""
Build rat_visium_SE.h5ad, the Visium dataset of Block C, from the full experiment.

The source is the instructor-only conversion of the Seurat object (16 sections,
see seurat_to_h5ad.py). The course uses four of them, two pairs mounted on the
same slide so that each control/SE comparison is free of slide batch:

    slide L1   B_L1_S2  control, day 5      A_L1_S1  SE, day 5
    slide L2   D_L2_S8  control, day 40     C_L2_S7  SE, day 40

Spatial domains come from the authors' integrated clustering across all 16
sections (Seurat, SCT + integration). Cortex is kept at sub-cluster resolution
(four bands), the hippocampal sub-clusters are merged. The domains are shown to
students as numbers 1-8 from a seeded shuffle, so this file is not an answer key
to Q6a; the student build drops obs['true_domain'].

    python prepare_visium.py SOURCE.h5ad OUTDIR [--with-truth]
"""

import sys, os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

SECTIONS = {
    "B_L1_S2": ("control", 5, "L1"),
    "A_L1_S1": ("SE", 5, "L1"),
    "D_L2_S8": ("control", 40, "L2"),
    "C_L2_S7": ("SE", 40, "L2"),
}

# authors' cluster / sub-cluster -> anatomical domain. Cortical bands are named
# by their position from the white matter and their markers (see the addendum).
DOMAIN_OF_CLUSTER = {
    "0.0": "cortex L6/6b",
    "0.3": "cortex L5/6 transition",
    "0.2": "cortex L5",
    "0.1": "cortex L2/3",
    "3.0": "hippocampus", "3.1": "hippocampus", "3.2": "hippocampus", "3.3": "hippocampus",
    "1": "white matter / fibre tracts",
    "2": "thalamus",
    "4": "striatum / amygdala",
}
DOMAINS = list(dict.fromkeys(DOMAIN_OF_CLUSTER.values()))
NUMBER_FOR_DOMAIN = dict(zip(DOMAINS, 1 + np.random.default_rng(5150).permutation(len(DOMAINS))))

# genes the notebook and the worksheet name explicitly: always kept
NAMED_GENES = [
    "Cux2", "Lamp5", "Calb1", "Rorb", "Fezf2", "Bcl11b", "Tle4", "Foxp2", "Ccn2", "Nr4a2",
    "Satb2", "Slc17a7", "Snap25", "Hpca", "Camk2a", "Nrgn", "Fibcd1", "Bok", "Prox1",
    "Mbp", "Plp1", "Mobp", "Prkcd", "Tcf7l2", "Ppp1r1b", "Penk", "Gpr88",
    "Gfap", "Aqp4", "Slc1a3", "Aldh1l1", "Vim", "C3", "Aif1", "Cd68", "Cx3cr1", "P2ry12",
    "Spp1", "Cd74", "Lcn2", "Ttr", "Hba-a1", "Hbb", "mt-Co1", "mt-Nd1",
]
N_GENES = 5000


def build(source, include_truth=False):
    A = ad.read_h5ad(source)
    A = A[A.obs["orig.ident"].isin(list(SECTIONS))].copy()

    # genes: the most variable ones across these four sections, plus the named ones
    X = A.X.tocsc()
    det = np.asarray((X > 0).mean(0)).ravel()
    lib = np.asarray(A.X.sum(1)).ravel()
    norm = sparse.diags(1e4 / lib) @ A.X
    norm.data = np.log1p(norm.data)
    mean = np.asarray(norm.mean(0)).ravel()
    var = np.asarray(norm.multiply(norm).mean(0)).ravel() - mean ** 2
    disp = np.where(det > 0.02, var / np.maximum(mean, 1e-3), 0)
    keep = set(A.var_names[np.argsort(-disp)[:N_GENES]]) | {g for g in NAMED_GENES if g in A.var_names}
    genes = [g for g in A.var_names if g in keep]
    # library size over ALL genes, before the gene subset: what normalisation must use
    n_genes = np.asarray((A.X > 0).sum(1)).ravel()
    A = A[:, genes].copy()

    o = A.obs
    cluster = o["seurat_custom_clusters"].astype(str)
    domain = cluster.map(DOMAIN_OF_CLUSTER)
    obs = pd.DataFrame({
        "section": o["orig.ident"].astype(str).values,
        "condition": [SECTIONS[s][0] for s in o["orig.ident"]],
        "days_post_SE": [SECTIONS[s][1] for s in o["orig.ident"]],
        "slide": [SECTIONS[s][2] for s in o["orig.ident"]],
        "domain": pd.Categorical([str(NUMBER_FOR_DOMAIN[d]) for d in domain],
                                 categories=[str(i) for i in range(1, len(DOMAINS) + 1)]),
        "total_counts": lib.astype(int),
        "n_genes": n_genes.astype(int),
        "array_row": o["array_row"].astype(int).values,
        "array_col": o["array_col"].astype(int).values,
    }, index=o.index)
    if include_truth:
        obs["true_domain"] = domain.values
        obs["author_cluster"] = cluster.values
    for c in ("section", "condition", "slide"):
        obs[c] = obs[c].astype("category")

    counts = sparse.csr_matrix(A.X).astype(np.int32)
    V = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    V.obsm["spatial"] = A.obsm["spatial"].astype(np.float32)
    V.uns["spatial"] = {}
    from scipy.spatial import cKDTree
    for s in SECTIONS:
        u = A.uns["spatial"][s]
        img = np.clip(np.asarray(u["images"]["lowres"]) * 255, 0, 255).astype(np.uint8)
        sf = {k: float(v) for k, v in u["scalefactors"].items()}
        # The Seurat object carries the hires scale factor in place of the spot
        # diameter. Recompute it: spots are 55 um wide on a 100 um centre grid.
        xy = V.obsm["spatial"][(V.obs.section == s).values]
        pitch = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
        sf["spot_diameter_fullres"] = float(0.55 * pitch)
        V.uns["spatial"][s] = {"images": {"lowres": img}, "scalefactors": sf}
    V.uns["about"] = (
        "Rat brain, coronal hemisphere sections, 10x Visium (55 um spots, 100 um apart). "
        "Status epilepticus (SE) vs control, 5 and 40 days after SE. Raw UMI counts for "
        f"{len(genes)} genes; obs['total_counts'] is the library size over all 26,658 genes. obs['domain'] numbers the spatial domains found by clustering "
        "spots across 16 sections; which is which is for you to work out.")
    return V


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    truth = "--with-truth" in sys.argv
    V = build(src, include_truth=truth)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "rat_visium_SE.h5ad")
    V.write_h5ad(path, compression="gzip", compression_opts=9)
    print("instructor build" if truth else "student build", V.shape,
          round(os.path.getsize(path) / 1e6, 1), "MB ->", path)
    print(pd.crosstab(V.obs.section, V.obs.domain))
