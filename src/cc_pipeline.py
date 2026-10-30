"""
The analysis behind Figures 1-6 of the M2BD CC.

Kept in one file so that (a) the notebook can import the plotting helpers,
(b) the instructor can regenerate the fallback figure handout, and
(c) the expected answers can be checked without opening Colab.

    python cc_pipeline.py DATADIR OUTDIR
"""

import os, sys, json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import sparse, stats

sc.settings.verbosity = 0
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})

INK = "#1a1a2e"
ACCENT = "#c8553d"
BLUE = "#2b6cb0"
GREY = "#b8b8c4"


def _dense(A):
    return np.asarray(A.X.todense()) if sparse.issparse(A.X) else np.asarray(A.X)


# ===========================================================================
# BLOCK A -- bulk
# ===========================================================================

def block_a(bulk):
    """Returns (adata_with_pca, de_table, pca_variance_ratio)."""
    b = bulk.copy()
    sc.pp.filter_genes(b, min_counts=10)
    b.layers["counts"] = b.X.copy()
    # variance-stabilising-ish transform: CPM -> log1p
    sc.pp.normalize_total(b, target_sum=1e6)
    sc.pp.log1p(b)
    b.raw = b
    sc.pp.highly_variable_genes(b, n_top_genes=2000)
    bh = b[:, b.var.highly_variable].copy()
    sc.pp.scale(bh, max_value=10)
    sc.tl.pca(bh, n_comps=10, svd_solver="arpack", random_state=0)
    b.obsm["X_pca"] = bh.obsm["X_pca"]
    b.uns["pca"] = bh.uns["pca"]

    # ---- differential expression on the contrast cohort only -------------
    cc = b[b.obs.cohort == "contrast"].copy()
    X = _dense(cc)                       # log CPM
    case = (cc.obs.diagnosis == "case").values
    lfc = X[case].mean(0) - X[~case].mean(0)
    lfc = lfc / np.log(2)                # natural log -> log2
    t, p = stats.ttest_ind(X[case], X[~case], axis=0, equal_var=False)
    p = np.nan_to_num(p, nan=1.0)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, p.size + 1)
    padj = np.minimum.accumulate((p * p.size / ranks)[order][::-1])[::-1]
    fdr = np.empty_like(padj)
    fdr[order] = np.clip(padj, 0, 1)

    de = pd.DataFrame({"gene": cc.var_names, "log2FC": lfc, "pval": p, "padj": fdr})
    de = de.set_index("gene")
    return b, de, np.asarray(b.uns["pca"]["variance_ratio"])


def figure1(b, ax=None):
    pcs = b.obsm["X_pca"]
    vr = np.asarray(b.uns["pca"]["variance_ratio"]) * 100
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
    sm = ax.scatter(pcs[:, 0], pcs[:, 1], c=b.obs.age_pcw, cmap="viridis",
                    s=62, edgecolor="white", linewidth=0.8, zorder=3)
    mark = {"control": "o", "case": "^"}
    for d, m in mark.items():
        k = (b.obs.diagnosis == d).values
        ax.scatter(pcs[k, 0], pcs[k, 1], facecolor="none", edgecolor=INK,
                   marker=m, s=95, linewidth=0.7, zorder=4)
    ax.set_xlabel(f"PC1  ({vr[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2  ({vr[1]:.0f}% of variance)")
    ax.set_title("Figure 1 — PCA of the bulk developmental cortex samples", loc="left")
    cb = plt.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("age (post-conceptional weeks)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="control"),
                       Line2D([], [], marker="^", ls="", mfc="none", mec=INK, label="case")],
              frameon=False, loc="best", fontsize=8)
    return ax


def figure2(de, synaptic, immune, ax=None, fdr=0.05, lfc_cut=0.58):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.6))
    x = de["log2FC"].values
    y = -np.log10(np.clip(de["pval"].values, 1e-300, None))
    sig = (de["padj"].values < fdr) & (np.abs(x) > lfc_cut)
    ax.scatter(x[~sig], y[~sig], s=6, c=GREY, alpha=0.55, linewidth=0)
    isyn = de.index.isin(synaptic)
    iimm = de.index.isin(immune)
    ax.scatter(x[sig & ~isyn & ~iimm], y[sig & ~isyn & ~iimm], s=9, c=INK, alpha=0.7, linewidth=0)
    ax.scatter(x[sig & isyn], y[sig & isyn], s=22, c=BLUE, linewidth=0, label="synaptic / neuronal")
    ax.scatter(x[sig & iimm], y[sig & iimm], s=22, c=ACCENT, linewidth=0, label="glial / immune")
    for v in (-lfc_cut, lfc_cut):
        ax.axvline(v, color=INK, lw=0.6, ls="--", alpha=0.5)
    thr = y[de["padj"].values < fdr]
    if thr.size:
        ax.axhline(thr.min(), color=INK, lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("log₂ fold change  (case vs control)")
    ax.set_ylabel("−log₁₀ p")
    ax.set_title("Figure 2 — Volcano plot, cases vs controls", loc="left")
    ax.text(0, 1.015, f"{sig.sum()} genes pass FDR {int(fdr*100)}% and |log₂FC| > {lfc_cut}  "
                      f"·  {int((x[sig] < 0).sum())} down, {int((x[sig] > 0).sum())} up in cases",
            transform=ax.transAxes, fontsize=8, color="#555")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    for g in ["SYN1", "SNAP25", "C1QB", "AIF1", "GFAP"]:
        if g in de.index and sig[de.index.get_loc(g)]:
            ax.annotate(g, (de.loc[g, "log2FC"], -np.log10(max(de.loc[g, "pval"], 1e-300))),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    return ax, int(sig.sum())


# ===========================================================================
# BLOCK B -- single cell
# ===========================================================================

def qc_metrics(s):
    s.var["mt"] = [g.startswith("MT-") for g in s.var_names]
    sc.pp.calculate_qc_metrics(s, qc_vars=["mt"], inplace=True, log1p=True, percent_top=None)
    return s


def mad_outlier(v, n=5):
    v = np.asarray(v, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        mad = 1e-9
    return np.abs(v - med) > n * 1.4826 * mad


def block_b_qc(s, n_mad=3, mt_mad=3):
    s = qc_metrics(s)
    o = s.obs
    o["out_counts"] = mad_outlier(np.log1p(o.total_counts), n_mad)
    o["out_genes"] = mad_outlier(np.log1p(o.n_genes_by_counts), n_mad)
    o["out_mt"] = mad_outlier(o.pct_counts_mt, mt_mad) & (o.pct_counts_mt > np.median(o.pct_counts_mt))
    o["qc_fail"] = o.out_counts | o.out_genes | o.out_mt
    return s


def figure3(s, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    o = s.obs
    panels = [("total_counts", "UMI counts per cell", True),
              ("n_genes_by_counts", "genes detected per cell", True),
              ("pct_counts_mt", "% mitochondrial counts", False)]
    rng = np.random.default_rng(0)
    for ax, (col, lab, logy) in zip(axes, panels):
        v = o[col].values.astype(float)
        keep, drop = ~o.qc_fail.values, o.qc_fail.values
        vv = np.log10(v + 1) if logy else v
        for pos, mask in [(0.0, keep), (1.0, drop)]:
            if mask.sum() < 5:
                continue
            parts = ax.violinplot([v[mask]], positions=[pos], showextrema=False, widths=0.75)
            for pc in parts["bodies"]:
                pc.set_facecolor("#e9e9ef"); pc.set_edgecolor(INK)
                pc.set_linewidth(0.7); pc.set_alpha(1); pc.set_zorder(1)
        jx = rng.normal(0, 0.055, v.size)
        ax.scatter(jx[keep], v[keep], s=2.2, c=BLUE, alpha=0.30, linewidth=0, zorder=3)
        ax.scatter(jx[drop] + 1.0, v[drop], s=3.2, c=ACCENT, alpha=0.65, linewidth=0, zorder=3)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"kept\n{int(keep.sum())}", f"flagged\n{int(drop.sum())}"], fontsize=8)
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel(lab)
    axes[2].axhline(5, color=INK, lw=0.8, ls=":")
    axes[2].text(1.42, 5, " fixed 5% cut-off", fontsize=7, va="bottom", ha="right", color=INK)
    axes[0].set_title("Figure 3 — single-cell QC metrics, MAD-based outlier flags", loc="left")
    return axes


def block_b_cluster(s, resolution=0.6):
    s = s[~s.obs.qc_fail].copy()
    s.layers["counts"] = s.X.copy()
    sc.pp.normalize_total(s, target_sum=1e4)
    sc.pp.log1p(s)
    s.raw = s
    sc.pp.highly_variable_genes(s, n_top_genes=1500)
    sh = s[:, s.var.highly_variable].copy()
    sc.pp.scale(sh, max_value=10)
    sc.tl.pca(sh, n_comps=30, svd_solver="arpack", random_state=0)
    s.obsm["X_pca"] = sh.obsm["X_pca"]
    sc.pp.neighbors(s, n_neighbors=15, n_pcs=30, random_state=0)
    sc.tl.leiden(s, resolution=resolution, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False, random_state=0)
    sc.tl.umap(s, random_state=0)
    # letters assigned by cluster size, largest first: deterministic and
    # explainable, and it puts the well-populated clusters at A, B, C.
    sizes = s.obs.leiden.value_counts()
    letters = {c: chr(ord("A") + i) for i, c in enumerate(sizes.index)}
    cats = [letters[c] for c in sizes.index]
    s.obs["cluster"] = pd.Categorical([letters[c] for c in s.obs.leiden], categories=cats)
    return s


MARKER_PANEL = {
    "radial glia":            ["VIM", "SOX2", "PAX6", "HES1"],
    "outer radial glia":      ["HOPX", "PTPRZ1", "FAM107A"],
    "intermediate prog.":     ["EOMES", "PPP1R17"],
    "excitatory, deep":       ["BCL11B", "FEZF2", "TBR1"],
    "excitatory, upper":      ["SATB2", "CUX2", "NEUROD6"],
    "interneuron":            ["DLX2", "GAD1", "LHX6", "SP8"],
    "Cajal–Retzius":          ["RELN"],
    "astrocyte":              ["AQP4", "ALDH1L1", "GFAP"],
    "OPC":                    ["OLIG2", "PDGFRA"],
    "microglia":              ["AIF1", "C1QB", "P2RY12"],
    "endothelial":            ["CLDN5", "PECAM1"],
}


def figure4(s, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.15], wspace=0.18)
    ax1 = fig.add_subplot(gs[0])
    u = s.obsm["X_umap"]
    cats = list(s.obs.cluster.cat.categories)
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(cats):
        k = (s.obs.cluster == c).values
        ax1.scatter(u[k, 0], u[k, 1], s=3, color=cmap(i % 20), linewidth=0, alpha=0.85)
        ax1.annotate(c, (np.median(u[k, 0]), np.median(u[k, 1])), fontsize=10, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=INK, lw=0.8, alpha=0.92))
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel("UMAP1"); ax1.set_ylabel("UMAP2")
    ax1.set_title("Figure 4 — UMAP, clusters unlabelled", loc="left")

    ax2 = fig.add_subplot(gs[1])
    genes, groups = [], []
    for grp, gl in MARKER_PANEL.items():
        gl = [g for g in gl if g in s.raw.var_names]
        genes += gl; groups += [grp] * len(gl)
    Xr = s.raw[:, genes].X
    Xr = np.asarray(Xr.todense()) if sparse.issparse(Xr) else np.asarray(Xr)
    mean = np.zeros((len(cats), len(genes)))
    frac = np.zeros_like(mean)
    for i, c in enumerate(cats):
        k = (s.obs.cluster == c).values
        mean[i] = Xr[k].mean(0)
        frac[i] = (Xr[k] > 0).mean(0)
    z = (mean - mean.mean(0)) / (mean.std(0) + 1e-9)
    yy, xx = np.meshgrid(np.arange(len(cats)), np.arange(len(genes)), indexing="ij")
    sm = ax2.scatter(xx.ravel(), yy.ravel(), s=frac.ravel() * 95 + 1,
                     c=np.clip(z.ravel(), -1.5, 2.8), cmap="RdBu_r", linewidth=0)
    ax2.set_xticks(range(len(genes)))
    ax2.set_xticklabels(genes, rotation=90, fontsize=7)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels(cats, fontsize=9, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_ylim(len(cats) - 0.4, -2.3)
    prev, start, row = groups[0], 0, 0
    for i, g in enumerate(groups + [None]):
        if g != prev:
            ax2.axvline(i - 0.5, color=INK, lw=0.5, alpha=0.3)
            y = -0.95 if row % 2 == 0 else -1.65
            ax2.plot([start - 0.35, i - 0.65], [y - 0.18, y - 0.18], color=INK, lw=1.6,
                     solid_capstyle="butt", alpha=0.7)
            ax2.text((start + i - 1) / 2, y, prev, ha="center", va="bottom",
                     fontsize=7, color=INK)
            prev, start, row = g, i, row + 1
    ax2.set_title("marker dot plot   ·   size = fraction of cells expressing, colour = scaled mean",
                  loc="left", fontsize=9, pad=16)
    ax2.grid(axis="y", lw=0.3, alpha=0.3)
    plt.colorbar(sm, ax=ax2, pad=0.01, shrink=0.6)
    return fig


# ===========================================================================
# BLOCK C -- deconvolution + spatial
# ===========================================================================

def signature_matrix(s, key="cluster", n_top=35, min_cells=40):
    """Cell-type signature, on the LINEAR CPM scale.

    Deconvolution solves  y = S p  and that equation only holds before the log,
    so the signature is built from counts, not from s.raw (which is logged).
    """
    C = s.layers["counts"]
    C = np.asarray(C.todense()) if sparse.issparse(C) else np.asarray(C)
    C = C / (C.sum(1, keepdims=True) + 1e-9) * 1e4          # CPM-like per cell
    cats = [c for c in s.obs[key].cat.categories if (s.obs[key] == c).sum() >= min_cells]
    M = np.vstack([C[(s.obs[key] == c).values].mean(0) for c in cats])
    sig = pd.DataFrame(M.T, index=s.var_names, columns=cats)
    sig = sig[sig.max(1) > 0.5]
    # keep the genes that discriminate between the clusters
    z = np.log1p(sig)
    z = (z.sub(z.mean(1), axis=0)).div(z.std(1) + 1e-9, axis=0)
    keep = set()
    for c in cats:
        keep |= set(z[c].sort_values(ascending=False).head(n_top).index)
    return sig.loc[sorted(keep)]


def deconvolve(mix_cpm, sig):
    """Scaled non-negative least squares; proportions constrained to sum to 1.

    Each signature gene is divided by its mean across cell types so that a
    handful of very highly expressed genes cannot dominate the fit.
    """
    from scipy.optimize import nnls
    genes = [g for g in sig.index if g in mix_cpm.columns]
    S = sig.loc[genes].values.astype(float)
    scale = S.mean(1) + 1e-9
    Ss = S / scale[:, None]
    Y = mix_cpm[genes].values.astype(float) / scale[None, :]
    out = []
    for i in range(Y.shape[0]):
        w, _ = nnls(Ss, Y[i])
        out.append(w / (w.sum() + 1e-12))
    return pd.DataFrame(out, index=mix_cpm.index, columns=sig.columns)


def figure5(props, obs, labels=None, ax=None):
    """Stacked composition + case/control comparison for the contrast cohort."""
    k = (obs.cohort == "contrast").values
    P = props[k]
    d = obs.diagnosis[k].values
    order = np.argsort([0 if x == "control" else 1 for x in d], kind="stable")
    P, d = P.iloc[order], d[order]

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.2),
                              gridspec_kw={"width_ratios": [2.1, 1]})
    a0, a1 = ax
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(P.shape[0])
    for i, c in enumerate(P.columns):
        lab = labels.get(c, c) if labels else c
        a0.bar(np.arange(P.shape[0]), P[c].values, bottom=bottom, width=0.85,
               color=cmap(i % 20), label=lab, linewidth=0)
        bottom += P[c].values
    n_ctl = int((d == "control").sum())
    a0.axvline(n_ctl - 0.5, color="k", lw=1.4)
    for xc, lab in [(n_ctl / 2 - 0.5, "controls"),
                    (n_ctl + (P.shape[0] - n_ctl) / 2 - 0.5, "cases")]:
        a0.text(xc, 0.5, lab, ha="center", va="center", fontsize=10, fontweight="bold",
                color=INK, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK, lw=0.8, alpha=0.9))
    a0.set_xticks([]); a0.set_ylim(0, 1); a0.set_ylabel("estimated proportion")
    a0.set_title("Figure 5 — estimated cell-type proportions, Block A bulk samples", loc="left")
    a0.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.005, 1), loc="upper left", frameon=False)

    rows = []
    for c in P.columns:
        a, b = P[c][d == "case"].values, P[c][d == "control"].values
        rows.append((labels.get(c, c) if labels else c, a.mean() - b.mean(),
                     stats.mannwhitneyu(a, b).pvalue))
    df = pd.DataFrame(rows, columns=["type", "delta", "p"]).sort_values("delta")
    col = [ACCENT if v > 0 else BLUE for v in df.delta]
    a1.barh(np.arange(len(df)), df.delta.values, color=col, height=0.7)
    a1.set_yticks(np.arange(len(df)))
    a1.set_yticklabels(df.type.values, fontsize=8)
    a1.axvline(0, color="k", lw=0.8)
    a1.set_xlabel("Δ proportion  (case − control)")
    a1.set_title("composition shift", loc="left", fontsize=9)
    for i, (v, p) in enumerate(zip(df.delta.values, df.p.values)):
        if p < 0.05:
            a1.text(v + np.sign(v) * 0.002, i, "*", va="center",
                    ha="left" if v > 0 else "right", fontsize=11)
    return ax, df


# ===========================================================================
# BLOCK C2 -- real Visium: rat brain after status epilepticus
# ===========================================================================

RAT_MARKER_PANEL = {
    "upper layers":     ["Cux2", "Lamp5", "Calb1"],
    "layer 5":          ["Fezf2", "Bcl11b"],
    "deep layers":      ["Tle4", "Foxp2", "Ccn2", "Nr4a2"],
    "hippocampus":      ["Hpca", "Fibcd1", "Bok"],
    "thalamus":         ["Prkcd", "Tcf7l2"],
    "striatum":         ["Ppp1r1b", "Penk", "Gpr88"],
    "oligodendrocytes": ["Mbp", "Plp1", "Mobp"],
}

# what Figure 6b puts side by side: a reactive gene, stable astrocyte and
# microglia markers, and neuronal genes, so that "more cells" and "more
# transcript per cell" can be told apart
GLIOSIS_PANEL = ["Gfap", "Vim", "Aqp4", "Slc1a3", "Aif1", "Cd68", "Hpca", "Camk2a", "Snap25"]

DOMAIN_COLOURS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
                  "#937860", "#da8bc3", "#8c8c8c"]


def visium_normalise(v):
    """Raw counts to layers['counts']; log1p of counts per 10,000 in .X.

    The file holds a subset of the genes, so each spot is scaled by its library
    size over the whole transcriptome (obs['total_counts']), not by the sum of
    the genes that happen to be kept.
    """
    v = v.copy()
    v.layers["counts"] = v.X.copy()
    v.X = sparse.diags(1e4 / v.obs.total_counts.values.astype(float)) @ sparse.csr_matrix(v.X, dtype=np.float32)
    v.X = sparse.csr_matrix(v.X, dtype=np.float32)
    v.X.data = np.log1p(v.X.data)
    return v


def _tissue(ax, v, section, alpha=1.0):
    """Draw the H&E of one section; return the spot coordinates in image pixels."""
    u = v.uns["spatial"][section]
    k = (v.obs.section == section).values
    xy = v.obsm["spatial"][k] * u["scalefactors"]["tissue_lowres_scalef"]
    ax.imshow(u["images"]["lowres"], alpha=alpha)
    pad = 12
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return k, xy


def _spots(ax, v, section, xy, **kw):
    """Draw spots as discs of their true 55 um diameter, in image coordinates,
    so that their size does not depend on the figure layout."""
    from matplotlib.collections import EllipseCollection
    u = v.uns["spatial"][section]
    d = u["scalefactors"]["spot_diameter_fullres"] * u["scalefactors"]["tissue_lowres_scalef"]
    n = len(xy)
    ec = EllipseCollection(np.full(n, d), np.full(n, d), np.zeros(n), units="xy",
                           offsets=xy, offset_transform=ax.transData, linewidth=0, **kw)
    ax.add_collection(ec)
    return ec


def _label_spot(xy, inside):
    """The spot of a domain that lies deepest inside it: thin curved bands have
    a centroid that can fall outside them."""
    from scipy.spatial import cKDTree
    out = xy[~inside]
    depth = cKDTree(out).query(xy[inside])[0] if len(out) else np.ones(inside.sum())
    c = np.median(xy[inside], 0)
    score = depth - 0.05 * np.sqrt(((xy[inside] - c) ** 2).sum(1))
    return xy[inside][np.argmax(score)]


def plot_spots(ax, v, section, what, vmax=None, cmap="magma_r"):
    """One section, spots coloured by a gene or an obs column, over the H&E."""
    k, xy = _tissue(ax, v, section, alpha=0.3)
    if what in v.obs:
        vals = v.obs[what].values[k].astype(float)
    else:
        X = v[k, what].X
        vals = np.asarray(X.todense()).ravel() if sparse.issparse(X) else np.asarray(X).ravel()
    ec = _spots(ax, v, section, xy, array=vals, cmap=cmap)
    ec.set_clim(0, vmax if vmax is not None else np.percentile(vals, 99.5))
    ax.set_title(f"{what}  ·  " + _section_title(v, section), loc="left", fontsize=9)
    plt.colorbar(ec, ax=ax, fraction=0.04, pad=0.01)
    return ec


def _section_title(v, section):
    o = v.obs[v.obs.section == section].iloc[0]
    return f"{section}  ·  {o.condition}, day {o.days_post_SE}"


def domain_means(v, genes, by="domain"):
    """Mean log-normalised expression of each gene in each domain."""
    genes = [g for g in genes if g in v.var_names]
    X = v[:, genes].X
    X = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
    return pd.DataFrame(X, index=v.obs_names, columns=genes).groupby(
        v.obs[by].values, observed=True).mean()


def figure6(v, section, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0),
                                 gridspec_kw={"width_ratios": [1, 1, 1.55]})
    a0, a1, a2 = axes
    _tissue(a0, v, section)
    a0.set_title("Figure 6 — " + _section_title(v, section) + "  ·  H&E", loc="left")

    k, xy = _tissue(a1, v, section, alpha=0.35)
    dom = v.obs.domain[k].astype(str).values
    cats = list(v.obs.domain.cat.categories)
    for i, d in enumerate(cats):
        m = dom == d
        if not m.any():
            continue
        _spots(a1, v, section, xy[m], facecolors=DOMAIN_COLOURS[i % 8])
        a1.annotate(d, _label_spot(xy, m), fontsize=11, fontweight="bold", ha="center",
                    va="center", color="white",
                    bbox=dict(boxstyle="circle,pad=0.2", fc=DOMAIN_COLOURS[i % 8], ec="white", lw=1.2))
    a1.set_title("numbered spatial domains (clustering of spots)", loc="left", fontsize=9)

    genes, groups = [], []
    for grp, gl in RAT_MARKER_PANEL.items():
        gl = [g for g in gl if g in v.var_names]
        genes += gl; groups += [grp] * len(gl)
    X = v[:, genes].X
    X = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
    mean = np.zeros((len(cats), len(genes))); frac = np.zeros_like(mean)
    for i, d in enumerate(cats):
        m = (v.obs.domain.astype(str) == d).values
        mean[i] = X[m].mean(0); frac[i] = (X[m] > 0).mean(0)
    z = (mean - mean.mean(0)) / (mean.std(0) + 1e-9)
    yy, xx = np.meshgrid(np.arange(len(cats)), np.arange(len(genes)), indexing="ij")
    sm = a2.scatter(xx.ravel(), yy.ravel(), s=frac.ravel() * 110 + 1,
                    c=np.clip(z.ravel(), -1.5, 2.5), cmap="RdBu_r", linewidth=0)
    a2.set_xticks(range(len(genes))); a2.set_xticklabels(genes, rotation=90, fontsize=8, style="italic")
    a2.set_yticks(range(len(cats))); a2.set_yticklabels(cats, fontsize=10, fontweight="bold")
    for t, d in zip(a2.get_yticklabels(), cats):
        t.set_color(DOMAIN_COLOURS[cats.index(d) % 8])
    a2.set_ylim(len(cats) - 0.4, -2.3)
    prev, start, row = groups[0], 0, 0
    for i, g in enumerate(groups + [None]):
        if g != prev:
            a2.axvline(i - 0.5, color=INK, lw=0.5, alpha=0.3)
            y = -0.95 if row % 2 == 0 else -1.65
            a2.plot([start - 0.35, i - 0.65], [y - 0.18, y - 0.18], color=INK, lw=1.6,
                    solid_capstyle="butt", alpha=0.7)
            a2.text((start + i - 1) / 2, y, prev, ha="center", va="bottom", fontsize=7, color=INK)
            prev, start, row = g, i, row + 1
    a2.set_ylabel("domain")
    a2.set_title("markers by domain (4 sections)  ·  size = fraction of spots detecting, "
                 "colour = scaled mean", loc="left", fontsize=9, pad=16)
    a2.grid(axis="y", lw=0.3, alpha=0.3)
    plt.colorbar(sm, ax=a2, pad=0.01, shrink=0.6)
    return axes


def pseudobulk_log2fc(v, genes, days, by="domain"):
    """log2 fold change SE / control per domain, on summed raw counts (CPM).

    Summing every spot of a domain before comparing is the spatial version of a
    pseudobulk: one number per gene per domain per section, as in bulk RNA-seq.
    """
    C = v.layers["counts"] if "counts" in v.layers else v.X
    genes = [g for g in genes if g in v.var_names]
    gi = [v.var_names.get_loc(g) for g in genes]
    out = {}
    for cond in ("control", "SE"):
        k = ((v.obs.condition == cond) & (v.obs.days_post_SE == days)).values
        rows = {}
        for d in v.obs.domain.cat.categories:
            m = k & (v.obs[by].astype(str) == d).values
            tot = np.asarray(C[m].sum(0)).ravel()
            lib = v.obs.total_counts.values[m].sum()
            rows[d] = tot[gi] / lib * 1e6 if lib > 0 else np.full(len(gi), np.nan)
        out[cond] = pd.DataFrame(rows, index=genes)
    return np.log2((out["SE"] + 1) / (out["control"] + 1))


def figure6b(v, gene="Gfap", fig=None):
    if fig is None:
        fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.22, wspace=0.08)
    order = [("control", 5), ("SE", 5), ("control", 40), ("SE", 40)]
    secs = [v.obs.section[(v.obs.condition == c) & (v.obs.days_post_SE == d)].iloc[0]
            for c, d in order]
    X = v[:, gene].X
    vals = np.asarray(X.todense()).ravel() if sparse.issparse(X) else np.asarray(X).ravel()
    vmax = np.percentile(vals, 99.5)
    for j, s in enumerate(secs):
        ax = fig.add_subplot(gs[0, j])
        k, xy = _tissue(ax, v, s, alpha=0.3)
        sm = _spots(ax, v, s, xy, array=vals[k], cmap="magma_r")
        sm.set_clim(0, vmax)
        ax.set_title(_section_title(v, s), loc="left", fontsize=9)
        if j == 0:
            ax.text(0, 1.13, f"Figure 6b — {gene} in space, control vs SE (same colour scale)",
                    transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK)
    cb = fig.colorbar(sm, ax=fig.axes[-1], fraction=0.04, pad=0.01)
    cb.set_label(f"{gene}, log1p(counts per 10k)", fontsize=8)

    for j, days in enumerate((5, 40)):
        ax = fig.add_subplot(gs[1, 2 * j: 2 * j + 2])
        lfc = pseudobulk_log2fc(v, GLIOSIS_PANEL, days)
        im = ax.imshow(lfc.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(lfc.shape[1])); ax.set_xticklabels(lfc.columns, fontweight="bold")
        for t, d in zip(ax.get_xticklabels(), lfc.columns):
            t.set_color(DOMAIN_COLOURS[list(v.obs.domain.cat.categories).index(d) % 8])
        ax.set_yticks(range(lfc.shape[0]))
        ax.set_yticklabels(lfc.index if j == 0 else [], style="italic")
        for (r, c), x in np.ndenumerate(lfc.values):
            if np.isfinite(x):
                ax.text(c, r, f"{x:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(x) > 1.4 else INK)
        ax.set_xlabel("domain")
        ax.set_title(f"log2 fold change SE / control, day {days}  ·  pseudobulk per domain",
                     loc="left", fontsize=9)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label("log2 FC", fontsize=8)
    return fig


NEURONAL_SCORE_GENES = ["SNAP25", "SYT1", "NEUROD2", "NEUROD6", "RBFOX3", "MAP2",
                        "GAD1", "GAD2", "DLX2", "STMN2", "SYN1"]


def neuronal_clusters(s, key="cluster", genes=None):
    """Clusters whose mean neuronal-marker score is above the cluster median.

    A fallback for the deconvolution step so that a student who has not yet
    filled in their annotation still gets a meaningful Figure 5b.
    """
    genes = [g for g in (genes or NEURONAL_SCORE_GENES) if g in s.raw.var_names]
    X = s.raw[:, genes].X
    X = np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)
    score = pd.Series(X.mean(1)).groupby(s.obs[key].values, observed=True).mean()
    z = (score - score.mean()) / (score.std() + 1e-9)
    return list(z[z > 0.35].index)


def adjusted_de(b, props, neuronal_clusters):
    """Re-test case vs control with cell-type composition as a covariate.

    This is the analysis behind Q5: it separates the part of the Figure 2
    signal that is a change in p (composition) from the part that survives
    as a change in e (regulation).
    """
    cc = b[b.obs.cohort == "contrast"]
    X = _dense(cc) / np.log(2)                       # log2 CPM
    case = (cc.obs.diagnosis == "case").values.astype(float)
    neuro = props.loc[cc.obs_names, neuronal_clusters].sum(1).values

    out = {}
    for name, D in [("crude", np.column_stack([np.ones_like(case), case])),
                    ("adjusted", np.column_stack([np.ones_like(case), case, neuro]))]:
        beta, *_ = np.linalg.lstsq(D, X, rcond=None)
        resid = X - D @ beta
        dof = D.shape[0] - D.shape[1]
        s2 = (resid ** 2).sum(0) / dof
        xtxi = np.linalg.pinv(D.T @ D)
        se = np.sqrt(np.clip(s2 * xtxi[1, 1], 1e-12, None))
        t = beta[1] / se
        p = 2 * stats.t.sf(np.abs(t), dof)
        out[name] = pd.DataFrame({"log2FC": beta[1], "pval": p}, index=cc.var_names)
    res = out["crude"].join(out["adjusted"], lsuffix="_crude", rsuffix="_adj")
    res["shrinkage"] = 1 - (res.log2FC_adj.abs() / res.log2FC_crude.abs().clip(1e-9))
    return res


def figure5b(res, synaptic, immune, ax=None):
    """Crude vs composition-adjusted effect size -- the Q5 payoff plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 4.6))
    lim = np.nanpercentile(np.abs(res.log2FC_crude), 99.8)
    ax.axline((0, 0), slope=1, color=INK, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color=INK, lw=0.6, alpha=0.4); ax.axvline(0, color=INK, lw=0.6, alpha=0.4)
    other = ~(res.index.isin(synaptic) | res.index.isin(immune))
    ax.scatter(res.log2FC_crude[other], res.log2FC_adj[other], s=4, c=GREY, alpha=0.4, linewidth=0)
    ax.scatter(res.log2FC_crude[res.index.isin(synaptic)], res.log2FC_adj[res.index.isin(synaptic)],
               s=10, c=BLUE, alpha=0.8, linewidth=0, label="synaptic / neuronal")
    ax.scatter(res.log2FC_crude[res.index.isin(immune)], res.log2FC_adj[res.index.isin(immune)],
               s=10, c=ACCENT, alpha=0.8, linewidth=0, label="glial / immune")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("log₂FC, cases vs controls")
    ax.set_ylabel("log₂FC, adjusted for neuronal fraction")
    ax.set_title("Composition or regulation?", loc="left")
    ax.text(0.98, 0.03, "on the dashed line: composition explains nothing\n"
                        "pulled toward zero: composition explained part of it",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="right", color="#444",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", lw=0.6, alpha=0.92))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return ax


# ===========================================================================

def main(datadir, outdir):
    # headless only here: imported from the notebook, the inline backend must
    # stay active or no figure is ever displayed
    plt.switch_backend("Agg")
    os.makedirs(outdir, exist_ok=True)
    bulk = ad.read_h5ad(os.path.join(datadir, "bulk_cortex.h5ad"))
    scd = ad.read_h5ad(os.path.join(datadir, "fetal_cortex_sc.h5ad"))
    vis = ad.read_h5ad(os.path.join(datadir, "rat_visium_SE.h5ad"))
    report = {}

    b, de, vr = block_a(bulk)
    report["pc_variance_ratio"] = [round(float(x), 4) for x in vr[:5]]
    for cov in ["age_pcw", "diagnosis", "region", "batch", "sex", "RIN", "PMI_h"]:
        vals = b.obs[cov]
        if vals.dtype.kind in "fi":
            report[f"corr_PC1_{cov}"] = round(float(np.corrcoef(b.obsm['X_pca'][:, 0], vals)[0, 1]), 3)
            report[f"corr_PC2_{cov}"] = round(float(np.corrcoef(b.obsm['X_pca'][:, 1], vals)[0, 1]), 3)
        else:
            for pc in (0, 1):
                gs = [b.obsm["X_pca"][(vals == lv).values, pc] for lv in vals.unique()]
                if len(gs) > 1:
                    report[f"anovaF_PC{pc+1}_{cov}"] = round(float(stats.f_oneway(*gs).statistic), 2)

    from cc_datasets import SYNAPTIC_MODULE as SYNAPTIC, IMMUNE_MODULE as IMMUNE_GLIAL
    fig = plt.figure(figsize=(6.4, 4.6)); figure1(b, fig.add_subplot(111))
    fig.tight_layout(); fig.savefig(f"{outdir}/figure1.png"); plt.close(fig)

    fig = plt.figure(figsize=(6.6, 4.8)); _, nsig = figure2(de, SYNAPTIC, IMMUNE_GLIAL, fig.add_subplot(111))
    fig.tight_layout(); fig.savefig(f"{outdir}/figure2.png"); plt.close(fig)
    report["n_significant_genes"] = nsig
    sig = de[(de.padj < 0.05) & (de.log2FC.abs() > 0.58)]
    report["n_up_in_cases"] = int((sig.log2FC > 0).sum())
    report["n_down_in_cases"] = int((sig.log2FC < 0).sum())
    report["synaptic_in_sig_down"] = int(sig[sig.log2FC < 0].index.isin(SYNAPTIC).sum())
    report["immune_glial_in_sig_up"] = int(sig[sig.log2FC > 0].index.isin(IMMUNE_GLIAL).sum())
    report["SYN1_log2FC"] = round(float(de.loc["SYN1", "log2FC"]), 3) if "SYN1" in de.index else None
    report["top_up"] = sig.sort_values("log2FC", ascending=False).head(12).index.tolist()
    report["top_down"] = sig.sort_values("log2FC").head(12).index.tolist()

    s = block_b_qc(scd.copy())
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6)); figure3(s, axes)
    fig.tight_layout(); fig.savefig(f"{outdir}/figure3.png"); plt.close(fig)
    report["n_cells_before_qc"] = int(s.n_obs)
    report["n_cells_flagged"] = int(s.obs.qc_fail.sum())
    report["qc_recall_by_kind"] = (s.obs.groupby("droplet_kind", observed=True).qc_fail.mean()
                                   .round(3).to_dict())
    report["median_pct_mt"] = round(float(np.median(s.obs.pct_counts_mt)), 2)
    report["pct_mt_by_type"] = (s.obs[s.obs.droplet_kind == "cell"]
                               .groupby("true_cell_type", observed=True).pct_counts_mt.median()
                               .round(2).sort_values(ascending=False).to_dict())
    report["cells_lost_at_fixed_5pct_MT"] = int((s.obs.pct_counts_mt > 5).sum())

    s = block_b_cluster(s)
    fig = figure4(s); fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.24); fig.savefig(f"{outdir}/figure4.png"); plt.close(fig)
    xt = pd.crosstab(s.obs.cluster, s.obs.true_cell_type)
    report["cluster_composition"] = {c: xt.loc[c].sort_values(ascending=False).head(3).to_dict()
                                     for c in xt.index}
    report["cluster_dominant_type"] = xt.idxmax(1).to_dict()
    report["n_clusters"] = int(xt.shape[0])

    sig_mat = signature_matrix(s)
    report["n_signature_genes"] = int(sig_mat.shape[0])
    Cb = _dense(bulk[:, b.var_names])
    Cb = Cb / Cb.sum(1, keepdims=True) * 1e4
    bulk_cpm = pd.DataFrame(Cb, index=b.obs_names, columns=b.var_names)
    props = deconvolve(bulk_cpm, sig_mat)
    labels = {c: f"{c} · {report['cluster_dominant_type'][c]}" for c in sig_mat.columns}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [2.1, 1]})
    _, comp = figure5(props, b.obs, labels, axes)
    fig.tight_layout(); fig.savefig(f"{outdir}/figure5.png"); plt.close(fig)
    report["composition_shift"] = comp.round(4).to_dict("records")
    truth = bulk.obsm["true_proportions"]
    kk = (b.obs.cohort == "contrast").values
    tc = truth[kk]
    dd = b.obs.diagnosis[kk].values
    report["true_composition_shift"] = {c: round(float(tc[c][dd == "case"].mean()
                                                       - tc[c][dd == "control"].mean()), 4)
                                        for c in tc.columns}

    neuronal_clusters = [c for c in props.columns
                         if labels[c].split("· ")[-1] in
                         ("ExN_deep", "ExN_upper", "IN_MGE", "IN_CGE", "CR")]
    adj = adjusted_de(b, props, neuronal_clusters)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    figure5b(adj, SYNAPTIC, IMMUNE_GLIAL, ax)
    fig.tight_layout(); fig.savefig(f"{outdir}/figure5b.png"); plt.close(fig)
    syn = adj.index.isin(SYNAPTIC); imm = adj.index.isin(IMMUNE_GLIAL)
    report["Q5_median_shrinkage_synaptic"] = round(float(adj.shrinkage[syn].median()), 3)
    report["Q5_median_shrinkage_immune"] = round(float(adj.shrinkage[imm].median()), 3)
    report["Q5_synaptic_still_sig_after_adjust"] = int(
        ((adj.pval_adj[syn] < 0.05) & (adj.log2FC_adj[syn] < 0)).sum())
    report["Q5_synaptic_sig_before_adjust"] = int(
        ((adj.pval_crude[syn] < 0.05) & (adj.log2FC_crude[syn] < 0)).sum())
    report["Q5_immune_still_sig_after_adjust"] = int(
        ((adj.pval_adj[imm] < 0.05) & (adj.log2FC_adj[imm] > 0)).sum())
    report["Q5_immune_sig_before_adjust"] = int(
        ((adj.pval_crude[imm] < 0.05) & (adj.log2FC_crude[imm] > 0)).sum())

    v = visium_normalise(vis)
    report["visium_spots_per_section"] = v.obs.section.value_counts().sort_index().to_dict()
    report["visium_median_umi_per_section"] = (
        v.obs.groupby("section", observed=True).total_counts.median().round(0).to_dict())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1, 1, 1.55]})
    figure6(v, "B_L1_S2", axes)
    fig.tight_layout(); fig.savefig(f"{outdir}/figure6.png"); plt.close(fig)
    fig = figure6b(v, "Gfap"); fig.savefig(f"{outdir}/figure6b.png", bbox_inches="tight"); plt.close(fig)
    if "true_domain" in v.obs:
        report["domain_number_to_anatomy"] = (v.obs.groupby("domain", observed=True).true_domain
                                              .agg(lambda x: x.mode()[0]).to_dict())
    # cortical depth: median distance from the white matter, in um (spot pitch = 100 um)
    from scipy.spatial import cKDTree
    wm = report.get("domain_number_to_anatomy", {})
    wm = [d for d, a in wm.items() if a.startswith("white matter")]
    if wm:
        depth = {}
        for s in v.obs.section.cat.categories:
            k = (v.obs.section == s).values
            xy = v.obsm["spatial"][k]
            pitch = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
            t = cKDTree(xy[(v.obs.domain[k].astype(str) == wm[0]).values])
            d, _ = t.query(xy)
            depth[s] = pd.Series(d / pitch * 100).groupby(v.obs.domain[k].astype(str).values).median()
        depth = pd.DataFrame(depth)
        report["median_distance_from_white_matter_um"] = depth.mean(1).round(0).to_dict()
    report["log2fc_SE_vs_control_day5"] = pseudobulk_log2fc(v, GLIOSIS_PANEL, 5).round(2).to_dict()
    report["log2fc_SE_vs_control_day40"] = pseudobulk_log2fc(v, GLIOSIS_PANEL, 40).round(2).to_dict()

    with open(f"{outdir}/expected_outputs.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
