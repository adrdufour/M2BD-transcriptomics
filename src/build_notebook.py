"""Assemble M2BD_CC_practical.ipynb from cc_datasets.py + cc_pipeline.py."""

import json, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = open(os.path.join(HERE, "cc_datasets.py")).read()
PIPELINE = open(os.path.join(HERE, "cc_pipeline.py")).read()

# the notebook writes the pipeline out as a helper module; strip its __main__
PIPELINE_LIB = PIPELINE.split('if __name__ == "__main__":')[0]
DATASETS_LIB = DATASETS.split('if __name__ == "__main__":')[0]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text, hide=False):
    meta = {"cellView": "form"} if hide else {}
    cells.append({"cell_type": "code", "execution_count": None, "metadata": meta,
                  "outputs": [], "source": text.strip("\n")})


# ==========================================================================
md("""
<a href="https://colab.research.google.com/github/adrdufour/M2BD-transcriptomics/blob/main/M2BD_CC_practical.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Transcriptomic analyses in neurodevelopment — practical session

**M2 Brain Disorders · UE2 Genetics and Neurodevelopment**  ·  Adrien Dufour, Inserm

> **Save your own copy first:** *File → Save a copy in Drive*. Otherwise nothing you type here
> survives the end of the session.

Three blocks, six numbered figures. Your graded worksheet asks about each figure **by its number**,
so when a cell prints `Figure 3`, that is the figure the worksheet means.

| Block | Time | You produce | Worksheet |
|---|---|---|---|
| **A** — bulk RNA-seq | 15 min | Figure 1, Figure 2 | Q1, Q2 |
| **B** — single-cell | 25 min | Figure 3, Figure 4 | Q3, Q4 |
| **C** — spatial + integration | 20 min | Figure 5, Figure 6 | Q5, Q6, Q7 |

**Two rules for today.** You are marked on how you *read* your own figures, not on whether your code
ran — so if a cell fails, move on and answer from the figure handout. And answer the worksheet **as you
go**: the last ten minutes are not enough.

---
### Run this first (≈ 2 min)
""")

code(f'''
# SETUP — run this cell once. About two minutes, most of it the pip install.

import os, sys, subprocess, textwrap, urllib.request, pathlib

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    subprocess.run("pip install -q scanpy leidenalg igraph", shell=True, check=False)

# The three datasets live in the course repository. If GitHub is unreachable
# from this room, the identical data is rebuilt locally from a fixed seed (~5 s),
# so the session works with no network at all.
REPO = "adrdufour/M2BD-transcriptomics"
DATA_URL_BASE = os.environ.get(
    "M2BD_DATA_URL", f"https://raw.githubusercontent.com/{{REPO}}/main/data")
DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
FILES = ["bulk_cortex.h5ad", "fetal_cortex_sc.h5ad", "cortex_visium.h5ad"]

# --- the two helper modules, written to disk so you can open and read them ---
pathlib.Path("cc_datasets.py").write_text({DATASETS_LIB!r})
pathlib.Path("cc_pipeline.py").write_text({PIPELINE_LIB!r})

def _looks_like_h5ad(path):
    # a 404 page or a Git-LFS pointer would download happily and fail later
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == b"\\x89HDF\\r\\n\\x1a\\n"
    except OSError:
        return False


got_remote = False
if DATA_URL_BASE:
    try:
        for f in FILES:
            if not _looks_like_h5ad(DATA_DIR / f):
                urllib.request.urlretrieve(DATA_URL_BASE.rstrip("/") + "/" + f, DATA_DIR / f)
        got_remote = all(_looks_like_h5ad(DATA_DIR / f) for f in FILES)
        if not got_remote:
            print("what downloaded was not an .h5ad — rebuilding locally instead")
    except Exception as e:
        print("download failed (" + type(e).__name__ + ") — rebuilding locally instead")

if not got_remote:
    import cc_datasets
    cc_datasets.build_all(str(DATA_DIR))

import numpy as np, pandas as pd, scanpy as sc, anndata as ad
import matplotlib.pyplot as plt
import cc_pipeline as cc
from cc_pipeline import figure1, figure2, figure3, figure4, figure5, figure5b, figure6

sc.settings.verbosity = 1
plt.rcParams["figure.dpi"] = 110
FIGDIR = pathlib.Path("figures"); FIGDIR.mkdir(exist_ok=True)

print("ready ·", "downloaded" if got_remote else "built locally")
for f in FILES:
    print(" ", f, round((DATA_DIR / f).stat().st_size / 1e6, 1), "MB")
''')

# ------------------------------------------------------------------ BLOCK A
md("""
---
# Block A — bulk RNA-seq  ·  15 minutes

`bulk_cortex.h5ad` holds 64 cortex samples: a **developmental series** of 24 control samples
from 8 to 39 post-conceptional weeks, and a **case/control contrast** of 40 age-matched late-fetal
samples (20 cases, 20 controls). Each row is one sample, each column one gene, each value a raw count.

The whole of Block A rests on one sentence: *a bulk measurement is a weighted average over a mixture
of cells.* Keep it in mind — Block C comes back to it.
""")

code('''
bulk = sc.read_h5ad("data/bulk_cortex.h5ad")
print(bulk)
bulk.obs.head(8)
''')

code('''
# What is actually in the metadata? Look before you model.
bulk.obs.groupby(["cohort", "diagnosis"], observed=True).size()
''')

md("""
### A1 · Normalise, then look at the samples — **Figure 1**

Counts are not comparable between samples until you correct for library size. Here: counts → CPM →
`log1p`, then PCA on the 2000 most variable genes.

PCA does not know what your samples are. It finds the directions along which they differ *most*.
Whatever comes out on PC1 is the largest source of variation in your experiment — and it is very
often not the thing you are studying.
""")

code('''
b, de, var_ratio = cc.block_a(bulk)        # normalise → log1p → HVG → PCA, then the case/control test
print("variance explained by the first five PCs:",
      np.round(var_ratio[:5] * 100, 1), "%")

fig, ax = plt.subplots(figsize=(6.4, 4.6))
figure1(b, ax); fig.tight_layout()
fig.savefig("figures/figure1.png", dpi=200)
''')

md("""
> **Worksheet Q1a.** Which variable best explains the separation along PC1? Justify it *from the plot*.
>
> Before you answer, colour the same plot by something else. The cell below does that — change
> `"batch"` to `"region"`, `"sex"`, `"RIN"` or `"PMI_h"` and look again. **Q1b** asks which covariate
> you would check before trusting any differential expression result, and what you would conclude if
> cases and controls separated on it.
""")

code('''
COLOUR_BY = "batch"          # <-- try: batch, region, sex, RIN, PMI_h, diagnosis

pcs = b.obsm["X_pca"]
v = b.obs[COLOUR_BY]
fig, ax = plt.subplots(figsize=(5.6, 4.4))
if v.dtype.kind in "fi":
    sm = ax.scatter(pcs[:, 0], pcs[:, 1], c=v, cmap="magma", s=55, edgecolor="w", lw=0.6)
    plt.colorbar(sm, ax=ax, label=COLOUR_BY)
else:
    for lev in sorted(v.unique()):
        k = (v == lev).values
        ax.scatter(pcs[k, 0], pcs[k, 1], s=55, edgecolor="w", lw=0.6, label=str(lev))
    ax.legend(frameon=False, title=COLOUR_BY)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("same PCA, coloured by " + COLOUR_BY)
fig.tight_layout()
''')

md("""
### A2 · Differential expression — **Figure 2**

The contrast is cases vs controls, restricted to the 40 age-matched samples. Two things decide whether
a gene is on the plot: the **effect size** (log₂ fold change, x-axis) and the **evidence** (p-value,
y-axis, corrected for testing ~2700 genes at once with Benjamini–Hochberg).

`|log₂FC| > 0.58` is a 1.5-fold change. Neither axis alone is enough: a tiny change can be highly
significant with a large n, and a huge fold change in a barely-expressed gene means nothing.
""")

code('''
from cc_datasets import SYNAPTIC_MODULE, IMMUNE_MODULE

fig, ax = plt.subplots(figsize=(6.6, 4.9))
_, n_sig = figure2(de, SYNAPTIC_MODULE, IMMUNE_MODULE, ax)
fig.tight_layout(); fig.savefig("figures/figure2.png", dpi=200)

sig = de[(de.padj < 0.05) & (de.log2FC.abs() > 0.58)]
print(f"{len(sig)} genes pass:  {int((sig.log2FC < 0).sum())} down and "
      f"{int((sig.log2FC > 0).sum())} up in cases")
''')

code('''
# The named genes on each side. This is the table Q2a wants you to read.
named = [g for g in sig.index if not g.startswith(("GENE", "AC0"))]
display(sig.loc[named].sort_values("log2FC").head(15).round(3))
display(sig.loc[named].sort_values("log2FC", ascending=False).head(15).round(3))
print("SYN1:", de.loc["SYN1"].round(4).to_dict())
''')

md("""
> **Worksheet Q2a.** Roughly how many genes pass, and which functional category dominates each side?
>
> **Worksheet Q2b.** A reviewer objects that a log₂FC of −0.4 on *SYN1* is too small to matter.
> Give one argument for the reviewer and one against. Look at how highly *SYN1* is expressed
> (`b.raw[:, "SYN1"]`) before you decide, and at how many other genes move in the same direction.
""")

# ------------------------------------------------------------------ BLOCK B
md("""
---
# Block B — single-cell  ·  25 minutes

`fetal_cortex_sc.h5ad` is 8,000 nuclei from mid-gestation human cortex, **raw counts with no QC
applied**. Every barcode in this file is a droplet, and a droplet is not the same thing as a cell:
some are empty and contain only ambient RNA, some contain two cells, and some contain a cell that
was already damaged when it went in.

Deciding which barcodes to keep is the first real analytical decision of the day, and it is a
*biological* decision, not a threshold you copy from a tutorial.
""")

code('''
sc_data = sc.read_h5ad("data/fetal_cortex_sc.h5ad")
print(sc_data)
print("\\nsparsity: {:.1%} of the matrix is zero".format(
    1 - sc_data.X.nnz / (sc_data.n_obs * sc_data.n_vars)))
''')

md("""
### B1 · QC as a decision — **Figure 3**

Three metrics per barcode: total UMI counts, number of genes detected, and the percentage of counts
coming from mitochondrial transcripts.

Instead of fixed cut-offs, flag barcodes that are more than *k* **MADs** (median absolute deviations)
from the median of *this* sample. A MAD is a median-based standard deviation: it is not dragged around
by the outliers you are trying to find, and it adapts to each dataset.
""")

code('''
N_MAD = 3        # <-- try 3 and 5 and watch how many cells you lose
MT_MAD = 3

s = cc.block_b_qc(sc_data.copy(), n_mad=N_MAD, mt_mad=MT_MAD)

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
figure3(s, axes); fig.tight_layout(); fig.savefig("figures/figure3.png", dpi=200)

print(s.obs[["out_counts", "out_genes", "out_mt", "qc_fail"]].sum())
print(f"\\n{int(s.obs.qc_fail.sum())} of {s.n_obs} barcodes flagged "
      f"({s.obs.qc_fail.mean():.1%})")
print("median %MT:", round(float(np.median(s.obs.pct_counts_mt)), 2))
print("a blanket 5% MT cut-off would remove:", int((s.obs.pct_counts_mt > 5).sum()), "barcodes")
''')

md("""
> **Worksheet Q3a.** Which cells would you remove, on which metric, at what threshold — and *how did
> you choose it*? "Because the tutorial said so" is worth half marks.
>
> **Worksheet Q3b.** Compare the two numbers the cell above printed. Which population are you deleting
> with a blanket 5% mitochondrial cut-off in fetal cortex, and why is that population MT-rich?
""")

md("""
### B2 · Embed, cluster, annotate

The standard route: normalise → log → pick highly variable genes → PCA → neighbour graph → Leiden
→ UMAP. This takes about a minute.

The UMAP is a *picture of the neighbour graph*, nothing more. Distance between clusters on it is not
meaningful, and a cluster is a hypothesis, not a cell type.
""")

code('''
RESOLUTION = 0.6      # <-- higher splits more; the number of clusters is your choice, not a fact

s = cc.block_b_cluster(s, resolution=RESOLUTION)
print(s.obs.cluster.value_counts().sort_index())
''')

md("""
### B3 · Markers make it a cell type — **Figure 4**

The dot plot: one row per cluster, one column per marker gene. **Dot size** = the fraction of cells in
that cluster expressing the gene; **colour** = the mean expression, scaled across clusters.

A good call needs a marker that is *specific*. `SLC1A3` alone will not separate radial glia from
astrocytes — both express it.
""")

code('''
fig = figure4(s)
fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.24)
fig.savefig("figures/figure4.png", dpi=200)
''')

code('''
# The unbiased version: what are the top-ranked genes of each cluster, marker panel or not?
sc.tl.rank_genes_groups(s, "cluster", method="wilcoxon")
pd.DataFrame(s.uns["rank_genes_groups"]["names"]).head(8)
''')

md("""
> **Worksheet Q4a.** Give clusters **A**, **B** and **C** a cell-type identity, and name the marker
> gene(s) you used for each. Identity without a marker scores half.
>
> **Worksheet Q4b.** One of the three is ambiguous. Say which, and name one extra gene or one analysis
> that would settle it. The `rank_genes_groups` table above is a good place to look.

Fill in your calls below — Block C uses them.
""")

code('''
# Your annotation. Leave a cluster as its letter if you are not sure.
MY_LABELS = {
    "A": "?",
    "B": "?",
    "C": "?",
}
for c in s.obs.cluster.cat.categories:
    MY_LABELS.setdefault(c, c)
MY_LABELS
''')

# ------------------------------------------------------------------ BLOCK C
md("""
---
# Block C — spatial and integration  ·  20 minutes

This is where the three technologies stop being three separate exercises.

### C1 · Deconvolve the Block A bulk with your Block B annotation — **Figure 5**

A bulk sample is a mixture. Written out:

$$ y_g \\;=\\; \\sum_k p_k \\, e_{g,k} $$

the bulk expression of gene *g* is the sum, over cell types *k*, of that type's proportion $p_k$ times
what one cell of that type transcribes, $e_{g,k}$. Your single-cell object gives you $e$. Solving for
$p$ is **deconvolution** — here, a non-negative least squares fit of each bulk sample against the
cluster signature.

Then look again at Figure 2. If $y_g$ changed, either $p$ changed, or $e$ changed, or both.
""")

code('''
signature = cc.signature_matrix(s)             # linear-scale mean expression per cluster
print("signature:", signature.shape[0], "genes ×", signature.shape[1], "clusters")

# bulk on the same linear CPM scale the equation is written on
Cb = cc._dense(bulk[:, b.var_names]); Cb = Cb / Cb.sum(1, keepdims=True) * 1e4
bulk_cpm = pd.DataFrame(Cb, index=b.obs_names, columns=b.var_names)

props = cc.deconvolve(bulk_cpm, signature)
labels = {c: f"{c} · {MY_LABELS.get(c, c)}" for c in props.columns}

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})
_, comp = figure5(props, b.obs, labels, axes)
fig.tight_layout(); fig.savefig("figures/figure5.png", dpi=200)
comp.round(4)
''')

md("""
Now the analysis that actually answers the question. Re-run the case/control test with the estimated
neuronal fraction as a covariate, and plot each gene's effect size before against after.

A gene that stays on the diagonal is one that composition does not explain. A gene pulled towards zero
is one whose apparent regulation was, at least partly, a change in how many neurons were in the tube.
""")

code('''
# Which of your clusters are neurons? Taken from your MY_LABELS if you filled it in;
# otherwise scored automatically from SNAP25 / NEUROD6 / GAD1 and friends.
NEURONAL = [c for c in props.columns
            if MY_LABELS.get(c, "").lower().startswith(("exn", "in_", "cr", "excit", "inter", "neuro"))]
if not NEURONAL:
    NEURONAL = [c for c in cc.neuronal_clusters(s) if c in props.columns]
    print("(MY_LABELS not filled in — scoring neuronal clusters from markers instead)")
print("treating these as neuronal:", NEURONAL)

adj = cc.adjusted_de(b, props, NEURONAL)
fig, ax = plt.subplots(figsize=(5.4, 5.0))
figure5b(adj, SYNAPTIC_MODULE, IMMUNE_MODULE, ax)
fig.tight_layout(); fig.savefig("figures/figure5b.png", dpi=200)

syn = adj.index.isin(SYNAPTIC_MODULE)
print("\\nsynaptic module, down in cases:")
print("  significant before adjusting:",
      int(((adj.pval_crude < 0.05) & (adj.log2FC_crude < 0) & syn).sum()))
print("  significant after  adjusting:",
      int(((adj.pval_adj < 0.05) & (adj.log2FC_adj < 0) & syn).sum()))
print("  median shrinkage of the effect: {:.0%}".format(adj.shrinkage[syn].median()))
''')

md("""
> **Worksheet Q5a — the keystone question.** Using the proportions in Figure 5, re-explain the
> differential expression result you described in Q2a. Composition, regulation, or both? Say what
> evidence you are using — the numbers the cell above printed are evidence.
>
> **Worksheet Q5b.** Name one limitation of *your* deconvolution that could make that conclusion
> wrong. Compare the microglial proportion you estimated to what you would expect in fetal cortex,
> and think about where the reference came from.
""")

md("""
### C2 · Where in the tissue? — **Figure 6**

`cortex_visium.h5ad` is one section: ~1,500 spots on a hexagonal grid, each 55 µm across, each with a
full transcriptome and an (x, y) position. The spots are numbered into six groups; nothing tells you
which is which.

Deconvolve each spot against the same single-cell signature, and the tissue's architecture appears —
because in a cortex built by radial migration, position *is* the phenotype.
""")

code('''
visium = sc.read_h5ad("data/cortex_visium.h5ad")

common = [g for g in signature.index if g in visium.var_names]
Xv = cc._dense(visium[:, common]) / cc._dense(visium).sum(1, keepdims=True) * 1e4
spot_props = cc.deconvolve(pd.DataFrame(Xv, index=visium.obs_names, columns=common),
                           signature.loc[common])

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4), gridspec_kw={"width_ratios": [1, 1.35]})
figure6(visium, spot_props, labels, axes)
fig.tight_layout(); fig.savefig("figures/figure6.png", dpi=200)
''')

code('''
# The composition of each numbered group, as estimated proportions.
grp = spot_props.groupby(visium.obs.spot_group.values).mean()
grp.columns = [labels.get(c, c) for c in grp.columns]
grp.round(3).T
''')

md("""
> **Worksheet Q6a.** Order the numbered spot groups from the **ventricular** surface to the **pial**
> surface, and name the compartment each one is (VZ, SVZ, IZ, SP, CP, MZ). The table above tells you
> what each group is made of; the marker biology from Block B tells you where that belongs.
>
> **Worksheet Q6b.** Roughly how many cells are in one 55 µm spot here, and what does that mean for
> the "cell type" you just assigned to it? The section's own estimate is in
> `visium.obs.n_cells_estimate` — look at its distribution.
""")

code('''
visium.obs.n_cells_estimate.describe()
''')

md("""
### C3 · Q7 — no code

> You want to test whether **outer radial glia are reduced in a cortical malformation**. You have 40
> frozen post-mortem samples (20 cases, 20 controls), and budget for bulk RNA-seq on all 40 plus
> **one** of the other two technologies on about 8 samples.
>
> In 4–6 sentences: what would you run on what, in which order, and what is your primary readout?
> Then name one question your design cannot answer.

Two things you have seen today are worth remembering while you write. Deconvolution needs a reference
built first, from the same stage and region. And the frozen tissue rules out one of the three assays
in the form you used it here.

---
### Before you hand in

All six figures are saved in `figures/`. Check that the numbers in your answers match your own output,
not the lecture slides.
""")

code('''
import glob
for f in sorted(glob.glob("figures/*.png")):
    print(f, round(os.path.getsize(f) / 1e3), "kB")

if IN_COLAB:
    from google.colab import files
    # uncomment to download all six figures as a zip
    # !zip -qr figures.zip figures && files.download("figures.zip")
''')

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = sys.argv[1] if len(sys.argv) > 1 else "M2BD_CC_practical.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "·", len(cells), "cells")
