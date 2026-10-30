"""
Convert the authors' Seurat RData (16 Visium sections) to one AnnData, without R.

The RData is 8 GB gzipped and 28 GB once loaded -- more than R can hold on a
laptop. rstream.py reads it as a stream and keeps only the raw counts, the
images, the spot coordinates, the metadata and the UMAP; the SCT and
integrated scale.data matrices are skipped on the fly (peak memory ~9 GB).

    python seurat_to_h5ad.py whole_clustering_subclustering_Seurat_objects.RData OUT.h5ad

The output is instructor material (all 16 sections, the authors' clusters);
prepare_visium.py turns it into the four-section student dataset.
"""
import sys, os, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rstream

r = rstream.Reader(sys.argv[1], keep_big=r"^seurat_object_whole_clustering/assays/\[\d+\]/counts|/image$")
r.header()
top = r.item()
objs = dict(top.value)
names = lambda o: o.a('names').value

def frame(o):
    d = {}
    for c, v in zip(names(o), o.value):
        if v.a('levels') is not None:
            lev = np.array(v.a('levels').value, dtype=object)
            d[c] = pd.Categorical.from_codes(np.where(v.value < 0, -1, v.value - 1), lev)
        elif v.type == 16:
            d[c] = np.array(v.value, dtype=object)
        else:
            d[c] = v.value
    rn = o.a('row.names')
    return pd.DataFrame(d, index=rn.value if rn.type == 16 else None)

def dgc(m):
    nr, nc = m.a('Dim').value
    M = sparse.csc_matrix((m.a('x').value, m.a('i').value, m.a('p').value), shape=(nr, nc))
    dn = m.a('Dimnames').value
    return M, dn[0].value, dn[1].value

W = objs['seurat_object_whole_clustering']
assays = dict(zip(names(W.a('assays')), W.a('assays').value))
X, genes, cells = dgc(assays['RNA'].a('counts'))
md = frame(W.a('meta.data')).loc[cells]
A = ad.AnnData(X=X.T.tocsr().astype(np.float32), obs=md, var=pd.DataFrame(index=genes))
red = dict(zip(names(W.a('reductions')), W.a('reductions').value))
for r in ('umap',):
    e = red[r].a('cell.embeddings')
    A.obsm['X_' + r] = e.value.reshape(e.a('dim').value[::-1]).T
# pixel coordinates in full-resolution image space, per section
A.obsm['spatial'] = np.zeros((A.n_obs, 2))
A.obs['array_row'] = 0; A.obs['array_col'] = 0
A.uns['spatial'] = {}
imgs = dict(zip(names(W.a('images')), W.a('images').value))
for lib, im in imgs.items():
    co = frame(im.a('coordinates'))
    idx = A.obs_names.get_indexer(co.index)
    ok = idx >= 0
    sf = dict(zip(names(im.a('scale.factors')), [float(v.value[0]) for v in im.a('scale.factors').value]))
    # Seurat stores imagerow/imagecol in full-resolution pixels
    A.obsm['spatial'][idx[ok]] = co[['imagecol', 'imagerow']].values[ok]
    A.obs.iloc[idx[ok], A.obs.columns.get_loc('array_row')] = co['row'].values[ok]
    A.obs.iloc[idx[ok], A.obs.columns.get_loc('array_col')] = co['col'].values[ok]
    arr = im.a('image'); dim = arr.a('dim').value
    img = arr.value.reshape(dim[::-1]).transpose(2, 1, 0)  # R is column-major: (h, w, c)
    A.uns['spatial'][lib] = {
        'images': {'lowres': img.astype(np.float32)},
        'scalefactors': {'tissue_lowres_scalef': sf['lowres'], 'tissue_hires_scalef': sf['hires'],
                         'spot_diameter_fullres': sf['spot'], 'fiducial_diameter_fullres': sf['fiducial']},
    }
    print(lib, ok.sum(), img.shape, sf)
for c in A.obs.columns:
    if A.obs[c].dtype == object:
        A.obs[c] = A.obs[c].astype(str)
A.write_h5ad(sys.argv[2], compression='gzip')
print(A)
