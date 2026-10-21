import scanpy as sc
import os 
import pandas as pd
import anndata as ad
import numpy as np
import pertpy as pt

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

adata_counts = sc.read('/lustre/scratch126/cellgen/lotfollahi/so21/neuroID/preprocessing/adatas/counts_filtered_correctGuides.h5ad')
adata_counts

print (adata_counts.X.shape)
print (adata_counts.obs.Replicate.value_counts())
adata_counts = adata_counts[~adata_counts.obs.target_gene.isin(['INT3','INT30'])].copy()
print (adata_counts.X.shape)
print (adata_counts.obs.Replicate.value_counts())

import decoupler as dc

pdata = dc.pp.pseudobulk(adata_counts,
                         sample_col="Replicate", 
                         groups_col="perturbation",
                         layer=None,
                         mode="sum",
                        )
dc.pp.filter_samples(pdata, min_cells=10, min_counts=1000, inplace=True, )
pdata

pds2 = pt.tl.PyDESeq2(adata=pdata, design="~Replicate+perturbation")
pds2

pds2.fit()

out_dir = "DEGs/pyDESeq2"
os.makedirs(out_dir, exist_ok=True)

results = {}
for pert in pdata.obs["perturbation"].cat.categories:
    if pert != "INT":
        contrast = pds2.contrast("perturbation", baseline="INT", group_to_compare=pert)
        results[pert] = pds2.test_contrasts(contrast)
        
output_path = os.path.join(out_dir, "DESeq2_iNeurons_results_v3_all.xlsx")
with pd.ExcelWriter(output_path) as writer:
    for pert, df in results.items():
        sheet_name = pert[:31]  # Excel sheet name limit
        df.to_excel(writer, sheet_name=sheet_name)

import pickle
with open(os.path.join(out_dir, "DESeq2_iNeurons_results_v3_all.pkl"), "wb") as f:
    pickle.dump(results, f)