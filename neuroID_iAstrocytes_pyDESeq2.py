import scanpy as sc
DPI=300
#sc.settings.set_figure_params(scanpy = True, dpi=60, transparent=True, vector_friendly = True, dpi_save=DPI) 
import os 
import pandas as pd
import anndata as ad
import numpy as np
import pertpy as pt

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

adata_counts = sc.read('/nfs/team233/Sam_O/Data/NeuroID/OTAR2071_Screen1_Dec24/iAstrocytes_SCREEN1/iAstrocytes_SCREEN1_filtered.h5ad')
adata_counts

import decoupler as dc

pdata = dc.pp.pseudobulk(adata_counts,
                         sample_col="sequencing_batch", 
                         groups_col="target_gene",
                         layer=None,
                         mode="sum",
                        )
dc.pp.filter_samples(pdata, min_cells=10, min_counts=1000, inplace=True, )
pdata

pds2 = pt.tl.PyDESeq2(adata=pdata, design="~sequencing_batch+target_gene")
pds2

pds2.fit()

out_dir = "DEGs/pyDESeq2"
os.makedirs(out_dir, exist_ok=True)

results = {}
for pert in pdata.obs["target_gene"].cat.categories:
    if pert != "INT":
        contrast = pds2.contrast("target_gene", baseline="INT", group_to_compare=pert)
        results[pert] = pds2.test_contrasts(contrast)
        
output_path = os.path.join(out_dir, "DESeq2_iAstrocytes_results_all.xlsx")
with pd.ExcelWriter(output_path) as writer:
    for pert, df in results.items():
        sheet_name = pert[:31]  # Excel sheet name limit
        df.to_excel(writer, sheet_name=sheet_name)
