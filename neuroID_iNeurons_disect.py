from celldisect import CellDISECT
import scvi
scvi.settings.seed = 42 # Setting a random seed for reproducibility
import scanpy as sc
from anndata import AnnData
import os
import shutil
import gc
import anndata as ad
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# adata
adata_path = '/lustre/scratch126/cellgen/lotfollahi/so21/neuroID/preprocessing/adatas/counts_filtered_correctGuides.h5ad'

# categorical covariates to disentangle
cats = ['Replicate', 'perturbation']

# Set up the directory for saving models
model_name = 'iNeurons'
pre_path = f'/lustre/scratch126/cellgen/lotfollahi/so21/neuroID/cellDisect/{model_name}_v3'
if not os.path.exists(pre_path):
    os.makedirs(pre_path)

# adata.X needs to be raw counts
adata = sc.read(adata_path)
print (adata.X.shape)
adata = adata[~adata.obs.target_gene.isin(['INT3','INT30'])].copy()
print (adata.X.shape)

adata.layers['counts'] = adata.X #set layer for later retrieval
print ('object:', adata.X.shape, adata.X.min(), adata.X.max())

for item in cats:
    print (adata.obs[item].value_counts())

# Normalizing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

sc.pl.umap(
    adata,
    color=['Replicate', 'perturbation'],
    wspace=0.2,
    save='_1_inital.pdf'
    )

adata.X = adata.layers['counts'].copy()
adata = adata[adata.X.sum(1) != 0].copy()
adata.X.min(), adata.X.max()

# model configuration

cell_type_included = False # Set to True if you have provided a cell type annotation in the cats list

# Architecture parameters
arch_dict = {'n_layers': 2,               # Number of hidden layers in encoder/decoder networks
 'n_hidden': 128,                         # Number of nodes per hidden layer
 'n_latent_shared': 32,                   # Dimensionality of the shared latent space (Z_0)
 'n_latent_attribute': 32,                # Dimensionality of each attribute-specific latent space (Z_i)
 'dropout_rate': 0.1,                     # Dropout rate for regularization
 'weighted_classifier': True,             # Whether to use weighted classifiers for imbalanced categories
}

# Training parameters
train_dict = {
 'max_epochs': 1000,                      # Maximum number of training epochs
 'batch_size': 256,                       # Number of samples per batch
 'recon_weight': 20,                      # Weight for reconstruction loss
 'cf_weight': 0.8,                        # Weight for counterfactual loss
 'beta': 0.003,                           # Weight for KL divergence (controls latent space regularization)
 'clf_weight': 0.05,                      # Weight for classifier loss
 'adv_clf_weight': 0.014,                 # Weight for adversarial classifier loss
 'adv_period': 5,                         # Period for adversarial training
 'n_cf': 1,                               # Number of counterfactual steps
 'early_stopping_patience': 6,            # Number of epochs to wait before early stopping
 'early_stopping': True,                  # Whether to use early stopping
 'save_best': True,                       # Whether to save the best model
 'kappa_optimizer2': False,               # Whether to use the kappa weight in optimizer 2
 'n_epochs_pretrain_ae': 0,               # Number of epochs for pretraining the autoencoder
}

# Optimizer parameters
plan_kwargs = {
 'lr': 0.003,                             # Learning rate
 'weight_decay': 0.00005,                 # L2 regularization strength
 'ensemble_method_cf': True,              # Whether to use ensemble method for counterfactuals
 'lr_patience': 5,                        # Patience for learning rate scheduler
 'lr_factor': 0.5,                        # Factor by which to reduce learning rate
 'lr_scheduler_metric': 'loss_validation',# Metric to monitor for learning rate scheduling
 'n_epochs_kl_warmup': 10,                # Number of epochs for KL divergence warmup
}

# Create a descriptive model name based on parameters
model_name = (
    f'pretrainAE_{train_dict["n_epochs_pretrain_ae"]}_'
    f'maxEpochs_{train_dict["max_epochs"]}_'
    f'reconW_{train_dict["recon_weight"]}_'
    f'cfWeight_{train_dict["cf_weight"]}_'
    f'beta_{train_dict["beta"]}_'
    f'clf_{train_dict["clf_weight"]}_'
    f'adv_{train_dict["adv_clf_weight"]}_'
    f'advp_{train_dict["adv_period"]}_'
    f'n_cf_{train_dict["n_cf"]}_'
    f'lr_{plan_kwargs["lr"]}_'
    f'wd_{plan_kwargs["weight_decay"]}_'
    f'ensemble_cf_{plan_kwargs["ensemble_method_cf"]}_'
    f'dropout_{arch_dict["dropout_rate"]}_'
    f'n_hidden_{arch_dict["n_hidden"]}_'
    f'n_latent_{arch_dict["n_latent_shared"]}_'
    f'n_layers_{arch_dict["n_layers"]}_'
    f'batch_size_{train_dict["batch_size"]}_'
    f'weighted_classifier_{arch_dict["weighted_classifier"]}_'
)
if cell_type_included:
    model_name = model_name + f'cellTypeIncluded'
else:
    model_name = model_name + f'cellTypeNotIncluded'

# Clean up existing model directory if it exists (caution: this will delete existing models)
try:
    shutil.rmtree(f"{pre_path}/{model_name}")
    print("Directory deleted successfully")
except OSError as e:
    print(f"Error deleting directory: {e}")

### anndata set up
CellDISECT.setup_anndata(
    adata,
    layer='counts',                                  # Layer containing raw count data
    categorical_covariate_keys=cats,                 # Categorical covariates to disentangle
    continuous_covariate_keys=[],                    # Continuous covariates (none in this example)
    add_cluster_covariate=not cell_type_included,    # Add cluster info if cell type is not included
)

### random splits
model = CellDISECT(adata, **arch_dict)

# Train the model with our training parameters
model.train(**train_dict, plan_kwargs=plan_kwargs, )

# Save the trained model
model.save(f"{pre_path}/{'iNeurons_v3_model'}", overwrite=True)
print(model_name)

# Get the latent representations
print(f"Getting the latent 0...")
# Z_0: Shared latent space (nullify all categorical covariates)
adata.obsm[f"CellDISECT_Z_0"] = model.get_latent_representation(
    nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False
)

for i in range(len(cats)):
    print(f"Getting the latent {i+1} / {len(cats)}...")
    null_idx = [s for s in range(len(cats)) if s != i]
    label = cats[i]
    # Z_i: Latent space specific to covariate i (nullify all other covariates and shared space)
    adata.obsm[f"CellDISECT_Z_{label}"] = model.get_latent_representation(
        nullify_cat_covs_indices=null_idx, nullify_shared=True
    )
    # Z_not_i: Latent space containing all information except covariate i
    adata.obsm[f"CellDISECT_Z_not_{label}"] = model.get_latent_representation(
        nullify_cat_covs_indices=[i], nullify_shared=False
    )

# Compute neighbors and UMAPs for the latent representations (this might take a while, consider running it using RAPIDS scanpy with a GPU if data is large)
for i in range(len(cats) + 1):  # loop over all Z_i | Neighbors and UMAPs for Z_i
    if i == 0:
        latent_name = f"CellDISECT_Z_{i}"
    else:
        label = cats[i - 1]
        latent_name = f"CellDISECT_Z_{label}"

    # Create a temporary AnnData object with the latent representation as the main matrix
    latent = ad.AnnData(X=adata.obsm[f"{latent_name}"], obs=adata.obs)
    sc.pp.neighbors(adata=latent, use_rep="X")
    sc.tl.umap(adata=latent)

    # Store the neighborhood graph and UMAP coordinates in the original AnnData object
    adata.uns[f"{latent_name}_neighbors"] = latent.uns["neighbors"]
    adata.obsm[f"{latent_name}_umap"] = latent.obsm["X_umap"]
    gc.collect()  # Free up memory

adata.write(f'iNeurons_cellDisect_v3.h5ad')
