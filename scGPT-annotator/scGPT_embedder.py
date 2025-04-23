import scipy.sparse
import numpy as np
import logging

class scGPT_embedder:
    def embed_data(self, adata):
        """
        Add preprocessing check to handle empty arrays during embedding
        """
        # Check for empty or zero-expression cells
        row_sums = adata.X.sum(axis=1)
        if scipy.sparse.issparse(row_sums):
            row_sums = row_sums.A1
        
        # Filter out cells with zero expression
        if (row_sums == 0).any():
            zero_expr_cells = np.where(row_sums == 0)[0]
            logging.warning(f"Found {len(zero_expr_cells)} cells with zero expression - removing these cells")
            adata = adata[~np.isin(np.arange(adata.n_obs), zero_expr_cells), :]
        
        # Also check for empty genes
        col_sums = adata.X.sum(axis=0)
        if scipy.sparse.issparse(col_sums):
            col_sums = col_sums.A1
        
        if (col_sums == 0).any():
            zero_expr_genes = np.where(col_sums == 0)[0]
            logging.warning(f"Found {len(zero_expr_genes)} genes with zero expression - removing these genes")
            adata = adata[:, ~np.isin(np.arange(adata.n_vars), zero_expr_genes)]
        
        # Proceed with embedding as before 