from typing import Dict, Optional, Union
import numpy as np
import torch
from scipy.sparse import issparse
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData
import logging

# Setup logger (since we don't have scgpt.logger)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        hvg_subset: bool = True,  # New parameter
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        """
        Set up the preprocessor with modified HVG handling.

        Args:
            use_key: The key of AnnData to use for preprocessing
            filter_gene_by_counts: Whether to filter genes by counts
            filter_cell_by_counts: Whether to filter cells by counts
            normalize_total: Whether to normalize the total counts of each cell
            result_normed_key: Key to store normalized data
            log1p: Whether to apply log1p transform
            result_log1p_key: Key to store log1p transformed data
            subset_hvg: Whether/how many highly variable genes to select
            hvg_use_key: Which layer to use for HVG calculation
            hvg_flavor: Method for HVG selection
            hvg_subset: Whether to subset data to HVGs or just mark them
            binning: Number of bins for discretization
            result_binned_key: Key to store binned data
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.hvg_subset = hvg_subset
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        Process the data according to the initialization parameters.

        Args:
            adata: AnnData object to process
            batch_key: Key for batch information in adata.obs

        Returns:
            Dict with processing information
        """
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None

        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # Step 1: Filter genes
        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # Step 2: Filter cells
        if isinstance(self.filter_cell_by_counts, int) and self.filter_cell_by_counts > 0:
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # Step 3: Normalize total
        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # Step 4: Log1p
        if self.log1p:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)

        # Step 5: HVG selection with modified behavior
        if self.subset_hvg:
            logger.info(
                "Calculating highly variable genes%s...", 
                " and subsetting" if self.hvg_subset else ""
            )
            if batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=self.hvg_subset
            )
            
            # Store HVG information
            if not self.hvg_subset:
                adata.uns['hvg_info'] = {
                    'n_selected': sum(adata.var['highly_variable']),
                    'flavor': self.hvg_flavor,
                    'batch_key': batch_key
                }

        # Step 6: Binning
        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    f"Binning arg must be an integer, but got {self.binning}."
                )
            n_bins = self.binning
            binned_rows = []
            bin_edges = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.A if issparse(layer_data) else layer_data
            
            if layer_data.min() < 0:
                raise ValueError(
                    f"Assuming non-negative data, but got min value {layer_data.min()}."
                )
                
            for row in layer_data:
                if row.max() == 0:
                    logger.warning(
                        "The input data contains all zero rows. Please make sure "
                        "this is expected. You can use the `filter_cell_by_counts` "
                        "arg to filter out all zero rows."
                    )
                    binned_rows.append(np.zeros_like(row, dtype=np.int64))
                    bin_edges.append(np.array([0] * n_bins))
                    continue
                    
                non_zero_ids = row.nonzero()
                non_zero_row = row[non_zero_ids]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                non_zero_digits = _digitize(non_zero_row, bins)
                binned_row = np.zeros_like(row, dtype=np.int64)
                binned_row[non_zero_ids] = non_zero_digits
                binned_rows.append(binned_row)
                bin_edges.append(np.concatenate([[0], bins]))
                
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)

        return {"Status": "Complete"}

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """Check if the data appears to be log-transformed already."""
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """Digitize data into bins with uniform spreading for equal values."""
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))
    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(row: Union[np.ndarray, torch.Tensor], n_bins: int) -> Union[np.ndarray, torch.Tensor]:
    """Bin a row of data into n_bins categories."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row

    if row.max() == 0:
        logger.warning(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)