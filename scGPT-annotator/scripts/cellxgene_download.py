import pathlib
from pathlib import Path


from pathlib import Path

data_dir = Path(".")

def download_cellxgene_v2(output_dir=None, data_dir=None):
    import boto3
    from botocore.config import Config

    bucket_name = "cdiam-h5ad-database"
    prefix = "cellxgene_v2/"
    if output_dir is None:
        output_dir = data_dir / "cellxgene_v2"
    else:
        output_dir = Path(output_dir)

    # fetch two example files
    s3_config = Config(
        retries=dict(max_attempts=10),
        max_pool_connections=50
    )
    s3_client = boto3.client('s3', config=s3_config)

    # List first two objects directly
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        MaxKeys=2
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the files
    for obj in response.get('Contents', []):
        file_name = obj['Key'].split('/')[-1]
        output_path = output_dir / file_name
        s3_client.download_file(bucket_name, obj['Key'], str(output_path))
    
    import os 
    print (os.getcwd())

    import os
    import h5py

    file_paths = [
        output_dir / item['Key'].split('/')[-1]
        for item in response['Contents']
    ]
    return file_paths





def main():
    import h5py
    file_paths = download_cellxgene_v2()
    for file_path in file_paths:
        print()
        print("-" * 5 + f" {file_path.name} " + "-" * 5)
        with h5py.File(file_path, 'r') as f:
                # Print the main groups in the HDF5 file
                print("\nMain HDF5 groups:", f.keys())
                
                # Print the type of the expression matrix storage
                print("\nExpression matrix (X) storage type:", type(f['X']))
                
                # Print the components of the sparse matrix storage
                print("\nSparse matrix components:", f['X'].keys())
                
                # Print the chunk size used for data storage
                print("\nChunk size for data storage:", f['X/data'].chunks)
                
                # Check what additional data matrices are stored
                print("\nStored embeddings (e.g. PCA, UMAP):", 
                    f['obsm'].keys() if 'obsm' in f else "No embeddings found")
                print("Pairwise relationships (e.g. distances, connectivities):", 
                    f['obsp'].keys() if 'obsp' in f else "No pairwise relationships found")
                print("Additional expression matrices (e.g. normalized, raw):", 
                    f['layers'].keys() if 'layers' in f else "No additional layers found")
    from utils import AnnDataChunker
    with AnnDataChunker(file_paths[0], obs_columns=None) as chunker:
        adata = chunker.load_subset(start_row=0, n_rows=100)
        print (adata)
        print (adata.obs)
        print (adata.var)
        print (adata.obsm)
        print (adata.obsp)
        print (adata.layers)

if __name__ == "__main__":
    main()





