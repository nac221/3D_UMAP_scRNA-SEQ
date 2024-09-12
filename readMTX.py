import numpy as np
import pandas as pd
from scipy import io
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import umap
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 1. Read the data
def read_mtx_data(mtx_file, genes_file, metadata_file):
    # Read the mtx file
    mat = io.mmread(mtx_file).tocsr()
    
    # Read gene names and cell metadata
    genes = pd.read_csv(genes_file, header=None, sep='\t')[0]
    metadata = pd.read_csv(metadata_file, header=0, sep='\t', index_col=0)
    
    return mat, genes, metadata

# 2. Preprocess the data
def preprocess_data(mat, min_cells=3, min_genes=200):
    # Filter genes
    cells_per_gene = np.array((mat > 0).sum(axis=0)).flatten()
    genes_to_keep = cells_per_gene >= min_cells
    
    # Filter cells
    genes_per_cell = np.array((mat > 0).sum(axis=1)).flatten()
    cells_to_keep = genes_per_cell >= min_genes
    
    # Apply filters
    mat_filtered = mat[cells_to_keep, :][:, genes_to_keep]
    
    # Normalize
    mat_normalized = normalize(mat_filtered, norm='l1', axis=1)
    
    # Log transform
    mat_log = csr_matrix(np.log1p(mat_normalized.toarray()))
    
    return mat_log

# 3. Dimensionality reduction
def reduce_dimensions(mat, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(mat.toarray())
    return reduced_data

# 4. Clustering
def cluster_cells(data, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

# 5. UMAP for 3D visualization
def umap_3d(data):
    umap_3d = umap.UMAP(n_components=3, random_state=42)
    umap_result = umap_3d.fit_transform(data)
    return umap_result

# 6. Visualize clusters in interactive 3D plot
def visualize_clusters_3d_interactive(umap_result, clusters, metadata):
    # Create a DataFrame with UMAP coordinates and cluster information
    df = pd.DataFrame({
        'UMAP1': umap_result[:, 0],
        'UMAP2': umap_result[:, 1],
        'UMAP3': umap_result[:, 2],
        'Cluster': clusters
    })
    
    # Add metadata information
    df = pd.concat([df, metadata], axis=1)
    
    # Create the 3D scatter plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data['UMAP1'],
                y=cluster_data['UMAP2'],
                z=cluster_data['UMAP3'],
                mode='markers',
                marker=dict(size=4),
                name=f'Cluster {cluster}',
                text=[f"Cell: {index}<br>Cluster: {row['Cluster']}<br>" + 
                      "<br>".join([f"{col}: {row[col]}" for col in metadata.columns])
                      for index, row in cluster_data.iterrows()],
                hoverinfo='text'
            )
        )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3'
        ),
        title='Interactive 3D UMAP of Single-cell RNA-seq Data',
        legend_title='Clusters'
    )
    
    fig.show()

# New function to get top expressed genes per cluster
def get_top_genes_per_cluster(mat, genes, clusters, n_top_genes=5):
    # Convert sparse matrix to dense if necessary
    if isinstance(mat, csr_matrix):
        mat = mat.toarray()

    # Find the center cell for each cluster
    cluster_centers = []
    for cluster in np.unique(clusters):
        cluster_points = mat[clusters == cluster]
        center_idx = pairwise_distances_argmin_min(cluster_points.mean(axis=0).reshape(1, -1), cluster_points)[0][0]
        cluster_centers.append(np.where(clusters == cluster)[0][center_idx])

    # Get top expressed genes for each center cell
    top_genes_per_cluster = {}
    for cluster, center_idx in enumerate(cluster_centers):
        cell_expression = mat[center_idx]
        top_gene_indices = cell_expression.argsort()[-n_top_genes:][::-1]
        top_genes = genes[top_gene_indices].tolist()
        top_gene_expressions = cell_expression[top_gene_indices].tolist()
        top_genes_per_cluster[cluster] = list(zip(top_genes, top_gene_expressions))

    return top_genes_per_cluster

# Main function to run the entire pipeline
def main(mtx_file, genes_file, metadata_file):
    # 1. Read data
    print("Reading data...")
    mat, genes, metadata = read_mtx_data(mtx_file, genes_file, metadata_file)
    
    # 2. Preprocess
    print("Preprocessing...")
    mat_processed = preprocess_data(mat)
    
    # 3. Dimensionality reduction
    print("Reducing dimensions...")
    reduced_data = reduce_dimensions(mat_processed)
    
    # 4. Clustering
    print("Clustering...")
    clusters = cluster_cells(reduced_data)
    
    # 5. Get top expressed genes per cluster
    print("Finding top expressed genes per cluster...")
    top_genes = get_top_genes_per_cluster(mat_processed, genes, clusters)

    # Print results
    for cluster, genes in top_genes.items():
        print(f"\nCluster {cluster} top genes:")
        for gene, expression in genes:
            print(f"{gene}: {expression:.2f}")
    
    # 6. UMAP for 3D visualization
    print("Applying UMAP...")
    umap_result = umap_3d(reduced_data)
    
    # 7. Visualize
    print("Visualizing...")
    visualize_clusters_3d_interactive(umap_result, clusters, metadata)

# Example usage
if __name__ == "__main__":
    mtx_file = "count_matrix.mtx"
    genes_file = "all_genes.csv"
    metadata_file = "cell_metadata.csv"
    main(mtx_file, genes_file, metadata_file)