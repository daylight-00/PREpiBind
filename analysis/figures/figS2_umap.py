import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rawpath as rp
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

blosum62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, -1, -1, -4],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, -2, 0, -1, -4],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 4, -3, 0, -1, -4],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, -3, 1, -1, -4],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -1, -3, -1, -4],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, -2, 4, -1, -4],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, -3, 4, -1, -4],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -4, -2, -1, -4],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, -3, 0, -1, -4],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, 3, -3, -1, -4],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, 3, -3, -1, -4],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, -3, 1, -1, -4],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, 2, -1, -1, -4],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, 0, -3, -1, -4],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -3, -1, -1, -4],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, -2, 0, -1, -4],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, -1, -1, -4],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -2, -2, -1, -4],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -1, -2, -1, -4],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, 2, -2, -1, -4],
    'B': [-2, -1, 4, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, -3, 0, -1, -4],
    'J': [-2, -3, -3, -1, -2, -3, -4, -3, 3, 3, -3, 2, 0, -3, -2, -1, -2, -1, 2, -3, 3, -3, -1, -4],
    'Z': [-1, 0, 0, 1, -3, 4, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -2, -2, -2, 0, -3, 4, -1, -4],
    'X': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -4],
    '*': [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1]
}

def get_blosum_emb(matrix, sequence, start_idx=None, end_idx=None):
    embedding = [matrix[aa] for aa in sequence]
    embedding = np.array(embedding, dtype=np.float32)
    if start_idx is not None and end_idx is not None:
        embedding = embedding[start_idx:end_idx]
    return embedding

def load_and_process_data(data_source, template_df, template_dict, keys):
    dpa_data = []
    dpb_data = []
    dqa_data = []
    dqb_data = []
    dra_data = []
    drb_data = []
    h2aa_data = []
    h2ab_data = []
    h2ea_data = []
    h2eb_data = []

    if data_source == 'blosum':
        for key in keys:
            sequence = template_dict[key]['HLA_Seq']
            start_idx = int(template_dict[key]['start_idx'])
            end_idx = int(template_dict[key]['end_idx'])
            embedding_data = get_blosum_emb(blosum62, sequence, start_idx, end_idx)
            classify_and_append(key, embedding_data, dpa_data, dpb_data, dqa_data, dqb_data, dra_data, drb_data, h2aa_data, h2ab_data, h2ea_data, h2eb_data)
            
    elif data_source == 'chai':
        patha = rp.external('EMB/emb_hla_chai_jack_single_0430.h5')
        pathb = rp.external('EMB/emb_hla_chai_jack_pair_side_0430.h5')
        with h5py.File(patha, 'r') as fa, h5py.File(pathb, 'r') as fb:
            for key in keys:
                embedding_data_a = fa[key][:]
                embedding_data_b = fb[key][:]
                start_idx = int(template_dict[key]['start_idx'])
                end_idx = int(template_dict[key]['end_idx'])
                embedding_data_a = embedding_data_a[start_idx:end_idx]
                embedding_data_b = embedding_data_b[start_idx:end_idx]
                try:
                    embedding_data = np.concatenate((embedding_data_a, embedding_data_b), axis=-1)
                    classify_and_append(key, embedding_data, dpa_data, dpb_data, dqa_data, dqb_data, dra_data, drb_data, h2aa_data, h2ab_data, h2ea_data, h2eb_data)
                except ValueError:
                    continue
                    
    elif data_source == 'esmc':
        path = rp.external('250601/2_full_hla_esm/emb_hla_esmc_small_0601.h5')
        with h5py.File(path, 'r') as f:
            for key in keys:
                embedding_data = f[key][:]
                start_idx = int(template_dict[key]['start_idx'])
                end_idx = int(template_dict[key]['end_idx'])
                embedding_data = embedding_data[start_idx:end_idx]
                classify_and_append(key, embedding_data, dpa_data, dpb_data, dqa_data, dqb_data, dra_data, drb_data, h2aa_data, h2ab_data, h2ea_data, h2eb_data)
                
    elif data_source == 'esm3':
        path = rp.external('EMB/esm_large/emb_hla_esm3_small_2408_0430.h5')
        with h5py.File(path, 'r') as f:
            for key in keys:
                embedding_data = f[key][:]
                start_idx = int(template_dict[key]['start_idx'])
                end_idx = int(template_dict[key]['end_idx'])
                embedding_data = embedding_data[start_idx:end_idx]
                classify_and_append(key, embedding_data, dpa_data, dpb_data, dqa_data, dqb_data, dra_data, drb_data, h2aa_data, h2ab_data, h2ea_data, h2eb_data)

    return process_to_mean_pooled(dpa_data, dpb_data, dqa_data, dqb_data, dra_data, drb_data, h2aa_data, h2ab_data, h2ea_data, h2eb_data)

def classify_and_append(key, embedding_data, dpa_data, dpb_data, dqa_data, dqb_data, dra_data, drb_data, h2aa_data, h2ab_data, h2ea_data, h2eb_data):
    if 'HLA-DPA' in key:
        dpa_data.append(embedding_data)
    elif 'HLA-DPB' in key:
        dpb_data.append(embedding_data)
    elif 'HLA-DQA' in key:
        dqa_data.append(embedding_data)
    elif 'HLA-DQB' in key:
        dqb_data.append(embedding_data)
    elif 'HLA-DRA' in key:
        dra_data.append(embedding_data)
    elif 'HLA-DRB' in key:
        drb_data.append(embedding_data)
    elif 'H2-IA' in key and key.endswith('A') and 'H2-IAd' not in key:
        h2aa_data.append(embedding_data)
    elif 'H2-IA' in key and key.endswith('B') and 'H2-IAd' not in key:
        h2ab_data.append(embedding_data)
    elif 'H2-IAdA' in key:
        h2ab_data.append(embedding_data)
    elif 'H2-IAdB' in key:
        h2aa_data.append(embedding_data)
    elif 'H2-IE' in key and key.endswith('A'):
        h2ea_data.append(embedding_data)
    elif 'H2-IE' in key and key.endswith('B'):
        h2eb_data.append(embedding_data)

def process_to_mean_pooled(dpa_data, dpb_data, dqa_data, dqb_data, dra_data, drb_data, h2aa_data, h2ab_data, h2ea_data, h2eb_data):
    dpa_data_mean_pooled = np.array([np.mean(dp, axis=0) for dp in dpa_data]) if dpa_data else np.empty((0, 25))
    dpb_data_mean_pooled = np.array([np.mean(dp, axis=0) for dp in dpb_data]) if dpb_data else np.empty((0, 25))
    dqa_data_mean_pooled = np.array([np.mean(dq, axis=0) for dq in dqa_data]) if dqa_data else np.empty((0, 25))
    dqb_data_mean_pooled = np.array([np.mean(dq, axis=0) for dq in dqb_data]) if dqb_data else np.empty((0, 25))
    dra_data_mean_pooled = np.array([np.mean(dr, axis=0) for dr in dra_data]) if dra_data else np.empty((0, 25))
    drb_data_mean_pooled = np.array([np.mean(dr, axis=0) for dr in drb_data]) if drb_data else np.empty((0, 25))
    h2aa_data_mean_pooled = np.array([np.mean(h2, axis=0) for h2 in h2aa_data]) if h2aa_data else np.empty((0, 25))
    h2ab_data_mean_pooled = np.array([np.mean(h2, axis=0) for h2 in h2ab_data]) if h2ab_data else np.empty((0, 25))
    h2ea_data_mean_pooled = np.array([np.mean(h2, axis=0) for h2 in h2ea_data]) if h2ea_data else np.empty((0, 25))
    h2eb_data_mean_pooled = np.array([np.mean(h2, axis=0) for h2 in h2eb_data]) if h2eb_data else np.empty((0, 25))
    
    return dpa_data_mean_pooled, dpb_data_mean_pooled, dqa_data_mean_pooled, dqb_data_mean_pooled, dra_data_mean_pooled, drb_data_mean_pooled, h2aa_data_mean_pooled, h2ab_data_mean_pooled, h2ea_data_mean_pooled, h2eb_data_mean_pooled

def run_umap_and_plot(all_data, title, ax):
    # Imported here, not at the top: with the cache below populated this file draws the figure on
    # a CPU-only machine. Only a recompute needs RAPIDS and a GPU.
    import cupy as cp
    from cuml.manifold import UMAP

    all_data_gpu = cp.asarray(all_data)
    
    min_dist = 0.3
    umap_model_gpu = UMAP(n_components=2, n_neighbors=16, min_dist=min_dist, random_state=42, build_algo='brute_force_knn', metric='cosine')
    umap_result_gpu = umap_model_gpu.fit_transform(all_data_gpu)
    
    umap_result_gpu_cpu = cp.asnumpy(umap_result_gpu)
    
    return umap_result_gpu_cpu

df_train = pd.read_csv(rp.data('dataset/full/train.csv'))
df_test = pd.read_csv(rp.data('dataset/full/test.csv'))
df_full = pd.concat([df_train, df_test], ignore_index=True)
mhc_alpha = df_full['HLA_Name_A'].sort_values().unique().tolist()
mhc_beta = df_full['HLA_Name_B'].sort_values().unique().tolist()
mhc_list = mhc_alpha + mhc_beta

template_df = pd.read_csv(rp.at('250714/2_umap_test/HLA2_IMGT_MSA_idx.csv'))
template_df = template_df[template_df['HLA_Name'].isin(mhc_list)]
template_dict = template_df.set_index('HLA_Name').T.to_dict()
keys = template_df['HLA_Name'].tolist()

serotypes = ['HLA-DPA', 'HLA-DPB', 'HLA-DQA', 'HLA-DQB', 'HLA-DRA', 'HLA-DRB', 'H2-IA (α)', 'H2-IA (β)', 'H2-IE (α)', 'H2-IE (β)']
colors  = ['#000000', '#000000',   # HLA-DP
           '#E69F00', '#E69F00',   # HLA-DQ
           '#56B4E9', '#56B4E9',   # HLA-DR
           '#009E73', '#009E73',   # H2-IA
           '#CC79A7', '#CC79A7']   # H2-IE
markers = ['o', '^', 'o', '^', 'o', '^', 'o', '^', 'o', '^']

# The four cached embeddings are tracked in this repository, so the figure redraws without the
# embedding stores. Delete umap_cache/ (or set USE_CACHE=False) to recompute; that needs a GPU.
USE_CACHE = True
CACHE_DIR = 'umap_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

methods = ['blosum', 'chai', 'esmc', 'esm3']
titles = ['BLOSUM62', 'Chai-1', 'ESM C 300M', 'ESM3 Small']

for idx, (method, title) in enumerate(zip(methods, titles)):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    cache_path = os.path.join(CACHE_DIR, f'{method}_umap.npz')

    if USE_CACHE and os.path.exists(cache_path):
        print(f"Loading cached data for {method}...")
        cache = np.load(cache_path)
        umap_result = cache['umap_result']
        sizes = cache['sizes']
    else:
        print(f"Processing {method}...")

        data_arrays = load_and_process_data(method, template_df, template_dict, keys)
        all_data = np.vstack([arr for arr in data_arrays if arr.size > 0])

        umap_result = run_umap_and_plot(all_data, title, ax)
        sizes = np.array([data_arrays[i].shape[0] for i in range(len(serotypes))])

        np.savez(cache_path, umap_result=umap_result, sizes=sizes)
        print(f"  -> Saved cache to {cache_path}")
    
    start_idx = 0
    for i, serotype in enumerate(serotypes):
        n = int(sizes[i])
        if n > 0:
            ax.scatter(
                umap_result[start_idx:start_idx+n, 0],
                umap_result[start_idx:start_idx+n, 1],
                c=colors[i], marker=markers[i], label=serotype, alpha=0.7, s=30, edgecolors='none',
            )
            start_idx += n
    
    ax.set_title(f'{title}')
    ax.set_xlabel('UMAP 1') if row == 1 and col == 0 else ''
    ax.set_ylabel('UMAP 2') if row == 0 and col == 0 else ''
    
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, title=None, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig('figS2.pdf', bbox_inches='tight')
plt.savefig('figS2.svg', bbox_inches='tight')
plt.show()
