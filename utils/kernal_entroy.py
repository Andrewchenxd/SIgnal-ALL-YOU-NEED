import numpy as np
import os
from DiffusionEMD.diffusion_emd import estimate_dos
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
import random
import warnings
warnings.filterwarnings("ignore")


def compute_diffusion_matrix(X: np.array, sigma: float = 10.0):
    '''
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input X returns a diffusion matrix P, as an numpy ndarray.
    Using the "anisotropic" kernel
    Inputs:
        X: a numpy array of size n x d
        sigma: a float
            conceptually, the neighborhood size of Gaussian kernel.
    Returns:
        K: a numpy array of size n x n that has the same eigenvalues as the diffusion matrix.
    '''

    # Construct the distance matrix.
    D = pairwise_distances(X)

    # Gaussian kernel
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))

    # Anisotropic density normalization.
    Deg = np.diag(1 / np.sum(G, axis=1)**0.5)
    K = Deg @ G @ Deg

    # Now K has the exact same eigenvalues as the diffusion matrix `P`
    # which is defined as `P = D^{-1} K`, with `D = np.diag(np.sum(K, axis=1))`.

    return K

def approx_eigvals(A: np.array, filter_thr: float = 1e-3):
    '''
    Estimate the eigenvalues of a matrix `A` using
    Chebyshev approximation of the eigenspectrum.

    Assuming the eigenvalues of `A` are within [-1, 1].

    There is no guarantee the set of eigenvalues are accurate.
    '''

    matrix = A.copy()
    N = matrix.shape[0]

    if filter_thr is not None:
        matrix[np.abs(matrix) < filter_thr] = 0

    # Chebyshev approximation of eigenspectrum.
    eigs, cdf = estimate_dos(matrix)

    # CDF to PDF conversion.
    pdf = np.zeros_like(cdf)
    for i in range(len(cdf) - 1):
        pdf[i] = cdf[i + 1] - cdf[i]

    # Estimate the set of eigenvalues.
    counts = N * pdf / np.sum(pdf)
    eigenvalues = []
    for i, count in enumerate(counts):
        if np.round(count) > 0:
            eigenvalues += [eigs[i]] * int(np.round(count))

    eigenvalues = np.array(eigenvalues)

    return eigenvalues


def exact_eigvals(A: np.array):
    '''
    Compute the exact eigenvalues.
    '''
    if np.allclose(A, A.T, rtol=1e-5, atol=1e-8):
        # Symmetric matrix.
        eigenvalues = np.linalg.eigvalsh(A)
    else:
        eigenvalues = np.linalg.eigvals(A)

    return eigenvalues


def exact_eig(A: np.array):
    '''
    Compute the exact eigenvalues & vecs.
    '''

    #return np.ones(A.shape[0]), np.ones((A.shape[0],A.shape[0]))
    if np.allclose(A, A.T, rtol=1e-5, atol=1e-8):
        # Symmetric matrix.
        eigenvalues_P, eigenvectors_P = np.linalg.eigh(A)
    else:
        eigenvalues_P, eigenvectors_P = np.linalg.eig(A)

    # Sort eigenvalues
    sorted_idx = np.argsort(eigenvalues_P)[::-1]
    eigenvalues_P = eigenvalues_P[sorted_idx]
    eigenvectors_P = eigenvectors_P[:, sorted_idx]

    return eigenvalues_P, eigenvectors_P

def diffusion_spectral_entropy(embedding_vectors: np.array,
                               gaussian_kernel_sigma: float = 10,
                               t: int = 1,
                               chebyshev_approx: bool = False,
                               eigval_save_path: str = None,
                               eigval_save_precision: np.dtype = np.float16,
                               classic_shannon_entropy: bool = False,
                               num_bins_per_dim: int = 2,
                               verbose: bool = False):
    '''
    >>> If `classic_shannon_entropy` is False (default)

    Diffusion Spectral Entropy over a set of N vectors, each of D dimensions.

    DSE = - sum_i [eig_i^t log eig_i^t]
        where each `eig_i` is an eigenvalue of `P`,
        where `P` is the diffusion matrix computed on the data graph of the [N, D] vectors.

    >>> If `classic_shannon_entropy` is True

    Classic Shannon Entropy over a set of N vectors, each of D dimensions.

    CSE = - sum_i [p(x) log p(x)]
        where each p(x) is the probability density of a histogram bin, after some sort of binning.

    args:
        embedding_vectors: np.array of shape [N, D]
            N: number of data points / samples
            D: number of feature dimensions of the neural representation

        gaussian_kernel_sigma: float
            The bandwidth of Gaussian kernel (for computation of the diffusion matrix)
            Can be adjusted per the dataset.
            Increase if the data points are very far away from each other.

        t: int
            Power of diffusion matrix (equivalent to power of diffusion eigenvalues)
            <-> Iteration of diffusion process
            Usually small, e.g., 1 or 2.
            Can be adjusted per dataset.
            Rule of thumb: after powering eigenvalues to `t`, there should be approximately
                           1 percent of eigenvalues that remain larger than 0.01

        chebyshev_approx: bool
            Whether or not to use Chebyshev moments for faster approximation of eigenvalues.
            Currently we DO NOT RECOMMEND USING THIS. Eigenvalues may be changed quite a bit.

        eigval_save_path: str
            If provided,
                (1) If running for the first time, will save the computed eigenvalues in this location.
                (2) Otherwise, if the file already exists, skip eigenvalue computation and load from this file.

        eigval_save_precision: np.dtype
            We use `np.float16` by default to reduce storage space required.
            For best precision, use `np.float64` instead.

        classic_shannon_entropy: bool
            Toggle between DSE and CSE. False (default) == DSE.

        num_bins_per_dim: int
            Number of bins per feature dim.
            Only relevant to CSE (i.e., `classic_shannon_entropy` is True).

        verbose: bool
            Whether or not to print progress to console.
    '''

    if not classic_shannon_entropy:
        # Computing Diffusion Spectral Entropy.
        if verbose: print('Computing Diffusion Spectral Entropy...')

        if eigval_save_path is not None and os.path.exists(eigval_save_path):
            if verbose:
                print('Loading pre-computed eigenvalues from %s' %
                      eigval_save_path)
            eigvals = np.load(eigval_save_path)['eigvals']
            eigvals = eigvals.astype(np.float64)  # mitigate rounding error.
            if verbose: print('Pre-computed eigenvalues loaded.')

        else:
            if verbose: print('Computing diffusion matrix.')
            # Note that `K` is a symmetric matrix with the same eigenvalues as the diffusion matrix `P`.
            K = compute_diffusion_matrix(embedding_vectors,
                                         sigma=gaussian_kernel_sigma)
            if verbose: print('Diffusion matrix computed.')

            if verbose: print('Computing eigenvalues.')
            if chebyshev_approx:
                if verbose: print('Using Chebyshev approximation.')
                eigvals = approx_eigvals(K)
            else:
                eigvals = exact_eigvals(K)
            if verbose: print('Eigenvalues computed.')

            if eigval_save_path is not None:
                os.makedirs(os.path.dirname(eigval_save_path), exist_ok=True)
                # Save eigenvalues.
                eigvals = eigvals.astype(
                    eigval_save_precision)  # reduce storage space.
                with open(eigval_save_path, 'wb+') as f:
                    np.savez(f, eigvals=eigvals)
                if verbose: print('Eigenvalues saved to %s' % eigval_save_path)

        # Eigenvalues may be negative. Only care about the magnitude, not the sign.
        eigvals = np.abs(eigvals)

        # Power eigenvalues to `t` to mitigate effect of noise.
        eigvals = eigvals**t

        prob = eigvals / eigvals.sum()

    else:
        # Computing Classic Shannon Entropy.
        if verbose: print('Computing Classic Shannon Entropy...')

        vecs = embedding_vectors.copy()

        # Min-Max scale each dimension.
        vecs = (vecs - np.min(vecs, axis=0)) / (np.max(vecs, axis=0) -
                                                np.min(vecs, axis=0))

        # Bin along each dimension.
        bins = np.linspace(0, 1, num_bins_per_dim + 1)[:-1]
        vecs = np.digitize(vecs, bins=bins)

        # Count probability.
        counts = np.unique(vecs, axis=0, return_counts=True)[1]
        prob = counts / np.sum(counts)

    prob = prob + np.finfo(float).eps
    entropy = -np.sum(prob * np.log2(prob))

    return entropy


def diffusion_spectral_mutual_information(
        embedding_vectors: np.array,
        reference_vectors: np.array,
        reference_discrete: bool = None,
        gaussian_kernel_sigma: float = 10,
        t: int = 1,
        chebyshev_approx: bool = False,
        num_repetitions: int = 5,
        n_clusters: int = 10,
        precomputed_clusters: np.array = None,
        classic_shannon_entropy: bool = False,
        num_bins_per_dim: int = 2,
        random_seed: int = 0,
        verbose: bool = False):
    '''
    DSMI between two sets of random variables.
    The first (`embedding_vectors`) must be a set of N vectors each of D dimension.
    The second (`reference_vectors`) must be a set of N vectors each of D' dimension.
        D is not necessarily the same as D'.
        In some common cases, we may have the following as `reference_vectors`
            - class labels (D' == 1) of shape [N, 1]
            - flattened input signals/images of shape [N, D']

    DSMI(A; B) = DSE(A) - DSE(A | B)
        where DSE is the diffusion spectral entropy.

    DSE(A | B) = sum_i [p(B = b_i) DSE(A | B = b_i)]
        where i = 0,1,...,m
            m = number of categories in random variable B
            if B itself is a discrete variable (e.g., class label), this is straightforward
            otherwise, we can use spectral clustering to create discrete categories/clusters in B

    For numerical consistency, instead of computing DSE(A) on all data points of A,
    we estimate it from a subset of A, with the size of subset equal to {B = b_i}.

    The final computation is:

    DSMI(A; B) = DSE(A) - DSE(A | B) = sum_i [p(B = b_i) (DSE(A*) - DSE(A | B = b_i))]
        where A* is a subsampled version of A, with len(A*) == len(B = b_i).

    args:
        embedding_vectors: np.array of shape [N, D]
            N: number of data points / samples
            D: number of feature dimensions of the neural representation

        reference_vectors: np.array of shape [N, D']
            N: number of data points / samples
            D': number of feature dimensions of the neural representation or input/output variable

        reference_discrete: bool
            Whether `reference_vectors` is discrete or continuous.
            This determines whether or not we perform clustering/binning on `reference_vectors`.
            NOTE: If True, we assume D' == 1. Common case: `reference_vectors` is the discrete class labels.
            If not provided, will be inferred from `reference_vectors`.

        gaussian_kernel_sigma: float
            The bandwidth of Gaussian kernel (for computation of the diffusion matrix)
            Can be adjusted per the dataset.
            Increase if the data points are very far away from each other.

        t: int
            Power of diffusion matrix (equivalent to power of diffusion eigenvalues)
            <-> Iteration of diffusion process
            Usually small, e.g., 1 or 2.
            Can be adjusted per dataset.
            Rule of thumb: after powering eigenvalues to `t`, there should be approximately
                           1 percent of eigenvalues that remain larger than 0.01

        chebyshev_approx: bool
            Whether or not to use Chebyshev moments for faster approximation of eigenvalues.
            Currently we DO NOT RECOMMEND USING THIS. Eigenvalues may be changed quite a bit.

        num_repetitions: int
            Number of repetition during DSE(A*) estimation.
            The variance is usually low, so a small number shall suffice.

        random_seed: int
            Random seed. For DSE(A*) estimation repeatability.

        n_clusters: int
            Number of clusters for `reference_vectors`.
            Only used when `reference_discrete` is False (`reference_vectors` is not discrete).
            If D' == 1 --> will use scalar binning.
            If D' > 1  --> will use spectral clustering.

        precomputed_clusters: np.array
            If provided, will directly use it as the cluster assignments for `reference_vectors`.
            Only used when `reference_discrete` is False (`reference_vectors` is not discrete).
            NOTE: When you have a fixed set of `reference_vectors` (e.g., a set of images),
            you probably want to only compute the spectral clustering once, and recycle the computed
            clusters for subsequent DSMI computations.

        classic_shannon_entropy: bool
            Whether or not we use CSE to replace DSE in the computation.
            NOTE: If true, the resulting mutual information will be CSMI instead of DSMI.

        num_bins_per_dim: int
            Number of bins per feature dim.
            Only relevant to CSE (i.e., `classic_shannon_entropy` is True).

        verbose: bool
            Whether or not to print progress to console.
    '''

    # Reshape from [N, ] to [N, 1].
    if len(reference_vectors.shape) == 1:
        reference_vectors = reference_vectors.reshape(
            reference_vectors.shape[0], 1)

    N_embedding, _ = embedding_vectors.shape
    N_reference, D_reference = reference_vectors.shape

    if N_embedding != N_reference:
        if verbose:
            print(
                'WARNING: DSMI embedding and reference do not have the same N: %s vs %s'
                % (N_embedding, N_reference))

    if reference_discrete is None:
        # Infer whether `reference_vectors` is discrete.
        # Criteria: D' == 1 and `reference_vectors` is an integer type.
        reference_discrete = D_reference == 1 \
            and np.issubdtype(
            reference_vectors.dtype, np.integer)

    #
    '''STEP 1. Prepare the category/cluster assignments.'''

    if reference_discrete:
        # `reference_vectors` is expected to be discrete class labels.
        assert D_reference == 1, \
            'DSMI `reference_discrete` is set to True, but shape of `reference_vectors` is not [N, 1].'
        precomputed_clusters = reference_vectors

    elif D_reference == 1:
        # `reference_vectors` is a set of continuous scalars.
        # Perform scalar binning if cluster assignments are not provided.
        if precomputed_clusters is None:
            vecs = reference_vectors.copy()
            # Min-Max scale each dimension.
            vecs = (vecs - np.min(vecs, axis=0)) / (np.max(vecs, axis=0) -
                                                    np.min(vecs, axis=0))
            # Bin along each dimension.
            bins = np.linspace(0, 1, n_clusters + 1)[:-1]
            vecs = np.digitize(vecs, bins=bins)
            precomputed_clusters = vecs

    else:
        # `reference_vectors` is a set of continuous vectors.
        # Perform spectral clustering if cluster assignments are not provided.
        if precomputed_clusters is None:
            cluster_op = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                assign_labels='cluster_qr',
                random_state=0).fit(reference_vectors)
            precomputed_clusters = cluster_op.labels_

    clusters_list, cluster_cnts = np.unique(precomputed_clusters,
                                            return_counts=True)

    #
    '''STEP 2. Compute DSMI.'''
    MI_by_class = []

    for cluster_idx in clusters_list:
        # DSE(A | B = b_i)
        inds = (precomputed_clusters == cluster_idx).reshape(-1)
        embeddings_curr_class = embedding_vectors[inds, :]

        entropy_AgivenB_curr_class = diffusion_spectral_entropy(
            embedding_vectors=embeddings_curr_class,
            gaussian_kernel_sigma=gaussian_kernel_sigma,
            t=t,
            chebyshev_approx=chebyshev_approx,
            classic_shannon_entropy=classic_shannon_entropy,
            num_bins_per_dim=num_bins_per_dim)

        # DSE(A*)
        if random_seed is not None:
            random.seed(random_seed)
        entropy_A_estimation_list = []
        for _ in np.arange(num_repetitions):
            rand_inds = np.array(
                random.sample(range(precomputed_clusters.shape[0]),
                              k=np.sum(precomputed_clusters == cluster_idx)))
            embeddings_random_subset = embedding_vectors[rand_inds, :]

            entropy_A_subsample_rep = diffusion_spectral_entropy(
                embedding_vectors=embeddings_random_subset,
                gaussian_kernel_sigma=gaussian_kernel_sigma,
                t=t,
                chebyshev_approx=chebyshev_approx,
                classic_shannon_entropy=classic_shannon_entropy,
                num_bins_per_dim=num_bins_per_dim)
            entropy_A_estimation_list.append(entropy_A_subsample_rep)

        entropy_A_estimation = np.mean(entropy_A_estimation_list)

        MI_by_class.append((entropy_A_estimation - entropy_AgivenB_curr_class))

    mutual_information = np.sum(cluster_cnts / np.sum(cluster_cnts) *
                                np.array(MI_by_class))

    return mutual_information, precomputed_clusters


if __name__ == '__main__':
    choose='DSMI'
    if choose=='DSE':
        print('Testing Diffusion Spectral Entropy.')
        print('\n1st run, random vecs, without saving eigvals.')
        embedding_vectors = np.random.uniform(0, 1, (1000, 256))
        DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors)
        print('DSE =', DSE)

        print('\n2nd run, random vecs, saving eigvals (np.float16).')
        tmp_path = './test_dse_eigval.npz'
        embedding_vectors = np.random.uniform(0, 1, (1000, 256))
        DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                         eigval_save_path=tmp_path)
        print('DSE =', DSE)

        print(
            '\n3rd run, loading eigvals from 2nd run. May be slightly off due to float16 saving.'
        )
        embedding_vectors = None  # does not matter, will be ignored anyways
        DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                         eigval_save_path=tmp_path)
        print('DSE =', DSE)
        os.remove(tmp_path)

        print('\n4th run, random vecs, saving eigvals (np.float64).')
        embedding_vectors = np.random.uniform(0, 1, (1000, 256))
        DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                         eigval_save_path=tmp_path,
                                         eigval_save_precision=np.float64)
        print('DSE =', DSE)

        print('\n5th run, loading eigvals from 4th run. Shall be identical.')
        embedding_vectors = None  # does not matter, will be ignored anyways
        DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                         eigval_save_path=tmp_path)
        print('DSE =', DSE)
        os.remove(tmp_path)

        print('\n6th run, Classic Shannon Entropy.')
        embedding_vectors = np.random.uniform(0, 1, (1000, 256))
        CSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                         classic_shannon_entropy=True)
        print('CSE =', CSE)
    elif choose == 'DSMI':
        print('Testing Diffusion Spectral Mutual Information.')
        print('\n1st run. DSMI, Embeddings vs discrete class labels.')
        embedding_vectors = np.random.uniform(0, 1, (1000, 256))
        class_labels = np.uint8(np.random.uniform(0, 11, (1000, 1)))
        DSMI, _ = diffusion_spectral_mutual_information(
            embedding_vectors=embedding_vectors, reference_vectors=class_labels)
        print('DSMI =', DSMI)

        print('\n2nd run. DSMI, Embeddings vs continuous scalars')
        embedding_vectors = np.random.uniform(0, 1, (1000, 256))
        continuous_scalars = np.random.uniform(-1, 1, (1000, 1))
        DSMI, _ = diffusion_spectral_mutual_information(
            embedding_vectors=embedding_vectors,
            reference_vectors=continuous_scalars)
        print('DSMI =', DSMI)

        print('\n3rd run. DSMI, Embeddings vs Input Image')
        embedding_vectors = np.random.uniform(0, 1, (1000, 256))
        input_image = np.random.uniform(-1, 1, (1000, 3, 32, 32))
        input_image = input_image.reshape(input_image.shape[0], -1)
        DSMI, _ = diffusion_spectral_mutual_information(
            embedding_vectors=embedding_vectors, reference_vectors=input_image)
        print('DSMI =', DSMI)

        print('\n4th run. DSMI, Classification dataset.')
        from sklearn.datasets import make_classification

        embedding_vectors, class_labels = make_classification(n_samples=1000,
                                                              n_features=5)
        DSMI, _ = diffusion_spectral_mutual_information(
            embedding_vectors=embedding_vectors, reference_vectors=class_labels)
        print('DSMI =', DSMI)

        print('\n5th run. CSMI, Classification dataset.')
        embedding_vectors, class_labels = make_classification(n_samples=1000,
                                                              n_features=5)
        CSMI, _ = diffusion_spectral_mutual_information(
            embedding_vectors=embedding_vectors,
            reference_vectors=class_labels,
            classic_shannon_entropy=True)
        print('CSMI =', CSMI)