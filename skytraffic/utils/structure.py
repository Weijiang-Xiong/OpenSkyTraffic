import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sp
from typing import Dict, Tuple


def norm_coords_and_grid_xy(node_coordinates: np.ndarray, grid_size: float = 220.0) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize coordinates and grid indices into a common grid scale.

    Converts real-world coordinates to grid-aligned coordinates in units of
    `grid_size`, with the minimum grid cell shifted to the origin. The returned
    `coords` are centered within grid cells (cell centers at integer grid coords),
    and `grid_xy` is the zero-based grid index for each node.

    Args:
        node_coordinates: Array of shape (N, 2) with real-world XY coordinates.
        grid_size: Size of each grid cell in the same units as `node_coordinates`.

    Returns:
        Tuple of (coords, grid_xy):
            coords: Normalized coordinates in grid units, shape (N, 2).
            grid_xy: the (x, y) grid indices of each node, shape (N, 2).
    """
    coords = np.asarray(node_coordinates, dtype=np.float32)

    grid_xy = coords // grid_size  # assuming each grid cell is grid_size x grid_size meters
    min_grid_x, min_grid_y = grid_xy.min(axis=0)

    # offset the coordinates and grid positions
    coords = (coords - np.array([(min_grid_x + 0.5) * grid_size, (min_grid_y + 0.5) * grid_size])) / grid_size
    grid_xy = grid_xy - np.array([min_grid_x, min_grid_y])

    return coords, grid_xy


def tuple_keys_to_str(d: Dict[Tuple[int, int], int]) -> Dict[str, int]:
    return {f"{k[0]}_{k[1]}": v for k, v in d.items()}


def build_grid_xy_to_id(grid_xy: np.ndarray, grid_id: np.ndarray) -> Dict[Tuple[int, int], int]:
    grid_xy_to_id_np = np.unique(np.concatenate([grid_xy, grid_id[:, None]], axis=1), axis=0)
    return {(int(x), int(y)): int(gid) for x, y, gid in grid_xy_to_id_np}


def build_grid_index(
    node_coordinates: np.ndarray, grid_size: float
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    grid_xy = np.floor_divide(node_coordinates, grid_size).astype(int)
    grid_xy = grid_xy - grid_xy.min(axis=0, keepdims=True)
    grid_width = int(grid_xy[:, 0].max() + 1)
    grid_id = grid_xy[:, 1] * grid_width + grid_xy[:, 0]
    grid_xy_to_id = build_grid_xy_to_id(grid_xy, grid_id)
    return grid_xy, grid_id, grid_xy_to_id


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()
