import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import threading
from tqdm import tqdm
from collections import defaultdict
from numba import njit, prange
from scipy import sparse

class TqdmParallel(Parallel):
    def __init__(self, use_tqdm=True, total=0, **kwargs):
        self._use_tqdm = use_tqdm
        self._lock = threading.Lock()
        self._pbar = tqdm(total=total) if use_tqdm else None
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        self._i = 0
        return super().__call__(*args, **kwargs)

    def print_progress(self):
        with self._lock:
            self._i += 1
            if self._pbar:
                self._pbar.update(1)

    def _backend_callback(self, *args, **kwargs):
        self.print_progress()
        return super()._backend_callback(*args, **kwargs)

    def __del__(self):
        if self._pbar:
            self._pbar.close()

# === 数据读取 ===
def load_snapshots(data_dir):
    T_snapshots, V_snapshots, BC = [], [], []
    coords = None
    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(data_dir, fname))
        if coords is None:
            coords = df[["Points:0", "Points:1", "Points:2"]].values
        T_snapshots.append(df['Temperature'].values)
        V_snapshots.append(df[['Velocity:0', 'Velocity:1', 'Velocity:2']].values)
        parts = fname.replace('.csv', '').split('_')
        BC.append([float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])])
    return np.array(coords), np.array(T_snapshots), np.array(V_snapshots), np.array(BC)

def load_BC(data_dir):
    BC = []
    coords = None
    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        parts = fname.replace('.csv', '').split('_')
        BC.append([float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])])
    return np.array(BC)


# === 降采样 ===
def downsample_snapshots_vectorized(coords, T_snapshots, V_snapshots, n_points):
    print(f"开始KMeans聚类: n_points={n_points} ...")
    kmeans = MiniBatchKMeans(n_clusters=n_points, random_state=0, batch_size=10000)
    labels = kmeans.fit_predict(coords)
    coords_ds = kmeans.cluster_centers_

    n_snap, n_total = T_snapshots.shape
    T_ds = np.zeros((n_snap, n_points))
    V_ds = np.zeros((n_snap, n_points, 3))

    print("构造稀疏矩阵表示 label 映射...")
    one_hot = sparse.coo_matrix(
        (np.ones_like(labels), (labels, np.arange(len(labels)))), shape=(n_points, len(labels))
    ).tocsr()

    print("向量化计算快照平均...")
    # 向量化温度聚合：稀疏矩阵乘原数据
    for i in range(n_snap):
        T_ds[i] = one_hot @ T_snapshots[i] / one_hot.sum(axis=1).A.ravel()
        for j in range(3):
            V_ds[i, :, j] = one_hot @ V_snapshots[i, :, j] / one_hot.sum(axis=1).A.ravel()

    print("降采样完成。")
    return coords_ds, T_ds, V_ds

DOWNSAMPLE_CACHE_PATH = "lib/downsampled_50000_snapshots.npz"

def save_downsampled_snapshots(path, coords, T_ds, V_ds):
    np.savez_compressed(path, coords=coords, T=T_ds, V=V_ds)
    print(f"降采样数据已保存至: {path}")

def load_downsampled_snapshots(path):
    if os.path.exists(path):
        data = np.load(path)
        print(f"加载已保存的降采样数据: {path}")
        return data['coords'], data['T'], data['V']
    return None, None, None

# === POD ===
def compute_POD(T_snapshots, r):
    mean_T = np.mean(T_snapshots, axis=0)
    fluctuations = T_snapshots - mean_T
    svd = TruncatedSVD(n_components=r)
    coeffs = svd.fit_transform(fluctuations)
    modes = svd.components_
    print("POD temperature modes computed.")
    return modes, coeffs, mean_T, svd

# === 邻域计算 ===
def compute_neighbor_data(coords, k=7):
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k)
    return idxs

# === 梯度与拉普拉斯 ===
def fast_gradient(phi_modes, coords, neighbors_idx):
    r, N = phi_modes.shape
    gradients = np.zeros((r, N, 3))
    for i in range(N):
        nbr_ids = neighbors_idx[i]
        xi = coords[i]
        X = coords[nbr_ids] - xi
        distances = np.linalg.norm(X, axis=1)
        weights = np.exp(-distances**2 / (np.mean(distances)**2 + 1e-12))
        W = np.diag(weights)
        XT_W = X.T @ W
        H = XT_W @ X
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)
        for j in range(r):
            dphi = phi_modes[j, nbr_ids] - phi_modes[j, i]
            gradients[j, i, :] = H_inv @ XT_W @ dphi
    return gradients

def fast_laplacian(phi_modes, coords, neighbors_idx):
    r, N = phi_modes.shape
    laplacian = np.zeros((r, N))
    for i in range(N):
        nbr_ids = neighbors_idx[i]
        xi = coords[i]
        X = coords[nbr_ids] - xi
        distances = np.linalg.norm(X, axis=1)
        weights = np.exp(-distances**2 / (np.mean(distances)**2 + 1e-12))
        W = weights / np.sum(weights)
        for j in range(r):
            dphi = phi_modes[j, nbr_ids] - phi_modes[j, i]
            laplacian[j, i] = 2 * np.sum(W * dphi) / (np.mean(distances**2) + 1e-12)
    return laplacian

# === Galerkin 系统 ===
def build_galerkin_matrix(phi_modes, velocity_field, coords, neighbors_idx, alpha):
    r, N = phi_modes.shape
    grad_phi = fast_gradient(phi_modes, coords, neighbors_idx)
    lap_phi = fast_laplacian(phi_modes, coords, neighbors_idx)
    L = np.zeros((r, r))
    for i in range(r):
        adv = np.sum(velocity_field * grad_phi[i], axis=1)
        for k in range(r):
            conv_term = np.dot(phi_modes[k], adv)
            diff_term = np.dot(phi_modes[k], lap_phi[i])
            L[k, i] = -conv_term + alpha * diff_term
    return L

# === 缓存系统 ===
L_CACHE_PATH = "lib/precomputed_L_matrices_r_12.npz"
NEIGHBOR_CACHE_PATH = "lib/neighbors_idx.npz"

def save_L_matrices(BC, L_list, path=L_CACHE_PATH):
    np.savez(path, BC=BC, L_list=L_list)
    print(f"L矩阵和BC已保存至: {path}")

def load_L_matrices(path=L_CACHE_PATH):
    if not os.path.exists(path): return None, None
    data = np.load(path)
    print(f"从 {path} 加载 L 矩阵和BC")
    return data['BC'], data['L_list']

def save_neighbors_idx(neighbors_idx, path=NEIGHBOR_CACHE_PATH):
    np.savez_compressed(path, neighbors_idx=neighbors_idx)
    print(f"邻居索引已保存至: {path}")

def load_neighbors_idx(path=NEIGHBOR_CACHE_PATH):
    if os.path.exists(path):
        data = np.load(path)
        print(f"从 {path} 加载邻居索引")
        return data['neighbors_idx']
    return None

def precompute_L_matrices_parallel(modes_T, coords, V_snapshots, alpha, neighbors_idx, n_jobs=6):
    total_jobs = len(V_snapshots)
    parallel = TqdmParallel(n_jobs=n_jobs, total=total_jobs, backend="threading")
    results = parallel(
        delayed(build_galerkin_matrix)(modes_T, V_snapshots[i], coords, neighbors_idx, alpha)
        for i in range(total_jobs)
    )
    return np.array(results)

# === Velocity POD ===
def train_velocity_POD_interpolator_rbf(V_snapshots, BC, r):
    N_snap, N_pts, _ = V_snapshots.shape
    V_reshaped = V_snapshots.reshape(N_snap, -1)
    svd = TruncatedSVD(n_components=r)
    coeffs = svd.fit_transform(V_reshaped)
    rbf_interp = RBFInterpolator(BC[:,[0,2,3]], coeffs, kernel='thin_plate_spline')
    print('Velocity POD RBF interpolator trained.')
    return svd, rbf_interp

def predict_velocity_field_rbf(new_bc, svd, rbf_interp):
    coeffs_pred = rbf_interp([[new_bc[0], new_bc[2], new_bc[3]]])[0]
    V_flat = svd.inverse_transform([coeffs_pred])[0]
    return V_flat.reshape(-1, 3), coeffs_pred

# === L 矩阵插值 ===
def train_L_interpolator(BC, L_list):
    return RBFInterpolator(BC, L_list.reshape(len(BC), -1), kernel='cubic')

def predict_L_matrix(new_bc, interpolator, r):
    L_flat = interpolator([new_bc])[0]
    return L_flat.reshape((r, r))

# === 预测温度 ===
def predict_temperature(new_bc, modes_T, coeffs_T, mean_T, svd_V, rbf_V_interp, coords, interpolator_L, r):
    V_field, _ = predict_velocity_field_rbf(new_bc, svd_V, rbf_V_interp)
    L = predict_L_matrix(new_bc, interpolator_L, r)
    a0 = coeffs_T.mean(axis=0)
    sol = solve_ivp(lambda t, a: L @ a, (0, 1), a0, t_eval=np.linspace(0, 1, 100))
    a_final = sol.y[:, -1]
    T_pred = mean_T + a_final @ modes_T
    df_save = pd.DataFrame({
        "Points:0": coords[:, 0], "Points:1": coords[:, 1], "Points:2": coords[:, 2],
        "Velocity:0": V_field[:, 0], "Velocity:1": V_field[:, 1], "Velocity:2": V_field[:, 2],
        "Temperature": T_pred
    })
    df_save.to_csv("predicted_snapshot.csv", index=False)
    return T_pred, V_field, sol.t, sol.y

# === 可视化 ===
# def onePlaneVisualize(T_true, T_predict, coords):
#     x_target = 0.95
#     mask = np.isclose(coords[:, 0], x_target, atol=1e-3)
#     y_plane, z_plane = coords[mask, 1], coords[mask, 2]
#     T_plane, T_construct = T_true[mask], T_predict[mask]
#     tri = Triangulation(y_plane, z_plane)
#     vmin, vmax = min(T_plane.min(), T_construct.min()), max(T_plane.max(), T_construct.max())
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
#     axes[0].tricontourf(tri, T_plane, levels=100, cmap='rainbow', vmin=vmin, vmax=vmax)
#     axes[0].set_title('True Temperature Snapshot')
#     axes[1].tricontourf(tri, T_construct, levels=100, cmap='rainbow', vmin=vmin, vmax=vmax)
#     axes[1].set_title('POD-Galerkin Reconstructed')
#     cbar = fig.colorbar(axes[1].collections[0], ax=axes.ravel().tolist(), shrink=0.9)
#     cbar.set_label("Temperature (K)")
#     plt.savefig('result.svg', dpi=900)
from matplotlib.tri import LinearTriInterpolator

def onePlaneVisualize(T_true, T_predict, coords, x_target=0.95):
    mask = np.isclose(coords[:, 0], x_target, atol=1e-1)
    if np.sum(mask) < 20:
        raise ValueError("截面点太少，无法可视化，请调整 x_target 或 atol。")

    y_plane = coords[mask, 1]
    z_plane = coords[mask, 2]
    T_plane = T_true[mask]
    T_construct = T_predict[mask]

    # 生成三角剖分
    tri = Triangulation(y_plane, z_plane)

    # 插值器
    interp_true = LinearTriInterpolator(tri, T_plane)
    interp_pred = LinearTriInterpolator(tri, T_construct)

    # 构造规则网格
    yi = np.linspace(y_plane.min(), y_plane.max(), 400)
    zi = np.linspace(z_plane.min(), z_plane.max(), 400)
    Y, Z = np.meshgrid(yi, zi)
    T_true_grid = interp_true(Y, Z)
    T_pred_grid = interp_pred(Y, Z)

    # 掩蔽掉插值结果中无效区域（NaN）
    T_true_masked = np.ma.masked_invalid(T_true_grid)
    T_pred_masked = np.ma.masked_invalid(T_pred_grid)

    # 设置颜色范围
    vmin = np.nanmin([T_true_grid, T_pred_grid])
    vmax = np.nanmax([T_true_grid, T_pred_grid])

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(T_true_masked, extent=(yi.min(), yi.max(), zi.min(), zi.max()),
                         origin='lower', cmap='rainbow', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('True Temperature Snapshot')
    axes[0].set_xlabel('Y'); axes[0].set_ylabel('Z')

    im1 = axes[1].imshow(T_pred_masked, extent=(yi.min(), yi.max(), zi.min(), zi.max()),
                         origin='lower', cmap='rainbow', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title('POD-Galerkin Reconstructed')
    axes[1].set_xlabel('Y'); axes[1].set_ylabel('Z')

    # 添加色条
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Temperature (K)")

    plt.savefig("result_fixed.svg", dpi=400)
    plt.show()

def export_downsampled_snapshot_to_csv(snapshot_index, npz_path=DOWNSAMPLE_CACHE_PATH, output_csv="lib/exported_snapshot_100000.csv"):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到降采样数据文件: {npz_path}")
    
    data = np.load(npz_path)
    coords = data['coords']            # (n_points, 3)
    T_snapshots = data['T']            # (n_snapshots, n_points)
    V_snapshots = data['V']            # (n_snapshots, n_points, 3)

    if snapshot_index < 0 or snapshot_index >= T_snapshots.shape[0]:
        raise IndexError(f"snapshot_index={snapshot_index} 超出范围 (共 {T_snapshots.shape[0]} 个 snapshot)")

    T = T_snapshots[snapshot_index]    # (n_points,)
    V = V_snapshots[snapshot_index]    # (n_points, 3)

    df = pd.DataFrame({
        "Points:0": coords[:, 0],
        "Points:1": coords[:, 1],
        "Points:2": coords[:, 2],
        "Velocity:0": V[:, 0],
        "Velocity:1": V[:, 1],
        "Velocity:2": V[:, 2],
        "Temperature": T
    })

    df.to_csv(output_csv, index=False)
    print(f"Snapshot {snapshot_index} 已保存为 CSV: {output_csv}")

# === 主程序 ===
def main(force_recompute_L=False, force_downsample=False):
    data_dir = "AutoCFD/Workdata/Fluent_Python/USEDATA"

    # 降采样：判断是否已有保存数据
    print("检查是否已有降采样数据...")
    coords, T_snaps, V_snaps = load_downsampled_snapshots(DOWNSAMPLE_CACHE_PATH)
    if coords is None or force_downsample:
        print("未发现缓存或强制重新降采样，开始降采样...")
        coords_raw, T_raw, V_raw, BC = load_snapshots(data_dir)
        coords, T_snaps, V_snaps = downsample_snapshots_vectorized(coords_raw, T_raw, V_raw, n_points=100000)
        save_downsampled_snapshots(DOWNSAMPLE_CACHE_PATH, coords, T_snaps, V_snaps)
        # export_downsampled_snapshots_to_csv(coords, T_snaps, V_snaps, BC)
    else:
        BC = load_BC(data_dir)
        print("使用缓存的降采样数据。")

    r, alpha = 12, 0.0034

    print('POD分解')
    modes_T, coeffs_T, mean_T, _ = compute_POD(T_snaps, r)
    svd_V, rbf_V_interp = train_velocity_POD_interpolator_rbf(V_snaps, BC, r)

    neighbors_idx = load_neighbors_idx()
    if neighbors_idx is None:
        print("计算邻居索引...")
        neighbors_idx = compute_neighbor_data(coords, k=7)
        save_neighbors_idx(neighbors_idx)

    loaded_BC, loaded_L_list = load_L_matrices()
    if loaded_BC is not None and loaded_L_list is not None and not force_recompute_L:
        print("使用缓存的 L 矩阵。")
        L_list = loaded_L_list
    else:
        print("重新计算 L 矩阵中...")
        L_list = precompute_L_matrices_parallel(modes_T, coords, V_snaps, alpha, neighbors_idx)
        save_L_matrices(BC, L_list)

    interpolator_L = train_L_interpolator(BC, L_list)
    new_bc = [0.045, 292, 0.5, 0.84]
    T_pred, V_pred, ts, a_sol = predict_temperature(new_bc, modes_T, coeffs_T, mean_T, svd_V, rbf_V_interp, coords, interpolator_L, r)
    # 找到最接近外推条件的真实 snapshot（降采样后）
    idx = np.argmin(np.linalg.norm(BC - new_bc, axis=1))
    T_true = T_snaps[idx]
    export_downsampled_snapshot_to_csv(idx)
    onePlaneVisualize(T_true, T_pred, coords)

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(T_true, T_pred)
    print(f"MSE with closest real snapshot: {mse:.4f}")

if __name__ == "__main__":
    start_time = time.time()
    main(force_recompute_L=False)
    end_time = time.time()
    print("总耗时: {:.2f} 秒".format(end_time - start_time))
