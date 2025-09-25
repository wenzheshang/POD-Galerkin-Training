import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from sklearn.decomposition import TruncatedSVD
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from matplotlib.tri import Triangulation
from scipy.interpolate import interpn, RBFInterpolator
import matplotlib.pyplot as plt
import pyvista as pv

# === 加速版梯度和拉普拉斯计算 ===

def compute_neighbor_data(coords, k=7):
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k)
    return idxs

def fast_gradient(phi_modes, coords, neighbors_idx):
    """
    向量化计算所有模态的梯度 (r, N, 3)
    """
    r, N = phi_modes.shape
    gradients = np.zeros((r, N, 3))
    for i in range(N):
        neighbor_ids = neighbors_idx[i]
        dx = coords[neighbor_ids] - coords[i]
        pinv = np.linalg.pinv(dx)
        dphi = phi_modes[:, neighbor_ids] - phi_modes[:, i][:, np.newaxis]
        gradients[:, i, :] = dphi @ pinv.T
    return gradients

def fast_laplacian(phi_modes, coords, neighbors_idx):
    """
    向量化近似计算拉普拉斯 (r, N)
    """
    r, N = phi_modes.shape
    laplacian = np.zeros((r, N))
    for i in range(N):
        neighbor_ids = neighbors_idx[i]
        h2 = np.mean(np.sum((coords[neighbor_ids] - coords[i])**2, axis=1)) + 1e-12
        laplacian[:, i] = (np.mean(phi_modes[:, neighbor_ids], axis=1) - phi_modes[:, i]) / h2
    return laplacian

# === 数据读取 ===
def load_snapshots(data_dir):
    T_snapshots = []
    V_snapshots = []
    BC = []
    coords = None

    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(data_dir, fname))
        if coords is None:
            coords = df[['Points:0', 'Points:1', 'Points:2']].values

        T_snapshots.append(df['Temperature'].values)
        V_snapshots.append(np.stack([df['Velocity:0'], df['Velocity:1'], df['Velocity:2']], axis=1))

        parts = fname.replace('.csv', '').split('_')
        V_in = float(parts[2])
        T_in = float(parts[3])
        BC.append([V_in, T_in])

    return np.array(coords), np.array(T_snapshots), np.array(V_snapshots), np.array(BC)

# === POD 分解 ===
def compute_POD(T_snapshots, r):
    mean_T = np.mean(T_snapshots, axis=0)
    fluctuations = T_snapshots - mean_T
    svd = TruncatedSVD(n_components=r)
    coeffs = svd.fit_transform(fluctuations)
    modes = svd.components_
    print('POD temperature modes computed.')
    return modes, coeffs, mean_T, svd

# === 速度场 POD + 插值 ===
def train_velocity_POD_interpolator_spline(V_snapshots, BC, r):
    N_snap, N_pts, _ = V_snapshots.shape
    V_reshaped = V_snapshots.reshape(N_snap, -1)
    svd = TruncatedSVD(n_components=r)
    coeffs = svd.fit_transform(V_reshaped)

    bc1_vals = np.unique(BC[:, 0])
    bc2_vals = np.unique(BC[:, 1])
    assert len(bc1_vals) * len(bc2_vals) == N_snap

    coeffs_grid = np.zeros((len(bc1_vals), len(bc2_vals), r))
    for i in range(N_snap):
        idx1 = np.where(bc1_vals == BC[i, 0])[0][0]
        idx2 = np.where(bc2_vals == BC[i, 1])[0][0]
        coeffs_grid[idx1, idx2, :] = coeffs[i]

    print('Velocity POD modes and spline coefficients prepared.')
    return svd, coeffs_grid, (bc1_vals, bc2_vals)

def predict_velocity_field_spline(new_bc, svd, coeffs_grid, bc_grid):
    r = coeffs_grid.shape[2]
    pred_coeffs = np.zeros(r)
    for i in range(r):
        pred_coeffs[i] = interpn(bc_grid, coeffs_grid[:, :, i], new_bc, method='cubic')
    V_flat = svd.inverse_transform([pred_coeffs])[0]
    return V_flat.reshape(-1, 3), pred_coeffs

# === Galerkin 矩阵计算 ===
def build_galerkin_matrix(phi_modes, velocity_field, coords, alpha):
    """
    使用向量化方式快速构建 Galerkin 矩阵
    """
    r, N = phi_modes.shape
    neighbors_idx = compute_neighbor_data(coords, k=7)
    grad_phi = fast_gradient(phi_modes, coords, neighbors_idx)  # shape: (r, N, 3)
    lap_phi = fast_laplacian(phi_modes, coords, neighbors_idx)  # shape: (r, N)

    L = np.zeros((r, r))
    for i in range(r):
        adv = np.sum(velocity_field * grad_phi[i], axis=1)
        for k in range(r):
            conv_term = np.dot(phi_modes[k], adv)
            diff_term = np.dot(phi_modes[k], lap_phi[i])
            L[k, i] = -conv_term + alpha * diff_term
    return L

# === L矩阵 并行构建 + 缓存 ===
L_CACHE_PATH = "lib/precomputed_L_matrices_cube.npz"

def save_L_matrices(BC, L_list, path=L_CACHE_PATH):
    np.savez(path, BC=BC, L_list=L_list)
    print(f"L矩阵和BC已保存至: {path}")

def load_L_matrices(path=L_CACHE_PATH):
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    print(f"从 {path} 加载 L 矩阵和BC")
    return data['BC'], data['L_list']

memory = Memory(location="./.l_cache", verbose=0)

@memory.cache
def cached_build_L_matrix(modes_T, V_field, coords, alpha):
    return build_galerkin_matrix(modes_T, V_field, coords, alpha)

def precompute_L_matrices_parallel(modes_T, coords, V_snapshots, alpha, n_jobs=6):
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(cached_build_L_matrix)(modes_T, V_snapshots[i], coords, alpha)
        for i in range(V_snapshots.shape[0])
    )
    return np.array(results)


def train_L_interpolator(BC, L_list):
    return RBFInterpolator(BC, L_list, kernel='cubic')

def predict_L_matrix(new_bc, interpolator, r):
    L_flat = interpolator([new_bc])[0]
    return L_flat.reshape((r, r))

# === 温度预测 ===
def predict_temperature(new_bc, modes_T, coeffs_T, mean_T, svd_V, coeffs_grid, bc_grid, coords, interpolator_L, r):
    V_field, _ = predict_velocity_field_spline(new_bc, svd_V, coeffs_grid, bc_grid)
    L = predict_L_matrix(new_bc, interpolator_L, r)

    a0 = coeffs_T.mean(axis=0)
    sol = solve_ivp(lambda t, a: L @ a, (0, 1), a0, t_eval=np.linspace(0, 1, 100))
    a_final = sol.y[:, -1]
    T_pred = mean_T + a_final @ modes_T

    df_save = pd.DataFrame({
        "Points:0": coords[:, 0],
        "Points:1": coords[:, 1],
        "Points:2": coords[:, 2],
        "Velocity:0": V_field[:, 0],
        "Velocity:1": V_field[:, 1],
        "Velocity:2": V_field[:, 2],
        "Temperature": T_pred
    })
    df_save.to_csv("predicted_snapshot.csv", index=False)
    return T_pred, V_field, sol.t, sol.y

# === 可视化 ===
def visualize(coords, T_field, title='Temperature Field'):
    grid = pv.PolyData(coords)
    grid["Temperature"] = T_field
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="Temperature", cmap="coolwarm")
    plotter.add_title(title)
    plotter.show()

def onePlaneVisualize(T_true, T_predict, coords):
    z_target = 1.22
    mask = np.isclose(coords[:, 2], z_target, atol=1e-4)

    x_plane = coords[mask, 0]
    y_plane = coords[mask, 1]
    T_plane = T_true[mask]
    T_construct = T_predict[mask]

    tri = Triangulation(x_plane, y_plane)
    vmin = min(T_plane.min(), T_construct.min())
    vmax = max(T_plane.max(), T_construct.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].tricontourf(tri, T_plane, levels=100, cmap='rainbow', vmin=vmin, vmax=vmax)
    axes[0].set_title('True Temperature Snapshot')
    axes[1].tricontourf(tri, T_construct, levels=100, cmap='rainbow', vmin=vmin, vmax=vmax)
    axes[1].set_title('POD-Galerkin Reconstructed')
    cbar = fig.colorbar(axes[1].collections[0], ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Temperature (K)")
    plt.savefig('result.svg', dpi=900)

# === 主程序 ===
def main(force_recompute_L=False):
    data_dir = "AutoCFD/Workdata/Fluent_Python/2025-06-04_16-21/saveData"
    coords, T_snaps, V_snaps, BC = load_snapshots(data_dir)

    r = 9
    alpha = 0.0034

    modes_T, coeffs_T, mean_T, svd_T = compute_POD(T_snaps, r)
    svd_V, coeffs_grid, bc_grid = train_velocity_POD_interpolator_spline(V_snaps, BC, r)

    # 检查是否已有缓存的L矩阵
    loaded_BC, loaded_L_list = load_L_matrices()
    if loaded_BC is not None and loaded_L_list is not None and not force_recompute_L:
        print("已检测到已保存的L矩阵，直接加载使用。")
        L_list = loaded_L_list
    else:
        print("正在并行预计算 Galerkin 矩阵 ...")
        L_list = precompute_L_matrices_parallel(modes_T, coords, V_snaps, alpha, n_jobs=6)
        save_L_matrices(BC, L_list)  # 仅在重算后保存

    print("正在训练 Galerkin 矩阵插值器 ...")
    interpolator_L = train_L_interpolator(BC, L_list)

    new_bc = [1.367, 295.35]
    T_pred, V_pred, ts, a_sol = predict_temperature(new_bc, modes_T, coeffs_T, mean_T,
                                                    svd_V, coeffs_grid, bc_grid,
                                                    coords, interpolator_L, r)

    #visualize(coords, T_pred, title=f"Predicted Temperature for BC={new_bc}")

    from sklearn.metrics import mean_squared_error
    idx = np.argmin(np.linalg.norm(BC - new_bc, axis=1))
    T_true = T_snaps[idx]
    onePlaneVisualize(T_true, T_pred, coords)
    mse = mean_squared_error(T_true, T_pred)
    print(f"MSE with closest real snapshot: {mse:.4f}")



if __name__ == "__main__":
    start_time = time.time()
    main(force_recompute_L=False)  # 设置为 True 可强制重算 L
    end_time = time.time()
    print("总耗时: {:.2f} 秒".format(end_time - start_time))

