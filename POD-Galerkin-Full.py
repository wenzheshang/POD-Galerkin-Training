import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, Memory
from sklearn.decomposition import TruncatedSVD
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import threading
from tqdm import tqdm
from scipy.linalg import eigvals
from scipy.spatial import Delaunay, ConvexHull

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


class LocalRBFVelocityInterpolator:
    def __init__(self, BC, V_snapshots, svd, kernel='thin_plate_spline', smoothing=1e-6, neighbors=30):
        self.newBC = BC[:,[0,2,3]]
        self.bc_min = self.newBC.min(axis=0)
        self.bc_max = self.newBC.max(axis=0)
        self.BC_norm = (self.newBC - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

        # —— 速度做“去均值”的 SVD（强烈推荐）——
        self.V_mean = V_snapshots.mean(axis=0)
        V_fluc = V_snapshots - self.V_mean
        self.svd = svd
        self.coeffs = self.svd.fit_transform(V_fluc)  # 用去均值后的数据拟合

        # 自动设置 epsilon（可选）：用中位数距离作为尺度
        from scipy.spatial import distance_matrix
        D = distance_matrix(self.BC_norm, self.BC_norm)
        med = np.median(D[D>0])
        self.epsilon = med if np.isfinite(med) and med>0 else 0.2

        # 直接用RBFInterpolator做多输出插值（目标是 coeffs）
        self.rbf = RBFInterpolator(
            self.BC_norm,
            self.coeffs,
            kernel=kernel, epsilon=self.epsilon,
            smoothing=smoothing,
            neighbors=neighbors  # 让它内部用平滑的局部近邻
        )

    def normalize(self, x):
        return (x - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

    def predict(self, new_bc):
        x = np.asarray(new_bc)[[0,2,3]]
        # 防止轻微出界的外推不稳：clip回训练范围（也可只发警告）
        x = np.minimum(np.maximum(x, self.bc_min), self.bc_max)
        x_norm = self.normalize(x)

        coeffs_pred = self.rbf(x_norm[None, :])[0]
        V_flat = self.svd.inverse_transform(coeffs_pred[None, :])[0] + self.V_mean
        return V_flat, coeffs_pred

# === L 矩阵插值 ===
class LocalRBFGalerkinMatrixInterpolator:
    def __init__(self, BC, L_list, r, 
                 kernel='multiquadric', smoothing=1e-6, neighbors=30):
        # 只取 [0,2,3] 维度
        self.newBC = BC
        self.bc_min = self.newBC.min(axis=0)
        self.bc_max = self.newBC.max(axis=0)
        self.BC_norm = (self.newBC - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

        self.r = r
        # 展平成二维
        self.L_list = L_list.reshape(len(BC), -1)

        # 自动设置 epsilon（与速度插值一致）
        from scipy.spatial import distance_matrix
        D = distance_matrix(self.BC_norm, self.BC_norm)
        med = np.median(D[D > 0])
        self.epsilon = med if np.isfinite(med) and med > 0 else 0.2

        # RBF 拟合 L 矩阵系数
        self.rbf = RBFInterpolator(
            self.BC_norm,
            self.L_list,
            kernel=kernel, epsilon=self.epsilon,
            smoothing=smoothing,
            neighbors=neighbors
        )

    def normalize(self, x):
        return (x - self.bc_min) / (self.bc_max - self.bc_min + 1e-12)

    def predict(self, new_bc):
        x = np.asarray(new_bc)
        # 裁剪到训练范围
        x = np.minimum(np.maximum(x, self.bc_min), self.bc_max)
        x_norm = self.normalize(x)

        L_flat = self.rbf(x_norm[None, :])[0]
        return L_flat.reshape((self.r, self.r))

# === 数据读取 ===
def load_snapshots(data_dir):
    T_snapshots, V_snapshots, Mass_snapshots, Vm_snapshots, BC = [], [], [], [], []
    coords = None

    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        filepath = os.path.join(data_dir, fname)
        df = pd.read_csv(filepath)
        if coords is None:
            coords = df[["Points:0", "Points:1", "Points:2"]].values
        T_snapshots.append(df['Temperature'].values)
        Mass_snapshots.append((df['Mass_fraction_of_co2'].values)**0.5)
        V_snapshots.append(df[['Velocity:0', 'Velocity:1', 'Velocity:2']].values)
        Vm_snapshots.append((df['Velocity'].values)**0.5)

        # 提取边界条件
        parts = fname.replace('.csv', '').split('_')
        BC.append([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])

    return np.array(coords), np.clip(np.array(T_snapshots),290,313), np.array(Mass_snapshots), np.array(V_snapshots), np.clip(np.array(Vm_snapshots),0,10), np.array(BC)

# === POD ===
def compute_POD(T_snapshots, r):
    mean_T = np.mean(T_snapshots, axis=0)
    fluctuations = T_snapshots - mean_T
    svd = TruncatedSVD(n_components=r)
    coeffs = svd.fit_transform(fluctuations)
    modes = svd.components_
    explained_ratios = svd.explained_variance_ratio_
    # 总共保留了多少能量
    energy_retention = np.sum(explained_ratios)
    print(f"当前选用的 {r} 个模态共保留浓度能量比例: {energy_retention:.4f}")
    print("POD temperature modes computed.")
    return modes, coeffs, mean_T, svd

# === 邻域 & 导数计算 ===
def compute_neighbor_data(coords, k=15):
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k)
    return idxs

def fast_gradient(phi_modes, coords, neighbors_idx):
    r, N = phi_modes.shape
    gradients = np.zeros((r, N, 3))

    for i in range(N):
        nbr_ids = neighbors_idx[i]
        xi = coords[i]
        X = coords[nbr_ids] - xi  # (k, 3)
        distances = np.linalg.norm(X, axis=1) + 1e-12  # 防止除以0
        X_normalized = X / distances[:, None]

        weights = np.exp(-distances**2 / (np.mean(distances)**2 + 1e-12))
        W = np.diag(weights)

        A = X_normalized
        AT_W = A.T @ W
        H = AT_W @ A + 1e-8 * np.eye(3)  # 正则化稳定求逆

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        for j in range(r):
            dphi = phi_modes[j, nbr_ids] - phi_modes[j, i]
            gradients[j, i] = H_inv @ AT_W @ dphi

    print("修正后梯度 max abs:", np.max(np.abs(gradients)))
    return gradients

# 网格权重计算
def compute_weights_delaunay(coords):
    """
    coords: (N, d) array, d=2 or 3
    returns: weights (N,), domain_volume (float)
    """
    coords = np.asarray(coords)
    N, d = coords.shape
    if d not in (2,3):
        raise ValueError("只支持 2D 或 3D 点云")

    # 1) 用 Delaunay 得到简单形
    tri = Delaunay(coords)  # 2D: triangles; 3D: tetrahedra
    simplices = tri.simplices  # shape (n_simplex, d+1)

    weights = np.zeros(N, dtype=float)

    if d == 2:
        # 对每个三角形，把面积 / 3 加给三个顶点
        for s in simplices:
            a, b, c = coords[s[0]], coords[s[1]], coords[s[2]]
            area = 0.5 * abs(np.cross(b - a, c - a))
            weights[s[0]] += area / 3.0
            weights[s[1]] += area / 3.0
            weights[s[2]] += area / 3.0
    else:  # d == 3
        # 对每个四面体，把体积 / 4 加给四个顶点
        for s in simplices:
            a, b, c, dpt = coords[s[0]], coords[s[1]], coords[s[2]], coords[s[3]]
            vol = abs(np.linalg.det(np.vstack([b-a, c-a, dpt-a]).T)) / 6.0
            weights[s[0]] += vol / 4.0
            weights[s[1]] += vol / 4.0
            weights[s[2]] += vol / 4.0
            weights[s[3]] += vol / 4.0

    # 规范化：用凸包体积（或面积）作参考，确保 sum(weights) == domain_volume
    hull = ConvexHull(coords)
    domain_vol = hull.volume  # 2D->area, 3D->volume
    sumw = weights.sum()
    if sumw <= 0:
        raise RuntimeError("计算权重出错，权重和为0")
    # 归一化以匹配凸包体积（修正数值误差）
    weights *= (domain_vol / sumw)

    return weights, domain_vol


def fast_laplacian(phi_modes, coords, neighbors_idx):
    r, N = phi_modes.shape
    laplacian = np.zeros((r, N))

    for i in range(N):
        nbr_ids = neighbors_idx[i]
        xi = coords[i]
        X = coords[nbr_ids] - xi  # (k, 3)

        # 特征矩阵 A: 多项式项
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        A = np.stack([
            np.ones_like(x), x, y, z,
            x*x, y*y, z*z,
            x*y, y*z, z*x
        ], axis=1)

        # 权重矩阵（基于距离）
        distances = np.linalg.norm(X, axis=1)
        weights = np.exp(-distances**2 / (np.mean(distances)**2 + 1e-12))
        W = np.diag(weights)

        AT_W = A.T @ W
        H = AT_W @ A + 1e-8 * np.eye(A.shape[1])
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H)

        for j in range(r):
            dphi = phi_modes[j, nbr_ids] - phi_modes[j, i]
            coeffs = H_inv @ AT_W @ dphi

            # 提取二阶导系数
            laplacian[j, i] = 2 * (coeffs[4] + coeffs[5] + coeffs[6])  # 2*(φ_xx + φ_yy + φ_zz)

    print("修正后Laplacian max abs:", np.max(np.abs(laplacian)))
    return laplacian

# === 缓存梯度与拉普拉斯 ===
def get_grad_lap_path(r, N):
    return f"lib/grad_lap_r{r}_N{N}_co2_NEWDATAnewCombine.npz"

def save_gradient_laplacian(gradients, laplacians, r, N):
    path = get_grad_lap_path(r, N)
    np.savez_compressed(path, gradients=gradients, laplacians=laplacians)
    print(f"梯度和拉普拉斯缓存已保存至: {path}")

def load_gradient_laplacian(r, N):
    path = get_grad_lap_path(r, N)
    if os.path.exists(path):
        data = np.load(path)
        print(f"从 {path} 加载梯度和拉普拉斯缓存")
        return data['gradients'], data['laplacians']
    return None, None

def ensure_gradient_laplacian_cached(phi_modes, coords, neighbors_idx):
    r, N = phi_modes.shape
    gradients, laplacians = load_gradient_laplacian(r, N)
    if gradients is None or laplacians is None:
        print("首次计算梯度与拉普拉斯（主线程）...")
        gradients = fast_gradient(phi_modes, coords, neighbors_idx)
        laplacians = fast_laplacian(phi_modes, coords, neighbors_idx)
        save_gradient_laplacian(gradients, laplacians, r, N)
    else:
        print("已检测到梯度与拉普拉斯缓存")
    return gradients, laplacians

# === Galerkin 矩阵 ===
def build_galerkin_matrix(phi_modes, velocity_field, gradients, laplacians,
                                   alpha, weights, skew_symm=True, enforce_mass_identity=False):
    """
    phi_modes: (r, N)
    velocity_field: (N, 3)
    gradients: (r, N, 3)  -- gradients[j,i,:] is grad of phi_j at node i
    laplacians: (r, N)    -- laplacian[j, i] is Δ phi_j at node i (optional if using gradients)
    weights: (N,) or None  -- quadrature weights; if None assume uniform weights=1
    skew_symm: whether to use skew-symmetric conv form
    enforce_mass_identity: if True, compute mass matrix M and return (M, L), else assume M = I
    Returns:
        if enforce_mass_identity: return M, L  (M: r x r, L: r x r such that M da/dt = L a)
        else: return L (such that da/dt = L a) assuming M = I
    """
    r, N = phi_modes.shape
    if weights is None:
        w = np.ones(N)
    else:
        w = weights

    # mass matrix M_{kl} = (phi_k, phi_l)
    M = np.zeros((r, r))
    for k in range(r):
        for l in range(r):
            M[k, l] = np.dot(w * phi_modes[k], phi_modes[l])

    # Precompute gradient inner products G_kl = (grad phi_k, grad phi_l)
    G = np.zeros((r, r))
    for k in range(r):
        for l in range(r):
            # inner product over domain: sum_n w[n] * grad_phi_k[n] · grad_phi_l[n]
            G[k, l] = np.sum(w * np.sum(gradients[k] * gradients[l], axis=1))

    # convective-like terms: A_kl = (phi_k, u·∇ phi_l)
    A = np.zeros((r, r))
    # compute u·∇ phi_l at nodes using gradients[l], then integrate
    for l in range(r):
        u_dot_grad = np.sum(velocity_field * gradients[l], axis=1)  # shape (N,)
        for k in range(r):
            A[k, l] = np.dot(w * phi_modes[k], u_dot_grad)

    # assemble L. There are two common choices:
    #  - direct form: L = -A + alpha * B  with B_{k,l} = (phi_k, Δ phi_l)
    #  - divergence/grad form (preferred for diffusion): use -alpha * G
    # Here we build L assuming we want M da/dt = L a
    # compute B via laplacians if provided, else use -G (since (phi_k, Δ phi_l) = -G + boundary)
    if laplacians is not None:
        B = np.zeros((r, r))
        for l in range(r):
            for k in range(r):
                B[k, l] = np.dot(w * phi_modes[k], laplacians[l])
        # but more stable is to use -G (no boundary considered)
        B_from_grad = -G
    else:
        B = None
        B_from_grad = -G

    # Choose convective treatment
    if skew_symm:
        # skew-symmetric form: C = 0.5*(A - A^T)
        C = 0.5 * (A - A.T)
    else:
        C = A

    # Now assemble L (right-hand side matrix) in form M da/dt = L a
    L = -C + alpha * (B_from_grad if B is None else B)

    if enforce_mass_identity:
        return M, L
    else:
        # if M is not identity, compute M^{-1} L (careful: consider regularization if M nearly singular)
        try:
            Minv = np.linalg.inv(M)
            L_effective = Minv @ L
        except np.linalg.LinAlgError:
            # pseudo-inverse fallback + Tikhonov regularization
            reg = 1e-8 * np.eye(r)
            Minv = np.linalg.pinv(M + reg)
            L_effective = Minv @ L
        # optional: spectral stabilization check
        eigs = eigvals(L_effective)
        max_real = np.max(eigs.real)
        if max_real > 1e-8:
            # warn or dampen
            # e.g., shift L_effective to reduce max_real
            L_effective = L_effective - (max_real + 1e-8) * np.eye(r)
        return L

# def build_galerkin_matrix(phi_modes, velocity_field, gradients, laplacians, alpha):
#     r, N = phi_modes.shape
#     L = np.zeros((r, r))
#     for i in range(r):
#         adv = np.sum(velocity_field * gradients[i], axis=1)
#         for k in range(r):
#             conv_term = np.dot(phi_modes[k], adv)
#             diff_term = np.dot(phi_modes[k], laplacians[i])
#             L[k, i] = -conv_term + alpha * diff_term
#     return L

# === 缓存系统 ===
L_CACHE_PATH = "lib/precomputed_L_matrices_ori_r_18_co2__NEWDATAnewCombine.npz"
L_T_CACHE_PATH = 'lib/precomputed_L_T_matrices_ori_r_18_co2__NEWDATAnewCombine.npz'
NEIGHBOR_CACHE_PATH = "lib/neighbors_idx_ori_r_18_co2_NEWDATAnewCombine.npz"

def save_L_matrices(BC, L_list, path=L_CACHE_PATH):
    np.savez(path, BC=BC, L_list=L_list)
    print(f"L矩阵和BC已保存至: {path}")

def save_L_T_matrices(BC, L_list, path=L_T_CACHE_PATH):
    np.savez(path, BC=BC, L_list=L_list)
    print(f"L矩阵和BC已保存至: {path}")

def load_L_matrices(path=L_CACHE_PATH):
    if not os.path.exists(path): return None, None
    data = np.load(path)

    df = pd.DataFrame({"L":str(data['L_list'])},index=[0])
    df.to_csv("dataL.csv", index=False)

    print(f"从 {path} 加载 L 矩阵和BC")
    return data['BC'], data['L_list']

def load_L_T_matrices(path=L_T_CACHE_PATH):
    if not os.path.exists(path): return None, None
    data = np.load(path)
    print(f"从 {path} 加载 L_T 矩阵和BC")
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

def precompute_L_matrices_parallel(modes_T, coords, V_snapshots, alpha, neighbors_idx, n_jobs=31):
    r, N = modes_T.shape
    gradients, laplacians = ensure_gradient_laplacian_cached(modes_T, coords, neighbors_idx)

    total_jobs = len(V_snapshots)
    parallel = TqdmParallel(n_jobs=n_jobs, total=total_jobs, backend="threading")

    weights, _ = compute_weights_delaunay(coords)

    results = parallel(
        delayed(build_galerkin_matrix)(modes_T, V_snapshots[i], gradients, laplacians, alpha, weights,skew_symm=True, enforce_mass_identity=False)
        for i in range(total_jobs)
    )
    return np.array(results)

# === 预测温度 ===
def predict_temperature(new_bc, modes_T, coeffs_T, mean_T,
                    modes_Mass, coeffs_Mass, mean_Mass,
                    velocity_interp, galerkin_interp, galerkin_interp_T,
                    coords, rbf_mass_coeff, rbf_temp_coeff):

    # --- 如果 velocity_interp 用合成速度，这里不用改 ---
    V_f, _ = velocity_interp.predict(new_bc)

    V_field = V_f**2

    # --- 用 RBF 插值得到初始系数（代替 mean） ---
    a0 = rbf_mass_coeff([new_bc])[0]
    a0_T = rbf_temp_coeff([new_bc])[0]

    # --- Galerkin ODE 演化 ---
    L = galerkin_interp.predict([new_bc[0],new_bc[2],new_bc[3]])

    df = pd.DataFrame({"L":str(L)},index=[0])
    df.to_csv("preL.csv", index=False)

    sol = solve_ivp(lambda t, a: L @ a, (0, 1), a0, t_eval=np.linspace(0, 1, 1000))
    a_final = sol.y[:, -1]
    Mass_pred = mean_Mass + a_final @ modes_Mass

    L_T = galerkin_interp_T.predict(new_bc)
    sol_T = solve_ivp(lambda t, a: L_T @ a, (0, 1), a0_T, t_eval=np.linspace(0, 1, 1000))
    a_T_final = sol_T.y[:, -1]
    T_pred = mean_T + a_T_final @ modes_T

    # --- 保存预测结果 ---
    df = pd.DataFrame({
        "Points:0": coords[:,0], "Points:1": coords[:,1], "Points:2": coords[:,2],
        "Velocity": V_field,  # 这里还是你原来的一维合成速度
        "Temperature": T_pred,
        "Mass_fraction_of_co2": Mass_pred**2
    })
    df.to_csv("predicted_snapshot_bc_test1_{}.csv".format("_".join(map(str,new_bc))), index=False)

    print("预测并保存完成")
    return T_pred, Mass_pred


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

# === 主程序 ===
def main(force_recompute_L=False):
    data_dir = "AutoCFD/Workdata/Fluent_Python/NEWUSEDATA"
    coords, T_snaps, Mass_snaps, V_snaps, Vm_snaps, BC = load_snapshots(data_dir)
    r, rt, alpha = 24, 18, 0.0034

    modes_T, coeffs_T, mean_T, _ = compute_POD(T_snaps, rt)
    modes_Mass, coeffs_Mass, mean_Mass, _ = compute_POD(Mass_snaps, rt)

    svd_V = TruncatedSVD(n_components=r)
    svd_V.fit(Vm_snaps)
    explained_ratios = svd_V.explained_variance_ratio_
    energy_retention = np.sum(explained_ratios)
    print(f"当前选用的 {r} 个模态共保留速度能量比例: {energy_retention:.4f}")
    print('Velocity POD RBF interpolator trained.')
    local_rbf_interp = LocalRBFVelocityInterpolator(BC, Vm_snaps, svd_V)

    neighbors_idx = load_neighbors_idx()
    if neighbors_idx is None:
        print("计算邻居索引...")
        neighbors_idx = compute_neighbor_data(coords, k=7)
        save_neighbors_idx(neighbors_idx)

    loaded_BC, loaded_L_list = load_L_matrices()
    if loaded_BC is not None and loaded_L_list is not None and not force_recompute_L:
        print("使用缓存 L 矩阵。")
        L_list = loaded_L_list
    else:
        print("计算 L 矩阵中...")
        L_list = precompute_L_matrices_parallel(modes_Mass, coords, V_snaps, alpha, neighbors_idx)

    loaded_BC, loaded_L_T_list = load_L_T_matrices()   
    if loaded_BC is not None and loaded_L_T_list is not None and not force_recompute_L:
        print("使用缓存 L_T 矩阵。")
        L_T_list = loaded_L_T_list
    else:
        print("计算 L_T 矩阵中...")
        L_T_list = precompute_L_matrices_parallel(modes_T, coords, V_snaps, alpha, neighbors_idx)
        save_L_T_matrices(BC, L_T_list)

    interpolator_L = LocalRBFGalerkinMatrixInterpolator(BC[:,[0,2,3]], L_list, rt)
    interpolator_L_T = LocalRBFGalerkinMatrixInterpolator(BC, L_T_list, rt)
    
    rbf_mass_coeff = RBFInterpolator(BC, coeffs_Mass, kernel="multiquadric", epsilon=1.5)
    rbf_temp_coeff = RBFInterpolator(BC, coeffs_T, kernel="multiquadric", epsilon=1.5)

    new_bc = [0.0361196802582503,297.353781909354,0,0.9659258]
    predict_temperature(new_bc, modes_T, coeffs_T, mean_T,
                modes_Mass, coeffs_Mass, mean_Mass,
                local_rbf_interp, interpolator_L, interpolator_L_T,
                coords, rbf_mass_coeff, rbf_temp_coeff)
    # idx = np.argmin(np.linalg.norm(BC - new_bc, axis=1))
    # T_true = T_snaps[idx]
    # # onePlaneVisualize(T_true, T_pred, coords)

    # from sklearn.metrics import mean_squared_error
    # mse = mean_squared_error(T_true, T_pred)
    # print(f"MSE with closest real snapshot: {mse:.4f}")

if __name__ == "__main__":
    start_time = time.time()
    main(force_recompute_L=False)
    end_time = time.time()
    print("总耗时: {:.2f} 秒".format(end_time - start_time))
