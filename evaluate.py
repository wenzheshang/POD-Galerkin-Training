import imp
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def load_snapshots(data_dir):

    coords = None

    df = pd.read_csv(data_dir)
    if coords is None:
        coords = df[['Points:0', 'Points:1', 'Points:2']].values

    Mass_snapshots=list(df['Mass_fraction_of_co2'].values)
    T_snapshots=list(df['Temperature'].values)
    Vx_snapshots=list(df['Velocity'].values)

    return np.clip(np.array(T_snapshots),290,313), np.clip(np.array(Vx_snapshots),0,10), np.array(Mass_snapshots)

def evaluate_temperature_similarity(T_true, T_pred):
    errors = T_true - T_pred
    mse = mean_squared_error(T_true, T_pred)
    rmse = np.sqrt(mse)
    max_ae = np.max(np.abs(errors))
    nrmse = rmse / (np.max(T_true) - np.min(T_true))
    r2 = r2_score(T_true, T_pred)
    cosine_sim = 1 - cosine(T_true, T_pred)
    se = np.std(errors, ddof=1) / np.sqrt(len(T_true))
    cv_true = np.std(T_true, ddof=1) / np.mean(np.abs(T_true))  # 相对于真实值均值
    cv_pre = np.std(T_pred, ddof=1) / np.mean(np.abs(T_pred))  # 相对于真实值均值

    return {
        "MSE": mse,
        "RMSE": rmse,
        "SE": se,
        "CV_true": cv_true,
        "CV_pred": cv_pre,
        "MaxAE": max_ae,
        "NRMSE": nrmse,
        "R²": r2,
        "Cosine Similarity": cosine_sim
    }

def evaluate_concentration_similarity(T_true, T_pred):
    errors = T_true - T_pred
    mse = mean_squared_error(T_true, T_pred)
    rmse = np.sqrt(mse)
    max_ae = np.max(np.abs(errors))
    nrmse = rmse / (np.max(T_true) - np.min(T_true))
    r2 = r2_score(T_true, T_pred)
    cosine_sim = 1 - cosine(T_true, T_pred)
    se = np.std(errors, ddof=1) / np.sqrt(len(T_true))
    cv_true = np.std(T_true, ddof=1) / np.mean(np.abs(T_true))  # 相对于真实值均值
    cv_pre = np.std(T_pred, ddof=1) / np.mean(np.abs(T_pred))  # 相对于真实值均值

    return {
        "MSE": mse,
        "RMSE": rmse,
        "SE": se,
        "CV_true": cv_true,
        "CV_pred": cv_pre,
        "MaxAE": max_ae,
        "NRMSE": nrmse,
        "R²": r2,
        "Cosine Similarity": cosine_sim
    }

def evaluate_velocity_similarity(V_true, V_pred):
    flat_true = V_true.flatten()
    flat_pred = V_pred.flatten()
    errors = flat_true - flat_pred

    mse = mean_squared_error(flat_true, flat_pred)
    rmse = np.sqrt(mse)
    max_ae = np.max(np.abs(errors))
    nrmse = rmse / (np.max(flat_true) - np.min(flat_true))
    r2 = r2_score(flat_true, flat_pred)
    cosine_sim = 1 - cosine(flat_true, flat_pred)
    se = np.std(errors, ddof=1) / np.sqrt(len(flat_true))
    cv_true = np.std(flat_true, ddof=1) / np.mean(np.abs(flat_true))  # 相对于真实值均值
    cv_pre = np.std(flat_pred, ddof=1) / np.mean(np.abs(flat_pred))  # 相对于真实值均值

    return {
        "MSE": mse,
        "RMSE": rmse,
        "SE": se,
        "CV_true": cv_true,
        "CV_pred": cv_pre,
        "MaxAE": max_ae,
        "NRMSE": nrmse,
        "R²": r2,
        "Cosine Similarity": cosine_sim
    }

def plot_prediction_vs_actual(y_true, y_pred, title="Velocity Prediction vs Actual", save_path=None):
    """
    绘制预测值 vs 实际值散点图，并计算 R² 分数及 ±0.1 区间内比例
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    r2 = r2_score(y_true, y_pred)

    # 计算 y = x ± 0.1 边界内的比例
    diffs = np.abs(y_pred - y_true)
    within_band = diffs <= 0.1
    inside_ratio = np.sum(within_band) / len(y_true)

    # 分开绘制区内点和区外点
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[~within_band], y_pred[~within_band], s=5, alpha=0.2, c='gray', label='Outside ±0.1')
    plt.scatter(y_true[within_band], y_pred[within_band], s=5, alpha=0.6, c='blue', label='Within ±0.1')

    # 对角线和边界线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
    plt.plot([min_val, max_val], [min_val + 0.1, max_val + 0.1], 'k--', lw=0.8, alpha=0.5)
    plt.plot([min_val, max_val], [min_val - 0.1, max_val - 0.1], 'k--', lw=0.8, alpha=0.5)

    # 标签和标题
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{title}\n$R^2$ = {r2:.4f}, Within ±0.1 = {inside_ratio*100:.2f}%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

def plot_prediction_vs_actual_concentration(y_true, y_pred, title="CO2 concentration Prediction vs Actual", save_path=None):
    """
    绘制预测值 vs 实际值散点图，并计算 R² 分数及 ±0.0001 区间内比例
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    r2 = r2_score(y_true, y_pred)

    # 计算 y = x ± 0.1 边界内的比例
    diffs = np.abs(y_pred - y_true)
    within_band = diffs <= 0.0001
    inside_ratio = np.sum(within_band) / len(y_true)

    # 分开绘制区内点和区外点
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[~within_band], y_pred[~within_band], s=5, alpha=0.2, c='gray', label='Outside ±0.0001')
    plt.scatter(y_true[within_band], y_pred[within_band], s=5, alpha=0.6, c='blue', label='Within ±0.0001')

    # 对角线和边界线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
    plt.plot([min_val, max_val], [min_val + 0.0001, max_val + 0.0001], 'k--', lw=0.8, alpha=0.5)
    plt.plot([min_val, max_val], [min_val - 0.0001, max_val - 0.0001], 'k--', lw=0.8, alpha=0.5)

    # 标签和标题
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{title}\n$R^2$ = {r2:.4f}, Within ±0.0001 = {inside_ratio*100:.2f}%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

def plot_prediction_vs_actual_temperature(y_true, y_pred, title="Temperature Prediction vs Actual", save_path=None):
    """
    绘制预测值 vs 实际值散点图，并计算 R² 分数及 ±0.5 区间内比例
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    r2 = r2_score(y_true, y_pred)

    # 计算 y = x ± 0.1 边界内的比例
    diffs = np.abs(y_pred - y_true)
    within_band = diffs <= 0.5
    inside_ratio = np.sum(within_band) / len(y_true)

    # 分开绘制区内点和区外点
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[~within_band], y_pred[~within_band], s=5, alpha=0.2, c='gray', label='Outside ±0.5')
    plt.scatter(y_true[within_band], y_pred[within_band], s=5, alpha=0.6, c='blue', label='Within ±0.5')

    # 对角线和边界线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
    plt.plot([min_val, max_val], [min_val + 0.5, max_val + 0.5], 'k--', lw=0.8, alpha=0.5)
    plt.plot([min_val, max_val], [min_val - 0.5, max_val - 0.5], 'k--', lw=0.8, alpha=0.5)

    # 标签和标题
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title(f"{title}\n$R^2$ = {r2:.4f}, Within ±0.5 = {inside_ratio*100:.2f}%")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

def save_metrics_to_txt(metrics, filename):
    """
    保存指标字典为txt文件
    """
    with open(filename, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6e}\n")
    print(f"指标已保存到 {filename}")

if __name__=="__main__":

    data_dir_true = "D:/NextPaper/code/comparedata/smac122/new_DataSave_32_0.07_290.57_0.0_0.26.csv"#comparedata/originPredictData/
    data_dir_pre = "D:/NextPaper/code/comparedata/smac122/127_predicted_snapshot_co2_0.07_290.57_0.0_0.26.csv"#comparedata/originPredictData/
    T_snaps_CFD, Vx_snaps_CFD, Mass_snaps_CFD = load_snapshots(data_dir_true)
    T_snaps_Predict, Vx_snaps_Predict, Mass_snaps_Predict = load_snapshots(data_dir_pre)

    T_metrics = evaluate_temperature_similarity(T_snaps_CFD, T_snaps_Predict)
    Mass_metrics = evaluate_concentration_similarity(Mass_snaps_CFD, Mass_snaps_Predict)
    Vx_metrics = evaluate_velocity_similarity(Vx_snaps_CFD, Vx_snaps_Predict)
    # Vy_metrics = evaluate_velocity_similarity(Vy_snaps_CFD, Vy_snaps_Predict)
    # Vz_metrics = evaluate_velocity_similarity(Vz_snaps_CFD, Vz_snaps_Predict)
    save_metrics_to_txt(T_metrics, "D:/NextPaper/code/comparedata/smac122/compare_122_Temperature_metrics.txt")
    save_metrics_to_txt(Mass_metrics, "D:/NextPaper/code/comparedata/smac122/compare_122_Concentration_metrics.txt")
    save_metrics_to_txt(Vx_metrics, "D:/NextPaper/code/comparedata/smac122/compare_122_Velocity_metrics.txt")

    plot_prediction_vs_actual_concentration(Mass_snaps_CFD, Mass_snaps_Predict,'Mass Prediction vs Actual','D:/NextPaper/code/comparedata/smac122/test_122_CO2_newdata_compare_TestBC.png')
    plot_prediction_vs_actual_temperature(T_snaps_CFD, T_snaps_Predict,'T Prediction vs Actual','D:/NextPaper/code/comparedata/smac122/test_122_T_newdata_compare_TestBC.png')
    plot_prediction_vs_actual(Vx_snaps_CFD, Vx_snaps_Predict,'Vx Prediction vs Actual','D:/NextPaper/code/comparedata/smac122/test_122_V_newdata_compare_TestBC.png')
    # plot_prediction_vs_actual(Vy_snaps_CFD, Vy_snaps_Predict,'Vy Prediction vs Actual','D:/NextPaper/code/evaluateResult/Vy_Test.png')
    # plot_prediction_vs_actual(Vz_snaps_CFD, Vz_snaps_Predict,'Vz Prediction vs Actual','D:/NextPaper/code/evaluateResult/Vz_Test.png')
    
    print("=== 温度场相似度指标 ===")
    for k, v in T_metrics.items():
        print(f"{k}: {v:.4e}")

    print("=== 浓度场相似度指标 ===")
    for k, v in Mass_metrics.items():
        print(f"{k}: {v:.4e}")

    print("=== 速度场相似度指标 ===")
    for k, v in Vx_metrics.items():
        print(f"{k}: {v:.4e}")

    # print("=== y速度场相似度指标 ===")
    # for k, v in Vy_metrics.items():
    #     print(f"{k}: {v:.4e}")
    
    # print("=== z速度场相似度指标 ===")
    # for k, v in Vz_metrics.items():
    #     print(f"{k}: {v:.4e}")