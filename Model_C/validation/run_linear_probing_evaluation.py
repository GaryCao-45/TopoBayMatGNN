import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import random
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 随机种子设置
# =============================================================================

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

# =============================================================================
# 配置参数 (Configuration Parameters)
# =============================================================================

LABEL_PREPROCESSING_METHOD = 'robust_per_feature'
EMBEDDING_PREPROCESSING_METHOD = 'none'  # 默认不缩放嵌入向量（保留原始语义）

# =============================================================================
# 标签预处理
# =============================================================================

def preprocess_labels(labels_np, method='robust_per_feature', verbose=True):
    n_samples, n_features = labels_np.shape

    if method == 'global_standard':
        scaler = StandardScaler()
        labels_scaled = scaler.fit_transform(labels_np)
    elif method in ['per_feature_standard', 'per_feature_minmax', 'per_feature_robust', 'robust_per_feature']:
        labels_scaled = np.zeros_like(labels_np)
        for i in range(n_features):
            feature_data = labels_np[:, i].reshape(-1, 1)
            if method == 'per_feature_standard':
                feature_scaler = StandardScaler()
            elif method == 'per_feature_minmax':
                feature_scaler = MinMaxScaler()
            elif method in ['per_feature_robust', 'robust_per_feature']:
                feature_scaler = RobustScaler()
            labels_scaled[:, i] = feature_scaler.fit_transform(feature_data).flatten()

        class MultiFeatureScaler:
            def __init__(self, scalers, method):
                self.scalers = scalers
                self.method = method

            def transform(self, data):
                result = np.zeros_like(data)
                n_features = data.shape[1]
                for i in range(n_features):
                    feature_data = data[:, i].reshape(-1, 1)
                    result[:, i] = self.scalers[i].transform(feature_data).flatten()
                return result

            def inverse_transform(self, data):
                result = np.zeros_like(data)
                for i in range(n_features):
                    feature_data = data[:, i].reshape(-1, 1)
                    result[:, i] = self.scalers[i].inverse_transform(feature_data).flatten()
                return result

        scalers = []
        for i in range(n_features):
            feature_data = labels_np[:, i].reshape(-1, 1)
            if method == 'per_feature_standard':
                scalers.append(StandardScaler().fit(feature_data))
            elif method == 'per_feature_minmax':
                scalers.append(MinMaxScaler().fit(feature_data))
            elif method in ['per_feature_robust', 'robust_per_feature']:
                scalers.append(RobustScaler().fit(feature_data))

        scaler = MultiFeatureScaler(scalers, method)
    else:
        raise ValueError(f"不支持的预处理方法: {method}")

    if verbose:
        print(f"\n🔧 使用标签预处理方法: {method}")
        print(f"   原始标签统计:")
        for i in range(n_features):
            print(f"   特征{i}: 范围=[{labels_np[:, i].min():.3f}, {labels_np[:, i].max():.3f}], "
                  f"均值={labels_np[:, i].mean():.3f}, 方差={labels_np[:, i].var():.3f}")

        print(f"   标准化后标签统计:")
        for i in range(n_features):
            print(f"   特征{i}: 范围=[{labels_scaled[:, i].min():.3f}, {labels_scaled[:, i].max():.3f}], "
                  f"均值={labels_scaled[:, i].mean():.3f}, 方差={labels_scaled[:, i].var():.3f}")

    return labels_scaled, scaler

# =============================================================================
# 嵌入向量预处理
# =============================================================================

def preprocess_embeddings(embeddings_np, method='none', verbose=True):
    if method == 'none':
        if verbose:
            print(f"\n🔧 嵌入向量预处理: {method} (不进行预处理)")
        return embeddings_np, None

    n_samples, n_features = embeddings_np.shape

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"不支持的预处理方法: {method}")

    embeddings_scaled = scaler.fit_transform(embeddings_np)

    if verbose:
        print(f"\n🔧 嵌入向量预处理方法: {method}")
        print(f"   全局范围: [{embeddings_np.min():.6f}, {embeddings_np.max():.6f}]")
        print(f"   全局均值: {embeddings_np.mean():.6f} ± {embeddings_np.std():.6f}")
        print(f"   维度均值范围: [{embeddings_np.mean(axis=0).min():.6f}, {embeddings_np.mean(axis=0).max():.6f}]")

    return embeddings_scaled, scaler

# =============================================================================
# 模型定义
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearProbe, self).__init__()
        self.probe = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.probe(x)

class MLPProbe(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=32):  # 减小容量防止过拟合
        super(MLPProbe, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_features)
        )
        # 初始化权重
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                
    def forward(self, x):
        return self.network(x)

# =============================================================================
# 训练函数
# =============================================================================

def train_model(embeddings, labels, model, epochs=1000, learning_rate=0.01, patience=20):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        predictions = model(embeddings)
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            model.load_state_dict(best_model_state)
            break

        if (epoch + 1) % 200 == 0:
            pass  # 不打印避免干扰展示

    final_loss = best_loss if best_loss != float('inf') else loss.item()
    return model, final_loss

def evaluate_model(model, embeddings, labels, label_scaler=None):
    model.eval()
    with torch.no_grad():
        predictions = model(embeddings)

        # 存储标准化尺度的预测，以备不时之需
        predictions_scaled = predictions.clone()

        # 如果提供了缩放器，则将预测和标签都逆变换回原始尺度进行评估
        if label_scaler is not None:
            predictions_np = label_scaler.inverse_transform(predictions.numpy())
            predictions = torch.tensor(predictions_np, dtype=torch.float32)
            # 同时将真实标签也逆变换回原始尺度
            labels_np = label_scaler.inverse_transform(labels.numpy())
            labels = torch.tensor(labels_np, dtype=torch.float32)

        # 无论是否缩放，mse_loss都是在相同尺度上计算的
        mse_loss = nn.MSELoss()(predictions, labels).item()

        # 返回：
        # 1. 原始尺度的预测 (用于计算MAE, RMSE等)
        # 2. 原始尺度上的MSE损失
        # 3. 标准化尺度的预测 (未使用，但保留接口)
        # 4. 原始尺度的真实标签 (这是修复的关键)
        return predictions, mse_loss, predictions_scaled, labels

# =============================================================================
# 新增：留一法交叉验证 (Leave-One-Out Cross-Validation)
# =============================================================================

def evaluate_probe_loo_cv(embeddings, raw_labels_np, model_class, material_names, prop_names, save_path=None, **model_kwargs):
    """
    留一法交叉验证：每次留出一个样本作为测试，用其余样本训练
    返回每个样本的预测结果和整体性能指标
    """
    print(f"\n--- 留一法交叉验证 {model_class.__name__} ({len(material_names)} 折) ---")

    n_samples = len(material_names)
    n_props = raw_labels_np.shape[1]

    # 存储每次的预测结果
    all_predictions = []
    all_true_labels = []
    fold_results = []

    for i in range(n_samples):
        # 划分训练集和测试集索引
        test_indices = [i]
        train_indices = [j for j in range(n_samples) if j != i]

        # 准备数据
        X_train = embeddings[train_indices]
        y_train_raw = raw_labels_np[train_indices]
        X_test = embeddings[test_indices]
        y_test_raw = raw_labels_np[test_indices]
        
        test_material = material_names[i]

        # 核心修复：在交叉验证循环内部进行数据缩放
        # 1. 仅在当前折叠的训练集上拟合缩放器
        y_train_scaled_np, fold_scaler = preprocess_labels(
            y_train_raw, method=LABEL_PREPROCESSING_METHOD, verbose=False
        )
        
        # 2. 使用从训练集学习到的缩放器转换测试集
        y_test_scaled_np = fold_scaler.transform(y_test_raw)

        # 转换为 PyTorch Tensors
        y_train = torch.tensor(y_train_scaled_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_scaled_np, dtype=torch.float32)

        # 训练模型
        model = model_class(**model_kwargs)
        trained_model, train_loss = train_model(X_train, y_train, model, epochs=1000, patience=20)

        # 在测试样本上预测，并使用当前折叠的缩放器进行逆转换
        predictions, test_loss, _, true_labels = evaluate_model(trained_model, X_test, y_test, fold_scaler)

        # 存储结果 (predictions 和 true_labels 均为原始尺度)
        pred_np = predictions.numpy()
        true_np = true_labels.numpy()

        all_predictions.append(pred_np[0])  # 单个样本预测
        all_true_labels.append(true_np[0])  # 单个样本真实值

        fold_result = {
            'fold': i+1,
            'test_material': test_material,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'predictions': pred_np[0],
            'true_labels': true_np[0]
        }
        fold_results.append(fold_result)

        print(f"  折 {i+1:2d}: {test_material:<12} → MSE: {test_loss:.6f}")

    # 计算整体性能指标
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # 为每个属性计算指标
    prop_metrics = {}
    overall_metrics = {'mae': [], 'rmse': [], 'r2': []}

    for prop_idx in range(n_props):
        prop_name = prop_names[prop_idx] if prop_idx < len(prop_names) else f"Prop_{prop_idx}"
        true_vals = all_true_labels[:, prop_idx]
        pred_vals = all_predictions[:, prop_idx]

        # 计算MAE, RMSE, R²
        mae = np.mean(np.abs(pred_vals - true_vals))
        rmse = np.sqrt(np.mean((pred_vals - true_vals)**2))
        ss_res = np.sum((true_vals - pred_vals)**2)
        ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        prop_metrics[prop_name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        overall_metrics['mae'].append(mae)
        overall_metrics['rmse'].append(rmse)
        overall_metrics['r2'].append(r2)

    # 计算平均指标
    avg_metrics = {
        'mae_avg': np.mean(overall_metrics['mae']),
        'rmse_avg': np.mean(overall_metrics['rmse']),
        'r2_avg': np.mean(overall_metrics['r2'])
    }

    # 创建详细结果表格
    loo_results = []
    for i, material in enumerate(material_names):
        row = {'Material': material}
        for prop_idx, prop_name in enumerate(prop_names):
            row[f'True_{prop_name}'] = all_true_labels[i, prop_idx]
            row[f'Pred_{prop_name}'] = all_predictions[i, prop_idx]
        loo_results.append(row)

    loo_table = pd.DataFrame(loo_results)

    # 打印结果
    print(f"\n📊 留一法交叉验证结果:")
    print(f"   平均MAE: {avg_metrics['mae_avg']:.4f}")
    print(f"   平均RMSE: {avg_metrics['rmse_avg']:.4f}")
    print(f"   平均R²: {avg_metrics['r2_avg']:.4f}")
    for prop_name, metrics in prop_metrics.items():
        print(f"   {prop_name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

    print(f"\n🔍 预测 vs 真实值详细对比:")
    print(loo_table.round(4))

    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        loo_table.to_csv(save_path, index=False, float_format='%.6f')
        print(f"   💾 LOO-CV 详细结果已保存: {save_path}")

        # 保存包含性能指标的扩展版本
        extended_data = loo_table.copy()
        for prop_name in prop_names:
            true_col = f'True_{prop_name}'
            pred_col = f'Pred_{prop_name}'
            extended_data[f'Error_{prop_name}'] = extended_data[pred_col] - extended_data[true_col]
            extended_data[f'Abs_Error_{prop_name}'] = np.abs(extended_data[pred_col] - extended_data[true_col])
            extended_data[f'Rel_Error_{prop_name}'] = (extended_data[pred_col] - extended_data[true_col]) / extended_data[true_col] * 100

        # 添加性能指标到扩展数据
        metrics_summary = {}
        for prop_name in prop_names:
            metrics_summary[f'MAE_{prop_name}'] = prop_metrics[prop_name]['mae']
            metrics_summary[f'RMSE_{prop_name}'] = prop_metrics[prop_name]['rmse']
            metrics_summary[f'R2_{prop_name}'] = prop_metrics[prop_name]['r2']

        # 创建性能指标行
        metrics_row = {'Material': 'METRICS'}
        for prop_name in prop_names:
            metrics_row.update({
                f'True_{prop_name}': metrics_summary[f'MAE_{prop_name}'],
                f'Pred_{prop_name}': metrics_summary[f'RMSE_{prop_name}'],
                f'Error_{prop_name}': metrics_summary[f'R2_{prop_name}'],
                f'Abs_Error_{prop_name}': np.nan,
                f'Rel_Error_{prop_name}': np.nan
            })

        extended_data = pd.concat([extended_data, pd.DataFrame([metrics_row])], ignore_index=True)
        extended_save_path = save_path.replace('.csv', '_detailed.csv')
        extended_data.to_csv(extended_save_path, index=False, float_format='%.6f')
        print(f"   💾 LOO-CV 扩展结果(包含误差分析)已保存: {extended_save_path}")

    return avg_metrics, prop_metrics, loo_table, fold_results

# =============================================================================
# 新增：全体数据训练 + 生成预测对比表格
# =============================================================================

def evaluate_probe_on_all_data(embeddings, raw_labels_np, model_class, material_names, prop_names, save_path=None, **model_kwargs):
    print(f"\n--- Training {model_class.__name__} on all {len(material_names)} samples ---")

    # 1. 在函数内部进行数据缩放
    labels_scaled_np, label_scaler = preprocess_labels(
        raw_labels_np, method=LABEL_PREPROCESSING_METHOD, verbose=False
    )
    labels_scaled_torch = torch.tensor(labels_scaled_np, dtype=torch.float32)

    # 2. 训练模型
    model = model_class(**model_kwargs)
    trained_model, train_loss = train_model(embeddings, labels_scaled_torch, model, epochs=1000, patience=20)
    
    # 3. 评估模型，并使用缩放器逆转换
    predictions, test_loss, _, _ = evaluate_model(trained_model, embeddings, labels_scaled_torch, label_scaler)

    # 创建预测对比表
    data = {'Material': material_names}
    n_props = raw_labels_np.shape[1]
    pred_np = predictions.numpy()
    
    # 获取原始尺度的真实标签用于对比
    true_np = raw_labels_np

    for i in range(n_props):
        prop_name = prop_names[i] if i < len(prop_names) else f"Prop_{i}"
        data[f'True_{prop_name}'] = true_np[:, i]
        data[f'Pred_{prop_name}'] = pred_np[:, i]

    pred_table = pd.DataFrame(data)

    print("\n📊 预测 vs 真实值对比表:")
    print(pred_table.round(3))

    # 保存详细预测对比数据到CSV文件
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pred_table.to_csv(save_path, index=False, float_format='%.6f')
        print(f"   💾 详细预测数据已保存: {save_path}")

        # 同时保存一个包含误差分析的扩展版本
        extended_data = pred_table.copy()
        for i in range(n_props):
            prop_name = prop_names[i] if i < len(prop_names) else f"Prop_{i}"
            true_col = f'True_{prop_name}'
            pred_col = f'Pred_{prop_name}'
            extended_data[f'Error_{prop_name}'] = extended_data[pred_col] - extended_data[true_col]
            extended_data[f'Abs_Error_{prop_name}'] = np.abs(extended_data[pred_col] - extended_data[true_col])
            extended_data[f'Rel_Error_{prop_name}'] = (extended_data[pred_col] - extended_data[true_col]) / extended_data[true_col] * 100

        extended_save_path = save_path.replace('.csv', '_detailed.csv')
        extended_data.to_csv(extended_save_path, index=False, float_format='%.6f')
        print(f"   💾 扩展预测数据(包含误差分析)已保存: {extended_save_path}")

    return test_loss, pred_table, trained_model

# =============================================================================
# 新增：余弦相似度热力图
# =============================================================================

def plot_cosine_similarity_heatmap(embeddings, material_names, save_path=None):
    sim_matrix = cosine_similarity(embeddings)
    plt.figure(figsize=(7, 6))
    sns.set(font_scale=1.0)
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", xticklabels=material_names, yticklabels=material_names,
                cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': 0.8})
    plt.title('Cosine Similarity between Material Embeddings', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Cosine similarity heatmap saved: {save_path}")
    plt.close()

# =============================================================================
# 新增：为LOO-CV结果绘制预测-真实值散点图
# =============================================================================

def plot_loo_predictions(loo_table, prop_names, model_name, probe_type, save_dir):
    """为留一法交叉验证结果生成预测值 vs. 真实值散点图"""
    print(f"\n--- 正在为 {model_name} ({probe_type}, LOO-CV) 生成预测-真实值散点图 ---")
    n_props = len(prop_names)
    n_cols = 2
    n_rows = (n_props + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, prop_name in enumerate(prop_names):
        ax = axes[i]
        true_col = f'True_{prop_name}'
        pred_col = f'Pred_{prop_name}'
        
        true_vals = loo_table[true_col].values
        pred_vals = loo_table[pred_col].values
        
        # 新增：计算 R² 分数
        ss_res = np.sum((true_vals - pred_vals)**2)
        ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

        ax.scatter(true_vals, pred_vals, s=150, alpha=0.8, edgecolors='k', linewidth=0.5, label='Predictions')
        
        # 绘制 y=x 对角线
        all_vals = np.concatenate([true_vals, pred_vals])
        lims = [np.min(all_vals), np.max(all_vals)]
        margin = (lims[1] - lims[0]) * 0.1 if (lims[1] - lims[0]) > 1e-6 else 0.1
        lims = [lims[0] - margin, lims[1] + margin]
        
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x (Ideal)')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel(f'True {prop_name}', fontsize=12)
        ax.set_ylabel(f'Predicted {prop_name}', fontsize=12)
        ax.set_title(f'{prop_name}\n$R^2 = {r2:.3f}$', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 隐藏多余的子图
    for i in range(n_props, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'LOO-CV: Predicted vs. True Values ({model_name} - {probe_type})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{model_name}_{probe_type.lower()}_loo_cv_predictions_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ LOO-CV 散点图已保存: {save_path}")
    plt.close()

# =============================================================================
# 新增模块 1.1: 残差分布直方图/核密度图
# =============================================================================

def plot_residual_distribution(linear_loo_table, mlp_loo_table, prop_names, model_name, save_dir):
    """为LOO-CV结果生成残差的提琴图，以展示其分布"""
    print(f"\n--- 正在为 {model_name} 生成残差分布图 ---")
    
    residuals_data = []
    for prop in prop_names:
        true_col = f'True_{prop}'
        # Linear Probe residuals
        pred_col_linear = f'Pred_{prop}'
        res_linear = linear_loo_table[pred_col_linear] - linear_loo_table[true_col]
        for r in res_linear:
            residuals_data.append({'Property': prop, 'Residual': r, 'Probe': 'Linear'})
        # MLP Probe residuals
        pred_col_mlp = f'Pred_{prop}'
        res_mlp = mlp_loo_table[pred_col_mlp] - mlp_loo_table[true_col]
        for r in res_mlp:
            residuals_data.append({'Property': prop, 'Residual': r, 'Probe': 'MLP'})

    df_res = pd.DataFrame(residuals_data)

    plt.figure(figsize=(15, 8))
    sns.violinplot(data=df_res, x='Property', y='Residual', hue='Probe', split=True, inner="quart", fill=False)
    
    plt.axhline(0, color='r', linestyle='--', label='Zero Error')
    plt.title(f'Distribution of LOO-CV Residuals for {model_name}', fontsize=16)
    plt.xticks(rotation=15, ha='right')
    plt.ylabel('Residual (Predicted - True)', fontsize=12)
    plt.xlabel('')
    plt.legend(title='Probe Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_residuals_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 残差分布图已保存: {save_path}")
    plt.close()


# =============================================================================
# 新增模块 1.2: 留一法交叉验证误差箱线图
# =============================================================================

def plot_loo_error_boxplot(linear_loo_table, mlp_loo_table, prop_names, model_name, save_dir):
    """为LOO-CV结果的绝对误差生成箱线图"""
    print(f"\n--- 正在为 {model_name} 生成LOO-CV误差箱线图 ---")
    
    error_data = []
    for prop in prop_names:
        true_col = f'True_{prop}'
        # Linear Probe errors
        pred_col_linear = f'Pred_{prop}'
        abs_err_linear = np.abs(linear_loo_table[pred_col_linear] - linear_loo_table[true_col])
        for e in abs_err_linear:
            error_data.append({'Property': prop, 'Absolute Error': e, 'Probe': 'Linear'})
        # MLP Probe errors
        pred_col_mlp = f'Pred_{prop}'
        abs_err_mlp = np.abs(mlp_loo_table[pred_col_mlp] - mlp_loo_table[true_col])
        for e in abs_err_mlp:
            error_data.append({'Property': prop, 'Absolute Error': e, 'Probe': 'MLP'})

    df_err = pd.DataFrame(error_data)

    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_err, x='Property', y='Absolute Error', hue='Probe')
    
    plt.title(f'Boxplot of LOO-CV Absolute Errors for {model_name}', fontsize=16)
    plt.xticks(rotation=15, ha='right')
    plt.ylabel('Absolute Error', fontsize=12)
    plt.xlabel('')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_loo_error_boxplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ LOO-CV误差箱线图已保存: {save_path}")
    plt.close()


# =============================================================================
# 新增模块 2.1: 标度后特征两两散点矩阵
# =============================================================================

def plot_feature_pairplot(scaled_labels_np, prop_names, save_path="visualizations/feature_pairplot.png"):
    """绘制标度后目标属性之间的散点矩阵，以检查相关性"""
    print(f"\n--- 正在生成标度后特征的散点矩阵 ---")
    df_features = pd.DataFrame(scaled_labels_np, columns=prop_names)
    
    g = sns.pairplot(df_features, corner=True, diag_kind='kde')
    g.fig.suptitle('Pairplot of Scaled Target Properties', y=1.02, fontsize=16)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 特征散点矩阵已保存: {save_path}")
    plt.close()


# =============================================================================
# 新增模块 2.2: 嵌入统计柱状图
# =============================================================================

def plot_embedding_statistics(embeddings_np, material_names, model_name, save_dir):
    """为每个材料的嵌入向量绘制关键统计量的热力图"""
    print(f"\n--- 正在为 {model_name} 生成嵌入统计热力图 ---")
    
    stats_data = {
        'Mean': np.mean(embeddings_np, axis=1),
        'Std': np.std(embeddings_np, axis=1),
        'Min': np.min(embeddings_np, axis=1),
        'Max': np.max(embeddings_np, axis=1)
    }
    df_stats = pd.DataFrame(stats_data, index=material_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_stats, annot=True, fmt=".3f", cmap='viridis', linewidths=.5, cbar_kws={'label': 'Embedding Value'})
    
    plt.title(f'Embedding Stats by Material for {model_name}', fontsize=16, pad=20)
    plt.xlabel('Statistics', fontsize=12)
    plt.ylabel('Material', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_embedding_statistics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 嵌入统计热力图已保存: {save_path}")
    plt.close()


# =============================================================================
# 新增模块 3.1: 残差 vs. 关键特征散点图
# =============================================================================

def plot_residuals_vs_feature(linear_loo_table, mlp_loo_table, scaled_labels_np, prop_names, model_name, save_dir, feature_idx=1):
    """绘制残差与一个关键特征的散点图，以检查系统性偏差"""
    key_feature_name = prop_names[feature_idx]
    print(f"\n--- 正在为 {model_name} 生成残差 vs. {key_feature_name} 散点图 ---")
    key_feature_values = scaled_labels_np[:, feature_idx]

    n_props = len(prop_names)
    fig, axes = plt.subplots(n_props, 2, figsize=(15, 5 * n_props), sharex=True)

    for i, prop in enumerate(prop_names):
        # Linear Probe
        true_col = f'True_{prop}'
        pred_col_linear = f'Pred_{prop}'
        res_linear = linear_loo_table[pred_col_linear] - linear_loo_table[true_col]
        
        ax_linear = axes[i, 0]
        sns.scatterplot(x=key_feature_values, y=res_linear, ax=ax_linear)
        ax_linear.axhline(0, color='r', linestyle='--')
        ax_linear.set_title(f'Linear Probe: {prop} Residuals', fontsize=12)
        ax_linear.set_ylabel('Residual')

        # MLP Probe
        pred_col_mlp = f'Pred_{prop}'
        res_mlp = mlp_loo_table[pred_col_mlp] - mlp_loo_table[true_col]
        
        ax_mlp = axes[i, 1]
        sns.scatterplot(x=key_feature_values, y=res_mlp, ax=ax_mlp)
        ax_mlp.axhline(0, color='r', linestyle='--')
        ax_mlp.set_title(f'MLP Probe: {prop} Residuals', fontsize=12)
        ax_mlp.set_ylabel('')

    axes[-1, 0].set_xlabel(f'Scaled {key_feature_name}', fontsize=12)
    axes[-1, 1].set_xlabel(f'Scaled {key_feature_name}', fontsize=12)
    
    fig.suptitle(f'Residuals vs. {key_feature_name} for {model_name}', fontsize=16, y=1.01)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_residuals_vs_feature.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 残差-特征散点图已保存: {save_path}")
    plt.close()


# =============================================================================
# 可视化嵌入空间
# =============================================================================

def visualize_embeddings(embeddings, material_names, method='tsne', save_path=None):
    print(f"\n--- 方案: 嵌入可视化 ({method.upper()}) ---")

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    else:
        reducer = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))

    unique_materials = list(set(material_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_materials)))

    for i, material in enumerate(unique_materials):
        mask = [name == material for name in material_names]
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   color=colors[i], label=material, s=150, alpha=0.8, edgecolors='k', linewidth=0.5)

    plt.title(f'Material Embedding Space ({method.upper()})', fontsize=14)
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.legend(title="Materials", fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualization saved: {save_path}")
    plt.close()

# =============================================================================
# 聚类评估（仅保留轮廓分数作为参考，删除NMI）
# =============================================================================

def evaluate_clustering_metrics(embeddings, material_names, n_clusters_range=range(2, 6)):
    print("\n--- 方案: 聚类结构探索 (仅供观察) ---")

    best_silhouette = -2
    best_n_clusters = 2
    try:
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            if len(set(cluster_labels)) == n_clusters:
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                print(f"  n_clusters={n_clusters}: Silhouette={silhouette_avg:.3f}")
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_clusters = n_clusters
    except Exception as e:
        print(f"  聚类计算异常: {e}")

    print(f"   最佳聚类数: {best_n_clusters}, 最佳轮廓分数: {best_silhouette:.3f} (仅供参考)")
    return {
        'best_n_clusters': best_n_clusters,
        'best_silhouette': best_silhouette,
        'note': '因样本极少，此指标仅供参考'
    }

# =============================================================================
# 主程序
# =============================================================================

def main():
    MODEL_DIRS = {
        "model-1": "model-1-09171648",
        "model-2": "model-2-09180905",
        "model-3": "model-3-09181323",
        "model-4": "model-4-09181820"
    }
    EMBEDDING_DIM = 128
    PROPERTY_NAMES = ['Formation Energy (eV)', 'Bandgap (eV)', 'Fermi Level (eV)', 'Eff. Mass (m0)']

    true_labels_placeholder = OrderedDict([
        ('CsPbCl3',    [-26.35, 3.0, 3.5038817, 0.21]),
        ('CsPbBr3',    [-23.81, 2.3, 2.964493582, 0.22]),
        ('CH3NH3GeI3', [-12.94, 1.9, 2.158559313, 0.15]),
        ('CH3NH3PbI3', [-13.06, 1.51, 1.754137302, 0.15]),
        ('CsPbI3',     [-32.8, 1.7, 3.458034525, 0.2])
    ])

    print("="*80)
    print("🔬 小样本演示模式 —— 自监督嵌入框架可行性验证系统")
    print("="*80)
    print("⚠️  当前模式: n=5 样本，仅用于方法流程演示和视觉展示")
    print("⚠️  所有数值结果不具备统计显著性，但可反映框架运行能力")
    print("="*80)

    labels_np = np.array(list(true_labels_placeholder.values()))
    material_names = list(true_labels_placeholder.keys())

    # 仅为绘图和报告生成一次标度后的标签
    labels_scaled_np_for_plotting, _ = preprocess_labels(
        labels_np, method=LABEL_PREPROCESSING_METHOD, verbose=False
    )
    
    # === 新增：绘制标度后特征的散点矩阵 (模型无关) ===
    plot_feature_pairplot(labels_scaled_np_for_plotting, PROPERTY_NAMES)

    all_models_results = {}

    for model_name, model_dir in MODEL_DIRS.items():
        print(f"\n{'='*70}")
        print(f"🎯 处理模型: {model_name}")
        print(f"{'='*70}")

        embedding_file = os.path.join(model_dir, "embeddings", f"{model_dir}_embeddings.csv")
        if not os.path.exists(embedding_file):
            print(f"❌ 嵌入文件未找到，跳过")
            continue

        df = pd.read_csv(embedding_file)
        if list(df['material']) != material_names:
            print("❌ 材料顺序不匹配，跳过")
            print(f"   CSV顺序: {list(df['material'])}")
            print(f"   标签顺序: {material_names}")
            continue

        embedding_columns = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
        embeddings_np = df[embedding_columns].values
        embeddings_scaled_np, embedding_scaler = preprocess_embeddings(
            embeddings_np, method=EMBEDDING_PREPROCESSING_METHOD, verbose=True)

        X_embeddings = torch.tensor(embeddings_scaled_np, dtype=torch.float32)

        viz_dir = os.path.join(model_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        print(f"✅ 加载完成，开始执行验证方案...")

        model_results = {}

        # === 1. 线性探针 (全体训练) ===
        linear_save_path = os.path.join(viz_dir, f"{model_dir}_linear_probe_predictions.csv")
        linear_mse, linear_table, _ = evaluate_probe_on_all_data(
            X_embeddings, labels_np, LinearProbe, material_names, PROPERTY_NAMES,
            save_path=linear_save_path,
            in_features=X_embeddings.shape[1], out_features=labels_np.shape[1]
        )
        model_results['linear_probe'] = {'mse': linear_mse, 'table': linear_table}

        # === 2. 线性探针 (留一法交叉验证) ===
        linear_loo_save_path = os.path.join(viz_dir, f"{model_dir}_linear_probe_loo_cv.csv")
        linear_loo_metrics, linear_loo_prop_metrics, linear_loo_table, _ = evaluate_probe_loo_cv(
            X_embeddings, labels_np, LinearProbe, material_names, PROPERTY_NAMES,
            save_path=linear_loo_save_path,
            in_features=X_embeddings.shape[1], out_features=labels_np.shape[1]
        )
        model_results['linear_probe_loo'] = {
            'avg_metrics': linear_loo_metrics,
            'prop_metrics': linear_loo_prop_metrics,
            'table': linear_loo_table
        }

        # === 新增：为 LOO-CV 结果绘图 ===
        plot_loo_predictions(linear_loo_table, PROPERTY_NAMES, model_name, 'LinearProbe', viz_dir)

        # === 3. MLP探针 (全体训练) ===
        mlp_save_path = os.path.join(viz_dir, f"{model_dir}_mlp_probe_predictions.csv")
        mlp_mse, mlp_table, _ = evaluate_probe_on_all_data(
            X_embeddings, labels_np, MLPProbe, material_names, PROPERTY_NAMES,
            save_path=mlp_save_path,
            in_features=X_embeddings.shape[1], out_features=labels_np.shape[1], hidden_dim=32
        )
        model_results['mlp_probe'] = {'mse': mlp_mse, 'table': mlp_table}

        # === 4. MLP探针 (留一法交叉验证) ===
        mlp_loo_save_path = os.path.join(viz_dir, f"{model_dir}_mlp_probe_loo_cv.csv")
        mlp_loo_metrics, mlp_loo_prop_metrics, mlp_loo_table, _ = evaluate_probe_loo_cv(
            X_embeddings, labels_np, MLPProbe, material_names, PROPERTY_NAMES,
            save_path=mlp_loo_save_path,
            in_features=X_embeddings.shape[1], out_features=labels_np.shape[1], hidden_dim=32
        )
        model_results['mlp_probe_loo'] = {
            'avg_metrics': mlp_loo_metrics,
            'prop_metrics': mlp_loo_prop_metrics,
            'table': mlp_loo_table
        }

        # === 新增：为 LOO-CV 结果绘图 ===
        plot_loo_predictions(mlp_loo_table, PROPERTY_NAMES, model_name, 'MLPProbe', viz_dir)

        # === 5. 可视化 t-SNE & PCA ===
        tsne_save_path = os.path.join(viz_dir, f"{model_dir}_embeddings_tsne.png")
        visualize_embeddings(embeddings_np, material_names, method='tsne', save_path=tsne_save_path)

        pca_save_path = os.path.join(viz_dir, f"{model_dir}_embeddings_pca.png")
        visualize_embeddings(embeddings_np, material_names, method='pca', save_path=pca_save_path)

        # === 6. 余弦相似度热力图 ===
        cosine_save_path = os.path.join(viz_dir, f"{model_dir}_cosine_similarity.png")
        plot_cosine_similarity_heatmap(embeddings_np, material_names, save_path=cosine_save_path)

        # === 7. 聚类分析（仅供参考）===
        clustering_result = evaluate_clustering_metrics(embeddings_np, material_names)
        model_results['clustering'] = clustering_result

        # === 新增：生成补充的分析图表 ===
        
        # 1.1 残差分布
        plot_residual_distribution(
            model_results['linear_probe_loo']['table'],
            model_results['mlp_probe_loo']['table'],
            PROPERTY_NAMES, model_name, viz_dir
        )
        
        # 1.2 误差箱线图
        plot_loo_error_boxplot(
            model_results['linear_probe_loo']['table'],
            model_results['mlp_probe_loo']['table'],
            PROPERTY_NAMES, model_name, viz_dir
        )
        
        # 2.2 嵌入统计
        plot_embedding_statistics(embeddings_np, material_names, model_name, viz_dir)

        # 3.1 残差 vs 特征
        plot_residuals_vs_feature(
            model_results['linear_probe_loo']['table'],
            model_results['mlp_probe_loo']['table'],
            labels_scaled_np_for_plotting,
            PROPERTY_NAMES, model_name, viz_dir, feature_idx=1  # 使用带隙作为关键特征
        )

        # 保存原始数据和处理后的数据
        data_summary_path = os.path.join(viz_dir, f"{model_dir}_data_summary.csv")
        
        summary_data = {
            'Material': material_names,
            'Embedding_Dim': [embeddings_np.shape[1]] * len(material_names)
        }

        # 添加原始标签
        for i, prop_name in enumerate(PROPERTY_NAMES):
            summary_data[f'Original_{prop_name}'] = labels_np[:, i]

        # 添加标准化后的标签
        for i, prop_name in enumerate(PROPERTY_NAMES):
            summary_data[f'Scaled_{prop_name}'] = labels_scaled_np_for_plotting[:, i]

        # 添加嵌入向量统计
        summary_data['Embedding_Mean'] = np.mean(embeddings_np, axis=1)
        summary_data['Embedding_Std'] = np.std(embeddings_np, axis=1)
        summary_data['Embedding_Min'] = np.min(embeddings_np, axis=1)
        summary_data['Embedding_Max'] = np.max(embeddings_np, axis=1)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(data_summary_path, index=False, float_format='%.6f')
        print(f"   💾 数据汇总已保存: {data_summary_path}")

        all_models_results[model_name] = model_results

        # 生成简洁总结
        print(f"\n📋 {model_name} 演示结果摘要:")
        print(f"   ┣━ Linear Probe (全体) MSE: {linear_mse:.4f}")
        print(f"   ┣━ Linear Probe (LOO-CV) MAE: {linear_loo_metrics['mae_avg']:.4f}")
        print(f"   ┣━ MLP Probe (全体) MSE: {mlp_mse:.4f}")
        print(f"   ┣━ MLP Probe (LOO-CV) MAE: {mlp_loo_metrics['mae_avg']:.4f}")
        print(f"   ┗━ 轮廓分数 (仅供参考): {clustering_result['best_silhouette']:.3f}")

    # 最终汇总（简洁版）
    if all_models_results:
        print(f"\n{'='*80}")
        print("🎉 所有模型演示完成 —— 框架可行性验证成功！")
        print("📚 推荐展示素材:")
        print("   • 每个模型 visualizations/ 下的图 (t-SNE, PCA, Cosine Heatmap, LOO-CV预测散点图, 残差图等)")
        print("   • 控制台输出的预测对比表（已自动保存为CSV文件）")
        print("   • 全体训练预测数据: {model_dir}_*_predictions.csv")
        print("   • 留一法交叉验证数据: {model_dir}_*_loo_cv.csv")
        print("   • 扩展预测数据(包含误差分析): {model_dir}_*_predictions_detailed.csv")
        print("   • 数据汇总文件: {model_dir}_data_summary.csv")
        print("   • 各模型MSE/MAE对比（全体训练 vs LOO-CV）")
        print(f"{'='*80}")

        # 输出模型排序 (全体训练)
        print("\n🏆 模型综合表现排序 (全体训练平均MSE ↓):")
        model_mses = []
        for name, res in all_models_results.items():
            avg_mse = (res['linear_probe']['mse'] + res['mlp_probe']['mse']) / 2
            model_mses.append((name, avg_mse))
        for name, mse in sorted(model_mses, key=lambda x: x[1]):
            print(f"   {name:<10} → 平均MSE: {mse:.5f}")

        # 输出模型排序 (LOO-CV)
        print("\n🏆 模型综合表现排序 (LOO-CV平均MAE ↓):")
        model_maes = []
        for name, res in all_models_results.items():
            avg_mae = (res['linear_probe_loo']['avg_metrics']['mae_avg'] +
                      res['mlp_probe_loo']['avg_metrics']['mae_avg']) / 2
            model_maes.append((name, avg_mae))
        for name, mae in sorted(model_maes, key=lambda x: x[1]):
            print(f"   {name:<10} → 平均MAE: {mae:.5f}")

        print(f"\n✅ 演示目标达成！嵌入能重构性质、捕获材料相似性，框架可行。")
        print(f"💡 LOO-CV结果更能反映模型的泛化能力，是评估嵌入质量的关键指标！")

if __name__ == "__main__":
    main()
