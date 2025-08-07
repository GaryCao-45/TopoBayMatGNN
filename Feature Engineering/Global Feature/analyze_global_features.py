import pandas as pd
import numpy as np
import os
import sys

def analyze_global_features():
    """
    Loads Global Feature CSVs for CsPbBr3, CsPbI3, and CsPbCl3,
    calculates and prints statistical summaries for key global features.
    """
    output_filename = "global_analysis_report.md"
    
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write("--- Global Feature Analysis Report (Committee Review Standard) ---\n")

        # 请根据您的实际文件路径进行修改
        # 假设全局特征文件位于根目录，并以特定名称保存
        cs_pb_br3_path = os.path.join("对照实验", "CsPbBr3", "CsPbBr3-supercell-Global-Features.csv")
        cs_pb_i3_path = os.path.join("对照实验", "CsPbI3", "CsPbI3-supercell-Global-Features.csv")
        cs_pb_cl3_path = os.path.join("对照实验", "CspbCl3", "CsPbCl3-supercell-Global-Features.csv")

        # 定义要分析的全局关键特征
        # 选择了A, B, C, D组中代表性的特征
        key_global_features = [
            # A组：基础统计与几何特征
            "mean_bond_length",
            "volume_per_fu",
            "lattice_anisotropy_ratio",
            "bulk_anisotropy_index",
            "octahedral_count",
            "packing_fraction",

            # B组：DFT计算的基态属性
            "total_energy_per_atom",
            "fermi_level",
            "electrostatic_potential_mean",
            "electrostatic_potential_variance",

            # C组：全局高阶代数特征
            # "structure_hash", # 字符串类型，不进行统计分析
            "lie_asymmetry_magnitude_entropy", # 替换原有的 wyckoff_position_entropy
            "symmetry_orbit_connectivity",
            "global_asymmetry_norm",
            "force_covariance_invariant_1",
            "force_covariance_invariant_2",
            "total_torsional_stress",
            "field_density_coupling_invariant_1",
            "field_density_coupling_invariant_2",
            "total_gradient_norm",
            "log_pseudo_symplectic_fluctuation_volume", # 替换原有的 pseudo_symplectic_fluctuation_volume

            # D组：图相关路径特征 (通常取均值进行摘要)
            "path_cov_torsional_stress_mean",
            "path_entropy_mean",
            "path_chempot_diff_mean",
            "path_max_torque_mean",
            "path_curvature_mean",
            "path_wrapping_norm_mean",
            "path_force_gradient_mean",
            "path_structure_autocorr_mean",
            "path_charge_potential_cov_mean"
        ]

        # Analyze CsPbBr3
        try:
            if not os.path.exists(cs_pb_br3_path):
                f.write(f"Error: 文件不存在: {os.path.basename(cs_pb_br3_path)}\n")
                raise FileNotFoundError(f"文件不存在: {cs_pb_br3_path}")
            df_br3 = pd.read_csv(cs_pb_br3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_br3_path)} (CsPbBr3, Pm3m).\n")
            f.write("\nStatistical Summary for CsPbBr3 (Pm3m) - Key Global Features:\n")
            for feature in key_global_features:
                if feature in df_br3.columns:
                    series = df_br3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        if len(series) > 1: # 方差需要至少两个数据点
                            f.write(f"    Variance: {series.var():.6f}\n")
                        else:
                            f.write("    Variance: N/A (single data point)\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found after dropping NaNs.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbBr3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError as e:
            f.write(f"An error occurred during CsPbBr3 Global Features analysis: {e}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbBr3 Global Features analysis: {e}\n")

        # Analyze CsPbI3
        try:
            if not os.path.exists(cs_pb_i3_path):
                f.write(f"Error: 文件不存在: {os.path.basename(cs_pb_i3_path)}\n")
                raise FileNotFoundError(f"文件不存在: {cs_pb_i3_path}")
            df_i3 = pd.read_csv(cs_pb_i3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_i3_path)} (CsPbI3, Pnma).\n")
            f.write("\nStatistical Summary for CsPbI3 (Pnma) - Key Global Features:\n")
            for feature in key_global_features:
                if feature in df_i3.columns:
                    series = df_i3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        if len(series) > 1: # 方差需要至少两个数据点
                            f.write(f"    Variance: {series.var():.6f}\n")
                        else:
                            f.write("    Variance: N/A (single data point)\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found after dropping NaNs.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbI3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError as e:
            f.write(f"An error occurred during CsPbI3 Global Features analysis: {e}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbI3 Global Features analysis: {e}\n")

        # Analyze CsPbCl3
        try:
            if not os.path.exists(cs_pb_cl3_path):
                f.write(f"Error: 文件不存在: {os.path.basename(cs_pb_cl3_path)}\n")
                raise FileNotFoundError(f"文件不存在: {cs_pb_cl3_path}")
            df_cl3 = pd.read_csv(cs_pb_cl3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_cl3_path)} (CsPbCl3, Pm3m).\n")
            f.write("\nStatistical Summary for CsPbCl3 (Pm3m) - Key Global Features:\n")
            for feature in key_global_features:
                if feature in df_cl3.columns:
                    series = df_cl3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        if len(series) > 1: # 方差需要至少两个数据点
                            f.write(f"    Variance: {series.var():.6f}\n")
                        else:
                            f.write("    Variance: N/A (single data point)\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found after dropping NaNs.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbCl3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError as e:
            f.write(f"An error occurred during CsPbCl3 Global Features analysis: {e}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbCl3 Global Features analysis: {e}\n")

        f.write("--- End of Report ---\n")

if __name__ == "__main__":
    analyze_global_features()