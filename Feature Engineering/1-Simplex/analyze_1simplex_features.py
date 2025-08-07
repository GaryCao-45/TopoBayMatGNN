import pandas as pd
import numpy as np
import os
import sys

def analyze_1simplex_features():
    """
    Loads 1-Simplex feature CSVs for CsPbBr3, CsPbI3, and CsPbCl3,
    calculates and prints statistical summaries for key features.
    """
    output_filename = "1_simplex_analysis_report.md"
    
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write("--- 1-Simplex Feature Analysis Report (Committee Review Standard) ---\n")

        # 请根据您的实际文件路径进行修改
        # 假设1-Simplex特征文件位于 '对照实验/CsPbBr3/' 和 '对照实验/CsPbI3/' 目录下
        cs_pb_br3_path = os.path.join("对照实验", "CsPbBr3", "CsPbBr3-supercell-1-Simplex-Features.csv")
        cs_pb_i3_path = os.path.join("对照实验", "CsPbI3", "CsPbI3-supercell-1-Simplex-Features.csv")
        cs_pb_cl3_path = os.path.join("对照实验", "CspbCl3", "CsPbCl3-supercell-1-Simplex-Features.csv")

        # 定义要分析的1-Simplex关键特征
        # 选择了一些代表性的A, B, C, D组特征
        key_1simplex_features = [
            # A组：几何与拓扑特征
            "bond_distance",
            "site1_coord_num",
            "site2_coord_num",

            # B组：派生特征 (基于0-Simplex特征)
            "delta_electronegativity",
            "delta_ionic_radius",
            "avg_bader_charge",

            # C组：量子化学特征 (修正后的无缺失值特征)
            "bond_midpoint_density",
            "bond_density_laplacian",
            "bond_midpoint_elf",
            "bond_charge_transfer",

            # D组：深度融合特征 (受五大代数思想启发)
            "lie_algebra_incompatibility",
            "quotient_algebra_orbit_size",
            "pseudo_symplectic_coupling",
            "tensor_algebraic_environment_alignment",
            "delta_structure_chemistry_incompatibility",
            "bond_ends_anisotropy_mismatch"
        ]

        # Analyze CsPbBr3
        try:
            if not os.path.exists(cs_pb_br3_path):
                f.write(f"Error: 文件不存在: {os.path.basename(cs_pb_br3_path)}\n")
                raise FileNotFoundError(f"文件不存在: {cs_pb_br3_path}")
            df_br3 = pd.read_csv(cs_pb_br3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_br3_path)} (CsPbBr3, Pm3m) with {len(df_br3)} bonds detected.\n")
            f.write("\nStatistical Summary for CsPbBr3 (Pm3m) - Key 1-Simplex Features:\n")
            for feature in key_1simplex_features:
                if feature in df_br3.columns:
                    series = df_br3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        f.write(f"    Variance: {series.var():.6f}\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found after dropping NaNs.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbBr3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError as e:
            f.write(f"An error occurred during CsPbBr3 1-Simplex analysis: {e}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbBr3 1-Simplex analysis: {e}\n")

        # Analyze CsPbI3
        try:
            if not os.path.exists(cs_pb_i3_path):
                f.write(f"Error: 文件不存在: {os.path.basename(cs_pb_i3_path)}\n")
                raise FileNotFoundError(f"文件不存在: {cs_pb_i3_path}")
            df_i3 = pd.read_csv(cs_pb_i3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_i3_path)} (CsPbI3, Pnma) with {len(df_i3)} bonds detected.\n")
            f.write("\nStatistical Summary for CsPbI3 (Pnma) - Key 1-Simplex Features:\n")
            for feature in key_1simplex_features:
                if feature in df_i3.columns:
                    series = df_i3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        f.write(f"    Variance: {series.var():.6f}\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found after dropping NaNs.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbI3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError as e:
            f.write(f"An error occurred during CsPbI3 1-Simplex analysis: {e}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbI3 1-Simplex analysis: {e}\n")

        # Analyze CsPbCl3
        try:
            if not os.path.exists(cs_pb_cl3_path):
                f.write(f"Error: 文件不存在: {os.path.basename(cs_pb_cl3_path)}\n")
                raise FileNotFoundError(f"文件不存在: {cs_pb_cl3_path}")
            df_cl3 = pd.read_csv(cs_pb_cl3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_cl3_path)} (CsPbCl3, Pm3m) with {len(df_cl3)} bonds detected.\n")
            f.write("\nStatistical Summary for CsPbCl3 (Pm3m) - Key 1-Simplex Features:\n")
            for feature in key_1simplex_features:
                if feature in df_cl3.columns:
                    series = df_cl3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        f.write(f"    Variance: {series.var():.6f}\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found after dropping NaNs.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbCl3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError as e:
            f.write(f"An error occurred during CsPbCl3 1-Simplex analysis: {e}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbCl3 1-Simplex analysis: {e}\n")

        f.write("--- End of Report ---\n")

if __name__ == "__main__":
    analyze_1simplex_features()
