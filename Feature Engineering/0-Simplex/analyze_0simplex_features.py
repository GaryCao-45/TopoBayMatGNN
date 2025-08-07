import pandas as pd
import numpy as np
import sys
import os

def analyze_0simplex_features():
    """
    Loads 0-Simplex feature CSVs for CsPbBr3, CsPbI3, and CsPbCl3,
    calculates and prints statistical summaries for key features.
    """
    output_filename = "0_simplex_analysis_report.md"
    
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write("--- 0-Simplex Feature Analysis Report (Committee Review Standard) ---\n")

        # 请根据您的实际文件路径进行修改
        cs_pb_br3_path = os.path.join("对照实验", "CsPbBr3", "CsPbBr3-supercell-0-Simplex-Features.csv")
        cs_pb_i3_path = os.path.join("对照实验", "CsPbI3", "CsPbI3-supercell-0-Simplex-Features.csv")
        cs_pb_cl3_path = os.path.join("对照实验", "CspbCl3", "CsPbCl3-supercell-0-Simplex-Features.csv")

        # 定义要分析的0-Simplex关键特征
        # 选择了一些代表性的A, B, C, D组特征
        key_0simplex_features = [
            # A组：基础物理化学特征
            "atomic_number",
            "electronegativity",
            "bond_valence_sum", # 新增的BVS

            # B组：量子化学特征
            "bader_charge",
            "local_dos_fermi",
            
            # C组：几何特征
            "vectorial_asymmetry_norm_sq",
            "mean_squared_neighbor_distance",
            "symmetry_breaking_quotient",

            # D组：融合特征
            "structure_chemistry_incompatibility",
            "charge_weighted_local_size"
        ]

        # Analyze CsPbBr3
        try:
            df_br3 = pd.read_csv(cs_pb_br3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_br3_path)} (CsPbBr3, Pm3m) with {len(df_br3)} atoms detected.\n")
            f.write("\nStatistical Summary for CsPbBr3 (Pm3m) - Key 0-Simplex Features:\n")
            for feature in key_0simplex_features:
                if feature in df_br3.columns:
                    series = df_br3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        f.write(f"    Variance: {series.var():.6f}\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbBr3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError:
            f.write(f"Error: CsPbBr3 0-Simplex file not found at {cs_pb_br3_path}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbBr3 0-Simplex analysis: {e}\n")

        # Analyze CsPbI3
        try:
            df_i3 = pd.read_csv(cs_pb_i3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_i3_path)} (CsPbI3, Pnma) with {len(df_i3)} atoms detected.\n")
            f.write("\nStatistical Summary for CsPbI3 (Pnma) - Key 0-Simplex Features:\n")
            for feature in key_0simplex_features:
                if feature in df_i3.columns:
                    series = df_i3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        f.write(f"    Variance: {series.var():.6f}\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbI3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError:
            f.write(f"Error: CsPbI3 0-Simplex file not found at {cs_pb_i3_path}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbI3 0-Simplex analysis: {e}\n")

        # Analyze CsPbCl3
        try:
            df_cl3 = pd.read_csv(cs_pb_cl3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_cl3_path)} (CsPbCl3, Pm3m) with {len(df_cl3)} atoms detected.\n")
            f.write("\nStatistical Summary for CsPbCl3 (Pm3m) - Key 0-Simplex Features:\n")
            for feature in key_0simplex_features:
                if feature in df_cl3.columns:
                    series = df_cl3[feature].dropna()
                    if not series.empty:
                        f.write(f"  Feature: {feature}\n")
                        f.write(f"    Mean: {series.mean():.6f}\n")
                        f.write(f"    Variance: {series.var():.6f}\n")
                        f.write(f"    Min: {series.min():.6f}\n")
                        f.write(f"    Max: {series.max():.6f}\n")
                    else:
                        f.write(f"  Feature: {feature} - No valid data found.\n")
                else:
                    f.write(f"  Feature: {feature} - Column not found in CsPbCl3 data.\n")
            f.write("\n" + "="*80 + "\n\n")
        except FileNotFoundError:
            f.write(f"Error: CsPbCl3 0-Simplex file not found at {cs_pb_cl3_path}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbCl3 0-Simplex analysis: {e}\n")

        f.write("--- End of Report ---\n")

if __name__ == "__main__":
    analyze_0simplex_features()