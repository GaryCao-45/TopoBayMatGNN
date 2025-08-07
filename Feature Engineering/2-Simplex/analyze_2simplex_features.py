
import pandas as pd
import numpy as np
import sys
import os

def analyze_2simplex_features():
    """
    Loads 2-Simplex feature CSVs for CsPbBr3, CsPbI3, and CsPbCl3,
    calculates and prints statistical summaries for "D" section features.
    """
    output_filename = "2_simplex_analysis_report.md"
    
    with open(output_filename, 'a', encoding='utf-8') as f:
        f.write("--- 2-Simplex Feature Analysis Report (Committee Review Standard) ---\n\n")

        cs_pb_br3_path = os.path.join("对照实验", "CsPbBr3", "CsPbBr3-supercell-2-Simplex-Features.csv")
        cs_pb_i3_path = os.path.join("对照实验", "CsPbI3", "CsPbI3-supercell-2-Simplex-Features.csv")
        cs_pb_cl3_path = os.path.join("对照实验", "CspbCl3", "CsPbCl3-supercell-2-Simplex-Features.csv")

        # Define the key 2-Simplex features to analyze across all sections
        key_2simplex_features = [
            # A组：几何特征 (3个)
            "triangle_area",
            "bond_angle_variance",
            "triangle_shape_factor",

            # B组：跨层级派生特征 (12个)
            "avg_atomic_incompatibility",
            "variance_atomic_incompatibility",
            "avg_bader_charge",
            "variance_bader_charge",
            "avg_vectorial_asymmetry",
            "max_vectorial_asymmetry",
            "avg_bond_alignment",
            "variance_bond_alignment",
            "avg_bond_gradient",
            "max_bond_gradient",
            "avg_bond_distance",
            "variance_bond_distance",

            # C组：量子化学特征 (3个, 在几何重心处评估)
            "geometric_centroid_density",
            "geometric_centroid_elf",
            "geometric_centroid_laplacian_of_density",

            # D组：代数融合特征 (5个)
            "structural_tensor_product_trace",
            "total_density_gradient_flux",
            "point_group_reduction_factor",
            "structure_tensor_normal_projection",
            "hierarchical_stress_flow"
        ]

        # Analyze CsPbBr3
        try:
            df_br3 = pd.read_csv(cs_pb_br3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_br3_path)} (CsPbBr3, Pm3m) with {len(df_br3)} triangles detected.\n")
            f.write("\nStatistical Summary for CsPbBr3 (Pm3m) - D Section Features:\n")
            for feature in key_2simplex_features:
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
            f.write(f"Error: CsPbBr3 file not found at {os.path.basename(cs_pb_br3_path)}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbBr3 analysis: {e}\n")

        # Analyze CsPbI3
        try:
            df_i3 = pd.read_csv(cs_pb_i3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_i3_path)} (CsPbI3, Pnma) with {len(df_i3)} triangles detected.\n")
            f.write("\nStatistical Summary for CsPbI3 (Pnma) - D Section Features:\n")
            for feature in key_2simplex_features:
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
            f.write(f"Error: CsPbI3 file not found at {os.path.basename(cs_pb_i3_path)}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbI3 analysis: {e}\n")

        # Analyze CsPbCl3
        try:
            df_cl3 = pd.read_csv(cs_pb_cl3_path)
            f.write(f"Analyzing {os.path.basename(cs_pb_cl3_path)} (CsPbCl3, Pm3m) with {len(df_cl3)} triangles detected.\n")
            f.write("\nStatistical Summary for CsPbCl3 (Pm3m) - D Section Features:\n")
            for feature in key_2simplex_features:
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
            f.write(f"Error: CsPbCl3 file not found at {os.path.basename(cs_pb_cl3_path)}\n")
        except Exception as e:
            f.write(f"An error occurred during CsPbCl3 analysis: {e}\n")

        f.write("--- End of Report ---\n\n")

if __name__ == "__main__":
    analyze_2simplex_features()