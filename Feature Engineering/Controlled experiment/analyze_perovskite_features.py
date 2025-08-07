import os
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd

class PerovskiteFeatureCalculator:
    """
    钙钛矿材料特征计算器，用于从CIF文件提取几何结构特征。
    """
    
    def __init__(self, cif_file_path):
        """
        初始化计算器。
        
        Parameters:
        -----
        cif_file_path : str
            CIF 文件路径。
        """
        if not os.path.exists(cif_file_path):
            raise FileNotFoundError(f"CIF file not found: {cif_file_path}")
        self.cif_path = cif_file_path
        self.pmg_structure = Structure.from_file(cif_file_path)
        
    def calculate_geometric_features(self):
        """
        直接从CIF文件计算可获得的几何特征。
        """
        features = {}
        
        # 1. 晶格参数
        features['lattice_a'] = self.pmg_structure.lattice.a
        features['lattice_b'] = self.pmg_structure.lattice.b
        features['lattice_c'] = self.pmg_structure.lattice.c
        features['lattice_alpha'] = self.pmg_structure.lattice.alpha
        features['lattice_beta'] = self.pmg_structure.lattice.beta
        features['lattice_gamma'] = self.pmg_structure.lattice.gamma
        features['volume'] = self.pmg_structure.volume
        
        # 2. 对称性分析
        try:
            sga = SpacegroupAnalyzer(self.pmg_structure)
            features['space_group'] = sga.get_space_group_symbol()
            features['crystal_system'] = sga.get_crystal_system()
        except Exception as e:
            print(f"Warning: Could not determine space group for {self.cif_path}: {e}")
            features['space_group'] = 'N/A'
            features['crystal_system'] = 'N/A'

        # 3. 原子数量和化学式
        features['num_atoms'] = len(self.pmg_structure)
        features['chemical_formula'] = self.pmg_structure.formula
        features['reduced_formula'] = self.pmg_structure.composition.reduced_formula
        
        # 4. 键长、键角、配位数等（示例，需要更复杂的实现）
        # 实际项目中，这些需要更详细的遍历和计算逻辑
        # 这里仅作示意，未来可根据需求扩展
        
        # 对于八面体倾斜/扭曲的量化，可能需要更高级的分析
        # 例如，计算Pb-X-Pb键角偏离180度的程度，或PbX6八面体的畸变指数。
        # 这部分需要详细的数学定义，并确保其不变性。
        
        # 高级几何特征：键角和八面体畸变
        self._calculate_advanced_geometric_features(features)
        
        return features

    def _calculate_advanced_geometric_features(self, features):
        """
        计算 Pb-X-Pb 键角和八面体畸变指数。
        """
        import numpy as np # Ensure numpy is imported
        from pymatgen.analysis.local_env import CrystalNN
        # from pymatgen.analysis.bond_valence import BVAnalyzer # Not used in current logic for this part

        cnn = CrystalNN(distance_cutoffs=None)
        # bva = BVAnalyzer() # Not used for this part of calculation

        # Iterate through Pb sites, getting both the site object and its integer index
        pb_sites_with_indices = [(site, i) for i, site in enumerate(self.pmg_structure) if site.specie.symbol == "Pb"]
        
        all_pbx_bond_lengths = []
        all_pxp_angles = []
        all_octahedral_distortion_indices = []

        for pb_site, pb_index in pb_sites_with_indices:
            # 1. 获取 Pb 的配位环境
            try:
                neighbors_info = cnn.get_nn_info(self.pmg_structure, pb_index)
                
                # Filter for halide neighbors, storing their original index and site
                x_neighbors_info = [n for n in neighbors_info if str(n['site'].specie) in ["Br", "Cl", "I"]]

                if len(x_neighbors_info) == 6: # 确保是六配位八面体
                    # Pb-X 键长
                    pbx_bond_lengths = [pb_site.distance(n['site']) for n in x_neighbors_info]
                    all_pbx_bond_lengths.extend(pbx_bond_lengths)

                    # Calculate X-Pb-X bond angle (within the octahedron)
                    xpx_angles = []
                    for i in range(len(x_neighbors_info)):
                        for j in range(i + 1, len(x_neighbors_info)):
                            halide_neighbor1_index = x_neighbors_info[i]['site_index']
                            halide_neighbor2_index = x_neighbors_info[j]['site_index']
                            
                            angle = self.pmg_structure.get_angle(
                                int(halide_neighbor1_index),
                                int(pb_index), # Use pb_index directly
                                int(halide_neighbor2_index)
                            )
                            xpx_angles.append(angle)

                    # Calculate B-X-B angles (Pb-X-Pb angles)
                    # Iterate through the halide neighbors to find bridging halogens and their Pb neighbors
                    for x_neighbor_entry in x_neighbors_info: 
                        x_index_center = x_neighbor_entry['site_index'] 

                        # Find neighboring Pb atoms for this central X atom
                        x_center_nn_info = cnn.get_nn_info(self.pmg_structure, x_index_center)
                        pb_nn_of_x_center_info = [n for n in x_center_nn_info if str(n['site'].specie) == "Pb"]

                        if len(pb_nn_of_x_center_info) >= 2: # X should bridge at least two Pb atoms
                            for i in range(len(pb_nn_of_x_center_info)):
                                for j in range(i + 1, len(pb_nn_of_x_center_info)):
                                    pb_neighbor1_index = pb_nn_of_x_center_info[i]['site_index']
                                    pb_neighbor2_index = pb_nn_of_x_center_info[j]['site_index']
                                    
                                    angle = self.pmg_structure.get_angle(
                                        int(pb_neighbor1_index),
                                        int(x_index_center),
                                        int(pb_neighbor2_index)
                                    )
                                    all_pxp_angles.append(angle)

                    # Octahedral distortion index (variance of bond lengths)
                    if pbx_bond_lengths:
                        distortion_index_bl = np.var(pbx_bond_lengths)
                        all_octahedral_distortion_indices.append(distortion_index_bl)
                else:
                    print(f"Warning: Pb atom at index {pb_index} is not 6-coordinated (found {len(x_neighbors_info)} halide neighbors). Skipping advanced feature calculation for this site.")
            except Exception as e:
                print(f"Error processing Pb site at index {pb_index}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                continue

        # 平均键长和键长方差
        if all_pbx_bond_lengths:
            features['avg_pbx_bond_length'] = np.mean(all_pbx_bond_lengths)
            features['std_pbx_bond_length'] = np.std(all_pbx_bond_lengths)
        else:
            features['avg_pbx_bond_length'] = np.nan
            features['std_pbx_bond_length'] = np.nan

        # Pb-X-Pb 键角统计
        if all_pxp_angles:
            pxp_angles_np = np.array(all_pxp_angles)
            features['min_pxp_angle'] = np.min(pxp_angles_np)
            features['max_pxp_angle'] = np.max(pxp_angles_np)
            features['avg_pxp_angle'] = np.mean(pxp_angles_np)
            features['std_pxp_angle'] = np.std(pxp_angles_np)
            features['pxp_angle_deviation_from_180'] = np.mean(np.abs(pxp_angles_np - 180.0))
        else:
            features['min_pxp_angle'] = np.nan
            features['max_pxp_angle'] = np.nan
            features['avg_pxp_angle'] = np.nan
            features['std_pxp_angle'] = np.nan
            features['pxp_angle_deviation_from_180'] = np.nan

        # 平均八面体键长畸变指数
        if all_octahedral_distortion_indices:
            features['avg_octahedral_bond_length_distortion'] = np.mean(all_octahedral_distortion_indices)
        else:
            features['avg_octahedral_bond_length_distortion'] = np.nan

        # X-Pb-X 键角统计 (新增)
        if xpx_angles:
            features['avg_xpx_angle'] = np.mean(xpx_angles)
            features['std_xpx_angle'] = np.std(xpx_angles)
        else:
            features['avg_xpx_angle'] = np.nan
            features['std_xpx_angle'] = np.nan

def analyze_perovskites(cif_files):
    """
    分析多个钙钛矿CIF文件并提取特征。
    """
    all_features = []
    for cif_file in cif_files:
        print(f"Analyzing {cif_file}...")
        try:
            calculator = PerovskiteFeatureCalculator(cif_file)
            geometric_features = calculator.calculate_geometric_features()
            geometric_features['material'] = os.path.basename(cif_file).split('-supercell-optimized.cif')[0]
            all_features.append(geometric_features)
        except Exception as e:
            print(f"Error processing {cif_file}: {e}")
            continue
            
    df = pd.DataFrame(all_features)
    return df

if __name__ == "__main__":
    cif_paths = [
        "对照实验/CsPbBr3/CsPbBr3-supercell-optimized.cif",
        "对照实验/CsPbCl3/CsPbCl3-supercell-optimized.cif",
        "对照实验/CsPbI3/CsPbI3-supercell-optimized.cif",
    ]
    
    features_df = analyze_perovskites(cif_paths)
    print("\n--- Extracted Geometric Features ---")
    print(features_df.to_markdown(index=False))

    # 预期结果的讨论
    print("\n--- Expected Results Discussion ---")
    print("基于已知的晶体结构，我们预期：")
    print("1. CsPbBr3 (立方相) 的晶格参数 a, b, c 会非常接近，且alpha, beta, gamma角接近90度，空间群为 Pm3m 或 P1 (由于超胞优化可能降对称)。")
    print("2. CsPbCl3 和 CsPbI3 (正交相，Pnma) 的晶格参数 a, b, c 会有明显差异，alpha, beta, gamma角保持90度，空间群可能显示更低的对称性（例如 Pnma 或 P1）。")
    print("3. 对于捕获不对称性：我们的特征工程应能通过量化 Pb-X-Pb 键角畸变、八面体倾斜/扭曲等指标，清晰地分辨出 CsPbI3 (Pnma) 的结构不对称性，而 CsPbBr3 (理想立方) 则应表现出对称性。")
    print("   这需要进一步深入实现八面体倾斜和扭曲相关的特征计算。当前主要展示基本晶格几何特征。")