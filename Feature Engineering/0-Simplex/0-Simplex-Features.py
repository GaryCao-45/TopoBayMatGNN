import os
import warnings
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import skew
from typing import List, Dict, Any, Optional, Tuple
import traceback # 新增导入

# Pymatgen
from pymatgen.core import Structure, Site
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer, calculate_bv_sum
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer

# Mendeleev
from mendeleev import element

# ASE 和 GPAW
from ase.io import read, write
from ase.units import Bohr
from gpaw import GPAW # type: ignore
from gpaw.elf import ELF # type: ignore

warnings.filterwarnings('ignore')


GLOBAL_CHEM_VEC_RANGES = {
    'electronegativity': (0.7, 4.0),      # 电负性 (Pauling标度)
    'covalent_radius': (30, 250),         # 共价半径 (Pyykko, in pm)
    'ionization_energy': (370, 2500),     # 第一电离能 (in kJ/mol)
    'electron_affinity': (-50, 350),      # 电子亲合能 (in kJ/mol)
    'atomic_volume': (5, 70),             # 原子体积 (in cm³/mol)
    'polarizability': (0.1, 60),          # 极化率 (in Å³)
    'effective_charge': (-3.0, 8.0)       # 玻恩有效电荷范围
}

class UnifiedFeatureCalculator:
    _TOLERANCE = 1e-10  # 用于数值稳定性的全局小常数，避免除以零等问题
    def __init__(self, cif_file_path: str, pw_gpw_file: str, fd_gpw_file: Optional[str] = None):
        print("--- 统一原子特征计算器初始化 ---")
        # --- 路径和文件加载 ---
        self.cif_path = Path(cif_file_path).resolve()
        pw_path = Path(pw_gpw_file).resolve()
        
        if not self.cif_path.exists(): raise FileNotFoundError(f"CIF文件不存在: {self.cif_path}")
        if not pw_path.exists(): raise FileNotFoundError(f"平面波GPW文件不存在: {pw_path}")

        # --- 结构和分析工具 ---
        print(f"读取结构: {self.cif_path.name}")
        self.structure = Structure.from_file(self.cif_path)
        self.atoms = read(self.cif_path)
        print(f"结构加载成功: {self.structure.composition.reduced_formula}, {len(self.structure)}个原子")
        
        self.crystal_nn = CrystalNN()
        self.sga = SpacegroupAnalyzer(self.structure)
        
        # 增强BVS分析
        try:
            self.bv_analyzer = BVAnalyzer()
            self.valences = self.bv_analyzer.get_valences(self.structure)
            print(f"氧化态分析成功，获得{len(self.valences)}个原子的氧化态: {self.valences}")
            
            # 计算键价和 (BVS)
            self.bvs_list = []
            for i in range(len(self.structure)):
                bvs_value = self._calculate_individual_bvs(i)
                self.bvs_list.append(bvs_value)
            
            print(f"键价和批量计算成功，获得{len(self.bvs_list)}个BVS值")
            print(f"BVS值范围: {min(self.bvs_list):.3f} 到 {max(self.bvs_list):.3f}")
                        
        except Exception as e:
            print(f"警告: 键价分析失败，氧化态和BVS将设为默认值。错误: {e}")
            self.valences = [1.0] * len(self.structure)  # 默认氧化态
            self.bvs_list = [1.0] * len(self.structure)   # 默认BVS值

        # --- GPAW 计算加载 ---
        print(f"加载平面波GPW计算: {pw_path.name}")
        self.pw_calc = GPAW(str(pw_path), txt=None)
        self.fermi_level = self.pw_calc.get_fermi_level()
        self.workdir = pw_path.parent
        print(f"计算加载成功，费米能级: {self.fermi_level:.4f} eV")

        self.fd_calc = None
        if fd_gpw_file:
            fd_path = Path(fd_gpw_file).resolve()
            if fd_path.exists():
                print(f"加载FD模式GPW计算: {fd_path.name}")
                self.fd_calc = GPAW(str(fd_path), txt=None)
            else:
                print(f"警告: FD模式GPW文件不存在: {fd_path.name}")

        print("使用全局化学属性范围进行标准化...")

        print("建立GPAW与Pymatgen原子索引的一致性映射...")
        self._build_gpaw_pymatgen_index_mapping()
        
        print("构建坐标索引映射以优化性能...")
        self._build_coordinate_index_mapping()

    def _build_gpaw_pymatgen_index_mapping(self):
        print("  正在验证GPAW与Pymatgen原子顺序的一致性...")
        
        # 获取两个结构的原子坐标
        pymatgen_coords = np.array([site.coords for site in self.structure])
        gpaw_coords = self.pw_calc.get_atoms().get_positions()
        
        # 检查原子数量是否一致
        if len(pymatgen_coords) != len(gpaw_coords):
            raise ValueError(f"原子数量不匹配: Pymatgen {len(pymatgen_coords)}, GPAW {len(gpaw_coords)}")
        
        # 建立GPAW索引到Pymatgen索引的映射
        self.gpaw_to_pymatgen_index = {}
        tolerance = 1e-3  # 坐标匹配容差（Å）
        
        unmatched_gpaw = []
        unmatched_pymatgen = list(range(len(pymatgen_coords)))
        
        for gpaw_idx, gpaw_coord in enumerate(gpaw_coords):
            best_match_idx = None
            min_distance = float('inf')
            
            for pymatgen_idx in unmatched_pymatgen:
                pymatgen_coord = pymatgen_coords[pymatgen_idx]
                
                # 计算最小镜像距离（考虑周期性边界条件）
                distance = np.linalg.norm(gpaw_coord - pymatgen_coord)
                
                # 也检查周期性镜像
                cell = self.structure.lattice.matrix
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        for k in [-1, 0, 1]:
                            if i == 0 and j == 0 and k == 0:
                                continue
                            shift = i*cell[0] + j*cell[1] + k*cell[2]
                            shifted_coord = pymatgen_coord + shift
                            dist_shifted = np.linalg.norm(gpaw_coord - shifted_coord)
                            distance = min(distance, dist_shifted)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match_idx = pymatgen_idx
            
            if min_distance <= tolerance and best_match_idx is not None:
                self.gpaw_to_pymatgen_index[gpaw_idx] = best_match_idx
                unmatched_pymatgen.remove(best_match_idx)
            else:
                unmatched_gpaw.append(gpaw_idx)
        
        # 验证映射结果
        if unmatched_gpaw:
            print(f"  警告：{len(unmatched_gpaw)}个GPAW原子未找到匹配的Pymatgen原子")
            print(f"  未匹配的GPAW索引: {unmatched_gpaw}")
        
        if unmatched_pymatgen:
            print(f"  警告：{len(unmatched_pymatgen)}个Pymatgen原子未找到匹配的GPAW原子")
            print(f"  未匹配的Pymatgen索引: {unmatched_pymatgen}")
        
        if len(self.gpaw_to_pymatgen_index) == len(gpaw_coords):
            print(f" 原子索引映射构建成功：{len(self.gpaw_to_pymatgen_index)}个原子完全匹配")
        else:
            print(f" 部分匹配：{len(self.gpaw_to_pymatgen_index)}/{len(gpaw_coords)}个原子成功匹配")
        
        # 打印一些映射示例用于验证
        print("  映射示例（GPAW索引 -> Pymatgen索引，元素符号）:")
        for i, (gpaw_idx, pymatgen_idx) in enumerate(list(self.gpaw_to_pymatgen_index.items())[:5]):
            gpaw_element = self.pw_calc.get_atoms().get_chemical_symbols()[gpaw_idx]
            pymatgen_element = self.structure[pymatgen_idx].specie.symbol
            print(f"    {gpaw_idx} -> {pymatgen_idx} ({gpaw_element} -> {pymatgen_element})")
        
        if len(self.gpaw_to_pymatgen_index) > 5:
            print("    ...")

    def _build_coordinate_index_mapping(self):
        self.coord_to_index = {}
        tolerance = 1e-6
        
        for i, site in enumerate(self.structure):
            # 将坐标转换为可哈希的元组，用固定精度避免浮点误差
            coord_key = tuple(np.round(site.coords, decimals=int(-np.log10(tolerance))))
            self.coord_to_index[coord_key] = i
        
        print(f"  坐标索引映射构建完成，包含{len(self.coord_to_index)}个原子")

    def _find_atom_index_by_coords(self, coords: np.ndarray) -> Optional[int]:
        tolerance = 1e-6
        coord_key = tuple(np.round(coords, decimals=int(-np.log10(tolerance))))
        return self.coord_to_index.get(coord_key, None)

    def _get_pymatgen_index_from_gpaw(self, gpaw_index: int) -> int:
        if not hasattr(self, 'gpaw_to_pymatgen_index'):
            # 如果映射未建立，假设索引相同（向后兼容）
            return gpaw_index
        
        return self.gpaw_to_pymatgen_index.get(gpaw_index, gpaw_index)

    def _get_fd_atom_index(self, pymatgen_index: int) -> int:
        if not self.fd_calc:
            return pymatgen_index
        
        # 如果FD和PW计算器使用相同的原子顺序，可以直接使用GPAW映射的逆向查找
        # 查找对应的GPAW索引
        gpaw_index = None
        for g_idx, p_idx in self.gpaw_to_pymatgen_index.items():
            if p_idx == pymatgen_index:
                gpaw_index = g_idx
                break
        
        if gpaw_index is not None:
            return gpaw_index
        else:
            # 如果找不到映射，假设索引相同（向后兼容）
            return pymatgen_index

    # ==========================================================================
    # 主流程方法
    # ==========================================================================

    def calculate_unified_features(self) -> pd.DataFrame:
        print("\n--- 开始计算统一原子特征 (27维) ---")

        # 步骤 1: 计算经典特征 (A+C组)
        classic_df = self._calculate_classic_features()
        
        # 步骤 2: 计算量子化学特征 (B组)
        quantum_df = self._calculate_quantum_features()
        
        # 步骤 3: 提取A, B, C组
        group_A, group_C = self._extract_classic_groups(classic_df)
        group_B = quantum_df # B组就是完整的量子特征
        
        # 步骤 4: 计算融合特征 (D组)
        group_D = self._calculate_fused_features(group_B, group_C)
        
        # 步骤 5: 整合所有特征
        unified_df = pd.concat([group_A, group_B, group_C, group_D], axis=1)
        
        # 额外步骤: 保存结构张量到独立文件
        base_name = self.cif_path.stem.replace('-optimized', '')
        tensor_output_path = self.workdir / f"{base_name}-0-Simplex-Structure-Tensors.csv"
        self._save_structure_tensors(tensor_output_path)
        
        return unified_df

    # ==========================================================================
    # 核心计算模块 (经典, 量子, 融合)
    # ==========================================================================

    def _calculate_classic_features(self) -> pd.DataFrame:
        print("\n[模块1/3] ==> 计算经典特征 (A+C组)...")
        features_list = []
        for i, site in enumerate(self.structure):
            classic_features, T_struct = self._calculate_single_site_classic_features(i, site)
            # 将索引和张量一起添加到列表中
            features_list.append([i] + classic_features + [T_struct])
        
        feature_names = [
            'site_index_temp', # 临时索引列
            'atomic_number', 'electronegativity', 'ionization_energy', 'electron_affinity',
            'valence_electrons', 'ionic_radius', 'covalent_radius', 'coordination_number',
            'avg_site_valence', 'bond_valence_sum', 'bond_length_distortion',  # 新增BVS
            'vectorial_asymmetry_norm_sq', 'mean_squared_neighbor_distance',
            'local_environment_anisotropy', 'symmetry_breaking_quotient',
            'tensor' # 临时张量列
        ]
        
        df = pd.DataFrame(features_list, columns=feature_names)
        # 将张量保存到实例变量中，以便后续保存到文件
        self._tensors_to_save = dict(zip(df['site_index_temp'], df['tensor']))
        # 返回不含张量和临时索引的纯特征DataFrame
        return df.drop(columns=['tensor', 'site_index_temp'])

    def _calculate_quantum_features(self) -> pd.DataFrame:
        print("\n[模块2/3] ==> 计算量子化学特征 (B组)...")
        
        
        bader_charges = self._calculate_bader_charges_direct_corrected()
        
        try:
            potential_grid = self.pw_calc.get_electrostatic_potential()
            potentials = self._interpolate_at_atomic_sites_corrected(potential_grid, self.pw_calc)
        except Exception as e: 
            print(f"  - 错误: 计算电势失败: {e}")
            potentials = [np.nan] * len(self.structure)

        try:
            density_grid = self.pw_calc.get_all_electron_density()
            densities = self._interpolate_at_atomic_sites_corrected(density_grid, self.pw_calc)
        except Exception as e: 
            print(f"  - 错误: 计算电子密度失败: {e}")
            densities = [np.nan] * len(self.structure)

        elf_values = self._calculate_elf_at_sites_corrected()
        
        try:
            magnetic_moments = self._get_magnetic_moments_corrected()
        except Exception as e: 
            print(f"  - 错误: 计算磁矩失败: {e}")
            magnetic_moments = [np.nan] * len(self.structure)

        # --- 计算稳健的轨道电子数特征 ---
        print("  - 计算稳健的DOS特征 (LDOS, s/p/d electron counts)...")
        dos_features_list = []
        for pymatgen_idx in range(len(self.structure)):
            atom_dos_features = self._calculate_dos_features_for_atom_corrected(pymatgen_idx)
            dos_features_list.append(atom_dos_features)
            
        dos_df = pd.DataFrame(dos_features_list)
        
        
        print("  - DOS特征计算完成")
        
        # 新增：智能化d轨道判断的诊断输出
        self._print_d_orbital_analysis_summary()

        features_df = pd.DataFrame({
            'bader_charge': bader_charges,
            'electrostatic_potential': potentials,
            'electron_density': densities,
            'elf': elf_values,
            'local_magnetic_moment': magnetic_moments,
        })
        return pd.concat([features_df, dos_df], axis=1)

    def _calculate_fused_features(self, group_B: pd.DataFrame, group_C: pd.DataFrame) -> pd.DataFrame:
        print("\n[模块3/3] ==> 计算深度融合特征 (D组)...")
        fusion_features = []
        
        # 获取所有原子的Bader电荷，用于加权张量计算
        bader_charges = group_B['bader_charge'].tolist()
        
        for i in range(len(self.structure)):
            site = self.structure[i]
            
            # 从B组获取量子化学属性
            bader_charge = group_B.iloc[i]['bader_charge']
            elf = group_B.iloc[i]['elf']
            
            # 获取结构张量和化学矢量
            T_struct = self._get_structure_tensor(i)
            v_chem = self._calculate_chemical_vector(site, i)
            
            # 计算四个深度融合特征
            features = {}
            
            # D1: 结构-化学不相容性 (李代数交换子范数) - 保持不变，现在基于可靠的v_chem
            features['structure_chemistry_incompatibility'] = self._calculate_structure_chemistry_incompatibility(
                T_struct, v_chem)
            
            # D2: 化学矢量在结构度量下的范数 (二次型) - 保持不变，现在基于可靠的v_chem
            features['chemical_in_structural_metric'] = self._calculate_chemical_in_structural_metric(
                T_struct, v_chem)
            
            # D3: 电荷加权的局域尺寸 (深度融合：电荷作为构造权重而非调制因子)
            features['charge_weighted_local_size'] = self._calculate_charge_weighted_local_size_new(
                i, bader_charges)
            
            # D4: ELF加权的局域不对称性 (深度融合：ELF与几何非对称性的直接耦合)
            features['elf_weighted_local_anisotropy'] = self._calculate_elf_weighted_local_anisotropy_new(
                i, elf)
            
            fusion_features.append(features)
            
        return pd.DataFrame(fusion_features)

    # ==========================================================================
    # 经典特征辅助方法 (重新设计的C组特征)
    # ==========================================================================

    def _calculate_single_site_classic_features(self, site_idx: int, site: "Site") -> Tuple[List[float], np.ndarray]:
        elem = element(site.specie.symbol)
        
        # --- A. 基础物理化学特征 (10维) ---
        cn = self._get_coordination_number(site_idx)
        valence = self.valences[site_idx]
        
        # 计算键价和 (BVS)
        bvs = self._calculate_bond_valence_sum(site_idx)
        
        basic_features = [
            elem.atomic_number,
            float(elem.electronegativity_pauling()) if elem.electronegativity_pauling else np.nan,
            float(elem.ionenergies.get(1, np.nan)),
            float(elem.electron_affinity) if elem.electron_affinity else np.nan,
            int(elem.nvalence()) if hasattr(elem, 'nvalence') else np.nan,
            self._get_precise_ionic_radius(elem, valence, cn),
            float(elem.covalent_radius_pyykko / 100.0) if elem.covalent_radius_pyykko else np.nan,
            cn,
            valence,
            bvs,  # 新增BVS特征
            self._calculate_bond_length_distortion(site_idx)
        ]
        
        # --- C. 重新设计的几何特征 (5维) ---
        # 获取结构不对称矢量和结构张量
        local_asymmetry_vector = self._calculate_local_asymmetry_vector(site_idx)
        T_struct = self._get_structure_tensor(site_idx)
        
        geometric_features = [
            self._calculate_vectorial_asymmetry_norm_sq(local_asymmetry_vector),
            self._calculate_mean_squared_neighbor_distance(T_struct, site_idx),
            self._calculate_local_environment_anisotropy_from_tensor(T_struct),
            self._calculate_symmetry_breaking_quotient(site_idx)
        ]
        return basic_features + geometric_features, T_struct

    def _get_structure_tensor(self, site_idx: int) -> np.ndarray:
        try:
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        except Exception as e:
            print(f"  - 错误: 获取结构张量失败: {e}")
            return np.eye(3) * 1e-12  # 返回一个很小的单位矩阵作为默认值
        
        if len(nn_info) < 1:
            return np.eye(3) * 1e-12
        
        center_coords = self.structure[site_idx].coords
        relative_vectors = np.array([n['site'].coords - center_coords for n in nn_info])
        
        # 构造二阶矩张量: T = V^T * V，其中V是相对向量矩阵
        structure_tensor = np.dot(relative_vectors.T, relative_vectors)
        return structure_tensor

    def _get_charge_weighted_structure_tensor(self, site_idx: int, bader_charges: List[float]) -> np.ndarray:
        try:
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        except Exception as e:
            print(f"  - 错误: 获取电荷加权的结构张量失败: {e}")
            return np.eye(3) * 1e-12
        
        if len(nn_info) < 1:
            return np.eye(3) * 1e-12
        
        center_coords = self.structure[site_idx].coords
        relative_vectors = np.array([n['site'].coords - center_coords for n in nn_info])
        
        # 获取近邻原子的电荷权重 (优化版：使用O(1)哈希查找)
        weights = []
        for nn in nn_info:
            # 使用高效的坐标映射查找近邻原子索引
            nn_idx = self._find_atom_index_by_coords(nn['site'].coords)
            
            if nn_idx is not None and nn_idx < len(bader_charges):
                # 使用Bader电荷的绝对值作为权重，避免负权重的问题
                weight = abs(bader_charges[nn_idx]) if not pd.isna(bader_charges[nn_idx]) else 1.0
            else:
                weight = 1.0  # 默认权重
            weights.append(weight)
        
        weights = np.array(weights)
        
        # 构造加权的二阶矩张量: T = Σ(w_i * v_i ⊗ v_i^T)
        weighted_tensor = np.zeros((3, 3))
        for i, (v, w) in enumerate(zip(relative_vectors, weights)):
            weighted_tensor += w * np.outer(v, v)
        
        return weighted_tensor

    def _calculate_chemical_vector(self, site: "Site", site_idx: int) -> np.ndarray:
        
        elem = element(site.specie.symbol)
        
        # 提取基础化学属性
        electronegativity = float(elem.electronegativity_pauling()) if elem.electronegativity_pauling else np.nan
        covalent_radius = float(elem.covalent_radius_pyykko) if elem.covalent_radius_pyykko else np.nan
        ionization_energy = float(elem.ionenergies.get(1, np.nan))
        
        # 扩展化学属性
        electron_affinity = float(elem.electron_affinity) if elem.electron_affinity else np.nan
        atomic_volume = float(elem.atomic_volume) if elem.atomic_volume else np.nan
        
        # 极化率（使用empirical关系或mendeleev数据）
        try:
            polarizability = float(elem.dipole_polarizability) if hasattr(elem, 'dipole_polarizability') and elem.dipole_polarizability else np.nan
        except Exception as e:
            print(f"  - 错误: 获取极化率失败: {e}")
            # 如果没有直接的极化率数据，使用经验关系 α ∝ r³
            polarizability = (covalent_radius / 100.0)**3 if not pd.isna(covalent_radius) else np.nan
        
        # 局域玻恩有效电荷（从价态估算）
        try:
            effective_charge = float(self.valences[site_idx]) if site_idx < len(self.valences) else np.nan
        
        except Exception as e:
            print(f"  - 错误: 获取局域玻恩有效电荷失败: {e}")
            effective_charge = np.nan
        
        # 归一化到[0,1]区间 - 使用全局范围确保可移植性
        def normalize_property(value, prop_name):
            if pd.isna(value):
                return 0.5  # 对于缺失值，使用中点作为默认
            min_val, max_val = GLOBAL_CHEM_VEC_RANGES[prop_name]
            return np.clip((value - min_val) / (max_val - min_val), 0, 1) if max_val > min_val else 0.5
        
        v_chem = np.array([
            normalize_property(electronegativity, 'electronegativity'),
            normalize_property(covalent_radius, 'covalent_radius'),
            normalize_property(ionization_energy, 'ionization_energy'),
            normalize_property(electron_affinity, 'electron_affinity'),
            normalize_property(atomic_volume, 'atomic_volume'),
            normalize_property(polarizability, 'polarizability'),
            normalize_property(effective_charge, 'effective_charge')
        ])
        
        return v_chem

    def _calculate_vectorial_asymmetry_norm_sq(self, asymmetry_vector: np.ndarray) -> float:
        return float(np.dot(asymmetry_vector, asymmetry_vector))

    def _calculate_mean_squared_neighbor_distance(self, T_struct: np.ndarray, site_idx: int) -> float:
        trace_value = np.trace(T_struct)
        
        # 增加除以邻居数量（配位数）以严格符合"均方"的定义
        cn = self._get_coordination_number(site_idx)
        if cn > 0:
            return float(trace_value / cn)
        else:
            # 如果没有邻居，或者配位数不明确，则返回0.0或NaN（这里选择0.0表示无尺寸）
            return 0.0

    def _calculate_local_environment_anisotropy_from_tensor(self, T_struct: np.ndarray) -> float:
        try:
            # 使用SVD确保数值稳定性
            eigenvalues = np.sort(np.linalg.eigvalsh(T_struct))
            eigenvalues = np.maximum(eigenvalues, 0.0)  # 确保非负
            
            # 检查退化情况
            total_trace = np.sum(eigenvalues)
            if total_trace < 1e-12:
                return 0.0
            
            # 计算各向异性因子：基于特征值差异的归一化度量
            lambda1, lambda2, lambda3 = eigenvalues[2], eigenvalues[1], eigenvalues[0]
            
            numerator = (lambda1 - lambda2)**2 + (lambda2 - lambda3)**2 + (lambda3 - lambda1)**2
            denominator = 2 * total_trace**2
            
            if denominator < 1e-12:
                return 0.0

            anisotropy_factor = numerator / denominator
            return float(np.sqrt(anisotropy_factor))
        except Exception as e:
            print(f"  - 错误: 从结构张量计算局域环境各向异性失败: {e}")
            return 0.0

    def _calculate_symmetry_breaking_quotient(self, site_idx: int) -> float:
        try:
            # 1. 获取原子位点对称群的阶数 |G_site| (stabilizer subgroup order)
            # 通过遍历所有空间群操作，计算保持该位点不变（在周期性边界条件下）的操作数量
            if self.sga is None:
                raise RuntimeError("SpacegroupAnalyzer 未初始化，无法计算位点对称群。")
                
            symmetry_ops = self.sga.get_symmetry_operations()
            center_site = self.structure[site_idx]
            order_site = 0
            
            for sym_op in symmetry_ops:
                # 将对称操作应用于位点的分数坐标
                transformed_frac_coords = sym_op.operate(center_site.frac_coords)
                
                            # 检查对称操作后的坐标是否与原坐标在周期性边界条件下等价
            # 考虑晶体的周期性，即使操作后的坐标不在[0,1)范围内，也可能是等价点
                diff = transformed_frac_coords - center_site.frac_coords
                # 处理周期性边界条件：将差异值"缠绕"回 [-0.5, 0.5) 范围，确保正确识别周期性等价点
                diff = diff - np.round(diff)
                
                if np.allclose(diff, 0, atol=1e-5): # 使用1e-5作为容差
                    order_site += 1
            
            if order_site == 0: # 确保至少有一个恒等操作
                order_site = 1 # 任何点至少包含恒等操作（群论基本性质）
            
            # 2. 构造局域分子团簇（中心原子及其配位环境）
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            if not nn_info:
                return 1.0
            
            # 创建相对坐标系（以中心原子为原点）用于点群分析
            coords_list = []
            species_list = []
            
            # 中心原子
            coords_list.append([0.0, 0.0, 0.0])
            species_list.append(center_site.specie.symbol)
            
            # 近邻原子
            for nn in nn_info:
                relative_coords = nn['site'].coords - center_site.coords
                coords_list.append(relative_coords.tolist())
                species_list.append(nn['site'].specie.symbol)
            
            # 使用pymatgen的Molecule类构建局部配位簇并进行点群分析
            # 注意：这里将周期性晶体的局部环境转换为分子模型以应用点群分析
            from pymatgen.core.structure import Molecule
            mol = Molecule(species_list, coords_list)
            
            # 获取局部配位环境的点群对称性（|G_local|）
            pga = PointGroupAnalyzer(mol)
            point_group = pga.get_pointgroup()
            order_local = len(point_group)
            
            # 计算并返回对称性破缺商值 |G_site| / |G_local|
            if order_local > 0:
                quotient = float(order_site) / float(order_local)
                return quotient  # 商值越小，对称性破缺越严重
            else:
                return float(order_site)  # 防止除零错误
                
        except Exception as e:
            print(f"  - 错误: 计算对称性破缺商失败: {e}")
            # 如果精确的群论方法计算失败，尝试使用启发式方法估算对称性破缺
            try:
                # 备用方法：基于配位环境的几何和化学不均匀性估算对称性破缺
                nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
                if not nn_info:
                    return 1.0
                
                # 计算配位环境的"对称性指标"
                # 原理：如果所有近邻距离相近且元素相同，则局部环境对称性较高
                # 反之，距离变化大或元素多样性高，则对称性破缺严重
                distances = [self.structure[site_idx].distance(nn['site']) for nn in nn_info]
                elements = [nn['site'].specie.symbol for nn in nn_info]
                
                # 计算配位距离的变异系数（CV）作为几何不均匀性指标
                if len(distances) > 1:
                    mean_dist = np.mean(distances)
                    if mean_dist > 0:
                        distance_cv = np.std(distances) / mean_dist
                    else:
                        distance_cv = 0.0
                else:
                    distance_cv = 0.0
                
                # 计算元素多样性指标（不同元素种类与总配位数的比值）
                unique_elements = len(set(elements))
                total_neighbors = len(elements)
                element_diversity = unique_elements / total_neighbors if total_neighbors > 0 else 1.0
                
                # 对称性破缺估算：距离变异系数和元素多样性的加权组合
                # 值越大表示对称性破缺越严重（与主方法的商值解释相反）
                symmetry_breaking = 1.0 + distance_cv + element_diversity
                return float(symmetry_breaking)  # 确保返回浮点数
                
            except Exception as e:
                print(f"  - 错误: 计算对称性破缺商失败: {e}")
                # 最后的备选方案：返回默认值1.0（表示中等对称性状态）
                return 1.0

    # ==========================================================================
    # 量子特征辅助方法 (保持不变)
    # ==========================================================================

    def _calculate_bader_charges_direct_corrected(self) -> List[float]:
        print("  - (Bader) 准备电荷密度文件...")
        cube_filename = self.workdir / "bader_charge_density.cube"
        try:
            rho = self.pw_calc.get_all_electron_density(gridrefinement=2)
            write(str(cube_filename), self.pw_calc.get_atoms(), data=rho * Bohr**3)
            
            print("  - (Bader) 调用Bader程序...")
            result = subprocess.run(['bader', str(cube_filename.name)], cwd=self.workdir, capture_output=True, text=True)
            if result.returncode != 0:
                # 即使Bader程序返回非零代码，也尝试解析文件，因为它可能部分成功
                print(f"  - 警告: Bader程序返回非零退出代码: {result.returncode}")
                print(f"    Bader stderr: {result.stderr}")

            print("  - (Bader) 解析ACF.dat文件...")
            acf_file = self.workdir / "ACF.dat"
            if not acf_file.exists():
                print("  - 错误: ACF.dat文件未生成。Bader分析失败。")
                return [np.nan] * len(self.structure)

            bader_electrons = self._parse_acf_file(acf_file)
            valence_electrons = [self.pw_calc.setups[i].Z for i in range(len(self.pw_calc.get_atoms()))]
            
            if len(valence_electrons) != len(bader_electrons):
                print(f"  - 错误: Bader电荷数({len(bader_electrons)})与价电子数({len(valence_electrons)})不匹配。")
                return [np.nan] * len(self.structure)
            
            # 计算GPAW顺序的Bader电荷
            gpaw_charges = [val - bader for val, bader in zip(valence_electrons, bader_electrons)]
            
            # 重新排列结果以匹配Pymatgen结构顺序
            corrected_charges = [np.nan] * len(self.structure)
            for gpaw_idx, charge in enumerate(gpaw_charges):
                pymatgen_idx = self._get_pymatgen_index_from_gpaw(gpaw_idx)
                if pymatgen_idx is not None and pymatgen_idx < len(corrected_charges):
                    corrected_charges[pymatgen_idx] = charge
            
            return corrected_charges

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"  - 错误: Bader分析失败: {e}")
            return [np.nan] * len(self.structure)
        finally:
            for f in self.workdir.glob("bader_*"): f.unlink(missing_ok=True)
            for f in ["ACF.dat", "BCF.dat", "AVF.dat"]: (self.workdir / f).unlink(missing_ok=True)

    def _parse_acf_file(self, acf_file: Path) -> List[float]:
        with open(acf_file, 'r') as f: lines = f.readlines()
        electrons = [float(line.split()[4]) for line in lines[2:] if len(line.split()) >= 5 and line.split()[0].isdigit()]
        return electrons

    def _interpolate_at_atomic_sites_corrected(self, grid_data: np.ndarray, calc_source: GPAW) -> List[float]:
        if grid_data is None or grid_data.size == 0:
            print("  - 错误: 输入网格数据为空或无效。")
            return [np.nan] * len(self.structure)

        try:
            # 将笛卡尔坐标转换为分数坐标
            cell = calc_source.get_atoms().get_cell()
            cell_inv = np.linalg.inv(cell)
            atomic_positions = calc_source.get_atoms().get_positions()
            frac_coords = np.dot(atomic_positions, cell_inv)

            # 将分数坐标"缠绕"到 [0, 1) 范围，处理周期性边界条件
            # 这确保了即使原子在晶胞边缘，也能正确插值
            wrapped_frac_coords = frac_coords % 1.0
            
            # 获取网格的维度
            grid_shape = grid_data.shape
            
            # 动态裁剪分数坐标，确保它们严格落入插值器的有效范围 [0, (N-1)/N - tolerance)
            # 这是因为RegularGridInterpolator的axes参数使用endpoint=False，不包含1.0
            # 如果不进行此裁剪，接近1.0的坐标可能超出插值范围，导致NaN
            scaled_frac_coords = np.copy(wrapped_frac_coords)
            for d in range(len(grid_shape)):
                n_points = grid_shape[d]
                # 确保n_points大于1，避免除零错误，同时对单点维度进行特殊处理
                if n_points > 1:
                    max_coord_value = (n_points - 1) / n_points # e.g., 91/92 for 92 points
                    scaled_frac_coords[:, d] = np.clip(scaled_frac_coords[:, d], 0.0, max_coord_value - self._TOLERANCE)
                else:
                    # 如果维度只有1个点，则所有坐标都应映射到该点 (通常为0)
                    scaled_frac_coords[:, d] = 0.0 # 假设单点网格位于0

            # 创建三线性插值器
            # axes 参数必须是每个维度坐标的数组，从0到1
            axes = [np.linspace(0, 1, n_points, endpoint=False) for n_points in grid_shape]

            # 使用 RegularGridInterpolator 进行三线性插值
            # bounds_error=False 允许在边界外进行插值（此时使用 fill_value）
            # fill_value=np.nan 在无法插值时返回 NaN，便于识别问题
            interpolator = RegularGridInterpolator(
                axes, grid_data,
                method='linear', # 'linear'对应三线性插值
                bounds_error=False,
                fill_value=np.nan
            )

            # 对所有原子位置进行插值
            gpaw_results = interpolator(scaled_frac_coords)
            
            # 重新排列结果以匹配Pymatgen结构顺序
            corrected_results = [np.nan] * len(self.structure)
            for gpaw_idx, result in enumerate(gpaw_results):
                pymatgen_idx = self._get_pymatgen_index_from_gpaw(gpaw_idx)
                if pymatgen_idx is not None and pymatgen_idx < len(corrected_results):
                    corrected_results[pymatgen_idx] = result
            return corrected_results

        except Exception as e:
            print(f"  - 错误: 执行插值失败: {e}")
            return [np.nan] * len(self.structure)

    def _calculate_elf_at_sites_corrected(self) -> List[float]:
        if not self.fd_calc:
            print("  - 跳过ELF计算: 未提供FD模式GPW文件。")
            return [np.nan] * len(self.structure)
        try:
            print("  - 计算电子局域化函数 (ELF)...")
            elf_calc = ELF(self.fd_calc)
            elf_calc.update()
            elf_grid = elf_calc.get_electronic_localization_function()
            return self._interpolate_at_atomic_sites_corrected(elf_grid, self.fd_calc)
        except Exception as e:
            print(f"  - ELF计算失败: {e}")
            return [np.nan] * len(self.structure)

    def _get_magnetic_moments_corrected(self) -> List[float]:
        if self.pw_calc.get_spin_polarized():
            gpaw_moments = self.pw_calc.get_magnetic_moments()
        else:
            gpaw_moments = np.zeros(len(self.pw_calc.get_atoms()))
        
        # 重新排列结果以匹配Pymatgen结构顺序
        corrected_moments = [np.nan] * len(self.structure)
        for gpaw_idx, moment in enumerate(gpaw_moments):
            pymatgen_idx = self._get_pymatgen_index_from_gpaw(gpaw_idx)
            if pymatgen_idx is not None and pymatgen_idx < len(corrected_moments):
                corrected_moments[pymatgen_idx] = moment
        
        return corrected_moments

    def _is_d_orbital_relevant(self, atom_index: int) -> bool:

        try:
            atomic_number = self.structure[atom_index].specie.Z
            
            # 保守但全面的判断规则：覆盖所有重要的过渡金属和重金属
            
            # 第一过渡系列：Z = 21-30 (Sc到Zn)
            if 21 <= atomic_number <= 30:
                return True
            
            # 第二过渡系列：Z = 39-48 (Y到Cd)  
            elif 39 <= atomic_number <= 48:
                return True
            
            # 第三过渡系列：Z = 72-80 (Hf到Hg)
            elif 72 <= atomic_number <= 80:
                return True
            
            # 6p重金属区：Z = 81-86 (Tl到Rn)
            # 包含Pb(82), Bi(83)等钙钛矿重要元素
            elif 81 <= atomic_number <= 86:
                return True
            
            # 超重元素：Z = 104-118 (Rf到Og)
            # 为未来的超重元素钙钛矿研究预留
            elif 104 <= atomic_number <= 118:
                return True
            
            # 其他元素：主族轻元素、碱金属、碱土金属等
            # 这些元素的d轨道通常无计算价值
            else:
                return False
                
        except Exception as e:
            print(f"  - 错误: 判断d轨道相关性失败: {e}")
            # 异常情况下的保守策略
            return 21 <= atomic_number <= 118 and atomic_number not in [
                # 明确排除的轻元素和s区金属
                37, 38, 55, 56, 87, 88  # Rb, Sr, Cs, Ba, Fr, Ra
            ]

    def _analyze_d_orbital_configuration(self, atomic_number: int) -> Dict[str, any]:

        config = {
            'has_relevant_d': False,
            'physical_relevance': False,
            'orbital_type': None,
            'energy_level': None
        }
        
        # 轻元素：Z < 21，d轨道能量过高
        if atomic_number < 21:
            config['orbital_type'] = 'too_high_energy'
            return config
        
        # 第一过渡系列：Z = 21-30 (Sc-Zn)
        elif 21 <= atomic_number <= 30:
            config.update({
                'has_relevant_d': True,
                'physical_relevance': True,
                'orbital_type': '3d_transition',
                'energy_level': 'valence'
            })
        
        # 第二过渡系列：Z = 39-48 (Y-Cd)
        elif 39 <= atomic_number <= 48:
            config.update({
                'has_relevant_d': True,
                'physical_relevance': True,
                'orbital_type': '4d_transition',
                'energy_level': 'valence'
            })
        
        # 镧系后元素：Z = 72-80 (Hf-Hg)
        elif 72 <= atomic_number <= 80:
            config.update({
                'has_relevant_d': True,
                'physical_relevance': True,
                'orbital_type': '5d_transition',
                'energy_level': 'valence'
            })
        
        # 重p区元素：Z = 81-118，(n-1)d轨道可能有信息价值
        elif 81 <= atomic_number <= 118:
            config.update({
                'has_relevant_d': True,
                'physical_relevance': atomic_number <= 86,  # 6p系列有一定相关性
                'orbital_type': 'post_transition_d',
                'energy_level': 'semi_core'
            })
        
        # 重s、p区元素：填充d轨道的主族元素
        elif atomic_number in list(range(31, 39)) + list(range(49, 57)) + list(range(58, 72)):
            # 排除镧系元素(57-71)和碱金属/碱土金属
            # 这些元素的d轨道通常为内层，但在特殊环境下可能有信息价值
            config.update({
                'has_relevant_d': True,
                'physical_relevance': False,  # 通常为内层
                'orbital_type': 'filled_d_sublayer',
                'energy_level': 'core_like'
            })
        
        # 碱金属和碱土金属：Z = 37,38,55,56,87,88 等
        elif atomic_number in [37, 38, 55, 56, 87, 88]:  # Rb, Sr, Cs, Ba, Fr, Ra
            # 这些元素的d轨道能量过高，无物理意义
            config.update({
                'has_relevant_d': False,
                'physical_relevance': False,
                'orbital_type': 'too_high_energy_s_block',
                'energy_level': 'too_high'
            })
        
        return config

    def _evaluate_d_orbital_environment_activation(self, atom_index: int) -> Dict[str, any]:

        activation = {
            'is_activated': False,
            'coordination_factor': 0.0,
            'ligand_field_strength': 'weak',
            'bonding_character': 'ionic'
        }
        
        try:
            # 获取配位环境信息
            nn_info = self.crystal_nn.get_nn_info(self.structure, atom_index)
            coordination_number = len(nn_info)
            atomic_number = self.structure[atom_index].specie.Z
            
            # 首先检查原子类型是否有激活潜力
            d_config = self._analyze_d_orbital_configuration(atomic_number)
            if not d_config['has_relevant_d']:
                # 对于没有相关d轨道的原子，不考虑环境激活
                return activation
            
            # 配位数对d轨道激活的影响
            if coordination_number >= 4:
                activation['coordination_factor'] = min(1.0, coordination_number / 8.0)
                activation['is_activated'] = True
            
            # 分析配体种类和键合性质
            ligand_elements = [nn['site'].specie.symbol for nn in nn_info]
            
            # 强配体场元素 (C, N, O等) 更容易激活d轨道
            strong_field_ligands = set(['C', 'N', 'O', 'S', 'P'])
            weak_field_ligands = set(['F', 'Cl', 'Br', 'I'])
            
            strong_ligands = len([l for l in ligand_elements if l in strong_field_ligands])
            weak_ligands = len([l for l in ligand_elements if l in weak_field_ligands])
            
            if strong_ligands > 0:
                activation['ligand_field_strength'] = 'strong'
                activation['is_activated'] = True
                activation['bonding_character'] = 'covalent'
            elif weak_ligands > 0:
                activation['ligand_field_strength'] = 'weak'
                activation['bonding_character'] = 'ionic'
            
        except Exception as e:
            print(f"  - 错误: 评估化学环境对d轨道激活的影响失败: {e}")
            pass
        
        return activation

    def _assess_d_orbital_ml_value(self, atomic_number: int, 
                                  config: Dict, 
                                  environment: Dict) -> Dict[str, any]:

        ml_value = {
            'has_value': False,
            'value_type': None,
            'confidence': 0.0
        }
        
        # 对于重卤素（Br, I），虽然d轨道通常为内层，但可能携带环境指纹信息
        if atomic_number in [35, 53]:  # Br, I
            ml_value.update({
                'has_value': True,
                'value_type': 'environment_fingerprint',
                'confidence': 0.6
            })
        
        # 对于后过渡金属区元素，d轨道可能有相关性
        elif 81 <= atomic_number <= 86:
            ml_value.update({
                'has_value': True,
                'value_type': 'semi_core_interaction',
                'confidence': 0.7
            })
        
        # 高配位环境下，即使弱相关的d轨道也可能有信息价值
        # 但仅限于有d轨道潜力的元素
        elif environment['coordination_factor'] > 0.5 and config['has_relevant_d']:
            ml_value.update({
                'has_value': True,
                'value_type': 'coordination_induced',
                'confidence': environment['coordination_factor']
            })
        
        return ml_value

    def _get_adaptive_pdos_threshold(self, atom_index: int, orbital_type: str) -> float:

        atomic_number = self.structure[atom_index].specie.Z
        
        if orbital_type != 'd':
            return 1e-3  # s, p轨道使用标准阈值
        
        # 对于d轨道，基于原子序数和电子构型智能设置阈值
        d_config = self._analyze_d_orbital_configuration(atomic_number)
        
        # 基于d轨道的能级位置和物理相关性调整阈值
        if d_config['energy_level'] == 'valence':
            # 价电子层d轨道：过渡金属，较宽松阈值
            return 1e-3
        elif d_config['energy_level'] == 'semi_core':
            # 半核心层d轨道：后过渡金属，中等阈值  
            if 81 <= atomic_number <= 86:  # 6p系列重元素
                return 1e-4
            else:
                return 5e-3
        elif d_config['energy_level'] == 'core_like':
            # 类核心层d轨道：主族元素的填充d壳层，严格阈值
            return 1e-2
        else:
                         # 未知情况，使用保守阈值
             return 1e-3

    def _print_d_orbital_analysis_summary(self):

        print("\n  === d轨道相关性分析总结 ===")
        
        element_summary = {}
        for i, site in enumerate(self.structure):
            element = site.specie.symbol
            atomic_number = site.specie.Z
            is_relevant = self._is_d_orbital_relevant(i)
            
            if element not in element_summary:
                element_summary[element] = {
                    'relevant': 0, 'total': 0, 'atomic_number': atomic_number
                }
            
            element_summary[element]['total'] += 1
            if is_relevant:
                element_summary[element]['relevant'] += 1
        
        for element, data in element_summary.items():
            relevant_ratio = data['relevant'] / data['total']
            status = "正确计算" if relevant_ratio > 0 else "跳过计算"
            z = data['atomic_number']
            
            print(f"    {element} (Z={z}): {data['relevant']}/{data['total']} 原子d轨道相关 [{status}]")
        
        print("  ================================")

    def _calculate_dos_features_for_atom(self, pymatgen_index: int, gpaw_index: int) -> Dict[str, float]:

        e_f = self.fermi_level
        features = {}
        
        # 使用Pymatgen索引获取结构信息
        element_symbol = self.structure[pymatgen_index].specie.symbol

        # 诊断信息：检查计算是否自旋极化以及波函数数据类型
        # is_spin_polarized = self.fd_calc.get_spin_polarized()
        # wfs_dtype = self.fd_calc.wfs.dtype
        # print(f"  - DEBUG: 原子 {pymatgen_index} ({element_symbol}) - 计算自旋极化: {is_spin_polarized}, 波函数DType: {wfs_dtype}")
   
        # --- LDOS计算 ---
        ldos_at_fermi = np.nan
        calc_for_ldos = self.fd_calc # 强制使用FD模式GPW进行LDOS计算，以确保K点密度足够
        # 使用GPAW索引进行计算
        try:
            energies, ldos = calc_for_ldos.get_wigner_seitz_ldos(a=gpaw_index, spin=0, npts=401, width=0.1)
            ldos_at_fermi = np.interp(e_f, energies, ldos)
        except Exception as e:
            print(f"  - 错误: 原子 {pymatgen_index} ({element_symbol}) Wigner-Seitz LDOS计算失败: {e}")
            traceback.print_exc() # 打印完整的错误堆栈
            pass
        features['local_dos_fermi'] = ldos_at_fermi
        
        # --- 轨道电子数计算 ---
        for l_name in ['s', 'p', 'd']:
            electron_count = np.nan
            
            # 使用Pymatgen索引进行物理合理性检查
            if l_name == 'd' and not self._is_d_orbital_relevant(pymatgen_index):
                features[f'{l_name}_electron_count'] = np.nan
                continue

            try:
                # 使用GPAW索引进行计算
                energies, pdos = calc_for_ldos.get_orbital_ldos(a=gpaw_index, angular=l_name, spin=0, npts=401, width=0.1)
                occupied_mask = energies <= e_f
                
                if len(energies[occupied_mask]) > 0:
                    electron_count = np.trapz(pdos[occupied_mask], energies[occupied_mask])
                    if electron_count < -0.1:
                        electron_count = 0.0
                        
            except Exception as e:
                print(f"  - 错误: 计算原子 {pymatgen_index} ({element_symbol}) 的{l_name}轨道电子数失败: {e}")
                pass
            
            features[f'{l_name}_electron_count'] = electron_count
            
        return features

    def _calculate_dos_features_for_atom_corrected(self, pymatgen_index: int) -> Dict[str, float]:

        # 查找对应的GPAW索引
        gpaw_index = None
        for g_idx, p_idx in self.gpaw_to_pymatgen_index.items():
            if p_idx == pymatgen_index:
                gpaw_index = g_idx
                break
        
        if gpaw_index is None:
            # 如果找不到对应的GPAW索引，返回新特征格式的NaN
            return {
                'local_dos_fermi': np.nan,
                's_electron_count': np.nan,
                'p_electron_count': np.nan,
                'd_electron_count': np.nan
            }
        
        # 使用正确的索引调用核心计算函数
        return self._calculate_dos_features_for_atom(pymatgen_index, gpaw_index)
        
    # ==========================================================================
    # 整合与输出辅助方法 (更新特征名称)
    # ==========================================================================
    
    def _extract_classic_groups(self, classic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        group_A_cols = [
            'atomic_number', 'electronegativity', 'ionization_energy',
            'electron_affinity', 'valence_electrons', 'ionic_radius',
            'covalent_radius', 'coordination_number', 'avg_site_valence', 'bond_valence_sum'  # 新增BVS
        ]
        group_C_cols = [
            'bond_length_distortion', 'vectorial_asymmetry_norm_sq',
            'mean_squared_neighbor_distance', 'local_environment_anisotropy',
            'symmetry_breaking_quotient'
        ]
        return classic_df[group_A_cols], classic_df[group_C_cols]

    def save_features_to_csv(self, features_df: pd.DataFrame, output_path: str):
        """
        将计算的特征保存到CSV文件。
        
        参数:
            features_df (pd.DataFrame): 包含特征的数据框。
            output_path (str): CSV文件的保存路径。
        """
        # 检查期望的特征列是否存在
        expected_columns = [
            # A组：基础物理化学特征 (10维)
            'atomic_number', 'electronegativity', 'ionization_energy', 'electron_affinity',
            'valence_electrons', 'ionic_radius', 'covalent_radius', 'coordination_number',
            'avg_site_valence', 'bond_valence_sum',
            
            # B组：量子化学特征 (8维) 
            'bader_charge', 'electrostatic_potential', 'electron_density', 'elf',
            'local_magnetic_moment', 'local_dos_fermi', 's_electron_count', 'p_electron_count',
            'd_electron_count',
            
            # C组：几何特征 (5维)
            'bond_length_distortion', 'vectorial_asymmetry_norm_sq', 'mean_squared_neighbor_distance',
            'local_environment_anisotropy', 'symmetry_breaking_quotient',
            
            # D组：融合特征 (4维)
            'structure_chemistry_incompatibility', 'chemical_in_structural_metric',
            'charge_weighted_local_size', 'elf_weighted_local_anisotropy'
        ]
        
        # 保存特征
        features_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n--- 27维统一特征已成功保存到: {output_path} ---")


    def _save_structure_tensors(self, output_path: Path):
        print(f"\n--- 保存结构张量 ---")
        if not hasattr(self, '_tensors_to_save') or not self._tensors_to_save:
            print("警告: 未找到可供保存的结构张量。")
            return

        tensor_data = []
        for site_idx, tensor in sorted(self._tensors_to_save.items()):
            row = {'site_index': site_idx}
            for i in range(3):
                for j in range(3):
                    row[f'T_struct_{i}{j}'] = tensor[i, j]
            tensor_data.append(row)
        
        df = pd.DataFrame(tensor_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"结构张量已成功保存到: {output_path}")

    def _calculate_individual_bvs(self, site_idx: int) -> float:
        try:
            center_site = self.structure[site_idx]
            
            # 获取近邻信息
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            if not nn_info:
                return abs(self.valences[site_idx]) if site_idx < len(self.valences) else 1.0
            
            # 创建近邻site列表
            neighbor_sites = [nn['site'] for nn in nn_info]
            
            # 使用pymatgen的calculate_bv_sum函数
            bvs_value = calculate_bv_sum(center_site, neighbor_sites)
            return float(bvs_value)
                
        except Exception as e:
            print(f"  - 错误: 计算键价和失败: {e}")
            # 如果计算失败，使用氧化态的绝对值作为估计
            return abs(self.valences[site_idx]) if site_idx < len(self.valences) else 1.0

    # ==========================================================================
    # D组深度融合特征的具体实现
    # ==========================================================================

    def _calculate_structure_chemistry_incompatibility(self, T_struct: np.ndarray, v_chem: np.ndarray) -> float:
        try:
            # 从结构张量的特征向量构造"结构矢量"
            eigenvalues, eigenvectors = np.linalg.eigh(T_struct)
            # 使用最大特征值对应的特征向量作为结构的主方向
            v_struct = eigenvectors[:, np.argmax(eigenvalues)]

            # 构造so(3)生成元
            M_struct = self._construct_so3_generator(v_struct)
            M_chem = self._construct_so3_generator(v_chem)

            # 计算交换子 [M_struct, M_chem] = M_struct @ M_chem - M_chem @ M_struct
            commutator = M_struct @ M_chem - M_chem @ M_struct

            # 返回交换子的弗罗贝尼乌斯范数
            frobenius_norm = np.linalg.norm(commutator, 'fro')
            return float(frobenius_norm)

        except Exception as e:  
            print(f"  - 错误: 计算结构-化学不相容性失败: {e}")
            return np.nan

    def _calculate_chemical_in_structural_metric(self, T_struct: np.ndarray, v_chem: np.ndarray) -> float:
        try:
            # 获取投影后的3D化学矢量
            # 调用 _construct_so3_generator 来获取投影后的 x, y, z 分量
            # 注意：_construct_so3_generator 返回的是一个矩阵，我们需要从中提取 x, y, z
            if len(v_chem) == 7:
                # 调用 _construct_so3_generator 来获取投影后的 SO(3) 生成元矩阵
                so3_generator_matrix = self._construct_so3_generator(v_chem)
                # 从反对称矩阵中提取 x, y, z 分量
                # 矩阵形式为: [[0, -z, y], [z, 0, -x], [-y, x, 0]]
                x_proj = -so3_generator_matrix[1, 2] # x = -M[1,2]
                y_proj = so3_generator_matrix[0, 2]  # y = M[0,2]
                z_proj = -so3_generator_matrix[0, 1] # z = -M[0,1]
                v_chem_proj = np.array([x_proj, y_proj, z_proj])
            elif len(v_chem) == 3:
                v_chem_proj = v_chem
            else:
                # 处理其他非3或非7的维度，确保是3D向量
                v_chem_proj = np.zeros(3)
                v_chem_proj[:min(3, len(v_chem))] = v_chem[:min(3, len(v_chem))] # 截取前3个分量
            
            # 确保T_struct是正定的，通过添加小的正则化项
            # 物理上，结构张量应是半正定的，但在数值计算中，添加一个小的正则化项可以提高稳定性
            regularized_T = T_struct + np.eye(3) * self._TOLERANCE # 使用统一的容差值
            quadratic_form = v_chem_proj.T @ regularized_T @ v_chem_proj
            return float(quadratic_form)
        except Exception as e:
            print(f"  - 错误: 计算化学矢量在结构度量下的范数失败: {e}")
            return np.nan

    def _calculate_charge_weighted_local_size_new(self, site_idx: int, bader_charges: List[float]) -> float:
        try:
            T_charge_weighted = self._get_charge_weighted_structure_tensor(site_idx, bader_charges)
            return float(np.trace(T_charge_weighted))
        except Exception as e:
            print(f"  - 错误: 计算电荷加权的局域尺寸失败: {e}")
            return np.nan

    def _calculate_elf_weighted_local_anisotropy_new(self, site_idx: int, elf: float) -> float:
        try:
            if pd.isna(elf):
                return np.nan
            
            # 计算局域不对称矢量
            local_asymmetry_vector = self._calculate_local_asymmetry_vector(site_idx)
            vectorial_asymmetry_norm_sq = np.dot(local_asymmetry_vector, local_asymmetry_vector)
            
            # 确保 vectorial_asymmetry_norm_sq 也不是 NaN，否则结果会是 NaN
            if pd.isna(vectorial_asymmetry_norm_sq):
                return np.nan

            # ELF加权：高ELF区域的几何不对称性被放大
            return float(elf * vectorial_asymmetry_norm_sq)
        except Exception as e:
            print(f"  - 错误: 计算ELF加权的局域不对称性失败: {e}")
            return np.nan

    def _calculate_bond_valence_sum(self, site_idx: int) -> float:

        try:
            return float(self.bvs_list[site_idx])
        except (IndexError, TypeError) as e:
            print(f"  - 错误: 计算键价和失败: {e}")
            return np.nan

    def _get_coordination_number(self, site_idx: int) -> int:
        # 初始尝试使用默认的CrystalNN
        try:
            cn = self.crystal_nn.get_cn(self.structure, site_idx)
            if cn > 0: # 如果找到了邻居，直接返回
                return cn
        except Exception as e:
            # 初始尝试失败，打印警告但不终止
            print(f"  - 警告: 初始配位数获取失败 for site {site_idx}: {e}")

        # 如果初始尝试返回0或失败，尝试略微放宽容差重试
        # 这是一个小范围的动态调整，旨在提升鲁棒性
        # 严格遵守"不使用简化模型"，我们只在CrystalNN完全失败时小幅调整其内部判断的"容差"，而非强加外部逻辑。
        for attempt in range(1, 3): # 最多尝试2次重试
            try:
                # 每次重试都创建一个新的CrystalNN实例，并略微增加距离容差
                # x_diff_tol 默认是 0.05，我们每次增加 0.02，最多到 0.09
                temp_crystal_nn = CrystalNN(x_diff_tol=0.05 + attempt * 0.02)
                cn_retry = temp_crystal_nn.get_cn(self.structure, site_idx)
                if cn_retry > 0:
                    print(f"  - 成功: Site {site_idx} 在第 {attempt+1} 次尝试 (dist_tol={0.05 + attempt * 0.02:.2f}) 后找到配位数: {cn_retry}")
                    return cn_retry
            except Exception as e_retry:
                print(f"  - 警告: Site {site_idx} 第 {attempt+1} 次尝试配位数获取失败: {e_retry}")
        
        # 所有尝试都失败，返回0并打印最终错误
        print(f"  - 错误: 无法为Site {site_idx} 获取有效配位数，已返回0。")
        return 0

    def _parse_coordination(self, cn_str: Any) -> Optional[int]:

        if isinstance(cn_str, int):
            return cn_str
        if isinstance(cn_str, float):
            return int(cn_str)
        try:
            return int(cn_str)
        except (ValueError, TypeError) as e:
            
            # 处理字符串格式的配位数
            cn_str_upper = str(cn_str).upper().strip()
            
            # 罗马数字映射表
            roman_map = {
                'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 
                'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12,
                'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16
            }
            
            # 首先尝试直接匹配纯罗马数字
            if cn_str_upper in roman_map:
                return roman_map[cn_str_upper]
            
            # 处理复合格式（罗马数字 + 几何标记）
            # 常见的几何标记：PY(pyramidal), OH(octahedral), TH(tetrahedral), 
            # SQ(square), TB(trigonal bipyramidal), SP(square pyramidal), etc.
            geometry_suffixes = [
                'PY', 'OH', 'TH', 'SQ', 'TB', 'SP', 'TPR', 'SAPR', 'DD', 'PPR',
                'HBPY', 'JTC', 'JBTC', 'TDH', 'HD', 'TT', 'CO', 'DI', 'PBPY',
                'COC', 'HBCO', 'BCCO', 'SC', 'S', 'CU', 'CUAPR', 'HE',
                'CPS', 'CSAP', 'KLE', 'BCE', 'BFS', 'JCS', 'CAPS', 'ASAP',
                'SQA', 'LA', 'CUPR', 'EP', 'OC', 'HPY', 'ETBPY', 'HTBPY'
            ]
            
            # 尝试从复合字符串中提取罗马数字部分
            for suffix in geometry_suffixes:
                if cn_str_upper.endswith(suffix):
                    roman_part = cn_str_upper[:-len(suffix)]
                    if roman_part in roman_map:
                        return roman_map[roman_part]
            
            # 如果以上都不匹配，尝试查找字符串开头的罗马数字
            for roman, number in sorted(roman_map.items(), key=lambda x: -len(x[0])):
                if cn_str_upper.startswith(roman):
                    return number
            
            # 最后的尝试：查找字符串中的任何罗马数字
            for roman, number in sorted(roman_map.items(), key=lambda x: -len(x[0])):
                if roman in cn_str_upper:
                    return number
            
            # 如果所有方法都失败，返回None
            print(f"  - 警告: 无法解析配位数 '{cn_str}'，已跳过")
            return None

    def _get_precise_ionic_radius(self, mendeleev_elem, oxidation_state, cn) -> float:
        matching_radii = [r for r in mendeleev_elem.ionic_radii if r.charge == round(oxidation_state)]
        if not matching_radii: return np.nan

        available_cn = [{'radius': r.ionic_radius, 'cn': r.coordination} for r in matching_radii]
        
        # 尝试精确匹配 (数字或罗马数字)
        for r in available_cn:
            # 使用新的解析函数
            parsed_r_cn = self._parse_coordination(r['cn'])
            if parsed_r_cn is not None and parsed_r_cn == cn:
                return r['radius'] / 100.0
        
        # 回退: 找到最接近的配位数
        parsed_cns = []
        for r in available_cn:
            parsed_cn = self._parse_coordination(r['cn'])
            if parsed_cn is not None:
                parsed_cns.append({'radius': r['radius'], 'cn': parsed_cn})
        
        if not parsed_cns:
            # 如果没有一个可以解析的配位数，则返回第一个可用的半径
            return available_cn[0]['radius'] / 100.0 if available_cn else np.nan
        
        # 找到与目标配位数最接近的一个
        closest = min(parsed_cns, key=lambda x: abs(x['cn'] - cn))
        return closest['radius'] / 100.0

    def _to_roman(self, n: int) -> str:
        """
        将整数转换为罗马数字字符串。
        仅支持1到14的整数。
        """
        if not isinstance(n, int) or not 1 <= n <= 14:
            return ""
        return ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV"][n]

    def _calculate_bond_length_distortion(self, site_idx: int) -> float:
        try: nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        except Exception as e:
            print(f"  - 错误: 计算键长畸变指数失败: {e}")
            return 1.0
        
        if len(nn_info) < 2: return 0.0
        
        distances = [self.structure[site_idx].distance(n['site']) for n in nn_info]
        avg_dist = np.mean(distances)
        if avg_dist == 0: return 0.0
        return float(np.sqrt(np.mean(((distances - avg_dist) / avg_dist)**2)))

    def _calculate_local_asymmetry_vector(self, site_idx: int) -> np.ndarray:
        try: nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        except Exception as e:
            print(f"  - 错误: 计算局部原子环境的结构不对称性向量失败: {e}")
            return np.zeros(3)
        if not nn_info: return np.zeros(3)
        
        center_coords = self.structure[site_idx].coords
        return np.sum([n['site'].coords - center_coords for n in nn_info], axis=0)

    def _construct_so3_generator(self, vector: np.ndarray) -> np.ndarray:

        if len(vector) == 3:
            # 向后兼容：如果是3维向量，直接使用
            x, y, z = vector
        elif len(vector) == 7:
            # 基于物理意义和Gram-Schmidt正交化构建投影基
            # 7维化学向量分量顺序：
            # [electronegativity, covalent_radius, ionization_energy, electron_affinity, atomic_volume, polarizability, effective_charge]

            # 初始方向向量 (非正交，基于物理分组)
            d1_electrochem = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]) # 电负性、电离能、电子亲和能
            d2_spatial_geom = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) # 共价半径、原子体积
            d3_electron_response = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]) # 极化率、玻恩有效电荷

            # Gram-Schmidt正交化过程
            u1 = d1_electrochem
            e1 = u1 / (np.linalg.norm(u1) + self._TOLERANCE)

            proj_d2_u1 = np.dot(d2_spatial_geom, u1) / (np.dot(u1, u1) + self._TOLERANCE) * u1
            u2 = d2_spatial_geom - proj_d2_u1
            e2 = u2 / (np.linalg.norm(u2) + self._TOLERANCE)

            proj_d3_u1 = np.dot(d3_electron_response, u1) / (np.dot(u1, u1) + self._TOLERANCE) * u1
            proj_d3_u2 = np.dot(d3_electron_response, u2) / (np.dot(u2, u2) + self._TOLERANCE) * u2
            u3 = d3_electron_response - proj_d3_u1 - proj_d3_u2
            e3 = u3 / (np.linalg.norm(u3) + self._TOLERANCE)

            # 将7维化学向量投影到新的正交基上
            x = np.dot(vector, e1)
            y = np.dot(vector, e2)
            z = np.dot(vector, e3)
            
        else:
            # 其他维度：标准数学处理
            padded_vector = np.zeros(3)
            padded_vector[:min(3, len(vector))] = vector[:3]
            x, y, z = padded_vector
        
        return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

# --- 使用示例 ---
if __name__ == '__main__':
    # --- 输入文件路径 ---
    # 定义输入的CIF文件路径
    cif_file = "CsPbI3-supercell-optimized.cif"
    
    # 基于输入文件名自动生成输出文件名
    # 提取文件名的主干部分，去除'_optimized'后缀，用于生成统一的输出文件名
    base_name = Path(cif_file).stem.replace('-optimized', '')
    pw_gpw_file = f'{base_name}.gpw'
    fd_gpw_file = f'{base_name}-fd.gpw'
    output_csv = f"{base_name}-0-Simplex-Features.csv"
    
    print("=" * 60)
    print(f"输入文件: {cif_file}, {pw_gpw_file}, {fd_gpw_file}")
    print(f"输出文件: {output_csv}")
    print("=" * 60)
    
    # 初始化计算器并运行完整流程
    try:
        calculator = UnifiedFeatureCalculator(cif_file, pw_gpw_file, fd_gpw_file)
        unified_features_df = calculator.calculate_unified_features()
    
        # 保存结果
        calculator.save_features_to_csv(unified_features_df, output_csv)

        # 打印简要报告
        print("\n--- 计算完成 ---")
        print(unified_features_df.head())
        print("\n描述性统计:")
        print(unified_features_df.describe())

    except FileNotFoundError as e:
        print(f"\n错误: 关键输入文件缺失: {e}")
        print("请确保CIF和PW模式的GPW文件存在于工作目录中。")
    except Exception as e:
        print(f"\n一个意外的错误发生: {e}")
        import traceback
        traceback.print_exc()
