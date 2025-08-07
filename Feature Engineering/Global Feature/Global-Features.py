import warnings
import numpy as np
import pandas as pd
import json
import hashlib
import traceback
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Tuple, List, Set

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.util.graph_hashing import weisfeiler_lehman_graph_hash
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams

from ase.io import read
from gpaw import GPAW # type: ignore
from scipy.ndimage import zoom
import networkx as nx
from scipy.interpolate import RegularGridInterpolator

class GlobalFeatureCalculator:
    # 基于物理定义的阈值（有明确物理学依据）
    OCTAHEDRAL_ORDER_PARAM_THRESHOLD = 0.75  # 基于Steinhardt等人的局域有序参数定义
    
    # 移除硬编码的算法参数，改为自适应计算
    
    _TOLERANCE = 1e-9  # 用于避免数值计算中的除零错误
    
    def __init__(self, cif_file_path: str, atomic_features_csv: str, bond_features_csv: str, 
                 pw_gpw_file: str = None, fd_gpw_file: str = None, 
                 random_seed: int = 42):
       
        # 设置可控随机种子
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.rng = np.random.default_rng(random_seed)
        
        self.cif_path = cif_file_path
        self.pw_gpw_file = pw_gpw_file
        self.fd_gpw_file = fd_gpw_file
            
        self.pmg_structure = Structure.from_file(cif_file_path)
        self.ase_structure = read(cif_file_path)
        self.local_features_df = pd.read_csv(atomic_features_csv)
        self.bond_features_df = pd.read_csv(bond_features_csv)
        self.gpaw_calc = None
        self.crystal_nn = CrystalNN()
        
        # 从0-单纯形特征文件加载氧化态（使用avg_site_valence），并设置到pymatgen结构中
        if 'avg_site_valence' in self.local_features_df.columns:
            try:
                # 确保原子顺序与pymatgen结构一致
                oxidation_states_from_0simplex = self.local_features_df['avg_site_valence'].tolist()
                if len(oxidation_states_from_0simplex) == len(self.pmg_structure):
                    # add_oxidation_state_by_site 接受列表，按顺序设置
                    self.pmg_structure.add_oxidation_state_by_site(oxidation_states_from_0simplex)
                    print("成功从0-单纯形特征文件加载并设置氧化态 (使用avg_site_valence)。")
                else:
                    raise ValueError("0-单纯形CSV中的原子数与CIF文件不匹配，无法设置氧化态。")
            except Exception as e:
                print(f"警告: 从0-单纯形设置氧化态失败: {e}。CrystalNN可能使用默认值。")
        else:
            print("警告: 0-单纯形特征文件中未找到'avg_site_valence'列，无法设置氧化态。")

        # 加载0-单纯形结构张量，避免重复计算
        self.atomic_tensors_from_0simplex = {}
        atomic_tensors_csv_path = Path(cif_file_path).parent / f"{Path(cif_file_path).stem.replace('-optimized', '')}-0-Simplex-Structure-Tensors.csv"
        if atomic_tensors_csv_path.exists():
            try:
                tensors_df = pd.read_csv(atomic_tensors_csv_path)
                for _, row in tensors_df.iterrows():
                    site_idx = int(row['site_index'])
                    tensor = np.array([
                        [row['T_struct_00'], row['T_struct_01'], row['T_struct_02']],
                        [row['T_struct_10'], row['T_struct_11'], row['T_struct_12']],
                        [row['T_struct_20'], row['T_struct_21'], row['T_struct_22']]
                    ])
                    self.atomic_tensors_from_0simplex[site_idx] = tensor
                print(f"成功加载0-单纯形结构张量文件: {atomic_tensors_csv_path.name}")
            except Exception as e:
                print(f"警告: 加载0-单纯形结构张量失败: {e}。部分特征计算可能受影响。")
        else:
            print(f"警告: 未找到0-单纯形结构张量文件: {atomic_tensors_csv_path.name}。部分特征计算将使用替代方法或NaN。")
        
        # 材料尺寸自适应参数
        self._setup_adaptive_parameters()
        
        print(f"随机种子: {random_seed}")
        print(f"成功读取晶体结构: {self.pmg_structure.composition.reduced_formula}")
        print(f"自适应参数: 最大路径长度={self.GRAPH_PATH_MAX_LENGTH}, 采样大小={self.PATH_SAMPLING_SIZE}")
        print(f"成功加载0-单纯形（原子）特征文件: {atomic_features_csv}")
        print(f"成功加载1-单纯形（化学键）特征文件: {bond_features_csv}")
        if self.pw_gpw_file:
            print(f"将使用PW模式GPW文件: {self.pw_gpw_file}")
        if self.fd_gpw_file and self.fd_gpw_file != self.pw_gpw_file:
            print(f"将使用FD模式GPW文件: {self.fd_gpw_file}")

        # 验证CSV中的原子数量是否与CIF文件匹配
        if len(self.local_features_df) != len(self.pmg_structure):
            raise ValueError(
                f"0-单纯形CSV文件中的原子数 ({len(self.local_features_df)}) "
                f"与CIF文件中的原子数 ({len(self.pmg_structure)}) 不匹配。"
            )

    def _setup_adaptive_parameters(self):
        """
        基于材料尺寸设置自适应参数，避免大超胞OOM。
        
        自适应公式：
        - 路径长度：基于原子数的阶梯式缩放，平衡计算深度与复杂度。
        - 采样大小：基于原子总数的阶梯式设置，需权衡计算效率与特征代表性。
        """
        num_atoms = len(self.pmg_structure)
        
        # 自适应最大路径长度：大系统使用较短路径
        # 物理意义：对于大型周期性系统，过长的路径容易引入冗余信息或超出局部关联范畴。
        # 较短路径的统计量已足以捕捉全局特性，同时显著降低图搜索复杂度。
        if num_atoms <= 50:
            self.GRAPH_PATH_MAX_LENGTH = 6  # 适用于小分子或小晶胞，允许探索更远的相互作用
        elif num_atoms <= 200:
            self.GRAPH_PATH_MAX_LENGTH = 5  # 中等大小，兼顾细节与性能
        else:
            self.GRAPH_PATH_MAX_LENGTH = 4  # 大型系统，聚焦主要相互作用，严格控制计算量

        # 自适应采样大小：根据原子总数进行阶梯式设置
        # 注意：当前设定 (50000, 150000) 可能会导致对于中大型系统计算耗时过长，甚至内存溢出。
        # 如果计算效率是首要考虑，建议适当降低这些值。
        if num_atoms <= 100:
            self.PATH_SAMPLING_SIZE = 5000 
        elif num_atoms <= 500:
            self.PATH_SAMPLING_SIZE = 10000 
        else:
            self.PATH_SAMPLING_SIZE = 50000 

        # 确保PATH_SAMPLING_SIZE至少大于或等于随机行走的最小路径长度（即使是随机行走也需要一些路径）
        self.PATH_SAMPLING_SIZE = max(self.PATH_SAMPLING_SIZE, 1000) # 确保有足够多的路径来计算度量指标

    def calculate_features(self) -> pd.DataFrame:
        """
        计算A、B、C、D部分的所有特征。

        Returns:
        --------
        pd.DataFrame: 包含A、B、C、D部分特征的单行DataFrame。
        """
        print("开始计算 A 部分的全局特征...")
        features_A = self._calculate_features_A()

        print("\n开始计算 B 部分的全局特征 (DFT)...")
        features_B = self._calculate_features_B()
        
        print("\n开始计算 C 部分的全局特征 (代数)...")
        features_C = self._calculate_features_C()
        
        print("\n开始计算 D 部分的全局特征 (图相关路径融合)...")
        features_D = self._calculate_graph_correlation_path_features()

        # 合并所有特征字典
        all_features = {**features_A, **features_B, **features_C, **features_D}
        
        df = pd.DataFrame([all_features])
        print("\nA, B, C, D 部分特征计算完成。")
        return df

    def _calculate_features_A(self) -> Dict[str, Any]:
        """A. 基础统计与几何 (6维)"""
        features = {}
        # 使用统一的、基于化学意义的CrystalNN
        voro = self.crystal_nn 
        
        # 1. mean_bond_length
        # 使用鲁棒的邻居查找方法
        all_bonds = []
        for i in range(len(self.pmg_structure)):
            all_bonds.extend(self._get_nn_info_robust(self.pmg_structure, i))

        # 直接从1-Simplex-Features.csv读取平均键长，确保数据一致性和准确性
        features['mean_bond_length'] = self.bond_features_df['bond_distance'].mean() if not self.bond_features_df.empty else 0

        # 2. volume_per_fu
        comp = self.pmg_structure.composition
        factor = comp.get_reduced_composition_and_factor()[1]
        features['volume_per_fu'] = self.pmg_structure.volume / factor if factor else 0

        # 3. lattice_anisotropy_ratio
        a, b, c = self.pmg_structure.lattice.abc
        features['lattice_anisotropy_ratio'] = max(a, b, c) / min(a, b, c) if min(a,b,c) > 0 else 0

        # 新增特征: bulk_anisotropy_index
        features['bulk_anisotropy_index'] = self._calculate_bulk_anisotropy()

        # 4. octahedral_count
        features['octahedral_count'] = self._calculate_octahedral_count()
        
        # 5. packing_fraction
        features['packing_fraction'] = self._calculate_packing_fraction()

        print("A 部分特征计算完成。")
        return features

    def _calculate_bulk_anisotropy(self):
        """计算体相各向异性指数，作为表面效应强度的代理特征。
        
        该特征通过分析体相晶格中原子局部环境的平均"形状"，
        定量捕捉晶体的内在结构各向异性程度。
        """
        structure = self.pmg_structure
        sga = SpacegroupAnalyzer(structure)
        symmetrized_structure = sga.get_symmetrized_structure()
        unique_site_indices = [sites[0] for sites in symmetrized_structure.equivalent_indices]

        if not unique_site_indices:
            return 0.0

        all_anisotropy_values = []

        # 直接从预加载的0-单纯形结构张量中获取
        if not self.atomic_tensors_from_0simplex:
            print("  - 警告: 未加载0-单纯形结构张量，无法计算bulk_anisotropy_index。")
            return np.nan # 如果没有张量数据，则无法计算此特征

        for site_idx in unique_site_indices:
            try:
                # 直接获取预计算的结构张量
                M_local = self.atomic_tensors_from_0simplex.get(site_idx)
                if M_local is None or np.any(np.isnan(M_local)):
                    print(f"  - 警告: 原子 {site_idx} 的结构张量无效或未找到，跳过。")
                    continue

                # 计算特征值
                # 使用 eigvalsh 确保对于实对称矩阵返回实数特征值，避免数值噪声
                eigenvalues = np.linalg.eigvalsh(M_local)
                eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列

                # 避免除零错误
                if np.sum(eigenvalues) > 1e-10:
                    # 计算各向异性指数 η = (λ_max - λ_min) / (λ_max + λ_min + λ_mid)
                    eta = (eigenvalues[0] - eigenvalues[2]) / np.sum(eigenvalues)
                    all_anisotropy_values.append(eta)

            except Exception as e:
                print(f"  - 错误: 计算原子 {site_idx} 的各向异性失败: {e}")
                continue

        if not all_anisotropy_values:
            return np.nan # 如果所有原子都无法计算，则返回NaN

        # 返回所有不等价位置的平均各向异性指数
        return np.mean(all_anisotropy_values)

    def _calculate_octahedral_count(self) -> int:

        octahedra = 0
        crystal_nn = self.crystal_nn
        lsop = LocalStructOrderParams(['oct'])
        
        for i, site in enumerate(self.pmg_structure):
            # 条件 1: 中心原子必须是金属 (例如 Cs, Pb)
            if not site.specie.is_metal:
                continue

            # 条件 2: 配位数必须精确为6
            # 由于 _get_nn_info_robust 返回的是列表，可以直接检查长度
            if len(self._get_nn_info_robust(self.pmg_structure, i)) != 6:
                continue

            # 获取近邻信息
            neighbors = self._get_nn_info_robust(self.pmg_structure, i)
            
            # 附加检查，确保确实找到了6个近邻
            if len(neighbors) != 6:
                continue
                
            # 条件 3: 所有近邻必须是非金属 (例如 Br)
            are_all_neighbors_non_metal = all(not nn['site'].specie.is_metal for nn in neighbors)
            if not are_all_neighbors_non_metal:
                continue

            # 如果所有化学和几何条件都满足，才计算有序参数
            neighbor_indices = [nn['site_index'] for nn in neighbors]
            ops = lsop.get_order_parameters(self.pmg_structure, i, indices_neighs=neighbor_indices)
            
            # 如果有序参数大于阈值，则认为是目标八面体
            if ops and ops[0] is not None and ops[0] > self.OCTAHEDRAL_ORDER_PARAM_THRESHOLD:
                octahedra += 1
                
        return octahedra

    def _calculate_packing_fraction(self) -> float:

        # 验证必需的列是否存在
        required_cols = ['atomic_number', 'ionic_radius']
        for col in required_cols:
            if col not in self.local_features_df.columns:
                raise KeyError(f"CSV文件中未找到必需的 '{col}' 列。")
            
        total_ionic_volume = 0.0
        
        # 使用zip同时遍历pymatgen结构中的原子和CSV数据行 (itertuples更高效)
        for i, (site, row) in enumerate(zip(self.pmg_structure, self.local_features_df.itertuples())):
            
            # 关键验证：确保pymatgen中的原子与CSV中的原子按序匹配
            if site.specie.Z != row.atomic_number:
                raise ValueError(
                    f"原子顺序不匹配：在索引 {i} 处, "
                    f"CIF文件中的原子为 {site.specie.symbol} (Z={site.specie.Z}), "
                    f"而CSV文件中的原子 Z={row.atomic_number}。"
                )

            radius = row.ionic_radius
            if pd.notna(radius) and radius > 0:
                # 计算离子的体积 (4/3 * π * r³)
                ionic_volume = (4.0 / 3.0) * np.pi * (radius ** 3)
                total_ionic_volume += ionic_volume
                
        # 获取晶胞体积
        unit_cell_volume = self.pmg_structure.volume
        
        # 计算堆积因子
        if unit_cell_volume > 0:
            packing_fraction = total_ionic_volume / unit_cell_volume
        else:
            packing_fraction = 0
        
        return packing_fraction

    def _get_gpw_filename(self) -> Path:
        """生成一个基于化学式和路径的唯一gpw文件名。"""
        formula = self.pmg_structure.composition.reduced_formula
        cif_path = Path(self.cif_path)
        return cif_path.parent / f"{formula}_{cif_path.stem}_scf.gpw"


    def _calculate_features_B(self) -> Dict[str, Any]:
        """B. DFT计算的基态属性 (4维) """
        features = {}
        
        # 优先使用PW模式GPW文件进行基态性质计算
        gpw_file_for_ground_state = self.pw_gpw_file
        if not gpw_file_for_ground_state:
            gpw_file_for_ground_state = self.fd_gpw_file
            
        if gpw_file_for_ground_state and Path(gpw_file_for_ground_state).exists():
            gpw_file = Path(gpw_file_for_ground_state)
            print(f"正在加载用于基态计算的GPW文件: {gpw_file}")
        else:
            # 如果都没有，则回退到自动缓存机制
            gpw_file = self._get_gpw_filename()
            print(f"正在检查或生成自动缓存的GPAW结果文件: {gpw_file}")
            if not gpw_file.exists():
                print("未找到 .gpw 文件, 请检查文件路径是否正确。")
                return {
                    'total_energy_per_atom': np.nan,
                    'fermi_level': np.nan,
                    'electrostatic_potential_mean': np.nan,
                    'electrostatic_potential_variance': np.nan
                }
            else:
                print(f"找到已存在的 .gpw 文件, 将从中加载结果。")
        
        # 从文件加载计算器以确保状态一致
        try:
            self.gpaw_calc = GPAW(gpw_file, txt=None)
            self.ase_structure.calc = self.gpaw_calc
        except Exception as e:
            print(f"加载 .gpw 文件失败: {e}。")
            return {
                'total_energy_per_atom': np.nan,
                'fermi_level': np.nan,
                'electrostatic_potential_mean': np.nan,
                'electrostatic_potential_variance': np.nan
            }

        # --- 从计算器中提取特征 ---
        print("从GPAW计算器中提取特征...")
        
        # 1. total_energy_per_atom - 增强稳健性
        try:
            total_energy = self.gpaw_calc.get_potential_energy()
            num_atoms = len(self.ase_structure)
            features['total_energy_per_atom'] = total_energy / num_atoms if num_atoms > 0 else 0
        except Exception as e:
            print(f"  - 错误: 无法获取总能量: {e}。该特征值将设为NaN。")
            features['total_energy_per_atom'] = np.nan

        # 2. fermi_level - 增强稳健性  
        try:
            features['fermi_level'] = self.gpaw_calc.get_fermi_level()
        except Exception as e:
            print(f"  - 错误: 无法获取费米能级: {e}。该特征值将设为NaN。")
            features['fermi_level'] = np.nan

        # 3. electrostatic_potential_mean & 4. variance - 委员会建议的改进插值
        try:
            potential = self.gpaw_calc.get_electrostatic_potential()
            # 检查网格质量
            # 确保网格非空
            if potential is not None and potential.size > 0 and \
               not np.all(np.isnan(potential)) and not np.all(np.isinf(potential)): # 确保不全是NaN或Inf
                
                # 过滤掉潜在的NaN/Inf值，确保统计量的稳健性
                valid_potential = potential[np.isfinite(potential)]
                
                if valid_potential.size > 0:
                    features['electrostatic_potential_mean'] = np.mean(valid_potential)
                    features['electrostatic_potential_variance'] = np.var(valid_potential)
                    print(f"  - 静电势均值: {features['electrostatic_potential_mean']}, 静电势方差: {features['electrostatic_potential_variance']}")
                else:
                    # 如果过滤后没有有效值，则设置为NaN
                    features['electrostatic_potential_mean'] = np.nan
                    features['electrostatic_potential_variance'] = np.nan
                    print("  - 警告: 静电势网格中无有效数值。")
            else:
                # 如果网格无效、为空或全是NaN/Inf
                raise ValueError("静电势网格为空、无效或全是NaN/无穷值。")
        except Exception as e:
            print(f"  - 错误: 无法获取静电势: {e}。该特征值将设为NaN。")
            features['electrostatic_potential_mean'] = np.nan
            features['electrostatic_potential_variance'] = np.nan
            
        print("B 部分特征提取完成。")
        return features

    def _calculate_features_C(self) -> Dict[str, Any]:
        """C. 全局高阶代数特征 (10维)"""
        
        # Part 1: 商代数 & 几何代数特征 (源自结构)
        # 方法现在返回特征字典和局部不对称矢量，使数据流更清晰
        features_C_geom, local_asymmetry_vectors = self._calculate_quotient_and_geometric_features()
        
        # Part 2: 基于力的局部扭转应力特征 (源自SCF计算) 
        features_C_force = self._calculate_local_torsional_stress_features(local_asymmetry_vectors)

        # Part 3: 基于场-梯度耦合的场-密度梯度耦合特征 (源自SCF计算)
        features_C_stress = self._calculate_field_density_coupling_features()
        
        # Part 4: 基于静态相空间的全局伪辛几何特征（物理类比）
        # 返回log(行列式)以避免数值下溢，并更新特征名称
        features_C_symplectic = self._calculate_symplectic_fluctuation_volume()

        return {**features_C_geom, **features_C_force, **features_C_stress, **features_C_symplectic}

    def _get_V_struct(self, center_index: int) -> np.ndarray:
        """
        计算指定原子的0-单纯形结构不对称性向量 V_struct = Σv_i。
        该方法直接从0-单纯形特征计算脚本中移植而来，确保了一致性。
        """
        center_site = self.pmg_structure[center_index]
        try:
            # 使用鲁棒的邻居查找方法
            neighbors = self._get_nn_info_robust(self.pmg_structure, center_index)
            if not neighbors:
                return np.zeros(3)
            # 使用真实坐标进行计算，得到物理意义明确的笛卡尔向量
            v_struct = np.sum([nn['site'].coords - center_site.coords for nn in neighbors], axis=0)
            return v_struct
        except Exception as e:
            print(f"  - 错误: 计算0-单纯形结构不对称性向量失败: {e}")
            return np.zeros(3)
            
    def _calculate_graph_correlation_path_features(self) -> Dict[str, Any]:
        """D. 图相关路径特征 (深度融合) - 采用统一重要性采样
        
        
        
        """
        features = {
            'path_cov_torsional_stress_mean': np.nan,
            'path_cov_torsional_stress_std': np.nan,
            'path_cov_torsional_stress_max': np.nan,
            
            'path_entropy_mean': np.nan,
            'path_entropy_std': np.nan,
            'path_entropy_max': np.nan,
            
            'path_chempot_diff_mean': np.nan,
            'path_chempot_diff_std': np.nan,
            'path_chempot_diff_max': np.nan,
            
            'path_max_torque_mean': np.nan,
            'path_max_torque_std': np.nan,
            'path_max_torque_max': np.nan,
            
            'path_curvature_mean': np.nan,
            'path_curvature_std': np.nan,
            'path_curvature_max': np.nan,

            'path_wrapping_norm_mean': np.nan,
            'path_wrapping_norm_std': np.nan,
            'path_wrapping_norm_max': np.nan,
            
            'path_force_gradient_mean': np.nan,
            'path_force_gradient_std': np.nan,
            'path_force_gradient_max': np.nan,

            'path_structure_autocorr_mean': np.nan,
            'path_structure_autocorr_std': np.nan,
            'path_structure_autocorr_max': np.nan,

            'path_charge_potential_cov_mean': np.nan,
            'path_charge_potential_cov_std': np.nan,
            'path_charge_potential_cov_max': np.nan
        }
        
        if self.gpaw_calc is None:
            warnings.warn("无法计算图相关路径特征，因为没有有效的 GPAW 计算器。")
            return features

        print("开始计算 D 部分的全局特征 (图相关路径融合)...")
        import time
        d_part_start_time = time.time()

        # --- I. 数据准备 ---
        print("  - 正在准备计算所需数据...")
        forces = self.gpaw_calc.get_forces()
        
        print("  - 正在获取静电势和Bader电荷...")
        esp_grid = self.gpaw_calc.get_electrostatic_potential()
        potential_at_atoms = self._interpolate_grid_at_atomic_sites(esp_grid)
        bader_charges = self.local_features_df['bader_charge'].values

        print("  - 构建晶体图...")
        graph = StructureGraph.from_local_env_strategy(self.pmg_structure, self.crystal_nn)
        nx_graph = graph.graph

        print(f"  - 搜索所有长度 3-{self.GRAPH_PATH_MAX_LENGTH} 的简单路径...")
        # 关键一步：将有向图转换为无向图进行路径搜索
        undirected_nx_graph = nx_graph.to_undirected()
        
        # 基于图大小智能选择路径搜索算法
        num_nodes = len(undirected_nx_graph.nodes)
        num_edges = len(undirected_nx_graph.edges)
        graph_complexity = num_nodes * num_edges
        
        # 智能算法选择，大幅降低复杂度阈值
        if graph_complexity > 2000 or num_nodes > 50:  # 更保守的阈值，避免性能瓶颈
            print(f"  - 检测到复杂图 (复杂度={graph_complexity}, 节点={num_nodes})，使用高效随机行走...")
            # 基于统计收敛性的自适应采样
            num_walks = self.PATH_SAMPLING_SIZE * 2  # 进一步降低随机行走次数的乘数，提高效率
            all_paths = self._find_paths_with_random_walk(undirected_nx_graph, num_walks=num_walks)
            print(f"  - 随机行走完成，采样了 {num_walks} 次，获得 {len(all_paths)} 条独特路径")
        else:  # 仅对小型图使用完全枚举
            print(f"  - 小型图 (复杂度={graph_complexity})，使用带早期停止的完全枚举...")
            all_paths = self._find_all_simple_paths_with_early_stop(undirected_nx_graph, max_paths=5000)
        
        if not all_paths:
            warnings.warn("警告: 在晶体图中未找到任何路径。")
            return features
            
        print(f"  - 找到 {len(all_paths)} 条路径，开始计算统一重要性权重...")
        
        # 添加进度报告和缓存机制
        path_metrics = []
        processed_count = 0
        total_paths = len(all_paths)
        
        # 预计算常用数据以避免重复计算
        incompatibility_data = self.local_features_df['structure_chemistry_incompatibility'].values
        electronegativity_data = self.local_features_df['electronegativity'].values
        
        for path in all_paths:
            # 进度报告（每处理1000条路径报告一次）
            if processed_count % 1000 == 0 and processed_count > 0:
                print(f"  - 已处理 {processed_count}/{total_paths} 条路径 ({100*processed_count/total_paths:.1f}%)")
            
            # 使用预计算的数据，提高访问效率
            path_indices = list(path)
            structure_incompatibility_seq = incompatibility_data[path_indices]
            en_seq = electronegativity_data[path_indices]
            
            torque_seq = self._get_torque_sequence(path, nx_graph, forces)
            if not torque_seq: 
                processed_count += 1
                continue

            # 计算三个核心指标
            s_struct = self._calculate_shannon_entropy(structure_incompatibility_seq)
            delta_chi = np.ptp(en_seq) # Peak-to-peak (max - min)
            tau_max = np.max(torque_seq)
            
            path_metrics.append({
                'path': path,
                's_struct': s_struct,
                'delta_chi': delta_chi,
                'tau_max': tau_max,
                'torque_seq': torque_seq,
                'structure_incompatibility_seq': structure_incompatibility_seq
            })
            
            processed_count += 1

        if not path_metrics:
            warnings.warn("警告: 无法为任何路径计算有效的度量指标。")
            return features
            
        metrics_df = pd.DataFrame(path_metrics)

        # 归一化处理
        for col in ['s_struct', 'delta_chi', 'tau_max']:
            min_val, max_val = metrics_df[col].min(), metrics_df[col].max()
            if max_val - min_val > 1e-9:
                metrics_df[f'{col}_norm'] = (metrics_df[col] - min_val) / (max_val - min_val)
            else:
                metrics_df[f'{col}_norm'] = 0.5 # 如果所有值都相同，则赋予一个中性权重
        
        # 使用Wasserstein-like距离度量的权重计算
        # 改用soft-min (exp(-β·metric)) 减缓数值下溢，具有明确几何意义
        metrics_df['distance_s'] = 1.0 - metrics_df['s_struct_norm']  # 转换为距离
        metrics_df['distance_c'] = 1.0 - metrics_df['delta_chi_norm']
        metrics_df['distance_t'] = 1.0 - metrics_df['tau_max_norm']
        
        # 使用自适应soft-min权重：基于数据分布自动确定β参数
        # 数学原理：β = 1 / std(distances)，确保权重分布有意义的区分度
        distance_std = max(metrics_df['distance_s'].std(), 1e-6)  # 避免除零
        beta = 1.0 / distance_std  # 自适应β参数，完全基于数据分布
        weights_s = np.exp(-beta * metrics_df['distance_s'])
        weights_c = np.exp(-beta * metrics_df['distance_c'])
        weights_t = np.exp(-beta * metrics_df['distance_t'])
        
        # Wasserstein-like综合距离：保证度量空间性质
        metrics_df['unified_weight'] = (weights_s * weights_c * weights_t)**(1/3)


        # --- 统一重要性采样 ---
        total_weight = metrics_df['unified_weight'].sum()
        if total_weight < 1e-9:
            # 如果所有权重都接近于零，则回退到简单随机抽样
            warnings.warn("警告: 所有路径的统一重要性权重都接近于零。回退到简单随机抽样。")
            if len(metrics_df) > self.PATH_SAMPLING_SIZE:
                sampled_df = metrics_df.sample(n=self.PATH_SAMPLING_SIZE, random_state=self.random_seed)
            else:
                sampled_df = metrics_df
        else:
            metrics_df['sampling_prob'] = metrics_df['unified_weight'] / total_weight
            sample_size = min(len(metrics_df), self.PATH_SAMPLING_SIZE)
            sampled_df = metrics_df.sample(n=sample_size, weights='sampling_prob', random_state=self.random_seed)

        print(f"  - 已从 {len(all_paths)} 条路径中采样 {len(sampled_df)} 条，开始计算最终特征...")
        
        # --- V. 最终特征计算 ---
        # 对采样出的路径计算所有我们关心的特征
        
        final_metrics = []
        for _, row in sampled_df.iterrows():
            path = row['path']
            
            # 1. 力场梯度 (归一化处理)
            force_seq = forces[list(path)]
            force_gradients = np.linalg.norm(np.diff(force_seq, axis=0), axis=1)**2
            path_force_gradient_raw = np.sum(force_gradients)
            
            # 按路径长度归一化，使不同长度路径可比较
            path_length = len(path) - 1  # 路径段数
            path_force_gradient = path_force_gradient_raw / max(path_length, 1) if path_length > 0 else 0

            # 2. 曲率和缠绕范数
            bond_props = self._get_bond_properties_sequence(path, nx_graph)
            if not bond_props: continue
            
            bond_vectors = [prop['vector'] for prop in bond_props]
            image_vectors = [prop['image'] for prop in bond_props]

            # 曲率计算归一化
            path_curvature = 0
            num_angles = 0
            for i in range(len(bond_vectors) - 1):
                v1 = bond_vectors[i]
                v2 = bond_vectors[i+1]
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm1 > 1e-10 and norm2 > 1e-10:  # 避免除零
                    cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    path_curvature += angle
                    num_angles += 1
            
            # 归一化曲率：按转折点数量平均
            path_curvature = path_curvature / max(num_angles, 1) if num_angles > 0 else 0
            
            # 缠绕范数归一化
            path_wrapping_vector = np.sum(image_vectors, axis=0)
            
            # 未考虑晶格非正交性，在分数坐标空间定义，需转换为笛卡尔坐标再取范数
            # W_P^frac = sum(f_k), ||W_P|| = ||L * W_P^frac||_2
            lattice_matrix = self.pmg_structure.lattice.matrix
            path_wrapping_vector_cartesian = np.dot(lattice_matrix, path_wrapping_vector)
            path_wrapping_norm_raw = np.linalg.norm(path_wrapping_vector_cartesian)
            
            # 按路径长度归一化
            path_wrapping_norm = path_wrapping_norm_raw / max(len(path), 1)

            # 3. 结构不相容性自相关 
            structure_incompatibility_seq = row['structure_incompatibility_seq']
            structure_autocorr = np.nan
            if len(structure_incompatibility_seq) >= 2:
                # 当序列方差为0时，autocorr返回NaN，这正是我们期望的行为
                autocorr = pd.Series(structure_incompatibility_seq).autocorr(lag=1)
                if pd.notna(autocorr):
                    structure_autocorr = autocorr

            # 4. 电荷-势能协方差
            charge_potential_cov = np.nan
            if len(path) >= 2:
                charge_seq = bader_charges[list(path)]
                potential_seq = potential_at_atoms[list(path)]
                cov_matrix = np.cov(charge_seq, potential_seq)
                if cov_matrix.shape == (2, 2) and not np.isnan(cov_matrix[0, 1]):
                     charge_potential_cov = cov_matrix[0, 1]

            final_metrics.append({
                's_struct': row['s_struct'],
                'delta_chi': row['delta_chi'],
                'tau_max': row['tau_max'],
                'force_gradient': path_force_gradient,
                'curvature': path_curvature,
                'wrapping_norm': path_wrapping_norm,
                'structure_autocorr': structure_autocorr,
                'charge_potential_cov': charge_potential_cov
            })

        final_metrics_df = pd.DataFrame(final_metrics).dropna()

        # --- VI. 协方差计算 ---
        path_covariances = []
        for _, row in sampled_df.iterrows():
            structure_incompatibility_seq = row['structure_incompatibility_seq']
            torque_seq = row['torque_seq']

            if len(structure_incompatibility_seq) >= 2 and len(torque_seq) >= 1:
                structure_seq_truncated = structure_incompatibility_seq[:len(torque_seq)]
                if len(structure_seq_truncated) == len(torque_seq) and len(torque_seq) >= 2:
                    if not (np.any(np.isnan(structure_seq_truncated)) or np.any(np.isnan(torque_seq))):
                        try:
                            cov_matrix = np.cov(structure_seq_truncated, torque_seq)
                            if cov_matrix.shape == (2, 2) and not np.isnan(cov_matrix[0, 1]):
                                path_covariances.append(cov_matrix[0, 1])
                        except Exception as e:
                            print(f"  - 错误: 计算路径协方差失败: {e}")
                            continue
        
        # --- VII. 最终结果填充 ---
        if path_covariances:
            features['path_cov_torsional_stress_mean'] = np.mean(path_covariances)
            features['path_cov_torsional_stress_std'] = np.std(path_covariances)
            features['path_cov_torsional_stress_max'] = np.max(path_covariances)
        
        # 全面记录所有新特征的统计量
        feature_mapping = {
            'path_entropy': 's_struct',
            'path_chempot_diff': 'delta_chi',
            'path_max_torque': 'tau_max',
            'path_force_gradient': 'force_gradient',
            'path_curvature': 'curvature',
            'path_wrapping_norm': 'wrapping_norm',
            'path_structure_autocorr': 'structure_autocorr',
            'path_charge_potential_cov': 'charge_potential_cov'
        }
        for f_prefix, col_name in feature_mapping.items():
            if col_name in final_metrics_df.columns and not final_metrics_df[col_name].empty:
                features[f'{f_prefix}_mean'] = final_metrics_df[col_name].mean()
                features[f'{f_prefix}_std'] = final_metrics_df[col_name].std()
                features[f'{f_prefix}_max'] = final_metrics_df[col_name].max()
            
        # 性能监控报告
        d_part_end_time = time.time()
        d_part_duration = d_part_end_time - d_part_start_time
        print(f"D 部分特征计算完成。总耗时: {d_part_duration:.2f} 秒")
        if d_part_duration > 300:  # 如果超过5分钟
            warnings.warn(f"D部分计算耗时 {d_part_duration:.1f} 秒，建议检查是否需要进一步优化。")
        
        return features

    def _get_bond_properties_sequence(self, path: Tuple, nx_graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """辅助函数：为给定路径计算键属性（向量和image）序列。"""
        bond_props = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data_forward = nx_graph.get_edge_data(u, v)
            edge_data_backward = nx_graph.get_edge_data(v, u)

            bond_vector, image_vector = None, np.zeros(3)
            if edge_data_forward:
                edge_data = edge_data_forward[0]
                image_vector = edge_data['to_jimage']
                p_u = self.pmg_structure[u].coords
                p_v_image = self.pmg_structure.lattice.get_cartesian_coords(
                    self.pmg_structure[v].frac_coords + image_vector
                )
                bond_vector = p_v_image - p_u
            elif edge_data_backward:
                edge_data = edge_data_backward[0]
                # 注意：这里的image是从v到u的，所以u到v的image是它的负值
                image_vector = -np.array(edge_data['to_jimage'])
                p_v = self.pmg_structure[v].coords
                p_u_image = self.pmg_structure.lattice.get_cartesian_coords(
                    self.pmg_structure[u].frac_coords + edge_data['to_jimage']
                )
                bond_vector_vu = p_u_image - p_v
                bond_vector = -bond_vector_vu
            else:
                return [] 

            bond_props.append({'vector': bond_vector, 'image': image_vector})
        return bond_props

    def _get_torque_sequence(self, path: Tuple, nx_graph: nx.MultiDiGraph, forces: np.ndarray) -> List[float]:
        """辅助函数：为给定路径计算辛力矩序列。"""
        bond_props = self._get_bond_properties_sequence(path, nx_graph)
        if not bond_props:
            return []
            
        torque_seq = []
        for i in range(len(path) - 1):
            bond_vector = bond_props[i]['vector']
            force_u, force_v = forces[path[i]], forces[path[i+1]]
            relative_force = force_v - force_u
            torque = np.linalg.norm(np.cross(bond_vector, relative_force))
            torque_seq.append(torque)
        return torque_seq

    def _calculate_shannon_entropy(self, data_seq: np.ndarray, bins: int = 10) -> float:
        """辅助函数：计算序列的香农熵。"""
        if len(data_seq) < 2:
            return 0.0
        # 使用基于计数的概率计算，确保熵为非负
        counts, _ = np.histogram(data_seq, bins=bins)
        probabilities = counts / len(data_seq)
        # 过滤掉概率为0的项，避免log2(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))


    def _interpolate_grid_at_atomic_sites(self, grid_data: np.ndarray) -> np.ndarray:
        """
        通用插值函数，使用物理一致的坐标变换在原子位点插值网格数据。
        
        物理一致性增强
        ===================================
        主要改进：
        1. 使用晶格矩阵G进行坐标变换，适用于非正交晶格
        2. 网格密度检查：确保DFT网格足够精细以支持准确插值
        3. 边界处理：改善晶胞边界附近的插值精度
        4. 插值方法选择：根据数据特性选择最佳插值策略
        5. 采用分层插值策略，优先三次插值，对NaN点回退到线性
        """
        if grid_data is None:
            return np.full(len(self.pmg_structure), np.nan)
        
        # 检查网格密度是否足够精细
        min_grid_spacing = min([1.0/n for n in grid_data.shape])
        if min_grid_spacing > 0.1:  # 网格间距 > 0.1 埃的分数坐标单位
            warnings.warn(
                f"DFT网格密度可能不足以进行精确插值。"
                f"最小网格间距: {min_grid_spacing:.3f} (分数坐标单位)。"
                f"建议使用更精细的网格进行DFT计算。"
            )
        
        # 使用晶格矩阵进行物理一致的坐标变换
        # 获取晶格矩阵 G = [a, b, c]^T，用于分数坐标到笛卡尔坐标的变换
        lattice_matrix = self.pmg_structure.lattice.matrix
        
        # 为非正交晶格正确设置插值网格轴
        # 注意：GPAW网格在分数坐标空间中是规则的
        axes = [np.linspace(0, 1, n, endpoint=False) for n in grid_data.shape]
        
        # 获取所有原子的分数坐标，并确保分数坐标在[0,1)范围内
        frac_coords = self.pmg_structure.frac_coords
        frac_coords_wrapped = frac_coords % 1.0
        
        # 分层插值策略
        interpolated_values = np.full(len(self.pmg_structure), np.nan) # 初始化结果数组
        
        # 第一步：尝试三次插值
        try:
            cubic_interpolator = RegularGridInterpolator(
                axes, grid_data, 
                method='cubic',
                bounds_error=False, 
                fill_value=np.nan
            )
            cubic_results = cubic_interpolator(frac_coords_wrapped)
            # 将三次插值结果写入最终数组
            interpolated_values = cubic_results
            
        except ValueError as e:
            print(f"  - 错误: 初始三次插值失败: {e}。将尝试全局线性插值。")
            # 如果三次插值器初始化失败（例如，数据不适合三次插值），则直接回退到全局线性插值
            linear_interpolator = RegularGridInterpolator(
                axes, grid_data, 
                method='linear',
                bounds_error=False, 
                fill_value=np.nan
            )
            interpolated_values = linear_interpolator(frac_coords_wrapped)
            warnings.warn("由于三次插值初始化失败，已回退到全局线性插值。")

        # 第二步：针对在三次插值中产生NaN的位点，回退到线性插值
        nan_indices = np.where(np.isnan(interpolated_values))[0]
        if len(nan_indices) > 0:
            print(f"  - 警告: 三次插值产生了 {len(nan_indices)} 个NaN值。对这些位点尝试线性插值。")
            # 打印导致NaN的原子位点索引和分数坐标，以便进一步诊断
            print(f"  - 发现NaN值的原子位点索引: {nan_indices}")
            print(f"  - 这些位点的分数坐标 (wrapped): {frac_coords_wrapped[nan_indices]}")
            try:
                linear_interpolator = RegularGridInterpolator(
                    axes, grid_data, 
                    method='linear',
                    bounds_error=False, 
                    fill_value=np.nan
                )
                # 仅对产生NaN的位点进行线性插值
                linear_fallback_results = linear_interpolator(frac_coords_wrapped[nan_indices])
                # 将线性插值结果填充回原数组
                interpolated_values[nan_indices] = linear_fallback_results
                
                # 再次检查，报告最终仍然为NaN的数量
                final_nan_count = np.where(np.isnan(interpolated_values))[0].size
                if final_nan_count > 0:
                    warnings.warn(f"分层插值后仍有 {final_nan_count} 个NaN值，可能影响特征质量。")
                else:
                    print("  - 成功: 所有三次插值产生的NaN值已通过线性插值修复。")

            except Exception as e:
                print(f"  - 错误: 线性插值回退过程失败: {e}")
                # 此时 interpolated_values 已经包含了cubic和部分NaN，不再进行额外操作
                warnings.warn(f"线性插值回退过程异常，插值结果可能包含未修复的NaN值。")

        # 确保返回值是一维数组，并处理可能由插值产生的维度问题
        if interpolated_values.ndim > 1:
            interpolated_values = interpolated_values.flatten()
            if len(interpolated_values) != len(self.pmg_structure):
                new_values = np.full(len(self.pmg_structure), np.nan)
                new_values[:len(interpolated_values)] = interpolated_values[:len(new_values)]
                interpolated_values = new_values

        # 最终NaN值处理，作为"安全网"
        # 如果经过三次和线性插值回退后仍然存在NaN，则用非NaN的平均值填充
        # 这确保了特征的数值完整性，避免NaN在后续计算中传播
        if np.any(np.isnan(interpolated_values)):
            valid_values = interpolated_values[~np.isnan(interpolated_values)]
            if len(valid_values) > 0:
                mean_valid_value = np.mean(valid_values)
                interpolated_values[np.isnan(interpolated_values)] = mean_valid_value
                warnings.warn(f"最终有 {np.where(np.isnan(interpolated_values))[0].size} 个NaN值被替换为有效值的平均值 ({mean_valid_value:.4f})。")
            else:
                # 如果所有插值都失败，则用0填充（作为极端情况下的默认值）
                interpolated_values[np.isnan(interpolated_values)] = 0.0
                warnings.warn("所有插值值均为NaN，已全部替换为0.0。")


        return interpolated_values

    def _find_paths_with_random_walk(self, nx_graph: nx.MultiDiGraph, num_walks: int = 1000, max_walk_length: int = None) -> Set[Tuple[int, ...]]:
        """
        使用随机行走策略高效采样图中的路径。
        
        替代完全枚举的高效方案
        ===============================================
        相比于_find_all_simple_paths的指数级复杂度，随机行走提供了线性时间的路径采样：
        - 时间复杂度：O(num_walks * max_walk_length) vs O(指数级)
        - 空间复杂度：O(采样路径数) vs O(所有可能路径数)
        - 适用于大型超胞和复杂晶体图
        
        策略：
        1. 从每个节点开始多次随机行走
        2. 确保路径的简单性（无重复节点）
        3. 自然的重要性采样：高连通性区域被更频繁采样
        
        Parameters:
        -----------
        nx_graph : nx.MultiDiGraph
            晶体图
        num_walks : int
            总的随机行走次数
        max_walk_length : int
            单次行走的最大长度，默认为TOPOLOGICAL_PATH_MAX_LENGTH
            
        Returns:
        --------
        Set[Tuple[int, ...]]
            采样得到的简单路径集合
        """
        if max_walk_length is None:
            max_walk_length = self.GRAPH_PATH_MAX_LENGTH
            
        unique_paths = set()
        nodes = list(nx_graph.nodes)
        
        if not nodes:
            return unique_paths
            
        for walk_idx in range(num_walks):
            # 进度报告 (每1000次行走报告一次)
            if walk_idx > 0 and walk_idx % 1000 == 0:
                print(f"  - 随机行走进度: {walk_idx}/{num_walks} ({100*walk_idx/num_walks:.1f}%) 完成.")

            # 使用可控随机数生成器
            start_node = self.rng.choice(nodes)
            current_path = [start_node]
            current_node = start_node
            
            # 进行随机行走
            for step in range(max_walk_length - 1):
                # 使用鲁棒的邻居查找方法
                connected_neighbors_info = self._get_nn_info_robust(self.pmg_structure, current_node)
                neighbors = [nn['site_index'] for nn in connected_neighbors_info]
                
                if not neighbors:
                    break
                    
                # 过滤已访问的邻居，确保路径简单性
                unvisited_neighbors = [n for n in neighbors if n not in current_path]
                if not unvisited_neighbors:
                    break
                    
                # 使用可控随机数生成器
                next_node = self.rng.choice(unvisited_neighbors)
                current_path.append(next_node)
                current_node = next_node
            
            # 只保留长度>=3的路径
            if len(current_path) >= 3:
                path_tuple = tuple(current_path)
                # 避免重复保存正反向路径
                if tuple(reversed(path_tuple)) not in unique_paths:
                    unique_paths.add(path_tuple)
                    
        return unique_paths

    def _find_all_simple_paths_with_early_stop(self, nx_graph: nx.MultiDiGraph) -> Set[Tuple[int, ...]]:
        """
        带早期停止机制的路径搜索。
        
        关键改进：
        ==========
        1. 早期停止：达到self.PATH_SAMPLING_SIZE（目标采样数）即停止，避免无意义的过度采样
        2. 智能源节点选择：基于节点度数进行优先级排序
        3. 路径长度自适应：动态调整搜索深度
        4. 重复检测优化：使用更高效的集合操作
        
        时间复杂度：从O(指数)降低到O(max_paths * log(max_paths))
        """
        unique_paths = set()
        nodes = list(nx_graph.nodes)
        
        # 按节点度数排序，优先处理高连通性节点
        nodes_by_degree = sorted(nodes, key=lambda n: nx_graph.degree(n), reverse=True)
        
        # 自适应路径长度：根据图规模动态调整
        adaptive_max_length = min(self.GRAPH_PATH_MAX_LENGTH, max(3, int(len(nodes) / 10)))
        
        # 使用全局设置的采样大小作为停止条件
        max_paths_to_find = self.PATH_SAMPLING_SIZE # 确保只寻找所需数量的路径
        
        for source in nodes_by_degree:
            if len(unique_paths) >= max_paths_to_find:
                print(f"  - 早期停止：已找到 {len(unique_paths)} 条路径，达到目标采样数")
                break
                
            # 只搜索度数较高的目标节点，提高路径质量
            target_candidates = [n for n in nodes_by_degree[:min(20, len(nodes))] if n != source]
            
            for target in target_candidates:
                if len(unique_paths) >= max_paths_to_find:
                    break
                    
                try:
                    # 恢复使用NetworkX内置的all_simple_paths函数
                    # nx_graph 已经通过 CrystalNN 构建，确保了图只包含物理键
                    # NetworkX的all_simple_paths函数是高度优化的，并且能够根据cutoff参数有效控制路径长度
                    paths = nx.all_simple_paths(nx_graph, source, target, cutoff=adaptive_max_length)

                    for path in paths:
                        if len(unique_paths) >= max_paths_to_find:
                            break
                            
                        if len(path) < 3: 
                            continue
                            
                        path_tuple = tuple(path)
                        path_tuple_rev = tuple(reversed(path_tuple))
                        
                        # 高效的重复检测
                        if path_tuple not in unique_paths and path_tuple_rev not in unique_paths:
                            unique_paths.add(path_tuple)
                            
                except nx.NetworkXNoPath as e:
                    print(f"  - 警告: 在 {source}->{target} 之间没有找到路径，可能由于图不连通或路径长度限制: {e}")
                    continue # 如果没有路径，继续下一个目标
                except Exception as e:
                    # 处理其他可能的NetworkX异常
                    print(f"  - 错误: 路径搜索异常 {source}->{target}: {e}")
                    continue
                        
        return unique_paths


    def _calculate_quotient_and_geometric_features(self) -> Tuple[Dict[str, Any], List[np.ndarray]]:
        """计算商代数特征和与几何相关的李代数特征。"""
        features = {}
        sga = SpacegroupAnalyzer(self.pmg_structure)
        # 统一使用CrystalNN
        voro = self.crystal_nn 
        
        # 1. structure_hash (商)
        # 使用pymatgen的Weisfeiler-Lehman图哈希方法
        try:
            # 使用CrystalNN作为键合策略来构建结构图
            crystal_nn_strategy = CrystalNN()
            graph = StructureGraph.from_local_env_strategy(self.pmg_structure, crystal_nn_strategy)
            # 使用正确的函数名
            features['structure_hash'] = weisfeiler_lehman_graph_hash(graph.graph)
        except Exception as e:
            print(f"  - 错误: 图哈希计算失败: {e}。使用备用的字典哈希方法。")
            struct_dict = self.pmg_structure.as_dict()
            struct_json = json.dumps(struct_dict, sort_keys=True)
            features['structure_hash'] = hashlib.md5(struct_json.encode()).hexdigest()

        # 3. 新: symmetry_orbit_connectivity (商)
        try:
            symmetrized_structure = sga.get_symmetrized_structure()
            equivalent_indices = symmetrized_structure.equivalent_indices
            
            site_to_orbit_map = {}
            for i, orbit in enumerate(equivalent_indices):
                for site_idx in orbit:
                    site_to_orbit_map[site_idx] = i

            graph = StructureGraph.from_local_env_strategy(self.pmg_structure, self.crystal_nn)
            
            intra_orbit_bonds = 0
            inter_orbit_bonds = 0
            counted_edges = set()

            for u, v, data in graph.graph.edges(data=True):
                edge_key = tuple(sorted((u, v)))
                if edge_key in counted_edges: continue
                counted_edges.add(edge_key)

                if site_to_orbit_map[u] == site_to_orbit_map[v]:
                    intra_orbit_bonds += 1
                else:
                    inter_orbit_bonds += 1
            
            total_bonds = intra_orbit_bonds + inter_orbit_bonds
            features['symmetry_orbit_connectivity'] = inter_orbit_bonds / total_bonds if total_bonds > 0 else 0.0

        except Exception as e:
            print(f"  - 错误: 对称轨道连通性计算失败: {e}。该特征值将设为0。")
            features['symmetry_orbit_connectivity'] = 0.0
        # 4. global_asymmetry_norm (李)
        # 使用鲁棒的邻居查找方法获取所有原子的邻居信息
        all_nn_info_robust = []
        for i in range(len(self.pmg_structure)):
            all_nn_info_robust.append(self._get_nn_info_robust(self.pmg_structure, i))

        local_asymmetry_vectors_cart = []
        for i in range(len(self.pmg_structure)): # 遍历所有原子，即使没有邻居也会被 _get_V_struct 处理
            # 使用新的辅助函数计算每个位点的V_struct
            v_struct_cart = self._get_V_struct(i)
            local_asymmetry_vectors_cart.append(v_struct_cart)
        
        global_asymmetry_vector = np.sum(local_asymmetry_vectors_cart, axis=0)
        global_asymmetry_norm = np.linalg.norm(global_asymmetry_vector)
        
        # 保持原始物理量，适用于跨材料数据库比较
        features['global_asymmetry_norm'] = global_asymmetry_norm
        
        # 修正 global_asymmetry_norm：计算所有原子局部不对称向量范数的均值
        # 这更能反映晶体整体的"平均局部不对称性"或"平均局部扭曲程度"，避免局部向量相互抵消
        individual_asymmetry_norms = [np.linalg.norm(v) for v in local_asymmetry_vectors_cart]
        features['global_asymmetry_norm'] = np.mean(individual_asymmetry_norms) if individual_asymmetry_norms else 0.0
        
        # 2. lie_asymmetry_magnitude_entropy (商) - 替换 wyckoff_position_entropy
        # 添加新的 lie_asymmetry_magnitude_entropy，确保在 local_asymmetry_vectors_cart 之后计算
        features['lie_asymmetry_magnitude_entropy'] = self._calculate_lie_asymmetry_magnitude_entropy(local_asymmetry_vectors_cart)

        # 返回特征字典和局部不对称矢量（笛卡尔坐标）
        return features, local_asymmetry_vectors_cart

    def _calculate_local_torsional_stress_features(self, local_asymmetry_vectors: List[np.ndarray]) -> Dict[str, Any]:
        """
        计算基于原子力的局部扭转应力特征。
        
        术语修正：这些特征基于so(3)李代数结构的几何-力学耦合，
        更准确地称为"局部扭转应力"而非"李代数特征"。
        """
        features = {}
        
        if self.gpaw_calc is None:
            warnings.warn("无法计算基于力的特征，因为没有有效的 GPAW 计算器。")
            return {
                'force_covariance_invariant_1': np.nan,
                'force_covariance_invariant_2': np.nan,
                'total_torsional_stress': np.nan,
            }

        forces = self.ase_structure.get_forces()
        num_atoms = len(self.ase_structure)
        
        if num_atoms > 0:
            # 单位换算和数值稳定性优化
            # 力的单位通常是 eV/Å，转换为更合适的单位以提高数值稳定性
            
            # 计算力的RMS（均方根）值作为特征尺度
            force_rms = np.sqrt(np.mean(forces.flatten()**2))
            
            # 使用标准的协方差计算，保持原始物理意义
            force_cov_matrix = (forces.T @ forces) / num_atoms
            features['force_covariance_invariant_1'] = np.trace(force_cov_matrix)
            features['force_covariance_invariant_2'] = np.linalg.det(force_cov_matrix)
        else:
            features['force_covariance_invariant_1'] = 0.0
            features['force_covariance_invariant_2'] = 0.0
            
        # 计算局部扭转应力：||V_i × f_i|| (物理意义增强版)
        torsions = []
        for V_i, f_i in zip(local_asymmetry_vectors, forces):
            cross_product = np.cross(V_i, f_i)
            torsion_magnitude = np.linalg.norm(cross_product)
            torsions.append(torsion_magnitude)
        
        total_torsion = np.sum(torsions)
        # 保持原始物理数值，不进行人为缩放
        features['total_torsional_stress'] = total_torsion
        
        return features

    def _calculate_field_density_coupling_features(self) -> Dict[str, Any]:
        """
        基于场-梯度耦合思想，从基态电子结构计算场-密度梯度耦合张量及其不变量。
        
        注意：
        - 更正术语：将"电子应力张量"改为"场-密度梯度耦合张量"
        - 这个张量 T_ij = E_i * ∇_j ρ 描述了静电场与电子密度梯度的耦合
        - 虽然与真实的物理应力张量相关，但不具备应力张量的所有物理性质
        - 其不变量可能与材料的介电响应和电子结构稳定性相关
        """
        features = {
            'field_density_coupling_invariant_1': np.nan,
            'field_density_coupling_invariant_2': np.nan,
            'total_gradient_norm': np.nan,
        }
        
        if self.gpaw_calc is None:
            print("错误: 无法计算场-密度梯度耦合特征，因为没有有效的 GPAW 计算器。")
            return features

        try:
            # 1. 获取基础场: 电子密度和静电势
            density_data = self.gpaw_calc.get_pseudo_density()
            potential = self.gpaw_calc.get_electrostatic_potential()
            
            # 确保电子密度网格与静电势网格一致 (通过插值)
            # GPAW的get_pseudo_density可能在粗网格上，而get_electrostatic_potential在细网格上
            if density_data.shape != potential.shape:
                print(f"  - 警告: 电子密度网格 {density_data.shape} 与静电势网格 {potential.shape} 不匹配。进行上采样...")
                zoom_factors = [pot_dim / den_dim for pot_dim, den_dim in zip(potential.shape, density_data.shape)]
                if all(f.is_integer() and f > 1 for f in zoom_factors):
                    # 使用三次插值进行上采样
                    density_data = zoom(density_data, zoom=zoom_factors, order=3)
                    print(f"  - 电子密度已上采样到新形状: {density_data.shape}")
                else:
                    raise ValueError(f"无法自动调整网格大小，Zoom因子不是整数倍: {zoom_factors}")

            # 2. 计算梯度场
            grad_potential = np.stack(np.gradient(potential), axis=0)
            electric_field = -grad_potential
            grad_density = np.stack(np.gradient(density_data), axis=0)

            # 3. 计算电子应力张量
            # 确保电场和密度梯度网格形状一致
            if electric_field.shape != grad_density.shape:
                raise ValueError(
                    f"错误: 电子场 ({electric_field.shape}) 和电子密度梯度 ({grad_density.shape}) 网格形状不匹配。"
                    "请检查GPAW计算的网格设置，确保所有物理场在一致的网格上生成。"
                )
            
            num_grid_points = np.prod(electric_field.shape[1:])
            if num_grid_points == 0:
                raise ValueError("电场格点数为零，无法进行平均。")
            electronic_stress_tensor = np.einsum('ixyz,jxyz->ij', electric_field, grad_density) / num_grid_points
            
            # 4. 计算张量不变量
            features['field_density_coupling_invariant_1'] = np.trace(electronic_stress_tensor)
            features['field_density_coupling_invariant_2'] = np.linalg.det(electronic_stress_tensor)

            # 5. 计算总梯度范数
            grad_density_sq_norm = np.sum(grad_density**2, axis=0)
            features['total_gradient_norm'] = np.mean(grad_density_sq_norm)

        except Exception as e:
            print(f"  - 错误: 计算场-密度梯度耦合特征时发生异常: {e}")
            
            print("相关特征值将设为NaN。")

        return features

    def _calculate_bader_charge_spatial_variance(self) -> Dict[str, Any]:
        """
        计算Bader电荷的空间方差。
        该特征量化了晶体中原子Bader电荷值的空间分布均匀性。
        对于对称结构，电荷分布趋于均匀，方差应接近零。
        对于扭曲或不对称结构，电荷分布可能不均匀，方差将显著非零。
        这提供了一个更稳健、更直接的衡量结构-电子耦合带来的"涨落"的指标。
        """
        features = {'bader_charge_spatial_variance': np.nan}
        
        # 验证必需的列是否存在
        if 'bader_charge' not in self.local_features_df.columns:
            warnings.warn("0-单纯形CSV文件中未找到 'bader_charge' 列，无法计算Bader电荷空间方差。")
            return features

        bader_charges = self.local_features_df['bader_charge'].values
        
        if len(bader_charges) < 2:
            warnings.warn("Bader电荷数据点不足2个，无法计算方差。")
            return features
        
        # 过滤掉NaN值，确保计算的鲁棒性
        valid_charges = bader_charges[~np.isnan(bader_charges)]
        
        if len(valid_charges) < 2:
            warnings.warn("有效Bader电荷数据点不足2个，无法计算方差。")
            return features
            
        features['bader_charge_spatial_variance'] = np.var(valid_charges)
        
        return features

    def _get_nn_info_robust(self, structure: Structure, site_idx: int) -> List[Dict[str, Any]]:
        """
        鲁棒地获取指定原子的CrystalNN邻居信息，包含重试机制。
        """
        neighbors_found = []
        try:
            neighbors_found = self.crystal_nn.get_nn_info(structure, site_idx)
        except Exception as e:
            # 初始尝试失败，打印警告但不终止
            print(f"  - 警告: Site {site_idx} 初始邻居识别失败: {e}")

        # 如果初始尝试失败或未找到邻居，尝试略微放宽容差重试
        if not neighbors_found:
            for attempt in range(1, 3): # 最多尝试2次重试
                try:
                    temp_crystal_nn = CrystalNN(x_diff_tol=0.05 + attempt * 0.02)
                    neighbors_retry = temp_crystal_nn.get_nn_info(structure, site_idx)
                    if neighbors_retry:
                        print(f"  - 成功: Site {site_idx} 在第 {attempt+1} 次尝试 (dist_tol={0.05 + attempt * 0.02:.2f}) 后找到邻居。")
                        neighbors_found = neighbors_retry
                        break # 找到即退出重试循环
                except Exception as e_retry:
                    print(f"  - 警告: Site {site_idx} 第 {attempt+1} 次尝试邻居识别失败: {e_retry}")
        
        if not neighbors_found:
            print(f"  - 错误: 无法为Site {site_idx} 识别任何邻居，已返回空列表。")
        
        return neighbors_found

    def _calculate_lie_asymmetry_magnitude_entropy(self, local_asymmetry_vectors: List[np.ndarray]) -> float:
        """
        计算基于原子局部不对称性向量范数的熵。
        该特征量化了晶体中原子局部结构不对称性程度分布的多样性。
        高熵值表示原子局部不对称性程度分布广泛，结构无序或复杂；
        低熵值表示原子局部不对称性程度相似，结构更规整。
        """
        if not local_asymmetry_vectors:
            warnings.warn("未找到局部不对称性向量，李代数不对称性范数熵将为0。")
            return 0.0

        # 计算每个向量的范数
        asymmetry_magnitudes = np.array([np.linalg.norm(vec) for vec in local_asymmetry_vectors])

        # 如果所有范数都相同（例如都是0），则熵为0
        if np.allclose(asymmetry_magnitudes, asymmetry_magnitudes[0], atol=self._TOLERANCE):
            return 0.0

        # 确定直方图的bin数量
        # 使用 'auto' 策略让 numpy.histogram 自动选择最佳bin宽度
        try:
            counts, _ = np.histogram(asymmetry_magnitudes, bins='auto', density=False)
        except Exception as e:
            warnings.warn(f"计算不对称性范数直方图失败: {e}。将尝试固定bin数量。")
            counts, _ = np.histogram(asymmetry_magnitudes, bins=10, density=False) # Fallback to fixed 10 bins

        # 计算概率
        total_magnitudes = np.sum(counts)
        if total_magnitudes == 0:
            return 0.0

        probabilities = counts / total_magnitudes
        probabilities = probabilities[probabilities > 0] # 过滤掉零概率

        if not probabilities.size > 0:
            return 0.0

        entropy = -sum(p * np.log2(p) for p in probabilities)
        return float(entropy)

    def _calculate_symplectic_fluctuation_volume(self) -> Dict[str, Any]:
        """
        基于"结构-电子"耦合思想，计算全局伪辛涨落体积（Pseudo-Symplectic Fluctuation Volume）。
        
        该特征采用了辛几何的构造思想作为物理类比，但并非严格的辛流形结构。
        为确保数学严谨性，将其重命名为"伪辛"或"类相空间"体积，明确其类比性质。
        
        物理类比构造：
        - 广义坐标 (q): 化学键的键向量，代表结构的几何自由度。
        - 广义动量 (p): 在化学键中点处的电子密度梯度，代表电子行为的动量。
        
        最终的特征值是此 (q, p) 代理相空间中状态分布协方差矩阵的行列式。
        此版本返回行列式的对数，以避免数值下溢，并更准确地反映其在不同量级上的变化。
        """
        features = {'log_pseudo_symplectic_fluctuation_volume': np.nan}
        if self.gpaw_calc is None:
            warnings.warn("警告: 无法计算伪辛涨落体积，因为没有有效的 GPAW 计算器。") # 修正：错误 -> 警告
            return features

        try:
            # 1. 获取电子密度及其梯度场
            density_data = self.gpaw_calc.get_pseudo_density()
            grad_density = np.stack(np.gradient(density_data), axis=0)

            # 2. 为梯度场的每个分量创建插值器
            axes = [np.linspace(0, 1, n, endpoint=False) for n in grad_density.shape[1:]]
            grad_interpolators = [
                RegularGridInterpolator(axes, grad_density[i], method='linear', bounds_error=False, fill_value=np.nan)
                for i in range(3)
            ]

            # 3. 遍历所有唯一的化学键，构建相空间状态矢量
            graph = StructureGraph.from_local_env_strategy(self.pmg_structure, self.crystal_nn)
            state_vectors = []
            processed_bonds = set()

            for u, v, data in graph.graph.edges(data=True):
                if tuple(sorted((u, v))) in processed_bonds:
                    continue
                processed_bonds.add(tuple(sorted((u, v))))

                site_u, site_v = self.pmg_structure[u], self.pmg_structure[v]
                
                # a. 计算键向量 (广义坐标 q)
                bond_vector = self.pmg_structure.lattice.get_cartesian_coords(
                    site_v.frac_coords + data['to_jimage']
                ) - site_u.coords
                
                # b. 计算键中点的分数坐标
                midpoint_frac = site_u.frac_coords + (site_v.frac_coords + data['to_jimage'] - site_u.frac_coords) / 2.0
                
                # c. 插值得到电子密度梯度 (广义动量 p)
                # 确保分数坐标在[0,1)范围内并且是正确的形状
                wrapped_frac = midpoint_frac % 1.0
                if wrapped_frac.ndim == 0:
                    wrapped_frac = np.array([wrapped_frac])
                elif wrapped_frac.shape[0] != 3:
                    # 如果不是3D坐标，跳过这个键
                    continue
                    
                try:
                    # 逐个计算每个分量的插值，确保返回标量
                    grad_components = []
                    for i, interp in enumerate(grad_interpolators):
                        grad_val = interp(wrapped_frac)
                        # 确保是标量
                        if np.isscalar(grad_val):
                            grad_components.append(grad_val)
                        elif hasattr(grad_val, '__len__') and len(grad_val) == 1:
                            grad_components.append(grad_val[0])
                        else:
                            grad_components.append(float(grad_val)) # 确保转换为浮点数，即使是单个数组元素
                    
                    grad_rho_at_midpoint = np.array(grad_components)
                    
                    # 确保键向量和梯度都是3D向量
                    if bond_vector.shape[0] == 3 and grad_rho_at_midpoint.shape[0] == 3:
                        state_vector = np.hstack([bond_vector, grad_rho_at_midpoint])
                        state_vectors.append(state_vector)
                        
                except Exception as interp_error:
                    # 如果插值失败，跳过这个键
                    warnings.warn(f"  - 警告: 键 {u}-{v} 的插值失败: {interp_error}。跳过该键。")
                    continue

            if not state_vectors: # 如果没有有效的键向量，直接返回NaN
                warnings.warn("未找到化学键或有效状态向量，无法计算伪辛涨落体积。")
                return features

            # 4. 计算6x6协方差矩阵的行列式（广义方差）
            if len(state_vectors) < 2:
                warnings.warn(f"有效化学键数量不足 ({len(state_vectors)})，无法计算协方差。")
                return features
            
            state_matrix = np.array(state_vectors)
            
            # 引入标准化操作，使其在不同量纲间可比
            standardized_state_matrix = np.zeros_like(state_matrix, dtype=float)
            for col_idx in range(state_matrix.shape[1]): # 遍历6个维度 (qx, qy, qz, px, py, pz)
                col_data = state_matrix[:, col_idx]
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                
                if std_val > self._TOLERANCE: # 避免除以零，使用更严格的阈值
                    standardized_state_matrix[:, col_idx] = (col_data - mean_val) / std_val
                else:
                    # 如果标准差为零 (所有值相同)，则该维度标准化后为零
                    # 这表示该维度上没有涨落，其贡献为0
                    standardized_state_matrix[:, col_idx] = 0.0 
            
            # 检查标准化后是否有无效值
            if np.any(np.isnan(standardized_state_matrix)) or np.any(np.isinf(standardized_state_matrix)):
                warnings.warn("标准化后的状态矩阵包含NaN或无穷大值，无法计算协方差。")
                return features
                
            try:
                # 对标准化后的数据计算协方差矩阵
                covariance_matrix = np.cov(standardized_state_matrix, rowvar=False)
                
                # 添加一个小的正扰动到对角线，确保协方差矩阵非奇异
                # 这能够避免行列式为零的问题，使得特征能够捕捉微弱的涨落
                # 使用一个比通常机器精度大一点的容差，但足够小以不影响物理意义
                covariance_matrix += np.eye(covariance_matrix.shape[0]) * (self._TOLERANCE * 100) # 稍微增大扰动以提高鲁棒性
                
                # 检查协方差矩阵的有效性
                if np.any(np.isnan(covariance_matrix)) or np.any(np.isinf(covariance_matrix)):
                    warnings.warn("协方差矩阵包含NaN或无穷大值。")
                    return features
                    
                # 计算行列式并取对数
                sign, logdet = np.linalg.slogdet(covariance_matrix)
                
                # 直接返回logdet，避免数值下溢
                if sign > 0 and np.isfinite(logdet): # 确保行列式为正且对数有效
                    features['log_pseudo_symplectic_fluctuation_volume'] = logdet
                else:
                    # 如果行列式为零或负数 (数值误差或真正奇异)，则返回负无穷或一个极小的负值
                    features['log_pseudo_symplectic_fluctuation_volume'] = -np.inf # 表示体积为0
                    
            except np.linalg.LinAlgError as linalg_error:
                warnings.warn(f"线性代数计算失败 (伪辛涨落体积): {linalg_error}")
                return features

        except Exception as e:
            warnings.warn(f"错误: 计算伪辛涨落体积时发生异常: {e}")
            
        return features

def main():
    """
    主函数：用于测试全局特征计算。
    """



    # 使用实际CIF文件路径
    cif_file = "CsPbI3-supercell-optimized.cif"
    base_name = Path(cif_file).stem.replace("-optimized", "")    



        # 构建完整的文件路径
    atomic_features_csv = f"{base_name}-0-Simplex-Features.csv"
    bond_features_csv = f"{base_name}-1-Simplex-Features.csv"
    pw_gpw_file = f"{base_name}.gpw"  # PW模式文件
    fd_gpw_file = f"{base_name}-fd.gpw"  # FD模式文件（如果有的话）

    print("=" * 60)
    print(f"全局特征计算器 (A、B、C、D 部分) - 处理 {Path(cif_file).name}")
    print("=" * 60)

    try:
        calculator = GlobalFeatureCalculator(
            cif_file_path=str(cif_file),
            atomic_features_csv=str(atomic_features_csv),
            bond_features_csv=str(bond_features_csv),
            pw_gpw_file=str(pw_gpw_file),
            fd_gpw_file=str(fd_gpw_file)
        )

        features_df = calculator.calculate_features()

        print("\n全局特征 (A、B、C、D部分):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print(features_df)

        output_path = f'{base_name}-Global-Features.csv'
        features_df.to_csv(output_path, index=False)
        print(f"\n特征已保存到: {output_path}")

    except (ValueError, KeyError, RuntimeError) as e:
        print(f"\n错误: {e}")
        

if __name__ == '__main__':
    main() 