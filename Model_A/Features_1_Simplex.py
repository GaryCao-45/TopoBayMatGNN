import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict, Set

from pymatgen.core import Structure, Molecule, Element
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.symmetry.analyzer import PointGroupAnalyzer # 新增：用于局部对称性分析
import networkx as nx

# GPAW 和 SciPy
from gpaw import GPAW # type: ignore
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import laplace

warnings.filterwarnings('ignore', category=UserWarning, module='pymatgen')
warnings.filterwarnings('ignore', category=FutureWarning)


class BondFeatureCalculator:
    """
    根据严格局部性原则计算1-单纯形（边或键）特征的计算器。
    该实现严格遵循 "Features_1_Simplex_Final_Plan.txt" 伪代码方案 V2.0。
    """
    _TOLERANCE = 1e-9  # 用于数值计算中的容差

    # ==========================================================================
    # 主计算器类结构 - 初始化
    # ==========================================================================
    def __init__(self, cif_file: str,
                 atomic_features_csv: str,
                 atomic_tensors_csv: str,
                 pw_gpw_file: Optional[str] = None,
                 fd_gpw_file: Optional[str] = None):
        """
        初始化计算器，加载所有输入文件并进行预计算。
        """
        print("="*80)
        print("初始化 BondFeatureCalculator (遵循伪代码 V2.0)")
        print("="*80)

        # 1. 加载所有输入文件
        print(f"1. 加载输入文件...")
        self.pmg_structure = Structure.from_file(cif_file)
        print(f"   - CIF: {cif_file} ({self.pmg_structure.composition.reduced_formula})")
        self.atomic_features = pd.read_csv(atomic_features_csv)
        print(f"   - 0-单纯形特征: {atomic_features_csv}")
        self.atomic_tensors = self._load_atomic_tensors(atomic_tensors_csv)
        print(f"   - 0-单纯形张量: {atomic_tensors_csv}")
        self._calculate_and_cache_local_vfe() # 新增：计算并缓存VFE
        self.pw_calc = self._load_gpaw_calc(pw_gpw_file, "平面波(PW)")
        self.fd_calc = self._load_gpaw_calc(fd_gpw_file, "有限差分(FD)")

        # 2. 初始化核心分析工具
        print("2. 初始化核心分析工具...")
        # 【委员会最终修正】为了完全移除硬编码参数，我们将让CrystalNN使用其默认的、化学智能的键识别逻辑。
        # 此举避免了向其传递任何参数，从而绕过Pymatgen内部的KeyError: 0问题，
        # 同时也满足了“不硬编码参数”的约束。
        self.neighbor_finder = CrystalNN()

        # 3. 预计算与缓存 (关键性能步骤)
        print("3. 预计算与缓存...")
        # 构建一个包含所有潜在相互作用的综合拓扑图
        self.topology_graph = self._build_comprehensive_topology_graph()
        # 将Pymatgen图转换为NetworkX简单图，用于高效的局部拓扑计算
        self.simple_graph = self._convert_structure_graph_to_networkx(self.topology_graph)

        # 缓存量子化学场的插值器
        self.interpolators = self._setup_all_interpolators()
        print("初始化完成。\n")

    def _load_gpaw_calc(self, gpw_file: Optional[str], mode_name: str) -> Optional[GPAW]:
        if not gpw_file or not Path(gpw_file).exists():
            print(f"   - {mode_name} gpw 文件未提供或不存在。")
            return None
        print(f"   - 正在加载 {mode_name} GPAW: {gpw_file}")
        return GPAW(str(gpw_file), txt=None)

    def _load_atomic_tensors(self, csv_path: str) -> Dict[int, np.ndarray]:
        tensors_df = pd.read_csv(csv_path)
        tensors_dict = {}
        for _, row in tensors_df.iterrows():
                    site_idx = int(row['site_index'])
                    tensor = np.array([
                        [row['T_struct_00'], row['T_struct_01'], row['T_struct_02']],
                        [row['T_struct_10'], row['T_struct_11'], row['T_struct_12']],
                        [row['T_struct_20'], row['T_struct_21'], row['T_struct_22']]
                    ])
                    tensors_dict[site_idx] = tensor
        return tensors_dict

    def _build_comprehensive_topology_graph(self) -> StructureGraph:
        """
        使用CrystalNN构建基础拓扑图。
        【委员会最终修正】直接使用StructureGraph.with_local_env_strategy，让CrystalNN自动识别键。
        这利用了CrystalNN内部的化学智能逻辑，避免了硬编码参数。
        """
        print("   - 构建综合拓扑图 (StructureGraph)...")
        # 使用 weight='solid_angle' 可以提供更丰富的几何信息 (可选)
        return StructureGraph.with_local_env_strategy(self.pmg_structure, self.neighbor_finder)

    def _convert_structure_graph_to_networkx(self, structure_graph: StructureGraph) -> nx.Graph:
        """
        将Pymatgen StructureGraph转换为NetworkX简单图，移除多重边和自环。
        """
        # G.graph已经是MultiDiGraph，直接转换为Graph可以去除方向和平行边
        return nx.Graph(structure_graph.graph)

    def _setup_all_interpolators(self) -> Dict[str, Optional[RegularGridInterpolator]]:
        """
        设置所有需要的场插值器。
        """
        print("   - 设置网格场插值器...")
        calc = self.pw_calc or self.fd_calc
        if not calc:
            print("     - 警告: 无可用GPAW计算，所有量子特征将为NaN。")
            return {
                'density': None, 'density_laplacian': None,
                'potential': None, 'density_gradient': None
            }

        interpolators = {
            'density': self._setup_grid_interpolator('density', calc),
            'density_laplacian': self._setup_grid_interpolator('density_laplacian', calc),
            'potential': self._setup_grid_interpolator('electrostatic_potential', calc),
            'density_gradient': self._setup_grid_interpolator('density_gradient', calc, is_vector=True),
        }
        return interpolators

    def _setup_grid_interpolator(self, field: str, calc: GPAW, is_vector: bool = False) -> Optional[RegularGridInterpolator]:
        """为给定的标量或矢量场创建插值器。"""
        try:
            grid_data = None
            if field == 'density':
                grid_data = calc.get_all_electron_density()
            elif field == 'density_laplacian':
                density = calc.get_all_electron_density()
                grid_data = laplace(density)
            elif field == 'electrostatic_potential':
                grid_data = calc.get_electrostatic_potential()
            elif field == 'density_gradient':
                density = calc.get_all_electron_density()
                # 计算分数坐标下的梯度
                grad_frac = np.gradient(density)
                # 转换为笛卡尔坐标下的梯度
                lattice_inv_T = np.linalg.inv(self.pmg_structure.lattice.matrix).T
                # 使用einsum进行高效的张量缩并
                grid_data = np.einsum('i...,ij->...j', grad_frac, lattice_inv_T)

            if grid_data is None:
                raise ValueError(f"无法获取场 '{field}' 的数据")

            axes = [np.linspace(0, 1, n, endpoint=False) for n in (grid_data.shape[:-1] if is_vector else grid_data.shape)]
            interpolator = RegularGridInterpolator(axes, grid_data, bounds_error=False, fill_value=np.nan)
            print(f"     - 成功创建 '{field}' 插值器。")
            return interpolator
        except Exception as e:
            print(f"     - 警告: 创建 '{field}' 插值器失败: {e}")
            return None

    def _calculate_and_cache_local_vfe(self):
        """
        根据已加载的0-单纯形特征，计算每个原子的局部变分自由能(VFE)，
        并将其作为新列添加到self.atomic_features中，以供后续派生特征计算使用。
        """
        print("   - 正在为所有原子计算和缓存局部变分自由能 (VFE)...")
        if 'local_variational_free_energy' in self.atomic_features.columns and \
           not self.atomic_features['local_variational_free_energy'].isnull().all():
            print("     - VFE特征已存在于0-单纯形特征文件中，跳过重新计算。")
            return

        try:
            # 准备计算VFE所需的所有原子级属性
            valences = [site.specie.oxi_state if hasattr(site.specie, 'oxi_state') else 0 for site in self.pmg_structure]
            
            vfe_values = []
            for i, row in self.atomic_features.iterrows():
                # --- 1. 能量项 U (Internal Energy Proxy) ---
                bond_length_distortion = row.get('bond_length_distortion', 0.0)
                local_environment_anisotropy = row.get('local_environment_anisotropy', 0.0)
                
                bond_valence_sum = row.get('bond_valence_sum', 0.0)
                ideal_valence = valences[i] if i < len(valences) else 0
                bvs_deviation = abs(bond_valence_sum - ideal_valence)
                
                U_proxy = bond_length_distortion + local_environment_anisotropy + bvs_deviation

                # --- 2. 熵项 S (Entropy Proxy) ---
                S_proxy = row.get('local_env_entropy', 0.0)

                # --- 3. 温度项 T (Temperature Proxy) ---
                symmetry_breaking_quotient = row.get('symmetry_breaking_quotient', 1.0) # 默认为1（高对称性）
                T_proxy = 1.0 - symmetry_breaking_quotient

                # --- 4. 组合成变分自由能 F = U - TS ---
                free_energy = U_proxy - T_proxy * S_proxy
                vfe_values.append(free_energy)

            self.atomic_features['local_variational_free_energy'] = vfe_values
            print("     - VFE计算和缓存完成。")
        except Exception as e:
            print(f"     - 警告: VFE计算失败: {e}. 将使用NaN值填充。")
            self.atomic_features['local_variational_free_energy'] = np.nan

    # ==========================================================================
    # 主计算流程
    # ==========================================================================
    def calculate_all_features(self) -> pd.DataFrame:
        """
        执行伪代码中定义的完整五步计算流程。
        """
        print("\n开始执行主计算流程...")
        # 步骤 1: 识别所有边，并计算基础几何和严格局部的拓扑特征
        print("步骤 1/5: 计算特征组 A (基础几何与严格局部拓扑)...")
        features_A_df = self._calculate_features_A()

        if features_A_df.empty:
            print("未找到任何边，计算提前终止。")
            return pd.DataFrame()
        print(f"步骤 1 完成: 找到 {len(features_A_df)} 条边。")

        # 步骤 2, 3, 4: 基于 features_A_df 定义的边集合计算其他特征组
        print("步骤 2/5: 计算特征组 B (0-单纯形派生特征)...")
        features_B_df = self._calculate_features_B(features_A_df)
        print("步骤 2 完成。")

        print("步骤 3/5: 计算特征组 C (沿键路径的量子特征)...")
        features_C_df = self._calculate_features_C(features_A_df)
        print("步骤 3 完成。")

        print("步骤 4/5: 计算特征组 D (深度融合代数特征)...")
        features_D_df = self._calculate_features_D(features_A_df)
        print("步骤 4 完成。")

        # 步骤 5: 合并所有特征组
        print("步骤 5/5: 合并所有特征组...")
        final_df = pd.concat([features_A_df, features_B_df, features_C_df, features_D_df], axis=1)
        
        # 将标识列移到最前面
        id_cols = ['site1_index', 'site2_index', 'site1_element', 'site2_element']
        feature_cols = [col for col in final_df.columns if col not in id_cols]
        final_df = final_df[id_cols + feature_cols]
        
        print(f"所有特征计算完成。最终DataFrame维度: {final_df.shape}")
        return final_df
    
    # ==========================================================================
    # 特征组 A: 基础几何与【严格局部】拓扑特征
    # ==========================================================================
    def _calculate_features_A(self) -> pd.DataFrame:
        all_edges_features = []
        for u, v in self.simple_graph.edges():
            # 1. 基础识别信息
            site1 = self.pmg_structure[u]
            site2 = self.pmg_structure[v]

            # 2. 基础几何
            distance, jimage = self.pmg_structure.lattice.get_distance_and_image(
                self.pmg_structure[u].frac_coords,
                self.pmg_structure[v].frac_coords
            )
            coord_num_u = self.simple_graph.degree[u]
            coord_num_v = self.simple_graph.degree[v]

            # 3. 【严格局部】拓扑上下文
            neighbors_u = set(self.simple_graph.neighbors(u))
            neighbors_v = set(self.simple_graph.neighbors(v))

            # 3a. 局部三元环计数
            common_neighbors = neighbors_u.intersection(neighbors_v)
            local_3_cycle_count = len(common_neighbors)

            # 3b. 局部四元环计数
            local_4_cycle_count = 0
            # 遍历u的邻居k (不包括v)
            for k in neighbors_u - {v}:
                # 遍历v的邻居l (不包括u)
                for l in neighbors_v - {u}:
                    if self.simple_graph.has_edge(k, l):
                        local_4_cycle_count += 1
            local_4_cycle_count //= 2 # 每个四元环被两个不同的(k,l)对计数

            # 4. 连续的化学相互作用描述符
            charge1 = self.atomic_features.loc[u, 'bader_charge']
            charge2 = self.atomic_features.loc[v, 'bader_charge']
            delta_bader_charge = abs(charge1 - charge2)

            en1 = site1.specie.X
            en2 = site2.specie.X
            delta_electronegativity = abs(en1 - en2)
            ionic_character = 1 - np.exp(-0.25 * delta_electronegativity**2)

            features = {
                'site1_index': u, 'site2_index': v,
                'site1_element': site1.specie.symbol, 'site2_element': site2.specie.symbol,
                'bond_distance': distance,
                'site1_coord_num': coord_num_u,
                'site2_coord_num': coord_num_v,
                'local_3_cycle_count': local_3_cycle_count,
                'local_4_cycle_count': local_4_cycle_count,
                'continuous_ionic_character_en': ionic_character,
                'continuous_ionic_character_charge': delta_bader_charge
            }
            all_edges_features.append(features)

        return pd.DataFrame(all_edges_features)

    # ==========================================================================
    # 特征组 B: 0-单纯形派生特征
    # ==========================================================================
    def _calculate_features_B(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        all_derived_features = []
        # 【委员会精简】不再使用全部34个特征，而是选择物理意义最核心的子集
        cols_to_process = [
            'electronegativity', 'bader_charge', 'electron_density', 'elf',
            'local_dos_fermi', 'local_environment_anisotropy', 'symmetry_breaking_quotient',
            'structure_chemistry_incompatibility', 'lie_algebra_norm', 'lie_algebra_principal_angle',
            'local_variational_free_energy' # 遵嘱加入贝叶斯力学特征
        ]
        # 确保我们选择的列确实存在
        cols_to_process = [col for col in cols_to_process if col in self.atomic_features.columns]
        print(f"   - 将为以下 {len(cols_to_process)} 个核心0-单纯形特征生成派生特征: {cols_to_process}")


        for _, row in edges_df.iterrows():
            u, v = int(row['site1_index']), int(row['site2_index'])
            props_u = self.atomic_features.iloc[u]
            props_v = self.atomic_features.iloc[v]

            features = {}
            for col in cols_to_process:
                val_u, val_v = props_u.get(col, np.nan), props_v.get(col, np.nan)
                if not (np.isnan(val_u) or np.isnan(val_v)):
                    features[f'delta_{col}'] = abs(val_u - val_v)
                    features[f'avg_{col}'] = (val_u + val_v) / 2.0
                else:
                    features[f'delta_{col}'] = np.nan
                    features[f'avg_{col}'] = np.nan
            all_derived_features.append(features)

        return pd.DataFrame(all_derived_features)

    # ==========================================================================
    # 特征组 C: 沿键路径的量子特征
    # ==========================================================================
    def _clip_frac_coords_for_interp(self, frac_coords: np.ndarray, grid_shape: Tuple[int, ...]) -> np.ndarray:
        """
        【委员会新增】将分数坐标安全地裁剪到插值网格的有效边界内。
        这可以防止由于浮点精度问题导致坐标略微超出 `[0, 1)` 范围而引起的插值失败。
        网格假定是使用 np.linspace(..., endpoint=False) 定义的。
        """
        clipped = np.copy(frac_coords)
        for i, n_points in enumerate(grid_shape):
            if n_points > 1:
                # 最大坐标为 (n-1)/n。我们将其裁剪到略低于此值的位置。
                max_coord_val = (n_points - 1) / n_points
                clipped[i] = np.clip(clipped[i], 0.0, max_coord_val - self._TOLERANCE)
            else:
                clipped[i] = None # 对于单点维度
        return clipped

    def _calculate_features_C(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        all_quantum_features = []
        for _, row in edges_df.iterrows():
            u, v = int(row['site1_index']), int(row['site2_index'])
            
            # 计算键中点 (考虑周期性)
            site1 = self.pmg_structure[u]
            _, jimage = self.pmg_structure.lattice.get_distance_and_image(
                self.pmg_structure[u].frac_coords,
                self.pmg_structure[v].frac_coords
            )
            site2 = self.pmg_structure[v]
            site2_shifted_coords = site2.lattice.get_cartesian_coords(site2.frac_coords + jimage)
            midpoint_cart = (site1.coords + site2_shifted_coords) / 2.0
            midpoint_frac = self.pmg_structure.lattice.get_fractional_coords(midpoint_cart) % 1.0

            features = {}
            for field, interp in self.interpolators.items():
                if interp:
                    # 【委员会修正】在插值前，对分数坐标进行裁剪以确保其在有效范围内
                    is_vector = interp.values.ndim == 4
                    grid_shape = interp.values.shape[:-1] if is_vector else interp.values.shape
                    clipped_midpoint_frac = self._clip_frac_coords_for_interp(midpoint_frac, grid_shape)

                    value = interp(clipped_midpoint_frac)
                    # 对于矢量场，取其模
                    if isinstance(value, np.ndarray) and value.ndim > 0:
                        value = np.linalg.norm(value)
                    features[f'{field}_at_midpoint'] = float(value) if not np.isnan(value) else np.nan
                else:
                    features[f'{field}_at_midpoint'] = np.nan
            all_quantum_features.append(features)

        return pd.DataFrame(all_quantum_features)

    # ==========================================================================
    # 特征组 D: 深度融合代数特征 (严格局部化)
    # ==========================================================================
    def _calculate_features_D(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        all_algebraic_features = []
        for _, row in edges_df.iterrows():
            u, v = int(row['site1_index']), int(row['site2_index'])

            # 获取数据
            _, jimage = self.pmg_structure.lattice.get_distance_and_image(
                self.pmg_structure[u].frac_coords,
                self.pmg_structure[v].frac_coords
            )
            site2 = self.pmg_structure[v]
            site2_shifted_coords_for_vec = site2.lattice.get_cartesian_coords(site2.frac_coords + jimage)
            bond_vector = site2_shifted_coords_for_vec - self.pmg_structure[u].coords
            T_u = self.atomic_tensors.get(u, np.full((3, 3), np.nan))
            T_v = self.atomic_tensors.get(v, np.full((3, 3), np.nan))

            # 1. 李代数不兼容性 (局部)
            if not (np.any(np.isnan(T_u)) or np.any(np.isnan(T_v))):
                lie_incompatibility = np.linalg.norm(T_u @ T_v - T_v @ T_u, 'fro')
            else:
                lie_incompatibility = np.nan

            # 2. 张量代数对齐性 (局部)
            norm_u = np.linalg.norm(T_u, 'fro')
            norm_v = np.linalg.norm(T_v, 'fro')
            if norm_u > self._TOLERANCE and norm_v > self._TOLERANCE:
                inner_product = np.trace(T_u.T @ T_v)
                tensor_alignment = inner_product / (norm_u * norm_v)
            else:
                tensor_alignment = None

            # 3. 【严格局部】商代数轨道大小
            neighbors_u = set(self.simple_graph.neighbors(u))
            neighbors_v = set(self.simple_graph.neighbors(v))
            local_cluster_indices = sorted(list({u, v} | neighbors_u | neighbors_v))
            
            # 创建局部原子簇的Molecule对象
            species = [self.pmg_structure[i].specie for i in local_cluster_indices]
            coords = [self.pmg_structure[i].coords for i in local_cluster_indices]
            local_molecule = Molecule(species, coords)
            
            # 计算局部点群
            try:
                pga = PointGroupAnalyzer(local_molecule, tolerance=0.3)
                local_symm_ops = pga.get_symmetry_operations()
                local_group_order = len(local_symm_ops)
                # 计算轨道大小
                orbit = {tuple(np.round(op.operate(bond_vector), 5)) for op in local_symm_ops}
                local_orbit_size = len(orbit)
                # 根据轨道-稳定子定理计算稳定子大小
                if local_orbit_size > 0:
                    stabilizer_size = int(round(local_group_order / local_orbit_size))
                else:
                    stabilizer_size = None # 理论上轨道大小至少为None

            except Exception as e: # 避免由于pga初始化失败导致的错误
                print(f"  - 警告: 计算局部点群失败: {e}")
                local_orbit_size = None # 失败则认为轨道大小为None
                stabilizer_size = None # 同样设置稳定子大小为None

            # 4. 辛几何耦合 (局部)
            # jimage 已在上文计算，site2_shifted_for_vec 也已计算
            midpoint_cart = (self.pmg_structure[u].coords + site2_shifted_coords_for_vec) / 2.0
            midpoint_frac = self.pmg_structure.lattice.get_fractional_coords(midpoint_cart) % 1.0
            grad_interp = self.interpolators.get('density_gradient')
            if grad_interp:
                # 【委员会二次修正】同样对D组中的插值坐标进行裁剪，防止NaN
                is_vector = grad_interp.values.ndim == 4
                grid_shape = grad_interp.values.shape[:-1] if is_vector else grad_interp.values.shape
                clipped_midpoint_frac = self._clip_frac_coords_for_interp(midpoint_frac, grid_shape)

                grad_rho_vec = grad_interp(clipped_midpoint_frac)
                if isinstance(grad_rho_vec, np.ndarray) and not np.any(np.isnan(grad_rho_vec)):
                    symplectic_coupling = np.linalg.norm(np.cross(bond_vector, grad_rho_vec))
                else:
                    symplectic_coupling = np.nan
            else:
                symplectic_coupling = np.nan

            # 5. 内在坐标系对齐 (局部)
            projections_on_u = [np.nan] * 3
            projections_on_v = [np.nan] * 3
            try:
                if not np.any(np.isnan(T_u)):
                    _, eigenvectors_u = np.linalg.eigh(T_u)
                    projections_on_u = [abs(np.dot(bond_vector, eig_vec)) for eig_vec in eigenvectors_u.T]
                if not np.any(np.isnan(T_v)):
                    _, eigenvectors_v = np.linalg.eigh(T_v)
                    projections_on_v = [abs(np.dot(bond_vector, eig_vec)) for eig_vec in eigenvectors_v.T]
            except np.linalg.LinAlgError as e:
                print(f"  - 警告: 计算内在坐标系对齐失败: {e}")
                pass # 保持NaN

            features = {
                'lie_incompatibility': lie_incompatibility,
                'tensor_alignment': tensor_alignment,
                'local_quotient_orbit_size': local_orbit_size,
                'local_bond_stabilizer_size': stabilizer_size, # 【委员会新增】
                'symplectic_coupling': symplectic_coupling,
                'align_proj_u1': projections_on_u[0], 'align_proj_u2': projections_on_u[1], 'align_proj_u3': projections_on_u[2],
                'align_proj_v1': projections_on_v[0], 'align_proj_v2': projections_on_v[1], 'align_proj_v3': projections_on_v[2],
            }
            all_algebraic_features.append(features)

        return pd.DataFrame(all_algebraic_features)

    def save_features_to_csv(self, features_df: pd.DataFrame, output_path: str):
        """将计算出的特征保存到CSV文件。"""
        features_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n1-单纯形特征已成功保存到: {output_path}")
        

def run_1_simplex_features(cif_file: str,
                           atomic_features_csv: str,
                           atomic_tensors_csv: str,
                           pw_gpw_file: Optional[str] = None,
                           fd_gpw_file: Optional[str] = None,
                           output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, StructureGraph]:
    """
    主运行函数，用于计算并保存1-单纯形（拓扑连接）特征。

    参数:
        cif_file (str): 输入CIF文件的路径。
        atomic_features_csv (str): 0-单纯形特征CSV文件的路径。
        atomic_tensors_csv (str): 0-单纯形结构张量CSV文件的路径。
        pw_gpw_file (Optional[str]): 平面波GPW文件的路径。
        fd_gpw_file (Optional[str]): FD模式GPW文件的路径。
        output_dir (Optional[str]): 输出CSV文件的目录。如果为None，则使用cif_file所在的目录。

    返回:
        pd.DataFrame: 包含所有1-单纯形特征的DataFrame。
        StructureGraph: 计算中使用的拓扑图对象。
    """
    # 确定输出路径
    if output_dir:
        output_path_dir = Path(output_dir)
        output_path_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path_dir = Path(cif_file).parent

    base_name = Path(cif_file).stem.replace('-optimized', '')
    output_csv = output_path_dir / f"{base_name}-1-Simplex-Features.csv"

    # 初始化计算器
    calculator = BondFeatureCalculator(
        cif_file=cif_file,
        atomic_features_csv=atomic_features_csv,
        atomic_tensors_csv=atomic_tensors_csv,
        pw_gpw_file=pw_gpw_file,
        fd_gpw_file=fd_gpw_file
    )

    # 计算所有特征
    final_features_df = calculator.calculate_all_features()

    # 保存结果
    if not final_features_df.empty:
        calculator.save_features_to_csv(final_features_df, str(output_csv))
    else:
        print("由于未计算出任何特征，因此不生成输出文件。")

    return final_features_df, calculator.topology_graph

