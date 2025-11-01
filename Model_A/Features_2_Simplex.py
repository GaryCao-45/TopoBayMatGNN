import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Optional

from pymatgen.core import Structure, Site
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph # 新增导入，尝试解决导入问题
from ase.io import read


from gpaw import GPAW # type: ignore
from gpaw.elf import ELF # type: ignore
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import laplace
from mendeleev import element
import networkx as nx # 新增导入，用于计算局部拓扑特征

warnings.filterwarnings('ignore')

class TriangleFeatureCalculator:
    """
    2-单纯形（三角形）特征计算器
    
    贝叶斯力学设计哲学 (源自《On Bayesian mechanics: a physics of and by beliefs》):
    1. 2-单纯形作为"信念"的物理载体: 每个三角形视为一个微观的"粒子"或"观测者"，
       其特征编码了对局部环境的"信念"或"模型"。
    2. 本体势 (Ontological Potential): 特征旨在探测驱动系统演化的本体势(-log p(x))
       在该微观区域的形态，如曲率和梯度。
    3. 流的分解 (Flow Decomposition): 特征体系能够区分信息流场中的耗散部分(梯度流)
       与孤立部分(环流)，以捕捉完整的动力学。
    4. 严格局部性与马尔可夫毯: 所有特征的计算域被严格限制在2-单纯形自身及其
       一阶邻域（即其马尔可夫毯）内，严禁任何形式的全局计算。
    """
    _TOLERANCE = 1e-9  # 用于避免数值计算中的除零错误

    def _clip_fractional_coords(self, fractional_coords: np.ndarray, grid_shape: Tuple[int, ...]) -> np.ndarray:
        """
        辅助函数：根据给定网格形状裁剪分数坐标，确保其在插值器有效范围内。
        """
        clipped_coords = np.copy(fractional_coords)
        for d in range(len(grid_shape)):
            n_points = grid_shape[d]
            if n_points > 1:
                max_coord_value = (n_points - 1) / n_points
                clipped_coords[d] = np.clip(clipped_coords[d], 0.0, max_coord_value - self._TOLERANCE)
            else:
                clipped_coords[d] = None # 对于单点维度，映射到None
        return clipped_coords


    def __init__(self, cif_file_path: str, atomic_features_csv_path: str, bond_features_csv_path: str, 
                 atomic_tensors_csv_path: str,
                 pw_gpw_file: Optional[str] = None, fd_gpw_file: Optional[str] = None,
                 random_seed: int = 42,
                 topology_graph: Optional[StructureGraph] = None):

        # 设置可控随机种子
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.rng = np.random.default_rng(random_seed)
        
        self.cif_path = cif_file_path
        self.pmg_structure: Structure = Structure.from_file(cif_file_path)
        self.crystal_nn = CrystalNN()
        self.topology_graph = topology_graph # 存储传入的拓扑图
        if self.topology_graph:
            # 【委员会修正】从 Pymatgen StructureGraph 内部的 MultiDiGraph 创建一个简单的、无向的图。
            # 这是进行拓扑分析（如寻找三角形）的正确图表示，避免了有向图导致的逻辑错误。
            self.simple_graph = nx.Graph(self.topology_graph.graph)
            print(f"从拓扑图构建了NetworkX简单图: {self.simple_graph.number_of_nodes()} 个节点, {self.simple_graph.number_of_edges()} 条边")
        else:
            self.simple_graph = nx.Graph()
            print("警告: 未提供拓扑图，将使用空图。")
            
        print(f"成功读取晶体结构: {self.pmg_structure.composition.reduced_formula}")
        print(f"结构包含 {len(self.pmg_structure)} 个原子。")
        
        # 加载0-Simplex和1-Simplex特征CSV文件
        self.atomic_features_df = pd.read_csv(atomic_features_csv_path)
        self.bond_features_df = pd.read_csv(bond_features_csv_path)
        print(f"成功加载0-Simplex特征: {len(self.atomic_features_df)} 个原子")
        print(f"成功加载1-Simplex特征: {len(self.bond_features_df)} 个键")

        # 加载0-Simplex结构张量
        self.load_atomic_tensors(atomic_tensors_csv_path)
        
        # 构建高效查询结构
        self._build_query_structures()
        
        self.pw_calc: Optional[GPAW] = self._load_gpaw_calc(pw_gpw_file, "平面波")
        self.fd_calc: Optional[GPAW] = self._load_gpaw_calc(fd_gpw_file, "FD模式")

        # 预加载SpacegroupAnalyzer以避免在循环中重复初始化
        try:
            print("正在初始化 SpacegroupAnalyzer...")
            self.sga = SpacegroupAnalyzer(self.pmg_structure)
            print("SpacegroupAnalyzer 初始化成功。")
        except Exception as e:
            self.sga = None
            print(f"警告: SpacegroupAnalyzer 初始化失败: {e}。位点对称特征将使用后备方案。")

    def load_atomic_tensors(self, atomic_tensors_csv_path: str):
        """
        从CSV文件加载0-单纯形结构张量到内存，以原子索引为键存储。

        参数:
            atomic_tensors_csv_path (str): 包含结构张量的CSV文件路径。
        """
        print(f"加载0-单纯形结构张量: {atomic_tensors_csv_path}")
        self.atomic_tensors: Dict[int, np.ndarray] = {}
        try:
            tensors_df = pd.read_csv(atomic_tensors_csv_path)
            for _, row in tensors_df.iterrows():
                site_idx = int(row['site_index'])
                tensor = np.array([
                    [row['T_struct_00'], row['T_struct_01'], row['T_struct_02']],
                    [row['T_struct_10'], row['T_struct_11'], row['T_struct_12']],
                    [row['T_struct_20'], row['T_struct_21'], row['T_struct_22']]
                ])
                self.atomic_tensors[site_idx] = tensor
            print(f"加载了 {len(self.atomic_tensors)} 个原子的结构张量。")
        except FileNotFoundError as e:
            print(f"错误: 结构张量文件未找到: {atomic_tensors_csv_path}, {e}")
            # 如果文件不存在，保持空字典，后续获取张量的操作会返回NaN
        except Exception as e:
            print(f"加载结构张量时出错: {e}")

    def _build_query_structures(self):
        """
        构建用于高效查询原子和键特征的内部数据结构。
        将原子特征存储为字典，键为原子索引；
        将键特征存储为字典，键为排序后的原子索引元组。
        """
        print("正在构建高效查询结构...")
        
        # 构建原子特征查询字典（以原子索引为键）
        self.atomic_features_dict = {}
        for idx, row in self.atomic_features_df.iterrows():
            atom_idx = idx  # 假设CSV中的行索引对应原子索引
            self.atomic_features_dict[atom_idx] = row.to_dict()
        
        # 构建键特征查询字典（以排序后的原子索引对为键）
        self.connection_features_dict = {}
        for _, row in self.bond_features_df.iterrows():
            site1_idx = int(row['site1_index'])
            site2_idx = int(row['site2_index'])
            bond_key = tuple(sorted([site1_idx, site2_idx]))
            self.connection_features_dict[bond_key] = row.to_dict()
        
        print(f"查询结构构建完成: {len(self.atomic_features_dict)} 个原子特征, {len(self.connection_features_dict)} 个键特征")

    def _compute_triangle_circumradius_sq(self, p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
        """
        计算三角形外接圆的平方半径 R^2 = (a^2 b^2 c^2)/(16 A^2)，用于Alpha过滤值的几何回退。
        """
        v_ij = p_j - p_i
        v_jk = p_k - p_j
        v_ki = p_i - p_k
        a = np.linalg.norm(v_jk)
        b = np.linalg.norm(v_ki)
        c = np.linalg.norm(v_ij)
        area = 0.5 * np.linalg.norm(np.cross(v_ij, -v_ki))
        if area <= self._TOLERANCE:
            return 0.0
        return float((a * a * b * b * c * c) / (16.0 * area * area))

    def _get_triangle_alpha_filtration(self, indices: Tuple[int, int, int]) -> float:
        """
        获取三角形在Alpha复形中的过滤值（若可用）；否则用几何外接圆平方半径回退。
        说明：在GUDHI的AlphaComplex中，2-单纯形的过滤值等于其外接圆的 R^2。
        """
        i, j, k = indices
        val: float = 0.0
        st = getattr(self, 'alpha_simplex_tree', None)
        if st is not None:
            try:
                val = float(st.filtration([i, j, k]))
                return val
            except Exception as e:
                print(f"  - 警告: 计算三角形过滤值失败: {e}")
                pass
        # 几何回退
        p_i, p_j, p_k = self.pmg_structure[i].coords, self.pmg_structure[j].coords, self.pmg_structure[k].coords
        return self._compute_triangle_circumradius_sq(np.array(p_i), np.array(p_j), np.array(p_k))

    def _load_gpaw_calc(self, gpw_file: Optional[str], mode_name: str) -> Optional[GPAW]:
        """
        加载GPAW计算器实例（平面波或FD模式）。

        参数:
            gpw_file (Optional[str]): GPW文件的路径。如果为None或文件不存在，则返回None。
            mode_name (str): 计算模式的名称（例如"平面波"或"FD模式"），用于打印信息。

        返回:
            Optional[GPAW]: GPAW计算器实例，如果加载失败则为None。
        """
        if not gpw_file:
            return None
        
        gpw_path = Path(gpw_file).resolve()
        if gpw_path.exists():
            print(f"--- 加载{mode_name} GPAW 计算结果: {gpw_path} ---")
            return GPAW(str(gpw_path), txt=None)
        else:
            print(f"警告: {mode_name} gpw文件 '{gpw_path}' 不存在。")
            return None

    def get_all_two_simplices(self) -> List[Tuple[int, int, int]]:
        """
        使用CrystalNN算法识别晶体结构中所有独特的2-单纯形（三角形）。
        每个三角形由一个中心原子及其两个近邻原子组成。
        使用CrystalNN识别邻居，并包含容差调整的鲁棒性处理。

        返回:
            List[Tuple[int, int, int]]: 包含所有独特三角形的列表，每个三角形由三个原子索引组成（已排序）。
        """
        print("正在识别所有独特的原子三元组（基于CrystalNN）...")
        unique_triangles: Set[Tuple[int, int, int]] = set()

        for i in range(len(self.pmg_structure)):
            neighbors_found = []
            try:
                neighbors_found = self.crystal_nn.get_nn_info(self.pmg_structure, i)
            except Exception as e:
                print(f"  - 警告: 原子 {i} 初始邻居识别失败: {e}")

            # 如果初始尝试失败或未找到邻居，尝试略微放宽容差重试
            if not neighbors_found:
                for attempt in range(1, 3): # 最多尝试2次重试
                    try:
                        temp_crystal_nn = CrystalNN(x_diff_tol=0.05 + attempt * 0.02)
                        neighbors_retry = temp_crystal_nn.get_nn_info(self.pmg_structure, i)
                        if neighbors_retry:
                            print(f"  - 成功: 原子 {i} 在第 {attempt+1} 次尝试 (dist_tol={0.05 + attempt * 0.02:.2f}) 后找到邻居。")
                            neighbors_found = neighbors_retry
                            break # 找到即退出重试循环
                    except Exception as e_retry:
                        print(f"  - 警告: 原子 {i} 第 {attempt+1} 次尝试邻居识别失败: {e_retry}")

            if not neighbors_found: # 如果所有尝试都失败
                print(f"  - 错误: 无法为原子 {i} 识别任何邻居，跳过其三角形构建。")
                continue # 跳过当前原子的三角形构建

            neighbor_indices = [nn['site_index'] for nn in neighbors_found]
            
            # 对于每个中心原子i，选择任意两个邻居j, k构成三角形
            if len(neighbor_indices) >= 2:
                for j, k in combinations(neighbor_indices, 2):
                    # 使用sorted确保唯一性
                    triangle_indices = tuple(sorted((i, j, k)))
                    unique_triangles.add(triangle_indices)
        
        print(f"识别到 {len(unique_triangles)} 个独特的三角形。")
        return list(unique_triangles)

    def _get_triangle_geometry(self, triangle_indices: Tuple[int, int, int]) -> Optional[
        Tuple[Tuple[Site, Site, Site], Dict[Tuple[int, int], Dict[str, Any]]]
    ]:
        """
        获取给定三角形的原子站点对象和边信息。
        该方法会验证三角形的三个原子是否互为近邻（至少是其中一个原子的近邻）。

        参数:
            triangle_indices (Tuple[int, int, int]): 组成三角形的三个原子索引。

        返回:
            Optional[Tuple[Tuple[Site, Site, Site], Dict[Tuple[int, int], Dict[str, Any]]]]:
                如果成功找到所有几何信息，则返回一个包含三个Site对象的元组和边信息字典；否则返回None。
        """
        i, j, k = triangle_indices
        base_sites = {idx: self.pmg_structure[idx] for idx in (i, j, k)}

        for center_idx, n1_idx, n2_idx in [(i, j, k), (j, i, k), (k, i, j)]:
            try:
                neighbors_of_center = self.crystal_nn.get_nn_info(self.pmg_structure, center_idx)
                n1_info = next((nn for nn in neighbors_of_center if nn['site_index'] == n1_idx), None)
                n2_info = next((nn for nn in neighbors_of_center if nn['site_index'] == n2_idx), None)

                if n1_info and n2_info:
                    # 找到中心，现在获取第三条边的信息（如果存在）
                    try:
                        neighbors_of_n1 = self.crystal_nn.get_nn_info(self.pmg_structure, n1_idx)
                        n1_n2_info = next((nn for nn in neighbors_of_n1 if nn['site_index'] == n2_idx), None)
                    except Exception as e:
                        print(f"  - 错误: 获取三角形几何关系失败: {e}")
                        n1_n2_info = None
                    
                    # 构建返回的site元组，顺序与输入索引一致
                    found_sites = {
                        center_idx: base_sites[center_idx],
                        n1_idx: n1_info['site'],
                        n2_idx: n2_info['site'],
                    }
                    sites_tuple = (found_sites[i], found_sites[j], found_sites[k])
                    
                    # 构建边信息字典
                    edge_info = {
                        tuple(sorted((center_idx, n1_idx))): n1_info,
                        tuple(sorted((center_idx, n2_idx))): n2_info,
                    }
                    if n1_n2_info:
                        edge_info[tuple(sorted((n1_idx, n2_idx)))] = n1_n2_info
                    
                    return sites_tuple, edge_info
            except Exception as e:
                print(f"  - 错误: 获取三角形几何关系失败: {e}")
                continue
        return None

    def calculate_all_features(self) -> pd.DataFrame:
        """
        计算所有2-单纯形（三角形）的融合特征。
        流程包括几何特征(A)、跨层级派生特征(B)、量子化学特征(C)和代数融合特征(D)。

        返回:
            pd.DataFrame: 包含所有计算特征的数据框。
        """
        print("开始统一计算流程...")
        all_two_simplices = self.get_all_two_simplices()
        all_features_list = []

        # --- C/D部分特征计算的预处理 ---
        # 统一为场特征准备插值器，包括密度、ELF、拉普拉斯和梯度
        field_interpolators = self._setup_field_interpolators()

        # --- 对每个三角形进行统一计算 ---
        for indices in all_two_simplices:
            geometry = self._get_triangle_geometry(indices)
            if not geometry:
                continue
            
            # 统一计算几何信息，避免重复
            sites, _ = geometry
            site_i, site_j, site_k = sites
            p_i, p_j, p_k = site_i.coords, site_j.coords, site_k.coords
            
            v_ij = p_j - p_i
            v_ik = p_k - p_i
            normal_vector = np.cross(v_ij, v_ik)
            norm_n = np.linalg.norm(normal_vector)
            unit_normal = normal_vector / norm_n if norm_n > self._TOLERANCE else None

            # --- A: 计算几何特征 ---
            features_A = self._compute_geometric_features_A(sites, v_ij, v_ik)

            # --- B: 计算跨层级派生特征 ---
            features_B = self._compute_derived_features_B(indices)

            # --- C: 计算量子化学特征 ---
            features_C = self._compute_quantum_features_C(indices, sites, field_interpolators)

            # --- D: 计算代数融合特征 ---
            # 传入场插值器和三角形面积
            features_D = self._compute_fused_algebraic_features_D(
                indices, sites, unit_normal, features_A['triangle_area'], field_interpolators
            )

            # --- E: (扩展) 局部拓扑与嵌入特征 ---
            features_E = self._compute_local_embedding_features_E(indices, unit_normal)


            # --- 合并所有特征 ---
            base_features = {
                'atom_index_i': indices[0], 'atom_index_j': indices[1], 'atom_index_k': indices[2],
                'atom_symbol_i': self.pmg_structure[indices[0]].specie.symbol,
                'atom_symbol_j': self.pmg_structure[indices[1]].specie.symbol,
                'atom_symbol_k': self.pmg_structure[indices[2]].specie.symbol,
            }
            
            # 过滤掉空的特征字典
            valid_features = [d for d in [base_features, features_A, features_B, features_C, features_D, features_E] if d]
            if not valid_features:
                continue

            all_features = {k: v for d in valid_features for k, v in d.items()}
            all_features_list.append(all_features)

        final_df = pd.DataFrame(all_features_list)
        
        # 将标识列移到最前面
        id_cols = [
            'atom_index_i', 'atom_index_j', 'atom_index_k', 
            'atom_symbol_i', 'atom_symbol_j', 'atom_symbol_k'
        ]

        # 增加健壮性：如果未找到2-单纯形，则返回带有正确列的空DataFrame
        if final_df.empty:
            print("\n警告: 未计算任何2-单纯形特征，因为未找到2-单纯形。将返回一个空的DataFrame。")
            return pd.DataFrame(columns=id_cols)

        feature_cols = [col for col in final_df.columns if col not in id_cols]
        # 确保所有id_cols都存在于DataFrame中，以避免KeyError
        existing_id_cols = [col for col in id_cols if col in final_df.columns]
        final_df = final_df[existing_id_cols + feature_cols]
        
        print(f"\n所有2-单纯形特征计算完成。总特征维度: {len(final_df.columns) - len(id_cols)}")
        print("  - A部分: 3个基础几何特征 (三角形面积, 键角方差, 形状因子)")
        print("  - B部分: 8个跨层级派生特征 (4个原子特征统计 + 4个键特征统计)") 
        print("  - C部分: 3个几何重心量子特征 (密度, 拉普拉斯, ELF)")
        print("  - D部分: 4个贝叶斯力学融合特征")
        print("  - E部分: 4个局部嵌入与拓扑特征 (二面角统计, 聚类系数, Alpha过滤值)")
        return final_df

    def _compute_geometric_features_A(self, sites: Tuple[Site, Site, Site], v_ij: np.ndarray, v_ik: np.ndarray) -> Dict[str, float]:
        """
        计算三角形的几何特征（A部分）。
        包括三角形面积、键角方差和三角形形状因子。

        参数:
            sites (Tuple[Site, Site, Site]): 组成三角形的三个Site对象。
            v_ij (np.ndarray): 从Site i到Site j的向量。
            v_ik (np.ndarray): 从Site i到Site k的向量。

        返回:
            Dict[str, float]: 包含几何特征及其值的字典。
        """
        p_i, p_j, p_k = sites[0].coords, sites[1].coords, sites[2].coords
        v_jk = p_k - p_j

        # 1. 三角形面积
        triangle_area = 0.5 * np.linalg.norm(np.cross(v_ij, v_ik))

        # 2. 键角方差
        cos_angle_i = np.dot(v_ij, v_ik) / (np.linalg.norm(v_ij) * np.linalg.norm(v_ik))
        cos_angle_j = np.dot(-v_ij, v_jk) / (np.linalg.norm(v_ij) * np.linalg.norm(v_jk))
        cos_angle_k = np.dot(-v_ik, -v_jk) / (np.linalg.norm(v_ik) * np.linalg.norm(v_jk))
        
        angles_radians = [
            np.arccos(np.clip(cos, -1.0, 1.0))
            for cos in [cos_angle_i, cos_angle_j, cos_angle_k]
        ]
        bond_angle_variance = np.var(angles_radians)

        edge_lengths = [np.linalg.norm(v) for v in (v_ij, v_ik, v_jk)]
        # 3. 三角形形状因子
        perimeter = np.sum(edge_lengths)
        triangle_shape_factor = 0.0
        # 只有当周长和面积都非零时，形状因子才有意义
        if perimeter > self._TOLERANCE and triangle_area > self._TOLERANCE:
            triangle_shape_factor = 12 * triangle_area * np.sqrt(3) / (perimeter**2)
        
        return {
            'triangle_area': triangle_area,
            'bond_angle_variance': bond_angle_variance,
            'triangle_shape_factor': float(np.clip(triangle_shape_factor, 0.0, 1.0)), # 确保在[0,1]
        }

    def _compute_derived_features_B(self, indices: Tuple[int, int, int]) -> Dict[str, float]:
        """
        计算跨层级派生特征（B部分）。
        
        贝叶斯力学视角：描述了构成2-单纯形的"粒子"集合的宏观统计性质。
        严格遵循伪代码方案，仅聚合bader_charge和structure_chemistry_incompatibility（0-单纯形），
        以及bond_distance和tensor_alignment（1-单纯形）。

        参数:
            indices (Tuple[int, int, int]): 组成三角形的三个原子索引。

        返回:
            Dict[str, float]: 包含派生特征及其值的字典。
        """
        i, j, k = indices
        
        # B1: 从0-单纯形派生（顶点特征统计）
        atomic_props_to_agg = ['bader_charge', 'structure_chemistry_incompatibility', 'tensor_variance']
        features_from_atoms = {}
        
        for prop in atomic_props_to_agg:
            values = []
            for idx in [i, j, k]:
                if idx in self.atomic_features_dict:
                    val = self.atomic_features_dict[idx].get(prop, np.nan)
                    if not pd.isna(val):
                        values.append(val)
            
            if values:
                features_from_atoms[f'avg_{prop}'] = np.mean(values)
                features_from_atoms[f'variance_{prop}'] = np.var(values)
            else:
                features_from_atoms[f'avg_{prop}'] = np.nan
                features_from_atoms[f'variance_{prop}'] = np.nan

        # 【委员会修正】重命名tensor_variance相关的特征以匹配用户期望
        if 'variance_tensor_variance' in features_from_atoms:
            features_from_atoms['var_tensor_variance'] = features_from_atoms.pop('variance_tensor_variance')

        # B2: 从1-单纯形派生（边特征统计）
        # 注意：伪代码中使用的是'tensor_alignment'，但现有数据可能使用不同的命名
        # 我们尝试映射到最接近的特征名
        bond_props_mapping = {
            'bond_distance': 'bond_distance',
            'tensor_alignment': 'tensor_alignment'  # 直接映射到现有特征名
        }
        
        features_from_bonds = {}
        
        for prop_alias, actual_prop in bond_props_mapping.items():
            bond_keys = [tuple(sorted([i, j])), tuple(sorted([j, k])), tuple(sorted([k, i]))]
            values = []

            for bond_key in bond_keys:
                if bond_key in self.connection_features_dict:
                    val = self.connection_features_dict[bond_key].get(actual_prop, np.nan)
                    if not pd.isna(val):
                        values.append(val)
                # 如果键不存在，我们简单地跳过它（这是正常的，因为不是所有原子对都是化学键）

            if values:
                features_from_bonds[f'avg_{prop_alias}'] = np.mean(values)
                features_from_bonds[f'variance_{prop_alias}'] = np.var(values)
            else:
                features_from_bonds[f'avg_{prop_alias}'] = np.nan
                features_from_bonds[f'variance_{prop_alias}'] = np.nan

        return {**features_from_atoms, **features_from_bonds}
        
    def _setup_field_interpolators(self) -> Dict[str, Any]:
        """
        设置用于插值量子场（如电子密度、拉普拉斯、ELF和密度梯度）的插值器。
        优先使用pw_calc获取密度，如果不存在则使用fd_calc。

        返回:
            Dict[str, Any]: 包含各种场插值器的字典。如果GPAW计算器不可用，则返回空字典。
        """
        if self.pw_calc is None and self.fd_calc is None:
            return {}
            
        density_calc = self.pw_calc or self.fd_calc
        if density_calc is None:
            return {}

        print("--- 预处理C/D部分：准备场和梯度插值器 ---")
        density_grid = density_calc.get_all_electron_density()
        laplacian_grid = laplace(density_grid)
        elf_grid = self._get_elf_grid(density_grid)
        
        # 使用np.gradient计算梯度场，以供后续高精度三线性插值
        print("  - 计算电子密度梯度场...")
        grad_x, grad_y, grad_z = np.gradient(density_grid)

        if elf_grid is None:
            elf_grid = np.full(density_grid.shape, np.nan)
        
        grids = {
            'density': density_grid,
            'laplacian': laplacian_grid,
            'elf': elf_grid,
            'grad_x': grad_x,
            'grad_y': grad_y,
            'grad_z': grad_z,
        }
        return self._create_interpolators_from_grids(grids)

    def _create_interpolators_from_grids(self, grids: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        从给定的三维网格数据创建RegularGridInterpolator实例。

        参数:
            grids (Dict[str, np.ndarray]): 包含网格数据（例如密度、拉普拉斯、ELF、梯度分量）的字典。

        返回:
            Dict[str, Any]: 包含创建的RegularGridInterpolator实例的字典。
        """
        print("  - 创建三维网格插值器...")
        interpolators = {}
        
        for name, grid_data in grids.items():
            if grid_data is not None:
                # 使用 np.linspace(..., endpoint=False) 与 RegularGridInterpolator 的标准用法匹配
                axes = [np.linspace(0, 1, n, endpoint=False) for n in grid_data.shape]
                # fill_value=None 会在越界时抛出异常，这里我们允许外插（默认行为）或设为0
                interpolators[name] = RegularGridInterpolator(
                    axes, grid_data, bounds_error=False, fill_value=np.nan
                )
                print(f"    - '{name}' 插值器创建成功 (网格: {grid_data.shape})")
            else:
                interpolators[name] = None
        
        print("  - 插值器创建完成。")
        return interpolators

    def _compute_quantum_features_C(self, indices: Tuple[int, int, int], sites: Tuple[Site, Site, Site], field_interpolators: Dict[str, Any]) -> Dict[str, float]:
        """
        计算几何重心量子特征（C部分）。
        
        贝叶斯力学视角：直接探测了2-单纯形在其状态空间位置上的量子化学势场。
        在三角形的几何重心（barycenter）处评估电子密度、拉普拉斯和ELF。

        参数:
            indices (Tuple[int, int, int]): 组成三角形的三个原子索引。
            sites (Tuple[Site, Site, Site]): 组成三角形的三个Site对象。
            field_interpolators (Dict[str, Any]): 包含各种场插值器的字典。

        返回:
            Dict[str, float]: 包含量子化学特征及其值的字典。
        """
        density_interp = field_interpolators.get('density')
        laplacian_interp = field_interpolators.get('laplacian')
        elf_interp = field_interpolators.get('elf')

        if not all([density_interp, laplacian_interp]):
            return {
                'density_at_barycenter': np.nan,
                'density_laplacian_at_barycenter': np.nan,
                'elf_at_barycenter': np.nan,
            }

        # 计算三角形的几何重心 (barycenter)
        barycenter_cart = (sites[0].coords + sites[1].coords + sites[2].coords) / 3.0
        barycenter_frac = self.pmg_structure.lattice.get_fractional_coords(barycenter_cart)
        barycenter_frac %= 1.0  # 确保在[0,1)范围内

        # 应用坐标裁剪，防止插值边界问题
        clipped_barycenter_frac = self._clip_fractional_coords(barycenter_frac, density_interp.values.shape)

        # 在重心点进行插值
        density = float(np.atleast_1d(density_interp(clipped_barycenter_frac))[0])
        laplacian = float(np.atleast_1d(laplacian_interp(clipped_barycenter_frac))[0])
        
        # ELF可能不存在（仅从FD模式计算器获取）
        elf_uncertainty = np.nan # (委员会新增) 初始化不确定性度量
        if elf_interp is not None:
            elf = float(np.atleast_1d(elf_interp(clipped_barycenter_frac))[0])
            # 如果是精确插值，不确定性为0
            if not pd.isna(elf):
                elf_uncertainty = 0.0
        else:
            elf = np.nan

        # --- (委员会修正) 缺失值修复策略与不确定性量化 ---
        # 如果ELF在重心的插值失败（通常是因为fd_gpw_file不可用），
        # 则使用构成三角形的三个顶点的原子ELF值的平均值作为后备估计，
        # 并使用其标准差作为该估计的不确定性度量。
        if pd.isna(elf):
            print(f"  - 警告: 三角形 {indices} 的 elf_at_barycenter 插值失败。正在使用顶点ELF统计进行修复...")
            vertex_elfs = []
            for idx in indices:
                site_elf = self.atomic_features_dict.get(idx, {}).get('elf', np.nan)
                if not pd.isna(site_elf):
                    vertex_elfs.append(site_elf)
            
            if len(vertex_elfs) > 0:
                elf = np.mean(vertex_elfs)
                # (委员会新增) 计算不确定性
                elf_uncertainty = np.std(vertex_elfs) if len(vertex_elfs) > 1 else 0.0
                print(f"    - 修复成功: 平均值={elf:.4f}, 不确定性(std)={elf_uncertainty:.4f}")
            else:
                print(f"    - 修复失败: 顶点的原子ELF值均不可用。")

        return {
            'density_at_barycenter': density,
            'density_laplacian_at_barycenter': laplacian,
            'elf_at_barycenter': elf,
            'elf_barycenter_uncertainty': elf_uncertainty, # (委员会新增)
        }

    def _get_elf_grid(self, density_grid: np.ndarray) -> Optional[np.ndarray]:
        """
        获取电子局域化函数（ELF）的网格数据。
        优先尝试从FD模式GPAW计算器获取高精度ELF，如果失败或未加载FD模式计算器，则返回None。

        参数:
            density_grid (np.ndarray): 电子密度网格数据，用于ELF计算的上下文。

        返回:
            Optional[np.ndarray]: ELF网格数据，如果无法获取则为None。
        """
        # 严格尝试从FD模式计算器获取ELF数据
        if self.fd_calc:
            print("  - 尝试从FD模式计算标准ELF...")
            try:
                elf_calculator = ELF(self.fd_calc)
                elf_calculator.update()
                print("  - 标准ELF网格获取成功 (FD模式)。")
                return elf_calculator.get_electronic_localization_function()
            except Exception as e:
                print(f"  - 警告: FD模式ELF计算失败: {e}。将返回None。")
                # 明确返回None，而非默认值，让上层函数处理缺失
                return None
        else:
            print("  - 警告: 未加载FD模式GPAW计算器，无法计算高精度ELF。将返回None。")
            return None

    def _compute_fused_algebraic_features_D(
        self, 
        indices: Tuple[int, int, int], 
        sites: Tuple[Site, Site, Site], 
        unit_normal: np.ndarray,
        triangle_area: float,
        field_interpolators: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        计算贝叶斯力学重铸的深度融合代数特征（D部分）。
        
        贝叶斯力学视角：这些特征探测了驱动系统演化的本体势场、信息流的分解，
        以及2-单纯形"粒子"内部的高阶几何耦合。

        参数:
            indices (Tuple[int, int, int]): 组成三角形的三个原子索引。
            sites (Tuple[Site, Site, Site]): 组成三角形的三个Site对象。
            unit_normal (np.ndarray): 三角形平面的单位法向量。
            triangle_area (float): 三角形的面积。
            field_interpolators (Dict[str, Any]): 包含各种场插值器的字典。

        返回:
            Dict[str, float]: 包含代数融合特征及其值的字典。
        """
        i, j, k = indices
        p_i, p_j, p_k = sites[0].coords, sites[1].coords, sites[2].coords

        # --- D1: 贝叶斯信息曲率 (Bayesian Information Curvature) ---
        # 物理意义: 量化了本体势场(Ontological Potential)在2-单纯形上的局部曲率。
        # 使用局部变分自由能的离散拉普拉斯来近似
        vfe_i = self.atomic_features_dict[i].get('local_variational_free_energy', np.nan)
        vfe_j = self.atomic_features_dict[j].get('local_variational_free_energy', np.nan)
        vfe_k = self.atomic_features_dict[k].get('local_variational_free_energy', np.nan)
        
        vfe_laplacian_variance = np.nan
        vfe_variance = np.nan
        
        vfe_values = [vfe_i, vfe_j, vfe_k]
        if not any(np.isnan(vfe_values)):
            # 新特征: VFE方差，描述信息势在单纯形上的异质性
            vfe_variance = np.var(vfe_values)

            # 修正原特征：离散拉普拉斯的平均值恒为零，改用方差来度量曲率的不均匀性
            lap_vfe_i = (vfe_j - vfe_i) + (vfe_k - vfe_i)
            lap_vfe_j = (vfe_i - vfe_j) + (vfe_k - vfe_j)
            lap_vfe_k = (vfe_i - vfe_k) + (vfe_j - vfe_k)
            vfe_laplacian_variance = np.var([lap_vfe_i, lap_vfe_j, lap_vfe_k])

        # --- D2: 李代数环流 (Lie Commutator Flux) ---
        # 物理意义: 描述了信息流场中的孤立子午分量(Solenoidal Flow)，即局部信息涡旋。
        # 通过沿三角形边界的lie_incompatibility环路积分来计算
        bond_keys = [tuple(sorted([i, j])), tuple(sorted([j, k])), tuple(sorted([k, i]))]
        lie_flux = 0.0
        valid_bonds = 0
        
        for bond_key in bond_keys:
            if bond_key in self.connection_features_dict:
                incompat = self.connection_features_dict[bond_key].get('lie_incompatibility', 0.0)
                if not np.isnan(incompat):
                    lie_flux += incompat
                    valid_bonds += 1
        
        lie_commutator_flux = lie_flux if valid_bonds > 0 else np.nan

        # --- D3: 结构张量积之迹 (Structural Tensor Product Trace) ---
        # 物理意义: 三体几何耦合，量化三个局部结构张量的高阶相互作用
        T_i = self._get_structure_tensor(i)
        T_j = self._get_structure_tensor(j)
        T_k = self._get_structure_tensor(k)
        
        try:
            structural_tensor_product_trace = np.trace(T_i @ T_j @ T_k)
        except Exception as e:
            print(f"  - 警告: 计算结构张量积之迹失败: {e}")
            structural_tensor_product_trace = np.nan

        # --- D4: 结构张量法向投影 (Structure Tensor Normal Projection) ---
        # 物理意义: 量化了三个原子的集体结构张量在三角形法向上的投影
        structure_tensor_normal_projection = np.nan
        try:
            # 【委员会修复】增加对unit_normal的检查，以处理共线原子的情况
            if unit_normal is not None and not np.any(np.isnan(unit_normal)):
                T_total = T_i + T_j + T_k
                # 仅在T_total不含NaN时进行计算
                if not np.any(np.isnan(T_total)):
                    structure_tensor_normal_projection = float(unit_normal.T @ T_total @ unit_normal)
        except Exception as e:
            print(f"  - 警告: 计算结构张量法向投影时发生意外错误: {e}")
            structure_tensor_normal_projection = np.nan

        return {
            'vfe_laplacian_variance': vfe_laplacian_variance,
            'vfe_variance': vfe_variance, # 新增的贝叶斯力学特征
            'lie_commutator_flux': lie_commutator_flux,
            'structural_tensor_product_trace': structural_tensor_product_trace,
            'structure_tensor_normal_projection': structure_tensor_normal_projection,
        }

    def _compute_local_embedding_features_E(self, indices: Tuple[int, int, int], unit_normal: np.ndarray) -> Dict[str, float]:
        """
        计算局部嵌入与拓扑特征（E部分）。
        
        贝叶斯力学视角：描述了2-单纯形"粒子"与其马尔可夫毯的几何关系。
        包括立体角方差、球形度（替换原二面角和聚类系数）和Alpha过滤值。
        """
        i, j, k = indices
        p_i, p_j, p_k = self.pmg_structure[i].coords, self.pmg_structure[j].coords, self.pmg_structure[k].coords

        # E1: 立体角方差 (Solid Angle Variance) - 【委员会修正】
        # 修正算法以正确处理低配位环境并使用物理上明确的定义：
        # 对中心原子的所有邻居三元组形成的立体角进行求和。
        solid_angles = []
        try:
            for idx in indices:
                neighbors = self.crystal_nn.get_nn_info(self.pmg_structure, idx)
                
                total_solid_angle = 0.0  # 配位数小于3时，立体角为0
                if len(neighbors) >= 3:
                    center_coords = self.pmg_structure[idx].coords
                    vectors = [np.array(nn['site'].coords) - center_coords for nn in neighbors]

                    # 遍历所有由三个邻居向量构成的三面角组合
                    for v1, v2, v3 in combinations(vectors, 3):
                        norm_v1 = np.linalg.norm(v1)
                        norm_v2 = np.linalg.norm(v2)
                        norm_v3 = np.linalg.norm(v3)

                        # 检查向量长度，避免除零错误
                        if norm_v1 < self._TOLERANCE or norm_v2 < self._TOLERANCE or norm_v3 < self._TOLERANCE:
                            continue

                        # 使用Oosterom和Strackee的精确公式计算三面角的立体角
                        numerator = np.abs(np.dot(v1, np.cross(v2, v3)))
                        denominator = (norm_v1 * norm_v2 * norm_v3 +
                                     np.dot(v1, v2) * norm_v3 +
                                     np.dot(v1, v3) * norm_v2 +
                                     np.dot(v2, v3) * norm_v1)
                        
                        if denominator > self._TOLERANCE:
                            # 使用atan2以增强数值稳定性
                            total_solid_angle += 2 * np.arctan2(numerator, denominator)
                
                solid_angles.append(total_solid_angle)

        except Exception as e:
            print(f"  - 警告: 计算立体角失败: {e}")
            # 如果发生意外错误，用NaN填充以确保列表长度为3
            while len(solid_angles) < 3:
                solid_angles.append(np.nan)

        solid_angle_variance = np.var(solid_angles) if len(solid_angles) == 3 else np.nan

        # E2: 三角形球形度 (Triangle Sphericity)
        # 描述三角形的形状畸变
        l_ij = np.linalg.norm(p_i - p_j)
        l_jk = np.linalg.norm(p_j - p_k)
        l_ki = np.linalg.norm(p_k - p_i)
        
        s = (l_ij + l_jk + l_ki) / 2.0  # 半周长
        # 使用海伦公式计算面积，避免重复计算
        area_sq = s * (s - l_ij) * (s - l_jk) * (s - l_ki)
        area = np.sqrt(area_sq) if area_sq > 0 else 0

        # 外接圆半径 R = abc / 4A
        circum_radius = (l_ij * l_jk * l_ki) / (4.0 * area) if area > self._TOLERANCE else 0
        # 内切圆半径 r = A / s
        inscribed_radius = area / s if s > self._TOLERANCE else 0

        triangle_sphericity = inscribed_radius / circum_radius if circum_radius > self._TOLERANCE else 0.0

        # E3: Alpha过滤值 (保留)
        alpha_filtration_value = self._get_triangle_alpha_filtration(indices)

        return {
            'solid_angle_variance': solid_angle_variance,
            'triangle_sphericity': triangle_sphericity,
            'alpha_filtration_value': alpha_filtration_value,
        }

    def _get_structure_tensor(self, center_index: int) -> np.ndarray:
        """
        获取指定原子位点的结构张量。

        参数:
            center_index (int): 原子位点的索引。

        返回:
            np.ndarray: 3x3的结构张量。如果找不到，则返回NaN填充的矩阵。
        """
        tensor = self.atomic_tensors.get(center_index)
        if tensor is None:
            # 如果由于某种原因找不到张量，返回一个由NaN填充的矩阵
            # 这样依赖于它的计算结果也会是NaN，使问题清晰可见
            return np.full((3, 3), np.nan)
        return tensor

    def _get_density_gradient(self, coords: np.ndarray, field_interpolators: Dict[str, Any]) -> np.ndarray:
        """
        在指定笛卡尔坐标处，通过插值获取电子密度的三维梯度向量。

        参数:
            coords (np.ndarray): 笛卡尔坐标 (x, y, z)。
            field_interpolators (Dict[str, Any]): 包含梯度分量插值器 (grad_x, grad_y, grad_z) 的字典。

        返回:
            np.ndarray: 电子密度的三维梯度向量。如果插值器缺失或插值失败，则返回NaN向量。
        """
        grad_x_interp = field_interpolators.get('grad_x')
        grad_y_interp = field_interpolators.get('grad_y')
        grad_z_interp = field_interpolators.get('grad_z')

        if not all([grad_x_interp, grad_y_interp, grad_z_interp]):
            return np.full(3, np.nan) # 如果插值器缺失，返回NaN向量
        
        try:
            fractional_coords = self.pmg_structure.lattice.get_fractional_coords(coords)
            # 确保分数坐标在 [0, 1) 区间内，与插值器定义域匹配
            fractional_coords %= 1.0 
            
            # 使用新的裁剪函数，并传入梯度网格的形状
            # 假定所有梯度分量的网格形状相同，从grad_x_interp获取
            clipped_frac_coords = self._clip_fractional_coords(fractional_coords, grad_x_interp.values.shape)

            gradient = np.array([
                grad_x_interp(clipped_frac_coords)[0],
                grad_y_interp(clipped_frac_coords)[0], 
                grad_z_interp(clipped_frac_coords)[0]
            ])
            return gradient
        except Exception as e:
            print(f"  - 错误: 在坐标 {coords} 处梯度插值失败: {e}")
            return np.full(3, np.nan) # 失败时返回NaN向量

    def _get_point_group_order(self, site_index: int) -> int:
        """
        获取指定原子位点的点群阶数（位点对称性）。
        此方法将依赖于 `SpacegroupAnalyzer`。
        """
        if self.sga:
            try:
                # 获取整体对称操作
                symmetry_ops = self.sga.get_symmetry_operations()
                # 对于位点对称，我们需要找到保持该位点不变的对称操作数
                site = self.pmg_structure[site_index]
                site_symm_count = 0
                for sym_op in symmetry_ops:
                    # 应用对称操作到位点
                    new_coords = sym_op.operate(site.frac_coords)
                    # 检查操作后的坐标是否与原坐标等价（考虑周期性边界条件）
                    diff = new_coords - site.frac_coords
                    diff = diff - np.round(diff)  # 处理周期性边界条件
                    if np.allclose(diff, 0, atol=1e-3):
                        site_symm_count += 1
                return site_symm_count
            except Exception as e:
                print(f"  - 错误: 使用预加载的SGA分析原子 {site_index} 的位点对称失败: {e}。将使用后备方案。")
        
        # 后备方案：如果SGA未初始化或在特定位点上失败
        # 鉴于CrystalNN已被移除，此后备方案将简化。
        # 一个简单的保守估计：如果存在拓扑图且原子有连接，则阶数至少为2，否则为1。
        if self.topology_graph and self.topology_graph.get_coordination_of_site(site_index) > 0:
             return 2 # 假设存在至少一个连接，则点群阶数至少为2
        return 1 # 否则，最小为1

def run_2_simplex_features(cif_file: str,
                           atomic_features_csv: str,
                           bond_features_csv: str,
                           atomic_tensors_csv: str,
                           pw_gpw_file: Optional[str] = None,
                           fd_gpw_file: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           topology_graph: Optional[StructureGraph] = None) -> pd.DataFrame:
    """
    计算并保存2-单纯形（三角形）特征。

    参数:
        cif_file (str): 输入CIF文件的路径。
        atomic_features_csv (str): 0-单纯形特征CSV文件的路径。
        bond_features_csv (str): 1-单纯形特征CSV文件的路径。
        atomic_tensors_csv (str): 0-单纯形结构张量CSV文件的路径。
        pw_gpw_file (Optional[str]): 平面波GPW文件的路径。
        fd_gpw_file (Optional[str]): FD模式GPW文件的路径。
        output_dir (Optional[str]): 输出CSV文件的目录。如果为None，则使用cif_file所在的目录。

    返回:
        pd.DataFrame: 包含所有2-单纯形特征的DataFrame。
    """
    print("=" * 80)
    print("开始计算2-单纯形（三角形）特征")
    print("贝叶斯力学强化版方案")
    print("特征组: A基础几何, B跨层级派生, C几何重心量子, D贝叶斯力学融合, E局部嵌入拓扑")
    print("=" * 80)

    # 确保 output_dir 是 Path 对象
    if output_dir:
        output_path_dir = Path(output_dir)
        output_path_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path_dir = Path(cif_file).parent

    base_name = Path(cif_file).stem.replace('-optimized', '')
    output_csv = output_path_dir / f'{base_name}-2-Simplex-Features.csv'

    # 检查所有必需的输入文件是否存在
    required_files = {
        "0-Simplex特征文件": atomic_features_csv,
        "1-Simplex特征文件": bond_features_csv,
        "0-Simplex张量文件": atomic_tensors_csv,
    }
    for name, path in required_files.items():
        if not Path(path).exists():
            print(f"错误: 必需的 {name} 未找到: {path}")
            print("请确保已成功运行 0-Simplex-Features.py 和 1-Simplex-Features.py 脚本。")
            raise FileNotFoundError(f"Missing required file: {path}")

    try:
        calculator = TriangleFeatureCalculator(
            cif_file_path=cif_file,
            atomic_features_csv_path=atomic_features_csv,
            bond_features_csv_path=bond_features_csv,
            atomic_tensors_csv_path=atomic_tensors_csv,
            pw_gpw_file=pw_gpw_file,
            fd_gpw_file=fd_gpw_file,
            topology_graph=topology_graph
        )

        features_df = calculator.calculate_all_features()

        features_df.to_csv(output_csv, index=False)
        print(f"\n特征已保存到: {output_csv}")

        print("\n--- 2-单纯形特征计算完成 ---")
        return features_df

    except FileNotFoundError as e:
        print(f"\n错误: 2-单纯形计算关键输入文件缺失: {e}")
        raise
    except Exception as e:
        print(f"\n一个意外的错误发生于2-单纯形计算: {e}")
        import traceback
        traceback.print_exc()
        raise 