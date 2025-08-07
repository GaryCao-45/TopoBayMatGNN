import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Optional

from pymatgen.core import Structure, Site
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.io import read


from gpaw import GPAW # type: ignore
from gpaw.elf import ELF # type: ignore
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import laplace
from mendeleev import element

warnings.filterwarnings('ignore')

class TriangleFeatureCalculator:
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
                clipped_coords[d] = 0.0 # 对于单点维度，映射到0
        return clipped_coords


    def __init__(self, cif_file_path: str, atomic_features_csv_path: str, bond_features_csv_path: str, 
                 atomic_tensors_csv_path: str,
                 pw_gpw_file: Optional[str] = None, fd_gpw_file: Optional[str] = None,
                 random_seed: int = 42):

        # 设置可控随机种子
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.rng = np.random.default_rng(random_seed)
        
        self.cif_path = cif_file_path
        self.pmg_structure: Structure = Structure.from_file(cif_file_path)
        # self.ase_atoms = read(cif_file_path) # 未使用，已移除
        self.crystal_nn = CrystalNN()
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
        self.bond_features_dict = {}
        for _, row in self.bond_features_df.iterrows():
            site1_idx = int(row['site1_index'])
            site2_idx = int(row['site2_index'])
            bond_key = tuple(sorted([site1_idx, site2_idx]))
            self.bond_features_dict[bond_key] = row.to_dict()
        
        print(f"查询结构构建完成: {len(self.atomic_features_dict)} 个原子特征, {len(self.bond_features_dict)} 个键特征")

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

    def get_all_triangles(self) -> List[Tuple[int, int, int]]:
        """
        识别晶体结构中所有独特的原子三元组（三角形）。
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
        all_triangles = self.get_all_triangles()
        all_features_list = []

        # --- C/D部分特征计算的预处理 ---
        # 统一为场特征准备插值器，包括密度、ELF、拉普拉斯和梯度
        field_interpolators = self._setup_field_interpolators()

        # --- 对每个三角形进行统一计算 ---
        for indices in all_triangles:
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
            unit_normal = normal_vector / norm_n if norm_n > self._TOLERANCE else np.zeros(3)

            # --- A: 计算几何特征 ---
            features_A = self._compute_geometric_features_A(sites, v_ij, v_ik)
            
            # --- B: 计算跨层级派生特征 ---
            features_B = self._compute_derived_features_B(indices)
            
            # --- C: 计算量子化学特征 ---
            features_C = self._compute_quantum_features_C(sites, field_interpolators)
            
            # --- D: 计算代数融合特征 ---
            # 传入场插值器和三角形面积
            features_D = self._compute_fused_algebraic_features_D(
                indices, sites, unit_normal, features_A['triangle_area'], field_interpolators
            )
            
            # --- 合并所有特征 ---
            base_features = {
                'atom_index_i': indices[0], 'atom_index_j': indices[1], 'atom_index_k': indices[2],
                'atom_symbol_i': self.pmg_structure[indices[0]].specie.symbol,
                'atom_symbol_j': self.pmg_structure[indices[1]].specie.symbol,
                'atom_symbol_k': self.pmg_structure[indices[2]].specie.symbol,
            }
            
            # 过滤掉空的特征字典
            valid_features = [d for d in [base_features, features_A, features_B, features_C, features_D] if d]
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
        feature_cols = [col for col in final_df.columns if col not in id_cols]
        final_df = final_df[id_cols + feature_cols]
        
        print(f"\n所有特征计算完成。总特征维度: {len(final_df.columns) - len(id_cols)}")
        print("A部分: 3个几何特征")
        print("B部分: 12个跨层级派生特征") 
        print("C部分: 3个量子化学特征 (在几何重心处评估)")
        print("D部分: 5个代数融合特征")
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
        计算基于0-单纯形和1-单纯形特征的派生特征（B部分）。
        包括顶点和边的特征统计量（如平均值、方差、最大值）。

        参数:
            indices (Tuple[int, int, int]): 组成三角形的三个原子索引。

        返回:
            Dict[str, float]: 包含派生特征及其值的字典。
        """
        i, j, k = indices
        
        # B1: 顶点0-Simplex特征统计
        atomic_features = [self.atomic_features_dict.get(idx) for idx in [i, j, k]]
        atomic_features = [f for f in atomic_features if f is not None]

        incompatibilities = [f['structure_chemistry_incompatibility'] for f in atomic_features if 'structure_chemistry_incompatibility' in f and not pd.isna(f['structure_chemistry_incompatibility'])]
        bader_charges = [f['bader_charge'] for f in atomic_features if 'bader_charge' in f and not pd.isna(f['bader_charge'])]
        vectorial_asymmetries = [f['vectorial_asymmetry_norm_sq'] for f in atomic_features if 'vectorial_asymmetry_norm_sq' in f and not pd.isna(f['vectorial_asymmetry_norm_sq'])]

        avg_atomic_incompatibility = np.mean(incompatibilities) if incompatibilities else np.nan
        variance_atomic_incompatibility = np.var(incompatibilities) if incompatibilities else np.nan
        avg_bader_charge = np.mean(bader_charges) if bader_charges else np.nan
        variance_bader_charge = np.var(bader_charges) if bader_charges else np.nan
        avg_vectorial_asymmetry = np.mean(vectorial_asymmetries) if vectorial_asymmetries else np.nan
        max_vectorial_asymmetry = np.max(vectorial_asymmetries) if vectorial_asymmetries else np.nan

        # B2: 边1-Simplex特征统计
        bond_keys = [tuple(sorted([i, j])), tuple(sorted([j, k])), tuple(sorted([k, i]))]
        bond_features = [self.bond_features_dict.get(key) for key in bond_keys]
        bond_features = [f for f in bond_features if f is not None]

        geometric_alignments = [f['tensor_algebraic_environment_alignment'] for f in bond_features if 'tensor_algebraic_environment_alignment' in f and not pd.isna(f['tensor_algebraic_environment_alignment'])]
        bond_density_gradients = [f['bond_density_gradient'] for f in bond_features if 'bond_density_gradient' in f and not pd.isna(f['bond_density_gradient'])]
        bond_distances = [f['bond_distance'] for f in bond_features if 'bond_distance' in f and not pd.isna(f['bond_distance'])]

        avg_bond_alignment = np.mean(geometric_alignments) if geometric_alignments else np.nan
        variance_bond_alignment = np.var(geometric_alignments) if geometric_alignments else np.nan
        avg_bond_gradient = np.mean(bond_density_gradients) if bond_density_gradients else np.nan
        max_bond_gradient = np.max(bond_density_gradients) if bond_density_gradients else np.nan
        avg_bond_distance = np.mean(bond_distances) if bond_distances else np.nan
        variance_bond_distance = np.var(bond_distances) if bond_distances else np.nan

        return {
            'avg_atomic_incompatibility': avg_atomic_incompatibility,
            'variance_atomic_incompatibility': variance_atomic_incompatibility,
            'avg_bader_charge': avg_bader_charge,
            'variance_bader_charge': variance_bader_charge,
            'avg_vectorial_asymmetry': avg_vectorial_asymmetry,
            'max_vectorial_asymmetry': max_vectorial_asymmetry,
            'avg_bond_alignment': avg_bond_alignment,
            'variance_bond_alignment': variance_bond_alignment,
            'avg_bond_gradient': avg_bond_gradient,
            'max_bond_gradient': max_bond_gradient,
            'avg_bond_distance': avg_bond_distance,
            'variance_bond_distance': variance_bond_distance,
        }
        
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

    def _compute_quantum_features_C(self, sites: Tuple[Site, Site, Site], field_interpolators: Dict[str, Any]) -> Dict[str, float]:
        """
        计算三角形几何重心处的量子化学特征（C部分）。
        包括电子密度、电子密度拉普拉斯和ELF值。

        参数:
            sites (Tuple[Site, Site, Site]): 组成三角形的三个Site对象。
            field_interpolators (Dict[str, Any]): 包含各种场插值器的字典。

        返回:
            Dict[str, float]: 包含量子化学特征及其值的字典。
        """
        density_interp = field_interpolators.get('density')
        laplacian_interp = field_interpolators.get('laplacian')
        elf_interp = field_interpolators.get('elf')

        # 移除调试打印语句
        # print(f"  - DEBUG: 密度插值器状态: {density_interp is not None}, ELF插值器状态: {elf_interp is not None}")

        if not all([density_interp, laplacian_interp, elf_interp]):
            return {
                'geometric_centroid_density': np.nan,
                'geometric_centroid_elf': np.nan,
                'geometric_centroid_laplacian_of_density': np.nan,
            }

        # 使用几何重心作为评估点
        centroid_cartesian = (sites[0].coords + sites[1].coords + sites[2].coords) / 3.0
        centroid_fractional = self.pmg_structure.lattice.get_fractional_coords(centroid_cartesian)
        centroid_fractional %= 1.0

        # 分别为每个插值器裁剪坐标
        clipped_density_coords = self._clip_fractional_coords(centroid_fractional, density_interp.values.shape)
        clipped_elf_coords = self._clip_fractional_coords(centroid_fractional, elf_interp.values.shape)
        clipped_laplacian_coords = self._clip_fractional_coords(centroid_fractional, laplacian_interp.values.shape)

        return {
            'geometric_centroid_density': float(np.atleast_1d(density_interp(clipped_density_coords))[0]),
            'geometric_centroid_elf': float(np.atleast_1d(elf_interp(clipped_elf_coords))[0]),
            'geometric_centroid_laplacian_of_density': float(np.atleast_1d(laplacian_interp(clipped_laplacian_coords))[0]),
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
        计算三角形的代数融合特征（D部分）。
        包括结构张量积的迹、密度梯度通量、点群约简因子、结构张量法向投影和层次应力流。

        参数:
            indices (Tuple[int, int, int]): 组成三角形的三个原子索引。
            sites (Tuple[Site, Site, Site]): 组成三角形的三个Site对象。
            unit_normal (np.ndarray): 三角形平面的单位法向量。
            triangle_area (float): 三角形的面积。
            field_interpolators (Dict[str, Any]): 包含各种场插值器的字典。

        返回:
            Dict[str, float]: 包含代数融合特征及其值的字典。
        """
        if not field_interpolators:
            return {} # 依赖于密度梯度的特征无法计算

        i, j, k = indices
        p_i, p_j, p_k = sites[0].coords, sites[1].coords, sites[2].coords

        # --- D1: 李代数 (Lie Algebra) ---
        T_i = self._get_structure_tensor(i)
        T_j = self._get_structure_tensor(j)
        T_k = self._get_structure_tensor(k)
        
        # 重新定义 D1 特征：三张量乘积的迹。
        # 原有的“标量三重积”在对称张量情况下恒为零，因此需要全新定义。
        # 新定义：Tr(Ti * Tj * Tk)，量化三个局部结构张量的高阶相互作用。
        # 材料学假说：此特征量化了三个相邻原子局部结构环境的复杂耦合效应。
        # 非零值表示局部几何和力场的非线性相互作用，这可能是由结构扭曲、
        # 键应变或集体原子协同运动引起的。绝对值越大，这种高阶相互作用越显著。
        structural_tensor_product_trace = np.trace(T_i @ T_j @ T_k)

        # --- D2: 向量场分析 ---
        total_density_gradient_flux = np.nan
        if all(field_interpolators.get(key) for key in ['grad_x', 'grad_y', 'grad_z']):
            grad_rho_i = self._get_density_gradient(p_i, field_interpolators)
            grad_rho_j = self._get_density_gradient(p_j, field_interpolators)
            grad_rho_k = self._get_density_gradient(p_k, field_interpolators)
            # 使用更精确的"平均梯度投影 × 面积"来近似通量
            avg_grad = (grad_rho_i + grad_rho_j + grad_rho_k) / 3.0
            total_density_gradient_flux = np.dot(avg_grad, unit_normal) * triangle_area

        # --- D3: 商代数 (Quotient Algebra) ---
        order_i = self._get_point_group_order(i)
        order_j = self._get_point_group_order(j)
        order_k = self._get_point_group_order(k)
        avg_atomic_order = np.mean([order_i, order_j, order_k])
        triangle_order = min(order_i, order_j, order_k)
        point_group_reduction_factor = triangle_order / avg_atomic_order if avg_atomic_order > 0 else 0

        # --- D4: 张量代数 (Tensor Algebra) ---
        total_structure_tensor = T_i + T_j + T_k
        structure_tensor_normal_projection = unit_normal @ total_structure_tensor @ unit_normal

        # --- D5: 层次代数 (Hierarchical Algebra) ---
        # 新名称: hierarchical_stress_flow
        # 定义: 量化以"键不兼容性"为大小的"应力"矢量, 沿着每条边流出三角形的净通量。
        # 原实现dot(edge, normal)恒为0，现改为基于2D平面内边法线的通量计算。
        bond_keys = [tuple(sorted([i, j])), tuple(sorted([j, k])), tuple(sorted([k, i]))]
        edge_vectors = [p_j - p_i, p_k - p_j, p_i - p_k] # 保持循环顺序
        
        stress_flow_vector = np.zeros(3)
        valid_bonds = 0

        for bond_key, edge_vec in zip(bond_keys, edge_vectors):
            if bond_key in self.bond_features_dict:
                incompatibility = self.bond_features_dict[bond_key]['lie_algebra_incompatibility']
                
                # 计算边的2D平面内法线（指向外部）
                edge_normal_3d = np.cross(unit_normal, edge_vec)
                norm_en = np.linalg.norm(edge_normal_3d)
                
                if norm_en > self._TOLERANCE:
                    unit_edge_normal = edge_normal_3d / norm_en
                    # 应力流贡献 = 不兼容性 * 边法线方向
                    stress_flow_vector += incompatibility * unit_edge_normal
                    valid_bonds += 1
        
        # 最终特征是应力流伪矢量的模长
        hierarchical_stress_flow = np.linalg.norm(stress_flow_vector)

        return {
            'structural_tensor_product_trace': structural_tensor_product_trace,
            'total_density_gradient_flux': total_density_gradient_flux,
            'point_group_reduction_factor': point_group_reduction_factor,
            'structure_tensor_normal_projection': structure_tensor_normal_projection,
            'hierarchical_stress_flow': hierarchical_stress_flow,
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
        优先尝试使用预加载的SpacegroupAnalyzer进行精确计算；
        如果失败，则使用基于近邻数量的保守后备方案。

        参数:
            site_index (int): 原子位点在结构中的索引。

        返回:
            int: 位点的点群阶数。
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
        try:
            neighbors = self.crystal_nn.get_nn_info(self.pmg_structure, site_index)
            # 配位数为1的情况下，最小点群是C_1v (阶数为1)
            # 配位数更高时，保守估计为2 (可能有反演中心)
            return 2 if len(neighbors) > 1 else 1
        except Exception as e:
            print(f"  - 错误: 获取原子 {site_index} 的位点对称失败: {e}。将使用后备方案。")
            return 1


def main():

    cif_file = "CsPbI3-supercell-optimized.cif"
    base_name = Path(cif_file).stem.replace('-optimized', '')
    
    # 输入文件路径
    atomic_features_csv = f'{base_name}-0-Simplex-Features.csv'
    bond_features_csv = f'{base_name}-1-Simplex-Features.csv'
    # 新增结构张量文件路径
    atomic_tensors_csv = f'{base_name}-0-Simplex-Structure-Tensors.csv'
    pw_gpw_file = f'{base_name}.gpw'  # 平面波模式
    fd_gpw_file = f'{base_name}-fd.gpw'  # FD模式
    
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
            return

    print("=" * 80)
    print("2-单纯形特征计算器")
    print("统一计算流程：A几何, B跨层级, C量子(几何重心), D代数融合")
    print("=" * 80)
    
    # 初始化计算器（层次化输入）
    calculator = TriangleFeatureCalculator(
        cif_file_path=cif_file,
        atomic_features_csv_path=atomic_features_csv,
        bond_features_csv_path=bond_features_csv,
        atomic_tensors_csv_path=atomic_tensors_csv,
        pw_gpw_file=pw_gpw_file,
        fd_gpw_file=fd_gpw_file
    )
    
    # 计算A, B, C, D部分所有特征
    features_df = calculator.calculate_all_features()
    
    # 显示前几行结果
    print("\n前5个三角形的特征:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(features_df.head())

    # 保存到CSV
    output_path = f'{base_name}-2-Simplex-Features.csv'
    features_df.to_csv(output_path, index=False)
    print(f"\n特征已保存到: {output_path}")
    
    # 特征统计信息
    print(f"\n=== 特征统计===")
    print(f"识别的三角形数量: {len(features_df)}")
    print(f"总特征维度: {len(features_df.columns) - 6}")  # 减去6个标识列
    print(f"A部分特征: triangle_area, bond_angle_variance, triangle_shape_factor")
    print(f"B部分特征: 基于0-Simplex和1-Simplex的12个跨层级统计特征")
    print(f"C部分特征: geometric_centroid_density, geometric_centroid_elf, geometric_centroid_laplacian_of_density")
    print(f"D部分特征: 5个代数融合特征")


if __name__ == '__main__':
    main() 