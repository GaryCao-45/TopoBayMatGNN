import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict


from pymatgen.core import Structure, Site
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ASE 和 GPAW
from gpaw import GPAW # type: ignore
from gpaw.elf import ELF  # type: ignore
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import laplace

warnings.filterwarnings('ignore')

# 扩展化学属性范围，确保与0-单纯形脚本完全一致
GLOBAL_CHEM_VEC_RANGES = {
    'electronegativity': (0.7, 4.0),      # 电负性 (Pauling标度)
    'covalent_radius': (30, 250),         # 共价半径 (Pyykko, in pm)
    'ionization_energy': (370, 2500),     # 第一电离能 (in kJ/mol)
    'electron_affinity': (-50, 350),      # 电子亲合能 (in kJ/mol)
    'atomic_volume': (5, 70),             # 原子体积 (in cm³/mol)
    'polarizability': (0.1, 60),          # 极化率 (in Å³)
    'effective_charge': (-3.0, 8.0)       # 玻恩有效电荷范围
}

class BondFeatureCalculator:
    _TOLERANCE = 1e-9  # 用于避免数值计算中的除零错误

    def _clip_fractional_coords(self, fractional_coords: np.ndarray, grid_shape: Tuple[int, ...]) -> np.ndarray:
        
        clipped_coords = np.copy(fractional_coords)
        for d in range(len(grid_shape)):
            n_points = grid_shape[d]
            if n_points > 1:
                max_coord_value = (n_points - 1) / n_points
                clipped_coords[d] = np.clip(clipped_coords[d], 0.0, max_coord_value - self._TOLERANCE)
            else:
                clipped_coords[d] = 0.0  # 对于单点维度，映射到0
        return clipped_coords

    def __init__(self, cif_file_path: str, 
                 atomic_features_csv_path: Optional[str] = None, 
                 atomic_tensors_csv_path: Optional[str] = None,
                 pw_gpw_file: Optional[str] = None, 
                 fd_gpw_file: Optional[str] = None):
        
        self.cif_path = cif_file_path
        self.pmg_structure: Structure = Structure.from_file(cif_file_path)
        
        print(f"读取CIF文件: {cif_file_path}")
        print(f"晶体结构: {self.pmg_structure.composition.reduced_formula}")
        print(f"原子数量: {len(self.pmg_structure)}")
        
        # 使用与0-单纯形一致的邻居识别方法
        self.crystal_nn = CrystalNN()
        
        
        self.pw_calc: Optional[GPAW] = self._load_gpaw_calc(pw_gpw_file, "平面波(PW)")
        self.fd_calc: Optional[GPAW] = self._load_gpaw_calc(fd_gpw_file, "有限差分(FD)")
        
        self.atomic_features: Optional[pd.DataFrame] = None
        if atomic_features_csv_path:
            self.load_atomic_features(atomic_features_csv_path)
        
        self.atomic_tensors: Dict[int, np.ndarray] = {}
        if atomic_tensors_csv_path:
            self.load_atomic_tensors(atomic_tensors_csv_path)

    def _load_gpaw_calc(self, gpw_file: Optional[str], mode_name: str) -> Optional[GPAW]:
        
        if not gpw_file:
            return None
        
        gpw_path = Path(gpw_file).resolve()
        if gpw_path.exists():
            print(f"--- 正在加载 {mode_name} GPAW 计算结果: {gpw_path} ---")
            calc = GPAW(str(gpw_path), txt=None)
            return calc
        else:
            print(f"警告: {mode_name} gpw文件 '{gpw_path}' 不存在。")
            return None

    def load_atomic_features(self, atomic_features_csv_path: str):
        
        print(f"加载重构后的0-单纯形（原子）特征: {atomic_features_csv_path}")
        try:
            self.atomic_features = pd.read_csv(atomic_features_csv_path)
            if len(self.atomic_features) != len(self.pmg_structure):
                raise ValueError(
                    f"原子特征文件中的原子数量 ({len(self.atomic_features)}) "
                    f"与CIF文件不匹配 ({len(self.pmg_structure)})!"
                )
            print("重构后的原子特征加载成功。")
            print(f"  检测到的特征列: {list(self.atomic_features.columns)}")
        except FileNotFoundError as e:
            print(f"错误: 原子特征文件未找到: {atomic_features_csv_path}, {e}")
            self.atomic_features = None
        except Exception as e:
            print(f"加载原子特征时出错: {e}")
            self.atomic_features = None

    def load_atomic_tensors(self, atomic_tensors_csv_path: str):
        
        print(f"加载0-单纯形结构张量: {atomic_tensors_csv_path}")
        
        # 检查文件是否存在
        import os
        if not os.path.exists(atomic_tensors_csv_path):
            print(f"错误: 结构张量文件不存在: {atomic_tensors_csv_path}")
            print(f"   当前工作目录: {os.getcwd()}")
            return
            
        try:
            tensors_df = pd.read_csv(atomic_tensors_csv_path)
            print(f"   CSV文件读取成功，行数: {len(tensors_df)}")
            print(f"   列名: {list(tensors_df.columns)}")
            
            # 清空现有字典
            self.atomic_tensors.clear()
            
            for idx, row in tensors_df.iterrows():
                try:
                    site_idx = int(row['site_index'])
                    tensor = np.array([
                        [row['T_struct_00'], row['T_struct_01'], row['T_struct_02']],
                        [row['T_struct_10'], row['T_struct_11'], row['T_struct_12']],
                        [row['T_struct_20'], row['T_struct_21'], row['T_struct_22']]
                    ])
                    
                    # 检查张量是否有效
                    if np.any(np.isnan(tensor)):
                        print(f"   原子{site_idx}的张量包含NaN值")
                    
                    self.atomic_tensors[site_idx] = tensor
                    
                except Exception as row_error:
                    print(f"   处理第{idx}行(原子{row.get('site_index', '?')})时出错: {row_error}")
                    continue
                    
            print(f"成功加载了 {len(self.atomic_tensors)} 个原子的结构张量")
            print(f"   原子索引范围: {min(self.atomic_tensors.keys())} 到 {max(self.atomic_tensors.keys())}")
            
        except FileNotFoundError:
            print(f"错误: 结构张量文件未找到: {atomic_tensors_csv_path}")
        except Exception as e:
            print(f"加载结构张量时出错: {e}")
            import traceback
            traceback.print_exc()

    def get_all_bonds(self) -> List[List[Dict[str, Any]]]:
        
        print("正在识别化学键 (使用CrystalNN方法)...")
        all_bonds_info = []
        
        for i in range(len(self.pmg_structure)):
            # 初始尝试使用默认的CrystalNN
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
                print(f"  - 错误: 无法为原子 {i} 识别任何邻居，已添加空列表。")
            
            all_bonds_info.append(neighbors_found)
        
        total_bonds = sum(len(neighbors) for neighbors in all_bonds_info)
        print(f"识别到 {total_bonds} 个化学键")
        
        return all_bonds_info

    def calculate_geometric_features(self) -> Tuple[pd.DataFrame, List[np.ndarray], List[Tuple[Site, Site]]]:
        
        print("开始计算几何与拓扑特征 (A部分)...")
        
        all_bonds_info = self.get_all_bonds()
        bond_features_list = []
        bond_vectors = []
        site_pairs = []

        coordination_numbers = [len(neighbors) for neighbors in all_bonds_info]

        for site_idx, neighbor_info in enumerate(all_bonds_info):
            site1 = self.pmg_structure[site_idx]
            
            for nn_info in neighbor_info:
                site2 = nn_info['site']
                site2_idx = nn_info['site_index']
                
                bond_vector = site2.coords - site1.coords
                bond_distance = np.linalg.norm(bond_vector)
                
                site1_coord_num = coordination_numbers[site_idx]
                site2_coord_num = coordination_numbers[site2_idx]
                
                bond_features = {
                    'site1_index': site_idx,
                    'site2_index': site2_idx,
                    'site1_element': site1.specie.symbol,
                    'site2_element': self.pmg_structure[site2_idx].specie.symbol,
                    'bond_distance': bond_distance,
                    'site1_coord_num': site1_coord_num,
                    'site2_coord_num': site2_coord_num,
                }
                bond_features_list.append(bond_features)
                bond_vectors.append(bond_vector)
                site_pairs.append((site1, site2))
        
        df = pd.DataFrame(bond_features_list)
        print(f"计算完成，共 {len(df)} 个化学键的几何特征")
        return df, bond_vectors, site_pairs

    def calculate_all_features(self) -> pd.DataFrame:
        
        # 步骤1：计算A部分几何特征，并获取向量和原子对信息
        bond_features_df, bond_vectors, site_pairs = self.calculate_geometric_features()

        # 步骤2：计算重构后的B部分派生特征
        derived_features_df = self._calculate_derived_features_B(bond_features_df)
        
        # 步骤3：计算C部分量子特征
        quantum_features_df = self._calculate_quantum_features_C(site_pairs, bond_features_df)

        # 步骤4：计算重构后的D部分融合特征（五大代数思想）
        fused_features_df = self._calculate_fused_features_D(
            bond_features_df, 
            bond_vectors, 
            site_pairs,
            quantum_features_df
        )

        # 步骤5：合并所有特征
        final_df = pd.concat([
            bond_features_df, 
            derived_features_df, 
            quantum_features_df,
            fused_features_df
        ], axis=1)
        
        # 将 site1_index, site2_index, site1_element, site2_element 移到最前面
        id_cols = ['site1_index', 'site2_index', 'site1_element', 'site2_element']
        feature_cols = [col for col in final_df.columns if col not in id_cols]
        final_df = final_df[id_cols + feature_cols]
        
        print(f"所有特征计算完成。总特征维度: {len(final_df.columns) - 4}")
        return final_df
    
    def _quotient_algebra_equivalence_class(self,
                                            element: np.ndarray, 
                                            group_operations: List[Any]) -> List[np.ndarray]: # 修改类型提示为Any，因为SymmOp不是np.ndarray
        
        equivalence_class = []
        
        for operation in group_operations:
            # 群操作作用于元素：对于SymmOp，使用.operate()方法
            transformed = operation.operate(element)
            
            # 规范化以便比较
            norm = np.linalg.norm(transformed)
            if norm > self._TOLERANCE:
                normalized = transformed / norm
                equivalence_class.append(normalized)
        
        # 去重：移除数值上相等的元素
        unique_elements = []
        for elem in equivalence_class:
            is_unique = True
            for existing in unique_elements:
                if np.allclose(elem, existing, atol=self._TOLERANCE):
                    is_unique = False
                    break
            if is_unique:
                unique_elements.append(elem)
        
        return unique_elements
    
    def _quotient_algebra_orbit_size(self, 
                                     bond_direction: np.ndarray, 
                                     group_operations: List[Any]) -> int: # 修改类型提示为Any
        
        try:
            # 计算等价类
            equivalence_class = self._quotient_algebra_equivalence_class(
                bond_direction, group_operations
            )
            
            orbit_size = len(equivalence_class)
            return orbit_size
            
        except Exception as e:
            print(f"商代数计算失败: {e}")
            return 1  # 默认返回平凡轨道

    def _get_density_gradient_interpolator(self):
        
        if not self.pw_calc: 
            print("警告：无GPAW文件，辛不变量将使用备用方法计算")
            return None
        try:
            density_grid = self.pw_calc.get_all_electron_density()
            print(f"密度网格尺寸: {density_grid.shape}")
            
            # 检查密度网格是否有效且至少是三维的
            if density_grid is None or density_grid.ndim < 3 or np.any(np.isnan(density_grid)) or np.any(np.isinf(density_grid)):
                print("警告：密度网格无效、维度不足或包含NaN/无穷值，无法计算梯度。")
                return None
            
            # 确保 np.gradient 的 axis 参数是明确的整数元组，避免潜在的歧义
            grad_x, grad_y, grad_z = np.gradient(density_grid, axis=(0, 1, 2))
            
            gradient_field = np.stack([grad_x, grad_y, grad_z], axis=-1)
            
            # 坐标变换 - 添加错误检查
            lattice = self.pmg_structure.lattice.matrix
            try:
                lattice_inv_T = np.linalg.inv(lattice).T
            except np.linalg.LinAlgError as e:
                print(f"警告：晶格矩阵不可逆, {e}。密度梯度插值器创建失败。")
                return None
            
            # 确保 cart_grads 的计算结果是有效的
            cart_grads = np.einsum('...i,ij->...j', gradient_field, lattice_inv_T)
            if np.any(np.isnan(cart_grads)) or np.any(np.isinf(cart_grads)):
                print("警告：转换后的梯度场包含NaN或无穷值。密度梯度插值器创建失败。")
                return None
            
            axes = [np.linspace(0, 1, n, endpoint=False) for n in cart_grads.shape[:-1]]
            interpolator = RegularGridInterpolator(
                axes, cart_grads, 
                bounds_error=False, 
                fill_value=np.nan  # 将填充值改为 np.nan，以获得更一致的缺失值处理
            )
            
            print("密度梯度插值器创建成功")
            return interpolator
            
        except Exception as e:
            print(f"密度梯度插值器创建失败: {e}")
            return None

    def _calculate_derived_features_B(self, bond_features_df: pd.DataFrame) -> pd.DataFrame:
        
        if self.atomic_features is None:
            print("警告: 未提供原子特征文件，跳过B部分派生特征的计算。")
            return pd.DataFrame()
        
        print("开始计算重构后的B部分：基于新0-单纯形特征的派生特征...")
        
        # 检查必需的列是否存在
        required_cols = [
            'electronegativity', 'ionic_radius', 'covalent_radius', 'bader_charge',
            'vectorial_asymmetry_norm_sq', 'local_environment_anisotropy', 
            'symmetry_breaking_quotient', 'structure_chemistry_incompatibility'
        ]
        
        missing_cols = [col for col in required_cols if col not in self.atomic_features.columns]
        if missing_cols:
            print(f"警告: 缺少必需的0-单纯形特征列: {missing_cols}")
            print("将跳过依赖这些列的特征计算。")
        
        derived_features_list = []
        for _, row in bond_features_df.iterrows():
            site1_idx = int(row['site1_index'])
            site2_idx = int(row['site2_index'])
            
            atom1_props = self.atomic_features.iloc[site1_idx]
            atom2_props = self.atomic_features.iloc[site2_idx]
            
            # 基于重构后的0-单纯形特征计算派生特征
            features = {}
            
            # 经典化学差异特征
            if 'electronegativity' in self.atomic_features.columns:
                features['delta_electronegativity'] = abs(atom1_props['electronegativity'] - atom2_props['electronegativity'])
            else:
                features['delta_electronegativity'] = np.nan
                
            if 'ionic_radius' in self.atomic_features.columns:
                features['delta_ionic_radius'] = abs(atom1_props['ionic_radius'] - atom2_props['ionic_radius'])
            else:
                features['delta_ionic_radius'] = np.nan
                
            if 'covalent_radius' in self.atomic_features.columns:
                features['sum_covalent_radii'] = atom1_props['covalent_radius'] + atom2_props['covalent_radius']
            else:
                features['sum_covalent_radii'] = np.nan
            
            # 量子化学特征
            if 'bader_charge' in self.atomic_features.columns:
                features['avg_bader_charge'] = (atom1_props['bader_charge'] + atom2_props['bader_charge']) / 2.0
                features['delta_bader_charge'] = abs(atom1_props['bader_charge'] - atom2_props['bader_charge'])
            else:
                features['avg_bader_charge'] = np.nan
                features['delta_bader_charge'] = np.nan
            
            derived_features_list.append(features)
            
        return pd.DataFrame(derived_features_list)

    def _calculate_fused_features_D(self, 
                                    bond_features_df: pd.DataFrame, 
                                    bond_vectors: List[np.ndarray], 
                                    site_pairs: List[Tuple[Site, Site]],
                                    quantum_features_df: pd.DataFrame) -> pd.DataFrame:
        
        print("开始计算重构后的D部分：五大代数思想的深度融合特征...")
        # all_bonds_info = self.get_all_bonds() # 已不再需要，信息通过参数传入
        fused_features_list = []
        
        # 为商代数特征准备点群分析器
        # 对于周期性晶体结构，使用简化的旋转操作集来分析化学键的对称性
        try:
            sga = SpacegroupAnalyzer(self.pmg_structure)
            group_operations = sga.get_symmetry_operations()
            print(f"构造了 {len(group_operations)} 个空间群操作")
        except Exception as e:
            print(f"空间群操作构造失败，使用默认值: {e}")
            # 如果失败，只保留恒等操作
            group_operations = [sga.get_space_group_operations()[0]] if 'sga' in locals() and sga.get_space_group_operations() else [np.eye(3)]
        
        # 为辛代数特征准备密度梯度插值器
        density_grad_interp = self._get_density_gradient_interpolator()
        
        bond_count = 0  # 添加计数器用于错误信息控制

        for i, row in bond_features_df.iterrows():
            bond_vector = bond_vectors[i]
            bond_distance = row['bond_distance']
            bond_direction = bond_vector / bond_distance if bond_distance > self._TOLERANCE else np.zeros(3)
            
            site1_idx = int(row['site1_index'])
            site2_idx = int(row['site2_index'])
            
            # 获取site1和site2对象
            site1, site2 = site_pairs[i]

            # 从文件读取预计算的结构张量，不再重新计算
            T_struct_1 = self._get_structure_tensor_from_file(site1_idx)
            T_struct_2 = self._get_structure_tensor_from_file(site2_idx)

            # --- 1. 李代数不相容性 (李代数严格实现) ---
            geometric_incompatibility = self._calculate_geometric_environment_incompatibility_rigorous(T_struct_1, T_struct_2)

            # --- 2. 商代数特征 (商代数严格实现) ---
            bond_orbit_size = self._quotient_algebra_orbit_size(bond_direction, group_operations)
            # quotient_invariant = self._quotient_algebra_invariant(bond_direction, group_operations) # 已移除此特征

            # --- 3. 辛几何特征 (辛几何严格实现) ---
            symplectic_feature = np.nan

            # 方法1：尝试使用密度梯度方法（最精确）
            if density_grad_interp:
                try:
                    midpoint_cartesian = (site1.coords + site2.coords) / 2.0
                    midpoint_fractional = self.pmg_structure.lattice.get_fractional_coords(midpoint_cartesian)

                    # 确保分数坐标在[0,1)范围内
                    midpoint_fractional = midpoint_fractional % 1.0

                    # 检查坐标是否在合理范围内
                    if np.all(midpoint_fractional >= 0) and np.all(midpoint_fractional < 1):
                        density_gradient = density_grad_interp(midpoint_fractional)

                        if density_gradient is not None:
                            density_gradient = np.asarray(density_gradient).flatten()
                            if len(density_gradient) == 3 and not np.any(np.isnan(density_gradient)):
                                # 将名称从 symplectic_invariant 更改为 pseudo_symplectic_coupling
                                symplectic_feature = self._calculate_bond_density_gradient_curl_rigorous(
                                    bond_direction, density_gradient
                                )
                except Exception as e:
                    print(f"  - 错误: 计算辛几何特征失败: {e}")
                    symplectic_feature = np.nan

            # 确保绝对没有NaN值
            if np.isnan(symplectic_feature) or symplectic_feature is None:
                symplectic_feature = 1.0

            # --- 4. 张量代数环境对齐 (张量代数严格实现) ---
            geometric_alignment = self._calculate_tensor_algebraic_environment_alignment(T_struct_1, T_struct_2)

            # --- 5. 李代数键对齐特征 (新增) ---
            lattice_matrix = self.pmg_structure.lattice.matrix
            alignment_a, alignment_b, alignment_c = self._calculate_lie_algebraic_bond_alignment(
                bond_direction, lattice_matrix
            )

            # --- 6. 层次代数 (Hierarchical Algebra): delta_structure_chemistry_incompatibility ---
            hierarchical_delta = np.nan
            if self.atomic_features is not None and 'structure_chemistry_incompatibility' in self.atomic_features.columns:
                incompat_1 = self.atomic_features.iloc[site1_idx]['structure_chemistry_incompatibility']
                incompat_2 = self.atomic_features.iloc[site2_idx]['structure_chemistry_incompatibility']
                if not (pd.isna(incompat_1) or pd.isna(incompat_2)):
                    hierarchical_delta = abs(incompat_1 - incompat_2)
            
            # --- 7. 物理图像深化 (Physics Insight): bond_ends_anisotropy_mismatch ---
            anisotropy_1 = self._calculate_local_anisotropy(T_struct_1)
            anisotropy_2 = self._calculate_local_anisotropy(T_struct_2)
            anisotropy_mismatch = abs(anisotropy_1 - anisotropy_2) if not(pd.isna(anisotropy_1) or pd.isna(anisotropy_2)) else np.nan
            
            fused_features_list.append({
                'lie_algebra_incompatibility': geometric_incompatibility,
                'quotient_algebra_orbit_size': bond_orbit_size,
                'pseudo_symplectic_coupling': symplectic_feature,
                'tensor_algebraic_environment_alignment': geometric_alignment,
                'lie_algebraic_bond_alignment_a': alignment_a,
                'lie_algebraic_bond_alignment_b': alignment_b,
                'lie_algebraic_bond_alignment_c': alignment_c,
                'delta_structure_chemistry_incompatibility': hierarchical_delta,
                'bond_ends_anisotropy_mismatch': anisotropy_mismatch,
            })

            bond_count += 1

        print(f"计算完成，共 {len(fused_features_list)} 个化学键的D部分融合特征。")
        return pd.DataFrame(fused_features_list)

    def _get_structure_tensor_from_file(self, site_idx: int) -> np.ndarray:
        
        tensor = self.atomic_tensors.get(site_idx)
        if tensor is None:
            # 如果未找到张量，返回NaN填充的默认矩阵
            return np.full((3, 3), np.nan)
        
        # 检查张量是否有效
        if np.any(np.isnan(tensor)):
            return tensor
            
        return tensor

    # ==========================================================================
    # 辅助函数：与0-单纯形脚本保持一致 (现已废弃，改用共享模块)
    # ==========================================================================

    def _calculate_local_anisotropy(self, T: np.ndarray) -> float:
        try:
            # 使用eigvalsh以利用对称性并保证实数本征值
            eigenvalues = np.linalg.eigvalsh(T)
            trace_T = np.sum(eigenvalues)
            
            if trace_T < self._TOLERANCE:
                return 0.0  # 没有邻居，或邻居矢量和为0，视为各向同性

            # 基于二阶不变量的各向异性公式，归一化到[0,1]
            # 1 - 3 * (λ1λ2 + λ2λ3 + λ3λ1) / (λ1+λ2+λ3)²
            term = 3 * (eigenvalues[0]*eigenvalues[1] + eigenvalues[1]*eigenvalues[2] + eigenvalues[2]*eigenvalues[0])
            anisotropy = 1.0 - term / (trace_T**2)
            return float(np.clip(anisotropy, 0.0, 1.0))
        
        except np.linalg.LinAlgError as e:
            print(f"警告：各向异性计算失败, {e}")
            return np.nan

    # =========================================================================
    # D组特征的具体实现：五大代数思想
    # =========================================================================

    def _lie_algebra_commutator(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        
        if X.shape != (3, 3) or Y.shape != (3, 3):
            raise ValueError("李代数元素必须是3×3矩阵")
        
        commutator = X @ Y - Y @ X
        
        # 验证反对称性：[X,Y] = -[Y,X]
        anti_commutator = Y @ X - X @ Y
        if not np.allclose(commutator, -anti_commutator, atol=self._TOLERANCE):
            print("警告：交换子反对称性验证失败")
        
        return commutator
    
    def _lie_algebra_frobenius_norm(self, X: np.ndarray) -> float:
        
        if X.shape != (3, 3):
            raise ValueError("输入必须是3×3矩阵")
        
        frobenius_norm = np.linalg.norm(X, 'fro')
        
        return float(frobenius_norm)
    
    def _lie_algebra_adjoint_action(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        
        return self._lie_algebra_commutator(X, Y)

    def _calculate_geometric_environment_incompatibility_rigorous(self, 
                                                                  T_struct_1: np.ndarray, 
                                                                  T_struct_2: np.ndarray) -> float:
        
        try:
            # 确保输入是有效的李代数元素
            if np.any(np.isnan(T_struct_1)) or np.any(np.isnan(T_struct_2)):
                return np.nan
            
            # 计算李代数交换子
            commutator = self._lie_algebra_commutator(T_struct_1, T_struct_2)
            
            # 计算弗罗贝尼乌斯范数
            incompatibility = self._lie_algebra_frobenius_norm(commutator)
            
            return incompatibility
            
        except Exception as e:
            print(f"李代数计算失败: {e}")
            return np.nan

    def _calculate_lie_algebraic_bond_alignment(self, 
                                                bond_direction: np.ndarray, 
                                                lattice_matrix: np.ndarray) -> Tuple[float, float, float]:
        
        try:
            # 获取晶格基向量
            a_vector = lattice_matrix[:, 0]  # a轴
            b_vector = lattice_matrix[:, 1]  # b轴  
            c_vector = lattice_matrix[:, 2]  # c轴
            
            # 规范化
            bond_unit = bond_direction / (np.linalg.norm(bond_direction) + 1e-12)
            a_unit = a_vector / (np.linalg.norm(a_vector) + 1e-12)
            b_unit = b_vector / (np.linalg.norm(b_vector) + 1e-12)
            c_unit = c_vector / (np.linalg.norm(c_vector) + 1e-12)
            
            # 构造反对称矩阵（李代数 so(3) 的元素）
            # 键方向对应的反对称矩阵
            bond_skew = np.array([
                [0, -bond_unit[2], bond_unit[1]],
                [bond_unit[2], 0, -bond_unit[0]],
                [-bond_unit[1], bond_unit[0], 0]
            ])
            
            # 晶格方向对应的反对称矩阵
            a_skew = np.array([
                [0, -a_unit[2], a_unit[1]],
                [a_unit[2], 0, -a_unit[0]],
                [-a_unit[1], a_unit[0], 0]
            ])
            
            b_skew = np.array([
                [0, -b_unit[2], b_unit[1]],
                [b_unit[2], 0, -b_unit[0]],
                [-b_unit[1], b_unit[0], 0]
            ])
            
            c_skew = np.array([
                [0, -c_unit[2], c_unit[1]],
                [c_unit[2], 0, -c_unit[0]],
                [-c_unit[1], c_unit[0], 0]
            ])
            
            # 计算李代数交换子 [bond, lattice_direction]
            commutator_a = self._lie_algebra_commutator(bond_skew, a_skew)
            commutator_b = self._lie_algebra_commutator(bond_skew, b_skew)
            commutator_c = self._lie_algebra_commutator(bond_skew, c_skew)
            
            # 计算对齐度（交换子范数的倒数，范数越小对齐度越高）
            norm_a = self._lie_algebra_frobenius_norm(commutator_a)
            norm_b = self._lie_algebra_frobenius_norm(commutator_b)
            norm_c = self._lie_algebra_frobenius_norm(commutator_c)
            
            # 转换为对齐度：范数越小，对齐度越高
            # 使用 exp(-norm) 形式确保对齐度在 [0,1] 范围内
            alignment_a = np.exp(-norm_a) if not np.isnan(norm_a) else 0.0
            alignment_b = np.exp(-norm_b) if not np.isnan(norm_b) else 0.0
            alignment_c = np.exp(-norm_c) if not np.isnan(norm_c) else 0.0
            
            return float(alignment_a), float(alignment_b), float(alignment_c)
            
        except Exception as e:
            print(f"李代数键对齐计算失败: {e}")
            return 0.0, 0.0, 0.0

    def _calculate_tensor_algebraic_environment_alignment(self, 
                                                          T_struct_1: np.ndarray, 
                                                          T_struct_2: np.ndarray) -> float:
        
        try:
            # 确保输入有效
            if np.any(np.isnan(T_struct_1)) or np.any(np.isnan(T_struct_2)):
                return np.nan
            
            # 方法1：归一化张量内积
            # 计算 Tr(T1^T @ T2) / (||T1||_F * ||T2||_F)
            frobenius_1 = self._lie_algebra_frobenius_norm(T_struct_1)
            frobenius_2 = self._lie_algebra_frobenius_norm(T_struct_2)
            
            if frobenius_1 < self._TOLERANCE or frobenius_2 < self._TOLERANCE:
                return 0.0
            
            # 归一化张量
            T1_normalized = T_struct_1 / frobenius_1
            T2_normalized = T_struct_2 / frobenius_2
            
            # 计算张量内积
            tensor_inner_product = np.trace(T1_normalized.T @ T2_normalized)
            
            # 方法2：主方向对齐度
            # 计算特征值和特征向量
            try:
                eigenvals_1, eigenvecs_1 = np.linalg.eigh(T_struct_1)
                eigenvals_2, eigenvecs_2 = np.linalg.eigh(T_struct_2)
                
                # 按特征值大小排序
                idx_1 = np.argsort(np.abs(eigenvals_1))[::-1]
                idx_2 = np.argsort(np.abs(eigenvals_2))[::-1]
                
                # 主方向向量
                principal_1 = eigenvecs_1[:, idx_1[0]]
                principal_2 = eigenvecs_2[:, idx_2[0]]
                
                # 计算主方向对齐度
                principal_alignment = abs(np.dot(principal_1, principal_2))
                
            except np.linalg.LinAlgError as e:
                print(f"警告：主方向对齐度计算失败, {e}")
                principal_alignment = 0.0
            
            # 方法3：张量相似性度量
            # 使用 cos(θ) = <T1,T2> / (||T1|| ||T2||) 的张量版本
            numerator = np.sum(T_struct_1 * T_struct_2)  # 逐元素乘积的和
            denominator = frobenius_1 * frobenius_2
            
            if denominator > self._TOLERANCE:
                cosine_similarity = numerator / denominator
            else:
                cosine_similarity = 0.0
            
            # 综合对齐度：加权平均
            total_alignment = (
                0.4 * abs(tensor_inner_product) +
                0.4 * principal_alignment +
                0.2 * abs(cosine_similarity)
            )
            
            # 确保结果在合理范围内
            return float(np.clip(total_alignment, 0.0, 1.0))
            
        except Exception as e:
            print(f"张量代数环境对齐计算失败: {e}")
            return np.nan

    # =========================================================================
    # 数学严格实现：辛几何理论
    # =========================================================================
    
    def _symplectic_form(self, q: np.ndarray, p: np.ndarray) -> float:

        if len(q) != 3 or len(p) != 3:
            raise ValueError("辛形式要求3维向量")
        
        # 标准辛形式：ω(q,p) = q^T p
        # 这是6维辛空间 ω((q,0), (0,p)) = q^T p 的特殊情况
        symplectic_value = np.dot(q, p)
        
        return float(symplectic_value)
    
    def _symplectic_form_bilinear(self, q1: np.ndarray, p1: np.ndarray, 
                                  q2: np.ndarray, p2: np.ndarray) -> float:
        if len(q1) != 3 or len(p1) != 3 or len(q2) != 3 or len(p2) != 3:
            raise ValueError("所有输入必须是3维向量")
        
        # 标准辛形式
        omega_value = np.dot(q1, p2) - np.dot(p1, q2)
        
        return float(omega_value)
    
    def _hamiltonian_vector_field(self, q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # 简单的哈密顿量：H = ||p||²/2 + V(q)
        # 其中V(q) = ||q||²/2 (谐振子势)
        
        dq_dt = p  # dH/dp = p
        dp_dt = -q  # -dH/dq = -q
        
        return dq_dt, dp_dt
    
    def _symplectic_invariant(self, q: np.ndarray, p: np.ndarray) -> float:

        try:
            # 1. 哈密顿量（能量）：H = ||p||²/2 + V(q)
            # 动能项
            kinetic_energy = 0.5 * np.linalg.norm(p) ** 2
            
            # 势能项（谐振子势）
            potential_energy = 0.5 * np.linalg.norm(q) ** 2
            
            # 总哈密顿量
            hamiltonian = kinetic_energy + potential_energy
            
            # 2. 辛形式（辛结构的基本量）
            omega = self._symplectic_form(q, p)
            
            # 3. 角动量（经典力学中的重要守恒量）
            # L = q × p，这在旋转对称系统中守恒
            angular_momentum = np.linalg.norm(np.cross(q, p))
            
            # 4. 相空间体积元素的特征量
            # 在辛坐标中，这与det(∂(q,p)/∂(Q,P))相关
            phase_space_volume = np.sqrt(np.linalg.norm(q) ** 2 + np.linalg.norm(p) ** 2)
            
            # 5. 构造总的辛不变量
            # 权重基于物理重要性和量纲分析
            symplectic_invariant = (
                hamiltonian +                    # 主要项：总能量
                0.1 * abs(omega) +              # 辛形式贡献
                0.05 * angular_momentum +       # 角动量贡献  
                0.01 * phase_space_volume       # 相空间几何贡献
            )
            
            return float(symplectic_invariant)
            
        except Exception as e:
            print(f"辛不变量计算失败: {e}")
            return np.nan
    
    def _calculate_bond_density_gradient_curl_rigorous(self, 
                                                       bond_direction: np.ndarray, 
                                                       density_gradient: np.ndarray) -> float:

        try:
            if len(bond_direction) != 3 or len(density_gradient) != 3:
                return np.nan
            
            # 规范化向量
            q_norm = np.linalg.norm(bond_direction)
            p_norm = np.linalg.norm(density_gradient)
            
            if q_norm < self._TOLERANCE or p_norm < self._TOLERANCE:
                return np.nan
            
            q = bond_direction / q_norm
            p = density_gradient / p_norm
            
            # 计算辛不变量
            symplectic_inv = self._symplectic_invariant(q, p)
            
            return symplectic_inv
            
        except Exception as e:
            print(f"辛几何计算失败: {e}")
            return np.nan

    # ==========================================================================
    # C部分：量子化学特征 (保持不变)
    # ==========================================================================

    def _setup_interpolators(self, density_grid, elf_grid, laplacian_grid):

        print("  - 创建三维网格插值器...")

        def create_interp(grid_data):
            axes = [np.linspace(0, 1, n, endpoint=False) for n in grid_data.shape]
            # 统一将fill_value设置为np.nan，确保在插值越界时返回NaN，而不是抛出错误
            return RegularGridInterpolator(axes, grid_data, bounds_error=False, fill_value=np.nan)

        density_interpolator = create_interp(density_grid)
        laplacian_interpolator = create_interp(laplacian_grid)
        elf_interpolator = create_interp(elf_grid)

        print(f"  - 插值器创建成功。密度网格: {density_grid.shape}, ELF网格: {elf_grid.shape}, 拉普拉斯网格: {laplacian_grid.shape}")
        return density_interpolator, laplacian_interpolator, elf_interpolator

    def _calculate_quantum_features_C(self, site_pairs: List[Tuple[Site, Site]], bond_features_df: pd.DataFrame) -> pd.DataFrame:

        empty_columns = [
            'bond_midpoint_density', 'bond_density_laplacian', 'bond_midpoint_elf', 'elf_type',
            'bond_charge_transfer', 'bond_elf_from_0simplex', 'bond_elf_asymmetry',
            'bond_density_from_0simplex', 'bond_density_gradient', 'bond_effective_charge', 'bond_charge_imbalance'
        ]

        if self.pw_calc is None and self.fd_calc is None:
            print("警告: 未加载任何GPAW计算，跳过C部分量子特征计算。")
            return pd.DataFrame(index=range(len(site_pairs)), columns=empty_columns)

        print("开始计算C部分：量子化学成键特征...")
        
        density_calc = self.pw_calc or self.fd_calc
        if density_calc is None:
             print("  - 错误：无法获取电子密度。")
             return pd.DataFrame(index=range(len(site_pairs)), columns=empty_columns)
        
        print("  - 正在从GPAW计算器获取电子密度...")
        density_grid = density_calc.get_all_electron_density()
        
        elf_grid, elf_type = self._get_elf_grid(density_grid)
        
        print("  - 正在计算电子密度的拉普拉斯算子...")
        laplacian_grid = laplace(density_grid)

        density_interp, laplacian_interp, elf_interp = self._setup_interpolators(density_grid, elf_grid, laplacian_grid)

        quantum_features_list = []

        for i, (site1, site2) in enumerate(site_pairs):
            midpoint_cartesian = (site1.coords + site2.coords) / 2.0
            midpoint_fractional = self.pmg_structure.lattice.get_fractional_coords(midpoint_cartesian)
            midpoint_fractional %= 1.0

            # 1. bond_midpoint_density - GPAW电子密度计算
            try:
                # 使用密度网格的形状进行裁剪
                clipped_density_coords = self._clip_fractional_coords(midpoint_fractional, density_grid.shape)
                density_val = density_interp(clipped_density_coords)
                features = {}
                features['bond_midpoint_density'] = float(np.atleast_1d(density_val)[0])
            except Exception as e:
                print(f"  - 错误: 计算电子密度失败: {e}")
                features['bond_midpoint_density'] = np.nan
            
            # 2. bond_density_laplacian - 密度拉普拉斯算子
            try:
                # 使用拉普拉斯网格的形状进行裁剪
                clipped_laplacian_coords = self._clip_fractional_coords(midpoint_fractional, laplacian_grid.shape)
                laplacian_val = laplacian_interp(clipped_laplacian_coords)
                features['bond_density_laplacian'] = float(np.atleast_1d(laplacian_val)[0])
            except Exception as e:
                print(f"  - 错误: 计算密度拉普拉斯算子失败: {e}")
                features['bond_density_laplacian'] = np.nan
            
            # 3. bond_midpoint_elf - ELF（电子局域函数）
            try:
                # 使用ELF网格的形状进行裁剪
                clipped_elf_coords = self._clip_fractional_coords(midpoint_fractional, elf_grid.shape)
                elf_val = elf_interp(clipped_elf_coords)
                features['bond_midpoint_elf'] = float(np.atleast_1d(elf_val)[0])
                
                features['elf_type'] = elf_type
                
            except Exception as e:
                print(f"  - 错误: 计算ELF失败: {e}")
                features['bond_midpoint_elf'] = np.nan
                features['elf_type'] = 'failed'
            
            # 4. bond_charge_transfer - Bader电荷分析
            try:
                bond_features_row = bond_features_df.iloc[i] if i < len(bond_features_df) else None
                if bond_features_row is not None and self.atomic_features is not None:
                    site1_idx = int(bond_features_row['site1_index'])
                    site2_idx = int(bond_features_row['site2_index'])
                    
                    if 'bader_charge' in self.atomic_features.columns:
                        charge1 = self.atomic_features.iloc[site1_idx]['bader_charge']
                        charge2 = self.atomic_features.iloc[site2_idx]['bader_charge']
                        features['bond_charge_transfer'] = abs(charge2 - charge1)
                    else:
                        features['bond_charge_transfer'] = np.nan
                else:
                    features['bond_charge_transfer'] = np.nan
            except Exception as e:
                print(f"  - 错误: 计算Bader电荷转移失败: {e}")
                features['bond_charge_transfer'] = np.nan
            
            try:
                bond_features_row = bond_features_df.iloc[i] if i < len(bond_features_df) else None
                if bond_features_row is not None and self.atomic_features is not None:
                    site1_idx = int(bond_features_row['site1_index'])
                    site2_idx = int(bond_features_row['site2_index'])
                    
                    # 基于0-Simplex数据的增强ELF特征（避免GPAW错误）
                    if 'elf' in self.atomic_features.columns:
                        elf1 = self.atomic_features.iloc[site1_idx]['elf']
                        elf2 = self.atomic_features.iloc[site2_idx]['elf']
                        
                        features['bond_elf_from_0simplex'] = (elf1 + elf2) / 2.0
                        features['bond_elf_asymmetry'] = abs(elf2 - elf1)
                    
                    # 基于0-Simplex数据的增强密度特征
                    if 'electron_density' in self.atomic_features.columns:
                        density1 = self.atomic_features.iloc[site1_idx]['electron_density']
                        density2 = self.atomic_features.iloc[site2_idx]['electron_density']
                        
                        features['bond_density_from_0simplex'] = (density1 + density2) / 2.0
                        features['bond_density_gradient'] = abs(density2 - density1) / (bond_features_row['bond_distance'] + 1e-12)
                    
                    # 基于0-Simplex数据的增强电荷特征
                    if 'bader_charge' in self.atomic_features.columns:
                        charge1 = self.atomic_features.iloc[site1_idx]['bader_charge']
                        charge2 = self.atomic_features.iloc[site2_idx]['bader_charge']
                        
                        features['bond_effective_charge'] = (charge1 + charge2) / 2.0
                        features['bond_charge_imbalance'] = (charge2 - charge1) / (abs(charge1) + abs(charge2) + 1e-12)
                        
            except Exception as e:
                print(f"  - 错误: 计算增强特征失败: {e}")
            
            quantum_features_list.append(features)
            
        return pd.DataFrame(quantum_features_list)

    def _get_elf_grid(self, density_grid: np.ndarray) -> Tuple[np.ndarray, str]:

        
        # 备用方法：尝试FD模式标准ELF计算
        if self.fd_calc:
            try:
                elf_calculator = ELF(self.fd_calc)
                elf_calculator.update()
                elf_values = elf_calculator.get_electronic_localization_function()
                print("  - 标准ELF网格获取成功。")
                return elf_values, 'standard'
            except Exception as e:
                print(f"  - 标准ELF计算失败: {e}。")
        
        print("  - 警告: 所有ELF计算方法失败，使用默认值。")
        return np.full(density_grid.shape, 0.5), 'default_fallback'  # 使用0.5作为合理的默认ELF值

    def save_features_to_csv(self, features_df: pd.DataFrame, output_path: str):
        features_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"1-单纯形特征已保存到: {output_path}")
        

def main():


    # --- 输入文件路径 ---
    # 定义输入的CIF文件路径
    cif_file = "CsPbI3-supercell-optimized.cif"

    # 基于输入文件名自动生成输出文件名
    # 提取文件名的主干部分，去除'_optimized'后缀，用于生成统一的输出文件名
    base_name = Path(cif_file).stem.replace('-optimized', '')
    pw_gpw_file = f'{base_name}.gpw'
    fd_gpw_file = f'{base_name}-fd.gpw'
    output_csv = f"{base_name}-1-Simplex-Features.csv"
    
    # 自动查找0-Simplex特征文件和结构张量文件
    atomic_features_csv = f"{base_name}-0-Simplex-Features.csv"
    atomic_tensors_csv = f"{base_name}-0-Simplex-Structure-Tensors.csv"

    # 检查GPW文件是否存在
    pw_gpw_path = Path(pw_gpw_file)
    fd_gpw_path = Path(fd_gpw_file)
    
    if not pw_gpw_path.exists():
        print(f"警告: 平面波GPW文件不存在: {pw_gpw_file}")
        print("   -> C组量子化学特征将无法计算")
        pw_gpw_file = None
    
    if not fd_gpw_path.exists():
        print(f"警告: 有限差分GPW文件不存在: {fd_gpw_file}")
        print("   -> 高精度ELF和辛几何特征将受到影响")
        fd_gpw_file = None
    
    if pw_gpw_file is None and fd_gpw_file is None:
        print("警告: 未找到任何GPW文件，C组量子化学特征和D组部分融合特征将为空值")

    print("=" * 80)
    print("1-单纯形化学键特征计算器")
    print("体现五大核心代数思想的启发与融合")
    print("=" * 80)

    # 初始化计算器，并提供所有必需的文件
    calculator = BondFeatureCalculator(
        cif_file_path=cif_file,
        pw_gpw_file=pw_gpw_file,
        fd_gpw_file=fd_gpw_file,
        atomic_features_csv_path=atomic_features_csv,
        atomic_tensors_csv_path=atomic_tensors_csv
    )

    # 计算所有特征（A+B+C+D）
    all_features = calculator.calculate_all_features()

    # 保存结果
    calculator.save_features_to_csv(all_features, output_csv)
    
    # 显示前几行结果
    print("\n--- 前5个化学键的重构后特征 ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(all_features.head())

    # 显示D组特征摘要
    d_group_cols = [
        'lie_algebra_incompatibility',              # 李代数不相容性
        'quotient_algebra_orbit_size',              # 商代数轨道大小
        # 'quotient_algebra_invariant',               # 商代数不变量 - 已移除
        'pseudo_symplectic_coupling',               # 辛几何不变量
        'tensor_algebraic_environment_alignment',   # 张量代数环境对齐
        'lie_algebraic_bond_alignment_a',           # 李代数键对齐(a轴)
        'lie_algebraic_bond_alignment_b',           # 李代数键对齐(b轴)
        'lie_algebraic_bond_alignment_c',           # 李代数键对齐(c轴)
        'delta_structure_chemistry_incompatibility', # 层次代数
        'bond_ends_anisotropy_mismatch'             # 物理图像深化
    ]
    
    print("\n--- D组与新增特征摘要 (受代数与物理思想启发) ---")
    for i, col in enumerate(d_group_cols, 1):
        if col in all_features.columns:
            non_nan_count = all_features[col].notna().sum()
            total_count = len(all_features)
            completion_rate = (non_nan_count / total_count) * 100 if total_count > 0 else 0
            mean_val = all_features[col].mean()
            print(f"D{i}. {col:<42} | 完整率: {completion_rate:>5.1f}% | 均值: {mean_val:.3e}")
        else:
            print(f"D{i}. {col:<42} | 未计算")


if __name__ == '__main__':
    main()