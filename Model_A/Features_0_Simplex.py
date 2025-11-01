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
import gudhi # 新增：拓扑数据分析库
from concurrent.futures import ProcessPoolExecutor, as_completed # 新增：用于并行计算
import mmap # 新增：用于内存映射文件
from scipy.optimize import linear_sum_assignment # 新增：用于解决高对称性原子匹配问题

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
    def __init__(self, cif_file_path: str, pw_gpw_file: str, fd_gpw_file: Optional[str] = None, output_dir_path: Optional[Path] = None, atomic_tensors_csv_path: Optional[str] = None):
        print("--- 统一原子特征计算器初始化 ---")
        # --- 路径和文件加载 ---
        self.cif_path = Path(cif_file_path).resolve()
        pw_path = Path(pw_gpw_file).resolve()
        self.output_dir_path = output_dir_path if output_dir_path else pw_path.parent
        
        if not self.cif_path.exists(): raise FileNotFoundError(f"CIF文件不存在: {self.cif_path}")
        if not pw_path.exists(): raise FileNotFoundError(f"平面波GPW文件不存在: {pw_path}")

        # --- 结构和分析工具 ---
        print(f"读取结构: {self.cif_path.name}")
        self.structure = Structure.from_file(self.cif_path)
        self.atoms = read(self.cif_path)
        print(f"结构加载成功: {self.structure.composition.reduced_formula}, {len(self.structure)}个原子")
        print(f"  Pymatgen晶格 (a,b,c,alpha,beta,gamma): {self.structure.lattice.abc}, {self.structure.lattice.angles}")
        print(f"  ASE晶胞 (a,b,c,alpha,beta,gamma): {self.atoms.cell.cellpar()}")
        print(f"  ASE元素顺序 (前5个): {self.atoms.get_chemical_symbols()[:5]}")
        
        print(f"  Pymatgen前5个原子信息:")
        for i in range(min(5, len(self.structure))):
            site = self.structure[i]
            print(f"    索引 {i}: 元素={site.specie.symbol}, 坐标={site.coords}")

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

        # --- (委员会修正) 性能优化相关初始化 ---
        # 必须在使用它们的函数被调用前定义
        self.interpolators: Dict[str, RegularGridInterpolator] = {} # 缓存网格插值器
        self.gpaw_to_pymatgen_index_reverse: Dict[int, int] = {} # Pymatgen到GPAW的逆向映射
        self.pre_extracted_dos_data: Dict[str, Any] = {} # 缓存预提取的DOS数据
        self.point_group_cache: Dict[Tuple, Any] = {} # 缓存PointGroupAnalyzer结果
        self.topo_feature_cache: Dict[Tuple, Tuple] = {} # 缓存拓扑特征
        self.persistence_max_radius: float = 6.0 # 拓扑持久同调的最大半径
        self.atomic_tensors: Dict[int, np.ndarray] = {}
        self._current_site_idx_for_feature_calc: int = -1

        # --- GPAW 计算加载 ---
        print(f"加载平面波GPW计算: {pw_path.name}")
        self.pw_calc = GPAW(str(pw_path), txt=None)
        self.fermi_level = self.pw_calc.get_fermi_level()
        self.workdir = pw_path.parent
        print(f"计算加载成功，费米能级: {self.fermi_level:.4f} eV")

        self.fd_calc = self._load_gpaw_calc(fd_gpw_file, "FD")

        # 预加载和缓存
        self._setup_grid_interpolators()

        print("使用全局化学属性范围进行标准化...")

        print("建立GPAW与Pymatgen原子索引的一致性映射...")
        self._build_gpaw_pymatgen_index_mapping()
        print("构建坐标索引映射以优化性能...")
        self._build_coordinate_index_mapping()
        
        # 立即构建逆向映射
        self._build_reverse_gpaw_pymatgen_mapping()

        # 预导出GPAW网格数据并设置插值器
        self._export_gpaw_grids_for_parallel() # 导出.npy文件
        self._setup_grid_interpolators() # 从.npy文件设置插值器

        # 预提取所有DOS数据 (如果fd_calc可用)
        if self.fd_calc:
            self.pre_extracted_dos_data = self._pre_extract_all_dos_data()

        # --- (新增) 李代数张量加载 ---
        if atomic_tensors_csv_path:
            self.load_atomic_tensors(atomic_tensors_csv_path)

    def load_atomic_tensors(self, atomic_tensors_csv_path: str):
        """
        (新增) 加载原子张量，表示局部电子密度梯度的旋转自由度。
        张量来源于晶体中原子位点的应力张量或电子云的角动量分布，
        反映局部的旋转对称性（so(3)李代数）。
        """
        print(f"  - (新增) 加载原子张量: {atomic_tensors_csv_path}")
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
            print(f"    加载了 {len(self.atomic_tensors)} 个原子的李代数张量。")
            self.verify_lie_algebra()
        except Exception as e:
            print(f"    错误: 加载原子张量失败: {e}")

    def verify_lie_algebra(self):
        """
        (新增) 验证原子张量是否满足so(3)李代数性质，并检查全局子空间的封闭性。
        1. 检查每个张量的反称性（tensor + tensor.T = 0）。
        2. 检查Jacobi恒等式（[A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0）。
        3. 验证张量集合生成的子空间是否为so(3)（维度为3）。
        """
        print("    验证原子张量的李代数性质...")
        # 反称性和Jacobi验证
        for site_idx, tensor in self.atomic_tensors.items():
            if not np.allclose(tensor + tensor.T, 0, atol=self._TOLERANCE):
                print(f"    警告: 位点 {site_idx} 的张量不是歪斜对称，可能不满足so(3)李代数性质。")
            for site_idx2, tensor2 in self.atomic_tensors.items():
                for site_idx3, tensor3 in self.atomic_tensors.items():
                    def lie_bracket(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                        return A @ B - B @ A
                    bracket12 = lie_bracket(tensor, tensor2)
                    bracket23 = lie_bracket(tensor2, tensor3)
                    bracket31 = lie_bracket(tensor3, tensor)
                    jacobi = lie_bracket(tensor, bracket23) + lie_bracket(tensor2, bracket31) + lie_bracket(tensor3, bracket12)
                    if not np.allclose(jacobi, 0, atol=self._TOLERANCE):
                        print(f"    警告: 位点 ({site_idx}, {site_idx2}, {site_idx3}) 的张量不满足Jacobi恒等式。")
        
        # 全局子空间验证
        if self.atomic_tensors:
            print("    验证张量集合的全局李代数子空间...")
            basis_tensors = list(self.atomic_tensors.values())
            lie_span = []
            for i, tensor1 in enumerate(basis_tensors):
                for tensor2 in basis_tensors[i+1:]:
                    bracket = tensor1 @ tensor2 - tensor2 @ tensor1
                    lie_span.append(bracket.flatten())
            if lie_span:
                lie_span_matrix = np.array(lie_span)
                rank = np.linalg.matrix_rank(lie_span_matrix, tol=self._TOLERANCE)
                expected_rank = 3  # so(3) 的维度
                if rank <= expected_rank:
                    print(f"    全局子空间验证通过: 张量生成子空间的秩为 {rank}，符合so(3)预期。")
                else:
                    print(f"    警告: 张量生成子空间的秩为 {rank}，超过so(3)的预期维度 {expected_rank}。")
            else:
                print("    警告: 无法生成李括号子空间，张量数量不足。")
        print("    李代数验证完成。")

    def _load_gpaw_calc(self, gpw_file: Optional[str], mode_name: str) -> Optional[GPAW]:
        """
        (委员会新增) 加载GPAW计算对象，并处理文件不存在或加载失败的情况。
        """
        if not gpw_file:
            # 静默处理，因为fd_gpw_file是可选的
            # print(f"  - ({mode_name}) GPAW文件路径未提供，跳过加载。")
            return None
        
        gpw_path = Path(gpw_file).resolve()
        if gpw_path.exists():
            print(f"  - ({mode_name}) 加载GPW计算: {gpw_path.name}")
            try:
                # txt=None to suppress GPAW stdout
                return GPAW(str(gpw_path), txt=None)
            except Exception as e:
                print(f"  - 错误: ({mode_name}) 加载GPW文件 '{gpw_path.name}' 失败: {e}")
                return None
        else:
            print(f"  - 警告: ({mode_name}) GPW文件不存在: {gpw_path}")
            return None

    def _create_local_environment_fingerprint(self, index: int, structure: Any, cutoff: float = 6.0) -> Tuple:
        """
        为指定原子创建局部化学环境指纹。

        该指纹基于邻近原子的元素类型和距离，并经过规范化处理，
        使其不受旋转、平移和原子顺序的影响。

        参数:
            index (int): 中心原子的索引。
            structure (Any): Pymatgen Structure 或 ASE Atoms 对象。
            cutoff (float): 寻找近邻的截断半径 (Å)。

        返回:
            Tuple: 一个可哈希的元组，代表原子的唯一指纹。
        """
        is_pymatgen = isinstance(structure, Structure)
        
        if is_pymatgen:
            center_site = structure[index]
            # Pymatgen的get_neighbors方法更强大，能直接处理周期性
            neighbors = structure.get_neighbors(center_site, cutoff)
        else: # is ASE Atoms
            # 对于ASE，我们需要手动设置周期性并计算距离
            # 为了简化，我们将ASE Atoms临时转换为Pymatgen Structure进行指纹生成
            from pymatgen.io.ase import AseAtomsAdaptor
            temp_pmg_structure = AseAtomsAdaptor.get_structure(structure)
            center_site = temp_pmg_structure[index]
            neighbors = temp_pmg_structure.get_neighbors(center_site, cutoff)

        fingerprint_data = []
        for neighbor_site, distance, _, _ in sorted(neighbors, key=lambda x: (x[0].specie.symbol, x[1])):
            element = neighbor_site.specie.symbol
            # 将距离四舍五入以消除浮点误差，使其更具鲁棒性
            rounded_distance = round(distance, 4)
            fingerprint_data.append((element, rounded_distance))
        
        # 返回一个元组，使其可哈希
        return tuple(fingerprint_data)

    def _build_gpaw_pymatgen_index_mapping(self):
        """
        (委员会修正) 使用指纹-坐标双重验证方法，稳健地构建GPAW到Pymatgen的原子索引映射。
        1.  使用局部化学环境指纹对原子进行分组。
        2.  对于存在多个相同指纹的高对称情况，使用匈牙利算法基于原子坐标进行最优匹配。
        3.  在所有映射建立后，进行最终的元素-坐标一致性校验。
        """
        print("  正在使用【指纹-坐标双重验证】方法构建GPAW与Pymatgen原子索引映射...")
        
        # 1. 为Pymatgen结构中的每个原子生成指纹
        pymatgen_fingerprints = {}
        for i in range(len(self.structure)):
            fp = self._create_local_environment_fingerprint(i, self.structure)
            if fp not in pymatgen_fingerprints:
                pymatgen_fingerprints[fp] = []
            pymatgen_fingerprints[fp].append(i)

        # 2. 为GPAW/ASE结构中的每个原子生成指纹
        gpaw_atoms = self.pw_calc.get_atoms()
        gpaw_fingerprints = {}
        for i in range(len(gpaw_atoms)):
            fp = self._create_local_environment_fingerprint(i, gpaw_atoms)
            if fp not in gpaw_fingerprints:
                gpaw_fingerprints[fp] = []
            gpaw_fingerprints[fp].append(i)

        # 3. 匹配指纹并使用匈牙利算法解决高对称性问题
        self.gpaw_to_pymatgen_index = {}
        unmatched_pymatgen_indices = set(range(len(self.structure)))
        unmatched_gpaw_indices = set(range(len(gpaw_atoms)))

        pymatgen_frac_coords = self.structure.frac_coords
        gpaw_frac_coords = gpaw_atoms.get_scaled_positions()

        for fp, p_indices in pymatgen_fingerprints.items():
            if fp not in gpaw_fingerprints:
                continue

            g_indices = gpaw_fingerprints[fp]
            if len(p_indices) != len(g_indices):
                print(f"  警告: 指纹 {fp} 的原子数量不匹配 (Pymatgen: {len(p_indices)}, GPAW: {len(g_indices)})")
                continue

            p_coords = pymatgen_frac_coords[p_indices]
            g_coords = gpaw_frac_coords[g_indices]

            # 计算距离矩阵，考虑周期性边界
            dist_matrix = np.zeros((len(p_indices), len(g_indices)))
            for i, p_coord in enumerate(p_coords):
                for j, g_coord in enumerate(g_coords):
                    delta = p_coord - g_coord
                    delta -= np.round(delta)  # 周期性边界条件
                    dist_matrix[i, j] = np.linalg.norm(np.dot(delta, self.structure.lattice.matrix))

            # 使用匈牙利算法解决分配问题
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            # 建立映射并检查匹配质量
            for r, c in zip(row_ind, col_ind):
                p_idx, g_idx = p_indices[r], g_indices[c]
                
                if dist_matrix[r, c] > 0.1:  # 0.1 Å 容差
                    print(f"  警告: 原子 {p_idx}(P) 和 {g_idx}(G) 指纹相同，但坐标相差 {dist_matrix[r, c]:.4f} Å")
                
                self.gpaw_to_pymatgen_index[g_idx] = p_idx
                unmatched_pymatgen_indices.discard(p_idx)
                unmatched_gpaw_indices.discard(g_idx)
        
        # 4. 报告未匹配的原子
        if unmatched_gpaw_indices or unmatched_pymatgen_indices:
            print(f"  部分匹配: {len(self.gpaw_to_pymatgen_index)}/{len(gpaw_atoms)} 个原子成功匹配")
            if unmatched_gpaw_indices: print(f"    未匹配的GPAW索引: {sorted(list(unmatched_gpaw_indices))}")
            if unmatched_pymatgen_indices: print(f"    未匹配的Pymatgen索引: {sorted(list(unmatched_pymatgen_indices))}")
        else:
            print(f"  所有 {len(self.gpaw_to_pymatgen_index)} 个原子初步匹配成功。")

        # 5. (新增) 双向一致性最终校验
        print("\n  正在执行最终双向一致性校验 (坐标 & 元素)...")
        mismatches = 0
        for g_idx, p_idx in self.gpaw_to_pymatgen_index.items():
            p_site = self.structure[p_idx]
            g_atom_sym = gpaw_atoms.get_chemical_symbols()[g_idx]

            if p_site.specie.symbol != g_atom_sym:
                print(f"  严重错误: 映射 {g_idx} -> {p_idx} 元素不匹配! ({g_atom_sym} vs {p_site.specie.symbol})")
                mismatches += 1
                continue
            
            p_coord = p_site.frac_coords
            g_coord = gpaw_frac_coords[g_idx]
            delta = p_coord - g_coord
            delta -= np.round(delta)
            dist = np.linalg.norm(np.dot(delta, self.structure.lattice.matrix))
            
            if dist > 0.1:
                print(f"  严重错误: 映射 {g_idx} -> {p_idx} ({g_atom_sym}) 坐标偏差过大: {dist:.4f} Å")
                mismatches += 1

        if mismatches == 0:
            print("  ✅ 所有映射均通过最终一致性校验。")
        else:
            print(f"  ❌ 发现 {mismatches} 个不一致的映射。请仔细检查输入文件和结构对称性。")

        # 6. 打印映射示例
        print("\n  映射示例（GPAW索引 -> Pymatgen索引，元素符号）:")
        sample_items = list(self.gpaw_to_pymatgen_index.items())
        num_samples = min(5, len(sample_items))
        gpaw_elements = gpaw_atoms.get_chemical_symbols()
        pymatgen_elements = [s.specie.symbol for s in self.structure]

        for g_idx, p_idx in sorted(sample_items)[:num_samples]:
            print(f"    {g_idx} -> {p_idx} ({gpaw_elements[g_idx]} -> {pymatgen_elements[p_idx]})")
        
        if len(sample_items) > num_samples:
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

    def _build_reverse_gpaw_pymatgen_mapping(self):
        """Helper to create reverse mapping from pymatgen_idx to gpaw_idx."""
        print("  - 构建Pymatgen到GPAW的逆向索引映射...")
        for g_idx, p_idx in self.gpaw_to_pymatgen_index.items():
            self.gpaw_to_pymatgen_index_reverse[p_idx] = g_idx
        print(f"    逆向映射构建完成，包含 {len(self.gpaw_to_pymatgen_index_reverse)} 个条目。")

    def _get_pymatgen_index_from_gpaw(self, gpaw_index: int) -> int:
        if not hasattr(self, 'gpaw_to_pymatgen_index'):
            # 如果映射未建立，假设索引相同（向后兼容）
            return gpaw_index
        
        return self.gpaw_to_pymatgen_index.get(gpaw_index, gpaw_index)


    def _export_gpaw_grids_for_parallel(self):
        """
        (新增) 预导出GPAW网格数据到内存映射的.npy文件，以供并行处理和插值器缓存。
        """
        print("  - 预导出GPAW网格数据...")
        grid_export_tasks = {
            "pw_density": (self.pw_calc, self.pw_calc.get_all_electron_density),
            "pw_potential": (self.pw_calc, self.pw_calc.get_electrostatic_potential),
        }
        if self.fd_calc:
            elf_calc = ELF(self.fd_calc)
            elf_calc.update()
            grid_export_tasks["fd_elf"] = (self.fd_calc, elf_calc.get_electronic_localization_function)
            # 如果需要梯度，也在这里导出
            # grid_export_tasks["fd_grad_x"] = (self.fd_calc, lambda: self.fd_calc.get_density_gradient()[0])
            # grid_export_tasks["fd_grad_y"] = (self.fd_calc, lambda: self.fd_calc.get_density_gradient()[1])
            # grid_export_tasks["fd_grad_z"] = (self.fd_calc, lambda: self.fd_calc.get_density_gradient()[2])

        for name, (calc_obj, get_grid_func) in grid_export_tasks.items():
            if calc_obj is None: continue
            try:
                grid_data = get_grid_func()
                if grid_data is not None and grid_data.size > 0:
                    npy_path = self.output_dir_path / f"{name}.npy"
                    np.save(npy_path, grid_data)
                    print(f"    导出 {name} 网格数据到 {npy_path.name} ({grid_data.shape})")
                else:
                    print(f"    警告: {name} 网格数据为空或无效，跳过导出。")
            except Exception as e:
                print(f"    错误: 导出 {name} 网格数据失败: {e}")

    def _load_grid_from_cache(self, grid_name: str) -> Optional[np.ndarray]:
        """
        (新增) 从内存映射的.npy文件加载网格数据。
        """
        npy_path = self.output_dir_path / f"{grid_name}.npy"
        if npy_path.exists():
            try:
                # 使用内存映射加载，避免将整个大文件读入内存
                return np.load(npy_path, mmap_mode='r')
            except Exception as e:
                print(f"  - 错误: 从缓存加载 {grid_name} 失败: {e}")
                return None
        return None

    def _setup_grid_interpolators(self):
        """
        (新增) 设置RegularGridInterpolator对象，从缓存的.npy文件加载数据。
        """
        print("  - 设置网格插值器...")
        grid_names = ["pw_density", "pw_potential"]
        if self.fd_calc:
            grid_names.extend(["fd_elf"]) # 如果需要梯度，也在这里添加
            # grid_names.extend(["fd_grad_x", "fd_grad_y", "fd_grad_z"])

        for name in grid_names:
            try:
                grid_data = self._load_grid_from_cache(name)
                if grid_data is None:
                    self.interpolators[name] = None
                    continue

                grid_shape = grid_data.shape
                # 确保n_points大于1，避免除零错误，同时对单点维度进行特殊处理
                axes = [np.linspace(0, 1, n_points, endpoint=False) if n_points > 1 else np.array([0.0]) for n_points in grid_shape]

                self.interpolators[name] = RegularGridInterpolator(
                    axes, grid_data,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                print(f"    插值器 '{name}' 设置成功。")
            except Exception as e:
                print(f"    警告: 插值器 '{name}' 设置失败: {e}")
                self.interpolators[name] = None
        print("  - 网格插值器设置完成。")

    # ==========================================================================
    # 主流程方法
    # ==========================================================================

    def calculate_unified_features(self) -> pd.DataFrame:
        print("\n--- 开始计算统一原子特征 (40维) ---") # 更新维度

        # 步骤 1: 计算经典特征 (A+C组)
        classic_df = self._calculate_classic_features()
        
        # 步骤 2: 计算量子化学特征 (B组)
        quantum_df = self._calculate_quantum_features()
        
        # 步骤 3: 提取A, B, C组
        group_A, group_C = self._extract_classic_groups(classic_df)
        group_B = quantum_df # B组就是完整的量子特征
        
        # 步骤 4: 计算融合特征 (D组)
        group_D = self._calculate_fused_features(classic_df, group_B) # 传递classic_df以获取结构张量和李代数不变量
        
        # 步骤 5: 整合所有特征
        unified_df = pd.concat([group_A, group_B, group_C, group_D], axis=1)
        
        # 额外步骤: 保存结构张量到独立文件
        base_name = self.cif_path.stem.replace('-optimized', '')
        tensor_output_path = self.output_dir_path / f"{base_name}-0-Simplex-Structure-Tensors.csv"
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
            'avg_site_valence', 'bond_valence_sum',
            'bond_length_distortion', 'vectorial_asymmetry_norm_sq', 'mean_squared_neighbor_distance',
            'local_environment_anisotropy', 'symmetry_breaking_quotient',
            'site_symmetry_order', 'H0_persistence_max_death', 'H0_persistence_death_std', # 【委员会重构】替换为高效的H0拓扑特征
            'local_env_entropy', 'local_variational_free_energy', # 新增贝叶斯力学特征
            'lie_algebra_norm', 'lie_algebra_principal_angle', # 新增李代数不变量
            'tensor_trace', 'tensor_determinant', 'tensor_eigenvalue_1', 'tensor_eigenvalue_2', 'tensor_eigenvalue_3', 'tensor_variance', # 【委员会新增】张量派生特征
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
            # 使用缓存的插值器
            potentials = self._interpolate_at_atomic_sites_corrected('pw_potential')
        except Exception as e: 
            print(f"  - 错误: 计算电势失败: {e}")
            potentials = [np.nan] * len(self.structure)

        try:
            # 使用缓存的插值器
            densities = self._interpolate_at_atomic_sites_corrected('pw_density')
        except Exception as e: 
            print(f"  - 错误: 计算电子密度失败: {e}")
            densities = [np.nan] * len(self.structure)

        # ELF计算也使用缓存的插值器
        elf_values = self._calculate_elf_at_sites_corrected()
        
        try:
            magnetic_moments = self._get_magnetic_moments_corrected()
        except Exception as e: 
            print(f"  - 错误: 计算磁矩失败: {e}")
            magnetic_moments = [np.nan] * len(self.structure)

        # --- 计算稳健的轨道电子数特征 (使用并行方法) ---
        print("  - 计算稳健的DOS特征 (LDOS, s/p/d electron counts) (并行)...")
        dos_features_list = self._calculate_dos_features_parallel() # 调用并行方法
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

    def _calculate_fused_features(self, classic_df: pd.DataFrame, group_B: pd.DataFrame) -> pd.DataFrame:
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
            
            # 获取李代数不变量 (从classic_df中获取，因为它们在_calculate_single_site_classic_features中计算)
            lie_algebra_norm = classic_df.iloc[i]['lie_algebra_norm']
            lie_algebra_principal_angle = classic_df.iloc[i]['lie_algebra_principal_angle']

            # 计算四个深度融合特征
            features = {}
            
            # D1: 结构-化学不相容性 (李代数交换子范数)
            features['structure_chemistry_incompatibility'] = self._calculate_structure_chemistry_incompatibility(
                T_struct, v_chem)
            
            # D2: 电荷加权的局域尺寸
            features['charge_weighted_local_size'] = self._calculate_charge_weighted_local_size_new(
                i, bader_charges)
            
            # D3: ELF加权的局域不对称性
            features['elf_weighted_local_anisotropy'] = self._calculate_elf_weighted_local_anisotropy_new(
                i, elf)
            
            # D4, D5: 李代数几何不变量 (直接从classic_df中获取)
            features['lie_algebra_norm'] = lie_algebra_norm
            features['lie_algebra_principal_angle'] = lie_algebra_principal_angle
            
            fusion_features.append(features)
            
        return pd.DataFrame(fusion_features)

    # ==========================================================================
    # 经典特征辅助方法 (重新设计的C组特征)
    # ==========================================================================

    def _calculate_single_site_classic_features(self, site_idx: int, site: "Site") -> Tuple[List[float], np.ndarray]:
        # (新增) 保存当前处理的位点索引，以供D组特征使用
        self._current_site_idx_for_feature_calc = site_idx
        
        elem = element(site.specie.symbol)
        
        # --- A. 基础物理化学特征 (10维) ---
        cn = self._get_coordination_number(site_idx)
        valence = self.valences[site_idx]
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
            bvs,
        ]
        
        # --- C. 重新设计的几何、对称性与拓扑特征 (10维) ---
        local_asymmetry_vector = self._calculate_local_asymmetry_vector(site_idx)
        T_struct = self._get_structure_tensor(site_idx)
        
        # (委员会新增) 计算张量派生特征
        tensor_derived_features = self._calculate_tensor_derived_features(T_struct)

        # 对称性特征 (使用缓存)
        site_symm_order = self._get_site_symmetry_order(site_idx)
        symmetry_breaking_quotient_val = self._calculate_symmetry_breaking_quotient(site_idx) # 此方法已包含缓存

        # 拓扑特征 (H0, 使用缓存)
        H0_max_death, H0_death_std = self._calculate_coord_shell_persistence_features(site_idx)

        # 贝叶斯力学特征
        local_env_entropy = self._calculate_local_environment_entropy(site_idx)
        local_variational_free_energy = self._calculate_local_variational_free_energy(site_idx, symmetry_breaking_quotient_val)

        # 李代数几何不变量 (D组的一部分，但在此处计算)
        lie_invariants = self._calculate_lie_algebra_geometric_invariants(site_idx)

        geometric_features = [
            self._calculate_bond_length_distortion(site_idx), # 补齐此方法
            self._calculate_vectorial_asymmetry_norm_sq(local_asymmetry_vector),
            self._calculate_mean_squared_neighbor_distance(T_struct, site_idx),
            self._calculate_local_environment_anisotropy_from_tensor(T_struct),
            symmetry_breaking_quotient_val,
            site_symm_order,
            H0_max_death, # 【委员会重构】
            H0_death_std, # 【委员会重构】
            local_env_entropy, # 新增
            local_variational_free_energy # 新增
        ]
        
        # 将李代数不变量也添加到返回的特征列表中，以便在classic_df中存储
        # 它们在D组中被引用，但为了避免重复计算，在此处计算并传递
        lie_features = [
            lie_invariants['lie_algebra_norm'],
            lie_invariants['lie_algebra_principal_angle']
        ]

        # (委员会新增)
        tensor_features = [
            tensor_derived_features['tensor_trace'],
            tensor_derived_features['tensor_determinant'],
            tensor_derived_features['tensor_eigenvalue_1'],
            tensor_derived_features['tensor_eigenvalue_2'],
            tensor_derived_features['tensor_eigenvalue_3'],
            tensor_derived_features['tensor_variance']
        ]

        return basic_features + geometric_features + lie_features + tensor_features, T_struct

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

    def _calculate_tensor_derived_features(self, T_struct: np.ndarray) -> Dict[str, float]:
        """
        (委员会新增) 计算结构张量的派生特征 (迹, 行列式, 特征值, 方差)。
        """
        features = {
            'tensor_trace': np.nan,
            'tensor_determinant': np.nan,
            'tensor_eigenvalue_1': np.nan,
            'tensor_eigenvalue_2': np.nan,
            'tensor_eigenvalue_3': np.nan,
            'tensor_variance': np.nan,
        }
        try:
            features['tensor_trace'] = float(np.trace(T_struct))
            features['tensor_determinant'] = float(np.linalg.det(T_struct))
            
            eigenvalues = np.sort(np.linalg.eigvalsh(T_struct)) # 使用eigvalsh以确保数值稳定性
            features['tensor_eigenvalue_1'] = float(eigenvalues[2]) # 最大
            features['tensor_eigenvalue_2'] = float(eigenvalues[1])
            features['tensor_eigenvalue_3'] = float(eigenvalues[0]) # 最小

            if len(eigenvalues) > 0:
                features['tensor_variance'] = float(np.var(eigenvalues))
        except Exception as e:
            print(f"  - 错误: 计算张量派生特征失败: {e}")
        return features

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
            return None

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
            order_site = self._get_site_symmetry_order(site_idx)
            
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            if not nn_info:
                return 1.0
            
            center_site = self.structure[site_idx]
            coords_list = [[0.0, 0.0, 0.0]]
            species_list = [center_site.specie.symbol]
            
            for nn in nn_info:
                relative_coords = nn['site'].coords - center_site.coords
                coords_list.append(relative_coords.tolist())
                species_list.append(nn['site'].specie.symbol)
            
            # 创建局部簇的指纹作为缓存键
            local_cluster_fingerprint = tuple(sorted(zip(species_list, [tuple(c) for c in coords_list])))

            order_local = 0
            if local_cluster_fingerprint in self.point_group_cache:
                order_local = len(self.point_group_cache[local_cluster_fingerprint])
            else:
                from pymatgen.core.structure import Molecule
                mol = Molecule(species_list, coords_list)
                pga = PointGroupAnalyzer(mol)
                point_group = pga.get_pointgroup()
                order_local = len(point_group)
                self.point_group_cache[local_cluster_fingerprint] = point_group # 缓存结果
            
            if order_local > 0:
                quotient = float(order_site) / float(order_local)
                return quotient
            else:
                return float(order_site)
                
        except Exception as e:
            print(f"  - 错误: 计算对称性破缺商失败: {e}")
            # 备用方法 (与您原代码相同)
            try:
                nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
                if not nn_info: return 1.0
                distances = [self.structure[site_idx].distance(nn['site']) for nn in nn_info]
                elements = [nn['site'].specie.symbol for nn in nn_info]
                if len(distances) > 1:
                    mean_dist = np.mean(distances)
                    distance_cv = np.std(distances) / mean_dist if mean_dist > 0 else 0.0
                else: distance_cv = 0.0
                unique_elements = len(set(elements))
                total_neighbors = len(elements)
                element_diversity = unique_elements / total_neighbors if total_neighbors > 0 else 1.0
                symmetry_breaking = 1.0 + distance_cv + element_diversity
                return float(symmetry_breaking)
            except Exception as e_fallback:
                print(f"  - 错误: 备用对称性破缺商计算失败: {e_fallback}")
                return None

    def _calculate_coord_shell_persistence_features(self, site_idx: int, max_radius: float = 6.0) -> Tuple[float, float]:
        """
        (委员会重构) 计算配位壳层的H0持久同调特征，以替代计算昂贵且信息稀疏的H1特征。
        - H0_persistence_max_death: H0同调群中有限生命周期点的最大死亡时间。这对应于连接所有近邻原子的最小生成树中的最长边，是局部环境尺寸的鲁棒度量。
        - H0_persistence_death_std: H0同调群中有限生命周期点的死亡时间的标准差。这反映了近邻距离分布的均匀性，是局部几何畸变的有效度量。
        """
        fingerprint = self._create_local_environment_fingerprint(site_idx, self.structure, cutoff=max_radius)
        if fingerprint in self.topo_feature_cache:
            return self.topo_feature_cache[fingerprint]

        nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        # 至少需要两个邻居才能有连接距离
        if len(nn_info) < 2:
            self.topo_feature_cache[fingerprint] = (0.0, 0.0)
            return 0.0, 0.0

        center_site = self.structure[site_idx]
        points = np.array([nn['site'].coords - center_site.coords for nn in nn_info])

        try:
            # H0特征只需要构建到1维（边）的Rips复形，计算速度非常快
            rips_complex = gudhi.RipsComplex(points=points, max_edge_length=max_radius)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

            simplex_tree.compute_persistence()
            
            # 提取0维的持久性区间
            h0_pairs = simplex_tree.persistence_intervals_in_dimension(0)
            
            # 有限生命周期的死亡时间对应于最小生成树的边长
            h0_deaths = [death for birth, death in h0_pairs if not np.isinf(death)]

            if not h0_deaths:
                result = (0.0, 0.0)
            else:
                max_death = float(np.max(h0_deaths))
                std_death = float(np.std(h0_deaths)) if len(h0_deaths) > 1 else 0.0
                result = (max_death, std_death)
            
            self.topo_feature_cache[fingerprint] = result
            return result

        except Exception as e:
            print(f"  - 错误: 计算位点 {site_idx} 的H0持久同调特征失败: {e}")
            self.topo_feature_cache[fingerprint] = (np.nan, np.nan)
            return np.nan, np.nan

    def _calculate_local_environment_entropy(self, site_idx: int) -> float:
        """
        (新增) 计算局部环境的香农熵，衡量局部环境的无序度或多样性。
        基于近邻的元素类型和距离分布。
        """
        try:
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            if not nn_info:
                return 0.0

            # 考虑元素类型和距离的组合
            env_descriptors = []
            for nn in nn_info:
                element_symbol = nn['site'].specie.symbol
                distance = round(self.structure[site_idx].distance(nn['site']), 2) # 四舍五入距离
                env_descriptors.append(f"{element_symbol}_{distance}")

            # 计算每个描述符的频率
            from collections import Counter
            counts = Counter(env_descriptors)
            total = len(env_descriptors)
            
            # 计算香农熵
            entropy = 0.0
            for count in counts.values():
                probability = count / total
                entropy -= probability * np.log2(probability)
            
            return float(entropy)
        except Exception as e:
            print(f"  - 错误: 计算局部环境熵失败: {e}")
            return np.nan

    def _calculate_local_variational_free_energy(self, site_idx: int, precomputed_symmetry_breaking: float) -> float:
        """
        (委员会重构) 计算局部变分自由能 (VFE)，作为局部结构稳定性和灵活性的综合度量。
        此实现严格遵循贝叶斯力学和热力学原理 F = U - TS，摒弃了所有硬编码权重。

        - 能量项 (U): 衡量局部几何和化学成键的畸变程度。
        - 熵项 (S): 衡量局部环境的化学和几何无序度。
        - 温度项 (T): 以对称性破缺程度作为局部环境"激发"程度的代理。
        """
        try:
            # --- 1. 能量项 U (Internal Energy Proxy) ---
            # 越高表示越不稳定或能量越高
            
            # 几何畸变
            bond_length_distortion = self._calculate_bond_length_distortion(site_idx)
            T_struct = self._get_structure_tensor(site_idx)
            local_environment_anisotropy = self._calculate_local_environment_anisotropy_from_tensor(T_struct)
            
            # 化学成键畸变
            bond_valence_sum = self._calculate_bond_valence_sum(site_idx)
            ideal_valence = self.valences[site_idx] if site_idx < len(self.valences) else None
            bvs_deviation = abs(bond_valence_sum - ideal_valence)

            # 直接相加，避免任意加权
            U_proxy = bond_length_distortion + local_environment_anisotropy + bvs_deviation

            # --- 2. 熵项 S (Entropy Proxy) ---
            # 越高表示越无序
            local_env_entropy = self._calculate_local_environment_entropy(site_idx)
            S_proxy = local_env_entropy

            # --- 3. 温度项 T (Temperature Proxy) ---
            # 对称性越低 (破缺商越小)，"温度"越高
            # precomputed_symmetry_breaking 接近1表示对称性高, 接近0表示对称性低
            T_proxy = 1.0 - precomputed_symmetry_breaking

            # --- 4. 组合成变分自由能 F = U - TS ---
            # 自由能越低，表示该局部结构越稳定、越"有序"
            free_energy = U_proxy - T_proxy * S_proxy
            
            return float(free_energy)

        except Exception as e:
            print(f"  - 错误: 计算局部变分自由能失败: {e}")
            return np.nan

    def _calculate_lie_algebra_geometric_invariants(self, site_idx: int) -> Dict[str, float]:
        """
        (新增) 计算李代数几何不变量：范数和主角度。
        这些不变量量化了局部环境的旋转对称性和不对称性。
        """
        invariants = {
            'lie_algebra_norm': np.nan,
            'lie_algebra_principal_angle': np.nan
        }
        
        try:
            # 优先使用预加载的原子张量
            if site_idx in self.atomic_tensors:
                M = self.atomic_tensors[site_idx]
            else:
                # 如果没有预加载，从结构张量构造一个代理李代数元素
                T_struct = self._get_structure_tensor(site_idx)
                eigenvalues, eigenvectors = np.linalg.eigh(T_struct)
                v_struct = eigenvectors[:, np.argmax(eigenvalues)] # 主方向
                M = self._construct_so3_generator(v_struct) # 构造so(3)生成元

            # 1. 李代数范数 (Frobenius范数)
            invariants['lie_algebra_norm'] = float(np.linalg.norm(M, 'fro'))

            # 2. 李代数主角度 (Principal Angle)
            # 对于so(3)的元素，其特征值是 0, +i*theta, -i*theta
            # 矩阵指数 exp(M) = I + M + M^2/2! + ...
            # 对于so(3)元素，exp(theta*M) 对应于旋转矩阵
            # 我们可以通过矩阵的迹来计算角度
            # 对于一个so(3)矩阵 M，其特征值是 0, i*omega, -i*omega
            # 迹(M^2) = -2 * omega^2
            # 因此 omega = sqrt(-迹(M^2)/2)
            # 主角度通常指这个 omega
            
            # 确保M是反称矩阵
            if not np.allclose(M + M.T, 0, atol=self._TOLERANCE):
                print(f"  - 警告: 位点 {site_idx} 的李代数元素不是反称矩阵，无法计算主角度。")
                return invariants

            M_squared = M @ M
            trace_M_squared = np.trace(M_squared)
            
            if trace_M_squared > -self._TOLERANCE: # 理论上应为负或零
                # 如果接近0或正，说明M接近零矩阵，角度为0
                invariants['lie_algebra_principal_angle'] = 0.0
            else:
                # omega^2 = -trace(M^2) / 2
                omega_squared = -trace_M_squared / 2.0
                omega = np.sqrt(omega_squared)
                invariants['lie_algebra_principal_angle'] = float(omega) # 角度以弧度表示

        except Exception as e:
            print(f"  - 错误: 计算位点 {site_idx} 的李代数几何不变量失败: {e}")
            pass
        
        return invariants

    def _construct_so3_generator(self, v: np.ndarray) -> np.ndarray:
        """
        (修改) 构造so(3)李代数的生成元矩阵。
        如果输入是7维化学矢量，则将其投影到3D空间。
        """
        if len(v) == 7:
            # 将7维化学矢量投影到3D空间，例如取前3个分量
            # 这是一个启发式投影，可以根据具体物理意义调整
            v_proj = v[:3]
        elif len(v) == 3:
            v_proj = v
        else:
            # 对于其他维度，返回零矩阵或进行更复杂的投影
            return np.zeros((3, 3))

        # 归一化矢量，避免过大或过小的值影响矩阵范数
        norm_v = np.linalg.norm(v_proj)
        if norm_v < self._TOLERANCE:
            return np.zeros((3, 3))
        v_norm = v_proj / norm_v

        # 构造反对称矩阵 (so(3)的生成元)
        # M = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
        x, y, z = v_norm[0], v_norm[1], v_norm[2]
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

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

    def _interpolate_at_atomic_sites_corrected(self, grid_name: str) -> List[float]:
        """
        (修改) 使用缓存的RegularGridInterpolator对象进行插值。
        参数:
            grid_name (str): 要使用的插值器的名称 (例如 "pw_density", "fd_elf")。
        """
        interpolator = self.interpolators.get(grid_name)
        if interpolator is None:
            print(f"  - 错误: 插值器 '{grid_name}' 未初始化或不可用。")
            return [np.nan] * len(self.structure)

        try:
            # 获取用于插值的原子位置 (GPAW顺序)
            calc_source = self.pw_calc if grid_name.startswith("pw_") else self.fd_calc
            if calc_source is None:
                print(f"  - 错误: 无法确定 {grid_name} 的计算源。")
                return [np.nan] * len(self.structure)

            cell = calc_source.get_atoms().get_cell()
            cell_inv = np.linalg.inv(cell)
            atomic_positions = calc_source.get_atoms().get_positions()
            frac_coords = np.dot(atomic_positions, cell_inv)
            wrapped_frac_coords = frac_coords % 1.0

            grid_shape = interpolator.values.shape # 从插值器获取网格形状
            scaled_frac_coords = np.copy(wrapped_frac_coords)
            for d in range(len(grid_shape)):
                n_points = grid_shape[d]
                if n_points > 1:
                    max_coord_value = (n_points - 1) / n_points
                    scaled_frac_coords[:, d] = np.clip(scaled_frac_coords[:, d], 0.0, max_coord_value - self._TOLERANCE)
                else:
                    scaled_frac_coords[:, d] = 0.0

            gpaw_results = interpolator(scaled_frac_coords)
            
            # 重新排列结果以匹配Pymatgen结构顺序
            corrected_results = [np.nan] * len(self.structure)
            for gpaw_idx, result in enumerate(gpaw_results):
                pymatgen_idx = self._get_pymatgen_index_from_gpaw(gpaw_idx)
                if pymatgen_idx is not None and pymatgen_idx < len(corrected_results):
                    corrected_results[pymatgen_idx] = result
            return corrected_results

        except Exception as e:
            print(f"  - 错误: 执行插值失败 for {grid_name}: {e}")
            return [np.nan] * len(self.structure)


    def _calculate_elf_at_sites_corrected(self) -> List[float]:
        if not self.fd_calc:
            print("  - 跳过ELF计算: 未提供FD模式GPW文件。")
            return [np.nan] * len(self.structure)
        try:
            print("  - 计算电子局域化函数 (ELF)...")
            # 直接使用缓存的插值器
            return self._interpolate_at_atomic_sites_corrected('fd_elf')
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

    def _pre_extract_all_dos_data(self) -> Dict[str, Any]:
        """
        (新增) 在主进程中预提取所有原子的DOS数据，避免子进程重复加载GPAW对象。
        """
        print("  - 预提取所有原子的DOS数据...")
        all_dos_data = {
            'energies': None,
            'ldos_data': {},
            'pdos_s_data': {}, 'pdos_p_data': {}, 'pdos_d_data': {}
        }
        if not self.fd_calc:
            print("    FD模式GPAW计算器不可用，跳过DOS数据预提取。")
            return all_dos_data

        # 获取公共能量网格 (假设所有原子使用相同的)
        try:
            # 使用第一个原子获取能量网格
            energies, _ = self.fd_calc.get_wigner_seitz_ldos(a=0, spin=0, npts=401, width=0.1)
            all_dos_data['energies'] = energies
        except Exception as e:
            print(f"  - 警告: 无法获取能量网格: {e}")
            return all_dos_data

        # 遍历所有GPAW索引，预提取LDOS和PDOS数据
        for gpaw_idx in range(len(self.pw_calc.get_atoms())):
            try:
                _, ldos = self.fd_calc.get_wigner_seitz_ldos(a=gpaw_idx, spin=0, npts=401, width=0.1)
                all_dos_data['ldos_data'][gpaw_idx] = ldos
            except Exception:
                all_dos_data['ldos_data'][gpaw_idx] = np.full(len(energies), np.nan)

            for l_name in ['s', 'p', 'd']:
                try:
                    _, pdos = self.fd_calc.get_orbital_ldos(a=gpaw_idx, angular=l_name, spin=0, npts=401, width=0.1)
                    all_dos_data[f'pdos_{l_name}_data'][gpaw_idx] = pdos
                except Exception:
                    all_dos_data[f'pdos_{l_name}_data'][gpaw_idx] = np.full(len(energies), np.nan)
        print("  - DOS数据预提取完成。")
        return all_dos_data

    def _calculate_dos_features_parallel(self, max_workers: Optional[int] = None) -> List[Dict[str, float]]:
        """
        (新增) 使用并行处理计算所有原子的DOS特征。
        """
        print("  - 并行计算DOS特征...")
        if os.environ.get('SKIP_DOS_CALCULATION', 'false').lower() == 'true':
            print("    环境变量SKIP_DOS_CALCULATION为true，跳过DOS计算。")
            return [{
                'local_dos_fermi': np.nan, 's_electron_count': np.nan,
                'p_electron_count': np.nan, 'd_electron_count': np.nan
            }] * len(self.structure)

        if not self.fd_calc or not self.pre_extracted_dos_data or self.pre_extracted_dos_data['energies'] is None:
            print("  - 警告: FD模式GPAW计算器或预提取的DOS数据不可用，DOS特征将为NaN。")
            return [{
                'local_dos_fermi': np.nan, 's_electron_count': np.nan,
                'p_electron_count': np.nan, 'd_electron_count': np.nan
            }] * len(self.structure)

        energies = self.pre_extracted_dos_data['energies']
        tasks = []
        for pymatgen_idx in range(len(self.structure)):
            gpaw_index = self.gpaw_to_pymatgen_index_reverse.get(pymatgen_idx, -1) # 使用逆向映射
            if gpaw_index == -1:
                print(f"  - 警告: 无法找到Pymatgen索引 {pymatgen_idx} 对应的GPAW索引，跳过DOS计算。")
                tasks.append((
                    pymatgen_idx, -1, energies,
                    None, None, None, None,
                    self.fermi_level, False # 传递False以跳过d轨道计算
                ))
                continue

            tasks.append((
                pymatgen_idx, gpaw_index, energies,
                self.pre_extracted_dos_data['ldos_data'].get(gpaw_index),
                self.pre_extracted_dos_data['pdos_s_data'].get(gpaw_index),
                self.pre_extracted_dos_data['pdos_p_data'].get(gpaw_index),
                self.pre_extracted_dos_data['pdos_d_data'].get(gpaw_index),
                self.fermi_level,
                self._is_d_orbital_relevant(pymatgen_idx) # 传递d轨道相关性判断结果
            ))

        dos_features_list = [{}] * len(tasks)
        # 默认使用CPU核心数，或指定max_workers
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for i, task_args in enumerate(tasks):
                future = executor.submit(self._calculate_single_dos_feature_worker_optimized, *task_args)
                future_to_index[future] = i
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    dos_features_list[idx] = future.result()
                except Exception as e:
                    print(f"  - 错误: 并行DOS计算任务 {idx} 失败: {e}")
                    dos_features_list[idx] = {
                        'local_dos_fermi': np.nan, 's_electron_count': np.nan,
                        'p_electron_count': np.nan, 'd_electron_count': np.nan
                    }
        print("  - 并行DOS特征计算完成。")
        return dos_features_list

    @staticmethod
    def _calculate_single_dos_feature_worker_optimized(
        pymatgen_idx: int, gpaw_idx: int, energies: np.ndarray, ldos_data: Optional[np.ndarray],
        pdos_s_data: Optional[np.ndarray], pdos_p_data: Optional[np.ndarray], pdos_d_data: Optional[np.ndarray],
        fermi_level: float, is_d_orbital_relevant: bool
    ) -> Dict[str, float]:
        """
        (新增) 静态工作函数：接收预提取的DOS数据，避免GPAW对象重新加载。
        """
        features = {
            'local_dos_fermi': np.nan,
            's_electron_count': np.nan,
            'p_electron_count': np.nan,
            'd_electron_count': np.nan
        }

        if gpaw_idx == -1: # 如果GPAW索引无效，直接返回NaN特征
            return features

        # LDOS
        if ldos_data is not None and not np.all(np.isnan(ldos_data)):
            features['local_dos_fermi'] = float(np.interp(fermi_level, energies, ldos_data))

        # PDOS
        for l_name, pdos_array in zip(['s', 'p', 'd'], [pdos_s_data, pdos_p_data, pdos_d_data]):
            if l_name == 'd' and not is_d_orbital_relevant:
                continue # 根据预判断结果跳过不相关的d轨道计算

            if pdos_array is not None and not np.all(np.isnan(pdos_array)):
                occupied_mask = energies <= fermi_level
                if len(energies[occupied_mask]) > 0:
                    electron_count = np.trapz(pdos_array[occupied_mask], energies[occupied_mask])
                    # 确保电子数非负
                    features[f'{l_name}_electron_count'] = float(max(0.0, electron_count))
                else:
                    features[f'{l_name}_electron_count'] = 0.0
        return features

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
            'atomic_number', 'electronegativity', 'ionization_energy', 'electron_affinity',
            'valence_electrons', 'ionic_radius', 'covalent_radius', 'coordination_number',
            'avg_site_valence', 'bond_valence_sum'
        ]
        group_C_cols = [
            'bond_length_distortion', 'vectorial_asymmetry_norm_sq', 'mean_squared_neighbor_distance',
            'local_environment_anisotropy', 'symmetry_breaking_quotient', 'site_symmetry_order',
            'H0_persistence_max_death', 'H0_persistence_death_std',
            'local_env_entropy', 'local_variational_free_energy',
            # 【委员会最终修复】将张量派生特征添加到C组
            'tensor_trace', 'tensor_determinant', 'tensor_eigenvalue_1', 'tensor_eigenvalue_2', 
            'tensor_eigenvalue_3', 'tensor_variance'
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
            
            # B组：量子化学特征 (9维)
            'bader_charge', 'electrostatic_potential', 'electron_density', 'elf',
            'local_magnetic_moment', 'local_dos_fermi', 's_electron_count', 'p_electron_count',
            'd_electron_count',
            
            # C组：几何、对称性、拓扑与张量特征 (16维)
            'bond_length_distortion', 'vectorial_asymmetry_norm_sq', 'mean_squared_neighbor_distance',
            'local_environment_anisotropy', 'symmetry_breaking_quotient', 'site_symmetry_order',
            'H0_persistence_max_death', 'H0_persistence_death_std',
            'local_env_entropy', 'local_variational_free_energy',
            'tensor_trace', 'tensor_determinant', 'tensor_eigenvalue_1', 'tensor_eigenvalue_2', 'tensor_eigenvalue_3', 'tensor_variance',
            
            # D组：融合特征 (5维)
            'structure_chemistry_incompatibility', 'charge_weighted_local_size', 'elf_weighted_local_anisotropy',
            'lie_algebra_norm', 'lie_algebra_principal_angle' # 李代数不变量
        ]
        
        # 确保所有期望的列都存在，如果缺失则填充NaN
        for col in expected_columns:
            if col not in features_df.columns:
                features_df[col] = np.nan
                print(f"  - 警告: 缺失特征列 '{col}'，已填充NaN。")

        # 重新排序列以匹配期望的顺序
        features_df = features_df[expected_columns]

        # 保存特征
        features_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n--- 40维统一特征已成功保存到: {output_path} ---")


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
                return abs(self.valences[site_idx]) if site_idx < len(self.valences) else None
            
            # 创建近邻site列表
            neighbor_sites = [nn['site'] for nn in nn_info]
            
            # 使用pymatgen的calculate_bv_sum函数
            bvs_value = calculate_bv_sum(center_site, neighbor_sites)
            return float(bvs_value)
                
        except Exception as e:
            print(f"  - 错误: 计算键价和失败: {e}")
            # 如果计算失败，使用氧化态的绝对值作为估计
            return abs(self.valences[site_idx]) if site_idx < len(self.valences) else None

    # ==========================================================================
    # D组深度融合特征的具体实现
    # ==========================================================================

    def _calculate_structure_chemistry_incompatibility(self, T_struct: np.ndarray, v_chem: np.ndarray) -> float:
        try:
            # (新增) 如果提供了外部原子张量，则优先使用它作为M_struct
            site_idx = self._current_site_idx_for_feature_calc
            if site_idx in self.atomic_tensors:
                M_struct = self.atomic_tensors[site_idx]
            else:
                # 从结构张量的特征向量构造"结构矢量"
                eigenvalues, eigenvectors = np.linalg.eigh(T_struct)
                # 使用最大特征值对应的特征向量作为结构的主方向
                v_struct = eigenvectors[:, np.argmax(eigenvalues)]
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
        try:
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            if not nn_info:
                return 0.0
            
            distances = [self.structure[site_idx].distance(nn['site']) for nn in nn_info]
            if len(distances) < 2: # 至少需要两个键才能计算畸变
                return 0.0
            
            mean_dist = np.mean(distances)
            if mean_dist < self._TOLERANCE: # 避免除以零
                return 0.0
            
            # 计算键长畸变：所有键长与平均键长差的平方和的平均值
            distortion = np.sqrt(np.sum([(d - mean_dist)**2 for d in distances]) / len(distances)) / mean_dist
            return float(distortion)
        except Exception as e:
            print(f"  - 错误: 计算键长畸变失败: {e}")
            return np.nan

    def _calculate_local_asymmetry_vector(self, site_idx: int) -> np.ndarray:
        try: nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        except Exception as e:
            print(f"  - 错误: 计算局部原子环境的结构不对称性向量失败: {e}")
            return np.zeros(3)
        if not nn_info: return np.zeros(3)
        
        center_coords = self.structure[site_idx].coords
        return np.sum([n['site'].coords - center_coords for n in nn_info], axis=0)




            
    def _get_site_symmetry_order(self, site_idx: int) -> int:
        """(新增) 计算并返回原子位点对称群的阶数 |G_site|。"""
        try:
            if self.sga is None:
                raise RuntimeError("SpacegroupAnalyzer 未初始化，无法计算位点对称群。")
                
            symmetry_ops = self.sga.get_symmetry_operations()
            center_site = self.structure[site_idx]
            order_site = 0
            
            for sym_op in symmetry_ops:
                transformed_frac_coords = sym_op.operate(center_site.frac_coords)
                diff = transformed_frac_coords - center_site.frac_coords
                diff = diff - np.round(diff)
                
                if np.allclose(diff, 0, atol=1e-5):
                    order_site += 1
            
            return order_site if order_site > 0 else None
        except Exception as e:
            print(f"  - 警告: 获取位点对称阶数失败 for site {site_idx}: {e}")
            return None # 返回None作为后备

    # ==========================================================================
    # 临时变量，用于在特征计算中传递上下文
    _current_site_idx_for_feature_calc: int = -1
    # ==========================================================================

def run_0_simplex_features(cif_file: str, pw_gpw_file: str, fd_gpw_file: Optional[str] = None, output_dir: Optional[str] = None, atomic_tensors_csv: Optional[str] = None) -> Tuple[pd.DataFrame, Path]:
    """
    计算并保存0-单纯形（原子）特征及其结构张量。

    参数:
        cif_file (str): 输入CIF文件的路径。
        pw_gpw_file (str): 平面波GPW文件的路径。
        fd_gpw_file (Optional[str]): FD模式GPW文件的路径。
        output_dir (Optional[str]): 输出CSV文件和张量文件的目录。如果为None，则使用cif_file所在的目录。
        atomic_tensors_csv (Optional[str]): (新增) 包含李代数原子张量的CSV文件路径。

    返回:
        Tuple[pd.DataFrame, Path]: 包含统一原子特征的DataFrame，以及保存结构张量CSV文件的路径。
    """
    print("=" * 60)
    print("开始计算0-单纯形（原子）特征")
    print("=" * 60)

    # 确保 output_dir 是 Path 对象
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(cif_file).parent
    
    # 基于输入文件名自动生成输出文件名
    base_name = Path(cif_file).stem.replace('-optimized', '')
    features_output_csv = output_path / f"{base_name}-0-Simplex-Features.csv"
    tensors_output_csv = output_path / f"{base_name}-0-Simplex-Structure-Tensors.csv"
    
    print(f"输入文件: {cif_file}, {pw_gpw_file}, {fd_gpw_file}")
    print(f"特征输出: {features_output_csv}")
    print(f"张量输出: {tensors_output_csv}")

    try:
        calculator = UnifiedFeatureCalculator(
            cif_file, 
            pw_gpw_file, 
            fd_gpw_file, 
            output_dir_path=output_path, 
            atomic_tensors_csv_path=atomic_tensors_csv
        )
        unified_features_df = calculator.calculate_unified_features()
        
        # 保存结果
        calculator.save_features_to_csv(unified_features_df, str(features_output_csv))
        # _save_structure_tensors 已经在 calculate_unified_features 内部调用，
        # 并将文件保存到 calculator.workdir / f"{base_name}-0-Simplex-Structure-Tensors.csv"
        # 因此，这里只需要返回该路径即可
        
        print("--- 0-单纯形特征计算完成 ---")
        return unified_features_df, tensors_output_csv

    except FileNotFoundError as e:
        print(f"错误: 0-单纯形计算关键输入文件缺失: {e}")
        raise
    except Exception as e:
        print(f"一个意外的错误发生于0-单纯形计算: {e}")
        import traceback
        traceback.print_exc()
        raise
