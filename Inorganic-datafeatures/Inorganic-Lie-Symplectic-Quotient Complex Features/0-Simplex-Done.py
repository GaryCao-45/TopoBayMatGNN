"""
无机钙钛矿材料数据库原子特征计算器
Inorganic Perovskite Database Atomic Feature Calculator

李-辛-商复形特征 - 0-单纯形特征：原子特征（28维）
Lie-Symplectic-Quotient Complex Features - 0-Simplex (Atomic) Features: 28 Dimensions
"""

import numpy as np  # 导入数值计算库（数学运算）/ Numerical computation library
import pandas as pd  # 导入数据处理库（表格数据）/ Data processing library
from pymatgen.core import Structure  # 晶体结构类 / Crystal structure class
from pymatgen.analysis.local_env import CrystalNN  # 最近邻分析器 / Nearest neighbor analyzer
from pymatgen.analysis.bond_valence import BVAnalyzer  # 键价分析器 / Bond valence analyzer
import hashlib  # 哈希库（唯一标识符）/ Hash library for unique identifiers
import warnings  # 警告处理库 / Warning control
warnings.filterwarnings('ignore')  # 忽略所有警告 / Ignore all warnings
from mendeleev import element  # 元素周期表库 / Periodic table library
from pymatgen.core.periodic_table import Specie, Element  # 离子种类与元素类 / Specie and Element classes
from collections import defaultdict  # 默认字典 / Default dictionary
import geomstats.backend as gs  # type: ignore # geomstats后端（numpy/pytorch）/ geomstats backend
from geomstats.geometry.hypersphere import Hypersphere  # 超球面几何 / Hypersphere geometry
from geomstats.learning.pca import TangentPCA  # type: ignore # 切空间主成分分析 / Tangent space PCA
import geomstats.geometry.euclidean as gs_euclidean  # 欧氏空间 / Euclidean geometry


class PerovskiteAtomicFeatureCalculator:
    """
    钙钛矿材料0-单纯形原子特征计算器
    0-simplex atomic feature calculator for perovskite materials
    """

    def __init__(self, cif_file_path):
        """
        初始化计算器，加载CIF文件和必要工具
        Initialize the calculator, load CIF file and required tools

        参数/Parameters:
        -----------
        cif_file_path : str
            CIF 文件路径 / Path to CIF file (input data source)
        """
        self.cif_path = cif_file_path  # 保存CIF文件路径 / Store CIF file path
        self.structure = Structure.from_file(cif_file_path)  # 读取晶体结构 / Read crystal structure
        self.crystal_nn = CrystalNN()  # 初始化最近邻分析器 / Initialize nearest neighbor analyzer
        self.site_assignments = self._assign_sites()  # 预分配A/B/X位点 / Pre-assign A/B/X sites

        # 打印基本信息 / Print basic info
        print(f"成功读取 CIF 文件: {cif_file_path}")  # CIF file loaded successfully
        print(f"化学式: {self.structure.composition.reduced_formula}")  # Chemical formula
        print(f"晶格参数: a={self.structure.lattice.a:.3f}, b={self.structure.lattice.b:.3f}, c={self.structure.lattice.c:.3f}")  # Lattice parameters

    def calculate_atomic_features(self):
        """
        计算所有原子的0-单纯形（28维）特征
        Calculate 0-simplex (28D) features for all atoms

        返回/Returns:
        --------
        pd.DataFrame: 每个原子的28维特征矩阵 / 28D feature matrix for each atom
        """
        features_list = []  # 存储所有原子的特征 / Store features for all atoms

        for site_idx, site in enumerate(self.structure.sites):
            site_features = self._calculate_single_site_features(site_idx, site)
            features_list.append(site_features)

        # 特征名称（中英双语注释）/ Feature names (with bilingual comments)
        feature_names = [
            'atomic_number',         # 原子序数 / Atomic number
            'group_number',          # 族号 / Group number
            'period_number',         # 周期号 / Period number
            'atomic_mass',           # 原子质量 / Atomic mass
            'electronegativity',     # 电负性 / Electronegativity
            'valence_electrons',     # 价电子数 / Valence electrons
            'ionization_energy',     # 电离能 / Ionization energy
            'electron_affinity',     # 电子亲和能 / Electron affinity
            'oxidation_state',       # 氧化态 / Oxidation state
            'covalent_radius',       # 共价半径 / Covalent radius
            'ionic_radius',          # 离子半径 / Ionic radius
            'van_der_waals_radius',  # 范德华半径 / van der Waals radius
            'coordination_number',   # 配位数 / Coordination number
            'heat_of_formation',     # 生成热 / Heat of formation
            'fusion_heat ',          # 熔化热 / Fusion heat
            'electrophilicity_index',# 亲电指数 / Electrophilicity index
            'tolerance_factor_contrib',      # 容忍因子贡献 / Tolerance factor contribution
            'octahedral_distortion_index',   # 八面体畸变指数 / Octahedral distortion index
            'frac_coord_x',          # 分数坐标x / Fractional coordinate x
            'frac_coord_y',          # 分数坐标y / Fractional coordinate y
            'frac_coord_z',          # 分数坐标z / Fractional coordinate z
            'quotient_hash',         # 商等价类哈希 / Quotient class hash
            'avg_site_valence',      # 平均位点价态 / Average site valence
            'atomic_casimir_invariant',      # 卡西米尔不变量 / Casimir invariant
            'atomic_symplectic_invariant',   # 辛不变量 / Symplectic invariant
            'atomic_quotient_metric_strict', # 严格商度量 / Strict quotient metric
            'sphere_exp_log_distance',       # 球面指数/对数映射距离 / Spherical exp/log distance
            'manifold_dimension_estimate'    # 流形维数估计 / Manifold dimension estimate
        ]

        df = pd.DataFrame(features_list, columns=feature_names)
        return df

    def _calculate_single_site_features(self, site_idx, site):
        """
        计算单个原子位点的28维特征
        Calculate 28D features for a single atomic site

        参数/Parameters:
        -----------
        site_idx : int
            原子索引 / Atom index
        site : PeriodicSite
            原子位点对象 / Atomic site object

        返回/Returns:
        --------
        list: 28维特征向量 / 28D feature vector
        """
        pymatgen_element = site.specie
        element_symbol = pymatgen_element.symbol
        mendeleev_elem = element(element_symbol)

        # 1. 基础原子特征 / Basic atomic features
        atomic_number = pymatgen_element.Z
        group_number = mendeleev_elem.group_id if mendeleev_elem.group_id else pymatgen_element.group
        period_number = mendeleev_elem.period if mendeleev_elem.period else pymatgen_element.row
        atomic_mass = mendeleev_elem.atomic_weight

        # 2. 电子特征 / Electronic features
        electronegativity = self._get_electronegativity_safely(mendeleev_elem)
        ionization_energy = self._get_ionization_energy_safely(mendeleev_elem)
        electron_affinity = self._get_electron_affinity_safely(mendeleev_elem)
        valence_electrons = self._calculate_precise_valence_electrons(mendeleev_elem)

        # 3. 氧化态 / Oxidation state
        oxidation_state = float(pymatgen_element.oxi_state) if pymatgen_element.oxi_state else 0.0

        # 4. 原子半径 / Atomic radii
        covalent_radius = self._get_precise_covalent_radius(mendeleev_elem)
        van_der_waals_radius = self._get_precise_vdw_radius(element_symbol, mendeleev_elem)
        coordination_number = self._get_coordination_number(site_idx)
        ionic_radius = self._get_precise_ionic_radius(element_symbol, oxidation_state)

        # 5. 元素分类 / Elemental classification
        heat_of_formation = self._get_heat_of_formation(mendeleev_elem)
        fusion_heat = self._get_fusion_heat(mendeleev_elem)
        electrophilicity_index = self._get_electrophilicity_index(mendeleev_elem)

        # 6. 钙钛矿特异性特征 / Perovskite-specific features
        tolerance_factor_contrib = self._calculate_tolerance_factor_contribution(pymatgen_element, ionic_radius)
        octahedral_distortion_index = self._calculate_octahedral_distortion_index(site_idx)

        # 7. 晶体学坐标 / Crystallographic coordinates
        frac_coords = site.frac_coords

        # 8. 代数特征 / Algebraic features
        quotient_hash = self._calculate_quotient_hash(site, frac_coords)
        atomic_quotient_metric_strict = self._calculate_atomic_quotient_metric_strict(site, frac_coords)
        atomic_casimir_invariant = self._calculate_atomic_casimir_invariant(frac_coords)
        atomic_symplectic_invariant = self._calculate_atomic_symplectic_invariant(frac_coords, site_idx)
        sphere_exp_log_distance = self._calculate_sphere_exp_log_distance(frac_coords)
        manifold_dimension_estimate = self._calculate_manifold_dimension_estimate(site_idx)

        # 9. 键价分析 / Bond valence analysis
        avg_site_valence = self._calculate_average_site_valence(site_idx)

        # 返回28维特征 / Return 28D feature vector
        return [
            atomic_number, group_number, period_number, atomic_mass,
            electronegativity, valence_electrons, ionization_energy, electron_affinity,
            oxidation_state, covalent_radius, ionic_radius, van_der_waals_radius,
            coordination_number, heat_of_formation, fusion_heat, electrophilicity_index,
            tolerance_factor_contrib, octahedral_distortion_index,
            frac_coords[0], frac_coords[1], frac_coords[2],
            quotient_hash, avg_site_valence,
            atomic_casimir_invariant, atomic_symplectic_invariant, atomic_quotient_metric_strict,
            sphere_exp_log_distance, manifold_dimension_estimate
        ]

    # 以下为各类特征的具体计算方法，均配有中英文注释
    # The following are specific feature calculation methods, all with bilingual comments

    def _get_heat_of_formation(self, mendeleev_elem):
        """获取元素的生成热(kJ/mol) / Get heat of formation (kJ/mol)"""
        if hasattr(mendeleev_elem, 'heat_of_formation') and mendeleev_elem.heat_of_formation:
            return float(mendeleev_elem.heat_of_formation)
        else:
            return np.nan

    def _get_fusion_heat(self, mendeleev_elem):
        """获取元素的熔化热(kJ/mol) / Get fusion heat (kJ/mol)"""
        if hasattr(mendeleev_elem, 'fusion_heat') and mendeleev_elem.fusion_heat:
            return float(mendeleev_elem.fusion_heat)
        else:
            return np.nan

    def _get_electrophilicity_index(self, mendeleev_elem):
        """获取元素的亲电性指数 / Get electrophilicity index"""
        if hasattr(mendeleev_elem, 'electrophilicity') and mendeleev_elem.electrophilicity:
            return float(mendeleev_elem.electrophilicity())
        else:
            return np.nan

    def _get_coordination_number(self, site_idx):
        """计算配位数 / Calculate coordination number"""
        nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        return len(nn_info)

    def _get_electronegativity_safely(self, mendeleev_elem):
        """获取电负性（Pauling标度）/ Get electronegativity (Pauling scale)"""
        if hasattr(mendeleev_elem, 'en_pauling') and mendeleev_elem.en_pauling is not None:
            return float(mendeleev_elem.en_pauling)
        else:
            # 若无数据，取结构中所有元素的平均电负性 / Use mean electronegativity of all elements in structure
            return np.mean([
                element(e.specie.symbol).en_pauling
                for e in self.structure.sites
                if hasattr(element(e.specie.symbol), 'en_pauling')
            ])

    def _get_ionization_energy_safely(self, mendeleev_elem):
        """获取第一电离能(eV) / Get first ionization energy (eV)"""
        if hasattr(mendeleev_elem, 'ionenergies') and mendeleev_elem.ionenergies:
            return float(mendeleev_elem.ionenergies[1])
        else:
            return np.nan

    def _get_electron_affinity_safely(self, mendeleev_elem):
        """获取电子亲和能(eV) / Get electron affinity (eV)"""
        return float(mendeleev_elem.electron_affinity)

    def _calculate_precise_valence_electrons(self, mendeleev_elem):
        """获取价电子数 / Get number of valence electrons"""
        if hasattr(mendeleev_elem, 'nvalence') and mendeleev_elem.nvalence:
            return int(mendeleev_elem.nvalence())
        else:
            return np.nan

    def _assign_sites(self):
        """
        预分配A/B/X位点类型
        Pre-assign A/B/X site types for perovskite structure
        """
        site_types = defaultdict(list)
        bv_analyzer = BVAnalyzer()
        valences = bv_analyzer.get_valences(self.structure)

        for i, site in enumerate(self.structure.sites):
            elem = site.specie
            oxi = valences[i]
            coord_num = self._get_coordination_number(i)
            ionic_r = self._get_precise_ionic_radius(elem.symbol, oxi)

            if oxi > 0 and coord_num >= 8:
                site_types['A'].append(i)
            elif oxi > 0 and 4 <= coord_num <= 6:
                site_types['B'].append(i)
            elif oxi < 0:
                site_types['X'].append(i)
            else:
                site_types['unknown'].append(i)
        return site_types

    def _is_x_site_element(self, elem, oxi_state):
        """
        判断是否为X位点元素
        Determine if the element is an X-site (anion) in perovskite
        """
        mendeleev_elem = element(elem.symbol)
        electroneg = self._get_electronegativity_safely(mendeleev_elem)
        return electroneg > 2.0 and oxi_state < 0

    def _get_precise_covalent_radius(self, mendeleev_elem):
        """获取共价半径(Å) / Get covalent radius (Å)"""
        if hasattr(mendeleev_elem, 'covalent_radius_pyykko') and mendeleev_elem.covalent_radius_pyykko:
            return float(mendeleev_elem.covalent_radius_pyykko) / 100.0
        else:
            return np.nan

    def _get_precise_vdw_radius(self, element_symbol, mendeleev_elem):
        """获取范德华半径(Å) / Get van der Waals radius (Å)"""
        if hasattr(mendeleev_elem, 'vdw_radius') and mendeleev_elem.vdw_radius:
            return float(mendeleev_elem.vdw_radius) / 100.0
        else:
            return np.nan

    def _get_precise_ionic_radius(self, element_symbol, oxidation_state):
        """获取离子半径(Å) / Get ionic radius (Å)"""
        if oxidation_state != 0:
            try:
                specie = Specie(element_symbol, oxidation_state)
                radius = specie.ionic_radius
                if radius is not None:
                    return float(radius)
            except Exception as e:
                print(f"获取离子半径时出错: {e}")
                return np.nan
        return np.nan

    def _calculate_quotient_hash(self, site, frac_coords):
        """
        计算商等价类哈希
        Calculate quotient class hash for atomic environment
        """
        try:
            normalized_coords = frac_coords % 1.0
            quantized_coords = np.round(normalized_coords * 1000).astype(int)
            hash_string = f"{site.specie.symbol}_{quantized_coords[0]}_{quantized_coords[1]}_{quantized_coords[2]}"
            hash_object = hashlib.md5(hash_string.encode())
            hash_hex = hash_object.hexdigest()
            return int(hash_hex[:8], 16)
        except (AttributeError, ValueError, TypeError):
            return np.nan

    def _construct_so3_generator(self, vector):
        """
        构造SO(3)李代数生成元（反对称矩阵）
        Construct SO(3) Lie algebra generator (antisymmetric matrix)
        """
        if len(vector) != 3:
            vector = np.pad(vector, (0, max(0, 3 - len(vector))))[:3]
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    def _calculate_atomic_casimir_invariant(self, frac_coords):
        """
        计算SO(3) Casimir不变量
        Calculate SO(3) Casimir invariant for atomic position
        """
        try:
            generator = self._construct_so3_generator(frac_coords)
            casimir_2 = float(np.trace(generator @ generator))
            return casimir_2
        except Exception as e:
            print(f"原子Casimir不变量计算警告: {e}")
            return np.nan

    def _calculate_atomic_symplectic_invariant(self, frac_coords, site_idx):
        """
        计算原子的辛不变量
        Calculate atomic symplectic invariant (phase space localization)
        """
        try:
            lattice_vectors = self.structure.lattice.matrix
            position = frac_coords @ lattice_vectors
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            if len(nn_info) > 0:
                avg_neighbor_displacement = np.mean([
                    neighbor['site'].coords - position
                    for neighbor in nn_info
                ], axis=0)
                momentum_approx = avg_neighbor_displacement / np.linalg.norm(avg_neighbor_displacement + 1e-10)
            else:
                lattice_mean = np.mean(self.structure.lattice.matrix, axis=0)
                momentum_approx = lattice_mean / np.linalg.norm(lattice_mean + 1e-10)
            phase_space = np.concatenate([position, momentum_approx])
            I = np.eye(3)
            O = np.zeros((3, 3))
            J_symplectic = np.block([[O, I], [-I, O]])
            symplectic_invariant = float(phase_space.T @ J_symplectic @ phase_space)
            return symplectic_invariant
        except Exception as e:
            print(f"原子辛不变量计算警告: {e}")
            return np.nan

    def _calculate_atomic_quotient_metric_strict(self, site, frac_coords):
        """
        计算严格的原子商代数度量（Killing形式）
        Calculate strict atomic quotient metric (Killing form)
        """
        normalized_coords = frac_coords % 1.0
        gen_position = self._construct_so3_generator(normalized_coords)
        element_vector = np.array([
            float(site.specie.Z % 10) / 10.0,
            float(site.specie.group % 10) / 10.0,
            float(site.specie.row % 10) / 10.0
        ])
        gen_element = self._construct_so3_generator(element_vector)
        killing_form = float(np.trace(gen_position @ gen_element))
        return killing_form

    def _calculate_sphere_exp_log_distance(self, frac_coords):
        """
        计算球面指数/对数映射距离
        Calculate spherical exp/log map distance (geomstats)
        """
        try:
            sphere = Hypersphere(dim=3)
            coords_3d = frac_coords / np.linalg.norm(frac_coords + 1e-10)
            w = np.sqrt(max(0, 1.0 - np.sum(coords_3d**2)))
            sphere_point = gs.array([coords_3d[0], coords_3d[1], coords_3d[2], w])
            north_pole = gs.array([0.0, 0.0, 0.0, 1.0])
            log_map = sphere.metric.log(sphere_point, north_pole)
            geodesic_distance = float(sphere.metric.norm(log_map))
            return geodesic_distance
        except Exception as e:
            print(f"球面指数/对数映射距离计算警告: {e}")
            return np.nan

    def _calculate_manifold_dimension_estimate(self, site_idx):
        """
        估算原子环境流形维数
        Estimate manifold dimension of atomic environment (geomstats)
        """
        nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        if len(nn_info) < 3:
            return 1.0
        neighbor_vectors = []
        center_coords = self.structure[site_idx].coords
        for neighbor in nn_info:
            neighbor_coords = neighbor['site'].coords
            vector = neighbor_coords - center_coords
            neighbor_vectors.append(vector)
        neighbor_matrix = np.array(neighbor_vectors)
        euclidean = gs_euclidean.Euclidean(dim=3)
        manifold_points = gs.array(neighbor_matrix)
        mean_point = gs.mean(manifold_points, axis=0)
        tangent_pca = TangentPCA(euclidean, n_components=min(len(neighbor_vectors), 3))
        tangent_pca.fit(manifold_points, base_point=mean_point)
        explained_variance_ratio = tangent_pca.explained_variance_ratio_
        cumulative_variance = gs.cumsum(explained_variance_ratio)
        effective_dimension = int(gs.argmax(cumulative_variance >= 0.95)) + 1
        return float(effective_dimension / 3.0)

    def _calculate_tolerance_factor_contribution(self, element, ionic_radius):
        """
        计算Goldschmidt容忍因子贡献
        Calculate Goldschmidt tolerance factor contribution
        """
        oxi_state = float(element.oxi_state) if element.oxi_state else 0
        avg_radius = np.mean([
            self._get_precise_ionic_radius(s.specie.symbol, s.specie.oxi_state)
            for s in self.structure.sites if s.specie != element
        ])
        if oxi_state > 0:
            return ionic_radius / avg_radius if avg_radius > 0 else np.nan
        else:
            return avg_radius / ionic_radius if ionic_radius > 0 else np.nan

    def _calculate_average_site_valence(self, site_idx):
        """
        计算平均位点价态
        Calculate average site valence (bond valence theory)
        """
        try:
            bv_analyzer = BVAnalyzer()
            valences = bv_analyzer.get_valences(self.structure)
            return float(valences[site_idx])
        except (ImportError, AttributeError, ValueError, TypeError):
            return np.nan

    def _calculate_octahedral_distortion_index(self, site_idx):
        """
        计算八面体畸变指数
        Calculate octahedral distortion index for B-site
        """
        site_type = None
        for typ, indices in self.site_assignments.items():
            if site_idx in indices:
                site_type = typ
                break
        if site_type != 'B':
            return 1.0

        nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
        if len(nn_info) < 4:
            return 1.0
        partial_factor = 0.0 if len(nn_info) == 6 else (6 - len(nn_info)) / 6.0

        valences = BVAnalyzer().get_valences(self.structure)
        x_neighbors = [
            neigh for neigh in nn_info
            if self._is_x_site_element(Element(neigh['site'].specie.symbol), valences[neigh['site'].index])
        ]
        if len(x_neighbors) < 4:
            return 1.0

        center_coords = self.structure[site_idx].coords
        distances = [np.linalg.norm(neigh['site'].coords - center_coords) for neigh in x_neighbors]

        angles_90 = []
        angles_180 = []
        for i in range(len(x_neighbors)):
            for j in range(i + 1, len(x_neighbors)):
                v1 = x_neighbors[i]['site'].coords - center_coords
                v2 = x_neighbors[j]['site'].coords - center_coords
                cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
                if abs(ang - 90) < 45:
                    angles_90.append(ang)
                elif abs(ang - 180) < 45:
                    angles_180.append(ang)

        if distances:
            avg_dist = np.mean(distances)
            dist_distortion = (np.max(distances) - np.min(distances)) / avg_dist if avg_dist > 0 else 0
        else:
            dist_distortion = 0

        angle_distortion_90 = np.std([abs(ang - 90) for ang in angles_90]) / 90 if angles_90 else 0
        angle_distortion_180 = np.std([abs(ang - 180) for ang in angles_180]) / 180 if angles_180 else 0
        angle_distortion = (angle_distortion_90 + angle_distortion_180) / 2

        distortion = (dist_distortion + angle_distortion) / 2 + partial_factor
        distortion = min(1.0, max(0.0, distortion))
        return float(distortion)

    def save_atom_features_to_csv(self, output_file):
        """
        计算并保存原子特征到CSV文件
        Calculate and save atomic features to CSV file

        参数/Parameters:
        -----------
        output_file : str
            输出CSV文件路径 / Output CSV file path
        """
        atom_features_df = self.calculate_atomic_features()
        atom_features_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"原子特征已保存到: {output_file}")  # 打印保存成功信息 / Print success message
        return atom_features_df

# 使用示例 / Example usage
if __name__ == "__main__":
    cif_file = "E:\\RA\\数据库搭建工作\\Examples\\CsPbBr3.cif"  # CIF文件路径 / Path to CIF file
    calculator = PerovskiteAtomicFeatureCalculator(cif_file)  # 初始化特征计算器 / Initialize feature calculator
    output_file = "CsPbBr3_atomic_features.csv"  # 输出文件路径 / Output file path
    calculator.save_atom_features_to_csv(output_file)  # 保存特征到CSV / Save features to CSV
    
    