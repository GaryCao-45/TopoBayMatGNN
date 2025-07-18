"""
无机钙钛矿材料数据库特征计算器
Inorganic Perovskite Database Feature Calculator

李-辛-商复形特征 - 1-单纯形特征：化学键特征（22维）
Lie-Symplectic-Quotient Complex Features - 1-Simplex (Bond) Features: 22 Dimensions
"""

import numpy as np  # 数值计算 / Numerical computation
import pandas as pd  # 数据处理 / Data processing
import math  # 数学运算 / Mathematical operations
from pymatgen.core import Structure  # 晶体结构处理 / Crystal structure processing
from pymatgen.analysis.local_env import CrystalNN  # 最近邻分析 / Nearest neighbor analysis
import warnings  # 警告信息处理 / Warning message handling
import geomstats.backend as gs  # type: ignore  # 几何统计后端 / Geometric statistics backend
from geomstats.geometry.hypersphere import Hypersphere  # 超球面几何 / Hypersphere geometry
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # 特殊正交群 / Special orthogonal group

warnings.filterwarnings('ignore')  # 忽略所有警告 / Ignore all warnings

class PerovskiteBondFeatureCalculator:
    """
    钙钛矿材料1-单纯形（化学键）特征计算器
    Perovskite 1-Simplex (Bond) Feature Calculator
    所有特征均从CIF文件自动计算，无需人工干预
    All features are automatically calculated from CIF files without manual intervention
    """

    def __init__(self, cif_file_path: str):
        """
        初始化特征计算器
        Initialize the feature calculator

        参数/Param:
            cif_file_path (str): CIF文件路径 / CIF file path
        """
        self.cif_path = cif_file_path  # 存储CIF文件路径 / Store CIF file path
        self.structure = Structure.from_file(cif_file_path)  # 从CIF文件加载晶体结构 / Load crystal structure from CIF file
        self.crystal_nn = CrystalNN()  # 初始化最近邻分析器 / Initialize nearest neighbor analyzer

        # 统计所有合理化学键长度，用于径向基函数展开
        # Count all reasonable bond lengths for radial basis function expansion
        bond_lengths = [
            self.structure.get_distance(i, j)
            for i in range(len(self.structure))
            for j in range(i + 1, len(self.structure))
            if 1.0 < self.structure.get_distance(i, j) < 6.0  # 过滤合理的键长范围 / Filter reasonable bond length range
        ]
        avg_bond = np.mean(bond_lengths)  # 平均键长 / Average bond length
        std_bond = np.std(bond_lengths) or 0.1  # 键长标准差 / Bond length standard deviation
        self.rbf_centers = [np.min(bond_lengths), avg_bond, np.max(bond_lengths)]  # RBF中心点 / RBF centers
        self.rbf_width = std_bond / 2  # RBF宽度 / RBF width

        # 打印基本信息 / Print basic information
        print(f"成功读取 CIF 文件: {cif_file_path} / Successfully loaded CIF file: {cif_file_path}")
        print(f"化学式: {self.structure.composition.reduced_formula} / Chemical formula: {self.structure.composition.reduced_formula}")
        print(f"晶格参数: a={self.structure.lattice.a:.3f}, b={self.structure.lattice.b:.3f}, c={self.structure.lattice.c:.3f} / Lattice parameters: a={self.structure.lattice.a:.3f}, b={self.structure.lattice.b:.3f}, c={self.structure.lattice.c:.3f}")

    def calculate_bond_features(self) -> pd.DataFrame:
        """
        计算所有化学键的1-单纯形特征（22维）
        Calculate all bond (1-simplex) features (22 dimensions)

        返回/Return:
            pd.DataFrame: 每个化学键的特征矩阵 / Feature matrix for each bond
        """
        bond_features_list = []  # 存储所有化学键特征 / Store all bond features

        # 遍历每个原子位点，获取其所有最近邻
        # Iterate through each atomic site and get all nearest neighbors
        for site_idx, site in enumerate(self.structure.sites):
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)  # 获取最近邻信息 / Get nearest neighbor information
            for neighbor_info in nn_info:
                bond_features = self._calculate_single_bond_features(site_idx, site, neighbor_info)  # 计算单个键的特征 / Calculate features for a single bond
                bond_features_list.append(bond_features)

        # 特征名称（22维） / Feature names (22 dimensions)
        feature_names = [
            'bond_distance', 'distance_inverse', 'bond_direction_x', 'bond_direction_y', 'bond_direction_z',
            'rbf_expansion_1', 'rbf_expansion_2', 'rbf_expansion_3', 'crosses_boundary', 'periodic_phase_x',
            'periodic_phase_y', 'wrap_vec_x', 'wrap_vec_y', 'wrap_vec_z', 'lie_bracket_mag',
            'bond_casimir_2', 'bond_symplectic_gen', 'bond_quotient_metric',
            'geodesic_length_sphere', 'geodesic_length_so3', 'so3_exp_log_distance', 'parallel_transport_norm'
        ]
        return pd.DataFrame(bond_features_list, columns=feature_names)

    def _calculate_single_bond_features(self, site_idx, site, neighbor_info):
        """
        计算单个化学键的所有特征
        Calculate all features for a single bond

        参数/Param:
            site_idx (int): 中心原子索引 / Central atom index
            site (PeriodicSite): 中心原子对象 / Central atom object
            neighbor_info (dict): 最近邻信息 / Nearest neighbor information

        返回/Return:
            list: 单个化学键的22维特征向量 / 22-dimensional feature vector for a single bond
        """
        # 计算化学键向量 / Calculate bond vector
        bond_vector = neighbor_info['image'] @ self.structure.lattice.matrix + \
                      self.structure.sites[neighbor_info['site_index']].coords - site.coords
        bond_distance = np.linalg.norm(bond_vector)  # 键长 / Bond length
        distance_inverse = 1.0 / bond_distance if bond_distance > 0 else 0.0  # 键长倒数 / Inverse bond length

        # 化学键方向 / Bond direction
        bond_direction = bond_vector / bond_distance  # 单位方向向量 / Unit direction vector
        bond_direction_x, bond_direction_y, bond_direction_z = bond_direction


        # 径向基函数展开 / Radial basis function expansion
        rbf_expansion_1 = math.exp(-((bond_distance - self.rbf_centers[0]) ** 2) / (2 * self.rbf_width ** 2))
        rbf_expansion_2 = math.exp(-((bond_distance - self.rbf_centers[1]) ** 2) / (2 * self.rbf_width ** 2))
        rbf_expansion_3 = math.exp(-((bond_distance - self.rbf_centers[2]) ** 2) / (2 * self.rbf_width ** 2))

        # 周期性边界信息 / Periodic boundary information
        image_vector = neighbor_info['image']  # 周期性镜像向量 / Periodic image vector
        crosses_boundary = float(np.any(image_vector != 0))  # 是否跨越边界 / Whether crosses boundary
        wrap_vec_x, wrap_vec_y, wrap_vec_z = image_vector  # 包装向量分量 / Wrap vector components

        # 晶格周期性相位 / Lattice periodic phase
        reciprocal_lattice = self.structure.lattice.reciprocal_lattice  # 倒格子 / Reciprocal lattice
        k_vectors = reciprocal_lattice.matrix  # k向量 / k-vectors
        k_point = np.mean(k_vectors, axis=0) / np.linalg.norm(np.mean(k_vectors, axis=0) + 1e-10)  # 归一化k点 / Normalized k-point
        periodic_phase_x = math.cos(2 * math.pi * np.dot(k_point, bond_vector))  # 周期性相位x分量 / Periodic phase x component
        periodic_phase_y = math.sin(2 * math.pi * np.dot(k_point, bond_vector))  # 周期性相位y分量 / Periodic phase y component

        # 李代数与几何特征 / Lie algebra and geometric features
        lie_bracket_mag = self._calculate_lie_bracket_magnitude(bond_direction, site_idx)  # 李括号幅值 / Lie bracket magnitude
        bond_casimir_2 = self._calculate_casimir_2_so3(bond_direction)  # Casimir不变量 / Casimir invariant
        bond_symplectic_gen = self._calculate_symplectic_generator(bond_vector)  # 辛生成元 / Symplectic generator
        bond_quotient_metric = self._calculate_bond_quotient_metric(bond_direction, image_vector)  # 商度量 / Quotient metric
        geodesic_length_sphere = self._calculate_geodesic_length_sphere(bond_vector)  # 球面测地线长度 / Sphere geodesic length
        geodesic_length_so3 = self._calculate_geodesic_length_so3(bond_vector)  # SO(3)测地线长度 / SO(3) geodesic length
        so3_exp_log_distance = self._calculate_so3_exp_log_distance(bond_vector)  # SO(3)指数对数距离 / SO(3) exp-log distance
        parallel_transport_norm = self._calculate_parallel_transport_norm(bond_vector)  # 平行输运范数 / Parallel transport norm

        return [
            bond_distance, distance_inverse, bond_direction_x, bond_direction_y, bond_direction_z,
            rbf_expansion_1, rbf_expansion_2, rbf_expansion_3, crosses_boundary, periodic_phase_x,
            periodic_phase_y, wrap_vec_x, wrap_vec_y, wrap_vec_z, lie_bracket_mag,
            bond_casimir_2, bond_symplectic_gen, bond_quotient_metric,
            geodesic_length_sphere, geodesic_length_so3, so3_exp_log_distance, parallel_transport_norm
        ]

    # 以下为各类李代数、辛、商、测地线等特征的计算函数
    # The following are calculation functions for various Lie algebra, symplectic, quotient, and geodesic features
    def _calculate_lie_bracket_magnitude(self, bond_direction, site_idx):
        """
        计算SO(3)李括号幅值
        Compute the magnitude of the SO(3) Lie bracket

        返回/Return:
            float: 李括号的Frobenius范数 / Frobenius norm of the Lie bracket
        """
        site_coords = self.structure.sites[site_idx].frac_coords  # 原子分数坐标 / Atomic fractional coordinates
        # 动态处理维度 / Dynamically handle dimensions
        bond_direction = self._pad_vector_to_3d(bond_direction)
        site_coords = self._pad_vector_to_3d(site_coords)
        gen_A = self._construct_so3_generator(bond_direction)  # 构造SO(3)生成元A / Construct SO(3) generator A
        gen_B = self._construct_so3_generator(site_coords)  # 构造SO(3)生成元B / Construct SO(3) generator B
        lie_bracket = gen_A @ gen_B - gen_B @ gen_A  # 计算李括号 / Compute Lie bracket
        return float(np.linalg.norm(lie_bracket, 'fro'))  # 返回Frobenius范数 / Return Frobenius norm

    def _construct_so3_generator(self, vector):
        """
        构造SO(3)李代数生成元（反对称矩阵）
        Construct SO(3) Lie algebra generator (antisymmetric matrix)
        """
        vector = self._pad_vector_to_3d(vector)  # 确保向量为3维 / Ensure vector is 3D
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])  # 反对称矩阵 / Antisymmetric matrix

    def _pad_vector_to_3d(self, vector):
        """
        保证向量为3维，若不足则用结构平均坐标补齐
        Ensure the vector is 3D, pad with mean coordinates if needed
        """
        vector = np.array(vector, dtype=float)
        if len(vector) < 3:
            avg_coords = np.mean([s.coords for s in self.structure.sites], axis=0)  # 计算平均坐标 / Calculate average coordinates
            vector = np.concatenate([vector, avg_coords[:3 - len(vector)]])  # 用平均坐标补齐 / Pad with average coordinates
        elif len(vector) > 3:
            vector = vector[:3]  # 截取前3维 / Take first 3 dimensions
        return vector

    def _calculate_casimir_2_so3(self, vector):
        """
        计算SO(3)的二阶Casimir不变量
        Compute the 2nd order Casimir invariant of SO(3)
        """
        try:
            vector = self._pad_vector_to_3d(vector)
            generator = self._construct_so3_generator(vector)  # 构造生成元 / Construct generator
            casimir_2 = float(np.trace(generator @ generator))  # 计算二阶Casimir不变量 / Compute 2nd order Casimir invariant
            return casimir_2
        except Exception as e:
            print(f"Casimir不变量计算警告 / Casimir invariant calculation warning: {e}")
            return np.nan

    def _calculate_symplectic_generator(self, bond_vector):
        """
        计算辛代数生成元幅值
        Compute the magnitude of the symplectic generator
        """
        bond_vector_padded = self._pad_vector_to_3d(bond_vector)
        n = 3
        I = np.eye(n)  # 单位矩阵 / Identity matrix
        O = np.zeros((n, n))  # 零矩阵 / Zero matrix
        J_symplectic = np.block([[O, I], [-I, O]])  # 辛矩阵 / Symplectic matrix
        phase_space_vector = np.concatenate([bond_vector_padded, np.zeros(n)])  # 相空间向量 / Phase space vector
        symplectic_gen = J_symplectic @ phase_space_vector  # 辛生成元 / Symplectic generator
        return float(np.linalg.norm(symplectic_gen))

    def _calculate_bond_quotient_metric(self, bond_direction, image_vector):
        """
        计算化学键的商代数度量
        Compute the quotient metric for the bond
        """
        bond_direction = self._pad_vector_to_3d(bond_direction)
        image_vector = self._pad_vector_to_3d(image_vector)
        bond_gen = self._construct_so3_generator(bond_direction)  # 键方向生成元 / Bond direction generator
        image_gen = self._construct_so3_generator(image_vector)  # 镜像向量生成元 / Image vector generator
        quotient_metric = float(np.trace(bond_gen @ image_gen))  # 商度量 / Quotient metric
        return quotient_metric

    def _calculate_geodesic_length_sphere(self, bond_vector):
        """
        计算球面测地线长度
        Compute geodesic length on the sphere
        """
        norm = np.linalg.norm(bond_vector)  # 向量范数 / Vector norm
        if norm < 1e-10:
            return 0.0
        sphere = Hypersphere(dim=2)  # 创建2维超球面 / Create 2D hypersphere
        normalized = self._pad_vector_to_3d(bond_vector / norm)  # 归一化向量 / Normalized vector
        point = gs.array(normalized)
        base_point = gs.array([1.0, 0.0, 0.0])  # 基点 / Base point
        if np.allclose(normalized, [-1.0, 0.0, 0.0], atol=1e-6):
            return np.pi  # 对径点情况 / Antipodal point case
        log = sphere.metric.log(point, base_point)  # 对数映射 / Logarithmic map
        result = float(sphere.metric.norm(log))
        if result < 0 or result > np.pi or np.isnan(result) or np.isinf(result):
            dot_product = np.clip(np.dot(normalized, [1.0, 0.0, 0.0]), -1.0, 1.0)  # 点积 / Dot product
            return float(np.arccos(np.abs(dot_product)))  # 弧度角 / Arc cosine angle
        return result

    def _calculate_geodesic_length_so3(self, bond_vector):
        """
        计算SO(3)测地线长度
        Compute geodesic length in SO(3)
        """
        norm = np.linalg.norm(bond_vector)
        if norm < 1e-10:
            return 0.0
        so3 = SpecialOrthogonal(n=3, point_type='vector')  # 创建SO(3)群 / Create SO(3) group
        vec = self._pad_vector_to_3d(bond_vector / norm)
        base = gs.array([0.0, 0.0, 0.0])  # 基向量 / Base vector
        log = so3.log(vec, base)  # 对数映射 / Logarithmic map
        metric = so3.metric
        result = float(metric.norm(log))
        if result < 0 or result > np.pi or np.isnan(result) or np.isinf(result):
            return float(np.linalg.norm(vec))
        return result

    def _calculate_so3_exp_log_distance(self, bond_vector):
        """
        计算SO(3)指数-对数距离
        Compute exp-log distance in SO(3)
        """
        norm = np.linalg.norm(bond_vector)
        if norm < 1e-10:
            return 0.0
        so3 = SpecialOrthogonal(n=3, point_type='vector')
        vec = self._pad_vector_to_3d(bond_vector / norm)
        base = gs.array([0.0, 0.0, 0.0])
        log = so3.log(vec, base)  # 对数映射 / Logarithmic map
        exp = so3.exp(log, base)  # 指数映射 / Exponential map
        metric = so3.metric
        result = float(metric.dist(exp, vec))  # 计算距离 / Compute distance
        if result < 0 or result > 2 * np.pi or np.isnan(result) or np.isinf(result):
            return 0.0
        return result

    def _calculate_parallel_transport_norm(self, bond_vector):
        """
        计算球面平行输运范数
        Compute the norm of parallel transport on the sphere
        """
        norm = np.linalg.norm(bond_vector)
        if norm < 1e-10:
            return 0.0
        sphere = Hypersphere(dim=2)
        p1 = gs.array([1.0, 0.0, 0.0])  # 起始点 / Starting point
        p2 = self._pad_vector_to_3d(bond_vector / norm)  # 终点 / End point
        vec = gs.array([0.0, 1.0, 0.0])  # 输运向量 / Transport vector
        if np.allclose(p1, p2, atol=1e-6) or np.allclose(p1, -p2, atol=1e-6):
            return 1.0  # 相同点或对径点情况 / Same point or antipodal point case
        transported = sphere.metric.parallel_transport(vec, p1, p2)  # 平行输运 / Parallel transport
        result = float(gs.linalg.norm(transported))
        if result < 0 or result > 2.0 or np.isnan(result) or np.isinf(result):
            return 1.0
        return result

    def save_bond_features_to_csv(self, output_file: str) -> pd.DataFrame:
        """
        计算并保存化学键特征到CSV
        Compute and save bond features to CSV

        参数/Param:
            output_file (str): 输出CSV文件路径 / Output CSV file path

        返回/Return:
            pd.DataFrame: 特征数据表 / Feature data table
        """
        bond_features_df = self.calculate_bond_features()  # 计算特征 / Calculate features
        bond_features_df.to_csv(output_file, index=False, encoding='utf-8')  # 保存到CSV / Save to CSV
        print(f"化学键特征已保存到 / Bond features saved to: {output_file}")

        return bond_features_df

# ================== 使用示例 Example Usage ==================
if __name__ == "__main__":
    # CIF文件路径（请根据实际情况修改） / CIF file path (please modify according to actual situation)
    cif_file = r"E:\RA\数据库搭建工作\Examples\CsPbBr3.cif"
    # 初始化特征计算器 / Initialize feature calculator
    calculator = PerovskiteBondFeatureCalculator(cif_file)
    # 输出文件路径 / Output file path
    output_file = "CsPbBr3_bond_features.csv"
    # 计算并保存特征 / Calculate and save features
    calculator.save_bond_features_to_csv(output_file)
