"""
无机钙钛矿材料数据库特征计算器
Inorganic Perovskite Database Feature Calculator

李-辛-商复形特征 - 2-单纯形特征：三体相互作用特征（22维）
Lie-Symplectic-Quotient Complex Features - 2-Simplex (Triangle) Features: 22 Dimensions
"""

import numpy as np  # 数值计算库 / Numerical computation
import pandas as pd  # 数据处理库 / Data processing
import math  # 数学运算 / Mathematical operations
from pymatgen.core import Structure  # 晶体结构处理 / Crystal structure processing
from pymatgen.analysis.local_env import CrystalNN  # 最近邻分析 / Nearest neighbor analysis
from itertools import combinations  # 组合生成器 / Combinatorial generator
import warnings  # 警告信息处理 / Warning message handling
warnings.filterwarnings('ignore')  # 忽略所有警告 / Ignore all warnings

import geomstats.backend as gs  # type: ignore  # 几何统计后端 / Geometric statistics backend
from geomstats.geometry.hypersphere import Hypersphere  # 超球面几何 / Hypersphere geometry
from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # 特殊正交群 / Special orthogonal group
from geomstats.geometry.euclidean import Euclidean  # 欧氏空间 / Euclidean geometry

class PerovskiteTriangleFeatureCalculator:
    """
    钙钛矿材料2-单纯形三体相互作用特征计算器
    Perovskite 2-Simplex (Triangle) Feature Calculator
    所有特征均从CIF文件自动计算，无需人工干预
    All features are automatically calculated from CIF files without manual intervention
    """

    def __init__(self, cif_file_path):
        """
        初始化特征计算器
        Initialize the feature calculator

        参数/Param:
            cif_file_path (str): CIF文件路径 / CIF file path
        """
        self.cif_path = cif_file_path  # 存储CIF文件路径 / Store CIF file path
        self.structure = Structure.from_file(cif_file_path)  # 读取晶体结构 / Load crystal structure
        self.crystal_nn = CrystalNN()  # 初始化最近邻分析器 / Initialize nearest neighbor analyzer
        # 预先缓存所有原子的最近邻信息 / Cache all nearest neighbor info
        self.nn_cache = {idx: self.crystal_nn.get_nn_info(self.structure, idx) for idx in range(len(self.structure))}

        # 统计所有三角形面积，用于径向基函数展开 / Collect all triangle areas for RBF expansion
        areas = []
        for site_idx in range(len(self.structure)):
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            for n1, n2 in combinations(nn_info, 2):
                c = self.structure[site_idx].coords
                p1 = self.structure[n1['site_index']].coords + n1['image'] @ self.structure.lattice.matrix
                p2 = self.structure[n2['site_index']].coords + n2['image'] @ self.structure.lattice.matrix
                edges = [np.linalg.norm(p1 - c), np.linalg.norm(p2 - c), np.linalg.norm(p2 - p1)]
                s = sum(edges) / 2
                area = math.sqrt(max(0, s * (s - edges[0]) * (s - edges[1]) * (s - edges[2])))
                areas.append(area)
        if areas:
            self.rbf_area_centers = [np.min(areas), np.mean(areas), np.max(areas)]
            self.rbf_area_width = (np.std(areas) or 0.1) / 2
        else:
            vol = self.structure.volume
            self.rbf_area_centers = [vol / 9, vol / 3, vol]
            self.rbf_area_width = vol / 10

        # 打印基本信息 / Print basic information
        print(f"成功读取 CIF 文件: {cif_file_path} / Successfully loaded CIF file: {cif_file_path}")
        print(f"化学式: {self.structure.composition.reduced_formula} / Chemical formula: {self.structure.composition.reduced_formula}")
        print(f"晶格参数: a={self.structure.lattice.a:.3f}, b={self.structure.lattice.b:.3f}, c={self.structure.lattice.c:.3f} / Lattice parameters: a={self.structure.lattice.a:.3f}, b={self.structure.lattice.b:.3f}, c={self.structure.lattice.c:.3f}")

    def calculate_triangle_features(self):
        """
        计算所有三体相互作用的2-单纯形特征（22维）
        Calculate all triangle (2-simplex) features (22 dimensions)

        返回/Return:
            pd.DataFrame: 每个三体相互作用的特征矩阵 / Feature matrix for each triangle
        """
        triangle_features_list = []  # 存储所有三体特征 / Store all triangle features

        # 遍历每个原子位点，获取其所有最近邻组合
        # Iterate through each atomic site and get all neighbor pairs
        for site_idx, site in enumerate(self.structure.sites):
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            for neighbor_pair in combinations(nn_info, 2):
                neighbor1, neighbor2 = neighbor_pair
                triangle_features = self._calculate_single_triangle_features(
                    site_idx, site, neighbor1, neighbor2
                )
                triangle_features_list.append(triangle_features)

        # 特征名称（22维） / Feature names (22 dimensions)
        feature_names = [
            'edge_length_1', 'edge_length_2', 'edge_length_3', 'triangle_area', 'triangle_perimeter',
            'shape_factor', 'rbf_area_small', 'rbf_area_medium', 'rbf_area_large', 'triangle_distortion_index',
            'angle_strain', 'coordination_type', 'tilt_gen_x', 'tilt_gen_y', 'tilt_gen_z',
            'casimir_C2', 'glazer_cont_param', 'mean_bond_angle_variance',
            'sectional_curvature_approx', 'ricci_curvature_approx', 'quotient_group_action_norm', 'periodic_boundary_curvature'
        ]
        return pd.DataFrame(triangle_features_list, columns=feature_names)

    def _calculate_single_triangle_features(self, center_site_idx, center_site, neighbor1, neighbor2):
        """
        计算单个三体相互作用的所有特征
        Calculate all features for a single triangle (2-simplex)

        参数/Param:
            center_site_idx (int): 中心原子索引 / Central atom index
            center_site (PeriodicSite): 中心原子对象 / Central atom object
            neighbor1 (dict): 第一个邻居信息 / First neighbor info
            neighbor2 (dict): 第二个邻居信息 / Second neighbor info

        返回/Return:
            list: 单个三体相互作用的22维特征向量 / 22-dimensional feature vector for one triangle
        """
        # 计算三角形三个顶点的坐标 / Get coordinates of the triangle vertices
        center_coords = center_site.coords
        neighbor1_coords = neighbor1['image'] @ self.structure.lattice.matrix + self.structure.sites[neighbor1['site_index']].coords
        neighbor2_coords = neighbor2['image'] @ self.structure.lattice.matrix + self.structure.sites[neighbor2['site_index']].coords

        # 计算三条边的长度 / Compute the three edge lengths
        edge1 = np.linalg.norm(neighbor1_coords - center_coords)
        edge2 = np.linalg.norm(neighbor2_coords - center_coords)
        edge3 = np.linalg.norm(neighbor2_coords - neighbor1_coords)
        edges = sorted([edge1, edge2, edge3])
        edge_length_1, edge_length_2, edge_length_3 = edges

        # 计算三角形面积（海伦公式）/ Compute triangle area (Heron's formula)
        s = (edge_length_1 + edge_length_2 + edge_length_3) / 2
        triangle_area = math.sqrt(max(0, s * (s - edge_length_1) * (s - edge_length_2) * (s - edge_length_3)))

        # 计算三角形周长 / Compute triangle perimeter
        triangle_perimeter = edge_length_1 + edge_length_2 + edge_length_3

        # 形状因子（等边三角形为1）/ Shape factor (1 for equilateral triangle)
        shape_factor = (12 * math.sqrt(3) * triangle_area) / (triangle_perimeter ** 2) if triangle_perimeter > 0 else 0.0

        # RBF径向基展开 / RBF expansion for area
        rbf_area_small = math.exp(-((triangle_area - self.rbf_area_centers[0]) ** 2) / (2 * self.rbf_area_width ** 2))
        rbf_area_medium = math.exp(-((triangle_area - self.rbf_area_centers[1]) ** 2) / (2 * self.rbf_area_width ** 2))
        rbf_area_large = math.exp(-((triangle_area - self.rbf_area_centers[2]) ** 2) / (2 * self.rbf_area_width ** 2))

        # 三角形畸变指数 / Triangle distortion index
        triangle_distortion_index = self._calculate_triangle_distortion_index(center_coords, neighbor1_coords, neighbor2_coords)

        # 角度应变 / Angle strain
        angle_strain = self._calculate_angle_strain(center_coords, neighbor1_coords, neighbor2_coords)

        # 配位类型编码 / Coordination type encoding
        coordination_type = self._calculate_coordination_type(center_site_idx)

        # 倾斜生成元（SO(3)李代数）/ Tilt generators (SO(3) Lie algebra)
        tilt_gen_x, tilt_gen_y, tilt_gen_z = self._calculate_tilt_generators(center_coords, neighbor1_coords, neighbor2_coords)

        # 二阶Casimir不变量 / Second-order Casimir invariant
        casimir_C2 = self._calculate_casimir_2_triangle(center_coords, neighbor1_coords, neighbor2_coords)

        # Glazer连续参数 / Glazer continuous parameter
        glazer_cont_param = self._calculate_glazer_continuous_parameter(tilt_gen_x, tilt_gen_y, tilt_gen_z)

        # 平均键角方差 / Mean bond angle variance
        mean_bond_angle_variance = self._calculate_bond_angle_variance(center_site_idx)

        # 曲率相关特征 / Curvature-related features
        sectional_curvature_approx = self._calculate_sectional_curvature_approx(center_coords, neighbor1_coords, neighbor2_coords)
        ricci_curvature_approx = self._calculate_ricci_curvature_approx(center_coords, neighbor1_coords, neighbor2_coords)
        quotient_group_action_norm = self._calculate_quotient_group_action_norm(center_coords, neighbor1_coords, neighbor2_coords)
        periodic_boundary_curvature = self._calculate_periodic_boundary_curvature(center_coords, neighbor1_coords, neighbor2_coords)

        return [
            edge_length_1, edge_length_2, edge_length_3, triangle_area, triangle_perimeter,
            shape_factor, rbf_area_small, rbf_area_medium, rbf_area_large, triangle_distortion_index,
            angle_strain, coordination_type, tilt_gen_x, tilt_gen_y, tilt_gen_z,
            casimir_C2, glazer_cont_param, mean_bond_angle_variance,
            sectional_curvature_approx, ricci_curvature_approx, quotient_group_action_norm, periodic_boundary_curvature
        ]

    def _calculate_triangle_distortion_index(self, center_coords, neighbor1_coords, neighbor2_coords):
        """
        计算三角形畸变指数
        Calculate triangle distortion index
        """
        # 边长标准差 / Edge length std
        distances = [
            np.linalg.norm(neighbor1_coords - center_coords),
            np.linalg.norm(neighbor2_coords - center_coords),
            np.linalg.norm(neighbor2_coords - neighbor1_coords)
        ]
        dist_std = np.std(distances) / np.mean(distances) if distances and np.mean(distances) > 0 else 0.0

        # 三角形内角 / Triangle angles
        vecs = [neighbor1_coords - center_coords, neighbor2_coords - center_coords, neighbor2_coords - neighbor1_coords]
        angles = []
        for i, j in [(0, 1), (2, 0), (1, 2)]:
            cos_ang = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-10)
            ang = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))
            angles.append(ang)

        # 理想角度统计 / Ideal angle statistics
        all_angles = []
        for idx in range(len(self.structure)):
            nn = self.crystal_nn.get_nn_info(self.structure, idx)[:5]
            for n1, n2 in combinations(nn[:3], 2):
                c = self.structure[idx].coords
                p1 = self.structure[n1['site_index']].coords + n1['image'] @ self.structure.lattice.matrix
                p2 = self.structure[n2['site_index']].coords + n2['image'] @ self.structure.lattice.matrix
                v1, v2 = p1 - c, p2 - c
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                all_angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        ideal_angle = np.mean(all_angles)

        angle_std = np.std([abs(ang - ideal_angle) for ang in angles]) / ideal_angle if angles else 0.0
        distortion = (dist_std + angle_std) / 2
        return float(min(1.0, max(0.0, distortion)))

    def _calculate_angle_strain(self, center_coords, neighbor1_coords, neighbor2_coords):
        """
        计算角度应变 - 偏离理想键角的程度
        Calculate angle strain (deviation from ideal bond angle)
        """
        # 计算键向量 / Compute bond vectors
        vec1 = neighbor1_coords - center_coords
        vec2 = neighbor2_coords - center_coords

        # 计算夹角 / Compute included angle
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = math.degrees(math.acos(cos_angle))

        # 统计全局理想角度 / Collect global ideal angles
        all_angles = []
        for idx in range(len(self.structure)):
            nn = self.crystal_nn.get_nn_info(self.structure, idx)
            for n1, n2 in combinations(nn, 2):
                c = self.structure[idx].coords
                p1 = self.structure[n1['site_index']].coords + n1['image'] @ self.structure.lattice.matrix
                p2 = self.structure[n2['site_index']].coords + n2['image'] @ self.structure.lattice.matrix
                v1, v2 = p1 - c, p2 - c
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                all_angles.append(math.degrees(math.acos(np.clip(cos, -1, 1))))
                if len(all_angles) > 100:
                    break
            if len(all_angles) > 100:
                break

        if not all_angles:
            all_dists = [np.linalg.norm(s.coords - self.structure[0].coords) for s in self.structure[1:]]
            return float(np.mean(all_dists) / 2.0)

        ideal_angle = np.mean(all_angles)
        angle_strain = abs(angle - ideal_angle)
        return float(angle_strain)

    def _calculate_coordination_type(self, site_idx):
        """
        计算配位类型编码
        Calculate coordination type encoding
        """
        try:
            nn_info = self.crystal_nn.get_nn_info(self.structure, site_idx)
            coordination_number = len(nn_info)
            # 配位数编码映射 / Coordination number mapping
            coordination_mapping = {
                1: 1,   # 单配位 / Monocoordinate
                2: 2,   # 双配位 / Dicoordinate
                3: 3,   # 三配位 / Tricoordinate
                4: 4,   # 四配位（四面体）/ Tetrahedral
                5: 5,   # 五配位 / Pentacoordinate
                6: 6,   # 六配位（八面体）/ Octahedral
                7: 7,   # 七配位 / Heptacoordinate
                8: 8,   # 八配位（立方体）/ Cubic
                9: 9,   # 九配位 / Nonacoordinate
                10: 10, # 十配位 / Decacoordinate
                11: 11, # 十一配位 / Undecacoordinate
                12: 12  # 十二配位（立方密堆积）/ Cubic close-packed
            }
            return coordination_mapping.get(coordination_number, 0)
        except Exception as e:
            print(f"配位类型计算警告: {e} / Coordination type calculation warning: {e}")
            return np.nan

    def _calculate_tilt_generators(self, center_coords, neighbor1_coords, neighbor2_coords):
        """
        计算倾斜生成元（SO(3)李代数）
        Calculate tilt generators (SO(3) Lie algebra)
        """
        try:
            # 计算三角形法向量 / Compute triangle normal vector
            vec1 = neighbor1_coords - center_coords
            vec2 = neighbor2_coords - center_coords
            normal = np.cross(vec1, vec2)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            # 构造SO(3)李代数生成元 / Construct SO(3) generator
            tilt_generator = self._construct_so3_generator(normal)
            # 提取三个轴的生成元分量 / Extract generator components
            tilt_gen_x = float(tilt_generator[2, 1])
            tilt_gen_y = float(tilt_generator[0, 2])
            tilt_gen_z = float(tilt_generator[1, 0])
            return tilt_gen_x, tilt_gen_y, tilt_gen_z
        except Exception as e:
            print(f"倾斜生成元计算警告: {e} / Tilt generator calculation warning: {e}")
            return np.nan, np.nan, np.nan

    def _construct_so3_generator(self, vector):
        """
        构造SO(3)李代数的生成元（反对称矩阵）
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

    def _calculate_casimir_2_triangle(self, center_coords, neighbor1_coords, neighbor2_coords):
        """
        计算三体相互作用的二阶Casimir不变量
        Calculate second-order Casimir invariant for triangle
        """
        try:
            tilt_gen_x, tilt_gen_y, tilt_gen_z = self._calculate_tilt_generators(center_coords, neighbor1_coords, neighbor2_coords)
            tilt_vector = np.array([tilt_gen_x, tilt_gen_y, tilt_gen_z])
            generator = self._construct_so3_generator(tilt_vector)
            casimir_2 = float(np.trace(generator @ generator))
            return casimir_2
        except Exception as e:
            print(f"Casimir不变量计算警告: {e} / Casimir invariant calculation warning: {e}")
            return np.nan

    def _calculate_glazer_continuous_parameter(self, tilt_gen_x, tilt_gen_y, tilt_gen_z):
        """
        计算Glazer连续参数 - 八面体倾斜的连续化描述
        Calculate Glazer continuous parameter (octahedral tilt)
        """
        tilt_magnitude = math.sqrt(tilt_gen_x**2 + tilt_gen_y**2 + tilt_gen_z**2)
        all_mags = []
        for i in range(len(self.structure)):
            nn = self.crystal_nn.get_nn_info(self.structure, i)[:2]
            if len(nn) < 2:
                continue
            c = self.structure[i].coords
            p1 = self.structure[nn[0]['site_index']].coords + nn[0]['image'] @ self.structure.lattice.matrix
            p2 = self.structure[nn[1]['site_index']].coords + nn[1]['image'] @ self.structure.lattice.matrix
            tx, ty, tz = self._calculate_tilt_generators(c, p1, p2)
            all_mags.append(math.sqrt(tx**2 + ty**2 + tz**2))
        max_tilt = max(all_mags) if all_mags else 1.0
        glazer_param = tilt_magnitude / max_tilt
        return float(glazer_param)

    def _calculate_bond_angle_variance(self, site_idx):
        """
        计算平均键角方差 - 配位几何的畸变程度
        Calculate mean bond angle variance (distortion of coordination geometry)
        """
        nn_info = self.nn_cache.get(site_idx, [])
        center_coords = self.structure.sites[site_idx].coords
        angles = []
        for i in range(len(nn_info)):
            for j in range(i + 1, len(nn_info)):
                neighbor1 = nn_info[i]
                neighbor2 = nn_info[j]
                neighbor1_coords = neighbor1['image'] @ self.structure.lattice.matrix + self.structure.sites[neighbor1['site_index']].coords
                neighbor2_coords = neighbor2['image'] @ self.structure.lattice.matrix + self.structure.sites[neighbor2['site_index']].coords
                vec1 = neighbor1_coords - center_coords
                vec2 = neighbor2_coords - center_coords
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        angle_variance = float(np.var(angles)) if len(angles) > 1 else 0.0
        return angle_variance

    def _calculate_sectional_curvature_approx(self, center, n1, n2):
        """
        近似计算截面曲率
        Approximate sectional curvature
        """
        try:
            points = gs.array([center, n1, n2])
            mean = gs.mean(points, axis=0)
            centered = points - mean
            cov = centered.T @ centered / 3
            eigvals = gs.linalg.eigvalsh(cov)
            return float(np.prod(eigvals[-2:]) / np.sum(eigvals[-2:]))
        except Exception as e:
            print(f"Sectional curvature approx warning: {e}")
            return np.nan

    def _calculate_ricci_curvature_approx(self, center, n1, n2):
        """
        近似计算Ricci曲率
        Approximate Ricci curvature
        """
        try:
            sphere = Hypersphere(dim=2)
            points = gs.array([center / np.linalg.norm(center+1e-10), n1 / np.linalg.norm(n1+1e-10), n2 / np.linalg.norm(n2+1e-10)])
            logs = [sphere.metric.log(p, points[0]) for p in points[1:]]
            return float(np.sum([sphere.metric.norm(log) for log in logs]))
        except Exception as e:
            print(f"Ricci curvature approx warning: {e}")
            return np.nan

    def _calculate_quotient_group_action_norm(self, center, n1, n2):
        """
        计算商群作用范数
        Calculate quotient group action norm
        """
        try:
            so3 = SpecialOrthogonal(n=3, point_type='vector')
            vec = (n1 - center) + (n2 - center)
            base = gs.array([0.0, 0.0, 0.0])
            log = so3.log(vec / np.linalg.norm(vec+1e-10), base)
            return float(so3.metric.norm(log))
        except Exception as e:
            print(f"Quotient group action norm warning: {e}")
            return np.nan

    def _calculate_periodic_boundary_curvature(self, center, n1, n2):
        """
        计算周期边界曲率
        Calculate periodic boundary curvature
        """
        try:
            points = gs.array([center, n1, n2])
            diffs = points[1:] - points[0]
            return float(np.linalg.norm(np.cross(diffs[0], diffs[1])) / (np.linalg.norm(diffs[0]) * np.linalg.norm(diffs[1])))
        except Exception as e:
            print(f"Periodic boundary curvature warning: {e}")
            return np.nan

    def save_triangle_features_to_csv(self, output_file):
        """
        计算并保存三体相互作用特征到CSV文件
        Compute and save triangle features to CSV

        参数/Param:
            output_file (str): 输出CSV文件路径 / Output CSV file path

        返回/Return:
            pd.DataFrame: 特征数据表 / Feature data table
        """
        triangle_features_df = self.calculate_triangle_features()
        triangle_features_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"三体相互作用特征已保存到: {output_file} / Triangle features saved to: {output_file}")
        return triangle_features_df

# ================== 使用示例 Example Usage ==================
if __name__ == "__main__":
    # CIF文件路径（请根据实际情况修改）/ CIF file path (please modify according to actual situation)
    cif_file = r"E:\RA\数据库搭建工作\Examples\CsPbBr3.cif"
    # 初始化特征计算器 / Initialize feature calculator
    calculator = PerovskiteTriangleFeatureCalculator(cif_file)
    # 输出文件路径 / Output file path
    output_file = "CsPbBr3_triangle_features_22d.csv"
    # 计算并保存特征 / Calculate and save features
    calculator.save_triangle_features_to_csv(output_file)