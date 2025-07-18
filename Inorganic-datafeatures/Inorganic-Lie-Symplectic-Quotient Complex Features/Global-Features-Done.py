"""
无机钙钛矿材料数据库特征计算器
Inorganic Perovskite Database Feature Calculator

李-辛-商复形特征 - 全局不变量特征（38维）
Lie-Symplectic-Quotient Complex Features - Global Invariant Features: 38 Dimensions
"""

import numpy as np
import pandas as pd
import math
import hashlib
import warnings
from typing import Dict, List, Optional
from mendeleev import element
from pymatgen.core import Structure, Element
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.linalg import eigvals
import scipy.constants as const  # 使用scipy获取物理常数 / Use scipy for physical constants
import geomstats.backend as gs # type: ignore
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.euclidean import Euclidean

# 忽略警告信息 / Suppress warnings
warnings.filterwarnings('ignore')

class PerovskiteGlobalFeatureCalculator:
    """
    标准无机钙钛矿38维全局特征计算器
    严格按照特征方案文档实现，使用化学库增强
    
    Standard Inorganic Perovskite 38-Dimensional Global Feature Calculator
    Strictly implemented according to feature specification document, enhanced with chemical libraries
    """
    
    def __init__(self, cif_file_path: str):
        """
        初始化全局特征计算器
        Initialize global feature calculator
        
        参数/Parameters:
            cif_file_path (str): CIF文件路径 / Path to CIF file
        """
        self.cif_path = cif_file_path
        self.structure = Structure.from_file(cif_file_path)
        self.primitive_structure = self.structure.get_primitive_structure()
        self.crystal_nn = CrystalNN()
        self.bv_analyzer = BVAnalyzer()
        # 识别钙钛矿晶位类型 / Identify perovskite site types
        self.site_assignments = self._identify_perovskite_sites_enhanced()
        
        # 打印基本信息 / Print basic information
        print(f"成功读取 CIF 文件: {cif_file_path} / Successfully loaded CIF file: {cif_file_path}")
        print(f"化学式: {self.structure.composition.reduced_formula} / Chemical formula: {self.structure.composition.reduced_formula}")
        print(f"晶格参数: a={self.structure.lattice.a:.3f}, b={self.structure.lattice.b:.3f}, c={self.structure.lattice.c:.3f} / Lattice parameters: a={self.structure.lattice.a:.3f}, b={self.structure.lattice.b:.3f}, c={self.structure.lattice.c:.3f}")
    
    def _identify_perovskite_sites_enhanced(self) -> Dict[str, List[int]]:
        """
        基于化学库增强的钙钛矿晶位识别
        Enhanced perovskite site identification based on chemical libraries
        
        返回/Returns:
            Dict[str, List[int]]: 各晶位类型的原子索引 / Atomic indices for each site type
        """
        sites = {'A': [], 'B': [], 'X': []}
        
        for i, site in enumerate(self.structure.sites):
            element_obj = Element(site.specie.symbol)
            
            if self._is_a_site_element(element_obj):
                sites['A'].append(i)
            elif self._is_b_site_element(element_obj):
                sites['B'].append(i)
            elif self._is_x_site_element(element_obj):
                sites['X'].append(i)
                
        return sites
    
    def _is_a_site_element(self, element: Element) -> bool:
        """
        判断是否为A位元素
        Determine if element belongs to A-site
        """
        # 碱金属和碱土金属 / Alkali and alkaline earth metals
        if element.is_alkali or element.is_alkaline:
            return True
            
        try:
            # 基于原子半径判断 / Based on atomic radius
            if hasattr(element, 'atomic_radius') and element.atomic_radius:
                if element.atomic_radius > 1.8:  # 大离子半径 / Large ionic radius
                    return True
                    
            # 基于族数判断 / Based on group number
            if element.group in [1, 2]:
                return True
                
            # 稀土元素 / Rare earth elements
            if element.symbol in ['La', 'Ce', 'Pr', 'Nd']:
                return True
                
        except Exception:
            pass
            
        return False
    
    def _is_b_site_element(self, element: Element) -> bool:
        """
        判断是否为B位元素
        Determine if element belongs to B-site
        """
        # 过渡金属 / Transition metals
        if element.is_transition_metal:
            return True
            
        # 主族金属元素 / Main group metals
        if element.group in [13, 14, 15, 16] and element.row >= 4:
            return True
            
        try:
            # 基于原子半径的精确判断 / Precise judgment based on atomic radius
            if hasattr(element, 'atomic_radius') and element.atomic_radius:
                if 0.5 < element.atomic_radius < 1.5:
                    if not self._is_a_site_element(element) and not self._is_x_site_element(element):
                        return True
        except Exception:
            pass
            
        return False
    
    def _is_x_site_element(self, element: Element) -> bool:
        """
        判断是否为X位元素（阴离子）
        Determine if element belongs to X-site (anion)
        """
        # 卤素 / Halogens
        if element.is_halogen:
            return True
            
        # 氧族元素 / Chalcogens
        if element.group == 16:
            return True
            
        try:
            # 基于电负性判断 / Based on electronegativity
            if hasattr(element, 'X') and element.X:
                if element.X > 2.5:  # 高电负性 / High electronegativity
                    return True
        except Exception:
            pass
            
        return False
    
    def _smart_oxidation_state_assignment(self, element_symbol: str, site_type: str) -> float:
        """
        智能氧化态分配
        Smart oxidation state assignment
        
        参数/Parameters:
            element_symbol (str): 元素符号 / Element symbol
            site_type (str): 晶位类型 / Site type
            
        返回/Returns:
            float: 氧化态 / Oxidation state
        """
        element_obj = Element(element_symbol)
        
        if site_type == 'A':
            # A位通常为阳离子 / A-site usually cations
            if element_obj.is_alkali:
                return 1.0
            elif element_obj.is_alkaline:
                return 2.0
            else:
                common_oxidation_states = element_obj.common_oxidation_states
                if common_oxidation_states:
                    positive_states = [state for state in common_oxidation_states if state > 0]
                    if positive_states:
                        return float(min(positive_states))
                return 2.0
                
        elif site_type == 'B':
            # B位金属离子 / B-site metal ions
            common_oxidation_states = element_obj.common_oxidation_states
            if common_oxidation_states:
                positive_states = [state for state in common_oxidation_states if state > 0]
                if positive_states:
                    # 优先选择常见氧化态 / Prefer common oxidation states
                    if 2 in positive_states:
                        return 2.0
                    elif 4 in positive_states:
                        return 4.0
                    else:
                        return float(min(positive_states))
            return 2.0
            
        elif site_type == 'X':
            # X位阴离子 / X-site anions
            if element_obj.is_halogen:
                return -1.0
            elif element_obj.group == 16:  # 氧族 / Chalcogens
                return -2.0
            else:
                common_oxidation_states = element_obj.common_oxidation_states
                if common_oxidation_states:
                    negative_states = [state for state in common_oxidation_states if state < 0]
                    if negative_states:
                        return float(max(negative_states))
                return -1.0
                
        return 0.0
    
    def calculate_global_features(self) -> Dict[str, float]:
        """
        计算全部38维全局特征
        Calculate all 38-dimensional global features
        
        返回/Returns:
            Dict[str, float]: 特征字典 / Feature dictionary
        """
        features = {}
        
        # 李群特征 (8个) / Lie Group Features (8)
        features['casimir_2_so3'] = self._calculate_casimir_2_so3()
        features['casimir_2_u1'] = self._calculate_casimir_2_u1()
        features['casimir_4_so3'] = self._calculate_casimir_4_so3()
        features['casimir_mixed'] = self._calculate_casimir_mixed()
        features['lie_dielectric_casimir'] = self._calculate_lie_dielectric_casimir()
        features['lie_polarization_casimir'] = self._calculate_lie_polarization_casimir()
        features['lie_energy_casimir'] = self._calculate_lie_energy_casimir()
        features['lie_group_commutator_norm'] = self._calculate_lie_group_commutator_norm()
        
        # 辛几何特征 (8个) / Symplectic Geometry Features (8)
        features['symplectic_casimir'] = self._calculate_symplectic_casimir()
        features['symplectic_gen_x'] = self._calculate_symplectic_gen_x()
        features['symplectic_gen_y'] = self._calculate_symplectic_gen_y()
        features['symplectic_weighted_casimir'] = self._calculate_symplectic_weighted_casimir()
        features['symplectic_dielectric_gen'] = self._calculate_symplectic_dielectric_gen()
        features['symplectic_absorption_gen'] = self._calculate_symplectic_absorption_gen()
        features['symplectic_form_norm'] = self._calculate_symplectic_form_norm()
        features['hamiltonian_flow_norm'] = self._calculate_hamiltonian_flow_norm()
        
        # 商空间特征 (8个) / Quotient Space Features (8)
        features['quotient_volume_metric'] = self._calculate_quotient_volume_metric()
        features['quotient_density_hash'] = self._calculate_quotient_density_hash()
        features['quotient_bartel_tau'] = self._calculate_quotient_bartel_tau()
        features['quotient_tau_prob'] = self._calculate_quotient_tau_prob()
        features['quotient_manifold_distance'] = self._calculate_quotient_manifold_distance()
        features['equivalence_class_entropy'] = self._calculate_equivalence_class_entropy()
        features['orbit_space_metric'] = self._calculate_orbit_space_metric()
        features['quotient_topology_invariant'] = self._calculate_quotient_topology_invariant()
        
        # 几何结构特征 (7个) / Geometric Structure Features (7)
        features['mean_bond_length'] = self._calculate_mean_bond_length()
        features['mean_tilt_angle'] = self._calculate_mean_tilt_angle()
        features['octahedral_count'] = self._calculate_octahedral_count()
        features['glazer_mode_ratio'] = self._calculate_glazer_mode_ratio()
        features['volume_per_fu'] = self._calculate_volume_per_fu()
        features['packing_fraction'] = self._calculate_packing_fraction()
        features['lattice_anisotropy_ratio'] = self._calculate_lattice_anisotropy_ratio()
        
        # 黎曼几何特征 (7个) / Riemannian Geometry Features (7)
        features['riemannian_distance_spd'] = self._calculate_riemannian_distance_spd()
        features['riemannian_variance'] = self._calculate_riemannian_variance()
        features['frechet_variance'] = self._calculate_frechet_variance()
        features['lie_algebra_bracket_norm'] = self._calculate_lie_algebra_bracket_norm()
        features['fundamental_domain_volume'] = self._calculate_fundamental_domain_volume()
        features['quotient_riemannian_volume'] = self._calculate_quotient_riemannian_volume()
        features['bond_valence_std'] = self._calculate_bond_valence_std()
        
        return features
    
    # ========== 李群特征计算方法 / Lie Group Feature Calculation Methods ==========
    
    def _calculate_casimir_2_so3(self) -> float:
        """
        计算SO(3)二次Casimir不变量
        Calculate SO(3) quadratic Casimir invariant
        """
        try:
            # 获取原子位置和质量 / Get atomic positions and masses
            positions = np.array([site.coords for site in self.structure.sites])
            masses = np.array([site.specie.atomic_mass for site in self.structure.sites])
            
            # 计算质心 / Calculate center of mass
            center_of_mass = np.average(positions, weights=masses, axis=0)
            rel_positions = positions - center_of_mass
            
            # 计算惯性张量 / Calculate inertia tensor
            I = np.zeros((3, 3))
            for r, m in zip(rel_positions, masses):
                r_sq = np.dot(r, r)
                I += m * (r_sq * np.eye(3) - np.outer(r, r))
            
            # 计算特征值并归一化 / Calculate eigenvalues and normalize
            eigenvals_ = np.abs(eigvals(I))
            eigenvals_ = eigenvals_[eigenvals_ > 1e-10]
            
            if len(eigenvals_) > 0:
                volume_scale = self.structure.volume**(1/3)
                mass_scale = np.mean(masses)
                casimir_value = np.sum(eigenvals_) / (mass_scale * volume_scale**2)
                return float(max(0.1, min(10.0, casimir_value)))
            else:
                # 备用计算方法 / Fallback calculation
                avg_mass = np.mean(masses)
                avg_dist = np.mean([np.linalg.norm(site.coords) for site in self.structure.sites])
                return float(avg_mass * avg_dist**2 / self.structure.volume)
                
        except Exception as e:
            print(f"SO(3)二次Casimir不变量计算出错: {e}")
            return np.nan
    
    def _calculate_casimir_2_u1(self) -> float:
        """
        计算U(1)二次Casimir不变量
        Calculate U(1) quadratic Casimir invariant
        """
        try:
            # 获取氧化态作为电荷 / Get oxidation states as charges
            charges = self._get_oxidation_states()
            if len(charges) == 0:
                return 0.1
                
            # 计算电荷相关量 / Calculate charge-related quantities
            charge_variance = np.var(charges)
            charge_asymmetry = abs(np.sum(charges))
            
            # 计算电荷分布的二阶矩 / Calculate second moment of charge distribution
            positions = np.array([site.coords for site in self.structure.sites])
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
            charge_moment_2nd = np.sum(charges**2 * distances**2)
            
            # 归一化处理 / Normalization
            volume_norm = self.structure.volume**(2/3)
            num_atoms = len(charges)
            casimir_u1 = (charge_variance + charge_moment_2nd/volume_norm + charge_asymmetry) / num_atoms
            
            return float(max(0.05, min(3.0, casimir_u1)))
            
        except Exception as e:
            print(f"U(1)二次Casimir不变量计算出错: {e}")
            return np.nan
    
    def _calculate_casimir_4_so3(self) -> float:
        """
        计算SO(3)四次Casimir不变量
        Calculate SO(3) quartic Casimir invariant
        """
        try:
            # 基于二次Casimir计算 / Based on quadratic Casimir
            casimir_2 = self._calculate_casimir_2_so3()
            casimir_4 = casimir_2**2
            return float(max(0.1, min(20.0, casimir_4)))
        except Exception:
            # 直接计算方法 / Direct calculation method
            positions = np.array([site.coords for site in self.structure.sites])
            I = np.zeros((3, 3))
            for r in positions:
                I += np.eye(3) * np.dot(r, r) - np.outer(r, r)
            return float(np.trace(I @ I @ I @ I)) / self.structure.volume**2
    
    def _calculate_casimir_mixed(self) -> float:
        """
        计算混合Casimir不变量
        Calculate mixed Casimir invariant
        """
        try:
            casimir_so3 = self._calculate_casimir_2_so3()
            casimir_u1 = self._calculate_casimir_2_u1()
            
            # 几何平均和耦合项 / Geometric mean and coupling term
            geometric_mean = np.sqrt(casimir_so3 * casimir_u1) if casimir_so3 > 0 and casimir_u1 > 0 else 0
            coupling_term = casimir_so3 * casimir_u1
            mixed_casimir = geometric_mean + coupling_term
            
            return float(max(1.0, min(100.0, mixed_casimir)))
        except Exception:
            return float(self._calculate_casimir_2_so3() * self._calculate_casimir_2_u1())
    
    def _calculate_lie_dielectric_casimir(self) -> float:
        """
        计算李介电Casimir不变量
        Calculate Lie dielectric Casimir invariant
        """
        try:
            charges = self._get_oxidation_states()
            positions = np.array([site.coords for site in self.structure.sites])
            
            # 计算偶极矩 / Calculate dipole moment
            dipole_moment = np.sum(charges[:, np.newaxis] * positions, axis=0)
            dipole_magnitude = np.linalg.norm(dipole_moment)
            
            # 归一化到体积 / Normalize to volume
            volume = self.structure.volume
            dielectric_casimir = dipole_magnitude / volume * 10
            
            return float(max(0.01, min(5.0, dielectric_casimir)))
            
        except Exception as e:
            print(f"李介电Casimir不变量计算出错: {e}")
            return np.nan
    
    def _calculate_lie_polarization_casimir(self) -> float:
        """
        计算李极化Casimir不变量
        Calculate Lie polarization Casimir invariant
        """
        try:
            charges = self._get_oxidation_states()
            positions = np.array([site.coords for site in self.structure.sites])
            
            # 计算极化张量 / Calculate polarization tensor
            polarization_tensor = np.zeros((3, 3))
            for i, charge in enumerate(charges):
                pos = positions[i]
                polarization_tensor += charge * np.outer(pos, pos)
            
            # 计算张量不变量 / Calculate tensor invariants
            trace = np.trace(polarization_tensor)
            det = np.linalg.det(polarization_tensor)
            polarization_casimir = trace / (1 + abs(det)**(1/3))
            
            # 体积归一化 / Volume normalization
            volume_norm = self.structure.volume**(2/3)
            normalized_value = polarization_casimir / volume_norm
            
            return float(max(0.01, min(10.0, normalized_value)))
            
        except Exception as e:
            print(f"李极化Casimir不变量计算出错: {e}")
            return np.nan
    
    def _calculate_lie_energy_casimir(self) -> float:
        """
        计算李能量Casimir不变量
        Calculate Lie energy Casimir invariant
        """
        try:
            charges = self._get_oxidation_states()
            positions = np.array([site.coords for site in self.structure.sites])
            
            # 计算库仑能 / Calculate Coulomb energy
            coulomb_energy = 0.0
            for i in range(len(charges)):
                for j in range(i+1, len(charges)):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance > 1e-6:  # 避免除零 / Avoid division by zero
                        coulomb_energy += charges[i] * charges[j] / distance
            
            # 归一化到体积尺度 / Normalize to volume scale
            volume_scale = self.structure.volume**(1/3)
            energy_casimir = abs(coulomb_energy) / volume_scale
            
            return float(max(0.1, min(50.0, energy_casimir)))
            
        except Exception as e:
            print(f"李能量Casimir不变量计算出错: {e}")
            return np.nan
    
    def _calculate_lie_group_commutator_norm(self) -> float:
        """
        计算李群对易子范数
        Calculate Lie group commutator norm
        """
        positions = np.array([site.coords for site in self.structure.sites])
        
        if len(positions) >= 2:
            # 选择两个位置向量 / Select two position vectors
            v1 = positions[0] - np.mean(positions, axis=0)
            v2 = positions[1] - np.mean(positions, axis=0)
            
            # 归一化向量 / Normalize vectors
            v1 = v1 / (np.linalg.norm(v1) + 1e-10)
            v2 = v2 / (np.linalg.norm(v2) + 1e-10)
            
            # 计算叉积（对应李代数对易子） / Calculate cross product (corresponding to Lie algebra commutator)
            so3 = SpecialOrthogonal(3, point_type='vector')
            commutator = gs.cross(v1, v2)
            commutator_norm = so3.metric.norm(commutator)
            
            return float(max(0.01, min(5.0, commutator_norm)))
        else:
            return 0.01
    
    # ========== 辛几何特征计算方法 / Symplectic Geometry Feature Calculation Methods ==========
    
    def _calculate_symplectic_casimir(self) -> float:
        """
        计算辛Casimir不变量
        Calculate symplectic Casimir invariant
        """
        try:
            positions = np.array([site.coords for site in self.structure.sites])
            velocities = self._get_thermal_velocities()
            
            # 构造相空间 / Construct phase space
            phase_space = np.concatenate([positions, velocities], axis=1)
            
            if phase_space.shape[1] >= 6:
                n = 3
                # 辛矩阵 / Symplectic matrix
                J = np.block([[np.zeros((n, n)), np.eye(n)], 
                             [-np.eye(n), np.zeros((n, n))]])
                
                # 相空间协方差 / Phase space covariance
                phase_cov = np.cov(phase_space.T)
                symplectic_det = np.abs(np.linalg.det(phase_cov @ J))
                
                # 归一化 / Normalization
                symplectic_casimir = symplectic_det / self.structure.volume
                return float(max(0.01, min(1.0, symplectic_casimir)))
            else:
                return 0.01
                
        except Exception:
            # 简化计算 / Simplified calculation
            positions = np.array([site.coords for site in self.structure.sites])
            cov = np.cov(positions.T)
            return float(np.abs(np.linalg.det(cov)) / self.structure.volume)
    
    def _calculate_symplectic_gen_x(self) -> float:
        """
        计算x方向辛生成元
        Calculate x-direction symplectic generator
        """
        try:
            positions = np.array([site.coords for site in self.structure.sites])
            charges = self._get_oxidation_states()
            
            # x方向电荷矩 / x-direction charge moment
            x_positions = positions[:, 0]
            charge_x_moment = np.sum(charges * x_positions)
            
            # 归一化到晶格参数 / Normalize to lattice parameter
            lattice_x = self.structure.lattice.a
            normalized_moment = charge_x_moment / lattice_x
            
            return float(max(0.001, min(1.0, abs(normalized_moment))))
            
        except Exception as e:
            print(f"x方向辛生成元计算出错: {e}")
            return np.nan
    
    def _calculate_symplectic_gen_y(self) -> float:
        """
        计算y方向辛生成元
        Calculate y-direction symplectic generator
        """
        try:
            positions = np.array([site.coords for site in self.structure.sites])
            charges = self._get_oxidation_states()
            
            # y方向电荷矩 / y-direction charge moment
            y_positions = positions[:, 1]
            charge_y_moment = np.sum(charges * y_positions)
            
            # 归一化到晶格参数 / Normalize to lattice parameter
            lattice_y = self.structure.lattice.b
            normalized_moment = charge_y_moment / lattice_y
            
            return float(max(0.001, min(1.0, abs(normalized_moment))))
            
        except Exception as e:
            print(f"y方向辛生成元计算出错: {e}")
            return np.nan
    
    def _calculate_symplectic_weighted_casimir(self) -> float:
        """
        计算有效质量加权辛Casimir不变量
        Calculate effective mass weighted symplectic Casimir invariant
        """
        try:
            masses = np.array([site.specie.atomic_mass for site in self.structure.sites])
            symplectic_basic = self._calculate_symplectic_casimir()
            
            # 质量方差修正因子 / Mass variance correction factor
            mass_variance = np.var(masses)
            mass_factor = 1.0 + 0.1 * mass_variance / np.mean(masses)**2
            
            weighted_casimir = symplectic_basic * mass_factor
            return float(max(0.01, min(2.0, weighted_casimir)))
            
        except Exception as e:
            print(f"有效质量加权辛Casimir计算出错: {e}")
            return np.nan
    
    def _calculate_symplectic_dielectric_gen(self) -> float:
        """
        计算辛介电生成元
        Calculate symplectic dielectric generator
        """
        try:
            charges = self._get_oxidation_states()
            positions = np.array([site.coords for site in self.structure.sites])
            
            # 构造极化向量 / Construct polarization vectors
            polarization_vectors = []
            for i, charge in enumerate(charges):
                if abs(charge) > 0.1:  # 忽略中性原子 / Ignore neutral atoms
                    position = positions[i]
                    dipole_moment = charge * position
                    polarization_vectors.append(dipole_moment)
            
            # 计算角动量型极化 / Calculate angular momentum type polarization
            total_angular_polarization = 0.0
            for i, pol_vec in enumerate(polarization_vectors):
                for other_pol in polarization_vectors[i+1:]:
                    cross_product = np.cross(pol_vec, other_pol)
                    total_angular_polarization += np.linalg.norm(cross_product)
            
            # 归一化 / Normalization
            volume_scale = self.structure.volume**(1/3)
            dielectric_gen = total_angular_polarization / (volume_scale * len(charges))
            
            return float(dielectric_gen)
            
        except Exception as e:
            print(f"辛介电生成元计算警告: {e}")
            return np.nan
    
    def _calculate_symplectic_absorption_gen(self) -> float:
        """
        计算辛吸收生成元
        Calculate symplectic absorption generator
        """
        try:
            charges = self._get_oxidation_states()
            positions = np.array([site.coords for site in self.structure.sites])
            
            # 电荷和位置的方差 / Variance of charges and positions
            charge_variance = np.var(charges)
            position_variance = np.var(np.linalg.norm(positions, axis=1))
            
            # 吸收测量 / Absorption measure
            absorption_measure = np.sqrt(charge_variance * position_variance)
            
            # 归一化 / Normalization
            volume_scale = self.structure.volume**(1/3)
            absorption_gen = absorption_measure / volume_scale
            
            return float(max(0.001, min(2.0, absorption_gen)))
            
        except Exception as e:
            print(f"辛吸收生成元计算出错: {e}")
            return np.nan
    
    def _calculate_symplectic_form_norm(self) -> float:
        """
        计算辛形式范数
        Calculate symplectic form norm
        """
        symplectic_casimir = self._calculate_symplectic_casimir()
        form_norm = np.sqrt(symplectic_casimir)
        return float(max(0.01, min(2.0, form_norm)))

    def _calculate_hamiltonian_flow_norm(self) -> float:
        """
        计算哈密顿流范数
        Calculate Hamiltonian flow norm
        """
        velocities = self._get_thermal_velocities()
        flow_magnitude = np.linalg.norm(np.mean(velocities, axis=0))
        normalized_flow = flow_magnitude / 1000.0  # 归一化因子 / Normalization factor
        return float(max(0.01, min(5.0, normalized_flow)))
    
    # ========== 商空间特征计算方法 / Quotient Space Feature Calculation Methods ==========
    
    def _calculate_quotient_volume_metric(self) -> float:
        """
        计算商体积度量
        Calculate quotient volume metric
        """
        try:
            volume = self.structure.volume
            primitive_volume = self.primitive_structure.volume
            
            # 体积比的对数 / Logarithm of volume ratio
            volume_ratio = volume / primitive_volume if primitive_volume > 1e-10 else 1.0
            quotient_metric = np.log(volume_ratio) if volume_ratio > 0 else 0.0
            
            return float(max(-5.0, min(5.0, quotient_metric)))
            
        except Exception as e:
            print(f"商体积度量计算出错: {e}")
            return np.nan
    
    def _calculate_quotient_density_hash(self) -> float:
        """
        计算商密度散列
        Calculate quotient density hash
        """
        try:
            density = self.structure.density
            density_str = f"{density:.6f}"
            
            # MD5散列转换为浮点数 / MD5 hash converted to float
            hash_value = int(hashlib.md5(density_str.encode()).hexdigest()[:8], 16)
            normalized_hash = (hash_value % 10000) / 10000.0
            
            return float(normalized_hash)
            
        except Exception as e:
            print(f"商密度散列计算出错: {e}")
            return np.nan
    
    def _calculate_quotient_bartel_tau(self) -> float:
        """
        计算商Bartel容忍因子
        Calculate quotient Bartel tolerance factor
        """
        try:
            # 初始化半径和氧化态 / Initialize radii and oxidation states
            A_radius = A_oxidation = None
            B_radius = None  
            X_radius = None
            
            # 遍历所有原子位点 / Iterate through all atomic sites
            for site in self.structure.sites:
                element_symbol = site.specie.symbol
                site_index = self.structure.sites.index(site)
                
                try:
                    # 确定晶位类型 / Determine site type
                    if site_index in self.site_assignments['A']:
                        site_type = 'A'
                        coordination = 12
                    elif site_index in self.site_assignments['B']:
                        site_type = 'B' 
                        coordination = 6
                    elif site_index in self.site_assignments['X']:
                        site_type = 'X'
                        coordination = 6
                    else:
                        continue
                    
                    # 获取氧化态和离子半径 / Get oxidation state and ionic radius
                    oxidation_state = self._smart_oxidation_state_assignment(element_symbol, site_type)
                    
                    if site_type == 'A':
                        ionic_radius = self._get_precise_ionic_radius(element_symbol, oxidation_state, 12)
                        if ionic_radius is not None and not np.isnan(ionic_radius):
                            if A_radius is None or ionic_radius > A_radius:
                                A_radius = ionic_radius
                                A_oxidation = abs(oxidation_state)
                                
                    elif site_type == 'B':
                        ionic_radius = self._get_b_site_ionic_radius(element_symbol, oxidation_state)
                        if ionic_radius is not None and not np.isnan(ionic_radius):
                            if B_radius is None or ionic_radius < B_radius:
                                B_radius = ionic_radius
                                
                    elif site_type == 'X':
                        ionic_radius = self._get_precise_ionic_radius(element_symbol, oxidation_state, 6)
                        if ionic_radius is not None and not np.isnan(ionic_radius):
                            if X_radius is None or ionic_radius > X_radius:
                                X_radius = ionic_radius
                                
                except Exception as e:
                    print(f"处理元素 {element_symbol} 时出错: {e}")
                    continue
            
            # 计算Bartel容忍因子 / Calculate Bartel tolerance factor
            r_A = A_radius
            r_B = B_radius  
            r_X = X_radius
            n_A = abs(A_oxidation)
            
            # Bartel公式的两个项 / Two terms of Bartel formula
            term1 = r_X / r_B
            
            ratio = r_A / r_B
            if ratio <= 1.0:
                ln_ratio = math.log(ratio + 1e-10)
            else:
                ln_ratio = math.log(ratio)
                
            if abs(ln_ratio) < 1e-10:
                ln_term = ratio
            else:
                ln_term = ratio / ln_ratio
                
            term2 = -n_A * (n_A - ln_term)
            
            tau = term1 + term2
            return float(tau)
            
        except Exception as e:
            # 备用计算 / Fallback calculation
            avg_r = np.mean([self._get_precise_ionic_radius(s.specie.symbol, 0) for s in self.structure.sites])
            return float(avg_r) if avg_r > 0 else np.nan
    
    def _calculate_quotient_tau_prob(self) -> float:
        """
        计算商稳定概率
        Calculate quotient stability probability
        """
        try:
            tau = self._calculate_quotient_bartel_tau()
            # Bartel公式中的概率函数 / Probability function in Bartel formula
            probability = 1.0 / (1.0 + np.exp(0.5 * (tau - 4.18)))
            return float(max(0.0, min(1.0, probability)))
            
        except Exception as e:
            print(f"商稳定概率计算出错: {e}")
            return np.nan

    def _calculate_quotient_manifold_distance(self) -> float:
        """
        计算商流形距离
        Calculate quotient manifold distance
        """
        original_volume = self.structure.volume
        primitive_volume = self.primitive_structure.volume
        volume_ratio = original_volume / primitive_volume if primitive_volume > 1e-10 else 1.0
        manifold_distance = abs(np.log(volume_ratio))
        return float(max(0.01, min(10.0, manifold_distance)))

    def _calculate_equivalence_class_entropy(self) -> float:
        """
        计算等价类熵
        Calculate equivalence class entropy
        """
        sga = SpacegroupAnalyzer(self.structure)
        symmetry_ops = sga.get_symmetry_operations()
        num_ops = len(symmetry_ops)
        entropy = np.log(num_ops) if num_ops > 1 else 0.0
        return float(max(0.01, min(5.0, entropy)))

    def _calculate_orbit_space_metric(self) -> float:
        """
        计算轨道空间度量
        Calculate orbit space metric
        """
        positions = np.array([site.coords for site in self.structure.sites])
        
        # 找到唯一位置（考虑对称性） / Find unique positions (considering symmetry)
        unique_positions = []
        for pos in positions:
            is_new = True
            for unique_pos in unique_positions:
                if np.linalg.norm(pos - unique_pos) < 0.1:  # 容差 / Tolerance
                    is_new = False
                    break
            if is_new:
                unique_positions.append(pos)
        
        orbit_metric = len(unique_positions) / len(positions)
        return float(max(0.1, min(1.0, orbit_metric)))

    def _calculate_quotient_topology_invariant(self) -> float:
        """
        计算商拓扑不变量
        Calculate quotient topology invariant
        """
        num_atoms = len(self.structure.sites)
        num_bonds = 0
        
        # 计算键数 / Calculate number of bonds
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                distance = self.structure.get_distance(i, j)
                if 1.0 < distance < 4.0:  # 合理的键长范围 / Reasonable bond length range
                    num_bonds += 1
        
        # 拓扑不变量 / Topology invariant
        topology_invariant = (num_atoms - num_bonds) / num_atoms if num_atoms > 0 else 0
        return float(max(-1.0, min(1.0, topology_invariant)))
    
    # ========== 几何结构特征计算方法 / Geometric Structure Feature Calculation Methods ==========
    
    def _calculate_mean_bond_length(self) -> float:
        """
        计算平均键长
        Calculate mean bond length
        """
        try:
            bond_lengths = []
            
            # 遍历所有原子对 / Iterate through all atom pairs
            for i in range(len(self.structure)):
                for j in range(i+1, len(self.structure)):
                    distance = self.structure.get_distance(i, j)
                    # 只考虑合理的键长 / Only consider reasonable bond lengths
                    if 1.0 < distance < 6.0:
                        bond_lengths.append(distance)
            
            if bond_lengths:
                return float(np.mean(bond_lengths))
            else:
                return 3.0  # 默认值 / Default value
                
        except Exception as e:
            print(f"平均键长计算出错: {e}")
            return np.nan
    
    def _calculate_mean_tilt_angle(self) -> float:
        """
        计算平均倾斜角
        Calculate mean tilt angle
        """
        try:
            # 晶格参数和角度 / Lattice parameters and angles
            a, b, c = self.structure.lattice.abc
            alpha, beta, gamma = self.structure.lattice.angles
            
            # 角度偏离度 / Angular deviation
            angular_deviation = (abs(alpha - 90) + abs(beta - 90) + abs(gamma - 90)) / 3.0
            
            # 晶格各向异性因子 / Lattice anisotropy factor
            max_param = max(a, b, c)
            min_param = min(a, b, c)
            anisotropy_factor = (max_param - min_param) / max_param * 10.0
            
            # 原子位移方差 / Atomic displacement variance
            displacement_variance = 0.0
            ideal_positions = {
                'A': [0, 0, 0], 
                'B': [0.5, 0.5, 0.5], 
                'X': [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
            }
            
            # 计算X原子的位移 / Calculate X atom displacement
            for site_type, indices in self.site_assignments.items():
                if site_type == 'X' and len(indices) >= 3:
                    actual_positions = [self.structure.sites[i].frac_coords for i in indices[:3]]
                    ideal_x_positions = ideal_positions['X']
                    
                    for actual, ideal in zip(actual_positions, ideal_x_positions):
                        displacement = np.linalg.norm(np.array(actual) - np.array(ideal))
                        displacement_variance += displacement
                    displacement_variance /= 3.0
            
            # 总倾斜因子 / Total tilt factor
            total_tilt_factor = angular_deviation + anisotropy_factor + displacement_variance
            tilt_angle = np.radians(total_tilt_factor)
            
            return float(tilt_angle)
            
        except Exception as e:
            print(f"平均倾斜角计算出错: {e}")
            return np.nan
    
    def _calculate_octahedral_count(self) -> float:
        """
        计算八面体数量
        Calculate octahedral count
        """
        try:
            octahedral_count = 0
            
            # 遍历B位原子 / Iterate through B-site atoms
            for i in self.site_assignments['B']:
                try:
                    neighbors = self.crystal_nn.get_nn_info(self.structure, i)
                    if len(neighbors) == 6:  # 完美八面体 / Perfect octahedron
                        octahedral_count += 1
                    elif len(neighbors) >= 4:  # 部分八面体 / Partial octahedron
                        octahedral_count += 0.5
                except Exception:
                    continue
                    
            return float(octahedral_count)
            
        except Exception as e:
            print(f"八面体数量计算出错: {e}")
            return np.nan
    
    def _calculate_glazer_mode_ratio(self) -> float:
        """
        计算Glazer模式占比
        Calculate Glazer mode ratio
        """
        try:
            tilt_angle = self._calculate_mean_tilt_angle()
            a, b, c = self.structure.lattice.abc
            
            # 平均晶格参数 / Average lattice parameter
            average_lattice = (a + b + c) / 3.0
            
            # 各方向应变 / Strain in each direction
            strain_a = abs(a - average_lattice) / average_lattice
            strain_b = abs(b - average_lattice) / average_lattice  
            strain_c = abs(c - average_lattice) / average_lattice
            
            max_strain = max(strain_a, strain_b, strain_c)
            strain_variance = np.var([strain_a, strain_b, strain_c])
            
            # 根据应变和倾斜角判断Glazer模式 / Determine Glazer mode based on strain and tilt
            if max_strain < 0.001 and tilt_angle < 0.01:
                glazer_ratio = 0.0  # 立方相 / Cubic phase
            elif max_strain < 0.01 and tilt_angle < 0.05:
                glazer_ratio = 0.1 + tilt_angle * 5.0  # 轻微倾斜 / Slight tilting
            elif strain_variance < 0.0001:
                glazer_ratio = 0.3 + min(tilt_angle * 10.0, 0.4)  # 均匀应变 / Uniform strain
            else:
                anisotropy_factor = strain_variance * 1000.0
                glazer_ratio = 0.5 + min(anisotropy_factor + tilt_angle * 5.0, 0.5)  # 复杂倾斜 / Complex tilting
            
            # 最小值限制 / Minimum value constraint
            if glazer_ratio < 0.05:
                glazer_ratio = 0.05
                
            return float(max(0.0, min(1.0, glazer_ratio)))
            
        except Exception as e:
            print(f"Glazer模式占比计算出错: {e}")
            return np.nan
    
    def _calculate_volume_per_fu(self) -> float:
        """
        计算每化学式单元体积
        Calculate volume per formula unit
        """
        try:
            volume = self.structure.volume
            
            # 统计各晶位原子数 / Count atoms in each site
            n_A = len(self.site_assignments['A'])
            n_B = len(self.site_assignments['B']) 
            n_X = len(self.site_assignments['X'])
            
            # 估算化学式单元数 / Estimate number of formula units
            if n_A > 0 and n_B > 0 and n_X > 0:
                if n_X >= 3 * n_B:  # 标准钙钛矿 / Standard perovskite
                    atoms_per_fu = n_A + n_B + 3 * n_B
                else:
                    atoms_per_fu = n_A + n_B + n_X
                num_formula_units = len(self.structure.sites) / atoms_per_fu
            else:
                num_formula_units = 1
            
            volume_per_fu = volume / num_formula_units if num_formula_units > 0 else volume
            return float(max(10.0, min(1000.0, volume_per_fu)))
            
        except Exception as e:
            print(f"每化学式单元体积计算出错: {e}")
            return np.nan
    
    def _calculate_packing_fraction(self) -> float:
        """
        计算堆积分数
        Calculate packing fraction
        """
        total_atomic_volume = 0.0
        
        # 计算所有原子的总体积 / Calculate total volume of all atoms
        for site in self.structure.sites:
            element_symbol = site.specie.symbol
            site_index = self.structure.sites.index(site)
            
            # 确定晶位类型和配位数 / Determine site type and coordination
            if site_index in self.site_assignments['A']:
                site_type = 'A'
                coordination = 12
            elif site_index in self.site_assignments['B']:
                site_type = 'B'
                coordination = 6
            elif site_index in self.site_assignments['X']:
                site_type = 'X'
                coordination = 6
            else:
                site_type = 'unknown'
                coordination = 6
            
            # 获取离子半径并计算体积 / Get ionic radius and calculate volume
            oxidation_state = self._smart_oxidation_state_assignment(element_symbol, site_type)
            radius = self._get_precise_ionic_radius(element_symbol, oxidation_state, coordination)
            atomic_volume = (4/3) * np.pi * radius**3
            total_atomic_volume += atomic_volume
        
        # 计算堆积分数 / Calculate packing fraction
        cell_volume = self.structure.volume
        packing_fraction = total_atomic_volume / cell_volume
        
        # 处理异常值 / Handle exceptional values
        if np.isnan(packing_fraction):
            return np.mean([e.atomic_radius**3 for e in Element if e.atomic_radius]) / self.structure.volume
            
        return float(packing_fraction)

    def _calculate_lattice_anisotropy_ratio(self) -> float:
        """
        计算晶格各向异性比
        Calculate lattice anisotropy ratio
        """
        try:
            a, b, c = self.structure.lattice.abc
            max_param = max(a, b, c)
            min_param = min(a, b, c)
            anisotropy_ratio = max_param / min_param if min_param > 1e-10 else 1.0
            return float(max(1.0, min(10.0, anisotropy_ratio)))
            
        except Exception as e:
            print(f"晶格各向异性比计算出错: {e}")
            return np.nan
    
    # ========== 黎曼几何特征计算方法 / Riemannian Geometry Feature Calculation Methods ==========
    
    def _calculate_riemannian_distance_spd(self) -> float:
        """
        计算SPD流形上的黎曼距离
        Calculate Riemannian distance on SPD manifold
        """
        # 晶格矩阵构造SPD矩阵 / Construct SPD matrix from lattice matrix
        lattice_matrix = self.structure.lattice.matrix
        spd_matrix = lattice_matrix @ lattice_matrix.T
        identity = np.eye(3)
        
        # 计算SPD流形上的距离 / Calculate distance on SPD manifold
        spd_manifold = SPDMatrices(3)
        distance = spd_manifold.metric.dist(identity, spd_matrix)
        
        return float(max(0.1, min(10.0, distance)))

    def _calculate_riemannian_variance(self) -> float:
        """
        计算黎曼方差
        Calculate Riemannian variance
        """
        positions = np.array([site.coords for site in self.structure.sites])
        euclidean = Euclidean(3)
        
        # 计算平均位置 / Calculate mean position
        mean_position = np.mean(positions, axis=0)
        
        # 计算方差 / Calculate variance
        variance = 0.0
        for pos in positions:
            distance = euclidean.metric.dist(mean_position, pos)
            variance += distance**2
        variance /= len(positions)
        
        return float(max(0.01, min(100.0, variance)))

    def _calculate_frechet_variance(self) -> float:
        """
        计算Fréchet方差
        Calculate Fréchet variance
        """
        positions = np.array([site.coords for site in self.structure.sites])
        euclidean = Euclidean(3)
        
        # Fréchet平均 / Fréchet mean
        frechet_mean = np.mean(positions, axis=0)
        
        # Fréchet方差 / Fréchet variance
        frechet_var = 0.0
        for pos in positions:
            dist = euclidean.metric.dist(frechet_mean, pos)
            frechet_var += dist**2
        frechet_var /= len(positions)
        
        return float(max(0.01, min(50.0, frechet_var)))

    def _calculate_lie_algebra_bracket_norm(self) -> float:
        """
        计算李代数括号范数
        Calculate Lie algebra bracket norm
        """
        commutator_norm = self._calculate_lie_group_commutator_norm()
        bracket_norm = commutator_norm * 0.8  # 李代数与李群的关系 / Relation between Lie algebra and Lie group
        return float(max(0.01, min(3.0, bracket_norm)))

    def _calculate_fundamental_domain_volume(self) -> float:
        """
        计算基本域体积
        Calculate fundamental domain volume
        """
        primitive_volume = self.primitive_structure.volume
        normalized_volume = np.log(primitive_volume) if primitive_volume > 1 else 0
        return float(max(1.0, min(1000.0, normalized_volume)))

    def _calculate_quotient_riemannian_volume(self) -> float:
        """
        计算商黎曼体积
        Calculate quotient Riemannian volume
        """
        lattice_matrix = self.structure.lattice.matrix
        riemannian_volume = abs(np.linalg.det(lattice_matrix))
        normalized_volume = np.log(riemannian_volume) if riemannian_volume > 1 else 0
        return float(max(1.0, min(500.0, normalized_volume)))
    
    def _calculate_bond_valence_std(self) -> float:
        """
        计算键价标准差
        Calculate bond valence standard deviation
        """
        try:
            if self.bv_analyzer:
                valences = self.bv_analyzer.get_valences(self.structure)
                bond_valence_std = np.std(valences)
                return float(max(0.01, min(3.0, bond_valence_std)))
            else:
                # 备用方法 / Fallback method
                oxidation_states = self._get_oxidation_states()
                return float(np.std(oxidation_states))
                
        except Exception as e:
            print(f"键价标准差计算出错: {e}")
            return np.nan
    
    # ========== 辅助方法 / Helper Methods ==========
    
    def _get_oxidation_states(self) -> np.ndarray:
        """
        获取氧化态数组
        Get oxidation states array
        """
        return np.array(self.bv_analyzer.get_valences(self.structure))
        
    def _get_thermal_velocities(self) -> np.ndarray:
        """
        计算热运动速度
        Calculate thermal velocities
        """
        # 原子质量（kg） / Atomic masses (kg)
        masses = np.array([site.specie.atomic_mass for site in self.structure.sites])
        masses_kg = masses * const.atomic_mass
        
        # 热运动速度计算 / Thermal velocity calculation
        k_B = const.Boltzmann
        T = 300.0  # 室温 / Room temperature
        thermal_speeds = np.sqrt(3 * k_B * T / masses_kg)
        
        # 生成随机方向的速度向量 / Generate velocity vectors with random directions
        np.random.seed(42)  # 确保可重现性 / Ensure reproducibility
        velocities = np.zeros((len(self.structure), 3))
        for i, speed in enumerate(thermal_speeds):
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            velocities[i] = speed * direction
            
        return velocities

    def _get_precise_ionic_radius(self, element_symbol: str, oxidation_state: float, coordination: int = 6) -> Optional[float]:
        """
        获取精确的离子半径
        Get precise ionic radius
        
        参数/Parameters:
            element_symbol (str): 元素符号 / Element symbol
            oxidation_state (float): 氧化态 / Oxidation state
            coordination (int): 配位数 / Coordination number
            
        返回/Returns:
            Optional[float]: 离子半径（Å） / Ionic radius (Å)
        """
        if oxidation_state == 0:
            return None
        return self._get_mendeleev_ionic_radius(element_symbol, oxidation_state, coordination)

    def _get_mendeleev_ionic_radius(self, element_symbol: str, oxidation_state: float, coordination: int = 6) -> Optional[float]:
        """
        从mendeleev库获取离子半径
        Get ionic radius from mendeleev library
        """
        elem = element(element_symbol)
        
        # 检查是否有离子半径数据 / Check if ionic radius data exists
        if not hasattr(elem, 'ionic_radii') or not elem.ionic_radii:
            return None
        
        # 配位数映射 / Coordination number mapping
        coord_mapping = {
            4: ['IV', 'IVSQ', 'IVPY'],
            5: ['V'],
            6: ['VI'],
            7: ['VII'],
            8: ['VIII'],
            9: ['IX'],
            10: ['X'],
            11: ['XI'],
            12: ['XII'],
        }
        
        target_coord_strs = coord_mapping.get(coordination, [])
        
        # 寻找匹配的离子半径 / Search for matching ionic radius
        for ionic_radius_record in elem.ionic_radii:
            if (ionic_radius_record.charge == int(oxidation_state) and 
                ionic_radius_record.coordination in target_coord_strs):
                radius = ionic_radius_record.ionic_radius
                if radius is not None:
                    radius_angstrom = float(radius) / 100.0  # pm转Å / Convert pm to Å
                    return radius_angstrom
        
        # 尝试其他配位数 / Try other coordination numbers
        priority_coords = [6, 8, 12, 4, 5, 7, 9, 10]
        for coord in priority_coords:
            if coord == coordination:
                continue
            alt_coord_strs = coord_mapping.get(coord, [])
            for ionic_radius_record in elem.ionic_radii:
                if (ionic_radius_record.charge == int(oxidation_state) and 
                    ionic_radius_record.coordination in alt_coord_strs):
                    radius = ionic_radius_record.ionic_radius
                    if radius is not None:
                        radius_angstrom = float(radius) / 100.0
                        print(f"使用{coord}配位代替{coordination}配位: {element_symbol}{oxidation_state:+.0f} = {radius_angstrom:.3f}Å")
                        return radius_angstrom
        
        return None

    def _get_b_site_ionic_radius(self, element_symbol: str, oxidation_state: float) -> float:
        """
        获取B位离子半径
        Get B-site ionic radius
        """
        radius = self._get_precise_ionic_radius(element_symbol, oxidation_state, 6)
        if radius is not None:
            return radius
        print(f"未找到B位离子半径数据: {element_symbol}{oxidation_state:+.0f}")
        return np.nan

    def save_global_features_to_csv(self, output_file: str) -> pd.DataFrame:
        """
        计算并保存全局特征到CSV文件
        Calculate and save global features to CSV file
        
        参数/Parameters:
            output_file (str): 输出CSV文件路径 / Output CSV file path
            
        返回/Returns:
            pd.DataFrame: 特征数据表 / Feature data table
        """
        # 计算特征 / Calculate features
        global_features = self.calculate_global_features()
        
        # 转换为DataFrame / Convert to DataFrame
        global_features_df = pd.DataFrame([global_features])
        
        # 保存到CSV / Save to CSV
        global_features_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"全局特征已保存到: {output_file} / Global features saved to: {output_file}")
        
        return global_features_df
    
if __name__ == "__main__":
    # 测试用例 / Test case
    cif_file = "E:\RA\数据库搭建工作\Examples\CsPbBr3.cif"
    
    # 初始化特征计算器 / Initialize feature calculator
    calculator = PerovskiteGlobalFeatureCalculator(cif_file)
    
    # 输出文件路径 / Output file path
    output_file = "CsPbBr3_global_features_38d.csv"
    
    # 计算并保存特征 / Calculate and save features
    calculator.save_global_features_to_csv(output_file)