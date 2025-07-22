from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import pandas as pd
import os


class XRDFeatureCalculator:
    """
    X射线衍射(XRD)特征计算器
    XRD Feature Calculator for crystalline materials
    
    该类从CIF文件计算XRD衍射图谱特征，包括峰位、强度、相信度等
    This class calculates XRD diffraction pattern features from CIF files,
    including peak positions, intensities, and phase confidence metrics.
    """
    
    def __init__(self, wavelength='CuKa'):
        """
        初始化XRD计算器
        Initialize XRD calculator
        
        Parameters:
        -----------
        wavelength : str, default='CuKa'
            X射线波长，默认使用铜靶Ka线
            X-ray wavelength, default uses Cu Ka radiation
        """
        self.calculator = XRDCalculator(wavelength=wavelength)
    
    def _extract_structure_info(self, cif_path):
        """
        从CIF文件提取晶体结构信息
        Extract crystal structure information from CIF file
        
        Parameters:
        -----------
        cif_path : str
            CIF文件路径 / Path to CIF file
            
        Returns:
        --------
        tuple
            (原始结构, 常规标准结构) / (original structure, conventional standard structure)
        """
        try:
            # 读取CIF文件中的晶体结构 / Read crystal structure from CIF file
            structure = Structure.from_file(cif_path)
            
            # 获取空间群分析器 / Get space group analyzer
            sga = SpacegroupAnalyzer(structure)
            
            # 获取常规标准结构，用于XRD计算 / Get conventional standard structure for XRD calculation
            conventional_structure = sga.get_conventional_standard_structure()
            
            return structure, conventional_structure
        except Exception as e:
            raise ValueError(f"无法读取CIF文件 / Failed to read CIF file: {cif_path}, 错误 / Error: {e}")
    
    def _calculate_xrd_pattern(self, structure):
        """
        计算XRD衍射图谱
        Calculate XRD diffraction pattern
        
        Parameters:
        -----------
        structure : Structure
            pymatgen Structure对象 / pymatgen Structure object
            
        Returns:
        --------
        tuple
            (2θ角度数组, 强度数组) / (2theta angles array, intensities array)
        """
        try:
            # 计算XRD衍射图谱 / Calculate XRD diffraction pattern
            pattern = self.calculator.get_pattern(structure)
            
            # 提取2θ角度和强度数据 / Extract 2theta angles and intensity data
            two_theta = pattern.x
            intensities = pattern.y
            
            return two_theta, intensities
        except Exception as e:
            raise ValueError(f"XRD图谱计算失败 / XRD pattern calculation failed: {e}")
    
    def _extract_peak_features(self, two_theta, intensities, top_n=5):
        """
        提取XRD峰特征
        Extract XRD peak features
        
        Parameters:
        -----------
        two_theta : array
            2θ角度数组 / 2theta angles array
        intensities : array
            强度数组 / Intensities array
        top_n : int, default=5
            提取前N个最强峰 / Extract top N strongest peaks
            
        Returns:
        --------
        dict
            峰特征字典 / Peak features dictionary
        """
        features = {}
        
        # 按强度排序，获取前N个最强峰 / Sort by intensity to get top N strongest peaks
        sorted_indices = np.argsort(intensities)[::-1]
        top_n_indices = sorted_indices[:top_n]
        
        top_two_theta = two_theta[top_n_indices]
        top_intensities = intensities[top_n_indices]
        
        # 提取前5个峰的2θ角度 / Extract 2theta angles of top 5 peaks
        for i in range(top_n):
            if i < len(top_two_theta):
                features[f'XRD_Peaks_{i+1}_2theta'] = top_two_theta[i]
            else:
                features[f'XRD_Peaks_{i+1}_2theta'] = 0.0
        
        # 归一化强度（除第一个峰外） / Normalize intensities (except the first peak)
        # 第一个峰强度固定为1.0，其他峰相对于第一个峰归一化
        # First peak intensity is fixed at 1.0, others normalized relative to first peak
        intensity_min = np.min(intensities)
        intensity_max = np.max(intensities)
        intensity_range = intensity_max - intensity_min
        
        for i in range(1, top_n):  # 从第2个峰开始 / Start from 2nd peak
            if i < len(top_intensities) and intensity_range > 0:
                normalized_intensity = (top_intensities[i] - intensity_min) / intensity_range
                features[f'XRD_Peaks_{i+1}_intensity'] = normalized_intensity
            else:
                features[f'XRD_Peaks_{i+1}_intensity'] = 0.0
        
        return features
    
    def _calculate_statistical_features(self, two_theta, intensities):
        """
        计算XRD图谱的统计特征
        Calculate statistical features of XRD pattern
        
        Parameters:
        -----------
        two_theta : array
            2θ角度数组 / 2theta angles array
        intensities : array
            强度数组 / Intensities array
            
        Returns:
        --------
        dict
            统计特征字典 / Statistical features dictionary
        """
        features = {}
        
        # 峰数量 / Number of peaks
        features['XRD_Peak_Count'] = len(two_theta)
        
        # 平均峰间距 / Average peak spacing
        if len(two_theta) > 1:
            sorted_two_theta = np.sort(two_theta)
            peak_spacings = np.diff(sorted_two_theta)
            features['XRD_Average_Peak_Spacing'] = np.mean(peak_spacings)
        else:
            features['XRD_Average_Peak_Spacing'] = 0.0
        
        # 强度标准差 / Intensity standard deviation
        features['XRD_Intensity_Std'] = np.std(intensities)
        
        return features
    
    def _calculate_entropy_features(self, intensities):
        """
        计算XRD图谱的熵特征和相信度
        Calculate entropy features and phase confidence of XRD pattern
        
        Parameters:
        -----------
        intensities : array
            强度数组 / Intensities array
            
        Returns:
        --------
        dict
            熵特征字典 / Entropy features dictionary
        """
        features = {}
        
        # 计算强度分布的信息熵 / Calculate information entropy of intensity distribution
        # 将强度转换为概率分布 / Convert intensities to probability distribution
        probs = intensities / np.sum(intensities)
        probs = probs[probs > 0]  # 移除零概率项 / Remove zero probability terms
        
        # 香农熵计算 / Shannon entropy calculation
        entropy = -np.sum(probs * np.log(probs))
        features['XRD_Intensity_Entropy'] = entropy
        
        # 相信度计算 / Phase confidence calculation
        # 基于熵的相信度：熵越低，相信度越高 / Entropy-based confidence: lower entropy, higher confidence
        max_entropy = np.log(len(intensities))  # 最大可能熵 / Maximum possible entropy
        if max_entropy > 0:
            confidence = 1 - (entropy / max_entropy)
            features['XRD_Phase_Confidence'] = confidence
        else:
            features['XRD_Phase_Confidence'] = 1.0
        
        return features
    
    def calculate_features_from_cif(self, cif_path):
        """
        从CIF文件计算完整的XRD特征
        Calculate complete XRD features from CIF file
        
        Parameters:
        -----------
        cif_path : str
            CIF文件路径 / Path to CIF file
            
        Returns:
        --------
        pd.DataFrame
            包含所有XRD特征的数据框 / DataFrame containing all XRD features
        """
        # 1. 提取晶体结构信息 / Extract crystal structure information
        original_structure, conventional_structure = self._extract_structure_info(cif_path)
        
        # 2. 计算XRD衍射图谱 / Calculate XRD diffraction pattern
        two_theta, intensities = self._calculate_xrd_pattern(conventional_structure)
        
        # 3. 提取各类特征 / Extract various features
        features = {}
        
        # 峰特征 / Peak features
        peak_features = self._extract_peak_features(two_theta, intensities)
        features.update(peak_features)
        
        # 统计特征 / Statistical features
        statistical_features = self._calculate_statistical_features(two_theta, intensities)
        features.update(statistical_features)
        
        # 熵特征和相信度 / Entropy features and phase confidence
        entropy_features = self._calculate_entropy_features(intensities)
        features.update(entropy_features)
        
        # 4. 创建数据框 / Create DataFrame
        df = pd.DataFrame([features])
        
        return df
    
    def save_features_to_csv(self, features_df, cif_path, output_dir=None):
        """
        将XRD特征保存到CSV文件
        Save XRD features to CSV file
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            特征数据框 / Features DataFrame
        cif_path : str
            原始CIF文件路径 / Original CIF file path
        output_dir : str, optional
            输出目录，默认为当前目录 / Output directory, default is current directory
        """
        # 生成输出文件名 / Generate output filename
        cif_filename = os.path.splitext(os.path.basename(cif_path))[0]
        csv_filename = f"{cif_filename}_XRD_Features.csv"
        
        if output_dir:
            csv_path = os.path.join(output_dir, csv_filename)
        else:
            csv_path = csv_filename
        
        # 保存到CSV文件 / Save to CSV file
        features_df.to_csv(csv_path, index=True)
        print(f"XRD特征已保存到 / XRD features saved to: {csv_path}")


def cal_xrd_features_from_cif(cif_path, save_csv=True, output_dir=None):
    """
    从CIF文件计算XRD特征的便捷函数
    Convenience function to calculate XRD features from CIF file
    
    Parameters:
    -----------
    cif_path : str
        CIF文件路径 / Path to CIF file
    save_csv : bool, default=True
        是否保存为CSV文件 / Whether to save as CSV file
    output_dir : str, optional
        输出目录 / Output directory
        
    Returns:
    --------
    pd.DataFrame
        XRD特征数据框 / XRD features DataFrame
    """
    # 创建XRD特征计算器 / Create XRD feature calculator
    calculator = XRDFeatureCalculator()
    
    # 计算特征 / Calculate features
    features_df = calculator.calculate_features_from_cif(cif_path)
    
    # 保存结果 / Save results
    if save_csv:
        calculator.save_features_to_csv(features_df, cif_path, output_dir)
    
    return features_df


if __name__ == "__main__":
    # 测试用例 / Test case
    cif_path = "E:\\RA\\数据库搭建工作\\Examples\\CsPbBr3.cif"
    
    print("开始计算XRD特征 / Starting XRD feature calculation...")
    features_df = cal_xrd_features_from_cif(cif_path)