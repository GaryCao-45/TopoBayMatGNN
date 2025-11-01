import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from Model_A.Features_0_Simplex import run_0_simplex_features
from Model_A.Features_1_Simplex import run_1_simplex_features
from Model_A.Features_2_Simplex import run_2_simplex_features
from pymatgen.analysis.graphs import StructureGraph # 导入 StructureGraph

def run_model_a_pipeline(
    cif_file: str,
    pw_gpw_file: str,
    fd_gpaw_file: Optional[str],  # 修改为 Optional[str]
    output_dir: str,
    atomic_tensors_csv: str,
    bond_persistence_max_radius: float = 4.0, # 传递新增参数
    max_workers: Optional[int] = None,  # 新增：并行进程数，默认自动检测
    use_parallel: bool = True  # 新增：是否启用并行计算
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StructureGraph]:
    """
    运行Model-A特征提取管道，依次计算0-单纯形、1-单纯形和2-单纯形的特征。
    支持CPU并行计算以加速特征提取过程。

    参数:
        cif_file_path (str): 输入CIF文件的路径。
        pw_gpw_file (str): 平面波GPW文件的路径。
        fd_gpaw_file (Optional[str]): FD模式GPW文件的路径。
        lie_algebra_tensors_csv_file (Optional[str]): (重命名) 外部计算的李代数原子张量文件路径。
        output_dir (Optional[str]): 输出CSV文件的根目录。实际保存路径为 "根目录/基名/"；
            若为 None，则使用 cif_file_path 所在目录作为根目录。
        bond_persistence_max_radius (float): 键级持久同调计算的最大半径。
        max_workers (Optional[int]): 并行进程数，默认自动检测CPU核心数。
        use_parallel (bool): 是否启用并行计算，默认True。

    返回:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            包含0-单纯形、1-单纯形和2-单纯形特征的DataFrame元组。
    """
    print("\n" + "=" * 80)
    print("开始运行Model-A特征提取管道")
    print("=" * 80)

    # 统一输出到 "根目录/基名/" 的子目录中，模仿 gpaw_opt.py 的目录组织
    base_name = Path(cif_file).stem.replace('-gpaw-optimized', '').replace('-optimized', '')
    output_root_dir = Path(output_dir) if output_dir else Path(cif_file).parent
    output_path_base = output_root_dir / base_name

    output_path_base.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_path_base}")
    print(f"并行计算: {'启用' if use_parallel else '禁用'}")
    if max_workers:
        print(f"最大进程数: {max_workers}")
    else:
        import os
        print(f"自动检测进程数: {os.cpu_count()}")

    # 0-单纯形特征
    print("\n--- 步骤 1: 计算0-单纯形（原子）特征 ---")
    try:
        # 返回的第二个值是代码内部生成的几何结构张量，重命名以避免混淆
        atomic_features_df, structure_tensors_csv_path = run_0_simplex_features(
            cif_file=cif_file,
            pw_gpw_file=pw_gpw_file,
            fd_gpw_file=fd_gpaw_file,
            atomic_tensors_csv=atomic_tensors_csv, # 传递外部李代数张量文件
            output_dir=str(output_path_base)
        )
        print(f"0-单纯形特征维度: {atomic_features_df.shape}")

    except Exception as e:
        print(f"错误: 0-单纯形特征计算失败: {e}")
        raise

    # 1-单纯形特征
    print("\n--- 步骤 2: 计算1-单纯形（化学键）特征 ---")
    try:
        bond_features_df, topology_graph = run_1_simplex_features(
            cif_file=cif_file,
            atomic_features_csv=str(output_path_base / f"{Path(cif_file).stem.replace('-optimized', '')}-0-Simplex-Features.csv"),
            atomic_tensors_csv=str(structure_tensors_csv_path), # 修正: 传递由0-Simplex生成的几何结构张量
            pw_gpw_file=pw_gpw_file,
            fd_gpw_file=fd_gpaw_file,
            output_dir=str(output_path_base)
        )
        print(f"1-单纯形特征维度: {bond_features_df.shape}")
    except Exception as e:
        print(f"错误: 1-单纯形特征计算失败: {e}")
        raise

    # 2-单纯形特征
    print("\n--- 步骤 3: 计算2-单纯形（三角形）特征 ---")
    try:
        triangle_features_df = run_2_simplex_features(
            cif_file=cif_file,
            atomic_features_csv=str(output_path_base / f"{Path(cif_file).stem.replace('-optimized', '')}-0-Simplex-Features.csv"),
            bond_features_csv=str(output_path_base / f"{Path(cif_file).stem.replace('-optimized', '')}-1-Simplex-Features.csv"),
            atomic_tensors_csv=str(structure_tensors_csv_path), # 修正: 传递由0-Simplex生成的几何结构张量
            pw_gpw_file=pw_gpw_file,
            fd_gpw_file=fd_gpaw_file,
            output_dir=str(output_path_base),
            topology_graph=topology_graph # 传递拓扑图
        )
        print(f"2-单纯形特征维度: {triangle_features_df.shape}")
    except Exception as e:
        print(f"错误: 2-单纯形特征计算失败: {e}")
        raise

    print("\n" + "=" * 80)
    print("Model-A特征提取管道运行完成")
    print("所有特征已计算并保存到指定目录")
    print("=" * 80)

    return atomic_features_df, bond_features_df, triangle_features_df, topology_graph

def test_0_simplex_only(cif_file: str, pw_gpw_file: str, fd_gpw_file: Optional[str] = None,
                       output_dir: str = None, atomic_tensors_csv: Optional[str] = None):
    """
    仅测试0-单纯形特征计算的简化函数。
    这是一个独立的测试函数，可以直接调用来验证0-单纯形代码。
    """
    print("\n" + "=" * 60)
    print("测试0-单纯形特征计算")
    print("=" * 60)

    try:
        # 调用0-单纯形特征计算
        atomic_features_df, structure_tensors_path = run_0_simplex_features(
            cif_file=cif_file,
            pw_gpw_file=pw_gpw_file,
            fd_gpw_file=fd_gpw_file,
            output_dir=output_dir,
            atomic_tensors_csv=atomic_tensors_csv
        )

        print("\n--- 0-单纯形测试结果 ---")
        print(f"特征DataFrame形状: {atomic_features_df.shape}")
        print(f"原子数量: {atomic_features_df.shape[0]}")
        print(f"特征维度: {atomic_features_df.shape[1]}")
        print(f"结构张量保存路径: {structure_tensors_path}")

        print("\n前5个原子的特征预览:")
        print(atomic_features_df.head())

        print("\n特征列名:")
        for i, col in enumerate(atomic_features_df.columns):
            print("2d")

        # 验证关键特征是否存在
        expected_features = [
            'atomic_number', 'electronegativity', 'covalent_radius',
            'bader_charge', 'electron_density', 'elf'
        ]

        missing_features = []
        for feature in expected_features:
            if feature not in atomic_features_df.columns:
                missing_features.append(feature)

        if missing_features:
            print(f"\n警告: 缺少以下预期特征: {missing_features}")
        else:
            print("\n✓ 所有关键特征都已成功计算")

        return atomic_features_df, structure_tensors_path

    except Exception as e:
        print(f"0-单纯形测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    # --- 示例用法 ---
    # 定义输入文件路径 (请根据实际情况修改)
    current_dir = Path(__file__).parent.resolve()

    # 示例文件路径 (请根据您的实际文件位置修改)
    cif_file = current_dir / "gpaw_results" / "CH3NH3PbI3" / "CH3NH3PbI3-gpaw-optimized.cif"
    pw_gpw_file = current_dir / "gpaw_results" / "CH3NH3PbI3" / "CH3NH3PbI3-gpaw.gpw"
    fd_gpw_file = current_dir / "fd_results" / "CH3NH3PbI3" / "CH3NH3PbI3-gpaw-fd.gpw"
    lie_algebra_tensors_file = current_dir / "atomic_tensors_results" / "CH3NH3PbI3" / "CH3NH3PbI3-lie-algebra-tensors.csv"

    # 输出目录
    output_directory = current_dir / "model_a_results"

    # 检查文件是否存在
    if not cif_file.exists():
        print(f"错误: CIF 文件不存在: {cif_file}")
        print("请修改文件路径或创建测试文件")
        exit(1)

    if not pw_gpw_file.exists():
        print(f"错误: PW GPW 文件不存在: {pw_gpw_file}")
        print("请修改文件路径或创建测试文件")
        exit(1)

    if fd_gpw_file and not fd_gpw_file.exists():
        print(f"警告: FD GPW 文件不存在: {fd_gpw_file}，ELF特征将无法计算")

    if lie_algebra_tensors_file and not lie_algebra_tensors_file.exists():
        print(f"警告: 李代数张量文件不存在: {lie_algebra_tensors_file}，将使用默认结构张量")

    # 选择测试模式
    test_mode = "full_pipeline"  # 可以改为 "0_simplex_only" 来仅测试0-单纯形

    if test_mode == "0_simplex_only":
        # 仅测试0-单纯形
        print("运行模式: 仅测试0-单纯形特征")
        try:
            atomic_df, tensors_path = test_0_simplex_only(
                cif_file=str(cif_file),
                pw_gpw_file=str(pw_gpw_file),
                fd_gpw_file=str(fd_gpw_file) if fd_gpw_file.exists() else None,
                output_dir=str(output_directory),
                atomic_tensors_csv=str(lie_algebra_tensors_file) if lie_algebra_tensors_file.exists() else None
            )
            print("\n✓ 0-单纯形测试完成！")

        except Exception as e:
            print(f"0-单纯形测试失败: {e}")

    else:
        # 运行完整管道
        print("运行模式: 完整Model-A管道")
        try:
            atomic_df, bond_df, triangle_df, topology_graph = run_model_a_pipeline(
                cif_file=str(cif_file),
                pw_gpw_file=str(pw_gpw_file),
                fd_gpaw_file=str(fd_gpw_file) if fd_gpw_file.exists() else None,
                output_dir=str(output_directory),
                atomic_tensors_csv=str(lie_algebra_tensors_file) if lie_algebra_tensors_file.exists() else None,
                bond_persistence_max_radius=4.0,
                max_workers=None,
                use_parallel=True
            )

            print("\n--- 最终特征摘要 ---")
            print(f"0-单纯形特征 (原子): {atomic_df.shape[0]} 个原子, {atomic_df.shape[1]} 维特征")
            print(atomic_df.head())
            print(f"1-单纯形特征 (化学键): {bond_df.shape[0]} 个键, {bond_df.shape[1]} 维特征")
            print(bond_df.head())
            print(f"2-单纯形特征 (三角形): {triangle_df.shape[0]} 个三角形, {triangle_df.shape[1]} 维特征")
            print(triangle_df.head())

        except Exception as e:
            print(f"运行Model-A管道时发生错误: {e}")
            import traceback
            traceback.print_exc()
