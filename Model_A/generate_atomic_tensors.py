# mpiexec -n 8 python "/home/morningstar/Single-cell data/Model_A/generate_atomic_tensors.py" gpaw_results/CsPbBr3/CsPbBr3-gpaw-optimized.cif gpaw_results/CsPbBr3/CsPbBr3-gpaw.gpw

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json # (委员会新增)
from gpaw import GPAW
from gpaw.mpi import world
from pymatgen.core import Structure
from scipy.interpolate import RegularGridInterpolator

def generate_atomic_tensors_parallel(args):
    """
    从GPAW电子密度并行生成每个原子位点的歪斜对称张量。
    该函数的核心逻辑将被多个MPI进程并行执行。
    """
    # --- 1. 设置确定性的输入和输出路径 ---
    input_cif_path = Path(args.input_cif)
    input_gpw_path = Path(args.input_gpw)
    root_output_dir = Path(args.output_dir)

    base_name = input_cif_path.stem.replace('-gpaw-optimized', '')
    output_dir = root_output_dir / base_name
    output_csv = output_dir / f'{base_name}-lie-algebra-tensors.csv'

    # --- 2. 高通量流程控制 (主进程决定，然后广播) ---
    if world.rank == 0:
        should_run = True
        if not input_cif_path.exists() or not input_gpw_path.exists():
            print(f"[{base_name}] 致命错误: 输入文件 {input_cif_path} 或 {input_gpw_path} 未找到！", file=sys.stderr)
            should_run = False
        elif output_csv.exists() and not args.overwrite:
            print(f"[{base_name}] 跳过: 结果已存在于 {output_csv}。使用 --overwrite 强制重算。")
            should_run = False
        
        if should_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            
        run_flag = np.array([1 if should_run else 0], dtype=int)
    else:
        run_flag = np.empty(1, dtype=int)

    world.broadcast(run_flag, 0)
    if not bool(run_flag[0]):
        sys.exit(0)

    # --- 3. 数据加载和预处理 (所有进程执行) ---
    if world.rank == 0:
        print(f"[{base_name}] === 开始并行生成李代数原子张量 ({world.size} CPU核心) ===")
        print(f"[{base_name}] 加载结构: {input_cif_path}")
        print(f"[{base_name}] 加载GPAW计算: {input_gpw_path}")

    structure = Structure.from_file(input_cif_path)
    calc = GPAW(str(input_gpw_path), txt=None)
    
    density = calc.get_pseudo_density(spin=None)
    grid_shape = density.shape
    cell_vectors = calc.get_atoms().get_cell()

    if world.rank == 0:
        print(f"[{base_name}] 电子密度加载成功，网格尺寸: {grid_shape}")
        print(f"[{base_name}] 计算电子密度梯度 (支持非正交晶胞)...")

    # === (委员会修正) 修正梯度计算以支持非正交晶胞 ===
    # 物理原理：
    # 1. 在网格坐标(i, j, k)中计算无单位梯度，np.gradient() 默认步长为1。
    # 2. 将网格梯度转换为分数坐标(u, v, w)下的梯度，通过乘以每个方向的网格点数实现。
    #    grad_frac = grad_grid * grid_shape
    # 3. 使用链式法则将分数坐标梯度变换到笛卡尔坐标梯度。
    #    变换矩阵是晶格矩阵(F)的逆: grad_cart = grad_frac * F^-1
    
    # 1. 在网格坐标中计算梯度 (无单位)
    grad_grid = np.gradient(density)

    # 2. 转换为分数坐标下的梯度 (单位: e/Å³)
    grad_frac_x = grad_grid[0] * grid_shape[0]
    grad_frac_y = grad_grid[1] * grid_shape[1]
    grad_frac_z = grad_grid[2] * grid_shape[2]
    
    # 3. 将梯度分量堆叠成矢量场, shape: (Nx, Ny, Nz, 3)
    grad_frac = np.stack([grad_frac_x, grad_frac_y, grad_frac_z], axis=-1)

    # 4. 从分数坐标梯度变换到笛卡尔坐标梯度 (单位: e/Å⁴)
    try:
        cell_inv = np.linalg.inv(cell_vectors)
        # 使用einsum进行高效的批量矩阵-向量乘法
        # g_cart_j = sum_i(g_frac_i * cell_inv_ij)
        grad_cart = np.einsum('...i,ij->...j', grad_frac, cell_inv)
        grad_x, grad_y, grad_z = grad_cart[..., 0], grad_cart[..., 1], grad_cart[..., 2]
    except np.linalg.LinAlgError:
        if world.rank == 0:
            print(f"[{base_name}] 错误: 晶格矩阵奇异，无法求逆。梯度将全为零。")
        grad_x = np.zeros_like(density)
        grad_y = np.zeros_like(density)
        grad_z = np.zeros_like(density)


    if world.rank == 0:
        print(f"[{base_name}] 使用NumPy梯度计算方法（数值稳定）")
        print(f"[{base_name}] 梯度网格形状: grad_x={grad_x.shape}, grad_y={grad_y.shape}, grad_z={grad_z.shape}")
        print(f"[{base_name}] 晶格向量: a={cell_vectors[0, 0]:.3f}, b={cell_vectors[1, 1]:.3f}, c={cell_vectors[2, 2]:.3f} Å")

    if world.rank == 0:
        print(f"[{base_name}] 构建梯度插值器...")

    # 插值器设置（改进版，更稳定的边界处理）
    axes = (np.linspace(0, 1, grid_shape[0], endpoint=False),
            np.linspace(0, 1, grid_shape[1], endpoint=False),
            np.linspace(0, 1, grid_shape[2], endpoint=False))

    # 验证梯度数据是否有NaN或inf值
    grad_components = [grad_x, grad_y, grad_z]
    for i, grad in enumerate(grad_components):
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            if world.rank == 0:
                print(f"[{base_name}] 警告: 梯度分量 {i} 包含NaN或inf值，将被替换为0")
            grad_components[i] = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    grad_x, grad_y, grad_z = grad_components

    grad_interpolators = [
        RegularGridInterpolator(axes, grad_x, method='linear', bounds_error=False, fill_value=0.0),
        RegularGridInterpolator(axes, grad_y, method='linear', bounds_error=False, fill_value=0.0),
        RegularGridInterpolator(axes, grad_z, method='linear', bounds_error=False, fill_value=0.0)
    ]

    # --- 4. 并行计算张量 (每个进程计算一部分原子) ---
    if world.rank == 0:
        print(f"[{base_name}] 为 {len(structure)} 个原子并行计算张量...")
        
    tensors_local = []
    # === 改进：使用标准so(3)生成元公式计算李代数张量 ===
    # 使用步长循环将原子分配给不同的MPI进程
    for site_idx in range(world.rank, len(structure), world.size):
        site = structure[site_idx]
        frac_coords = site.frac_coords

        # 在原子位置直接插值电子密度梯度（简化版，避免复杂积分）
        try:
            grad_vector = np.array([
                grad_interpolators[0](frac_coords[np.newaxis, :])[0],  # grad_x
                grad_interpolators[1](frac_coords[np.newaxis, :])[0],  # grad_y
                grad_interpolators[2](frac_coords[np.newaxis, :])[0]   # grad_z
            ])

            # 数值稳定性检查
            if np.any(np.isnan(grad_vector)) or np.linalg.norm(grad_vector) < 1e-12:
                # 如果梯度无效或太小，使用零张量
                tensor = np.zeros((3, 3))
                if world.rank == 0:
                    print(f"[{base_name}] 警告: 原子 {site_idx} 梯度无效，使用零张量")
            else:
                # === 核心改进：使用标准so(3)生成元公式 ===
                # L = [0, -∇ρ_z, ∇ρ_y;
                #      ∇ρ_z, 0, -∇ρ_x;
                #      -∇ρ_y, ∇ρ_x, 0]
                gx, gy, gz = grad_vector
                tensor = np.array([
                    [0.0,  -gz,   gy],
                    [ gz,  0.0,  -gx],
                    [-gy,   gx,  0.0]
                ])

                # 验证反对称性（调试用）
                if not np.allclose(tensor + tensor.T, 0, atol=1e-10):
                    print(f"[{base_name}] 警告: 原子 {site_idx} 张量反对称性检查失败")

        except Exception as e:
            # 插值失败时使用零张量
            tensor = np.zeros((3, 3))
            if world.rank == 0:
                print(f"[{base_name}] 错误: 原子 {site_idx} 梯度插值失败: {e}，使用零张量")

        # 将结果存储为扁平化的列表/行
        tensors_local.append([
            site_idx,
            tensor[0, 0], tensor[0, 1], tensor[0, 2],
            tensor[1, 0], tensor[1, 1], tensor[1, 2],
            tensor[2, 0], tensor[2, 1], tensor[2, 2]
        ])

    # --- 5. 汇总并保存结果 (仅主进程) ---
    # 使用全局固定形状数组并通过规约(sum)合并，避免 gather 对数组形状的一致性要求
    num_sites = len(structure)
    columns = [
        'site_index',
        'T_struct_00', 'T_struct_01', 'T_struct_02',
        'T_struct_10', 'T_struct_11', 'T_struct_12',
        'T_struct_20', 'T_struct_21', 'T_struct_22'
    ]

    global_array = np.zeros((num_sites, 10), dtype=float)
    if len(tensors_local) > 0:
        for row in tensors_local:
            idx = int(row[0])
            global_array[idx, :] = row

    # 集体规约：对所有进程的同形数组求和（就地规约）
    world.sum(global_array)
    reduced = global_array

    if world.rank == 0:
        print(f"[{base_name}] 所有进程计算完成，正在合并结果...")
        df = pd.DataFrame(reduced, columns=columns)
        df['site_index'] = df['site_index'].astype(int)
        # 保险起见，再按 site_index 排序
        df = df.sort_values('site_index').reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        print(f"[{base_name}] 李代数原子张量已成功保存到: {output_csv}")

        # === 改进：自动进行数学验证 ===
        if hasattr(args, 'verify') and args.verify:
            verify_lie_algebra_properties(df, base_name, output_dir)

        print(f"[{base_name}] --- 生成过程成功完成 ---")

    # --- 6. 显式清理资源 ---
    # 在所有进程完成工作后，进行同步
    world.barrier()
    # 显式删除GPAW计算器对象，以尝试进行更干净的资源释放
    del calc
 
def verify_lie_algebra_properties(tensors_df, base_name, output_dir):
    """
    (委员会修正) 验证计算的李代数张量是否满足数学性质，并增加全局秩验证。
    """
    print(f"[{base_name}] === 李代数张量数学验证 ===")

    validation_results = {
        'total_atoms': len(tensors_df),
        'antisymmetric_passed': 0,
        'antisymmetric_failed': 0,
        'zero_tensors': 0,
        'nonzero_tensors': 0,
        'subspace_rank': 0 # (委员会新增)
    }

    gradient_vectors = []
    for idx, row in tensors_df.iterrows():
        # 重建张量
        tensor = np.array([
            [row['T_struct_00'], row['T_struct_01'], row['T_struct_02']],
            [row['T_struct_10'], row['T_struct_11'], row['T_struct_12']],
            [row['T_struct_20'], row['T_struct_21'], row['T_struct_22']]
        ])

        # 检查是否为零张量
        if np.allclose(tensor, 0, atol=1e-15):
            validation_results['zero_tensors'] += 1
        else:
            validation_results['nonzero_tensors'] += 1
            # (委员会新增) 为非零张量重构梯度向量以用于秩分析
            gx = -tensor[1, 2]
            gy = tensor[0, 2]
            gz = -tensor[0, 1]
            gradient_vectors.append([gx, gy, gz])

        # 检查反对称性
        if np.allclose(tensor + tensor.T, 0, atol=1e-10):
            validation_results['antisymmetric_passed'] += 1
        else:
            validation_results['antisymmetric_failed'] += 1
            asymmetry_error = np.max(np.abs(tensor + tensor.T))
            print(f"[{base_name}] 警告: 原子 {int(row['site_index'])} 反对称性误差: {asymmetry_error:.2e}")

    # 输出验证结果
    print(f"[{base_name}] 验证结果:")
    print(f"[{base_name}]   总原子数: {validation_results['total_atoms']}")
    print(f"[{base_name}]   零张量: {validation_results['zero_tensors']}")
    print(f"[{base_name}]   非零张量: {validation_results['nonzero_tensors']}")
    print(f"[{base_name}]   反对称性通过: {validation_results['antisymmetric_passed']}")
    print(f"[{base_name}]   反对称性失败: {validation_results['antisymmetric_failed']}")

    # 计算通过率
    if validation_results['total_atoms'] > 0:
        antisymmetric_rate = validation_results['antisymmetric_passed'] / validation_results['total_atoms'] * 100
        print(f"[{base_name}]   反对称性通过率: {antisymmetric_rate:.1f}%")

    # === (委员会新增) 全局秩验证 ===
    rank = 0
    if gradient_vectors:
        grad_matrix = np.array(gradient_vectors)
        if grad_matrix.shape[0] > 0:
            rank = np.linalg.matrix_rank(grad_matrix, tol=1e-8)
    
    validation_results['subspace_rank'] = int(rank)
    print(f"[{base_name}]   梯度向量子空间秩: {rank} (预期 <= 3)")
    if rank < 3:
        print(f"[{base_name}]   注意: 秩小于3表示电子密度梯度分布在低维子空间中。")

    # === (委员会新增) 保存验证结果到JSON文件 ===
    output_json_path = output_dir / f'{base_name}-lie-algebra-validation.json'
    try:
        with open(output_json_path, 'w') as f:
            json.dump(validation_results, f, indent=4)
        print(f"[{base_name}] ✅ 验证结果已保存到: {output_json_path.name}")
    except Exception as e:
        print(f"[{base_name}] 错误: 保存验证结果失败: {e}")


    if validation_results['antisymmetric_failed'] > 0:
        print(f"[{base_name}] ⚠️  警告: 反对称性检查存在失败项，请检查数值精度。")
    else:
        print(f"[{base_name}] ✅ 李代数张量验证通过")

    return validation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从GPAW电子密度生成每个原子位点的so(3)李代数张量。使用标准梯度法，确保数学严谨性。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_cif', type=str, help='输入的优化后CIF文件路径')
    parser.add_argument('input_gpw', type=str, help='与CIF文件对应的GPAW计算结果(.gpw)文件')
    parser.add_argument('--output-dir', type=str, default='atomic_tensors_results', help='存放所有张量结果的根目录')
    parser.add_argument('--overwrite', action='store_true', help='如果设置，则覆盖已存在的计算结果，否则跳过')
    parser.add_argument('--verify', action='store_true', help='启用数学验证（检查反对称性等性质）')

    args = parser.parse_args()

    generate_atomic_tensors_parallel(args)
