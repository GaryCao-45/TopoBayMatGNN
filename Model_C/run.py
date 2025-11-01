#!/usr/bin/env python3
"""
TEN-FMA Framework 统一运行器
Unified Runner for TEN-FMA Framework

此脚本提供统一的接口来运行TEN-FMA框架的各种功能：
- 可视化预训练模型
- 生成全局嵌入向量CSV
- 批量处理多个材料和模型

使用方法:
python run.py [command] [options]

可用命令:
- visualize    : 可视化预训练模型
- embeddings   : 生成嵌入向量CSV
- batch        : 批量处理
"""

import argparse
import sys
import os
from pathlib import Path

# 设置Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

def run_visualization(args):
    """运行可视化功能"""
    print("🎨 启动可视化功能...")

    # 导入可视化脚本
    try:
        sys.path.insert(0, os.path.join(current_dir, 'visualization'))
        from visualization import visualize_pretrained_models as viz

        # 设置参数
        viz_args = [
            '--model1', args.model1 or 'model-1-09171648',
            '--model2', args.model2 or 'model-2-09180905',
            '--model3', args.model3 or 'model-3-09181323',
            '--model4', args.model4 or 'model-4-09181820'
        ]

        if args.material:
            viz_args.extend(['--material', args.material])

        if args.batch_all:
            viz_args.append('--batch-all')

        if hasattr(args, 'save_embeddings') and not args.save_embeddings:
            viz_args.append('--no-save-embeddings')

        # 运行可视化
        sys.argv = ['visualize_pretrained_models.py'] + viz_args
        viz.main()

    except ImportError as e:
        print(f"❌ 无法导入可视化模块: {e}")
        return False

    return True

def run_embeddings(args):
    """运行嵌入向量生成功能"""
    print("🔬 启动嵌入向量生成功能...")

    # 导入嵌入向量生成脚本
    try:
        sys.path.insert(0, os.path.join(current_dir, 'scripts'))
        # 修复导入问题：使用相对导入或绝对路径导入
        
        from scripts import generate_embeddings_batch as emb

        # 设置参数
        emb_args = [
            '--model1', args.model1 or 'model-1-09171648',
            '--model2', args.model2 or 'model-2-09180905',
            '--model3', args.model3 or 'model-3-09181323',
            '--model4', args.model4 or 'model-4-09181820'
        ]

        if args.all_materials:
            emb_args.append('--all-materials')
        elif args.materials:
            emb_args.extend(['--materials'] + args.materials)

        if args.output_dir:
            emb_args.extend(['--output-dir', args.output_dir])

        # 运行嵌入向量生成
        sys.argv = ['generate_embeddings_batch.py'] + emb_args
        emb.main()

    except ImportError as e:
        print(f"❌ 无法导入嵌入向量生成模块: {e}")
        return False

    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TEN-FMA Framework 统一运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

# 可视化特定材料
python run.py visualize --material CsPbI3

# 可视化所有材料
python run.py visualize --batch-all

# 生成特定材料的嵌入向量
python run.py embeddings --materials CsPbCl3 CsPbBr3

# 生成所有材料的嵌入向量
python run.py embeddings --all-materials

# 指定模型目录
python run.py visualize --model1 my-model-1 --model2 my-model-2 --model3 my-model-3 --model4 my-model-4 --batch-all
        """
    )

    parser.add_argument('command', choices=['visualize', 'embeddings'],
                       help='要执行的命令')

    parser.add_argument('--model1', type=str,
                       help='第一个模型目录名')
    parser.add_argument('--model2', type=str,
                       help='第二个模型目录名')
    parser.add_argument('--model3', type=str,
                       help='第三个模型目录名')
    parser.add_argument('--model4', type=str,
                       help='第四个模型目录名')
    # 可视化相关参数
    parser.add_argument('--material', type=str,
                       help='要可视化的材料名称')
    parser.add_argument('--batch-all', action='store_true',
                       help='批量处理所有5个材料')
    parser.add_argument('--save-embeddings', action='store_true', default=True,
                       help='可视化时同时保存嵌入向量 (默认开启)')

    # 嵌入向量相关参数
    parser.add_argument('--materials', type=str, nargs='+',
                       help='要生成嵌入向量的材料名称')
    parser.add_argument('--all-materials', action='store_true',
                       help='生成所有5个材料的嵌入向量')
    parser.add_argument('--output-dir', type=str, default='embeddings',
                       help='嵌入向量输出目录')

    args = parser.parse_args()

    print("🎯 TEN-FMA Framework 统一运行器")
    print("=" * 50)

    # 检查必要的模块是否可以导入
    try:
        from src.simplex_data_loader import PerovskiteSimplexDataset
        from src.snn_model import SNN, SNNHamiltonianDynamicsSDE
        print("✓ TEN-FMA核心模块导入成功")
    except ImportError as e:
        print("❌ 无法导入TEN-FMA核心模块")
        print(f"错误详情: {e}")
        print("\n请确保:")
        print("1. 您在项目根目录运行此脚本")
        print("2. src目录存在且包含必要的模块")
        return False

    # 执行相应命令
    success = False

    if args.command == 'visualize':
        success = run_visualization(args)
    elif args.command == 'embeddings':
        success = run_embeddings(args)

    if success:
        print("\n✅ 命令执行成功！")
    else:
        print("\n❌ 命令执行失败！")
        return False

    return True

if __name__ == "__main__":
    main()
