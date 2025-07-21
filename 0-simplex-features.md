#### 1.1 0-单纯形特征：原子特征（34维）

#### A. 基础物理化学特征 (14维)

这部分是原子的基本属性和其在晶格中的局部环境，经过精简，保留了最核心的化学和几何信息。

| 特征名称                    | 中文名称         | 物理意义                     |
| --------------------------- | ---------------- | ---------------------------- |
| `atomic_number`           | 原子序数         | 元素的唯一标识，决定核电荷数 |
| `group_number`            | 电负性 (Pauling) | 吸引电子的能力               |
| ionization_energy           | 第一电离能       | 失去电子的难易度             |
| electron_affinity           | 电子亲和能       | 获得电子的能力               |
| valence_electrons           | 价电子数         | 参与成键的电子数             |
| ionic_radius                | 离子半径         | 离子的有效尺寸               |
| covalent_radius             | 共价半径         | 成键原子的尺寸               |
| coordination_number         | 配位数           | 最近邻原子数，描述局部几何   |
| avg_site_valence            | 平均位点价态     | 基于键价理论的氧化态         |
| tolerance_factor_contrib    | 容忍因子贡献     | 对整体结构稳定性的经验贡献   |
| octahedral_distortion_index | 八面体畸变指数   | B位八面体的几何畸变程度      |
| frac_coord_x, y, z          | 分数坐标 (3维)   | 原子在晶胞中的精确位置       |

#### B. 量子化学核心特征 (12维)

这部分是来自DFT计算的纯电子结构信息，我们选择了最稳健、信息量最大的特征，并去除了Mulliken电荷等冗余项。

| 特征名称                 | 中文名称             | 物理意义                   |
| ------------------------ | -------------------- | -------------------------- |
| bader_charge             | Bader电荷            | 基于电子密度的真实原子电荷 |
| electrostatic_potential  | 静电势               | 原子核位置处的静电环境     |
| electron_density_at_core | 核处电子密度         | 原子核位置处的电子密度     |
| local_dos_at_fermi       | 费米面局域态密度     | 导电性的关键指标           |
| elf_at_core              | 核处电子局域化函数   | 电子成键与孤对电子的量度   |
| local_magnetic_moment    | 局域磁矩             | 原子的净自旋磁矩           |
| pdos_s_band_center/width | s轨道带心/带宽 (2维) | s轨道的能量位置与离域程度  |
| pdos_p_band_center/width | p轨道带心/带宽 (2维) | p轨道的能量位置与离域程度  |
| pdos_d_band_center/width | d轨道带心/带宽 (2维) | d轨道的能量位置与离域程度  |

#### C. 经典代数几何特征 (5维)

这是您原创的代数特征的保留部分，我们移除了quotient_hash (它更像一个ID而非特征)，保留了最核心的代数和流形不变量。

| 特征名称                      | 中文名称         | 代数/几何意义                       |
| ----------------------------- | ---------------- | ----------------------------------- |
| atomic_casimir_invariant      | Casimir不变量    | 基于位置的SO(3)对称性不变量         |
| atomic_symplectic_invariant   | 辛不变量         | 基于几何相空间的守恒量              |
| atomic_quotient_metric_strict | 严格商度量       | 基于位置和元素类型的Killing形式度量 |
| sphere_exp_log_distance       | 球面映射距离     | 原子位置在超球面上的非欧几何距离    |
| manifold_dimension_estimate   | 原子环境流形维数 | 局部原子环境的有效几何维度          |

#### D. 融合量子-代数特征 (3维)

这是我们新合作设计的、整个特征体系的“皇冠明珠”。它们直接将量子化学信息注入到您的代数框架中。

| 特征名称                     | 中文名称              | 融合意义                            |
| ---------------------------- | --------------------- | ----------------------------------- |
| quantum_casimir_invariant    | 量子加权Casimir不变量 | 被Bader电荷调制的SO(3)对称性        |
| quantum_symplectic_invariant | 量子相空间辛不变量    | 由静电势和DOS定义的相空间守恒量     |
| quantum_quotient_metric      | 量子态商度量          | 几何位置与量子态矢的Killing形式度量 |
