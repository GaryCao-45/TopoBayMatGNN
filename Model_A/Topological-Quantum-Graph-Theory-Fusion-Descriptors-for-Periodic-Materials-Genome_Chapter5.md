# Chapter 5: 2-Simplex Descriptors: Quantification of Three-Body Interaction Configuration and Complexity

A 2-simplex, i.e., a triangle, represents three-body interactions between three atoms in materials science, serving as a key unit for understanding local geometric configurations, stress distributions, and charge transport pathways. This section will detail how to construct a set of descriptors that comprehensively quantify 2-simplex features in crystal structures through geometry, cross-level feature fusion, quantum chemistry, and advanced algebraic theories. These features not only capture the local characteristics of three-body interactions but also provide deeper physicochemical insights through the fusion of multi-level information.

## 5.1 2-Simplex Identification and Geometric Definition

This framework constructs 2-simplices by identifying all unique atomic triplets in the crystal structure. A 2-simplex consists of a central atom and its two nearest neighbors, precisely identified using Pymatgen's CrystalNN tool, with robust handling including tolerance adjustments.

## 5.2 Geometric Features

Geometric features directly quantify the shape and spatial arrangement of the 2-simplex itself. These features are direct manifestations of the local structural rigidity and flexibility of the crystal.

### 5.2.1 Triangle Area (`triangle_area`)

`triangle_area` descriptor quantifies the actual area of a 2-simplex (i.e., a triangle formed by three atomic nuclei) in space. It is precisely calculated as half the magnitude of the cross product of two vectors formed by the Cartesian coordinates of the three atoms.
$$
\text{Area} = \frac{1}{2} \| (\mathbf{p}_j - \mathbf{p}_i) \times (\mathbf{p}_k - \mathbf{p}_i) \|_2
$$
where $\mathbf{p}_i, \mathbf{p}_j, \mathbf{p}_k$ are the Cartesian coordinate vectors of the three atoms $i, j, k$ forming the triangle.

Physically, triangle area is a direct indicator of the compactness of local atomic arrangement and spatial occupancy. A larger area may indicate relatively larger interatomic distances or a more open local structure. Chemically, it reflects the geometric configuration of a three-atom system, e.g., it is significant in determining bond angles and coordination polyhedron shapes. In materials science, changes in triangle area can indicate local strain, defect regions, or structural reconstruction during phase transitions. In computer science and artificial intelligence fields, `triangle_area` provides a basic and invariant geometric feature, for machine learning models to understand and predict local structural properties of materials, e.g., with potential applications in determining phonon scattering mechanisms or thermal expansion coefficients.

### 5.2.2 Bond Angle Variance (`bond_angle_variance`)

`bond_angle_variance` descriptor quantifies the dispersion of the three bond angles (i.e., $\angle ijk, \angle jki, \angle kij$) forming a 2-simplex. It is obtained by calculating the statistical variance of these three angles (expressed in radians).
$$
\text{Variance}(\theta_1, \theta_2, \theta_3) = \frac{1}{3} \sum_{n=1}^{3} (\theta_n - \bar{\theta})^2
$$
where $\theta_n$ are the three internal angles of the triangle, and $\bar{\theta}$ is their average value.

Physically, bond angle variance is a key indicator of local geometric distortion and bonding flexibility. For example, in an ideal equilateral triangle, the bond angle variance is zero; while in a highly distorted or flattened triangle, the variance significantly increases. Chemically, it directly relates to molecular conformational strain and bonding flexibility, e.g., it is important in studies of polyhedron distortion. In materials science, high bond angle variance may indicate local structural instability, defects, or precursor states to phase transitions, which may affect material mechanical properties (e.g., hardness, ductility) and thermodynamic stability. In computer science and artificial intelligence fields, `bond_angle_variance` provides machine learning models with a quantitative description of the degree of local structural distortion within materials, aiding models in identifying and predicting lattice defects, phase behavior, and structural stability of materials under extreme conditions.

### 5.2.3 Triangle Shape Factor (`triangle_shape_factor`)

`triangle_shape_factor` descriptor quantifies the deviation of the shape of a 2-simplex (triangle) from an ideal equilateral triangle. It is a dimensionless normalized factor, ranging from 0 to 1, where 1 represents a perfect equilateral triangle and 0 represents an extremely degenerate triangle (e.g., collinear). Its definition is as follows:
$$
\text{ShapeFactor} = \frac{12\sqrt{3} \cdot \text{Area}}{\text{Perimeter}^2}
$$
where `Area` is the area of the triangle and `Perimeter` is the perimeter of the triangle.

Physically, the triangle shape factor reflects the local lattice anisotropy and geometric ideality. Values close to 1 indicate an approximately ideal symmetric local environment, while values close to 0 indicate severe geometric distortion. Chemically, it provides a quantitative evaluation of the geometric regularity of a three-atom coordination polyhedron or bonding network, which is crucial for understanding the influence of the local chemical environment on electronic structure. In materials science, `triangle_shape_factor` is a key feature for evaluating material structural stability, phase behavior, and stress distribution in specific crystal structures; for example, in analyzing octahedral tilts in perovskites, it can effectively quantify the distortion of individual faces. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of the local geometric regularity within materials, aiding models in predicting material symmetry breaking, lattice relaxation effects, and responses to external stresses.

## 5.3 Cross-Level Derived Features

These features integrate lower-level physicochemical information into the 2-simplex description through statistical aggregation of 0-simplex (atom) and 1-simplex (bond) features, thereby capturing the average behavior and fluctuations of atoms and bonds in three-body interactions.

### 5.3.1 Atomic Incompatibility Statistics (`avg_atomic_incompatibility`, `variance_atomic_incompatibility`)

These two features quantify, respectively, the average and variance of the `structure-chemistry incompatibility` (from Section 3.4.1) for the three atoms comprising a 2-simplex. They reflect the average level and fluctuation of local strain or geometric distortion at the atomic level within the triangle.
$$
\begin{aligned}
\text{avg\_atomic\_incomp} &= \frac{1}{3} \sum_{i \in \text{triangle}} \text{Incomp}_{\text{S-C},i} \\
\text{variance\_atomic\_incomp} &= \frac{1}{3} \sum_{i \in \text{triangle}} (\text{Incomp}_{\text{S-C},i} - \text{avg\_atomic\_incomp})^2
\end{aligned}
$$
where $\text{Incomp}_{\text{S-C},i}$ is the structure-chemistry incompatibility of atom $i$.

Physically, high average incompatibility may indicate that the triangular region is generally under high stress or has significant lattice distortions. High variance, conversely, suggests non-uniform stress distribution among atoms within this region, with localized stress concentrations or soft spots. Chemically, these features reveal differences in the structure-chemical environment among the three atoms in the system. In materials science, they are key indicators for assessing material toughness, fracture toughness, and phase transition stability, crucial for understanding local stress accumulation and failure mechanisms. In computer science and artificial intelligence fields, these features provide machine learning models with a quantitative description of the local stress distribution and structural heterogeneity within materials.

### 5.3.2 Bader Charge Statistics (`avg_bader_charge`, `variance_bader_charge`)

These two features quantify, respectively, the average and variance of the Bader charges (from Section 3.3.1) for the three atoms comprising a 2-simplex. They reflect the average state and non-uniformity of charge distribution within the triangular region.
$$
\begin{aligned}
\text{avg\_bader\_charge} &= \frac{1}{3} \sum_{i \in \text{triangle}} \text{Charge}_i \\
\text{variance\_bader\_charge} &= \frac{1}{3} \sum_{i \in \text{triangle}} (\text{Charge}_i - \text{avg\_bader\_charge})^2
\end{aligned}
$$
where $\text{Charge}_i$ is the Bader charge of atom $i$.

Physically, the average Bader charge reflects the overall electron enrichment or depletion of the region, related to material charge transport and potential distribution. The variance, conversely, measures the degree of delocalization or localization of charge among the three atoms, with high variance potentially indicating charge polarization or charge transfer effects. Chemically, these features reveal the ionic or covalent characteristics of local chemical bonding. In materials science, these features are key indicators for designing and optimizing electrochemical materials (e.g., battery electrodes, catalysts) and semiconductor materials, as they directly relate to material charge transport efficiency and active sites. In computer science and artificial intelligence fields, these features help machine learning models predict material conductivity, catalytic activity, and photoelectric conversion efficiency.

### 5.3.3 Vectorial Asymmetry Statistics (`avg_vectorial_asymmetry`, `max_vectorial_asymmetry`)

These two features quantify, respectively, the average and maximum of the squared norm of the vectorial asymmetry (from Section 3.4.1) for the three atoms comprising a 2-simplex. They capture the average strength and the strongest point of symmetry breaking in the local atomic environment within the triangle.
$$
\begin{aligned}
\text{avg\_vectorial\_asymmetry} &= \frac{1}{3} \sum_{i \in \text{triangle}} \|\mathbf{V}_{\text{asym},i}\|^2 \\
\text{max\_vectorial\_asymmetry} &= \max_{i \in \text{triangle}} (\|\mathbf{V}_{\text{asym},i}\|^2)
\end{aligned}
$$
where $\|\mathbf{V}_{\text{asym},i}\|^2$ is the squared norm of the vectorial asymmetry of atom $i$.

Physically, high average vectorial asymmetry indicates that the region is generally in an asymmetric local potential field, which may lead to piezoelectric, ferroelectric, or nonlinear optical effects. The maximum value, conversely, indicates the most asymmetric atomic site within that triangle. Chemically, these features relate to local coordination environment distortion and stereochemical effects. In materials science, they are key indicators for screening functional materials with specific symmetry-related properties (e.g., piezoelectric, pyroelectric materials), crucial for understanding microscopic mechanisms of symmetry breaking during phase transitions. In computer science and artificial intelligence fields, these features help machine learning models predict material polarization behavior, nonlinear optical response, and sensitivity to external fields.

### 5.3.4 Bond Alignment Statistics (`avg_bond_alignment`, `variance_bond_alignment`)

These two features quantify, respectively, the average and variance of the tensor algebraic environment alignment (from Section 4.4.4) for the three bonds comprising a 2-simplex. They reflect the average degree of alignment of bonds with the local atomic environment within the triangle and its fluctuation.
$$
\begin{aligned}
\text{avg\_bond\_alignment} &= \frac{1}{3} \sum_{b \in \text{triangle bonds}} \text{Alignment}_b \\
\text{variance\_bond\_alignment} &= \frac{1}{3} \sum_{b \in \text{triangle bonds}} (\text{Alignment}_b - \text{avg\_bond\_alignment})^2
\end{aligned}
$$
where $\text{Alignment}_b$ is the tensor algebraic environment alignment of bond $b$.

Physically, high bond alignment indicates that the bond orientation is highly consistent with the intrinsic symmetry of the local atomic arrangement, predicting a more stable structure. High variance, conversely, may indicate non-uniformity of orientation in the bonding network, which could lead to anisotropy or localized stress concentrations. Chemically, they relate to the rigidity of chemical bonds, orientation selectivity, and crystal field effects. In materials science, these features are key indicators for predicting mechanical anisotropy, optoelectronic response, and preferential crystal growth directions, crucial for understanding crystal growth and macroscopic property anisotropy. In computer science and artificial intelligence fields, these features help machine learning models predict material mechanical strength, toughness, and deformation behavior under external stress.

### 5.3.5 Bond Density Gradient Statistics (`avg_bond_gradient`, `max_bond_gradient`)

These two features quantify, respectively, the average and maximum of the bond density gradients (from Section 4.3.4) for the three bonds comprising a 2-simplex. They reflect the average intensity and maximum change point of electron density variation within the triangular region.
$$
\begin{aligned}
\text{avg\_bond\_gradient} &= \frac{1}{3} \sum_{b \in \text{triangle bonds}} |\nabla \rho|_b \\
\text{max\_bond\_gradient} &= \max_{b \in \text{triangle bonds}} (|\nabla \rho|_b)
\end{aligned}
$$
where $|\nabla \rho|_b$ is the magnitude of the bond density gradient of bond $b$.

Physically, a high average bond density gradient indicates rapid electron density changes within the region, potentially due to strong covalent bonding or charge transfer pathways. The maximum value, conversely, indicates the bond within the triangle with the steepest electron density change. Chemically, these features relate to the strength of covalent bonds, bond dissociation energies, and active sites for chemical reactions. In materials science, they are key indicators for designing and optimizing catalysts, battery electrode materials, and optoelectronic conversion materials, as they directly relate to material electron transport efficiency and chemical reactivity. In computer science and artificial intelligence fields, these features help machine learning models predict material catalytic activity, electron mobility, and performance in electrochemical reactions.

### 5.3.6 Bond Length Statistics (`avg_bond_distance`, `variance_bond_distance`)

These two features quantify, respectively, the average and variance of the bond lengths (from Section 4.2.1) for the three bonds comprising a 2-simplex. They reflect the average level and non-uniformity of bond lengths within the triangular region.
$$
\begin{aligned}
\text{avg\_bond\_distance} &= \frac{1}{3} \sum_{b \in \text{triangle bonds}} L_b \\
\text{variance\_bond\_distance} &= \frac{1}{3} \sum_{b \in \text{triangle bonds}} (L_b - \text{avg\_bond\_distance})^2
\end{aligned}
$$
where $L_b$ is the bond length of bond $b$.

Physically, the average bond length reflects the overall strength of bonding and interatomic distances in the region. The variance, conversely, measures the degree of bond length distortion, with high variance potentially indicating non-uniform strain or defects in the bonding network. Chemically, they directly relate to chemical bond stability, bond energy, and local geometric configuration. In materials science, these features are key indicators for predicting mechanical properties (e.g., modulus, hardness), thermal expansion coefficients, and phase behavior, crucial for understanding crystal structural stability and mechanical response. In computer science and artificial intelligence fields, these features help machine learning models predict material mechanical strength, thermal stability, and structural changes under different temperatures and pressures.

## 5.4 Quantum Chemical Features

These features are extracted directly from the electron density and Electron Localization Function (ELF) fields obtained from first-principles calculations, used to quantify the electronic structure characteristics at the geometric centroid of a 2-simplex, thereby revealing the quantum nature of the local electronic environment.

### 5.4.1 Geometric Centroid Electron Density (`geometric_centroid_density`)

`geometric_centroid_density` descriptor quantifies the electron density value at the geometric centroid of a 2-simplex (triangle). This value is obtained by high-precision trilinear interpolation of the 3D electron density grid for the entire unit cell.
$$
\rho_{\text{centroid}} = \text{Interpolate}(\rho_{\text{grid}}, \mathbf{r}_{\text{centroid}})
$$
where $\rho_{\text{grid}}$ is the electron density grid data obtained from DFT calculation, and $\mathbf{r}_{\text{centroid}}$ is the Cartesian coordinate of the triangle's geometric centroid.

Physically, geometric centroid electron density directly reflects the degree of electron enrichment at that spatial point, which is closely related to interatomic bonding, charge transfer, and local chemical environment. High electron density usually indicates the presence of covalent bonds or local electron delocalization. Chemically, it provides microscopic information for understanding bonding properties and reactivity. In materials science, `geometric_centroid_density` is an important indicator for predicting material conductivity, photoelectric conversion efficiency, and catalytic active sites, as electron density is a fundamental factor determining these macroscopic properties. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of the local electronic environment and bonding properties within materials, aiding models in predicting material electrical, optical, and catalytic properties.

### 5.4.2 Geometric Centroid Electron Localization Function (`geometric_centroid_elf`)

`geometric_centroid_elf` descriptor quantifies the Electron Localization Function (ELF) value at the geometric centroid of a 2-simplex (triangle). This value is obtained by high-precision trilinear interpolation of the 3D ELF grid for the entire unit cell.
$$
\text{ELF}_{\text{centroid}} = \text{Interpolate}(\text{ELF}_{\text{grid}}, \mathbf{r}_{\text{centroid}})
$$
where $\text{ELF}_{\text{grid}}$ is the ELF grid data obtained from the GPAW calculator, and $\mathbf{r}_{\text{centroid}}$ is the Cartesian coordinate of the triangle's geometric centroid.

Physically, ELF is a dimensionless function between 0 and 1, used to identify regions of electron localization (high values, e.g., covalent bonds and lone pairs) and delocalization (low values, e.g., metallic bonds and van der Waals regions). The ELF value at the geometric centroid can precisely reveal whether electron pairs are localized at that spatial point, thereby determining its bonding characteristics. Chemically, high ELF values indicate strong covalent bond characteristics or the presence of lone pairs, while low values may indicate ionic bonds or weak interactions. In materials science, `geometric_centroid_elf` is a key feature for analyzing material chemical bonding types, interatomic interaction strengths, and for determining material stability and mechanical behavior. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of the local bonding type and electron localization characteristics within materials, aiding models in predicting material mechanical strength, thermal stability, and chemical reactivity.

### 5.4.3 Geometric Centroid Laplacian of Electron Density (`geometric_centroid_laplacian_of_density`)

`geometric_centroid_laplacian_of_density` descriptor quantifies the Laplacian of electron density ($\nabla^2 \rho$) value at the geometric centroid of a 2-simplex (triangle). This value is obtained by high-precision trilinear interpolation of the 3D $\nabla^2 \rho$ grid for the entire unit cell.
$$
\nabla^2 \rho_{\text{centroid}} = \text{Interpolate}(\nabla^2 \rho_{\text{grid}}, \mathbf{r}_{\text{centroid}})
$$
where $\nabla^2 \rho_{\text{grid}}$ is the Laplacian field calculated from the electron density grid data, and $\mathbf{r}_{\text{centroid}}$ is the Cartesian coordinate of the triangle's geometric centroid.

Physically, the sign of the electron density Laplacian provides key information about electron localization and deformation: $\nabla^2 \rho < 0$ indicates electron accumulation (usually at bond centers of covalent bonds and lone pair regions), while $\nabla^2 \rho > 0$ indicates electron depletion (usually at atomic nuclei regions or interatomic regions of ionic bonds). The Laplacian value at the geometric centroid can precisely identify whether this point is an electron accumulation basin or a depletion region. Chemically, negative values indicate the presence of bonding interactions, while positive values indicate non-bonding regions. In materials science, `geometric_centroid_laplacian_of_density` is an important feature for understanding material bonding properties, charge transport pathways, and local stress distribution, especially with unique advantages in analyzing the nature and strength of chemical bonds. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of local electron deformation and bonding topology within materials, aiding models in predicting material electrical properties, mechanical response, and chemical reactivity.

## 5.5 Algebraic Fusion Features

These features fuse concepts of structure tensors, density gradient fields, point group symmetry, and hierarchical stress through algebraic methods to capture higher-order interactions, fluxes, symmetry reductions, and complex stress flow patterns at the 2-simplex level.

### 5.5.1 Trace of Structural Tensor Product (`structural_tensor_product_trace`)

`structural_tensor_product_trace` descriptor quantifies the higher-order interactions of the local structure tensors $\mathbf{T}_i, \mathbf{T}_j, \mathbf{T}_k$ corresponding to the three atoms forming a 2-simplex (from the structure tensor in Section 3.4.2). It is defined as the trace of the product of these three tensors:
$$
\text{Tr}(\mathbf{T}_i \cdot \mathbf{T}_j \cdot \mathbf{T}_k)
$$
where $\mathbf{T}_i, \mathbf{T}_j, \mathbf{T}_k$ are the 3x3 structure tensors of atoms $i, j, k$ respectively.

Physically, this feature quantifies the complex coupling effects of the local structural environments of three neighboring atoms. Non-zero values indicate non-linear interactions of local geometry and force fields, which may be caused by structural distortion, bond strain, or collective atomic cooperative motion. The larger the absolute value, the more significant this higher-order interaction. Chemically, it can capture synergistic bonding or repulsive effects among three interacting centers. In materials science, `structural_tensor_product_trace` is a key indicator for evaluating material mechanical response, phonon scattering, and many-body effects during phase transitions, crucial for understanding coupling phenomena in complex crystal structures. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of higher-order structural coupling and many-body interactions within materials, aiding models in predicting material nonlinear mechanical properties, thermodynamic stability, and complex behavior under external fields.

### 5.5.2 Total Density Gradient Flux (`total_density_gradient_flux`)

`total_density_gradient_flux` descriptor quantifies the net flux of the electron density gradient vector field through the plane of a 2-simplex (triangle). It is approximated by calculating the dot product of the average electron density gradient vector at the triangle's geometric centroid and the unit normal vector to the triangle's plane, then multiplying by the triangle's area.
$$
\Phi = (\frac{\nabla \rho_i + \nabla \rho_j + \nabla \rho_k}{3}) \cdot \mathbf{n} \cdot \text{Area}
$$
where $\nabla \rho_i, \nabla \rho_j, \nabla \rho_k$ are the electron density gradient vectors at atoms $i, j, k$ respectively (obtained via interpolation), $\mathbf{n}$ is the unit normal vector to the triangle's plane, and `Area` is the triangle's area. This feature relies on electron density gradient fields obtained from plane wave or FD mode GPAW calculators.

Physically, total density gradient flux reflects the net flow trend of electron density through the triangle plane. Positive values indicate electrons tend to flow out of the plane, negative values indicate flow in. This is closely related to charge transport pathways, local electric field strength, and bond polarity. Chemically, it reveals the direction and strength of electron transfer in three-atom systems. In materials science, `total_density_gradient_flux` is a key feature for designing and optimizing electrochemical materials (e.g., battery electrodes, electrolytes), semiconductors, and catalysts, as it directly relates to material charge transport efficiency and active sites. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of the charge transport microenvironment and local charge polarization within materials, aiding models in predicting material conductivity, ionic conductivity, and catalytic activity.

### 5.5.3 Point Group Reduction Factor (`point_group_reduction_factor`)

`point_group_reduction_factor` descriptor quantifies the reduction degree of the local point group symmetry order (from Section 3.4.3) for the three atoms comprising a 2-simplex, relative to their average value. It is defined as the ratio of the minimum of the three atomic site symmetry orders to their average value:
$$
\text{Reduction Factor} = \frac{\min(\text{Order}_i, \text{Order}_j, \text{Order}_k)}{\text{avg}(\text{Order}_i, \text{Order}_j, \text{Order}_k)}
$$
where $\text{Order}_i$ is the order of the point group of atom $i$'s site.

Physically, this feature reflects the severity of local symmetry breaking. Lower values indicate that at least one atom in the triangle has significantly lower symmetry than the average, which may predict local structural distortions or defects. Higher values indicate more consistent local symmetry among the three atoms. Chemically, it reveals the regularity of the local coordination environment, crucial for understanding the orientation selectivity of chemical bonds and crystal field effects. In materials science, `point_group_reduction_factor` is a key indicator for evaluating material structural stability, phase behavior, and predicting macroscopic anisotropy (e.g., optical, electrical anisotropy), especially for functional materials. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of local symmetry heterogeneity and the degree of symmetry breaking within materials, aiding models in identifying and predicting lattice distortions, precursor states to phase transitions, and specific functional materials.

### 5.5.4 Structure Tensor Normal Projection (`structure_tensor_normal_projection`)

`structure_tensor_normal_projection` descriptor quantifies the projection of the total structure tensor (from Section 3.4.2) of the three atoms forming a 2-simplex onto the normal of the triangle's plane. It is defined as the double dot product of the unit normal vector $\mathbf{n}$ to the triangle's plane and the total structure tensor $\mathbf{T}_{\text{total}} = \mathbf{T}_i + \mathbf{T}_j + \mathbf{T}_k$. Its rigorous expression and equivalence to matrix multiplication are derived as follows:

Let the unit normal vector $\mathbf{n}$ be a column vector with components $\mathbf{n} = \begin{pmatrix} n_x \\
        n_y \\
        n_z 
        \end{pmatrix}$.
Let the total structure tensor $\mathbf{T}_{\text{total}}$ be a $3 \times 3$ matrix with components $\mathbf{T}_{\text{total}} = \begin{pmatrix} T_{xx} & T_{xy} & T_{xz} \\
        T_{yx} & T_{yy} & T_{yz} \\
        T_{zx} & T_{zy} & T_{zz} 
        \end{pmatrix}$.

1.  **Component Expansion of Double Dot Product $\mathbf{n} \cdot \mathbf{T}_{\text{total}} \cdot \mathbf{n}$**

First, calculate the dot product $\mathbf{v} = \mathbf{n} \cdot \mathbf{T}_{\text{total}}$ of vector $\mathbf{n}$ and tensor $\mathbf{T}_{\text{total}}$. This operation will yield a new vector $\mathbf{v}$, whose $j$-th component is:
$$
v_j = \sum_i n_i T_{ij}
$$
Then, calculate the dot product of the resulting vector $\mathbf{v}$ with vector $\mathbf{n}$:
$$
\mathbf{n} \cdot \mathbf{T}_{\text{total}} \cdot \mathbf{n} = \mathbf{v} \cdot \mathbf{n} = \sum_j v_j n_j
$$
Substituting the expression for $v_j$, the final component form of the double dot product is obtained:
$$
\mathbf{n} \cdot \mathbf{T}_{\text{total}} \cdot \mathbf{n} = \sum_j \left( \sum_i n_i T_{ij} \right) n_j = \sum_i \sum_j n_i T_{ij} n_j \quad
$$

2.  **Component Expansion of Matrix Product $\mathbf{n}^T \mathbf{T}_{\text{total}} \mathbf{n}$**

The transpose of vector $\mathbf{n}$ is the row vector $\mathbf{n}^T = \begin{pmatrix} n_x & n_y & n_z \end{pmatrix}$.
First, calculate the product $\mathbf{u} = \mathbf{T}_{\text{total}} \mathbf{n}$ of matrix $\mathbf{T}_{\text{total}}$ and column vector $\mathbf{n}$. This operation will yield a new column vector $\mathbf{u}$, whose $i$-th component is:
$$
u_i = \sum_j T_{ij} n_j
$$
Then, calculate the product of row vector $\mathbf{n}^T$ and column vector $\mathbf{u}$:
$$
\mathbf{n}^T \mathbf{T}_{\text{total}} \mathbf{n} = \mathbf{n}^T \mathbf{u} = \sum_i n_i u_i
$$
Substituting the expression for $u_i$, the final component form of the matrix product is obtained:
$$
\mathbf{n}^T \mathbf{T}_{\text{total}} \mathbf{n} = \sum_i n_i \left( \sum_j T_{ij} n_j \right) = \sum_i \sum_j n_i T_{ij} n_j \quad
$$

Conclusion:

By comparing the component expanded results of (1) and (2), we can see that they are identical. Therefore, under tensor and matrix representations, the following equality holds:
$$
\mathbf{n} \cdot \mathbf{T}_{\text{total}} \cdot \mathbf{n} = \mathbf{n}^T \mathbf{T}_{\text{total}} \mathbf{n}
$$
This value is a scalar that quantifies the strength of the total structure tensor component in the direction normal to the triangle's plane, reflecting the anisotropy or stress condition of the triangle's plane within the overall structure. This scalar value not only has a clear mathematical definition, but its physical and chemical significance also provides profound insights into the connection between the microscopic structure and macroscopic properties of materials. From a multi-disciplinary perspective, the meaning of this feature can be further elaborated as follows:

Physically, this feature reflects the component of local structural stress or distortion field perpendicular to the triangle's plane. A larger absolute value indicates that the triangle's plane is subjected to perpendicular compressive or tensile stress, or there is structural asymmetry perpendicular to this plane. Chemically, it is related to the distribution of bonding forces in 3D space and the deformation of local electron clouds. In materials science, `structure_tensor_normal_projection` is a key indicator for evaluating material anisotropic mechanical response, piezoelectric effects, and structural stability at specific crystal planes, crucial for understanding stress transfer within lattice planes and crystal plane reconstruction. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of local stress directionality and crystal plane response within materials, aiding models in predicting material macroscopic mechanical properties, piezoelectric properties, and structural stability under external stress.

### 5.5.5 Hierarchical Stress Flow (`hierarchical_stress_flow`)

`hierarchical_stress_flow` descriptor quantifies the net flux of "stress" vectors, whose "magnitude" is given by bond incompatibility (from `lie_algebr-incompatibility` in Section 4.4.1), flowing out of each side of the triangle. It is obtained by multiplying the Lie algebraic incompatibility of each bond by its unit normal vector perpendicular to that bond within the triangle's plane and pointing outwards, then vector-summing over all three sides, and finally taking the magnitude of the resulting vector.
$$
\begin{aligned}
\mathbf{F}_{\text{stress}} &= \sum_{b \in \text{triangle bonds}} \text{Incomp}_{b} \cdot \mathbf{n}_{b}^{\perp} \\
\text{StressFlow} &= \|\mathbf{F}_{\text{stress}}\|_2
\end{aligned}
$$
where $\text{Incomp}_{b}$ is the Lie algebraic incompatibility of bond $b$, and $\mathbf{n}_{b}^{\perp}$ is the unit normal vector perpendicular to bond $b$ and pointing outwards within the triangle's plane.

Physically, this feature captures how "stress" caused by bonding incompatibility within a 2-simplex is transmitted and accumulated in the local region, forming a pseudo-vector field. High values indicate significant, directional internal stress accumulation or release in that triangular region. Chemically, it reveals synergistic or antagonistic effects of bonding stress in three-body interactions. In materials science, `hierarchical_stress_flow` is a key indicator for understanding material plastic deformation, fatigue crack initiation, and collective atomic motion during phase transitions, crucial for analyzing material failure mechanisms under complex stress conditions. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of local stress transmission pathways and energy dissipation modes within materials, aiding models in predicting material plasticity, toughness, and fatigue life under cyclic loading.
