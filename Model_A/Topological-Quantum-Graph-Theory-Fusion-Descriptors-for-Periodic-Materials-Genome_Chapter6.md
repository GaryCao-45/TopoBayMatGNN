# Chapter 6: Global Descriptors: Macroscopic Quantification and Characterization of Material Properties

This chapter focuses on the construction and analysis of global descriptors, which aim to quantify the macroscopic physical, chemical, and geometric properties of the entire crystal structure. Unlike the previous chapters that focused on local features of atoms, chemical bonds, and three-body clusters, global descriptors provide a generalized understanding of the material's overall characteristics. They typically serve as complements or higher-level abstractions of local features, playing an irreplaceable role in predicting macroscopic material properties (e.g., total energy, stability, lattice anisotropy, packing efficiency) and in rapid evaluation during material design and screening.

## 6.1 Basic Statistical and Geometric Features: First-Order Quantification of Crystal Macroscopic Structure

This section includes features primarily based on macroscopic geometric parameters and statistical averages of the crystal structure, providing a fundamental description of the material's overall size, shape, packing efficiency, and bonding state. These features are intuitive and easy to calculate, serving as starting points for understanding the global characteristics of crystalline materials.

### 6.1.1 Mean Bond Length (`mean_bond_length`)

`mean_bond_length` descriptor is strictly defined as the arithmetic mean of all chemical bond lengths in the crystal structure. It provides a basic quantification of the material's overall interatomic connection strength and spatial scale. Its calculation directly stems from the global average of the `bond_distance` feature pre-calculated in Chapter 4, 1-simplex descriptors.

Physically, mean bond length is directly related to the material's average interatomic distance, influencing the crystal's thermal expansion coefficient, melting point, and mechanical rigidity. Chemically, it reflects the average bonding type and strength of the material; covalent bonds are typically shorter, while ionic and metallic bonds may exhibit different average lengths. In materials science, mean bond length is an important indicator of material structural stability and density, and an effective feature for quickly assessing the compactness of crystal structures. In computer science and artificial intelligence fields, as one of the most fundamental structural descriptors, it provides machine learning models with direct input on material scale and bonding characteristics, often used in the initial screening phase for material selection and performance prediction.

### 6.1.2 Volume Per Formula Unit (`volume_per_fu`)

`volume_per_fu` descriptor is strictly defined as the crystal's unit cell volume divided by the number of formula units. This quantifies the average space occupied by each unit cell in the material's chemical formula (e.g., "1" unit for $\text{CsPbI}_3$).

Physically, this feature directly reflects the material's macroscopic density and spatial packing efficiency. Lower `volume_per_fu` typically implies higher atomic packing density. Chemically, it is closely related to atomic sizes and the relative contributions of ionic and covalent bonds. In materials science, this feature is a key parameter for evaluating material stability and predicting high-pressure synthesis, as smaller unit volumes typically mean higher energy density or more compact atomic packing, significantly affecting the material's thermodynamic stability. In computer science and artificial intelligence fields, `volume_per_fu` is an important size feature that can be used to predict material density, compressibility, and structural behavior under different conditions.

### 6.1.3 Lattice Anisotropy Ratio (`lattice_anisotropy_ratio`)

`lattice_anisotropy_ratio` descriptor is strictly defined as the ratio of the maximum to the minimum length among the three basis vectors (a, b, c) of the crystal lattice. It quantifies the geometric asymmetry of the crystal in different crystallographic directions, serving as a key indicator for determining whether the material has a layered, chain-like, or highly anisotropic structure.
$$
\text{Lattice Anisotropy Ratio} = \frac{\max(a, b, c)}{\min(a, b, c)}
$$
Physically, a high anisotropy ratio usually indicates that the material possesses different physical properties in different directions, such as mechanical strength, thermal conductivity, electrical conductivity, or optical response. Chemically, it may be related to weak bonds (e.g., van der Waals forces) or strong covalent networks in specific directions, leading to structures that are more easily deformable in certain directions. In materials science, `lattice_anisotropy_ratio` is a key feature for designing materials with directional properties (e.g., 2D materials, thermoelectric materials, photovoltaic materials); high anisotropy is often the source of these materials' unique functionalities. In computer science and artificial intelligence fields, it provides machine learning models with an intuitive quantification of the degree of material structural anisotropy, aiding in predicting material directional transport properties, mechanical stability, and performance in specific application scenarios.

### 6.1.4 Bulk Anisotropy Index (`bulk_anisotropy_index`)

`bulk_anisotropy_index` descriptor aims to quantify the average "shape" anisotropy of atomic local environments in the bulk crystal lattice, thereby serving as a proxy for the material's intrinsic structural anisotropy. It is obtained by calculating the eigenvalue anisotropy of the local structure tensors (from the structure tensor concept introduced in Section 3.4.2) for all inequivalent sites (identified by space group analysis) in the crystal, and then taking their average. Specifically, for each site $i$, its structure tensor $\mathbf{T}_i$ has three eigenvalues $\lambda_{i,1} \ge \lambda_{i,2} \ge \lambda_{i,3}$, and the local anisotropy index $\et-i = (\lambda_{i,1} - \lambda_{i,3}) / (\lambda_{i,1} + \lambda_{i,2} + \lambda_{i,3})$ is calculated. The final `bulk_anisotropy_index` is the average of all inequivalent site $\et-i$.
$$
\text{Bulk Anisotropy Index} = \frac{1}{N_{\text{unique}}} \sum_{i \in \text{unique sites}} \frac{\lambda_{i,1} - \lambda_{i,3}}{\sum_{j=1}^3 \lambda_{i,j}}
$$
where $N_{\text{unique}}$ is the number of inequivalent sites, and $\lambda_{i,j}$ are the eigenvalues of the structure tensor for the $i$-th site.

Physically, this feature captures the local symmetry breaking and non-spherical characteristics of atomic arrangements within the crystal, serving as a direct manifestation of microscopic structural stress, local distortions, and anisotropic lattice vibrations. Chemically, it is closely related to interatomic bonding directionality, coordination environment distortion, and electron orbital hybridization patterns. In materials science, `bulk_anisotropy_index` is an important indicator for predicting material elastic anisotropy, phonon transport, thermal conductivity, and even functional properties like ferroelectricity and piezoelectricity, as these properties often originate from microscopic structural anisotropy. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of structural non-uniformity and local strain fields within materials, aiding models in understanding and predicting material responses under complex environments, thereby accelerating material design and performance optimization.

### 6.1.5 Octahedral Count (`octahedral_count`)

`octahedral_count` descriptor strictly calculates the number of octahedra in the crystal structure that conform to specific chemical and geometric standards. To ensure physical correctness, this feature only counts atoms satisfying the following conditions: (1) the central atom must be a metal (e.g., Pb); (2) its coordination number must be exactly 6; (3) all 6 nearest neighbors must be non-metals (e.g., Br). For central atoms meeting these criteria, the system calculates their local structural order parameter; only when this order parameter is greater than a preset physical threshold of 0.75 is it considered a valid octahedron.

Physically, octahedra are fundamental structural units in many functional materials like perovskites, and their number and connectivity directly determine the material's crystal structure type, stability, and many macroscopic physical properties, such as band gap, exciton binding energy, and ferroelectricity. Chemically, the presence and integrity of octahedra reflect the bonding characteristics and coordination environment stability between the central metal ion and coordinating non-metal ions. In materials science, `octahedral_count` is a key characterization for octahedron-containing materials like perovskites and spinels; for example, in perovskite photovoltaic materials, the degree of octahedral tilt and distortion directly affects carrier transport efficiency and material stability. In computer science and artificial intelligence fields, it provides machine learning models with a direct quantification of the number of basic structural units in materials, aiding models in identifying the impact of specific structural motifs on material properties, and is an important feature for material classification and performance prediction.

### 6.1.6 Packing Fraction (`packing_fraction`)

`packing_fraction` descriptor is strictly defined as the ratio of the sum of the ionic volumes of all atoms in the unit cell to the total volume of the unit cell. It quantifies the atomic spatial packing efficiency of the crystal structure, reflecting the degree to which atoms are tightly packed in the unit cell. Its calculation relies on pre-calculated ionic radii data obtained from Section 4.2.6, ensuring atomic order matches the CIF file. The specific calculation formula is:
$$
\text{Packing Fraction} = \frac{\sum_{i=1}^{N} \frac{4}{3}\pi r_i^3}{V_{\text{cell}}}
$$
where $N$ is the total number of atoms in the unit cell, $r_i$ is the ionic radius of the $i$-th atom, and $V_{\text{cell}}$ is the volume of the unit cell.

Physically, a high packing fraction usually implies a more stable structure and higher density, influencing material hardness, melting point, and phonon propagation speed. Chemically, it is related to interatomic repulsion and bonding types; close packing typically occurs in materials with stronger isotropic bonding. In materials science, `packing_fraction` is an important indicator for evaluating material thermodynamic stability, predicting phase behavior, and designing new material structures; for example, in glasses and amorphous materials, packing fraction is closely related to glass-forming ability and structural relaxation behavior. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of the compactness of material atomic arrangements, aiding models in predicting material mechanical properties, thermodynamic stability, and structural behavior under different pressure and temperature conditions.

## 6.2 DFT-Calculated Ground State Properties: Macroscopic Insight into Material Energy and Electronic Structure

The global descriptors in this section are directly derived from the crystal ground state properties obtained through density functional theory (DFT) calculations. These features quantify the material's overall energy stability, electronic energy level characteristics, and average electrostatic environment, serving as key elements for understanding and predicting material thermodynamic stability, electron transport properties, and chemical reactivity.

### 6.2.1 Total Energy per Atom (`total_energy_per_atom`)

`total_energy_per_atom` descriptor is strictly defined as the total energy of the crystal system divided by the number of atoms in the unit cell. It represents the average energy per atom in the material at absolute zero temperature, serving as one of the most fundamental and important indicators of material thermodynamic stability.
$$
E_{\text{atom}} = \frac{E_{\text{total}}}{N_{\text{atoms}}}
$$
where $E_{\text{total}}$ is the total energy of the system, and $N_{\text{atoms}}$ is the total number of atoms in the unit cell.

Physically, lower total energy per atom typically implies higher thermodynamic stability of the material structure, as it resides in a deeper energy well. Chemically, this feature reflects the overall strength and stability of interatomic bonding; lower energy indicates stronger, more optimized bonding. In materials science, `total_energy_per_atom` is a key parameter for predicting material phase stability, assessing the feasibility of synthesizing new materials, and evaluating defect formation energies, playing a central role in crystal structure searching and phase diagram construction. In computer science and artificial intelligence fields, it is one of the most commonly used output or input features in material property prediction models, directly related to material formation energy, decomposition energy, and thermodynamic phase transition behavior, having a decisive impact on machine learning models' understanding of material stability.

### 6.2.2 Fermi Level (`fermi_level`)

`fermi_level` descriptor is strictly defined as the Fermi level of the crystal system. At absolute zero temperature, the Fermi level is the highest occupied energy state for electrons. For conductors, it lies within the conduction band; for semiconductors and insulators, it lies within the band gap. The position of the Fermi level determines the material's conductivity type and the ease of electron transitions.

Physically, the Fermi level is a core concept in electronic structure and electron transport properties. Its precise position has a decisive impact on a material's electrical properties (e.g., conductivity, carrier concentration), thermoelectric properties, and optical absorption edge. Chemically, the Fermi level can be regarded as the material's chemical potential, reflecting the ease with which electrons are gained or lost; a high Fermi level means the material tends to lose electrons (strong reducing ability), while a low Fermi level means it tends to gain electrons (strong oxidizing ability). In materials science, `fermi_level` is a key parameter for designing semiconductors, optoelectronic devices, thermoelectric materials, and catalysts; for example, in photovoltaic materials, the relative position of the Fermi level to the band edges directly affects solar energy conversion efficiency. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material electronic states and transport properties, serving as an important feature for predicting material conductivity, optical absorption, catalytic activity, and designing novel electronic devices.

### 6.2.3 Mean Electrostatic Potential (`electrostatic_potential_mean`)

`electrostatic_potential_mean` descriptor is strictly defined as the average value of the electrostatic potential over all spatial grid points within the crystal system. The electrostatic potential is the energy experienced by charged particles in space due to electric fields, and its average value reflects the overall uniformity of charge distribution and the average potential environment within the crystal.
$$
\Phi_{\text{mean}} = \frac{1}{N_{\text{grid}}} \sum_{j=1}^{N_{\text{grid}}} \Phi(\mathbf{r}_j)
$$
where $\Phi(\mathbf{r}_j)$ is the electrostatic potential at grid point $\mathbf{r}_j$, and $N_{\text{grid}}$ is the total number of effective grid points.

Physically, the mean electrostatic potential is related to the overall charge neutrality of the system and the average strength of long-range electrostatic interactions. It provides a macroscopic potential baseline that aids in understanding the crystal's response to external fields. Chemically, although a single average value may not reveal local charge transfer, it provides a preliminary assessment of the overall charge environment. In materials science, `electrostatic_potential_mean` can serve as an auxiliary indicator for evaluating material dielectric properties, surface work function, and charge injection/extraction efficiency; for example, in capacitor and battery materials, the average potential environment affects charge storage and transport. In computer science and artificial intelligence fields, it provides machine learning models with a macroscopic description of the material's overall electrostatic environment, aiding in predicting material electrical properties, surface behavior, and response under electric fields.

### 6.2.4 Variance of Electrostatic Potential (`electrostatic_potential_variance`)

`electrostatic_potential_variance` descriptor is strictly defined as the variance of the electrostatic potential over all spatial grid points within the crystal system. It quantifies the dispersion or non-uniformity of the electrostatic potential distribution within the crystal. High variance means the electrostatic potential fluctuates greatly in space, potentially indicating significant non-uniform charge distribution or polarization effects; low variance indicates a relatively uniform potential distribution.
$$
\sigma^2_{\Phi} = \frac{1}{N_{\text{grid}}} \sum_{j=1}^{N_{\text{grid}}} (\Phi(\mathbf{r}_j) - \Phi_{\text{mean}})^2
$$
where $\Phi(\mathbf{r}_j)$ is the electrostatic potential at grid point $\mathbf{r}_j$, $\Phi_{\text{mean}}$ is the mean electrostatic potential, and $N_{\text{grid}}$ is the total number of effective grid points.

Physically, the variance of electrostatic potential is an important indicator for quantifying charge non-uniformity and local polarization strength within a crystal. It is directly related to a material's dielectric constant, dipole moment, and potential ferroelectric or piezoelectric properties. Chemically, high electrostatic potential variance typically implies strong ionic bonding, significant charge transfer, or localized charge accumulation regions. In materials science, `electrostatic_potential_variance` is a key feature for designing high-performance dielectric materials, ferroelectrics, piezoelectrics, and novel catalysts, as it directly reflects the complexity and functionality of charge distribution within the material. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of charge fluctuations and local field effects within materials, aiding in predicting material functional properties, charge transport pathways, and responses under external fields, serving as a crucial input for understanding and optimizing functional materials.

## 6.3 Global Higher-Order Algebraic Features: Abstract Quantification Beyond Traditional Descriptions

This section introduces global descriptors that are among the core innovations of this framework. They incorporate abstract quantifications derived from higher-order mathematical concepts such as geometric algebra, Lie algebra, and symplectic geometry. These features aim to capture deeper structural symmetries, force field responses, and complex couplings between electronic and structural degrees of freedom in crystalline materials, providing a more refined and physically insightful description of macroscopic functional properties.

### 6.3.1 Quotient Algebra and Structural Geometric Features

This subsection's features primarily utilize the concept of quotient algebra to quantify crystal structural symmetry and describe global structural asymmetry from a geometric algebra perspective. They offer an abstract and precise way to measure the degree of order and disorder in a crystal, as well as its deviation from ideal symmetry.

#### 6.3.1.1 Structural Hash Value (`structure_hash`)

The `structure_hash` descriptor is strictly defined as a unique graph hash value for the crystal structure. It abstracts the crystal structure as a graph (atoms as nodes, chemical bonds as edges) and uses the Weisfeiler-Lehman algorithm to generate a compact, highly distinctive string. This hash value uniquely identifies a crystal structure, even if its space group is the same but atomic arrangement details differ.

Physically, the structural hash value itself does not directly have physical dimensions, but its uniqueness ensures precise identification of material "fingerprints." It can quickly distinguish different crystal structure isomers, even if they may appear macroscopically similar. Chemically, it ensures accurate identification of isomers and polymorphic materials, facilitating the establishment and management of material databases. In materials science, `structure_hash` is a key tool for structural identification, deduplication, and building material database indices in materials informatics, which is significant for accelerating high-throughput calculations and experimental screening. In computer science and artificial intelligence fields, it provides machine learning models with a discretized representation of material structural uniqueness, particularly suitable for material classification, recommendation systems, and matching specific structural targets in inverse design based on structural similarity.

#### 6.3.1.2 Symmetry Orbit Connectivity (`symmetry_orbit_connectivity`)

The `symmetry_orbit_connectivity` descriptor is strictly defined as the proportion of chemical bonds between different symmetry-equivalent atomic orbits in the crystal structure to the total number of chemical bonds. It quantifies the degree of symmetry breaking within the crystal and the structural connection strength between different symmetry-independent regions.
$$
\text{Symmetry Orbit Connectivity} = \frac{N_{\text{inter-orbit bonds}}}{N_{\text{total bonds}}}
$$
where $N_{\text{inter-orbit bonds}}$ is the number of chemical bonds connecting different symmetry-equivalent atoms, and $N_{\text{total bonds}}$ is the total number of chemical bonds in the crystal.

Physically, a high `symmetry_orbit_connectivity` value usually implies a more complex crystal structure, potentially accompanied by lower symmetry and strong interactions between different crystallographic sites. This may lead to macroscopic functional properties such as polarity, chirality, or anisotropy in the material. Chemically, it reflects how atoms in different chemical environments are connected through bonding, which is crucial for understanding the formation and stability of complex compounds. In materials science, `symmetry_orbit_connectivity` is a key feature for designing functional materials with specific symmetry-dependent properties; for example, in ferroelectrics, symmetry breaking leads to atomic displacements, which in turn generate electric polarization. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of the degree of symmetry breaking within the material and its impact on long-range structural order, helping models predict symmetry-related material functions and guide the discovery of new functional materials.

#### 6.3.1.3 Global Asymmetry Norm (`global_asymmetry_norm`)

The `global_asymmetry_norm` descriptor is strictly defined as the average of the Euclidean norms of all atomic local asymmetry vectors in the crystal structure. It quantifies the overall "average local asymmetry" or "average local distortion degree" of the crystal, rather than simply summing all local vectors (which might result in vector cancellation and fail to reflect actual local distortions). Each atom's local asymmetry vector $V_{\text{struct}}$ (from Section 3.4.1) represents the degree to which its coordination environment deviates from ideal symmetry. The final `global_asymmetry_norm` is the average of the Euclidean norms $\|V_{\text{struct}}\|$ of all atomic local asymmetry vectors.
$$
\text{Global Asymmetry Norm} = \frac{1}{N_{\text{atoms}}} \sum_{i=1}^{N_{\text{atoms}}} \|V_{\text{struct},i}\|
$$
where $N_{\text{atoms}}$ is the total number of atoms in the unit cell, and $\|V_{\text{struct},i}\|$ is the Euclidean norm of the local asymmetry vector for the $i$-th atom.

Physically, `global_asymmetry_norm` provides a macroscopic quantification of the overall structural distortion of the crystal. High values indicate widespread local structural distortions within the crystal, which may be related to the material's phase transition behavior, low-frequency phonon modes, and certain functional properties (e.g., ferroelectricity, piezoelectricity). Chemically, it reflects the average deviation of coordination environments from their ideal states in the crystal, helping to understand the heterogeneity of bonding environments. In materials science, this feature is an important indicator for evaluating material structural stability and functionality, especially suitable for analyzing functional materials like perovskites with complex structural distortions (e.g., octahedral tilting, atomic displacements). In computer science and artificial intelligence fields, it provides machine learning models with a continuous and differentiable quantification of the overall structural distortion and strain field of materials, serving as a key feature for predicting material structural phase transitions, mechanical responses, and symmetry-related functions (e.g., polarity).

#### 6.3.1.4 Lie Asymmetry Magnitude Entropy (`lie_asymmetry_magnitude_entropy`)

The `lie_asymmetry_magnitude_entropy` descriptor is strictly defined as the Shannon entropy of the distribution of the Euclidean norms of all atomic local asymmetry vectors ($\|V_{\text{struct}}\|$) in the crystal. It quantifies the diversity or complexity of the distribution of local structural asymmetry degrees within the crystal. High entropy values indicate a wide distribution of atomic local asymmetry degrees, suggesting a more disordered or complex structure with various degrees of local distortions; low entropy values indicate similar degrees of atomic local asymmetry, with a more regular structure. This entropy is obtained by histogramming the local asymmetry norms and then calculating the Shannon entropy of their probability distribution.
$$
H = -\sum_{k=1}^{M} p_k \log_2(p_k)
$$
where $M$ is the number of bins in the histogram, and $p_k$ is the probability of the norm falling into the $k$-th bin.

Physically, this entropy provides a measure of the "information content" or "complexity" of the crystal structure, reflecting the diversity of local distortion patterns. High entropy may be associated with disordered phases, glassy states, or materials with complex configurations. Chemically, it reveals the heterogeneity of charge or geometric environments at different atomic sites. In materials science, `lie_asymmetry_magnitude_entropy` helps to understand material structural phase transition mechanisms, degree of disorder, and thermodynamic stability, and is an important feature for evaluating material structural flexibility under specific temperatures or pressures. In computer science and artificial intelligence fields, it provides machine learning models with a high-level abstract quantification of material internal structural variability and complexity, helping models identify and predict functional properties of materials with specific configurational diversity (e.g., for high-entropy alloys) or structural disorder (e.g., for amorphous materials), thereby playing a key role in the design of complex material systems.

### 6.3.2 Local Torsional Stress Features Based on Forces

This subsection's features combine atomic force information with local structural asymmetry vectors, aiming to quantify "torsional stress" within the crystal induced by local distortions. These features provide deep insights into material mechanical stability and dynamic behavior, particularly useful for understanding inelastic deformation, phonon modes, and phase transition driving forces.

#### 6.3.2.1 Force Covariance Invariant 1 (`force_covariance_invariant_1`)

The `force_covariance_invariant_1` descriptor is strictly defined as the trace of the covariance matrix of forces acting on all atoms in the unit cell. It quantifies the overall dispersion of atomic force distribution in the crystal, reflecting the average stress fluctuation or force field inhomogeneity within the material. The calculation of this feature requires first obtaining the force vectors $\mathbf{f}_i$ acting on all atoms from the GPAW calculator, then constructing the force covariance matrix $\mathbf{C}_{\mathbf{f}} = \frac{1}{N_{\text{atoms}}} \sum_i \mathbf{f}_i \mathbf{f}_i^T$, and finally calculating the trace of this matrix $\text{Tr}(\mathbf{C}_{\mathbf{f}})=\sum_k \sigma_{kk}$.
$$
\text{Force Covariance Invariant}_1 = \text{Tr}\left(\frac{1}{N_{\text{atoms}}} \sum_{i=1}^{N_{\text{atoms}}} \mathbf{f}_i \mathbf{f}_i^T\right)
$$
where $\mathbf{f}_i$ is the force vector acting on the $i$-th atom, and $N_{\text{atoms}}$ is the total number of atoms.

Physically, `force_covariance_invariant_1` is related to the material's overall strain energy density, local lattice soft modes, or phonon instabilities. High values may indicate that the material is under high stress, or that there are significant local lattice distortions. Chemically, it reflects the complexity and inhomogeneity of the interatomic force field, helping to understand the heterogeneity of the bonding environment. In materials science, this feature is an important indicator for predicting material mechanical properties (e.g., toughness, brittleness), fatigue behavior, and structural phase transition tendencies; for example, in materials with negative thermal expansion, their internal force fields may exhibit abnormal fluctuations. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material internal force field distribution and stress fluctuations, helping models predict material response under external loads, defect formation energies, and thermodynamic stability, and is a key feature for understanding material mechanical behavior and structural dynamics.

#### 6.3.2.2 Force Covariance Invariant 2 (`force_covariance_invariant_2`)

The `force_covariance_invariant_2` descriptor is strictly defined as the determinant of the covariance matrix of forces acting on all atoms in the unit cell. As the second invariant of the covariance matrix, it further quantifies the multi-dimensional anisotropy and "volume" of the force field distribution. Unlike the trace (total variance), the determinant can capture the correlation of force fluctuations in different orthogonal directions, as well as the overall "divergence" or "shape" of the force field distribution.
$$
\text{Force Covariance Invariant}_2 = \text{Det}\left(\frac{1}{N_{\text{atoms}}} \sum_{i=1}^{N_{\text{atoms}}} \mathbf{f}_i \mathbf{f}_i^T\right)
$$
where $\mathbf{f}_i$ is the force vector acting on the $i$-th atom, and $N_{\text{atoms}}$ is the total number of atoms.

Physically, high `force_covariance_invariant_2` values may imply strong anisotropic distribution of the force field in space, which may be related to the material's inelastic response, specific phonon modes, or anisotropic driving forces of structural phase transitions. Chemically, it can more precisely reveal the complex anisotropy of interatomic interactions, for example, high values may be observed in materials with strong directional bonds (e.g., covalent bonds). In materials science, this feature is an important indicator for predicting material anisotropic mechanical properties, anisotropic heat transport, and understanding the anisotropy of the energy landscape in structural phase transitions. In computer science and artificial intelligence fields, it provides machine learning models with a higher-order quantification of the material's internal force field tensor properties, helping models understand material structural flexibility, deformation mechanisms, and mechanical response in specific directions, and is an advanced feature in material mechanical design and functional prediction.

#### 6.3.2.3 Total Torsional Stress (`total_torsional_stress`)

The `total_torsional_stress` descriptor is strictly defined as the sum of the magnitudes of local torsional stresses acting on all atoms in the crystal. The local torsional stress on a single atom is defined as the Euclidean norm of the cross product of its local structural asymmetry vector $\mathbf{V}_{\text{struct},i}$ (from Section 3.4.1 vectorial asymmetry squared norm) and the force vector $\mathbf{f}_i$ acting on that atom: $\|\mathbf{V}_{\text{struct},i} \times \mathbf{f}_i\|$. This cross product quantifies the "torque" induced by local structural distortions, i.e., the "torsional" effect of the force on structural asymmetry. The final `total_torsional_stress` is the sum of the magnitudes of torsional stresses of all atoms.
$$
\text{Total Torsional Stress} = \sum_{i=1}^{N_{\text{atoms}}} \|\mathbf{V}_{\text{struct},i} \times \mathbf{f}_i\|
$$
where $\mathbf{V}_{\text{struct},i}$ is the local asymmetry vector of the $i$-th atom, and $\mathbf{f}_i$ is the force vector acting on the $i$-th atom.

Physically, `total_torsional_stress` directly reflects the coupling strength between local distortions and the acting force field within the crystal. High values indicate significant local structural distortions and associated internal torques within the crystal, which may lead to structural instability, propensity for structural phase transitions, or nonlinear responses. Chemically, it helps to understand the bonding tension in specific coordination environments and the mode of stress transfer between different atoms. In materials science, `total_torsional_stress` is a key feature for predicting material mechanical properties (e.g., plasticity, creep), precursors to structural phase transitions, and evaluating material response to external stimuli; for example, in ferroelastic materials, this feature can quantify the coupling between structural distortion and elastic stress. In computer science and artificial intelligence fields, it provides machine learning models with a continuous and differentiable quantification of the material's internal "geometric-mechanical" coupling state, and is an important feature for predicting material structural stability, mechanical response, and functional transitions (e.g., phase transitions), having unique application value in material design and optimization.

### 6.3.3 Field-Density Gradient Coupling Features

This subsection's features focus on extracting higher-order information from electron density fields and electrostatic potential fields, quantifying the complex interactions between charge distribution and electric fields within the crystal by constructing a field-density gradient coupling tensor. These features provide deep insights into material electrical response, dielectric properties, and electronic structure stability.

#### 6.3.3.1 Field-Density Coupling Invariant 1 (`field_density_coupling_invariant_1`)

The `field_density_coupling_invariant_1` descriptor is strictly defined as the trace of the field-density gradient coupling tensor $\mathbf{M}$, constructed from the internal electric field $\mathbf{E}(\mathbf{r})$ and electron density gradient $\nabla \rho(\mathbf{r})$ within the crystal. This tensor $\mathbf{M}_{ij} = \langle E_i \nabla_j \rho \rangle$ describes the coupling between the electrostatic field and the electron density gradient, where $\langle \cdot \rangle$ denotes spatial averaging. Its trace $\text{Tr}(\mathbf{M})$ quantifies the total effect of the overall electric field on the spatial variation of electron density, reflecting the material's average "electronic stress" or the responsiveness of its charge distribution. The calculation process involves first obtaining electron density and electrostatic potential grid data from GPAW, then calculating the electric field (negative gradient of electrostatic potential) and electron density gradient, and finally constructing and tracing the tensor.

Its rigorous mathematical expression in continuous form is:
$$
\text{Field Density Coupling Invariant}_1 = \text{Tr}\left( \frac{1}{V_{\text{grid}}} \int_V \mathbf{E}(\mathbf{r}) \otimes \nabla \rho(\mathbf{r}) \, d^3\mathbf{r} \right)
$$
where $V_{\text{grid}}$ is the total unit cell volume of the integration region.

In practical DFT calculations, physical quantities like electron density and electric field are defined on discrete grid points. Therefore, the continuous integral above needs to be approximated by discrete summation. Let there be $N_{\text{grid}}$ uniformly distributed grid points in the unit cell, with each grid point occupying a volume element $d^3\mathbf{r} \approx \Delta V = V_{\text{grid}} / N_{\text{grid}}$.

Then the integral approximation for the field-density gradient coupling tensor $\mathbf{M}$ is:
$$
\begin{aligned}
\mathbf{M} &\approx \frac{1}{V_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k) \Delta V \\
\mathbf{M} &\approx \frac{1}{V_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k) \frac{V_{\text{grid}}}{N_{\text{grid}}} \\
\mathbf{M} &\approx \frac{1}{N_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k)
\end{aligned}
$$
Here, $\mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k)$ is the outer product of the electric field vector and the electron density gradient vector at grid point $\mathbf{r}_k$, resulting in a second-order tensor (matrix).

Then, we take the trace of this approximate tensor. Using the property of the trace of an outer product of tensors $\text{Tr}(\mathbf{A} \otimes \mathbf{B}) = \mathbf{A} \cdot \mathbf{B} = \sum_i A_i B_i$ (i.e., the dot product of two vectors), we get:
$$
\begin{aligned}
\text{Field Density Coupling Invariant}_1 &= \text{Tr}(\mathbf{M}) \\
&\approx \text{Tr}\left( \frac{1}{N_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k) \right) \\
&= \frac{1}{N_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \text{Tr}(\mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k)) \\
&= \frac{1}{N_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \cdot \nabla \rho(\mathbf{r}_k) \\
&= \frac{1}{N_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \sum_i E_i(\mathbf{r}_k) \frac{\partial \rho(\mathbf{r}_k)}{\partial x_i}
\end{aligned}
$$
where $\mathbf{E}(\mathbf{r}_k)$ is the electric field at grid point $\mathbf{r}_k$, $\nabla \rho(\mathbf{r}_k)$ is the electron density gradient, and $N_{\text{grid}}$ is the total number of grid points.

This is completely consistent with the actual calculation method of summing and averaging the point-wise products of electric field and electron density gradient components.

After clarifying the rigorous mathematical definition and calculation method of this feature, its profound physical, chemical, and materials science significance will provide us with key insights into understanding the microscopic charge distribution and macroscopic response mechanisms of materials. The following will elaborate on it from a multi-disciplinary perspective:

Physically, `field_density_coupling_invariant_1` is closely related to the material's dielectric response, polarization intensity, and electron-phonon coupling strength. High values may indicate that the material's electron distribution will change significantly under the action of an electric field, leading to strong dielectric effects or nonlinear optical responses. Chemically, it reflects how electric fields affect local bonding and charge transfer processes. In materials science, this feature is a key indicator for designing high-performance dielectric materials, ferroelectrics, and nonlinear optical materials, providing a quantitative description of the interaction between material microscopic charge distribution and macroscopic electric fields. In computer science and artificial intelligence fields, it provides machine learning models with a continuous and differentiable quantification of the material's internal "electric field-electron response" coupling, helping models predict material response under electric fields, charge transport properties, and electrical polarization behavior, and is a key advanced feature for understanding and optimizing functional materials.

#### 6.3.3.2 Field-Density Coupling Invariant 2 (`field_density_coupling_invariant_2`)

The `field_density_coupling_invariant_2` descriptor is strictly defined as the determinant of the field-density gradient coupling tensor $\mathbf{M}$. As the second invariant of the coupling tensor, it further quantifies the multi-dimensional anisotropy and "non-orthogonality" of the electric field's influence on the spatial variation of electron density. Unlike the trace (total coupling strength), the determinant can capture complex correlations between the electric field and electron density gradient in different orthogonal directions, reflecting the degree of anisotropy of the charge response and the "volume" of the spatial distribution.
Its rigorous mathematical expression in continuous form is:
$$
\text{Field Density Coupling Invariant}_2 = \text{Det}\left( \frac{1}{V_{\text{grid}}} \int_V \mathbf{E}(\mathbf{r}) \otimes \nabla \rho(\mathbf{r}) \, d^3\mathbf{r} \right)
$$
where $V_{\text{grid}}$ is the total unit cell volume of the integration region.
In practical DFT calculations, physical quantities like electron density and electric field are defined on discrete grid points. Therefore, the continuous integral above needs to be approximated by discrete summation. Let there be $N_{\text{grid}}$ uniformly distributed grid points in the unit cell, with each grid point occupying a volume element $d^3\mathbf{r} \approx \Delta V = V_{\text{grid}} / N_{\text{grid}}$.
Then the integral approximation for the field-density gradient coupling tensor $\mathbf{M}$ is:
$$
\begin{aligned}
\mathbf{M} &= \frac{1}{V_{\text{grid}}} \int_V \mathbf{E}(\mathbf{r}) \otimes \nabla \rho(\mathbf{r}) \, d^3\mathbf{r} \\
&\approx \frac{1}{V_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k) \, \Delta V \\
&= \frac{1}{V_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k) \frac{V_{\text{grid}}}{N_{\text{grid}}} \\
&= \frac{1}{N_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k)
\end{aligned}
$$
Therefore, the approximate discrete form of `field_density_coupling_invariant_2` is:
$$
\text{Field Density Coupling Invariant}_2 \approx \text{Det}\left( \frac{1}{N_{\text{grid}}} \sum_{k=1}^{N_{\text{grid}}} \mathbf{E}(\mathbf{r}_k) \otimes \nabla \rho(\mathbf{r}_k) \right)
$$
where $\mathbf{E}(\mathbf{r}_k)$ is the electric field at grid point $\mathbf{r}_k$, $\nabla \rho(\mathbf{r}_k)$ is the electron density gradient, and $N_{\text{grid}}$ is the total number of grid points.

This is completely consistent with the actual calculation method of summing and averaging the point-wise tensor products of electric field and electron density gradient components, and finally calculating the determinant.

After clarifying the rigorous mathematical definition and calculation method of this feature, its profound significance in physics, chemistry, and materials science will provide us with key insights into understanding the microscopic charge distribution and macroscopic response mechanisms of materials. The following will elaborate on it from a multi-disciplinary perspective:

Physically, high `field_density_coupling_invariant_2` values may imply strong anisotropic coupling between the electric field and electron density gradient, which may lead to significant anisotropic dielectric behavior, nonlinear optical effects, or specific charge transport pathways in the material. Chemically, it can more precisely reveal how electric fields affect charge polarization and bonding reconstruction in specific directions. In materials science, this feature is a key indicator for designing materials with directional electrical properties, optoelectronic devices, and novel catalysts; for example, in low-symmetry crystals, this feature can quantify the anisotropy of their electrical response. In computer science and artificial intelligence fields, it provides machine learning models with a higher-order quantification of the material's internal "electric field-electron response" tensor properties, helping models understand material electrical anisotropy, carrier transport mechanisms, and functional transformations in specific electric field directions, thereby playing a key role in the design of advanced functional materials.

#### 6.3.3.3 Total Gradient Norm (`total_gradient_norm`)

The `total_gradient_norm` descriptor is strictly defined as the average of the squared norms of electron density gradients over all spatial grid points within the crystal. It quantifies the overall severity or "ruggedness" of the electron density variation in space within the crystal. High values indicate rapid changes in electron density in space, possibly implying clear chemical bonds, charge transfer regions, or areas near atomic nuclei; low values indicate a relatively smooth electron density distribution. This feature is obtained by acquiring electron density grid data from the GPAW calculator, calculating its gradient, then calculating the squared norm of the gradient, and finally averaging over all effective grid points.
$$
\text{Total Gradient Norm} = \frac{1}{N_{\text{grid}}} \sum_{j=1}^{N_{\text{grid}}} \|\nabla \rho(\mathbf{r}_j)\|^2
$$
where $\|\nabla \rho(\mathbf{r}_j)\|$ is the Euclidean norm of the electron density gradient at grid point $\mathbf{r}_j$.

Physically, `total_gradient_norm` is closely related to the material's electron localization degree, chemical bond strength, and electron kinetic energy density. High values usually imply strong covalent bonds or localized electronic states. Chemically, it directly reflects the spatial distribution characteristics of charge density, especially the charge concentration and depletion in bonding regions and around atomic nuclei. In materials science, `total_gradient_norm` is an important indicator for evaluating material chemical bonding types, electron transport pathways, and structural stability; for example, in semiconductors and insulators, it is related to band gap size and electron effective mass. In computer science and artificial intelligence fields, it provides machine learning models with a continuous and differentiable quantification of the material's internal electron density spatial variation patterns, helping models predict material electron transport properties, dielectric properties, and chemical reactivity, and is a key feature for understanding the relationship between material electronic structure and function.

### 6.3.4 Global Pseudo-Symplectic Geometric Features Based on Static Phase Space

This subsection's features draw upon the profound ideas of Symplectic Geometry, constructing a "pseudo-symplectic phase space" to quantify the coupled fluctuations between the geometric degrees of freedom of chemical bonds and the electron density gradients at bond midpoints within the crystal. Although not a strictly symplectic manifold construction, this analogy provides a powerful framework for describing the deep coupling between structural and electronic behaviors in materials. This feature aims to capture the complexity and intrinsic "volume" of structure-electron interactions in materials, reflecting the synergistic fluctuations of structural and electronic responses in the static lattice. The final feature value returns the logarithm of this coupling measure to avoid numerical underflow and better reflect changes across different orders of magnitude.

#### 6.3.4.1 Logarithm of Pseudo-Symplectic Fluctuation Volume (`log_pseudo_symplectic_fluctuation_volume`)

The `log_pseudo_symplectic_fluctuation_volume` descriptor is strictly defined as the natural logarithm of the determinant of the covariance matrix of state vector distributions in the "pseudo-symplectic phase space" formed by all chemical bonds in the crystal. Each chemical bond $k$ is represented as a six-dimensional state vector $\mathbf{s}_k = (\mathbf{q}_k, \mathbf{p}_k)$ in this pseudo-symplectic phase space, where:

$\mathbf{q}_k$ is the bond direction vector of the chemical bond (represented by its Cartesian coordinates), representing the generalized coordinates of the structure (related to bond geometry as involved in Section 4.1.1).

$\mathbf{p}_k$ is the electron density gradient at the midpoint of the chemical bond (represented by its Cartesian coordinates), representing the generalized momentum of the electronic behavior. This gradient is obtained by high-precision trilinear interpolation of the electron density gradient field obtained from GPAW calculations.

The set of state vectors of all chemical bonds $\{\mathbf{s}_k\}$ forms a point cloud in the pseudo-symplectic phase space. We calculate the covariance matrix $\mathbf{C}_{\mathbf{s}}$ of this point cloud. The determinant of the covariance matrix $\text{Det}(\mathbf{C}_{\mathbf{s}})$ quantifies the "volume" occupied by this point cloud in the six-dimensional space, reflecting the degree of coupled fluctuations between bond geometry and electron density gradient at the bond midpoint. To avoid numerical underflow and better handle order-of-magnitude differences, the final feature takes its natural logarithm $\ln(\text{Det}(\mathbf{C}_{\mathbf{s}}))$. In the calculation, standardization operations are introduced to ensure comparability of features with different dimensions, and a small perturbation is added to the diagonal of the covariance matrix to ensure numerical stability.
$$
\text{Log Pseudo Symplectic Fluctuation Volume} = \ln\left( \text{Det}\left( \text{Cov}(\{\mathbf{s}_k\}) \right) \right)
$$
where $\mathbf{s}_k = (\mathbf{q}_k, \mathbf{p}_k)$ is the state vector of the $k$-th chemical bond, and $\text{Cov}$ denotes the covariance operation.

Physically, high `log_pseudo_symplectic_fluctuation_volume` values imply complex and extensive coupled fluctuations between structural and electronic behaviors within the crystal, which may be closely related to the origin mechanisms of material anharmonic vibrations, thermodynamic instability, and certain functional properties (e.g., superconductivity, ferroelectricity). It provides a novel perspective for quantifying "structure-electron coupling strength." Chemically, it reveals how the geometric configuration of chemical bonds dynamically interacts with their local electronic environment, helping to understand bond flexibility and reactivity. In materials science, this feature is a key indicator for designing and screening materials with strong structure-electron coupling effects, especially suitable for exploring new optoelectronic materials, thermoelectric materials, and superconducting materials. In computer science and artificial intelligence fields, it provides machine learning models with a continuous and differentiable quantification of the material's internal "structure-electron coupling fluctuation space," and is a key advanced feature for predicting complex functional properties of materials, understanding material response mechanisms, and discovering novel structure-property relationships in data-driven material design, with its abstractness giving it powerful generalization capabilities.

## 6.4 Graph-Related Path Features: Graph-Theoretic Insights into Long-Range Correlations within Crystals

This section introduces global descriptors that employ innovative graph-theoretic path analysis methods, aiming to capture more complex and long-range interactions and correlations between atoms in crystal structures. These features quantify dynamic correlations within the crystal, such as energy transfer, charge flow, stress propagation, and structural heterogeneity, by sampling various paths on the crystal graph (atoms as nodes, chemical bonds as edges) and then aggregating the local features of atoms or bonds along these paths. Unlike traditional global averages, path features can reveal patterns and directionality of spatial correlations. To address the computational challenges of complex crystal graphs, this method adopts a unified importance sampling strategy to efficiently select the most representative subset from a large number of possible paths.

### 6.4.1 Path Covariance Torsional Stress (`path_cov_torsional_stress`)

The `path_cov_torsional_stress` descriptor quantifies the covariance between "structure-chemistry incompatibility" and "local torsional stress" along sampled paths in the crystal structure. It reflects the linear correlation strength between the degree of local structural distortion and the torque it experiences along the path. The calculation of this feature first involves sampling paths on the crystal graph (via random walks or complete enumeration with early stopping), then obtaining the sequence of structure-chemistry incompatibility (from Section 4.4.1) and local torsional stress (magnitude of torsional stress of each atom, as defined in Section 6.3.2.3) for each atom along the path, and finally calculating the covariance of these two sequences. The final feature values include the mean (`path_cov_torsional_stress_mean`), standard deviation (`path_cov_torsional_stress_std`), and maximum (`path_cov_torsional_stress_max`) of these path covariances to comprehensively describe this correlation.
$$
\text{Cov}(\text{Incomp}_{\text{S-C}}, \text{Torsion}) = E[(\text{Incomp}_{\text{S-C}} - E[\text{Incomp}_{\text{S-C}}])(\text{Torsion} - E[\text{Torsion}])]
$$
where $\text{Incomp}_{\text{S-C}}$ represents the structure-chemistry incompatibility along the path, and $\text{Torsion}$ represents the local torsional stress along the path.

Physically, high covariance indicates that larger structural distortions are accompanied by larger torques, which may suggest synergistic structural instability or stress concentration regions within the material, affecting its mechanical response and phase transition behavior. Chemically, it reveals how local mismatches in chemical bonding within the structure propagate and accumulate through the force field, forming long-range stress paths. In materials science, `path_cov_torsional_stress` is a key indicator for evaluating material toughness, fracture toughness, fatigue strength, and crack propagation paths; high covariance paths may indicate weak points in the material. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material internal "stress propagation paths" and "structure-mechanical coupling," helping models predict material behavior and failure modes under extreme conditions, thereby guiding material structural optimization and damage tolerance design.

### 6.4.2 Path Entropy (`path_entropy`)

The `path_entropy` descriptor quantifies the Shannon entropy of the "structure-chemistry incompatibility" distribution along sampled paths in the crystal structure. It measures the diversity or randomness of local structural distortion patterns along the path. High entropy values indicate a wide distribution of atomic or bond structure-chemistry incompatibility (from Section 4.4.1) along the path, with various degrees of distortion; low entropy values indicate similar degrees of distortion along the path, with a more uniform structure. The final feature values include the mean (`path_entropy_mean`), standard deviation (`path_entropy_std`), and maximum (`path_entropy_max`) of these path entropies.
$$
H = -\sum_{k=1}^{M} p_k \log_2(p_k)
$$
where $M$ is the number of bins in the histogram, and $p_k$ is the probability that the structure-chemistry incompatibility along the path falls into the $k$-th bin.

Physically, `path_entropy` reflects the spatial distribution of structural defects, local disorder, or heterogeneity within the material. High entropy paths may suggest an increase in the disorder of energy transfer or charge transfer, affecting the material's transport properties. Chemically, it reveals the heterogeneity of bonding environments or coordination structures along specific paths, helping to understand the distribution of active sites for chemical reactions. In materials science, `path_entropy` is an important indicator for evaluating material defect tolerance, formation of disordered phases, and thermodynamic stability, especially suitable for analyzing high-entropy alloys, amorphous materials, and complex solid solutions. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material internal "path structural heterogeneity" and "degree of disorder," helping models predict material phase transition behavior, thermodynamic stability, and response to random perturbations, thereby playing a role in material phase design and performance tuning.

### 6.4.3 Path Chemical Potential Difference (`path_chempot_diff`)

The `path_chempot_diff` descriptor quantifies the difference between the maximum and minimum electronegativity (from Section 4.2.2) along sampled paths in the crystal structure. It reflects the macroscopic fluctuation of atomic charge distribution or electron attraction ability along the path, suggesting potential charge transport channels or electrostatic potential gradients. High values indicate significant electronegativity (chemical potential) differences along the path, which may form a driving force for electron flow; low values indicate a relatively uniform charge environment along the path. The final feature values include the mean (`path_chempot_diff_mean`), standard deviation (`path_chempot_diff_std`), and maximum (`path_chempot_diff_max`) of these path chemical potential differences.
$$
\Delta \chi_{\text{path}} = \max_{i \in \text{path}}(\chi_i) - \min_{i \in \text{path}}(\chi_i)
$$
where $\chi_i$ is the electronegativity of atom $i$ along the path.

Physically, `path_chempot_diff` is closely related to material charge transport, carrier mobility, and local electric field strength. High chemical potential difference paths may indicate preferred electron or ion transport channels, for example, in battery electrode materials or catalysts. Chemically, it reveals the anisotropic distribution of charge polarization and electron attraction between atoms along specific paths. In materials science, `path_chempot_diff` is a key feature for designing and optimizing electrochemical materials (e.g., batteries, fuel cells), semiconductors, and catalysts, as it directly relates to material charge transport efficiency and activity. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material internal "charge transport paths" and "charge environment gradients," helping models predict material electrical properties, ionic conductivity, and catalytic activity, thereby playing a key role in the design of energy materials and catalysts.

### 6.4.4 Path Maximum Torque (`path_max_torque`)

The `path_max_torque` descriptor quantifies the maximum value of the local torsional stress (magnitude of torsional stress of each atom, as defined in Section 6.3.2.3) of all atoms along sampled paths in the crystal structure. It identifies atomic sites along the path that bear the maximum "torque" or the most significant local distortion-force coupling. High values indicate that one or more atoms along the path experience strong torsional effects, which may suggest local structural instability or high strain regions. The final feature values include the mean (`path_max_torque_mean`), standard deviation (`path_max_torque_std`), and maximum (`path_max_torque_max`) of these path maximum torques.
$$
\text{MaxTorque}_{\text{path}} = \max_{i \in \text{path}} (\|\mathbf{V}_{\text{struct},i} \times \mathbf{f}_i\|)
$$
where $\|\mathbf{V}_{\text{struct},i} \times \mathbf{f}_i\|$ is the magnitude of the local torsional stress of the $i$-th atom.

Physically, `path_max_torque` is related to material local soft modes, phase transition precursor effects, and defect-induced stress concentrations. It can identify the most sensitive "weak points" in the crystal structure to external stimuli. Chemically, it reveals the maximum tension experienced by specific atoms or bonding environments along the path under the action of a force field. In materials science, `path_max_torque` is a key feature for predicting early material failure, fatigue crack initiation, and designing stable materials under high stress environments, which is crucial for understanding material reliability under operating conditions. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material internal "critical stress points" and "most vulnerable paths," helping models predict material ultimate strength, fatigue life, and structural stability under extreme conditions, thereby playing a key role in the reliability design of structural and functional materials.

### 6.4.5 Path Curvature (`path_curvature`)

The `path_curvature` descriptor quantifies the average bending degree of bond vector sequences along sampled paths in the crystal structure. It reflects the degree to which a path deviates from linearity in space. For each simple path $P = (v_1, v_2, \ldots, v_L)$ in the graph, we consider its consecutive bond vectors $\mathbf{b}_k = \mathbf{r}_{v_{k+1}} - \mathbf{r}_{v_k}$. Path curvature is approximated by summing the angles between consecutive bond vectors and normalizing by the number of turning points. This provides a continuous quantification of path geometric complexity.

Its mathematical definition is as follows:
$$
\text{Curvature}_P = \frac{1}{N_{\text{angles}}} \sum_{k=1}^{L-1} \theta(\mathbf{b}_k, \mathbf{b}_{k+1})
$$
where $N_{\text{angles}}$ is the number of turning points on the path (i.e., $L-2$), and $\theta(\mathbf{u}, \mathbf{v}) = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}\right)$ is the angle between two vectors. The final global features are obtained by statistical analysis of these curvature values for all sampled paths, including their mean (`path_curvature_mean`), standard deviation (`path_curvature_std`), and maximum (`path_curvature_max`), to comprehensively capture the characteristics of path bending patterns within the crystal.

Physically, `path_curvature` reflects the degree of structural "bending" or "folding" along the path. High curvature paths may imply significant lattice distortions in that region, or more complex atomic arrangements, which may affect material phonon transport and heat conduction. Chemically, this feature is closely related to bond angle strain, conformational degrees of freedom, and the flexibility of molecular chains, especially important for understanding flexible structures in polymers and biological macromolecules. In materials science, `path_curvature` is a key indicator for understanding material flexibility, plastic deformation, crystal structure reconstruction, and phonon propagation paths; for example, paths with high curvature in specific directions may indicate weak points in the material under mechanical loads. In computer science and artificial intelligence fields, it provides a quantification of crystal internal path geometric complexity, enabling machine learning models to more accurately predict material mechanical flexibility, phase transition behavior, and energy transport properties, thereby playing a role in material structural design and performance tuning.

### 6.4.6 Path Wrapping Norm (`path_wrapping_norm`)

The `path_wrapping_norm` descriptor quantifies the degree of "periodic boundary condition" wrapping along sampled paths in the crystal structure. It reflects the Euclidean norm of the "total displacement vector" accumulated by the path as it crosses unit cell boundaries. For each simple path $P = (v_1, v_2, \ldots, v_L)$ in the graph, we consider the periodic images of its consecutive bond vectors in fractional coordinates. These image vectors $\mathbf{f}_k^{\text{image}}$ (representing the integer cell displacement crossed by bond $v_k \to v_{k+1}$) are summed along the path to obtain a total wrapping vector $\mathbf{W}_P^{\text{frac}} = \sum \mathbf{f}_k^{\text{image}}$. To obtain a physically meaningful norm in Cartesian coordinates, this fractional wrapping vector is converted to the Cartesian coordinate system by multiplying it with the lattice matrix $\mathbf{L}$ to get $\mathbf{W}_P^{\text{cart}} = \mathbf{L} \mathbf{W}_P^{\text{frac}}$. The final path wrapping norm is the norm of this Cartesian wrapping vector, $\|\mathbf{W}_P^{\text{cart}}\|_2$, normalized by the path length.

Its mathematical definition is as follows:
$$
\begin{aligned}
\mathbf{W}_P^{\text{frac}} &= \sum_{k=1}^{L-1} \mathbf{f}_k^{\text{image}} \\
\mathbf{W}_P^{\text{cart}} &= \mathbf{L} \mathbf{W}_P^{\text{frac}} \\
\text{WrappingNorm}_P &= \frac{\|\mathbf{W}_P^{\text{cart}}\|_2}{L}
\end{aligned}
$$
where $\mathbf{f}_k^{\text{image}}$ is the periodic image vector of bond $k$, $\mathbf{L}$ is the lattice matrix, and $L$ is the path length (number of atoms). The final global features are obtained by statistical analysis of these wrapping norm values for all sampled paths, including their mean (`path_wrapping_norm_mean`), standard deviation (`path_wrapping_norm_std`), and maximum (`path_wrapping_norm_max`).

Physically, `path_wrapping_norm` reflects the "long-range topological connectivity" or "periodic deformation accumulation" along specific paths in the crystal structure. High wrapping norm paths may imply that the atomic arrangement in that region is strongly influenced by periodic boundary conditions, or that there are microscopic structural paths leading to macroscopic strain accumulation. Chemically, it is related to atomic migration paths in the lattice, diffusion mechanisms, and the propagation of crystal defects (e.g., dislocations), as these processes often involve crossing unit cells. In materials science, `path_wrapping_norm` is an important indicator for understanding long-range transport properties (e.g., ion diffusion, thermal expansion), plastic deformation mechanisms, and lattice matching/mismatch, especially suitable for analyzing grain boundary behavior and diffusion channels in polycrystalline materials. In computer science and artificial intelligence fields, it provides a quantification of the complexity of crystal internal path crossing periodic boundaries, helping machine learning models predict material diffusion coefficients, phase transition mechanisms, and structural response under periodic stress, thereby playing a key role in energy materials and lattice engineering design.

### 6.4.7 Path Force Gradient (`path_force_gradient`)

The `path_force_gradient` descriptor quantifies the rate of change or inhomogeneity of force vectors acting on atoms along sampled paths in the crystal structure. It reflects the accumulation of force field spatial gradients along the path, suggesting potential stress concentrations, lattice soft modes, or energy transport channels. For each simple path $P = (v_1, v_2, \ldots, v_L)$ in the graph, we obtain the force vector $\mathbf{f}_k$ acting on each atom $v_k$ along the path. The path force gradient is approximated by summing the squared norms of the differences between consecutive atomic force vectors and normalizing by the path length.

Its mathematical definition is as follows:
$$
\text{ForceGradient}_P = \frac{1}{L-1} \sum_{k=1}^{L-1} \|\mathbf{f}_{v_{k+1}} - \mathbf{f}_{v_k}\|^2
$$
where $\mathbf{f}_v$ is the force vector acting on atom $v$, and $L$ is the path length (number of atoms). These force vectors are directly obtained from DFT calculation results. The final global features are obtained by statistical analysis of these force field gradient values for all sampled paths, including their mean (`path_force_gradient_mean`), standard deviation (`path_force_gradient_std`), and maximum (`path_force_gradient_max`).

Physically, `path_force_gradient` reflects the "ruggedness" of the internal force field and the severity of local stress distribution within the crystal. High gradient paths may imply significant lattice distortions, atomic displacements, or phase transition precursor effects in that region, all of which are closely related to material mechanical stability and phonon behavior. Chemically, it is related to the steepness of interatomic potential energy surfaces and changes in bonding stiffness. In materials science, `path_force_gradient` is a key indicator for predicting material mechanical properties (e.g., hardness, fracture toughness), heat transport (via phonon scattering), and understanding structural phase transition mechanisms; for example, in superelastic materials, the nonlinear gradient of the force field is crucial for their flexibility. In computer science and artificial intelligence fields, it provides a quantification of crystal internal "force field paths" and "stress wave propagation," helping machine learning models predict material mechanical response, thermal conductivity, and structural stability under different stress conditions, thereby playing a key role in the design of structural and functional materials.

### 6.4.8 Path Structure Autocorrelation (`path_structure_autocorr`)

The `path_structure_autocorr` descriptor quantifies the first-order autocorrelation coefficient of the "structure-chemistry incompatibility" sequence (from Section 4.4.1) along sampled paths in the crystal structure. It reflects the "memory" or "continuity" of the local structural distortion degree along the path, i.e., the strength of the linear correlation between one atom's incompatibility and its next nearest neighbor atom's incompatibility. High positive values indicate that adjacent atoms have similar degrees of structural distortion, while low or negative values indicate significant changes in distortion patterns along the path. The final global features are obtained by statistical analysis of these autocorrelation coefficients for all sampled paths, including their mean (`path_structure_autocorr_mean`), standard deviation (`path_structure_autocorr_std`), and maximum (`path_structure_autocorr_max`).
$$
\text{Autocorr}_P(\text{Incomp}_{\text{S-C}}) = \frac{\text{Cov}(\text{Incomp}_{\text{S-C},k}, \text{Incomp}_{\text{S-C},k+1})}{\sigma(\text{Incomp}_{\text{S-C},k})\sigma(\text{Incomp}_{\text{S-C},k+1})}
$$
where $\text{Incomp}_{\text{S-C},k}$ is the structure-chemistry incompatibility of the $k$-th atom along the path.

Physically, `path_structure_autocorr` reflects the spatial continuity of structural distortion or strain fields within the crystal. High autocorrelation paths may indicate the presence of long-range ordered domains, covalent networks, or rigid chains along specific crystallographic directions in the crystal, which affect material lattice stability, phonon transport and macroscopic anisotropy. Chemically, it reveals how the similarity or dissimilarity of chemical bonding environments propagates along specific paths. In materials science, `path_structure_autocorr` is a key indicator for understanding material microstructure homogeneity, phase boundary behavior, defect distribution, and crystal growth patterns, which is crucial for controlling material microstructure to achieve desired macroscopic properties. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material internal "structural sequence patterns" and "spatial correlations," helping models identify specific structural motifs in crystals, predict material microstructural evolution, and morphological features under different synthesis conditions, thereby playing a role in material structural design and process optimization.

### 6.4.9 Path Charge-Potential Covariance (`path_charge_potential_cov`)

The `path_charge_potential_cov` descriptor quantifies the covariance between atomic Bader charges (from Section 3.3.1) and electrostatic potentials at atomic sites (from Section 6.2.3 mean electrostatic potential, then interpolated) along sampled paths in the crystal structure. It reflects the linear correlation strength between charge distribution and potential environment along the path, suggesting potential charge transport efficiency or electrochemical activity. High positive covariance indicates that charge-rich regions are accompanied by high potentials, or charge-depleted regions are accompanied by low potentials; negative covariance may indicate the directionality of charge transfer. The final global features are obtained by statistical analysis of these covariance values for all sampled paths, including their mean (`path_charge_potential_cov_mean`), standard deviation (`path_charge_potential_cov_std`), and maximum (`path_charge_potential_cov_max`).
$$
\text{Cov}(\text{Charge}, \text{Potential}) = E[(\text{Charge} - E[\text{Charge}])(\text{Potential} - E[\text{Potential}])]
$$
where $\text{Charge}$ represents the Bader charge along the path, and $\text{Potential}$ represents the electrostatic potential along the path.

Physically, `path_charge_potential_cov` is closely related to material charge transport mechanisms, electron scattering processes, and local electric field strength. High covariance paths may indicate preferred electron or ion conduction channels, or regions with high polarizability. Chemically, it reveals charge transfer patterns and chemical potential gradients along specific paths, helping to understand charge transfer steps in chemical reactions. In materials science, `path_charge_potential_cov` is a key feature for designing and optimizing electrochemical materials (e.g., batteries, supercapacitors), semiconductors, and catalysts, as it directly relates to material charge transport efficiency, energy storage capacity, and surface reactivity. In computer science and artificial intelligence fields, it provides machine learning models with a quantitative description of material internal "charge-potential paths" and "charge transport microenvironments," helping models predict material conductivity, capacitance performance, and catalytic activity, thereby playing a key role in energy materials and electronic device design.
