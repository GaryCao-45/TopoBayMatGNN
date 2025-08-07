# Chapter 4: 1-Simplex Descriptors: Quantitative Characterization of Chemical Bonds and Interactions

1-simplex descriptors focus on chemical bonds (atomic pairs) and their immediate environments in crystalline materials, aiming to precisely quantify the intrinsic properties of bonds and how they mediate interatomic interactions from geometric, physicochemical, and quantum mechanical perspectives. This chapter constructs a multi-dimensional, high-content bond-level description system through in-depth analysis of key attributes such as bond length, coordination number, charge transfer, and electron localization. These descriptors serve as crucial bridges for understanding crystal structure stability, structure-property relationships, and predicting material functional characteristics.

## 4.1 Geometric Descriptors: Bond Length and Coordination Environment

This section primarily focuses on the most direct geometric attributes of chemical bonds—bond length—and the local coordination environments of the atoms at both ends of the bond. These features provide a foundation for understanding spatial arrangements and fundamental geometric constraints between atoms in crystalline materials.

### 4.1.1 Bond Length (`bond_distance`)

Bond length $d_{ij}$ is strictly defined as the Euclidean distance connecting two bonded atoms $i$ and $j$. If atom $i$ has coordinates $\mathbf{r}_i$ and atom $j$ has coordinates $\mathbf{r}_j$, the formula for calculating bond length $d_{ij}$ is:
    $$
    d_{ij} = \| \mathbf{r}_j - \mathbf{r}_i \|_2 = \sqrt{(x_j - x_i)^2 + (y_j - y_i)^2 + (z_j - z_i)^2}
    $$

Bond length as a direct spatial scale of interatomic interaction, is an important physical representation of chemical bond strength and bond type; shorter bond lengths typically correspond to stronger bonding. Chemically, it is closely related to covalent radius, ionic radius, and bonding potential energy, serving as a key parameter for determining bond types (covalent, ionic, metallic) and predicting molecular stability; for example, short and strong covalent bonds usually have higher bond energies and directionality. In materials science, bond length directly influences crystal lattice constants, density, thermal expansion coefficients, and mechanical strength, where minor changes can lead to significant differences in macroscopic material properties, e.g., in semiconductors, bond length changes affect the electronic band structure. In computer science and artificial intelligence fields, `bond_distance` as the most basic geometric feature, is the cornerstone for constructing complex higher-order descriptors, providing low-dimensional, directly computable input, widely applicable in machine learning models for material property prediction, easy to understand and requiring no complex feature engineering.

### 4.1.2 Bond End Atom Coordination Numbers (`site1_coord_num`, `site2_coord_num`)

`site1_coord_num` and `site2_coord_num` descriptors are strictly defined as the coordination numbers of atoms $i$ and $j$ at both ends of a chemical bond, i.e., the number of nearest-neighbor atoms directly connected to an atom in a crystal structure. Their calculation relies on the `pymatgen.analysis.local_env.CrystalNN` strategy, which determines coordination by intelligent judgment based on bond distances and chemical environments. The mathematical expression for coordination number is:
    $$
    \text{CoordNum}_k = |\text{NN}(k)|
    $$
where $\text{NN}(k)$ denotes the set of nearest-neighbor atoms of atom $k$. These values are obtained from pre-calculated atomic coordination number lists in Section 4.2.8. The CrystalNN algorithm identifies chemical bonds by analyzing interatomic distances, elemental properties, and periodic boundary conditions, and counts the number of atoms directly connected to the central atom. This is a complex process combining geometric judgment and chemical rules, aiming to simulate the real chemical bonding environment. From physics, chemistry, and materials science perspectives, coordination number directly reflects the spatial packing and local interaction strength of an atom within the crystal lattice. It is a core indicator for determining atomic valence, bond types, and local geometric configurations, directly influencing the shape and connectivity of coordination polyhedra, which in turn affects material density, stability, electronic structure (e.g., crystal field splitting), and many macroscopic physical properties (e.g., hardness, melting point, ion mobility). Changes in coordination number are often associated with phase transitions or defect formation. In computer science and artificial intelligence, as discrete but physically meaningful features, coordination numbers can be part of node features in graph neural networks, or used as classification or regression features in traditional machine learning models, providing a basic description for models to understand the topological properties of crystal networks.

---

## 4.2 Derived and Relational Descriptors: Bond-Level Information Integration Based on 0-Simplex Features

The descriptors introduced in this section are obtained by aggregating or differentiating pre-calculated features of atoms (0-simplices) at both ends of a chemical bond, aiming to capture the interaction and information transfer of atomic properties during bonding.

### 4.2.1 Electronegativity Difference (`delta_electronegativity`)

`delta_electronegativity` descriptor is strictly defined as the absolute difference in electronegativity $\chi$ between atoms $i$ and $j$ at both ends of a chemical bond. Its mathematical expression is:
$$
\Delta \chi_{ij} = |\chi_j - \chi_i|
$$
where atomic electronegativity data is sourced from Section 4.2.2.

Physically, the electronegativity difference is a direct indicator of the degree of ionic character of a bond; a larger difference typically predicts stronger ionic bond character and a tendency for electron transfer from one atom to another. Chemically, it directly relates to the polarity, dipole moment, and dielectric properties of chemical bonds; in chemical reactions, bonds with high electronegativity differences are often active sites for reactions. In materials science, this feature influences material polarity, dielectric constant, piezoelectricity, and ionic conductivity; for ionic conductors, a larger electronegativity difference can promote ion migration within the crystal lattice. In computer science and artificial intelligence fields, as an important feature measuring bond polarity, it provides machine learning models with intuitive information about the nature of bonding, playing an important role in predicting thermodynamic stability, material band gap, or surface activity.

### 4.2.2 Ionic Radius Difference (`delt-ionic_radius`)

`delt-ionic_radius` descriptor is strictly defined as the absolute difference in ionic radii $r^{\text{ion}}$ between atoms $i$ and $j$ at both ends of a chemical bond. Its mathematical expression is:
$$
\Delta r_{ij}^{\text{ion}} = |r_j^{\text{ion}} - r_i^{\text{ion}}|
$$
where atomic ionic radius data is sourced from Section 4.2.6.

Physically, the ionic radius difference directly reflects the degree of spatial size matching between bonded atoms; a larger difference may lead to structural distortion or stress. Chemically, for ionic crystals, the ionic radius ratio is a key factor for determining crystal structure type and stability (e.g., Pauling's rules), and it also influences the strength and directionality of ionic bonds. In materials science, this feature is closely related to lattice distortion, defect formation energy, phase behavior, and ion mobility; for example, in perovskite crystal materials, appropriate matching of ionic radii is crucial for forming stable ABX₃ structures. In computer science and artificial intelligence fields, ionic radius difference provides information about atomic size matching, which is an important input for models predicting material structural stability, synthesizability, and certain ion transport properties.

### 4.2.3 Sum of Covalent Radii (`sum_covalent_radii`)

`sum_covalent_radii` descriptor is strictly defined as the sum of covalent radii $r^{\text{Covalent Radii}}$ for atoms $i$ and $j$ at both ends of a chemical bond. Its mathematical expression is:
$$
\Sigma r_{ij}^{\text{Covalent Radii}} = r_i^{\text{Covalent Radii}} + r_j^{\text{Covalent Radii}}
$$
where atomic covalent radius data is sourced from Section 4.2.7.

Physically, the sum of covalent radii is generally regarded as an approximation of ideal covalent bond lengths, providing a baseline for evaluating whether actual bond lengths are compressed or stretched. Chemically, it is directly related to the properties of covalent bonds and the overlap integral between atoms; in covalent crystals, the sum of covalent radii can help predict the strength and directionality of bonding. In materials science, this feature affects crystal material structure, mechanical properties (e.g., hardness, elastic modulus), and thermal expansion; in covalently dominant materials, the sum of covalent radii is crucial for understanding the bonding network and its stability. In computer science and artificial intelligence fields, `sum_covalent_radii` can be used in conjunction with actual bond lengths to form higher-order features like bond length distortion, providing fine-grained information about the bonding environment for models.

### 4.2.4 Average Bader Charge (`avg_bader_charge`)

`avg_bader_charge` descriptor is strictly defined as the arithmetic mean of the Bader charges $Q^{\text{Bader}}$ of atoms $i$ and $j$ at both ends of a chemical bond. Its mathematical expression is:
$$
\overline{Q}_{ij}^{\text{Bader}} = \frac{Q_i^{\text{Bader}} + Q_j^{\text{Bader}}}{2}
$$
where atomic Bader charge data is sourced from Section 3.3.1.

Physically, the average Bader charge reflects the average contribution of bonded atoms to the overall charge, serving as an indicator of the average charge state of the bonding environment. Chemically, it is related to average oxidation states, total electron count in bonds, and material chemical stability, being very useful for understanding the overall electron enrichment or depletion state of a bonding region. In materials science, this feature influences crystal material conductivity, optical properties, and chemical reactivity; for example, the average charge along a bonding chain can indicate electron transport pathways. In computer science and artificial intelligence fields, `avg_bader_charge` provides statistical information about the overall electronic environment of the bonding region, aiding models in understanding the electronic properties of bonds.

### 4.2.5 Bader Charge Difference (`delta_bader_charge`)

`delta_bader_charge` descriptor is strictly defined as the absolute difference in Bader charges $Q^{\text{Bader}}$ between atoms $i$ and $j$ at both ends of a chemical bond. Its mathematical expression is:
$$
\Delta Q_{ij}^{\text{Bader}} = |Q_j^{\text{Bader}} - Q_i^{\text{Bader}}|
$$
where atomic Bader charge data is sourced from the same section as above.

Physically, the Bader charge difference directly reflects the strength of charge transfer and the non-uniformity of charge distribution between bonded atoms, serving as an important quantitative indicator of bond ionicity; a larger difference implies more significant charge polarization. Chemically, it is related to bond polarity, dipole moment (though it does not directly provide direction), ionic bond character, and chemical bond stability, serving as a key parameter for understanding how electrons redistribute within a bond. In materials science, this feature influences crystal material dielectric properties, ferroelectricity, piezoelectricity, and electron/ion transport pathways; in polar materials, the Bader charge difference is an important metric for quantifying their polarization strength. In computer science and artificial intelligence fields, `delta_bader_charge` complements electronegativity difference by providing more direct information about charge distribution non-uniformity, holding significant value for predicting properties of polar materials and guiding the design of new materials.

---

## 4.3 Quantum Chemical Bonding Descriptors: Microscopic Insight into Bond Electronic Structure

The descriptors in this section are derived directly from quantum physical quantities such as electron density, ELF (Electron Localization Function), and Bader charge, obtained from density functional theory (DFT) first-principles calculations. They deeply reveal the electronic structure details within chemical bonds, serving as cornerstones for understanding the microscopic bonding mechanisms and macroscopic quantum effects of crystalline materials.

### 4.3.1 Bond Midpoint Electron Density (`bond_midpoint_density`)

`bond_midpoint_density` descriptor is strictly defined as the total electron density $\rho(\mathbf{r}_{\text{mid}})$ at the midpoint $\mathbf{r}_{\text{mid}}$ of a chemical bond. The bond midpoint is the geometric midpoint of the line segment connecting the two atomic nuclei. Electron density $\rho(\mathbf{r})$ measures the probability of finding an electron at a certain point in space; higher values indicate greater electron enrichment in that region. Its mathematical expression is:
$$
\rho(\mathbf{r}_{\text{mid}}) = \rho\left(\frac{\mathbf{r}_i + \mathbf{r}_j}{2}\right)
$$

Physically, bond midpoint electron density is a direct indicator of covalent bond strength and multiplicity; covalent bonds typically exhibit significant electron density accumulation in the bonding region (especially at the midpoint). Chemically, it is closely related to bond order, covalent character of the bond, and the degree of electron sharing between atoms; high bond midpoint density usually implies strong covalent interaction. In materials science, this feature affects crystal material mechanical strength, hardness, thermal conductivity, and electron transport properties; for example, high bond midpoint density leads to excellent mechanical properties in covalent crystals like diamond. In computer science and artificial intelligence fields, `bond_midpoint_density` provides the most core electronic structure information of chemical bonds, serving as a key input for machine learning models predicting properties of covalently dominated materials, and its continuous nature makes it suitable for gradient-based optimization models.

### 4.3.2 Laplacian of Bond Midpoint Electron Density (`bond_density_laplacian`)

`bond_density_laplacian` descriptor is strictly defined as the Laplacian of electron density $\nabla^2 \rho(\mathbf{r}_{\text{mid}})$ at the midpoint $\mathbf{r}_{\text{mid}}$ of a chemical bond. The Laplacian operator $\nabla^2 \rho(\mathbf{r})$ measures the local concentration or dispersion trend of electron density at a certain point in space. According to QTAIM theory, $\nabla^2 \rho(\mathbf{r}) < 0$ indicates electron concentration at $\mathbf{r}$ (bonding region), while $\nabla^2 \rho(\mathbf{r}) > 0$ indicates electron dispersion at $\mathbf{r}$ (nuclear region). Its mathematical expression is:
$$
\nabla^2 \rho(\mathbf{r}_{\text{mid}}) = \left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}\right) \rho(\mathbf{r})\Big|_{\mathbf{r}=\mathbf{r}_{\text{mid}}}
$$

Physically, the Laplacian of bond midpoint electron density is a powerful tool for discriminating bonding types. In covalent bonds, the Laplacian at the bond critical point (BCP, where $\nabla \rho = 0$) is usually negative, indicating electron sharing within the bonding region; whereas in ionic bonds, this value may be positive or close to zero, indicating charge accumulation around atomic nuclei. Chemically, it is directly related to covalent/ionic character of bonding, existence of bond paths, and chemical reactivity, providing richer topological information than simple electron density. In materials science, this feature affects crystal material bonding rigidity, mechanical response, and electron localization, e.g., a negative Laplacian value at the bond midpoint indicates a stronger covalent network. In computer science and artificial intelligence fields, as a precise quantum quantity for measuring bonding topological features, it provides machine learning models with deeper insights into the nature of bonding, helping models distinguish different bonding types, thereby improving the accuracy of material property prediction.

### 4.3.3 Bond Midpoint ELF (`bond_midpoint_elf`)

`bond_midpoint_elf` descriptor is strictly defined as the Electron Localization Function $\text{ELF}(\mathbf{r}_{\text{mid}})$ at the midpoint $\mathbf{r}_{\text{mid}}$ of a chemical bond. ELF is a dimensionless function between 0 and 1, quantifying the degree of electron localization in that region. High ELF values (close to 1) indicate strong electron localization (e.g., covalent bonds or lone pairs), low ELF values (close to 0.5) indicate delocalization (e.g., metallic bonds), and very low ELF values (close to 0) indicate electron deficiency. Its mathematical expression is:
$$
\text{ELF}(\mathbf{r}) = \frac{1}{1 + (\mathcal{D}(\mathbf{r})/D_0(\mathbf{r}))^2}
$$
where $\mathcal{D}(\mathbf{r})$ is the electron kinetic energy density, and $D_0(\mathbf{r})$ is the kinetic energy density reference for a uniform electron gas. High ELF values (close to 1) indicate strong electron localization (e.g., covalent bonds or lone pairs), low ELF values (close to 0.5) indicate delocalization (e.g., metallic bonds), and very low ELF values (close to 0) indicate electron deficiency.

Physically, bond midpoint ELF directly visualizes the electron localization characteristics in the core region of a three-body cluster, helping to identify the behavior of covalent, ionic, or metallic bonds in a multi-body environment. Chemically, it is used to determine the strength and type of multi-center bonding, e.g., complex bonding in intermetallic compounds or molecular clusters. Materials science, this feature affects the local bonding rigidity, electron transport properties, dielectric response, and phonon scattering of materials; for example, in semiconductors, high ELF value regions are often barriers to electron transport. In computer science and artificial intelligence fields, it provides intuitive and quantitative three-body region electron localization information, serving as a key feature for building machine learning models that can distinguish complex bonding types and predict material many-body interactions and functional properties.

### 4.3.4 Mean ELF at Bond Ends (`bond_elf_from_0simplex`)

`bond_elf_from_0simplex` descriptor is strictly defined as the arithmetic mean of the Electron Localization Function (ELF) values of atoms $i$ and $j$ at both ends of a chemical bond. This feature is directly aggregated from 0-simplex features (see Section 3.3.4), providing a general description of the overall electron localization degree in the bonding region. Its mathematical expression is:
$$
\overline{\text{ELF}}_{ij} = \frac{\text{ELF}_i + \text{ELF}_j}{2}
$$
where $\text{ELF}_i$ and $\text{ELF}_j$ are the ELF values of atoms $i$ and $j$ at their respective sites.

From physics, chemistry, and materials science perspectives, this feature reflects the overall electron localization status of a chemical bond. A high mean indicates that electrons in the bonding region are generally highly localized, potentially corresponding to strong covalent or ionic bonds; a low mean might indicate more delocalized electron characteristics (e.g., metallic bonds). It helps understand the average bonding nature of the bond. In computer science and artificial intelligence fields, as a continuous numerical feature, it provides a simple and effective quantification of the degree of electron localization in the bonding region, which can be used to predict material bonding types, electrical properties, and stability.

### 4.3.5 ELF Asymmetry at Bond Ends (`bond_elf_asymmetry`)

`bond_elf_asymmetry` descriptor is strictly defined as the absolute difference in Electron Localization Function (ELF) values between atoms $i$ and $j$ at both ends of a chemical bond. This feature is derived from 0-simplex features (see Section 3.3.4), quantifying the difference in electron localization degree between the two ends of the bond. Its mathematical expression is:
$$
\Delta \text{ELF}_{ij} = |\text{ELF}_j - \text{ELF}_i|
$$
where $\text{ELF}_i$ and $\text{ELF}_j$ are the ELF values of atoms $i$ and $j$ at their respective sites.

From physics, chemistry, and materials science perspectives, this feature reflects the difference or non-uniformity in the degree of electron localization at the two ends of a chemical bond. A high asymmetry value may indicate strong bond polarity, or significant differences in the ability of the two atoms at the bond ends to share electrons. It helps identify the distribution of electron localization strength within the bond. In computer science and artificial intelligence fields, as a continuous numerical feature, it provides a quantification of the non-uniformity of electron localization in the bonding region, holding significant importance for predicting properties of polar materials, charge transport pathways, and chemical reactivity sites.

### 4.3.6 Mean Electron Density at Bond Ends (`bond_density_from_0simplex`)

`bond_density_from_0simplex` descriptor is strictly defined as the arithmetic mean of the electron density values (taken from their respective atomic sites, see Section 3.3.3) of atoms $i$ and $j$ at both ends of a chemical bond. It provides a generalized measure of the overall electron abundance in the bonding region. Its mathematical expression is:
$$
\overline{\rho}_{ij} = \frac{\rho_i + \rho_j}{2}
$$
where $\rho_i$ and $\rho_j$ are the electron density values of atoms $i$ and $j$ at their respective sites.

From physics, chemistry, and materials science perspectives, this feature reflects the average electron abundance around the two atoms connected by a chemical bond. A high mean indicates that the bonding environment is generally rich in electrons, potentially related to stronger bonding interactions. It helps understand the average electronic environment of the bond. In computer science and artificial intelligence fields, as a continuous numerical feature, it provides a simple quantification of electron abundance in the bonding region, which can be used to predict material bonding strength, density, and electronic properties.

### 4.3.7 Bond Electron Density Gradient (`bond_density_gradient`)

`bond_density_gradient` descriptor is strictly defined as the absolute difference in electron density values (taken from their respective atomic sites, see Section 3.3.3) between atoms $i$ and $j$ at both ends of a chemical bond, divided by the bond length. This quantifies the rate of change of electron density along the chemical bond direction, reflecting the degree of charge polarization or sharing in the bond. Its mathematical expression is:
$$
\text{Density Gradient}_{ij} = \frac{|\rho_j - \rho_i|}{d_{ij}}
$$
where $\rho_i$ and $\rho_j$ are the electron density values of atoms $i$ and $j$ at their respective sites, and $d_{ij}$ is the bond length (see Section 4.1.1). When the bond length approaches zero, a small tolerance value is added to the denominator to avoid division by zero errors.

From physics, chemistry, and materials science perspectives, this feature reflects the electron density slope or gradient along a chemical bond. A high gradient indicates significant charge transfer from one atom to another or strong bond polarity. It helps identify charge transport pathways and the strength of polar bonds. In computer science and artificial intelligence fields, as a continuous numerical feature, it provides a quantification of the rate of change of electron density in the bonding region, holding significant importance for predicting material charge transport properties, dielectric properties, and chemical reactivity sites.

### 4.3.8 Bond Effective Charge (`bond_effective_charge`)

`bond_effective_charge` descriptor is strictly defined as the arithmetic mean of the Bader charge values (see Section 3.3.1) of atoms $i$ and $j$ at both ends of a chemical bond. This is numerically identical to the average Bader charge from Section 4.2.4, but in this section, its role as a direct reflection of the overall charge state of the bonding system in quantum chemical analysis is re-emphasized. Its mathematical expression is:
$$
\overline{Q}_{ij}^{\text{Bader}} = \frac{Q_i^{\text{Bader}} + Q_j^{\text{Bader}}}{2}
$$
where $Q_i^{\text{Bader}}$ and $Q_j^{\text{Bader}}$ are the Bader charges of atoms $i$ and $j$.

From physics, chemistry, and materials science perspectives, this feature reflects the average charge contribution of bonded atoms to the entire bonding region. It helps understand the overall electronic environment and the balance between ionic and covalent character of the bond. In computer science and artificial intelligence fields, as a continuous numerical feature, it provides a quantification of the charge state in the bonding region, which can be used to predict material electrical properties and chemical stability.

### 4.3.9 Bond Charge Imbalance (`bond_charge_imbalance`)

`bond_charge_imbalance` descriptor is strictly defined as the normalized value of the difference in Bader charges between atoms $i$ and $j$ at both ends of a chemical bond, relative to the sum of their absolute values. It quantifies the charge asymmetry or imbalance between bonded atoms. Its mathematical expression is:
$$
\text{Charge Imbalance}_{ij} = \frac{Q_j^{\text{Bader}} - Q_i^{\text{Bader}}}{|Q_i^{\text{Bader}}| + |Q_j^{\text{Bader}}|}
$$
where $Q_i^{\text{Bader}}$ and $Q_j^{\text{Bader}}$ are the Bader charges of atoms $i$ and $j$. A small tolerance value is added to the denominator to avoid division by zero errors.

From physics, chemistry, and materials science perspectives, this feature reflects the degree of electron distribution polarization within the bond. Non-zero values indicate a net charge transfer along the bond direction, which may be related to dipole formation, local polarization, and piezoelectricity. It helps identify the direction and strength of charge transfer in bonding. In computer science and artificial intelligence fields, as a continuous numerical feature, it provides a quantification of the non-uniformity of electron localization in the bonding region, holding significant importance for predicting properties of polar materials, charge transport pathways, and chemical reactivity sites.

---

## 4.4 Deep Fusion Algebraic Descriptors: Abstract Characterization of Complex Chemical Bond Interactions

This section's descriptors are a core innovation of this chapter, going beyond traditional geometric and quantum descriptions by pioneeringly fusing five core algebraic ideas. They aim to capture complex interactions in chemical bonds at higher-order geometric, topological, symmetry, and field-matter coupling levels. These features provide continuous, differentiable, and profoundly physically interpretable mathematical tools, crucial for constructing interpretable artificial intelligence models capable of insight into the deep laws of crystalline material "genomes."

### 4.4.1 Geometric Environment Incompatibility (`lie_algebr-incompatibility`)

`lie_algebr-incompatibility` descriptor is strictly defined as the Frobenius norm of the commutator of the local environment structure tensors $\mathbf{T}_{\text{struct},i}$ and $\mathbf{T}_{\text{struct},j}$ of atoms $i$ and $j$ at both ends of a chemical bond, on the Lie algebra $\mathfrak{so}(3)$ of the rotation group. This feature quantifies the non-commutativity or non-isomorphism of two local geometric environments through the Lie algebra commutator $\mathcal{C}(\mathbf{T}_{\text{struct},i}, \mathbf{T}_{\text{struct},j}) = \mathbf{T}_{\text{struct},i} \mathbf{T}_{\text{struct},j} - \mathbf{T}_{\text{struct},j} \mathbf{T}_{\text{struct},i}$. Its mathematical expression is:
$$
\text{Incomp}_{ij} = \| \mathcal{C}(\mathbf{T}_{\text{struct},i}, \mathbf{T}_{\text{struct},j}) \|_F = \sqrt{\text{Tr}\left( (\mathcal{C}(\mathbf{T}_{\text{struct},i}, \mathbf{T}_{\text{struct},j}))^T \mathcal{C}(\mathbf{T}_{\text{struct},i}, \mathbf{T}_{\text{struct},j}) \right)}
$$
where $\mathbf{T}_{\text{struct},k}$ is the local environment structure tensor of atom $k$, sourced from Section 3.4.2.

Physically, this descriptor quantifies the degree of geometric structure matching and stress state of the local environments of atoms at both ends of a bond. A non-zero incompatibility value indicates that the two environments cannot be perfectly aligned by simple linear transformations, predicting the presence of shear stress, torsion, or distortion within the lattice, analogous to strain tensors or dislocation cores in crystallography. Chemically, it is related to the mismatch of local chemical bonding environments and the accumulation of bonding stress; high incompatibility may lead to reduced bond energy or instability. In materials science, `lie_algebr-incompatibility` is key to understanding crystal defect formation, phase transition pathways, mechanical properties (e.g., brittleness, toughness), and transport properties (e.g., hindered ion migration); in specific crystals like perovskites, incompatibility due to octahedral distortion directly affects their optoelectronic properties. In computer science and artificial intelligence fields, this feature provides a continuous, differentiable quantification of local structural strain and geometric topological mismatch in crystals, serving as a powerful feature for building machine learning models that can predict material structural stability, defect formation energies, and mechanical response. Its mathematical rigor provides high generalization capability and interpretability.

### 4.4.2 Bond Direction Quotient Algebra Feature: Orbit Size (`quotient_algebra_orbit_size`)

This feature, based on quotient algebra theory (or orbit theory of group action), aims to quantify the degree of symmetric equivalence of chemical bond directions under the action of the crystal symmetry group.
`quotient_algebra_orbit_size`: Strictly defined as the size of the orbit $O(\mathbf{v}_{\text{bond}}) = \{g \cdot \mathbf{v}_{\text{bond}} \mid g \in G\}$ formed by the bond direction vector $\mathbf{v}_{\text{bond}}$ under the action of the crystal symmetry group $G$ (specifically, in calculations, this typically refers to the set of rotation and reflection operations contained in the crystal's space group, i.e., its point group operations).

An orbit is a core concept in group theory, representing an equivalence class of elements in a set $X$ (here, bond direction vectors in 3D space) that can be transformed into each other through group operations, under the action of a group $G$ on the set $X$. Specifically for chemical bond directions, if two bond directions can be transformed into each other by a crystal symmetry operation, they belong to the same orbit.

Its mathematical expression is:
$$
\text{OrbitSize}(\mathbf{v}_{\text{bond}}) = |O(\mathbf{v}_{\text{bond}})| = |\{g \cdot \mathbf{v}_{\text{bond}} \mid g \in G \}|
$$
Where:

  `\mathbf{v}_{\text{bond}}` is the normalized chemical bond direction vector; $G$ is the crystal's symmetry group (usually the point group or its associated rotation operations), whose elements $g$ represent a symmetry operation (e.g., rotation, inversion); $g \cdot \mathbf{v}_{\text{bond}}$ represents the new vector after the symmetry operation $g$ acts on the bond direction vector $\mathbf{v}_{\text{bond}}$; $|\cdot|$ denotes the number of elements in a set.

The orbit size reflects the symmetric uniqueness of this bond direction within the crystal: a larger orbit (i.e., containing more unique directions) indicates that the direction is less symmetrically special, having more equivalent directions; a smaller orbit indicates higher or more unique symmetry (e.g., directions along major lattice axes usually have smaller orbits because only a few operations can map them to different directions, or perhaps only to themselves). In computational implementation, we iterate through all symmetry operations, transform the bond direction vector, and collect all numerically unique transformed results, the count of which is the orbit size.

### 4.4.3 Pseudo-Symplectic Coupling (`pseudo_symplectic_coupling`)

`pseudo_symplectic_coupling` descriptor, inspired by symplectic geometry theory, aims to quantify the deep coupling relationship between the geometric direction of a chemical bond and the electron density gradient at the bond's midpoint. Although strictly speaking, symplectic geometry usually deals with even-dimensional phase spaces, this feature constructs an analogous phase space by treating the bond's direction vector as generalized coordinates $\mathbf{q}$ and the electron density gradient as generalized momenta $\mathbf{p}$. The physical basis for choosing electron density gradient as generalized momenta is that it not only quantifies the spatial rate of change of electron density but also profoundly reflects the effective force field and potential driving force for charge transport experienced by local electrons. In quantum mechanics, the concept of force is closely related to momentum, and the spatial distribution of the electron density gradient can be viewed as a potential energy slope or a driving force field, whose direction indicates the tendency for electrons to flow from high-density to low-density regions, which conceptually resembles the physical meaning of momentum driving system evolution in classical mechanics. Based on this, a composite invariant is defined to capture the strength of their interaction. A larger value of this feature indicates stronger curling or coupling effects between the bond direction and the local change trend of electron density, predicting unique physicochemical behaviors.

The core of this feature's calculation lies in defining a pseudo-symplectic invariant $S(\mathbf{q}, \mathbf{p})$ that combines concepts like Hamiltonian, symplectic form, angular momentum, and phase space scale. First, the bond direction vector `bond_direction` and bond midpoint electron density gradient `density_gradient` are normalized to obtain unit vectors $\hat{\mathbf{q}}$ and $\hat{\mathbf{p}}$. Then, the pseudo-symplectic invariant $S(\hat{\mathbf{q}}, \hat{\mathbf{p}})$ is calculated using the following composite formula:
$$
\begin{aligned}
H(\hat{\mathbf{q}}, \hat{\mathbf{p}}) &= \frac{1}{2} \|\hat{\mathbf{p}}\|^2 + \frac{1}{2} \|\hat{\mathbf{q}}\|^2 \\
\omega(\hat{\mathbf{q}}, \hat{\mathbf{p}}) &= \hat{\mathbf{q}} \cdot \hat{\mathbf{p}} \\
L(\hat{\mathbf{q}}, \hat{\mathbf{p}}) &= \|\hat{\mathbf{q}} \times \hat{\mathbf{p}}\| \\
\text{PS\_Scale}(\hat{\mathbf{q}}, \hat{\mathbf{p}}) &= \sqrt{\|\hat{\mathbf{q}}\|^2 + \|\hat{\mathbf{p}}\|^2} \\
S(\hat{\mathbf{q}}, \hat{\mathbf{p}}) &= H(\hat{\mathbf{q}}, \hat{\mathbf{p}}) + 0.1 \cdot |\omega(\hat{\mathbf{q}}, \hat{\mathbf{p}})| + 0.05 \cdot L(\hat{\mathbf{q}}, \hat{\mathbf{p}}) + 0.01 \cdot \text{PS\_Scale}(\hat{\mathbf{q}}, \hat{\mathbf{p}})
\end{aligned}
$$
where $H(\hat{\mathbf{q}}, \hat{\mathbf{p}})$ is analogous to the total energy of the system, $ \omega(\hat{\mathbf{q}}, \hat{\mathbf{p}})$ is the dot product of the bond direction and density gradient (representing a direct analogy to the symplectic form), $L(\hat{\mathbf{q}}, \hat{\mathbf{p}})$ is the magnitude of the cross product of the two vectors (representing an analogy to angular momentum), and $\text{PS\_Scale}(\hat{\mathbf{q}}, \hat{\mathbf{p}})$ reflects the overall scale of the phase space. The weights of these components are heuristic, aiming to balance the contribution of each physical quantity.

Physically, `pseudo_symplectic_coupling` quantifies the deep interaction between the chemical bond's geometric structure (direction) and the change in electron density at the bond midpoint (gradient). High values indicate strong coupling between the bond direction and the electron flow or charge polarization direction, which is crucial for understanding electron transport, optoelectronic response, and nonlinear optical properties. Chemically, this coupling reflects the sensitivity of electron distribution to spatial geometric arrangement during bonding, helping to explain the stability or reactivity of specific bonds. In materials science, this feature can serve as an indicator for predicting functional properties such as material conductivity, dielectric constant, and piezoelectricity, especially in materials with significant structural-electronic coupling effects like perovskites, where it can reveal how microscopic structural distortions affect electron behavior. In computer science and artificial intelligence fields, `pseudo_symplectic_coupling` provides a continuous, differentiable, and profoundly physically meaningful quantification of structure-electron coupling, serving as a powerful feature for building machine learning models that can predict the properties of complex functional materials and optimize material design, aiding in understanding the intrinsic mechanisms of materials from a white-box perspective.

### 4.4.4 Tensor Algebraic Environment Alignment (`tensor_algebraic_environment_alignment`)

`tensor_algebraic_environment_alignment` descriptor is strictly defined as the tensor alignment degree between the local environment structure tensors $\mathbf{T}_{\text{struct},i}$ and $\mathbf{T}_{\text{struct},j}$ of atoms $i$ and $j$ at both ends of a chemical bond. This feature aims to quantify the similarity or consistency of these two local geometric environments in terms of spatial orientation and shape. High alignment indicates that the local structural environments of the atoms at both bond ends are geometrically highly coordinated, while low alignment predicts structural mismatch or stress.

The calculation of this alignment degree integrates multiple tensor similarity measures to provide a robust comprehensive indicator. Specifically, it combines normalized tensor inner product, principal direction alignment, and tensor cosine similarity. The calculation steps are as follows:

1.  **Normalized Tensor Inner Product:** First, calculate the Frobenius norm of the two structure tensors $\|\mathbf{T}\|_F = \sqrt{\text{Tr}(\mathbf{T}^T \mathbf{T})}$. Then, the overall similarity is measured by calculating the trace inner product $\text{Tr}(\mathbf{T}_{\text{norm},i}^T \mathbf{T}_{\text{norm},j})$ between the normalized tensors $\mathbf{T}_{\text{norm},k} = \mathbf{T}_k / \|\mathbf{T}_k\|_F$.
2.  **Principal Direction Alignment:** Extract the principal direction vectors corresponding to the maximum eigenvalues of the two structural tensors through their respective eigenvalue decompositions. Then calculate the cosine value (absolute value) of the angle between these two principal direction vectors to assess the alignment of their primary anisotropic directions.
3.  **Tensor Cosine Similarity:** Treat the two tensors as high-dimensional vectors and calculate their cosine similarity, i.e., $\frac{\sum_{m,n} (\mathbf{T}_{i})_{mn} (\mathbf{T}_{j})_{mn}}{\|\mathbf{T}_i\|_F \|\mathbf{T}_j\|_F}$.

The final `tensor_algebraic_environment_alignment` value is the weighted average of these three components (normalized tensor inner product, principal direction alignment, and tensor cosine similarity), with weights of 0.4, 0.4, and 0.2 respectively, and is clipped to the [0, 1] range. The structure tensors $\mathbf{T}_{\text{struct},k}$ of the bond-end atoms originate from the concept of the structure tensor introduced in Section 3.4.2's mean squared neighbor distance.

Physically, this feature provides a continuous quantification of the matching degree of local geometric environments at both ends of a bond. High alignment indicates that the stress distribution and geometric deformation trends in the local regions are highly consistent, which is crucial for understanding lattice strain propagation and structural stability. Chemically, it reflects the compatibility of adjacent atomic clusters in shape and orientation, aiding in predicting the strength and directionality of bonding. In materials science, `tensor_algebraic_environment_alignment` is key to understanding crystal structural stability, phase behavior, mechanical properties (e.g., elastic modulus, hardness), and the formation of grain boundaries and defects, providing guidance for designing materials with specific mechanical or thermodynamic properties. In computer science and artificial intelligence fields, this feature provides machine learning models with a higher-order description of complex local structural geometric compatibility, aiding models in predicting macroscopic material properties, especially in high-throughput screening and inverse design, where it can identify candidate materials with ideal local structural matching.

### 4.4.5 Lie Algebraic Bond Alignment (`lie_algebraic_bond_alignment_a`, `lie_algebraic_bond_alignment_b`, `lie_algebraic_bond_alignment_c`)

`lie_algebraic_bond_alignment_a/b/c` descriptors utilize the adjoint action of the Lie algebra $\mathfrak{so}(3)$ to quantify the algebraic alignment between chemical bond directions and the three principal axes (a, b, c) of the crystal lattice. This feature treats bond direction vectors and lattice basis vectors as elements in the Lie algebra $\mathfrak{so}(3)$ (represented by their corresponding antisymmetric matrices) and calculates the Frobenius norm of their Lie algebra commutator. A small norm indicates high alignment (i.e., less symmetry breaking of the bond direction with respect to the lattice axes), while a large norm indicates low alignment or significant symmetry breaking effects. This method transcends simple geometric angles, algebraically evaluating the coordination between bond direction and lattice periodicity.

Specifically, for the normalized bond direction vector $\hat{\mathbf{v}}_{\text{bond}}$ and the unit basis vectors of the lattice $\hat{\mathbf{a}}, \hat{\mathbf{b}}, \hat{\mathbf{c}}$, we construct their corresponding antisymmetric matrices $\mathbf{M}_{\text{bond}}, \mathbf{M}_{\text{a}}, \mathbf{M}_{\text{b}}$, and $\mathbf{M}_{\text{c}}$. The Lie algebraic bond alignment is measured by calculating the Frobenius norm of the Lie algebra commutator $\mathcal{C}(\mathbf{M}_{\text{bond}}, \mathbf{M}_{\text{axis}}) = \mathbf{M}_{\text{bond}} \mathbf{M}_{\text{axis}} - \mathbf{M}_{\text{axis}} \mathbf{M}_{\text{bond}}$, and then normalized to the [0, 1] range using an exponential function, such that smaller norms result in larger alignment values (closer to 1). Its core mathematical expression is:
$$
\text{Alignment}_{\text{axis}} = \exp\left(-\|\mathcal{C}(\mathbf{M}_{\text{bond}}, \mathbf{M}_{\text{axis}})\|_F\right)
$$
where $\|\cdot\|_F$ denotes the Frobenius norm.

Physically, `lie_algebraic_bond_alignment` quantifies the deep interaction between chemical bond orientation and lattice anisotropy. High alignment indicates that the bond direction is highly coordinated with the crystallographic principal axes, which is crucial for understanding anisotropic physical properties of materials (e.g., conductivity, optical response, mechanical deformation). Chemically, it reflects the preferential orientation of chemical bonds in the crystal environment, aiding in explaining the stability or strengthening effect of bonding in specific directions. In materials science, this feature is the microscopic root of macroscopic anisotropic properties of crystalline materials, providing guidance for designing materials with specific crystal orientation-dependent functions (e.g., piezoelectric, ferroelectric, thermoelectric). In computer science and artificial intelligence fields, this feature provides a continuous, differentiable quantification of the coupling between bond direction and lattice symmetry, enabling machine learning models to more accurately predict the anisotropic behavior of materials and to explore candidate structures with ideal crystallographic orientations in inverse design.

### 4.4.6 Structure-Chemistry Incompatibility Difference (`delta_structure_chemistry_incompatibility`)

`delta_structure_chemistry_incompatibility` descriptor is strictly defined as the absolute difference between the `structure_chemistry_incompatibility` features of atoms $i$ and $j$ at both ends of a chemical bond. This feature aims to capture the difference in the matching degree between local structure and chemical environment for the atoms at both ends of the chemical bond. A high difference value indicates significant inconsistency in the coupling of structural and chemical information in the local environments of the atoms at both bond ends, which may predict stress concentration or heterogeneity in the bonding region.

The calculation of this feature directly originates from the `structure_chemistry_incompatibility` feature defined in Section 3.5.1. Its mathematical expression is:
$$
\Delta \text{Incomp}_{ij}^{\text{struct-chem}} = |\text{Incomp}_{j}^{\text{struct-chem}} - \text{Incomp}_{i}^{\text{struct-chem}}|
$$
where $\text{Incomp}_{k}^{\text{struct-chem}}$ is the structure-chemistry incompatibility of atom $k$.

Physically, this feature quantifies the difference in the degree of coupling between local structure and electronic/chemical properties at the two ends of a bond. A high incompatibility difference may indicate local strain, non-uniform charge distribution, or geometric distortion in the bonding region. Chemically, it aids in understanding the heterogeneity of bonding and how electrons redistribute within the bonding region, which is crucial for explaining the formation of polar bonds, stabilization of defects, and identification of chemical reaction active sites. In materials science, `delta_structure_chemistry_incompatibility` is key to understanding local structural distortions, grain boundary characteristics, phase transition mechanisms, and the microscopic driving forces behind certain functions (e.g., ion transport, catalysis) in materials. In computer science and artificial intelligence fields, this feature, by quantifying the structure-chemistry coupling differences at the atomic level, provides machine learning models with deeper microscopic heterogeneity information, aiding in predicting material local stress, defect tolerance, and behavior under extreme conditions.

### 4.4.7 Bond Ends Anisotropy Mismatch (`bond_ends_anisotropy_mismatch`)

`bond_ends_anisotropy_mismatch` descriptor is strictly defined as the absolute difference in the `local_environment_anisotropy` features of atoms $i$ and $j$ at both ends of a chemical bond. This feature aims to quantify the difference in the degree of shape anisotropy of the local geometric environments around the atoms at both bond ends. A high mismatch value indicates significant inconsistency in the degree of symmetry deviation between the environments of the atoms at both bond ends, which may lead to structural stress or heterogeneity in the bonding region.

The calculation of this feature directly originates from the `local_environment_anisotropy` feature defined in Section 3.4.3. Its mathematical expression is:
$$
\Delta \text{Anisotropy}_{ij}^{\text{local}} = |\text{Anisotropy}_{j}^{\text{local}} - \text{Anisotropy}_{i}^{\text{local}}|
$$
where $\text{Anisotropy}_{k}^{\text{local}}$ is the local environment anisotropy of atom $k$.

Physically, this feature quantifies the difference in local spatial packing efficiency or geometric symmetry deviation between the atoms at both ends of a bond. A high mismatch value may indicate local strain concentration in the bonding region, especially at interfaces between regions of different symmetries. Chemically, it aids in understanding the heterogeneity of the local bonding environment and how this heterogeneity affects bond properties and stability. In materials science, `bond_ends_anisotropy_mismatch` is key to understanding local structural distortions, phase transition mechanisms, mechanical response (e.g., brittleness), and certain transport properties (e.g., heat conduction, phonon transport) in crystalline materials, particularly in those with significant lattice distortions. In computer science and artificial intelligence fields, this feature provides a continuous quantification of the difference in local geometric anisotropy in the bonding region, providing machine learning models with information about microscopic stress distribution and local structural heterogeneity within materials, aiding in predicting material mechanical strength, thermodynamic stability, and behavior under complex stress.
