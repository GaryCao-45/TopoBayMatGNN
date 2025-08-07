# Chapter 3: 0-Simplex Descriptors - In-depth Characterization and Multi-dimensional Representation of Atom-Centered Environments

## 3.1 Introduction: Atoms as 0-Simplices – Fine-grained Decoding of Periodic Materials' Microscopic Building Blocks

In the descriptor system for crystalline materials, the 0-simplex is the most fundamental constituent unit, physically corresponding directly to individual atoms within the system. The primary goal of the 0-simplex descriptors constructed in this chapter is to transcend the elemental properties of atoms themselves and delve into a refined, quantitative characterization of their local chemical and physical environments. Traditional descriptors, such as atomic number or electronegativity, can only provide information about isolated atoms and fail to capture the complex changes an atom undergoes within the lattice due to interactions with other atoms.

To this end, our designed 0-simplex descriptors not only cover the intrinsic properties of atoms but, more importantly, anchor a series of local physical fields (such as electron density, Bader charge, electrostatic potential, etc.) obtained from first-principles calculations, as well as structural geometric information defined through algebraic topological methods, onto each atomic center. This approach transforms each 0-simplex (atom) from a simple elemental symbol into a high-dimensional vector carrying rich structural-physical coupling information.

This chapter will elaborate on the mathematical definitions, physicochemical implications, and computational implementations of each type of 0-simplex descriptor. These descriptors collectively form a high-dimensional atomic "fingerprint," laying a solid foundation for constructing more complex cross-scale features in higher-order simplices (e.g., 1-simplex/chemical bonds, 2-simplex/three-body interactions).

## 3.2 Basic Physicochemical Descriptors: Quantification of Intrinsic Atomic Properties and Classical Environment

This section aims to detail a series of basic physicochemical descriptors at the atomic level, which serve as cornerstones for defining atomic identity and their fundamental roles in crystalline materials. These attributes are not only central to elemental periodic behavior but also starting points for building more complex material features. These descriptors are primarily obtained directly from standard chemical information libraries (e.g., Mendeleev, Pymatgen) or derived through basic crystallochemical analysis.

### 3.2.1 Atomic Number (`atomic_number`)

Atomic number $Z$ is strictly defined as the number of protons in an atomic nucleus, a fundamental quantity determining an element's chemical properties and its position in the periodic table. From a physics, chemistry, and materials science perspective, atomic number, as the most basic identifier, directly determines an atom's nuclear charge and electron configuration, thereby influencing its size and all chemical properties. In crystalline material descriptions, it is an indispensable fundamental constituent feature, laying the groundwork for constructing combinatorial chemical features and understanding interatomic interactions.

### 3.2.2 Electronegativity (`electronegativity`)

Electronegativity $\chi$ (Pauling scale) is strictly defined as the relative measure of an atom's ability to attract electrons in a chemical bond. From physics, chemistry, and materials science perspectives, electronegativity is a crucial indicator for determining chemical bond types (ionic, covalent, polar covalent) and bonding polarity; a larger electronegativity difference between atoms implies stronger ionic character of the bond. In crystalline materials, electronegativity influences the material's dielectric constant, work function, band structure, and chemical reactivity. Particularly in composite materials, differences in electronegativity among different elements drive charge transfer, forming dipoles, which in turn affect the material's polarization behavior and interfacial properties, being especially critical for the functionality of polar materials like perovskites. For computer science and artificial intelligence disciplines, as a continuous numerical feature, electronegativity can help machine learning models learn and identify the polarity strength of chemical bonds, thereby predicting material properties related to electronic structure and interfacial characteristics.

### 3.2.3 First Ionization Energy (`ionization_energy`)

First ionization energy $I_1$ is strictly defined as the minimum energy required to remove one electron from a gaseous isolated ground-state atom, with its unit being kJ/mol. From physics, chemistry, and materials science perspectives, ionization energy reflects an atom's ability to bind outer-shell electrons; higher energy means electrons are more difficult to remove. It is closely related to an atom's metallic character, reducing ability, and ease of participating in redox reactions; atoms with low ionization energy tend to form cations and contribute free electrons, thereby influencing the conductivity of crystalline materials (e.g., electron excitation thresholds in metallic conductors and semiconductors), as well as the excitation thresholds in photovoltaic materials during photoelectric effects. In computer science and artificial intelligence disciplines, as a continuous numerical feature, ionization energy helps models understand an atom's tendency to form cations and its role in electron transport and electrochemical reactions, serving as an important input feature for predicting the electrical and optoelectronic properties of materials.

### 3.2.4 Electron Affinity (`electron_affinity`)

Electron affinity $E_{ea}$ is strictly defined as the energy released when a gaseous isolated ground-state atom accepts an electron to form a negative ion, with its unit being kJ/mol. From physics, chemistry, and materials science perspectives, electron affinity reflects an atom's ability to capture electrons; higher energy (more energy released) means the atom more readily forms a negative ion. It is closely related to an atom's non-metallic character, oxidizing ability, and ease of participating in chemical reactions; atoms with high electron affinity can influence the work function of crystalline materials (critical for electron injection/extraction interfaces), the conductivity type of semiconductors (n-type or p-type doping), and electron capture capabilities in charge transport and catalytic reactions. For computer science and artificial intelligence disciplines, as a continuous numerical feature, electron affinity helps models understand an atom's tendency to form anions and its role in electron transport. Together with ionization energy, it provides models with a comprehensive quantification of an atom's electron gain/loss abilities, offering guidance for predicting the performance of semiconductors, photovoltaic materials, and catalysts.

### 3.2.5 Valence Electrons (`valence_electrons`)

Valence electron count $N_v$ is strictly defined as the number of electrons in an atom's outermost shell, which typically participate in chemical bond formation. From physics, chemistry, and materials science perspectives, the valence electron count directly determines an atom's bonding capacity and its valence state in chemical bonds. It is a core parameter for chemical bond formation, valency, and crystal material structure prediction, influencing material conductivity (valence electrons forming conduction or valence bands), optical properties (electron transitions), and chemical reactivity. For semiconductors, the valence electron count determines their potential as n-type or p-type dopants. In computer science and artificial intelligence disciplines, it is a discrete numerical feature reflecting an atom's activity in chemical bonding, highly useful for predicting electronic structure-related material properties (e.g., conductivity, optical absorption).

### 3.2.6 Ionic Radius (`ionic_radius`)

Ionic radius $r_{ion}$ is strictly defined as the effective spatial size occupied by an ion in a crystalline material, depending on the ion's charge, coordination number, and spin state. From physics, chemistry, and materials science perspectives, ionic radius determines an ion's spatial packing efficiency, lattice constant, and unit cell volume, serving as a key factor for predicting the structural stability of ionic crystalline materials (e.g., Goldschmidt tolerance factor, Pauling's rules), ion mobility, and solid solution formation. It directly influences a material's density, melting point, hardness, and the size of ion migration channels in ionic conductors. In computer science and artificial intelligence disciplines, as a continuous numerical feature, ionic radius reflects an ion's spatial contribution to crystal packing, playing an important role in predicting material structural stability, synthesizability, density, and in assessing the "openness" of ion transport channels in solid-state ionic conductors.

### 3.2.7 Covalent Radius (`covalent_radius`)

Covalent radius $r_{cov}$ is strictly defined as the effective spatial size occupied by an atom in a covalent bond, with units of Angstroms (Å). From physics, chemistry, and materials science perspectives, covalent radius determines bond lengths and bond angles in covalent bonds, serving as a key factor for predicting covalent bond lengths, molecular geometry, and the structure of covalent crystalline materials. It influences the density, hardness, and covalent bond-related electronic structures (e.g., band structure) of covalent materials. In computer science and artificial intelligence disciplines, covalent radius is a continuous numerical feature reflecting an atom's spatial contribution in covalent bonding. It is often used in conjunction with actual bond lengths to form higher-order features like bond length distortion, providing fine-grained information about the bonding environment for machine learning models.

### 3.2.8 Coordination Number (`coordination_number`)

Coordination number $CN$ is strictly defined as the number of nearest-neighbor atoms directly bonded to a central atom in a crystalline material. This framework uniformly employs the `pymatgen.analysis.local_env.CrystalNN` algorithm to determine coordination numbers. The CrystalNN algorithm identifies chemical bonds by analyzing interatomic distances, elemental properties, and periodic boundary conditions, and then counts the number of atoms directly connected to the central atom. This is a complex process combining geometric judgment and chemical rules, aiming to simulate the real chemical bonding environment. From physics, chemistry, and materials science perspectives, coordination number directly reflects the spatial packing and local interaction strength of an atom within the crystal lattice. It is a core indicator for determining atomic valence, bond types, and local geometric configurations, directly influencing the shape and connectivity of coordination polyhedra, which in turn affects material density, stability, electronic structure (e.g., crystal field splitting), and many macroscopic physical properties (e.g., hardness, melting point, ion mobility). Changes in coordination number are often associated with phase transitions or defect formation. In computer science and artificial intelligence disciplines, as a discrete numerical feature with clear physical meaning, coordination number can be part of node features in graph neural networks or used as classification or regression features in traditional machine learning models, providing a basic description for models to understand the topological properties of crystal networks.

### 3.2.9 Average Site Valence (`avg_site_valence`)

Average site valence $V_{avg}$ is strictly defined as the formal oxidation state of an atom as calculated by `pymatgen.analysis.bond_valence.BVAnalyzer`. This analyzer attempts to determine the most probable oxidation state for each atom based on the crystal structure and chemical formula of the crystalline material. The BVAnalyzer internally contains a set of heuristic rules and a bond valence model, designed to infer the most reasonable atomic oxidation states from crystal structure and chemical formula. From physics, chemistry, and materials science perspectives, this feature reflects the average charge state of an atom within the crystal, related to electron transfer and localization between atoms. It is an important parameter for predicting stoichiometry, redox reactions, and compound stability. In materials science, it influences material conductivity (valence changes can lead to semiconductor or insulator behavior), magnetism (unpaired electrons), and catalytic activity (redox centers). In computer science and artificial intelligence disciplines, average site valence reflects the redox state of an atom in crystalline materials, holding significant importance for predicting material electrochemical and catalytic properties.

### 3.2.10 Bond Valence Sum (`bond_valence_sum`)

Bond valence sum $\text{BVS}_i$ is strictly defined as the total sum of bond valences for all chemical bonds formed by a central atom $i$. The bond valence $s_{ij}$ for each bond is calculated based on bond length $d_{ij}$ and empirical parameters $R_0, B$:
    $$
    s_{ij} = \exp\left(\frac{R_0 - d_{ij}}{B}\right)
    $$
where $R_0$ and $B$ are empirical constants related to specific element pairs, typically sourced from specialized bond valence parameter databases (e.g., the Brese and O'Keeffe parameters integrated within the pymatgen library). The bond valence sum $\text{BVS}_i$ for central atom $i$ is:
    $$
    \text{BVS}_i = \sum_{j \in \text{NN}(i)} s_{ij}
    $$
Ideally, $\text{BVS}_i$ should approach the formal oxidation state of central atom $i$. BVS is a semi-empirical model whose core idea is that the sum of bond valences for an atom should equal its formal oxidation state. Each bond's bond valence $s_{ij}$ converts bond length into a valence contribution via an empirical formula, where $R_0$ and $B$ parameters are obtained by fitting a large number of known structures. The sum is an accumulation of bond valences from all nearest neighbors. From physics, chemistry, and materials science perspectives, bond valence sum reflects the charge balance of the atom's local bonding environment and the presence of stress (over-coordination or under-coordination). It is a crucial tool for validating assumed atomic oxidation states and diagnosing unusual bonding states in crystalline material structures, being highly sensitive to geometric distortions. In materials science, it is often used to assess material structural stability, ion migration pathways (deviation of BVS from ideal values may indicate ion migration sites), and catalytic activity. High BVS deviation may imply local strain or a tendency towards amorphization. In computer science and artificial intelligence disciplines, it is a continuous numerical feature providing a quantification of local charge balance and strain states, capable of capturing subtle changes in structure, and is highly useful for predicting material structural stability, phase behavior, and ion transport properties.

### 3.2.11 Bond Length Distortion (`bond_length_distortion`)

The bond length distortion descriptor is strictly defined as the relative standard deviation of all nearest-neighbor bond lengths around a central atom. It quantifies the uniformity of bond lengths within the local coordination polyhedron, reflecting the degree of geometric distortion. Its formula is defined as:
    $$
\text{Bond Length Distortion} = \sqrt{\frac{1}{N_{\text{NN}}} \sum_{j=1}^{N_{\text{NN}}} \left(\frac{d_{ij} - \overline{d_i}}{\overline{d_i}}\right)^2} \quad (\text{when } N_{\text{NN}} > 0 \text{})
    $$
where $d_{ij}$ is the bond length from central atom $i$ to its nearest neighbor $j$, $\overline{d_i}$ is the average bond length of central atom $i$, and $N_{\text{NN}}$ is the coordination number (obtained from Section 4.2.8). When $N_{\text{NN}} \le 1$ , the distortion index is defined as 0.

---

## 3.3 Quantum Chemical Core Descriptors: Fine-grained Characterization of Electronic Structure and Local Physical Fields

The descriptors in this section are extracted directly from density functional theory (DFT) first-principles calculation results, aiming to deeply reveal the electronic structure details, charge distribution characteristics, local magnetism, and energy density of states of atoms in crystalline materials. These features constitute the quantum mechanical basis for understanding the intrinsic physicochemical properties of crystalline materials (e.g., conductivity, optical properties, catalytic activity, magnetism), possessing high physical authenticity and predictive power.

### 3.3.1 Bader Charge (`bader_charge`)

Bader charge $q_i$ is strictly defined as the difference between the number of electrons in an atom's nucleus and its Bader volume. The Bader volume is determined by the Quantum Theory of Atoms in Molecules (QTAIM) concept, through topological analysis of the electron density $\rho(\mathbf{r})$. Its core lies in the zero-flux surface, where the electron density gradient $\nabla\rho(\mathbf{r})$ is everywhere zero or perpendicular to the surface. The Bader charge $q_i$ of atom $i$ is strictly defined as:
    $$
    q_i = Z_i - \int_{\Omeg-i} \rho(\mathbf{r}) d^3\mathbf{r}
    $$
where $Z_i$ is the atomic number of atom $i$ (nuclear charge), $\Omeg-i$ is the Bader volume of atom $i$ delineated by QTAIM, and $\int_{\Omeg-i} \rho(\mathbf{r}) d^3\mathbf{r}$ is the total number of electrons integrated within this volume.

From physics, chemistry, and materials science perspectives, Bader charge directly quantifies the actual charge state of an atom in a crystalline material, reflecting the degree of charge transfer and localization between atoms. A positive value indicates the atom has lost electrons (cationic character), while a negative value indicates the atom has gained electrons (anionic character). It is a quantitative indicator for judging the strength of ionic/covalent character of chemical bonds, and is closely related to chemical reactivity, redox ability, and intermolecular interactions. For multi-valent atoms, Bader charge can distinguish their actual valence states in different environments. In materials science, it influences charge transport (e.g., carriers in ionic conductors and semiconductors), dielectric properties (e.g., polarization intensity), catalytic activity (charge state of active sites), and surface properties (e.g., work function). In ionic perovskites and other crystalline materials, the distribution of Bader charge significantly affects lattice stability, ferroelectricity, and band gap properties. In computer science and artificial intelligence disciplines, as a continuous numerical feature, Bader charge provides a high-precision, physically authentic quantification of atomic charge states, serving as a core input for building electron-structure-property prediction models.

### 3.3.2 Electrostatic Potential (`electrostatic_potential`)

Electrostatic potential $V(\mathbf{r})$ is strictly defined as the electrostatic potential value at each atomic position $\mathbf{r}$ in a crystalline material. In DFT, the electrostatic potential is the average electric field distribution generated by electrons and atomic nuclei in space, which can be obtained by high-precision trilinear interpolation of the electrostatic potential grid data calculated by DFT.

Trilinear interpolation is a type of multi-dimensional interpolation that estimates the value at an arbitrary point by performing three linear interpolations on known data points within a small cube (defined by eight nearest grid points) containing the target point. Suppose we have a function $f(x, y, z)$, and a target point $(x_d, y_d, z_d)$. This target point is enclosed by a cube defined by $(x_1, y_1, z_1)$ and $(x_2, y_2, z_2)$, where $x_1 \le x_d \le x_2$, $y_1 \le y_d \le y_2$, $z_1 \le z_d \le z_2$. The interpolation process can be decomposed into the following steps:

First, perform two linear interpolations along the $x$-axis to get interpolation results for four points on four edges:
    $$
    \begin{aligned}
    f_{11}(x_d) &= f(x_1, y_1, z_1) \frac{x_2 - x_d}{x_2 - x_1} + f(x_2, y_1, z_1) \frac{x_d - x_1}{x_2 - x_1} \\
    f_{21}(x_d) &= f(x_1, y_2, z_1) \frac{x_2 - x_d}{x_2 - x_1} + f(x_2, y_2, z_1) \frac{x_d - x_1}{x_2 - x_1} \\
    f_{12}(x_d) &= f(x_1, y_1, z_2) \frac{x_2 - x_d}{x_2 - x_1} + f(x_2, y_1, z_2) \frac{x_d - x_1}{x_2 - x_1} \\
    f_{22}(x_d) &= f(x_1, y_2, z_2) \frac{x_2 - x_d}{x_2 - x_1} + f(x_2, y_2, z_2) \frac{x_d - x_1}{x_2 - x_1}
    \end{aligned}
    $$
Next, use these four points to perform two linear interpolations along the $y$-axis:
    $$
    \begin{aligned}
    f_1(x_d, y_d) &= f_{11}(x_d) \frac{y_2 - y_d}{y_2 - y_1} + f_{21}(x_d) \frac{y_d - y_1}{y_2 - y_1} \\
    f_2(x_d, y_d) &= f_{12}(x_d) \frac{y_2 - y_d}{y_2 - y_1} + f_{22}(x_d) \frac{y_d - y_1}{y_2 - y_1}
    \end{aligned}
    $$
Finally, perform one linear interpolation on these two results along the $z$-axis to get the final target point value:
    $$
    f(x_d, y_d, z_d) = f_1(x_d, y_d) \frac{z_2 - z_d}{z_2 - z_1} + f_2(x_d, y_d) \frac{z_d - z_1}{z_2 - z_1}
    $$
This process ensures that local values at atomic nuclear sites or key surrounding regions are precisely estimated from continuous field distributions, thereby bridging the gap between grid-based calculations and atom-local property descriptions, laying the data foundation for subsequent advanced feature construction. Through this precise interpolation method, various physical quantities at atomic sites can be further defined and calculated.

The electrostatic potential $V_i$ at atom $i$, represents the local value obtained at the Cartesian coordinates $\mathbf{r}_i$ of atom $i$ from the 3D grid data of electrostatic potential $V_{\text{grid}}$ calculated by DFT, using the high-precision trilinear interpolation method ($\text{Interpolate}$) described above. It is strictly defined as:
    $$
    V_i = V(\mathbf{r}_i) = \text{Interpolate}(V_{\text{grid}}, \mathbf{r}_i)
    $$
where $V(\mathbf{r}_i)$ is the numerical value of the electrostatic potential at spatial point $\mathbf{r}_i$, which is precisely obtained by trilinear interpolation of $V_{\text{grid}}$ (3D grid data of electrostatic potential for the entire unit cell calculated by DFT) at $\mathbf{r}_i$.

From physics, chemistry, and materials science perspectives, electrostatic potential reflects the strength and direction of the electric field in the atom's local environment. Local highs and lows of electrostatic potential directly influence electron migration pathways and energy barriers within the lattice. It is closely related to an atom's electrophilicity/nucleophilicity, Lewis acidity/basicity, and chemical reactivity sites. For instance, highly negative potential regions are typically targets for electrophilic attack. In materials science, it affects a material's dielectric constant, work function, charge transport (e.g., in ionic and electronic conductors), catalytic activity, and adsorption properties. The gradient of the electrostatic potential drives the movement of charge carriers. In computer science and artificial intelligence disciplines, as a continuous numerical feature, electrostatic potential provides a fine-grained quantification of the atom's local potential environment, aiding models in understanding and predicting charge behavior in materials.

### 3.3.3 Electron Density (`electron_density`)

Electron density $\rho(\mathbf{r})$ is strictly defined as the electron probability density value at each atomic position $\mathbf{r}$ in a crystalline material. It quantifies the probability of finding an electron at a given spatial point. In DFT, the total electron density is given by the sum of electron densities of all occupied orbitals. The electron density $\rho_i$ at atom $i$ is strictly defined as:
    $$
    \rho_i = \rho(\mathbf{r}_i) = \text{Interpolate}(\rho_{\text{grid}}, \mathbf{r}_i)
    $$
where $\rho_{\text{grid}}$ is the 3D grid data of electron density for the entire unit cell calculated by DFT, and $\mathbf{r}_i$ is the Cartesian coordinate of atom $i$.

From physics, chemistry, and materials science perspectives, electron density is a fundamental quantity in quantum mechanics, and its distribution determines almost all physical properties of crystalline materials. Local peaks in electron density correspond to atomic nuclei or covalent bond regions, while low values correspond to interatomic spaces. It is related to chemical bond formation, bond order, interatomic interaction strength, and the Kato Cusp Condition (electron density exhibits sharp peaks at atomic nuclei). In materials science, it affects material bonding types (e.g., covalent, ionic, metallic bonds), hardness, elastic moduli, phonon spectra (depending on bonding strength), optical properties (e.g., refractive index), and Mössbauer and NMR spectroscopy (directly related to the electronic environment around atomic nuclei). In computer science and artificial intelligence disciplines, as a continuous numerical feature, electron density provides the most direct and fundamental quantum mechanical quantification of the atom's local electronic environment.

### 3.3.4 Electron Localization Function (ELF) (`elf`)

The Electron Localization Function (ELF) is a dimensionless function ranging from 0 to 1 that quantifies the degree of electron localization in space. An ELF value close to 1 indicates high electron localization (e.g., in covalent bonds or lone pair regions), a value close to 0.5 indicates free-electron-like behavior (e.g., in metals), and a value close to 0 indicates delocalization or interatomic regions. The strict definition of ELF is based on a comparison of the electron pair probability density with that in a uniform electron gas. In DFT calculations, it is typically calculated via the kinetic energy density. The ELF value $\text{ELF}_i$ at atom $i$ is strictly defined as:
    $$
    \text{ELF}_i = \text{ELF}(\mathbf{r}_i) = \text{Interpolate}(\text{ELF}_{\text{grid}}, \mathbf{r}_i)
    $$
where $\text{ELF}_{\text{grid}}$ is the 3D grid data of ELF for the entire unit cell obtained from GPAW's ELF module.

From physics, chemistry, and materials science perspectives, ELF visually reflects the "shell" or "bonding" characteristics of electrons in space, providing a way to visualize chemical bonds (covalent, ionic, metallic bonds), lone pairs, and core electron regions. It is a powerful tool for determining chemical bond types, bond strengths, atomic valence, and the presence of lone pairs. High ELF regions correspond to the center of covalent bonds or non-bonding electron pairs. In materials science, it influences the bonding strength, hardness, insulating properties (high ELF regions indicate electrons are not easily mobile), and active sites in chemical reactions of crystalline materials. In covalent materials and semiconductors, the distribution of ELF is crucial for understanding carrier transport mechanisms. In computer science and artificial intelligence disciplines, as a continuous numerical feature, ELF provides a high-resolution quantification of the degree of local electron localization at atoms, aiding models in learning and predicting material properties related to bonding type and electron behavior.

### 3.3.5 Local Magnetic Moment (`local_magnetic_moment`)

Local magnetic moment $\mu_i$ is strictly defined as the net magnetic moment contributed by each atom, typically obtained through atomic orbital projection or Wigner-Seitz sphere integration in spin-polarized DFT calculations. For spin-polarized systems, it represents the integral of the difference between up-spin and down-spin electron densities within the atomic region. The local magnetic moment $\mu_i$ of atom $i$ is strictly defined as:
    $$
    \mu_i = \int_{\Omeg-i} (\rho_{\uparrow}(\mathbf{r}) - \rho_{\downarrow}(\mathbf{r})) d^3\mathbf{r}
    $$
where $\rho_{\uparrow}$ and $\rho_{\downarrow}$ are up-spin and down-spin electron densities, respectively, and $\Omeg-i$ is the region of atom $i$.

From physics, chemistry, and materials science perspectives, local magnetic moment directly reflects the magnetic state of an atom in a crystalline material (e.g., ferromagnetism, anti-ferromagnetism, paramagnetism). Non-zero magnetic moments indicate the presence of unpaired electrons. It is closely related to the d-orbital electron configuration of transition metal ions, spin states, and ligand field theory. In materials science, it is a key parameter for designing and understanding magnetic materials, spintronic materials, magnetic storage materials, and magnetic catalysts. The distribution of local magnetic moments determines the overall magnetic order and magnetic anisotropy of a material. In computer science and artificial intelligence disciplines, as a continuous numerical feature, local magnetic moment provides a precise quantification of the atom's local magnetic state, serving as a core input for building magnetic material design models.

### 3.3.6 Local Density of States at Fermi Level (`local_dos_fermi`)

The local density of states (LDOS) $\text{LDOS}_i(\varepsilon_F)$ at the Fermi level $\varepsilon_F$ for atom $i$ is strictly defined as the number of electronic states contributed by atom $i$ within a unit energy interval near the Fermi level. It is usually obtained by interpolation of the projected density of states (PDOS) or by integration near the Fermi level.

From physics, chemistry, and materials science perspectives, `local_dos_fermi` reflects the electronic local density of states near the Fermi level for an atom, serving as a key indicator for determining the conductivity of crystalline materials (metal, semiconductor, insulator). High LDOS values usually indicate a significant contribution of the atom to conductivity. It is closely related to the atom's bonding ability in the crystal, orbital hybridization modes, and chemical reactivity sites. In materials science, it directly affects material electrical conductivity, semiconductor band gap properties, Schottky barrier, and photoelectric conversion efficiency. In computer science and artificial intelligence disciplines, as a quantum descriptor directly reflecting the atom's local electronic structure and bonding characteristics, it is an indispensable input for building high-precision material property prediction models.

### 3.3.7 s-orbital Electron Count (`s_electron_count`)

The s-orbital electron count $N_{i,s}$ for atom $i$ is strictly defined as the total number of electrons integrated from the valence band minimum to the Fermi level in the s-orbital projected density of states of that atom:
$$
N_{i,s} = \int_{E_{\text{valence}}}^{E_F} \text{PDOS}_{i,s}(E) dE
$$
where $\text{PDOS}_{i,s}(E)$ is the s-orbital projected density of states of atom $i$ at energy $E$.

From physics, chemistry, and materials science perspectives, `s_electron_count` reveals the electron occupancy in the s-orbital and the bonding type of the atom, typically associated with isotropic spherical symmetrical bonding. In materials science, it influences the stability, mechanical strength (e.g., through the formation of strong directional covalent bonds), and work function of crystalline materials.

### 3.3.8 p-orbital Electron Count (`p_electron_count`)

The p-orbital electron count $N_{i,p}$ for atom $i$ is strictly defined as the total number of electrons integrated from the valence band minimum to the Fermi level in the p-orbital projected density of states of that atom:
$$
N_{i,p} = \int_{E_{\text{valence}}}^{E_F} \text{PDOS}_{i,p}(E) dE
$$
where $\text{PDOS}_{i,p}(E)$ is the p-orbital projected density of states of atom $i$ at energy $E$.

From physics, chemistry, and materials science perspectives, `p_electron_count` reveals the electron occupancy in the p-orbital and the bonding type of the atom, typically associated with directional covalent bonds and $\pi$ bonds. In materials science, it influences the conductivity (through the formation of conductive networks), optical properties (through $\pi$-electron conjugation), and chemical reactivity of crystalline materials.

### 3.3.9 d-orbital Electron Count (`d_electron_count`)

The d-orbital electron count $N_{i,d}$ for atom $i$ is strictly defined as the total number of electrons integrated from the valence band minimum to the Fermi level in the d-orbital projected density of states of that atom:
$$
N_{i,d} = \int_{E_{\text{valence}}}^{E_F} \text{PDOS}_{i,d}(E) dE
$$
where $\text{PDOS}_{i,d}(E)$ is the d-orbital projected density of states of atom $i$ at energy $E$.

From physics, chemistry, and materials science perspectives, `d_electron_count` reveals the electron occupancy in the d-orbital and the bonding type of the atom, which is particularly crucial for the magnetism, catalytic activity, and strong correlation effects of transition metals. For example, transition metals with half-filled d-orbitals generally exhibit high catalytic activity. In materials science, it directly determines the origin of magnetism and magnetic phase transitions, catalytic activity (closely related to the d-band center theory), and electronic band structure of materials. In this framework, when calculating this feature, we intelligently judge the atomic number and chemical environment of the atom and apply adaptive thresholds to ensure that calculations are performed only when d-orbitals are physically relevant, significantly enhancing the robustness and data quality of the feature. In computer science and artificial intelligence disciplines, as a quantum descriptor directly reflecting the atom's local electronic structure and bonding characteristics, it is an indispensable input for building high-precision material property prediction models (e.g., conductivity, catalytic activity, thermoelectric properties, superconductivity), and its physically driven construction endows models with stronger generalization capabilities and physical insights.

---

## 3.4 Classical Geometric and Algebraic Descriptors: Abstract Quantification of 0-Simplex Local Geometry

This section introduces a set of descriptors based on classical geometry and abstract algebraic concepts, aiming to quantify the shape, symmetry deviation, and structural rigidity of local atomic environments in crystalline materials from a higher mathematical perspective. These features go beyond simple bond lengths and angles, providing deep insights into the geometric invariance and algebraic structure of atomic spatial distribution.

### 3.4.1 Vectorial Asymmetry Norm Squared (`vectorial_asymmetry_norm_sq`)

The vectorial asymmetry norm squared $\|\mathbf{V}_{\text{asymmetry}}\|^2$ descriptor is strictly defined as the squared Euclidean norm of the vector sum of displacement vectors from the central atom to all its nearest neighbors. It directly quantifies the degree to which the local atomic environment deviates from centrosymmetry. Its formula is:
    $$
    \|\mathbf{V}_{\text{asymmetry}}\|^2 = \left\| \sum_{j \in \text{NN}(i)} (\mathbf{r}_j - \mathbf{r}_i) \right\|_2^2
    $$
where $\mathbf{r}_i$ is the coordinate of the central atom $i$, $\mathbf{r}_j$ is the coordinate of its nearest neighbor $j$, and $\text{NN}(i)$ is the set of nearest neighbors of atom $i$. When the local environment possesses centrosymmetry, this vector sum is zero, and thus the squared norm is also zero.

From physics, chemistry, and materials science perspectives, this feature directly reflects the geometric polarity of the atom's local environment. Non-zero values indicate the presence of a net dipole moment or a deviation of the charge center, which is related to piezoelectricity, ferroelectricity, and spontaneous polarization in crystalline materials. It is closely associated with coordination polyhedron distortion, bond angle strain, and stereochemical activity. For example, in ABX3 perovskites, the asymmetry of the B-site cation is directly linked to ferroelectricity. In materials science, it affects material polarity, dielectric constant, nonlinear optical effects, and mechanical response. High squared norm values may indicate that the material possesses significant spontaneous polarization intensity or is prone to deformation under external fields. In computer science and artificial intelligence disciplines, as a continuous numerical feature, it provides precise quantification of local geometric polarity, aiding machine learning models in predicting the functional properties of materials.

### 3.4.2 Mean Squared Neighbor Distance (`mean_squared_neighbor_distance`)

The mean squared neighbor distance descriptor is strictly defined as the trace of the local environment structure tensor (Structure Tensor) of a central atom divided by its coordination number (number of nearest neighbors). The structure tensor $\mathbf{T}_{\text{struct},i}$ is a second-order tensor, its definition as the sum of outer products of displacement vectors from the central atom to all its nearest neighbors:
    $$
    \mathbf{T}_{\text{struct},i} = \sum_{j \in \text{NN}(i)} \mathbf{v}_{ij} \otimes \mathbf{v}_{ij} = \sum_{j \in \text{NN}(i)} (\mathbf{r}_j - \mathbf{r}_i) \otimes (\mathbf{r}_j - \mathbf{r}_i)
    $$
where $\otimes$ denotes the outer product, and $\mathbf{v}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ represents the displacement vector from central atom $i$ to nearest neighbor $j$. The trace of this tensor $\text{Tr}(\mathbf{T}_{\text{struct},i})$ is strictly defined as:
    $$
    \text{Tr}(\mathbf{T}_{\text{struct},i}) = \sum_{j \in \text{NN}(i)} \|\mathbf{r}_j - \mathbf{r}_i\|_2^2 = \sum_{j \in \text{NN}(i)} d_{ij}^2
    $$
The final formula for mean squared neighbor distance is:
$$
\text{Mean Squared Neighbor Distance} = \frac{\text{Tr}(\mathbf{T}_{\text{struct},i})}{N_{\text{NN}}} = \frac{1}{N_{\text{NN}}} \sum_{j \in \text{NN}(i)} d_{ij}^2
$$
where $N_{\text{NN}}$ is the coordination number of central atom $i$. When $N_{\text{NN}} = 0$, this value is 0.

Although its calculation ultimately equates to the average of the squared bond lengths to all nearest neighbors, this definition based on the structure tensor provides a more rigorous mathematical foundation and conceptual universality. It quantifies the "size" or "dispersion" of the local environment as a fundamental geometric invariant, and also provides a unified framework for future extensions to characterize more complex geometric properties (through other tensor invariants) such as the anisotropy of the local environment.

From physics, chemistry, and materials science perspectives, this feature intuitively reflects the spatial "scale" of the atom's local environment. Larger values indicate a sparser distribution of nearest-neighbor atoms or longer average bond lengths. It is closely related to local volume, atomic packing density, and phonon vibration modes, among other macroscopic properties. For example, in ionic conductors, a larger mean squared neighbor distance might imply more open ion migration pathways. In computer science and artificial intelligence disciplines, as a continuous numerical feature, it provides a quantification of the spatial scale of local atomic distribution, and the rigor of its derivation ensures the reliability and interpretability of the feature.

### 3.4.3 Local Environment Anisotropy (`local_environment_anisotropy`)

The local environment anisotropy descriptor $\alph-i$ is strictly defined as a dimensionless index calculated from the eigenvalues of the atom's local environment structure tensor, used to quantify the degree of geometric anisotropy of the local environment. For the structure tensor $\mathbf{T}_{\text{struct},i}$, three non-negative real eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$ are calculated. The anisotropy index $\alph-i$ is strictly defined as:
    $$
    \alph-i = \sqrt{\frac{(\lambda_1 - \lambda_2)^2 + (\lambda_2 - \lambda_3)^2 + (\lambda_3 - \lambda_1)^2}{2 (\lambda_1 + \lambda_2 + \lambda_3)^2}}
    $$

To demonstrate the intrinsic connection of this index to the fundamental invariants of the structure tensor and to enhance its mathematical rigor, its square $\alph-i^2$ can be further derived as a function of the trace (first invariant) and squared trace (second invariant) of the structure tensor. This equivalent form clearly reveals that this descriptor is based on the intrinsic geometric relationship of the sum of squared bond lengths in the atomic local environment (given by $\text{Tr}(\mathbf{T}_{\text{struct},i})$) and its second-order moment information (given by $\text{Tr}(\mathbf{T}_{\text{struct},i}^2)$). This derivation based on tensor invariants not only provides deeper mathematical insight but also enhances the theoretical completeness and interpretability of the feature.

Mathematical Derivation:

We know that:
1.  Trace of the structure tensor: $\text{Tr}(\mathbf{T}_{\text{struct},i}) = \lambda_1 + \lambda_2 + \lambda_3$
2.  Trace of the squared structure tensor: $\text{Tr}(\mathbf{T}_{\text{struct},i}^2) = \lambda_1^2 + \lambda_2^2 + \lambda_3^2$

Now, expand the numerator of $\alph-i^2$:
$$
\begin{aligned}
& (\lambda_1 - \lambda_2)^2 + (\lambda_2 - \lambda_3)^2 + (\lambda_3 - \lambda_1)^2 \\
&\quad = (\lambda_1^2 - 2\lambda_1\lambda_2 + \lambda_2^2) + (\lambda_2^2 - 2\lambda_2\lambda_3 + \lambda_3^2) + (\lambda_3^2 - 2\lambda_3\lambda_1 + \lambda_1^2) \\
&\quad = 2(\lambda_1^2 + \lambda_2^2 + \lambda_3^2) - 2(\lambda_1\lambda_2 + \lambda_2\lambda_3 + \lambda_3\lambda_1)
\end{aligned}
$$
Using the algebraic identity $(\lambda_1 + \lambda_2 + \lambda_3)^2 = \lambda_1^2 + \lambda_2^2 + \lambda_3^2 + 2(\lambda_1\lambda_2 + \lambda_2\lambda_3 + \lambda_3\lambda_1)$, we can obtain:
$$
2(\lambda_1\lambda_2 + \lambda_2\lambda_3 + \lambda_3\lambda_1) = (\lambda_1 + \lambda_2 + \lambda_3)^2 - (\lambda_1^2 + \lambda_2^2 + \lambda_3^2)
$$
Substituting this back into the numerator expression:
$$
\begin{aligned}
\text{Numerator } &= 2(\lambda_1^2 + \lambda_2^2 + \lambda_3^2) - \left[(\lambda_1 + \lambda_2 + \lambda_3)^2 - (\lambda_1^2 + \lambda_2^2 + \lambda_3^2)\right] \\
&= 3(\lambda_1^2 + \lambda_2^2 + \lambda_3^2) - (\lambda_1 + \lambda_2 + \lambda_3)^2
\end{aligned}
$$
Replacing with trace notation:
$$
\text{Numerator } = 3 \cdot \text{Tr}(\mathbf{T}_{\text{struct},i}^2) - (\text{Tr}(\mathbf{T}_{\text{struct},i}))^2
$$
Now, substitute both the numerator and denominator with trace notation back into the definition of $\alph-i^2$:
$$
\begin{aligned}
\alph-i^2 &= \frac{3 \cdot \text{Tr}(\mathbf{T}_{\text{struct},i}^2) - (\text{Tr}(\mathbf{T}_{\text{struct},i}))^2}{2 (\text{Tr}(\mathbf{T}_{\text{struct},i}))^2} \\
&= \frac{3}{2} \frac{\text{Tr}(\mathbf{T}_{\text{struct},i}^2)}{(\text{Tr}(\mathbf{T}_{\text{struct},i}))^2} - \frac{1}{2}
\end{aligned}
$$
Therefore, the local environment anisotropy index can be expressed as:
$$
\alph-i = \sqrt{\frac{3}{2} \frac{\text{Tr}(\mathbf{T}_{\text{struct},i}^2)}{(\text{Tr}(\mathbf{T}_{\text{struct},i}))^2} - \frac{1}{2}}
$$
This index ranges from $[0, 1]$: when $\alph-i = 0$, the local environment is completely isotropic (e.g., perfect spherical or cubic symmetry); when $\alph-i = 1$, the local environment is highly anisotropic (e.g., a one-dimensional chain-like structure).

From physics, chemistry, and materials science perspectives, this feature reflects the degree of geometric distortion and spatial anisotropy of the atom's local environment. A high index indicates that local electric fields, strains, or energy distributions may be non-uniform, thereby affecting physical processes such as charge transport and phonon scattering. It is related to the shape of coordination polyhedra and the non-uniformity of bond angle and bond length distributions, aiding in understanding the influence of local bonding on macroscopic properties. In materials science, it affects the mechanical properties (e.g., anisotropic elastic moduli), heat transport (e.g., anisotropic thermal conductivity), optical properties (e.g., birefringence), and electron transport of crystalline materials. Highly anisotropic local environments play crucial roles in functional materials (e.g., ferroelectrics, piezoelectrics, thermoelectric materials). In computer science and artificial intelligence disciplines, as a continuous numerical feature, it provides high-resolution quantification of local environment geometric anisotropy, aiding machine learning models in predicting structure-related functional properties.

### 3.4.4 Symmetry Breaking Quotient (`symmetry_breaking_quotient`)

The symmetry breaking quotient descriptor is strictly defined as the ratio of the order of the atomic crystallographic site symmetry group to the order of the local point group of the atom's local environment (including its nearest neighbors). This dimensionless indicator aims to quantitatively assess the degree to which an atom's actual symmetry deviates from its ideal crystallographic symmetry, i.e., the reduction in symmetry caused by geometric or chemical asymmetry of the local environment. Its mathematical expression is:
    $$
    \text{Symmetry Breaking Quotient} = \frac{|\text{G}_{\text{site}}|}{|\text{G}_{\text{local}}|}
    $$
where $\text{G}_{\text{site}}$ represents the stabilizer subgroup of the atomic site under crystal periodicity, and its order $|\text{G}_{\text{site}}|$ is the number of space group operations that leave the site invariant; $\text{G}_{\text{local}}$ is the molecular point group formed by the local coordination cluster (including the central atom and all its nearest neighbors) centered at that atom. This quotient value is strictly limited to $[0, 1]$. Ideally, if the local environment can perfectly maintain the crystallographic symmetry of the atomic site, the quotient value is 1; any form of symmetry breaking will result in a quotient value less than 1, and smaller values indicating more drastic symmetry breaking.

From physics, chemistry, and materials science perspectives, the symmetry breaking quotient directly reflects the degree of local symmetry breaking within crystalline materials. This feature is closely related to local strain, electric dipole formation, and macroscopic physical effects (e.g., piezoelectric and ferroelectric effects) induced by lattice distortions. It can quantify coordination polyhedron distortions, bond angle strain, and the tendency of ions to be "off-center" in the lattice, thereby aiding in a deeper understanding of how chemical bonding influences local symmetry. In materials science, it directly affects a material's ferroelectricity, piezoelectricity, dielectric constant, and optical nonlinearity. High symmetry breaking values may indicate that the material possesses large spontaneous polarization or is more easily responsive to external electric fields. For example, in perovskite materials, the symmetry breaking of the B-site cation is one of the key driving forces for inducing ferroelectric phase transitions. In computer science and artificial intelligence disciplines, as a continuous numerical feature, the symmetry breaking quotient provides a precise, group-theoretical quantification of local symmetry deviation, helping machine learning models capture symmetry-related functional properties, thereby enabling the prediction and design of materials with specific symmetry-related performance.

---

## 3.5 Deep Fusion Quantum-Algebraic Descriptors: Intrinsic Coupling of Quantum Information and Geometric Algebra

This section introduces a groundbreaking set of deep fusion descriptors that break through the limitations of simple concatenation or linear combination in traditional feature engineering, achieving intrinsic coupling of quantum chemical information (e.g., electron density, Bader charge, ELF) with geometric algebra (e.g., structure tensor, Lie algebra). These features construct more physically profound and abstractly rigorous descriptors by introducing quantum information as "weights" or "actions" from the very beginning of tensor construction or mathematical operations. This "physics-driven fusion paradigm" reveals higher-order interactions between atomic local structure and electronic behavior, providing unprecedented tools for understanding the deep "genotype" of crystalline materials.

### 3.5.1 Structure-Chemistry Incompatibility (`structure_chemistry_incompatibility`)

The `structure_chemistry_incompatibility` descriptor is strictly defined as the Frobenius norm of the commutator of the primary structural direction of an atom's local environment and the atom's extended chemical vector in the Lie algebra $\mathfrak{so}(3)$, aiming to quantify the "non-commutativity" or "degree of conflict" between local geometric structure and intrinsic atomic chemical properties. The calculation of this feature first involves selecting the eigenvector corresponding to the maximum eigenvalue from the eigenvalue decomposition of the atom's local structural tensor $\mathbf{T}_{\text{struct},i}$ (defined in Section 3.4.2) as the primary extension direction of the local structure, $\mathbf{v}_{\text{struct}}$. Concurrently, a 7-dimensional normalized chemical vector $\mathbf{v}_{\text{chem}}$ is constructed, which includes various atomic intrinsic chemical properties: [electronegativity, covalent_radius, ionization_energy, electron_affinity, atomic_volume, polarizability, effective_charge]. For comparison within the Lie algebra framework, the 3D structural primary direction and the 7-dimensional chemical vector must be strictly mapped to the generators of the Lie algebra $\mathfrak{so}(3)$ of the 3D rotation group $SO(3)$. To precisely achieve this mapping, we first need to deeply understand the mathematical structure of the Lie group $SO(3)$ and its Lie algebra $\mathfrak{so}(3)$ and the properties of its generators.

The three-dimensional rotation group $SO(3)$ is a Lie group composed of all $3 \times 3$ orthogonal matrices with a determinant of 1, where its elements $\mathbf{R}$ satisfy $\mathbf{R}\mathbf{R}^T = \mathbf{I}$ and $\det(\mathbf{R}) = 1$. The Lie algebra $\mathfrak{so}(3)$ is the tangent space of the Lie group $SO(3)$ at the identity element (i.e., the identity matrix $\mathbf{I}$). Elements $\mathbf{A}$ in the Lie algebra are related to Lie group elements $\mathbf{R}(t)$ via the exponential map, i.e., $\mathbf{R}(t) = \exp(t\mathbf{A})$. To clarify the specific form of elements in $\mathfrak{so}(3)$, we consider a smooth path in $SO(3)$ that passes through the identity element.

If $\mathbf{R}(t)$ is a path in $SO(3)$ such that $\mathbf{R}(0) = \mathbf{I}$, then its derivative at $t=0$, $\mathbf{A} = \frac{d\mathbf{R}}{dt}\Big|_{t=0}$, belongs to the Lie algebra $\mathfrak{so}(3)$.
Since $\mathbf{R}(t)\mathbf{R}(t)^T = \mathbf{I}$, differentiating with respect to $t$ yields:
$$
\frac{d}{dt}(\mathbf{R}(t)\mathbf{R}(t)^T) = \frac{d\mathbf{I}}{dt} \\
\frac{d\mathbf{R}}{dt}\mathbf{R}(t)^T + \mathbf{R}(t)\frac{d\mathbf{R}^T}{dt} = \mathbf{0}
$$
At $t=0$, $\mathbf{R}(0) = \mathbf{I}$, and $\frac{d\mathbf{R}}{dt}\Big|_{t=0} = \mathbf{A}$, so the above equation becomes:
$$
\mathbf{A}\mathbf{I}^T + \mathbf{I}\mathbf{A}^T = \mathbf{0} \\
\mathbf{A} + \mathbf{A}^T = \mathbf{0}
$$
This proves that the elements of the Lie algebra $\mathfrak{so}(3)$ are all antisymmetric matrices. In other words, $\mathfrak{so}(3)$ is the set of all $3 \times 3$ antisymmetric real matrices.

For any three-dimensional vector $\mathbf{a} = (a_x, a_y, a_z)^T$, an isomorphic relationship can be established with an antisymmetric matrix via the cross product operation. Specifically, the antisymmetric matrix $\mathbf{M}_{\mathbf{a}}$ corresponding to vector $\mathbf{a}$ is such that for any vector $\mathbf{v}$, $\mathbf{a} \times \mathbf{v} = \mathbf{M}_{\mathbf{a}}\mathbf{v}$. Its strict form is:
        $$
        \mathbf{M}_{\mathbf{a}} = \begin{pmatrix} 0 & -a_z & a_y \\
        a_z & 0 & -a_x \\
        -a_y & a_x & 0 
        \end{pmatrix}
        $$
The Lie bracket operation on the Lie algebra $\mathfrak{so}(3)$ is defined as the commutator of matrices $[\mathbf{A}, \mathbf{B}] = \mathbf{A}\mathbf{B} - \mathbf{B}\mathbf{A}$. If $\mathbf{A}$ and $\mathbf{B}$ are antisymmetric matrices, then their commutator $[\mathbf{A}, \mathbf{B}]$ is still an antisymmetric matrix, thus the Lie algebra is closed under the Lie bracket operation.

Having clarified the correspondence between three-dimensional vectors and antisymmetric matrices in $\mathfrak{so}(3)$ and the properties of Lie algebras, the key is how to effectively map higher-dimensional chemical information (such as the 7-dimensional chemical vector $\mathbf{v}_{\text{chem}}$) into this three-dimensional Lie algebra space.

For the mapping of the 7-dimensional chemical vector $\mathbf{v}_{\text{chem}}$ to a 3-dimensional projected vector $(x, y, z)$, this study employs a rigorous Gram-Schmidt orthogonalization projection method based on physical meaning. This method defines three sets of initial direction vectors with clear physical meanings (respectively focusing on electrochemical properties, spatial geometric properties, and electron response properties), and constructs an orthonormal basis $\left\{ \mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3 \right\}$ for projection through orthogonalization. The final 3-dimensional chemical projection components $(x, y, z)$ are given by the inner product of the 7-dimensional chemical vector $\mathbf{v}_{\text{chem}}$ with these orthonormal bases:
$$
\begin{aligned}
x &= \mathbf{v}_{\text{chem}} \cdot \mathbf{e}_1 \\
y &= \mathbf{v}_{\text{chem}} \cdot \mathbf{e}_2 \\
z &= \mathbf{v}_{\text{chem}} \cdot \mathbf{e}_3
\end{aligned}
$$
This method ensures the physical interpretability and mathematical rigor of the projection. Finally, calculate the Lie algebra commutator $\mathcal{C}(\mathbf{M}_{\text{struct}}, \mathbf{M}_{\text{chem}}) = \mathbf{M}_{\text{struct}} \mathbf{M}_{\text{chem}} - \mathbf{M}_{\text{chem}} \mathbf{M}_{\text{struct}}$ of the structural generator $\mathbf{M}_{\text{struct}}$ (mapped from $\mathbf{v}_{\text{struct}}$) and the chemical generator $\mathbf{M}_{\text{chem}}$ (mapped from the 3D projection of $\mathbf{v}_{\text{chem}}$), and the final descriptor is the Frobenius norm of this commutator matrix:
        $$
        \|\mathcal{C}(\mathbf{M}_{\text{struct}}, \mathbf{M}_{\text{chem}})\|_F = \sqrt{\text{Tr}(\mathcal{C}(\mathbf{M}_{\text{struct}}, \mathbf{M}_{\text{chem}})^T \mathcal{C}(\mathbf{M}_{\text{struct}}, \mathbf{M}_{\text{chem}}))}
        $$
A non-zero Frobenius norm indicates "non-commutativity" or "conflict" between structural and chemical properties under rotational operations, i.e., they cannot be simultaneously aligned by simple rotational transformations. This feature physically quantifies the intrinsic contradiction or driving force between the local geometric structure and the intrinsic chemical nature of atoms in a crystal. High incompatibility values may indicate local strain, structural instability, or spontaneous distortion, which are closely related to phase transitions, polarization effects, and defect formation energies in crystalline materials. It reveals the degree of mismatch between the atomic bonding environment and the atom's own chemical properties such as electron affinity and ionization energy; high incompatibility may cause atoms to deviate from ideal lattice positions to accommodate their chemical bonding preferences, thereby driving structural rearrangement or phase transition. In materials science, it directly affects material structural stability, ferroelectricity, piezoelectricity, and phase behavior. For example, in perovskite materials, the structure-chemistry incompatibility of the B-site cation may be an important source of its ferroelectricity. This feature provides new guiding principles for designing materials with specific structural distortions or functional responses. In computer science and artificial intelligence disciplines, as a continuous numerical feature, it provides an abstract quantification of the deep interaction between high-dimensional chemical information and geometric structure, and by combining Lie algebra and norm theory, it can capture structure-electron coupling information that is difficult to express with traditional descriptors, and its physically driven construction endows models with stronger generalization capabilities and physical insights.

### 3.5.2 Chemical Vector Norm in Structural Metric (`chemical_in_structural_metric`)

The `chemical_in_structural_metric` descriptor is strictly defined as the quadratic form of the atom's 3D chemical projected vector $\mathbf{v}_{\text{chem,proj}}$ under its local environment structure tensor $\mathbf{T}_{\text{struct},i}$ as the metric tensor, aiming to quantify the "extension" or "damping" of atomic chemical properties in the local structural space. Its mathematical expression is:
    $$
    \text{Norm}_{\mathbf{T}_{\text{struct},i}}(\mathbf{v}_{\text{chem,proj}}) = \mathbf{v}_{\text{chem,proj}}^T \mathbf{T}_{\text{struct},i} \mathbf{v}_{\text{chem,proj}}
    $$
Here, $\mathbf{v}_{\text{chem,proj}}$ is the 3D projected vector obtained from the 7-dimensional chemical vector through the Gram-Schmidt projection method described in Section 3.5.1, and $\mathbf{T}_{\text{struct},i}$ is the structure tensor of the atom's local environment (defined in Section 3.4.2). This quadratic form provides an intuitive geometric explanation: the structure tensor defines "distance" or "shape" in local space, while this feature measures the "length" or "energy" of the chemical vector in this "distorted" space defined by atomic arrangement. This calculation includes regularization of the structure tensor (by adding a small identity matrix term) to ensure its positive definiteness, thereby enhancing numerical stability so it can reliably serve as a metric tensor. Physically, this feature reflects the "coupling strength" between the atom's chemical nature and its local geometric structure. High values may imply that the atom's chemical preferences highly match the structural geometry, or that chemical properties are "amplified" in the structural space, leading to stronger interactions. It reveals how the atom's chemical potential field interacts with the local atomic arrangement. For example, atoms with specific chemical properties may tend to occupy sites with certain geometric configurations, maximizing the norm of their chemical vector under this structural metric, thereby achieving energetic stability. In materials science, this feature directly influences material stability, solid solubility, atomic diffusion pathways, and phase behavior. High coupling strength may imply that the atom has higher structural adaptability at that site, it provides new guidance for optimizing material composition and structure and predicting alloying behavior. In computer science and artificial intelligence disciplines, as a continuous numerical feature, it provides a "metric" of high-dimensional chemical information in structural space, realizing feature construction with deep coupling of multiple physical quantities. This feature, by introducing the structure tensor as a metric, enables machine learning models to better understand the manifestation of chemical information in specific geometric environments, being key to building multi-physical field coupled models.

### 3.5.3 Charge-Weighted Local Size (`charge_weighted_local_size`)

The `charge_weighted_local_size` descriptor is strictly defined as the trace of the charge-weighted local structure tensor $\mathbf{T}_{\text{charge\_weighted},i}$. This tensor is constructed similarly to the conventional structure tensor, but the absolute value of the Bader charge of the nearest-neighbor atom (from Section 3.3.1) is introduced as a weight when calculating the outer product of each nearest-neighbor displacement vector:
    $$
    \mathbf{T}_{\text{charge\_weighted},i} = \sum_{j \in \text{NN}(i)} |q_j| (\mathbf{r}_j - \mathbf{r}_i) \otimes (\mathbf{r}_j - \mathbf{r}_i)
    $$
The final `charge_weighted_local_size` is the trace of this weighted tensor:
    $$
    \text{Charge Weighted Local Size} = \text{Tr}(\mathbf{T}_{\text{charge\_weighted},i}) = \sum_{j \in \text{NN}(i)} |q_j| \|\mathbf{r}_j - \mathbf{r}_i\|_2^2
    $$
This "deep fusion" paradigm means that charge is no longer an afterthought modulator, but a weight that participates from the very beginning of tensor construction, representing a "charge interaction-aware" measure of local size. Physically, this feature reflects the "effective" spatial size of the atom's local environment, where nearest neighbors with larger charges contribute more to the size. This is closely related to the non-uniformity of local charge distribution, ionic bond strength, and electrostatic interaction range, enabling a more accurate reflection of the influence of charge on structural space. It reveals the impact of charge transfer and ionicity on local geometric structure. In crystals dominated by ionic bonds, this feature will highlight the contribution of ions with high net charges to the local size, thereby more accurately capturing charge-geometry interactions. In materials science, it directly affects ion mobility (in solid electrolytes, charge-weighted size may be related to the "effective" size of ion diffusion channels), dielectric response, and defect formation energy. For example, a larger charge-weighted size may indicate strong Coulomb repulsion between ions, thereby affecting lattice stability. In computer science and artificial intelligence disciplines, as a continuous numerical feature, it through intrinsically fusing quantum information (charge) into geometric tensors, achieves multi-physical quantity deep coupled feature construction. This enables models to learn the fine influence of charge distribution on local structural size, improving the physical interpretability and predictive power of features.

### 3.5.4 ELF-Weighted Local Asymmetry (`elf_weighted_local_anisotropy`)

The `elf_weighted_local_anisotropy` descriptor is strictly defined as the product of the Electron Localization Function (ELF) value at the atomic center (from Section 3.3.4) and its vectorial asymmetry norm squared (from Section 3.4.1). It directly links the localization characteristics of the electronic structure with the geometric asymmetry of the atomic environment. Its mathematical expression is:
    $$
    \text{ELF Weighted Local Asymmetry} = \text{ELF}_i \times \|\mathbf{V}_{\text{asymmetry},i}\|^2
    $$
where $\text{ELF}_i$ is the ELF value at atom $i$, and $\|\mathbf{V}_{\text{asymmetry},i}\|^2$ is the vectorial asymmetry norm squared of atom $i$. The physical picture of this feature is: in regions of high electron localization (high ELF values), the "weight" of geometric asymmetry (measured by the vectorial asymmetry norm squared) is greater. This means that geometric asymmetry, combined with electron localization information, may have a more significant impact on the physicochemical properties and functionality of materials. Physically, this feature directly quantifies the strengthening effect of electron localization on local structural asymmetry. High values indicate significant geometric asymmetry in regions of high electron localization, which may be closely related to the formation of local electric dipoles, band structure changes caused by non-centrosymmetry, and optoelectronic response. It reveals the synergistic effect between electron bonding patterns (covalent character, lone pairs) and local geometric distortions. For example, in strongly covalent bonding environments, even small geometric asymmetries can produce significant local polarization. In materials science, this feature directly affects the piezoelectricity, ferroelectricity, nonlinear optical effects, and photovoltaic performance of crystalline materials. High ELF-weighted asymmetry may indicate that the material possesses excellent polarity-related functions, it provides a new perspective for understanding and designing functional materials with specific optoelectronic and piezoelectric properties. In computer science and artificial intelligence disciplines, as a continuous numerical feature, it through deeply fusing electronic structure information (ELF) with geometric information (vectorial asymmetry norm squared), achieves physics-driven feature engineering. This enables machine learning models to more effectively capture the complex interaction between electron localization and the functional impact of geometric asymmetry, thereby improving the prediction accuracy and interpretability of models for material functional properties (e.g., polarity, optical response).
