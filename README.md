# -Molecular-Dynamics-Project-on-Li-Solvation
Radial Distribution Function (RDF) plot → This shows how water oxygen atoms are distributed around the Li⁺ ion (the solvation structure). Peaks in the RDF correspond to solvation shells.  Mean Square Displacement (MSD) plot → This shows the motion of Li⁺ over time, from which you can estimate diffusivity.
# Install dependencies
!pip install ase mdanalysis matplotlib tidynamics

from ase.build import molecule
from ase import Atoms
from ase.md.langevin import Langevin
from ase import units
from ase.io import Trajectory, write
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from MDAnalysis.analysis.msd import EinsteinMSD


# 1. Build system: Li+ in water

water = molecule("H2O")
waters = [water.copy() for _ in range(10)]
cell = np.eye(3) * 15.0  # box size

system = Atoms(cell=cell, pbc=True)
for w in waters:
    w.positions += np.random.rand(*w.positions.shape) * 15.0
    system += w

# Add Li+
system += Atoms("Li", positions=[[7.5, 7.5, 7.5]])


# 2. Use a toy calculator (Harmonic potential) since EMT/LAMMPS not working in Colab
from ase.calculators.lj import LennardJones
system.calc = LennardJones()  # simple potential for demo

# 3. Run short MD
dyn = Langevin(system, 1 * units.fs, temperature_K=300, friction=0.002)
traj = Trajectory("li_solvation.traj", 'w', system)

for step in range(200):  # short run
    dyn.run(1)
    traj.write()

traj.close() # Close the trajectory file before converting

# Convert to XYZ format for MDAnalysis
write("li_solvation.xyz", Trajectory("li_solvation.traj"), format="xyz")

print(" MD finished. Trajectory saved in .traj and .xyz formats.")

# 4. Analysis with MDAnalysis
# Load the XYZ trajectory with MDAnalysis
u = mda.Universe("li_solvation.xyz")

# Explicitly set the box dimensions after creating the Universe
u.dimensions = system.get_cell().cellpar()


Li = u.select_atoms("name Li")
O = u.select_atoms("name O")

# RDF
rdf = InterRDF(Li, O, range=(0.0, 6.0))
rdf.run()
plt.plot(rdf.bins, rdf.rdf)
plt.xlabel("Distance (Å)")
plt.ylabel("g(r)")
plt.title("Li-O Radial Distribution Function")
plt.show()

# MSD
msd = EinsteinMSD(u, select="name Li", msd_type='xyz', fft=True)
msd.run()
print(msd.results) # Print the results object to inspect its contents
plt.plot(msd.times, msd.results['timeseries'])
plt.xlabel("Time (ps)")
plt.ylabel("MSD (Å^2)")
plt.title("Li+ Mean Square Displacement")
plt.show()
