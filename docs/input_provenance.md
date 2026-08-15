# Input provenance

## Primitive layers

The independently oriented four-atom monolayer primitives are stored as:

- `inputs/source_layers/POSCAR_LOWER_PRIMITIVE`
  - SHA-256: `83d992a964f62b1b9b736978767b10c6d7e2819143c68ca2510f4bd759f01cc0`
- `inputs/source_layers/POSCAR_UPPER_PRIMITIVE`
  - SHA-256: `7026ff7bf47e131d840bd9cc0dd885e7a382af7f11f9f8c7df56c705f15e5921`

Both cells contain `Sn2Se2`. The upper motif is mapped to the lower convention
by a Cartesian y reflection before the commensurate-cell construction.

## Registry inputs

`inputs/registry/registry_ix??_iy??_n3x3/POSCAR` contains the nine fractional
translations `(ix/3, iy/3)`. The compact CSV files preserve primitive basis,
atom mapping, registry translation, geometry gates, and input hashes. They are
input/provenance metadata, not relaxed DFT results.

## DFT settings

`configs/dft_registry_relaxation/INCAR` and `KPOINTS` record the strict
fixed-cell registry protocol: PBE-D3(BJ), 600 eV cutoff, `EDIFF=1E-8`,
`EDIFFG=-0.001`, no symmetry, and a Gamma-centered 16 x 16 x 1 mesh.

`POTCAR` is excluded because it is licensed VASP content.

## Published geometry parameter

The default initial layer-center separation is
`5.9127456535574385 A`, the mean of the nine relaxed registry geometries used
for construction. It is encoded directly in `configs/angle_series.json`, so no
calculated registry-result table is required to generate the seven initial
twisted structures.
