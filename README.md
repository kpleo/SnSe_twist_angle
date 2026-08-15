# Reproducibility code for twisted bilayer SnSe

This repository contains the scripts, compact structural inputs, and parameter
files needed to reproduce the computational workflow associated with
**Broadband phonon-velocity suppression and a finite anisotropic crossover in
twisted bilayer SnSe**.

The release is intentionally code-first. It does **not** contain calculated
trajectories, raw VASP outputs, trained model weights, force arrays, force
constants, generated figures, source-data tables, reviewer material, or the
manuscript source. Those files are excluded to keep the repository focused and
lightweight. The calculation data needed for an independent rerun are available
from the authors upon reasonable request, subject to software and data-license
restrictions.

## What is included

- construction and independent audit of seven commensurate bilayer cells;
- fixed-cell MACE relaxation and matched untwisted-control construction;
- centered finite-displacement harmonic phonons;
- analytical Phonopy dynamical-matrix-derivative group velocities;
- 4 x 4 x 1 full-zone stability, ASR, and path-density checks;
- DFT registry and MACE-vs-DFT interatomic-force-constant validation scripts;
- the restrained 7.61-degree relaxation-sensitivity diagnostics;
- scripts that regenerate the reported plots from recomputed result tables;
- compact primitive and registry input structures, DFT settings, and hashes.

## Repository layout

```text
configs/       Published angle series, DFT settings, and MACE fit parameters
docs/          Workflow, data contract, and input-provenance notes
environment/   Pinned Python environments for production and analysis
inputs/        Small structural inputs only
scripts/       Structure, phonon, validation, analysis, diagnostic, and plot code
tools/         Release-integrity checks
```

Generated files belong under `work/` and `outputs/`; both are ignored by Git.

## Quick start

Create the structure-generation environment and build all seven cells:

```bash
python3 -m venv .venv-phonons
source .venv-phonons/bin/activate
python -m pip install -r environment/phonon-production.txt

python scripts/structures/prepare_corrected_2d_twist_series.py \
  --output-dir work \
  --layer-separation 5.9127456535574385
python scripts/structures/audit_corrected_2d_twist_series.py --root work
```

The full calculation sequence, file contracts, and representative commands are
in [docs/workflow.md](docs/workflow.md). Input origins and checksums are listed
in [docs/input_provenance.md](docs/input_provenance.md).

## Central numerical protocol

- angles: 8.771750, 7.611379, 6.017285, 4.780192, 3.822554,
  3.583322, and 3.184739 degrees;
- commensurate family:
  `M_lower=[[p,0],[0,q]]`, `M_upper=[[p,1],[-1,q]]`;
- periodic empty gap: 25 A;
- centered displacements: +/-0.010 A, no symmetry reduction, 1 x 1 x 1;
- analytical path: nine midpoint samples per segment on Gamma-X-S-Y-Gamma;
- primary thermal metric: 300 K with a 0.05 THz positive-frequency cutoff;
- full-zone check: uniform 4 x 4 x 1 q grid for every twisted structure;
- displayed band path: 18 points per segment.

The 7.61-degree configuration is treated only as a relaxation-sensitive special
case. It is not used to claim an intrinsic instability or a magic angle.

## External software and licensed inputs

VASP is not distributed here. `POTCAR` files are never included and must be
obtained through a valid VASP license. MACE model checkpoints are also excluded;
the expected checkpoint or a compatible retrained model must be supplied with
`--model-path`. See [docs/data_contract.md](docs/data_contract.md).

## Integrity check

Run before publishing or archiving the repository:

```bash
python tools/audit_release.py
```

The audit rejects raw calculation outputs, manuscript files, generated figures,
model checkpoints, large binary arrays, personal absolute paths, and files over
the release size limit.
