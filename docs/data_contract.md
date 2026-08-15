# External data contract

This code release deliberately excludes large or licensed artifacts. An
independent full rerun needs the following externally supplied items.

| Item | Required for | Expected interface |
|---|---|---|
| Compatible MACE checkpoint | relaxation, forces, phonons | pass with `--model-path`; SHA-256 should be recorded in each run |
| DFT training labels and split files | retraining | EXTXYZ with `REF_energy`, `REF_forces`, and configuration types |
| VASP executable and PAW datasets | DFT registry/IFC validation | locally licensed installation; never commit `POTCAR` |
| Completed displacement forces | analysis without rerunning forces | ordered arrays matching each generated displacement manifest |
| Force constants and compact case exports | analytical-only rerun | `phonopy_structure.yaml`, `force_constants.hdf5`, and `case.json` |

The repository does not supply calculated values as substitutes for these
inputs. Data may be obtained from the authors upon reasonable request, subject
to VASP, model, institutional, and storage constraints.

Every imported external artifact should be checked against its recorded SHA-256
before analysis. Do not mix force constants from different structures, models,
displacement manifests, Phonopy conventions, or atom orderings.
