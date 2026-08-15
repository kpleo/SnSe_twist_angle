# Reproduction workflow

All commands below are run from the repository root. Put generated content
under `work/` and publication-ready derived content under `outputs/`.

## 1. Build and audit the seven commensurate cells

```bash
python scripts/structures/prepare_corrected_2d_twist_series.py \
  --output-dir work \
  --layer-separation 5.9127456535574385
python scripts/structures/audit_corrected_2d_twist_series.py --root work
```

The generator applies equal and opposite logarithmic area accommodation,
area-preserving aspect matching, centered Sn anchors, and a 25 A periodic empty
gap. The seven `(p,q)` pairs are defined in `configs/angle_series.json`.

## 2. Fixed-cell relaxation

Supply either the compatible MACE checkpoint used in the study or a model
retrained with `configs/mace_finetune.yaml`:

```bash
python scripts/phonons/strict_relax_mace.py \
  --input work/structures/initial/8p77deg/initial.extxyz \
  --out-dir work/relaxations/8p77deg/confirmed_fmax0p001 \
  --fmax 0.001 \
  --model-path /path/to/model.model

python scripts/structures/audit_corrected_2d_relaxation.py \
  --initial work/structures/initial/8p77deg/initial.extxyz \
  --relaxed work/relaxations/8p77deg/confirmed_fmax0p001/relaxed.extxyz \
  --relax-stats work/relaxations/8p77deg/confirmed_fmax0p001/relax_stats.json \
  --output work/relaxations/8p77deg/confirmed_fmax0p001/relaxation_audit.json
```

Repeat for each angle. Construct both matched periodic controls after the
twisted-cell relaxations pass their gates:

```bash
python scripts/structures/prepare_corrected_2d_matched_controls.py --root work
python scripts/structures/audit_corrected_2d_matched_controls.py --root work
```

### Optional model retraining

If the compatible checkpoint is unavailable, extract the registry trajectories
beside the completed VASP calculations with
`scripts/model/extract_snse_registry_training_data.py`, place the compact phase
files under `work/training_data/`, and run:

```bash
python scripts/model/prepare_snse_registry_finetuning_dataset.py --root work
mace_run_train --config configs/mace_finetune.yaml
```

The registry-disjoint train/validation/test split is fixed in the dataset script.
Evaluate predictions with `evaluate_snse_registry_finetuned_model.py` before any
large-cell production run.

## 3. Centered finite-displacement phonons

For every twisted structure and both matched controls, run a 1 x 1 x 1
finite-displacement calculation with +/-0.010 A centered displacements:

```bash
python scripts/phonons/mlip_full_phonon.py \
  --model mace-checkpoint \
  --structure /path/to/relaxed.extxyz \
  --out-dir work/phonons/8p77deg/fmax0p001_disp0p01_plusminus \
  --model-path /path/to/model.model \
  --displacement 0.01 \
  --plusminus \
  --disable-symmetry \
  --supercell 1,1,1 \
  --band-points 18 \
  --device cpu
```

The calculation requires `6N` force evaluations for an `N`-atom cell. The
script is restartable because each displacement force is written independently.
These arrays are calculation outputs and must not be committed.

## 4. Export compact Phonopy cases

```bash
python scripts/phonons/export_corrected_2d_review_phonons.py \
  --root work \
  --output-dir work/review_cases
```

This step reconstructs centered force constants, records the raw translational
row-sum residual, applies one Phonopy translational/permutation symmetrization,
and writes the compact case metadata required by the analytical runner.

## 5. Analytical group velocities and stability

Use the dedicated Phonopy 2.43.4 environment for all reported velocity values:

```bash
python scripts/phonons/run_corrected_2d_review_phonon_case.py \
  --case-dir work/review_cases/8p77deg/twist \
  --output work/review_json/8p77deg_twist.json \
  --band-points 9 \
  --temperatures 100,300,500,700 \
  --cutoffs 0.02,0.05,0.10 \
  --q-grid 4
```

For each control use `--q-grid 0`. Repeat all 21 cases. The method is the
analytical derivative of the dynamical matrix projected along each path segment;
finite-difference absolute velocities are not part of the final workflow.

Run the ASR sensitivity triplet from raw force constants and the path-density
triplets with `--band-points 18` where required.

## 6. Aggregate the seven-angle evidence

```bash
python scripts/analysis/aggregate_corrected_2d_review_phonons.py \
  --input-dir work/review_json \
  --raw-asr-dir work/review_json/asr_raw \
  --path18-dir work/review_json/path18 \
  --output-dir outputs/analysis/review_analytical_velocity
```

Optional `--production-job-id`, `--asr-job-id`, and `--path18-job-id` arguments
can enforce exact scheduler provenance for an HPC rerun. When omitted, the
scientific case, method, task ordering, hashes, path count, ASR, finiteness, and
q-grid gates are still enforced.

## 7. DFT and MACE validation

The nine registry POSCAR inputs and VASP settings are in `inputs/registry/` and
`configs/dft_registry_relaxation/`. `POTCAR` is deliberately absent. After the
DFT relaxation outputs are available:

```bash
python scripts/dft_validation/prepare_registry_vasp_inputs.py
```

Add a locally licensed `POTCAR` to each generated case, run VASP, and retain the
completed `CONTCAR`. For the direct IFC benchmark, place the selected strict
results at
`work/dft_registry_phase_b_results/{registry_ix00_iy00_n3x3,registry_ix00_iy02_n3x3,registry_ix01_iy01_n3x3}/CONTCAR`.
Then generate the centered-displacement benchmark:

```bash
python scripts/dft_validation/prepare_snse_mlip_dft_ifc_benchmark.py \
  --registry-root work/dft_registry_phase_b_results \
  --output work/dft_mlip_ifc_benchmark_v3 \
  --ediff 1e-8
```

Run VASP for the generated tasks, parse each completed task with
`parse_snse_mlip_dft_ifc_task.py`, evaluate the same structures with
`run_snse_mlip_dft_ifc_mace.py`, and aggregate with
`analyze_snse_mlip_dft_ifc_benchmark.py --protocol-version v3`, supplying the
new run's smoke and production job IDs so the parser can enforce exact scheduler
provenance.

## 8. 7.61-degree diagnostic

Use `run_corrected_2d_7p61_tight_relax_probe.py` to close the residual-force
gate and `run_corrected_2d_frozen_soft_mode.py` for centered mode amplitudes.
The same full-phonon and analytical-velocity scripts used above apply to the
tight endpoint. This branch is diagnostic only and must remain separate from
the six-structure equilibrium trend.

## 9. Figures

After the aggregate and diagnostic tables have been regenerated, the scripts in
`scripts/figures/` write figures to `outputs/figures/` and their derived source
tables to `outputs/source_data/`. Generated images and tables are intentionally
not versioned in this repository.
