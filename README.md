
# MolGenBench: CodeBase for "Benchmarking Real-World Applicability of Molecular Generative Models from De novo Design to Lead Optimization with MolGenBench"
![MolGenBench overview](./FigShow/MolGenBench.svg "Overview of MolGenBench pipeline")

## 🔔 News
[2026-01-09] We have released Version 4 of the [dataset](https://zenodo.org/records/20233505). In this release, we fixed the raw CSV results of MotifDist for the H2L model. The aggregated YAML results under the `paper_results` folder were already correct as of commit [`afae9326`](https://github.com/Intelligent-Drug-Discovery-Lab/MolGenBench/tree/afae93266cd4b7e7f63878adfa7612b746b8cac9).

[2026-01-09] We have released Version 3 of the [dataset](https://zenodo.org/records/18183463). In this release, we added pre-computed evaluation CSV results to each results folder. We also uploaded the aggregated results under the `paper_results` folder in the repository.

[2025-12-12] We have released Version 2 of the [dataset](https://zenodo.org/records/17890389). Compared with Version 1, this update additionally includes precomputed InChI and SMILES for active molecules, which significantly accelerates the HitRediscovery calculation.

# 🛠️ Environment Setup
```bash
conda create --name MolGenBench python=3.11
mamba install -c conda-forge numpy pandas seaborn scipy -y
pip install --use-pep517 EFGs
pip install rdkit==2025.9.1 prolif==2.0.3 mdanalysis==2.7.0
pip install posebusters==0.3.1 spyrmsd
pip install tqdm joblib pytest swifter medchem
mamba install -c conda-forge lilly-medchem-rules
mamba install openbabel

# for vina docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# for posecheck evaluation
git clone https://github.com/cch1999/posecheck.git
cd posecheck
git checkout 57a1938  # the calculation of strain energy used in our paper
pip install -e .
pip install -r requirements.txt
conda install -c mx reduce
```

# 📦 Datasets & Benchmark Results
Please download from [Zenodo dataset](https://zenodo.org/records/20233505) the result on your device and unzip the files. The downloaded dataset already follows the required folder structure, so you can directly use it for evaluation without any reorganization.

# 📁 Required Directory Structure

> [!NOTE]
> If you generate molecules with your own model, please ensure that your output is saved following the same directory structure as the official dataset.
> Below is the expected structure for each UniProt target:

```
P12345/     # Uniprot ID
├─ reference_active_molecules/
├─ Round1/
│  ├─ De_novo_Results/
│  │  └─ <YOUR_MODEL_NAME>/
│  │     ├─ results/    # Evaluation results will be saved here
│  │     ├─ P12345_<YOUR_MODEL_NAME>.sdf
│  │     └─ P12345_<YOUR_MODEL_NAME>_vina_docked.sdf     # Docked pose generated using vina_docking.py
│  │
│  └─ Hit_to_Lead_Results/
│     └─ Sries001/      # Series ID
│        └─ <YOUR_MODEL_NAME>/
│           ├─ results/ 
│           ├─ P12345_Sries001_<YOUR_MODEL_NAME>.sdf
│           └─ P12345_Sries001_<YOUR_MODEL_NAME>_vina_docked.sdf
│
├─ P12345_prep.pdb
├─ P12345_pocket10.pdb
└─ P12345_lig.sdf
```

# 🧬 Run Generation with Your Model

This section explains how to use the MolGenBench data structure to run *your own
molecule generation model*. After molecules are generated, they can be evaluated
with the unified MolGenBench evaluation pipeline.

MolGenBench supports two generation scenarios:

1. **De novo design**  
2. **Hit-to-lead optimization**

## De novo design
**Input requirement:**

- You may use either the full protein (`*_prep.pdb`) or the pocket file (`*_pocket10.pdb`) as model input; the ligand file (`*_lig.sdf`) is used to define the pocket position.

**Generation rules:**

- Set the generation  to **1000 samples per UniProt ID**.
- Do **NOT** perform any filtering or post-processing beyond what your model naturally produces.  
  → If your model produces fewer than 1000 valid molecules, keep all valid molecules it can generate.
- Save the generated SDF file as:
Generated molecules should be saved as `<UniprotID>_<YOUR_MODEL_NAME>.sdf`; each molecule must have its `_Name` field set to a unique index (0, 1, 2, …) to distinguish individual samples.

## Hit-to-lead optimization

Hit-to-lead optimization is performed **per chemical series**.  

**Generation rules:**

- Set the generation  to **200 samples per Series ID**.
- Do **NOT** perform any filtering or post-processing beyond what your model naturally produces.  
  → If your model produces fewer than 200 valid molecules, keep all valid molecules it can generate.
- Save the generated SDF file as:
Generated molecules should be saved as `<UniprotID>_<SeriesID>_<YOUR_MODEL_NAME>.sdf`; each molecule must have its `_Name` field set to a unique index (0, 1, 2, …) to distinguish individual samples.

---
Depending on your model type, use different inputs:

### **1. Structure-based generation models**

**Input requirement:**

- Protein pocket: `<UniprotID>_pocket10.pdb` or `<UniprotID>_prep.pdb`
- Reference ligand pose for the specific series:
  `<UniprotID>_<SeriesID>_reference_ligand_pose_with_h.sdf`
- **Conserved scaffold extraction for model input**:
The conserved core scaffold is extracted using the corresponding scaffold SMARTS defined in top5_common_scaffold_info.csv for each series and provided as the input to the model. We provide a concrete example in get_h2l_scaf.py.

### **2. Ligand-based generation models**

**Input requirement:**

Use ONLY:

- `<UniprotID>_<SeriesID>_reference_ligand_pose_with_h.sdf`


# ⚙️ Running the Evaluation

After generating molecules with your model, you can evaluate them using the unified
evaluation pipeline provided in `eval.py`.

```bash
python eval.py \
    --data_path "/path/to/data" \
    --round_name "Round1" \
    --mode "De_novo_Results" \  # or "Hit_to_Lead_Results"
    --model_name "YOUR_MODEL_NAME" \
    --fixStereoFrom3D # close this if your model direct output smiles
```

You can **comment out metrics in `eval.py` that you do not wish to compute.**
For example:

```python
evaluator = Evaluator(
    [
        "Validity", 
        "QED",
        "SA",
        "Uniqueness",
        "Diversity",
        "MotifDist",
        "ChemFilter",
        "HitRediscover",

        # ----below are 3D metrics----
        # "PoseBuster", # comment out if 3D checks are not needed
        # "StrainEnergy",
        # "RMSD",
        # "InteractionScore",
        # "ClashScore",
        
      ]
)
```

> [!TIP]
> RMSD metrics require the `*_vina_docked.sdf` files.  
> Please run `vina_docking.py` to generate the docked poses before evaluation.
> You can generate them by running:
> ```bash
> python vina_docking.py \
>    --data_path "/path/to/data" \
>    --output_path "/path/to/output" \
>    --round_name "Round1" \
>    --mode "De_novo_Results" \ or "Hit_to_Lead_Results"
>    --model_name "YOUR_MODEL_NAME"
> ```

# 📊 Visualizing Final Results (Notebook)
``` For Example:
   denovo hit rate (please replace the name of dir)
   relative_dir/FigShow/Denovo_hit_recovery/Deonovo_repeats_hit_rate_boxplot.ipynb

    denovo hit fraction
   relative_dir/FigShow/Denovo_hit_recovery/Deonovo_repeats_hit_fraction_boxplot.ipynb
``` 

