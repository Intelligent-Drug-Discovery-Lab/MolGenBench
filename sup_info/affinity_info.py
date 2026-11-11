import os
import pickle
import numpy as np
from collections import defaultdict
from rdkit import Chem


def get_pAffinity(x):
    '''
    transform affinity value to pAffinity value
    '''
    if not isinstance(x, (int, float)):
        raise TypeError("Input must be an integer or float.")
    x = x * 1e-9
    x = -np.log10(x)
    return x

def normalize_affinity_map(info_map):
    """
    Normalize the affinity values in the info_map to a range between 0 and 1
    Using max normaliztion
    Args:
        info_map (_type_): A dictionary with affinity values to be normalized
        {smiles: affinity_value, ...}
    Returns:
        dict: A dictionary with normalized affinity values
    """
    # Sort the dictornary by values in ascending order
    sorted_dict = {
        k: get_pAffinity(float(v)) 
        for k, v in sorted(info_map.items(), key=lambda item: float(item[1]), reverse=False) 
        if float(v) >= 0.0
    }
    # Normalize the values to range [0, 1]
    max_value = max(sorted_dict.values())
    min_value = min(sorted_dict.values())
    normalized_dict = {k: (v - min_value) / (max_value - min_value) for k, v in sorted_dict.items()}
    
    return normalized_dict

def normalize_affinty_map_discount(info_map):
    """
    Normalize affinity values with a discounted cumulative gain (DCG)-style positional discount
    for NADR calculation.
    
    Args:
        info_map (dict): A dictionary mapping SMILES strings to raw affinity values (in nM).
            {smiles: affinity_value, ...}
    
    Returns:
        dict: A dictionary mapping SMILES strings to discounted normalized scores.
            Score = (pAffinity / max_pAffinity) / log2(rank + 1)
            where rank is the ascending affinity order.
    """
    # Sort the dictionary by values in ascending order
    sorted_dict = {
        k: get_pAffinity(float(v)) 
        for k, v in sorted(info_map.items(), key=lambda item: float(item[1]), reverse=False) 
        if float(v) >= 0.0
    }

    # Normalize the values
    min_value = min(sorted_dict.values())
    max_value = max(sorted_dict.values())
    normalized_dict = {k: ((v - min_value) / (max_value - min_value) / np.log2(idx + 1 + 1)) for idx, (k, v) in enumerate(sorted_dict.items())}

    return normalized_dict

# # Generate affinity info map and save as pickle file
# affinity_info_map = defaultdict(dict)
# data_path = "MolGenBench/MolGenBench_Version2"

# uniprot_series_list = []
# for uniprot in os.listdir(data_path):
#     uniprot_path = os.path.join(data_path, uniprot, "Round1", "Hit_to_Lead_Results")
#     series_list = os.listdir(uniprot_path)
#     for series in series_list:
#         uniprot_series_list.append(f'{uniprot}_{series}')

# for uniprot in os.listdir(data_path):
#     ref_active_path = os.path.join(data_path, uniprot, "reference_active_molecules", f"{uniprot}_reference_active_molecules.sdf")
#     suppl = Chem.SDMolSupplier(ref_active_path)
#     for ref_mol in suppl:
#         if ref_mol is None:
#             continue
#         uniprot_series = '_'.join(ref_mol.GetProp("_Name").split("_")[:2])
#         if uniprot_series in uniprot_series_list:
#             affinity = ref_mol.GetProp("Affinity")
#             affinity_info_map[uniprot_series][Chem.MolToSmiles(ref_mol)] = affinity

# data_path = "sup_info/affinity_info_map.pkl"
# with open(data_path, "wb") as f:
#     pickle.dump(affinity_info_map, f)

# Load affinity info map
### This file only contain affinity data in special series ###

            
data_path = "sup_info/affinity_info_map.pkl"

with open(data_path, "rb") as f:
    affinity_info_map = pickle.load(f)
# Normalize affinity info map
normalized_affinity_info_map = {}
for uniprot_series, info_map in affinity_info_map.items():
    normalized_map = normalize_affinity_map(info_map)
    normalized_affinity_info_map[uniprot_series] = normalized_map

normalized_affinity_info_map_discount = {}
for uniprot_series, info_map in affinity_info_map.items():
    normalized_map = normalize_affinty_map_discount(info_map)
    normalized_affinity_info_map_discount[uniprot_series] = normalized_map