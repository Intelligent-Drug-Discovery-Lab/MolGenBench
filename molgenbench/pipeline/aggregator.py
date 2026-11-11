import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from sup_info.utils import uniprot_in_trainset, ref_smiles_scaffold_unique_count_denovo, ref_smiles_scaffold_unique_count_h2l
from sup_info.affinity_info import normalized_affinity_info_map, normalized_affinity_info_map_discount


class Aggregator:
    """
    Aggregates evaluation metrics from CSV files and outputs results as YAML.
    First collects all CSVs into DataFrames, then computes aggregated statistics.
    """

    def __init__(self, metric_names: List[str] = None):
        self.metric_names = metric_names or [
            "Validity", "QED", "SA", "Uniqueness", "Diversity",
            "MotifDist", "ChemFilter",
            "PoseBuster", "StrainEnergy", "RMSD", "InteractionScore", "ClashScore",
            "HitRediscover","StrainEnergy_docked", "InteractionScore_docked", "ClashScore_docked"
        ]
        
        # Metric -> aggregation function mapping
        self.aggregation_map = {
            "Validity": self._aggregate_validity,
            "QED": self._aggregate_qed,
            "SA": self._aggregate_sa,
            "Uniqueness": self._aggregate_uniqueness,
            "Diversity": self._aggregate_diversity,
            
            "MotifDist": self._aggregate_motif_dist,
            "ChemFilter": self._aggregate_chemfilter,
            "PoseBuster": self._aggregate_posebuster,
            "StrainEnergy": self._aggregate_strain_energy,
            "StrainEnergy_docked": self._aggregate_strain_energy,
            "RMSD": self._aggregate_rmsd,
            "InteractionScore": self._aggregate_interaction_score,
            "InteractionScore_docked": self._aggregate_interaction_score,
            "ClashScore": self._aggregate_clash_score,
            "ClashScore_docked": self._aggregate_clash_score,
            "HitRediscover": self._aggregate_hit_rediscover,
        }
        
        self.mode = None
        self.output_dir = None
        # Store reference percentiles for InteractionScore
        self.ref_interaction_percentiles = None
        
        self.ref_smiles = None
        self.ref_scaffold = None

    # ========================== #
    #   CSV Collection Methods
    # ========================== #

    def collect_all_csvs(
        self,
        root_dir: str,
        model_name: str,
        round: str,
        mode: str = "De_novo_Results",
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all CSV files for each metric and concatenate them into DataFrames.
        
        Returns:
            Dict mapping metric_name -> concatenated DataFrame
        """
        metric_dfs = {metric: [] for metric in self.metric_names}
        
        for uniprot in os.listdir(root_dir):
            uniprot_path = os.path.join(root_dir, uniprot, round, mode)
            
            if mode == "De_novo_Results":
                results_dir = os.path.join(uniprot_path, model_name, "results")
                self._collect_from_results_dir(results_dir, metric_dfs)
            
            elif mode == "Hit_to_Lead_Results":
                for series_id in os.listdir(uniprot_path):
                    series_path = os.path.join(uniprot_path, series_id)
                    results_dir = os.path.join(series_path, model_name, "results")
                    self._collect_from_results_dir(results_dir, metric_dfs)
        
        # Concatenate all collected DataFrames
        concatenated = {}
        for metric_name, df_list in metric_dfs.items():
            if df_list:
                concatenated[metric_name] = pd.concat(df_list, ignore_index=True)
        
        return concatenated

    def _collect_from_results_dir(
        self,
        results_dir: str,
        metric_dfs: Dict[str, List[pd.DataFrame]],
    ):
        """Collect CSVs from a single results directory"""
        if not os.path.isdir(results_dir):
            return
        
        for metric_name in self.metric_names:
            csv_path = os.path.join(results_dir, f"{metric_name}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                metric_dfs[metric_name].append(df)

    # ========================== #
    #   Aggregation Methods
    # ========================== #
    def _aggregate_validity(self, df: pd.DataFrame, metric_name: str = "Validity") -> Dict[str, Any]:
        values_all = df[metric_name].sum() / 120000
        values_seen = df[df['uniprot'].isin(uniprot_in_trainset)][metric_name].sum() / 85000
        values_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)][metric_name].sum() / 35000
        
        return {
            "all": {"Validity": values_all},
            "seen": {"Validity": values_seen},
            "unseen": {"Validity": values_unseen},
        }
    
    def _aggregate_qed(self, df: pd.DataFrame, metric_name: str = "QED") -> Dict[str, Any]:
        values_all = df[metric_name].mean()
        values_seen = df[df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        values_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        
        return {
            "all": {"QED": values_all},
            "seen": {"QED": values_seen},
            "unseen": {"QED": values_unseen},
        }

    def _aggregate_sa(self, df: pd.DataFrame, metric_name: str = "SA") -> Dict[str, Any]:
        values_all = df[metric_name].mean()
        values_seen = df[df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        values_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        
        return {
            "all": {"SA": values_all},
            "seen": {"SA": values_seen},
            "unseen": {"SA": values_unseen},
        }

    def _aggregate_uniqueness(self, df: pd.DataFrame, metric_name: str = "Uniqueness") -> Dict[str, Any]:
        values_all = df[metric_name].mean()
        values_seen = df[df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        values_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        
        return {
            "all": {"Uniqueness": values_all},
            "seen": {"Uniqueness": values_seen},
            "unseen": {"Uniqueness": values_unseen},
        }

    def _aggregate_diversity(self, df: pd.DataFrame, metric_name: str = "Diversity") -> Dict[str, Any]:
        values_all = df[metric_name].mean()
        values_seen = df[df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        values_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        
        return {
            "all": {"Diversity": values_all},
            "seen": {"Diversity": values_seen},
            "unseen": {"Diversity": values_unseen},
        }
    
    def _aggregate_motif_dist(self, df: pd.DataFrame, metric_name: str = "MotifDist") -> Dict[str, Any]:
        result = {"all": {}, "seen": {}, "unseen": {}}
        
        for subtype in ["Atom Type", "Ring Type", "Functional Group"]:
            js_col = f"{subtype} JS"
            if js_col in df.columns:
                # Compute score as 1 - JS, then take mean
                score_all = (1 - df[js_col]).mean()
                score_seen = (1 - df[df['uniprot'].isin(uniprot_in_trainset)][js_col]).mean()
                score_unseen = (1 - df[~df['uniprot'].isin(uniprot_in_trainset)][js_col]).mean()
                
                result["all"][f"{subtype}_Score"] = score_all
                result["seen"][f"{subtype}_Score"] = score_seen
                result["unseen"][f"{subtype}_Score"] = score_unseen
        return result
        
    def _aggregate_chemfilter(self, df: pd.DataFrame, metric_name: str = "ChemFilter") -> Dict[str, Any]:
        df_seen = df[df['uniprot'].isin(uniprot_in_trainset)]
        df_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)]
        
        # SMILES pass rate: count True values divided by total expected
        smiles_pass_all = df[metric_name].sum() / 120000
        smiles_pass_seen = df_seen[metric_name].sum() / 85000
        smiles_pass_unseen = df_unseen[metric_name].sum() / 35000
        
        # Scaffold pass rate: filter to passed molecules, then count unique scaffolds per uniprot
        df_passed = df[df[metric_name] == True]
        df_passed_seen = df_passed[df_passed['uniprot'].isin(uniprot_in_trainset)]
        df_passed_unseen = df_passed[~df_passed['uniprot'].isin(uniprot_in_trainset)]
        
        # TODO think if group by series to calculate unique scaffolds
        # Count unique scaffolds per uniprot, then sum
        if self.mode == "De_novo_Results":
            unique_scaffolds_all = df_passed.groupby('uniprot')['scaffold'].nunique().sum()
            unique_scaffolds_seen = df_passed_seen.groupby('uniprot')['scaffold'].nunique().sum()
            unique_scaffolds_unseen = df_passed_unseen.groupby('uniprot')['scaffold'].nunique().sum()
        elif self.mode == "Hit_to_Lead_Results":
            unique_scaffolds_all = df_passed.groupby('series')['scaffold'].nunique().sum()
            unique_scaffolds_seen = df_passed_seen.groupby('series')['scaffold'].nunique().sum()
            unique_scaffolds_unseen = df_passed_unseen.groupby('series')['scaffold'].nunique().sum()
        
        return {
            "all": {
                "smiles_pass_rate": smiles_pass_all,
                "scaffold_pass_rate": unique_scaffolds_all / 120000,
            },
            "seen": {
                "smiles_pass_rate": smiles_pass_seen,
                "scaffold_pass_rate": unique_scaffolds_seen / 85000,
            },
            "unseen": {
                "smiles_pass_rate": smiles_pass_unseen,
                "scaffold_pass_rate": unique_scaffolds_unseen / 35000,
            },
        }
    
    def _aggregate_posebuster(self, df: pd.DataFrame, metric_name: str = "PoseBuster") -> Dict[str, Any]:
        """Aggregate PoseBuster metrics, compute pass rate for each sub-metric"""
        exclude_cols = ["id", "smiles", "uniprot", "series", "num_rotatable_bonds"]
        
        # filter to valid molecules only
        df = df[df['all_atoms_connected'] == True]
        
        check_cols = [col for col in df.columns if col not in exclude_cols]
        df['pass_all'] = df[check_cols].all(axis=1)
        
        df_seen = df[df['uniprot'].isin(uniprot_in_trainset)]
        df_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)]
        
        # columns to calculate (including pass_all)
        metric_cols = check_cols + ['pass_all']
        
        return {
            'all': {col: df[col].sum() / 120000 for col in metric_cols},
            'seen': {col: df_seen[col].sum() / 85000 for col in metric_cols},
            'unseen': {col: df_unseen[col].sum() / 35000 for col in metric_cols},
        }
    
    def _aggregate_strain_energy(self, df: pd.DataFrame, metric_name: str = "StrainEnergy") -> Dict[str, Any]:
        df_clean = df.dropna(subset=[metric_name])
        
        values_all = df_clean[metric_name]
        values_seen = df_clean[df_clean['uniprot'].isin(uniprot_in_trainset)][metric_name]
        values_unseen = df_clean[~df_clean['uniprot'].isin(uniprot_in_trainset)][metric_name]
        
        return {
            "all": {
                "StrainEnergy25%": values_all.quantile(0.25),
                "StrainEnergy50%": values_all.quantile(0.50),
                "StrainEnergy75%": values_all.quantile(0.75),
            },
            "seen": {
                "StrainEnergy25%": values_seen.quantile(0.25),
                "StrainEnergy50%": values_seen.quantile(0.50),
                "StrainEnergy75%": values_seen.quantile(0.75),
            },
            "unseen": {
                "StrainEnergy25%": values_unseen.quantile(0.25),
                "StrainEnergy50%": values_unseen.quantile(0.50),
                "StrainEnergy75%": values_unseen.quantile(0.75),
            },
        }
    
    def _aggregate_rmsd(self, df: pd.DataFrame, metric_name: str = "RMSD") -> Dict[str, Any]:
        df = df[df['num_rotatable_bonds'] < 15]
        df_clean = df.dropna(subset=[metric_name])
        
        values_all = df_clean[metric_name]
        values_seen = df_clean[df_clean['uniprot'].isin(uniprot_in_trainset)][metric_name]
        values_unseen = df_clean[~df_clean['uniprot'].isin(uniprot_in_trainset)][metric_name]
        
        return {
            "all": {"RMSD_<2": (values_all < 2).mean()},
            "seen": {"RMSD_<2": (values_seen < 2).mean()},
            "unseen": {"RMSD_<2": (values_unseen < 2).mean()},
        }
        
    def _aggregate_interaction_score(self, df: pd.DataFrame, metric_name: str = "InteractionScore") -> Dict[str, Any]:
        
        # Group by uniprot and calculate proportion exceeding each percentile
        uniprot_stats = []
        for uniprot, group in df.groupby('uniprot'):
            if len(group) == 0:
                continue
            
            ref = self.ref_interaction_percentiles[uniprot]
            scores = group[metric_name]
            
            uniprot_stats.append({
                'uniprot': uniprot,
                'exceeds_p25': (scores > ref['p25']).mean(),
                'exceeds_p50': (scores > ref['p50']).mean(),
                'exceeds_p75': (scores > ref['p75']).mean(),
            })
        
        stats_df = pd.DataFrame(uniprot_stats)
        
        # Split into seen and unseen
        stats_seen = stats_df[stats_df['uniprot'].isin(uniprot_in_trainset)]
        stats_unseen = stats_df[~stats_df['uniprot'].isin(uniprot_in_trainset)]
        
        # Calculate mean across uniprots
        return {
            "all": {
                "InteractionScore>p25": stats_df['exceeds_p25'].mean(),
                "InteractionScore>p50": stats_df['exceeds_p50'].mean(),
                "InteractionScore>p75": stats_df['exceeds_p75'].mean(),
            },
            "seen": {
                "InteractionScore>p25": stats_seen['exceeds_p25'].mean(),
                "InteractionScore>p50": stats_seen['exceeds_p50'].mean(),
                "InteractionScore>p75": stats_seen['exceeds_p75'].mean(),
            },
            "unseen": {
                "InteractionScore>p25": stats_unseen['exceeds_p25'].mean(),
                "InteractionScore>p50": stats_unseen['exceeds_p50'].mean(),
                "InteractionScore>p75": stats_unseen['exceeds_p75'].mean(),
            },
        }
    
    def _aggregate_clash_score(self, df: pd.DataFrame, metric_name: str = "ClashScore") -> Dict[str, Any]:
        values_all = df[metric_name].mean()
        values_seen = df[df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        values_unseen = df[~df['uniprot'].isin(uniprot_in_trainset)][metric_name].mean()
        
        return {
            "all": {"ClashScore": values_all},
            "seen": {"ClashScore": values_seen},
            "unseen": {"ClashScore": values_unseen},
        }
    
    def _aggregate_hit_rediscover(self, df: pd.DataFrame, metric_name: str = "HitRediscover") -> Dict[str, Any]:
        if self.mode == "De_novo_Results":
            return self._aggregate_hit_rediscover_denovo(df, metric_name)
        elif self.mode == "Hit_to_Lead_Results":
            return self._aggrefate_hit_rediscover_h2l(df, metric_name)
    
    def _aggrefate_hit_rediscover_h2l(self, df: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        series_stats = []
        for series, group in df.groupby('series'):
            if len(group) == 0:
                continue
            uniprot = group['uniprot'].iloc[0]
            uniprot_series = f'{uniprot}_{series}'
            
            normalized_affinity_info = normalized_affinity_info_map[uniprot_series]
            normalized_affinity_info_discount = normalized_affinity_info_map_discount[uniprot_series]
            
            # SMILES hits
            unique_smiles_hits = group[group['smiles_hit'] == True]['found_smiles'].nunique()
            has_smiles_hit = unique_smiles_hits > 0
            
            # Scaffold hits
            unique_scaffold_hits = group[group['scaffold_hit'] == True]['found_scaffold'].nunique()
            unique_scaffold_hits_to_smiles = group[group['scaffold_hit'] == True]['gen_smiles'].nunique()
            has_scaffold_hit = unique_scaffold_hits > 0
            
            # Get reference counts for this series
            n_ref_smiles = ref_smiles_scaffold_unique_count_h2l[uniprot_series]
            n_ref_scaffolds = ref_smiles_scaffold_unique_count_h2l[uniprot_series + "_scaffold"]
            
            # Affinity sum
            found_smiles = group['found_smiles'].unique()
            affinity_values = [
                normalized_affinity_info[smi] 
                for smi in found_smiles 
                if smi in normalized_affinity_info
            ]
            affinity_discount_values = [
                normalized_affinity_info_discount[smi]
                for smi in found_smiles
                if smi in normalized_affinity_info_discount
            ]
            
            normalized_affinity_sum = sum(affinity_values)
            max_normalized_affinity = max(affinity_values) if affinity_values else None
            
            normalized_affinity_discount_sum = sum(affinity_discount_values)
            
            # Counts above thresholds
            thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
            affinity_counts = {
                t: sum(v >= t for v in affinity_values) for t in thresholds
            }
            max_affinity_counts = {
                t: max_normalized_affinity >= t if max_normalized_affinity is not None else 0 for t in thresholds
            }
            
            series_stats.append({
                'uniprot': uniprot,
                'series': series,
                # SMILES metrics
                'has_smiles_hit': has_smiles_hit,
                'smiles_hit_num': unique_smiles_hits,
                'smiles_hit_rate': unique_smiles_hits / 200,
                'smiles_hit_fraction': unique_smiles_hits / n_ref_smiles,
                # Scaffold metrics
                'has_scaffold_hit': has_scaffold_hit,
                'scaffold_hit_rate': unique_scaffold_hits_to_smiles / 200,
                'scaffold_hit_fraction': unique_scaffold_hits / n_ref_scaffolds,
                # Affinity
                'normalized_affinity_sum': normalized_affinity_sum,
                'max_normalized_affinity': max_normalized_affinity,
                **{f'normalized_affinity_count_>=_{t}': affinity_counts[t] for t in thresholds},
                **{f'max_normalized_affinity_count_>=_{t}': max_affinity_counts[t] for t in thresholds},
                'NADR': normalized_affinity_discount_sum,
            })
            
        stats_df = pd.DataFrame(series_stats)
        stats_df.to_csv(os.path.join(self.output_dir, 'hitrediscovery_detail.csv'), index=False)
        
        # Split into seen and unseen
        stats_seen = stats_df[stats_df['uniprot'].isin(uniprot_in_trainset)]
        stats_unseen = stats_df[~stats_df['uniprot'].isin(uniprot_in_trainset)]
        
        def calc_metrics(sub_df):
            thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
            return {
                # SMILES metrics
                "smiles_hit_recovery": sub_df['has_smiles_hit'].sum(),
                "smiles_hit_rate": sub_df['smiles_hit_rate'].mean(),
                "smiles_hit_fraction": sub_df['smiles_hit_fraction'].mean(),
                # Scaffold metrics
                "scaffold_hit_recovery": sub_df['has_scaffold_hit'].sum(),
                "scaffold_hit_rate": sub_df['scaffold_hit_rate'].mean(),
                "scaffold_hit_fraction": sub_df['scaffold_hit_fraction'].mean(),
                # Affinity
                'smiles_hit_num': sub_df['smiles_hit_num'].sum(),
                'Mean_normalized_affinity': sub_df['normalized_affinity_sum'].sum() / sub_df['smiles_hit_num'].sum() if sub_df['smiles_hit_num'].sum() > 0 else 0.0,
                **{f'sum_normalized_affinity_count_>=_{t}': sub_df[f'normalized_affinity_count_>=_{t}'].sum() for t in thresholds},
                **{f'sum_max_normalized_affinity_count_>=_{t}': sub_df[f'max_normalized_affinity_count_>=_{t}'].sum() for t in thresholds},
                'Mean_NADR': sub_df.loc[sub_df['smiles_hit_num'] > 0, 'NADR'].mean() if (sub_df['smiles_hit_num'] > 0).any() else 0.0,
               }
        
        return {
            "all": calc_metrics(stats_df),
            "seen": calc_metrics(stats_seen),
            "unseen": calc_metrics(stats_unseen),
        }
        
    def _aggregate_hit_rediscover_denovo(self, df: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        # Group by uniprot to calculate per-uniprot stats
        uniprot_stats = []
        for uniprot, group in df.groupby('uniprot'):
            n_generated = len(group)
            if n_generated == 0:
                continue
            
            # SMILES hits
            unique_smiles_hits = group[group['smiles_hit'] == True]['found_smiles'].nunique()
            has_smiles_hit = unique_smiles_hits > 0
            unique_smiles_hits_not_in_trainset = group[(group['smiles_hit'] == True) & (group['found_smiles_inchi_in_trainset'] == False)]['found_smiles'].nunique()
            has_smiles_hit_not_in_trainset = unique_smiles_hits_not_in_trainset > 0
            
            # Scaffold hits
            unique_scaffold_hits = group[group['scaffold_hit'] == True]['found_scaffold'].nunique()
            unique_scaffold_hits_to_smiles = group[group['scaffold_hit'] == True]['gen_smiles'].nunique()
            has_scaffold_hit = unique_scaffold_hits > 0
            unique_scaffold_hits_not_in_trainset = group[(group['scaffold_hit'] == True) & (group['found_scaffold_inchi_in_trainset'] == False)]['found_scaffold'].nunique()
            unique_scaffold_hits_not_in_trainset_to_smiles = group[(group['scaffold_hit'] == True) & (group['found_scaffold_inchi_in_trainset'] == False)]['gen_smiles'].nunique()
            has_scaffold_hit_not_in_trainset = unique_scaffold_hits_not_in_trainset > 0
            
            # Get reference counts for this uniprot
            n_ref_smiles = ref_smiles_scaffold_unique_count_denovo[uniprot]
            n_ref_scaffolds = ref_smiles_scaffold_unique_count_denovo[uniprot + "_scaffold"]
            
            # Target Aware Score calculations
            ref_smiles = set(self.ref_smiles[uniprot])
            smiles_intersection_specific = len(ref_smiles.intersection(set(group['gen_smiles']))) / group['gen_smiles'].nunique()
            smiles_intersection_all = len(ref_smiles.intersection(set(df['gen_smiles']))) / df['gen_smiles'].nunique()
            smiles_target_aware_score = smiles_intersection_specific / (smiles_intersection_all + 1e-10)
            
            ref_scaffold = set(self.ref_scaffold[uniprot])
            scaffold_intersection_specific = group[group['gen_scaffold'].isin(ref_scaffold)]['gen_smiles'].nunique() / group['gen_smiles'].nunique()
            all_intersection_scaffold = ref_scaffold.intersection(set(df['gen_scaffold']))
            # Build dict only for intersection scaffolds
            gen_all_scaffold_to_smiles = df.groupby('gen_scaffold')['gen_smiles'].apply(set).to_dict()
            find_scaffold_to_smiles = set()
            for scaffold in all_intersection_scaffold:
                if scaffold in gen_all_scaffold_to_smiles:
                    find_scaffold_to_smiles.update(gen_all_scaffold_to_smiles[scaffold])
            scaffold_intersection_all = len(find_scaffold_to_smiles) / df['gen_smiles'].nunique()
            scaffold_target_aware_score = scaffold_intersection_specific / (scaffold_intersection_all + 1e-10)
            
            uniprot_stats.append({
                'uniprot': uniprot,
                # SMILES metrics
                'has_smiles_hit': has_smiles_hit,
                'has_smiles_hit_not_in_trainset': has_smiles_hit_not_in_trainset,
                'smiles_hit_rate': unique_smiles_hits / 1000,
                'smiles_hit_rate_not_in_trainset': unique_smiles_hits_not_in_trainset / 1000,
                'smiles_hit_fraction': unique_smiles_hits / n_ref_smiles,
                'smiles_hit_fraction_not_in_trainset': unique_smiles_hits_not_in_trainset / n_ref_smiles,
                'smiles_TAScore': smiles_target_aware_score,
                # Scaffold metrics
                'has_scaffold_hit': has_scaffold_hit,
                'has_scaffold_hit_not_in_trainset': has_scaffold_hit_not_in_trainset,
                'scaffold_hit_rate': unique_scaffold_hits_to_smiles / 1000,
                'scaffold_hit_rate_not_in_trainset': unique_scaffold_hits_not_in_trainset_to_smiles / 1000,
                'scaffold_hit_fraction': unique_scaffold_hits / n_ref_scaffolds,
                'scaffold_hit_fraction_not_in_trainset': unique_scaffold_hits_not_in_trainset / n_ref_scaffolds,
                'scaffold_TAScore': scaffold_target_aware_score,
                # debug
                'smiles_intersection_specific': smiles_intersection_specific,
                'smiles_intersection_all': smiles_intersection_all,
                'scaffold_intersection_specific': scaffold_intersection_specific,
                'scaffold_intersection_all': scaffold_intersection_all,
            })
        
        stats_df = pd.DataFrame(uniprot_stats)
        stats_df.to_csv(os.path.join(self.output_dir, 'hitrediscovery_detail.csv'), index=False)
        
        # Split into seen and unseen
        stats_seen = stats_df[stats_df['uniprot'].isin(uniprot_in_trainset)]
        stats_unseen = stats_df[~stats_df['uniprot'].isin(uniprot_in_trainset)]
        
        def calc_metrics(sub_df):
            smiles_ta = sub_df['smiles_TAScore']
            scaffold_ta = sub_df['scaffold_TAScore']
            
            return {
                # SMILES metrics
                "smiles_hit_recovery": sub_df['has_smiles_hit'].sum(),
                "smiles_hit_recovery_not_in_trainset": sub_df['has_smiles_hit_not_in_trainset'].sum(),
                "smiles_hit_rate": sub_df['smiles_hit_rate'].mean(),
                "smiles_hit_rate_not_in_trainset": sub_df['smiles_hit_rate_not_in_trainset'].mean(),
                "smiles_hit_fraction": sub_df['smiles_hit_fraction'].mean(),
                "smiles_hit_fraction_not_in_trainset": sub_df['smiles_hit_fraction_not_in_trainset'].mean(),
                "smiles_TAScore_0-1_count": ((smiles_ta >= 0) & (smiles_ta < 1)).sum(),
                "smiles_TAScore_1-10_count": ((smiles_ta >= 1) & (smiles_ta < 10)).sum(),
                "smiles_TAScore_10-100_count": ((smiles_ta >= 10) & (smiles_ta < 100)).sum(),
                "smiles_TAScore_>100_count": (smiles_ta >= 100).sum(),
                # Scaffold metrics
                "scaffold_hit_recovery": sub_df['has_scaffold_hit'].sum(),
                "scaffold_hit_recovery_not_in_trainset": sub_df['has_scaffold_hit_not_in_trainset'].sum(),
                "scaffold_hit_rate": sub_df['scaffold_hit_rate'].mean(),
                "scaffold_hit_rate_not_in_trainset": sub_df['scaffold_hit_rate_not_in_trainset'].mean(),
                "scaffold_hit_fraction": sub_df['scaffold_hit_fraction'].mean(),
                "scaffold_hit_fraction_not_in_trainset": sub_df['scaffold_hit_fraction_not_in_trainset'].mean(),
                # "scaffold_intersection_specific": sub_df['scaffold_intersection_specific'].mean(),
                "scaffold_TAScore_0-1_count": ((scaffold_ta >= 0) & (scaffold_ta < 1)).sum(),
                "scaffold_TAScore_1-10_count": ((scaffold_ta >= 1) & (scaffold_ta < 10)).sum(),
                "scaffold_TAScore_10-100_count": ((scaffold_ta >= 10) & (scaffold_ta < 100)).sum(),
                "scaffold_TAScore_>100_count": (scaffold_ta >= 100).sum(),
            }
        
        # Calculate for all, seen, unseen
        return {
            "all": calc_metrics(stats_df),
            "seen": calc_metrics(stats_seen),
            "unseen": calc_metrics(stats_unseen),
        }

    # ========================== #
    #   Main Aggregation Logic
    # ========================== #

    def aggregate_metric(
        self,
        df: pd.DataFrame,
        metric_name: str,
    ) -> Dict[str, Any]:
        """
        Aggregate a single metric DataFrame.
        """
        
        if metric_name in self.aggregation_map:
            return self.aggregation_map[metric_name](df)

    def run(
        self,
        root_dir: str,
        model_name: str,
        round: str,
        mode: str = "De_novo_Results",
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the aggregation pipeline.
        
        Args:
            root_dir: Root directory
            model_name: Model name
            round: Round identifier
            mode: 'De_novo_Results' or 'Hit_to_Lead_Results'
            output_path: YAML output path
        
        Returns:
            Aggregated results dictionary
        """
        # Step 0: Load reference interaction scores and reference SMILES/scaffolds
        self.mode = mode
        if "InteractionScore" in self.metric_names or "InteractionScore_docked" in self.metric_names:
            self.ref_interaction_percentiles = self.load_reference_interaction_scores(root_dir)
        self.ref_smiles, self.ref_scaffold = self.load_reference_smiles_and_scaffolds(root_dir)
        self.output_dir = os.path.dirname(output_path)
        os.makedirs(self.output_dir, exist_ok=True)
        # Step 1: Collect all CSVs
        all_dfs = self.collect_all_csvs(root_dir, model_name, round, mode)
        
        # Step 2: Aggregate each metric
        all_results = {
            "model_name": model_name,
            "round": round,
            "mode": mode,
            "metrics": {},
        }
        
        for metric_name, df in all_dfs.items():
            if df is not None and len(df) > 0:
                all_results["metrics"][metric_name] = self.aggregate_metric(df, metric_name)
        
        # Step 3: Output YAML
        if output_path:
            self.save_to_yaml(all_results, output_path)
        
        return all_results

    def load_reference_smiles_and_scaffolds(self, root_dir: str):
        """
        Load reference SMILES and scaffolds for each uniprot.
        
        Returns:
            Dict mapping uniprot -> set of reference SMILES
            Dict mapping uniprot -> set of reference scaffolds
            Dict mapping scaffold -> set of all SMILES across uniprots
        """
        ref_smiles = {}
        ref_scaffolds = {}
        
        for uniprot in os.listdir(root_dir):
            cached_ref_smiles_path = os.path.join(
                root_dir,
                uniprot,
                "reference_active_molecules",
                f"{uniprot}_smiles.pkl"
            )
            cached_ref_scaffold_path = os.path.join(
                root_dir,
                uniprot,
                "reference_active_molecules",
                f"{uniprot}_scaffold.pkl"
            )
            
            cached_ref_smiles = pd.read_pickle(cached_ref_smiles_path)
            cached_ref_scaffold = pd.read_pickle(cached_ref_scaffold_path)
            
            ref_smiles[uniprot] = cached_ref_smiles
            ref_scaffolds[uniprot] = set(cached_ref_scaffold.keys())
            
        return ref_smiles, ref_scaffolds

    def load_reference_interaction_scores(self, root_dir: str) -> Dict[str, Dict[str, float]]:
        """
        Load reference interaction scores and compute percentiles per uniprot.
        
        Returns:
            Dict mapping uniprot -> {"p25": float, "p50": float, "p75": float}
        """
        ref_percentiles = {}
        
        for uniprot in os.listdir(root_dir):
            csv_path = os.path.join(
                root_dir, 
                uniprot, 
                "reference_active_molecules", 
                f"{uniprot}_reference_active_molecules_vina_docked_interaction_scores.csv"
            )
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Reference interaction scores CSV not found for {uniprot} at {csv_path}")
            
            df = pd.read_csv(csv_path)
            ref_percentiles[uniprot] = {
                "p25": df["InteractionScore"].quantile(0.25),
                "p50": df["InteractionScore"].quantile(0.50),
                "p75": df["InteractionScore"].quantile(0.75),
            }
        
        return ref_percentiles

    def save_to_yaml(self, results: Dict[str, Any], output_path: str):
        """Save results to a YAML file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results = convert_to_native(results)
        
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"Results saved to {output_path}")
