import pandas as pd
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
import logging
from .metrics import GlaucomaMetrics
from typing import Dict, Any, List
from ..models.segmentation.unet import OpticDiscCupPredictor


# Import the progression analyzer (assuming it will be in src/analysis)
# from ..analysis.progression_analyzer import analyze_progression
# Placeholder for the actual progression analysis function
def build_baseline_features(df: pd.DataFrame,
                            cup_disc_model: OpticDiscCupPredictor
                           ) -> pd.DataFrame:
    """
    Extracts features using ONLY the first available visit for ALL subjects.

    This is useful for training a model that predicts progression risk based
    solely on baseline information. Includes subjects even if they only have one visit.

    Args:
        df: DataFrame with dataset metadata including 'Subject Number', 'Laterality',
            'Visit Number', 'Image Path', 'Progression Status'.
        cup_disc_model: Instantiated segmentation model predictor.

    Returns:
        A pandas DataFrame where each row represents one subject/eye, containing
        baseline metrics as features and the 'progression_status'. Returns an
        empty DataFrame on critical errors.
    """
    if cup_disc_model is None:
        print("Error (Build Baseline): No segmentation model provided.")
        return pd.DataFrame()

    metrics_calculator = GlaucomaMetrics()
    baseline_features_list: List[Dict[str, Any]] = []

    required_cols = ['Subject Number', 'Laterality', 'Visit Number', 'Image Path', 'Progression Status']
    if not all(col in df.columns for col in required_cols):
        print(f"Error (Build Baseline): Input DataFrame missing columns: {required_cols}")
        return pd.DataFrame()

    print("\nBuilding BASELINE features (using first visit)...")
    grouped = df.groupby(['Subject Number', 'Laterality'])
    total_groups = len(grouped)
    print(f"Processing {total_groups} subject-laterality groups...")
    processed_count, failed_count = 0, 0

    for (subject, lat), subject_data in grouped:
        # Ensure we get the actual first visit
        baseline_visit_data = subject_data.sort_values('Visit Number').iloc[0]
        visit_num = baseline_visit_data['Visit Number'] # For logging

        try:
            image = cv2.imread(baseline_visit_data['Image Path'])
            if image is None: failed_count += 1; continue

            try: disc_mask, cup_mask = cup_disc_model.predict(image)
            except Exception: failed_count += 1; continue
            if disc_mask is None or cup_mask is None or np.sum(disc_mask) == 0: failed_count += 1; continue

            metrics = metrics_calculator.extract_metrics(disc_mask, cup_mask)

            if metrics:
                feature_row = {'subject_id': subject, 'laterality': lat,
                               'progression_status': int(baseline_visit_data['Progression Status'])}
                # Add extracted metrics directly as baseline features
                feature_row.update({k: float(v) for k, v in metrics.items()
                                    if isinstance(v, (int, float, bool, np.number))})
                baseline_features_list.append(feature_row)
                processed_count += 1
            else: failed_count += 1; continue # Metrics extraction failed

        except Exception as e:
            print(f"Error processing baseline Subj {subject}, Lat {lat}, Visit {visit_num}: {e}")
            failed_count += 1
            continue

    print(f"\nBaseline feature building summary:")
    print(f"  Successfully processed groups: {processed_count}")
    print(f"  Groups failed (load/seg/metrics): {failed_count}")

    features_df = pd.DataFrame(baseline_features_list)
    if features_df.empty: print("Warning: Resulting baseline features DataFrame is empty.")
    # Optional: Add checks for NaNs/Infs in the final DataFrame here
    return features_df


def build_longitudinal_features(df: pd.DataFrame,
                                cup_disc_model: OpticDiscCupPredictor
                               ) -> pd.DataFrame:
    """
    Extracts features using changes between visits for subjects with >= 2 visits.

    Includes change metrics (e.g., VCDR change, rim area change rate),
    baseline metrics (prefixed with 'baseline_'), and final metrics (prefixed
    with 'final_'). Only subjects with at least two successfully processed visits
    are included.

    Args:
        df: DataFrame with dataset metadata.
        cup_disc_model: Instantiated segmentation model predictor.

    Returns:
        A pandas DataFrame where each row represents one subject/eye with >=2 visits,
        containing longitudinal and static features. Returns an empty DataFrame on
        critical errors.
    """
    if cup_disc_model is None:
        print("Error (Build Longitudinal): No segmentation model provided.")
        return pd.DataFrame()

    metrics_calculator = GlaucomaMetrics()
    longitudinal_features_list: List[Dict[str, Any]] = []

    required_cols = ['Subject Number', 'Laterality', 'Visit Number', 'Image Path', 'Progression Status']
    if not all(col in df.columns for col in required_cols):
        print(f"Error (Build Longitudinal): Input DataFrame missing columns: {required_cols}")
        return pd.DataFrame()

    print("\nBuilding LONGITUDINAL features (using >= 2 visits)...")
    grouped = df.groupby(['Subject Number', 'Laterality'])
    total_groups = len(grouped)
    print(f"Processing {total_groups} potential subject-laterality groups...")

    processed_groups, skipped_lt2_visits, skipped_lt2_metrics, skipped_prog_fail = 0, 0, 0, 0

    for (subject, lat), subject_data in grouped:
        subject_data = subject_data.sort_values('Visit Number').copy()

        if len(subject_data) < 2: # Filter groups with < 2 visits
            skipped_lt2_visits += 1
            continue

        progression_status = subject_data['Progression Status'].iloc[0] # Use first visit's status as ground truth

        # Process all visits for this group
        metrics_list: List[Dict[str, Any]] = []
        for idx, row in subject_data.iterrows():
            try:
                image = cv2.imread(row['Image Path'])
                if image is None: continue
                try: disc_mask, cup_mask = cup_disc_model.predict(image)
                except Exception: continue
                if disc_mask is None or cup_mask is None or np.sum(disc_mask) == 0: continue
                metrics = metrics_calculator.extract_metrics(disc_mask, cup_mask)
                if metrics: metrics_list.append(metrics) # Only append successful metrics
            except Exception: continue # Catch any other processing error for the visit

        # Need >= 2 *valid* metrics for progression analysis
        if len(metrics_list) < 2:
            skipped_lt2_metrics += 1
            continue

        # Calculate progression metrics
        progression_metrics = metrics_calculator.analyze_progression(metrics_list)
        if 'error' in progression_metrics:
            skipped_prog_fail += 1
            continue

        # Assemble feature row
        feature_row = {'subject_id': subject, 'laterality': lat,
                       'progression_status': int(progression_status)}

        # Add progression change features (change, rate, etc.)
        exclude_prog_keys = ['error', 'progression_likely', 'num_valid_visits']
        feature_row.update({k: float(v) for k, v in progression_metrics.items()
                            if k not in exclude_prog_keys and isinstance(v, (int, float, bool, np.number))})

        # Add baseline metrics (from first valid visit) with prefix
        if metrics_list[0]:
            feature_row.update({f'baseline_{k}': float(v) for k, v in metrics_list[0].items()
                                if isinstance(v, (int, float, bool, np.number))})

        # Add final metrics (from last valid visit) with prefix
        if metrics_list[-1]:
            feature_row.update({f'final_{k}': float(v) for k, v in metrics_list[-1].items()
                                if isinstance(v, (int, float, bool, np.number))})

        # Clean potential infinities introduced earlier (e.g., percent change)
        for key, value in feature_row.items():
            if isinstance(value, float) and np.isinf(value):
                feature_row[key] = np.sign(value) * 1e9 # Replace with large number

        longitudinal_features_list.append(feature_row)
        processed_groups += 1

    print(f"\nLongitudinal feature building summary:")
    print(f"  Groups with >= 2 visits in data: {total_groups - skipped_lt2_visits}")
    print(f"  Successfully processed for features: {processed_groups}")
    print(f"  Skipped (fewer than 2 visits originally): {skipped_lt2_visits}")
    print(f"  Skipped (fewer than 2 valid metrics obtained): {skipped_lt2_metrics}")
    print(f"  Skipped (progression analysis failed): {skipped_prog_fail}")

    features_df = pd.DataFrame(longitudinal_features_list)
    if features_df.empty: print("Warning: Resulting longitudinal features DataFrame is empty.")
    # Optional: Add checks for NaNs/Infs in the final DataFrame here
    return features_df