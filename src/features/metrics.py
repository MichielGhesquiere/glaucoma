import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
import logging
from typing import Dict, Any, Optional, List, Union
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlaucomaMetrics:
    """
    Extracts and calculates clinically relevant metrics from optic cup and disc
    segmentation masks. Automatically infers laterality (OD/OS) based on the
    horizontal position of the optic disc centroid relative to the image center.
    """
    # Example thresholds (should be clinically validated)
    VCDR_PROGRESSION_THRESHOLD: float = 0.05
    RIM_AREA_PERCENT_PROGRESSION_THRESHOLD: float = -10.0

    def __init__(self, pixel_to_mm_ratio: Optional[float] = None):
        """
        Initializes the metrics calculator.

        Args:
            pixel_to_mm_ratio: Optional conversion factor from pixels to
                               millimeters, if available from image metadata.
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio

    def _validate_masks(self, disc_mask: np.ndarray, cup_mask: np.ndarray) -> bool:
        """Checks if masks are valid for processing."""
        if disc_mask.shape != cup_mask.shape:
            print("Error: Disc and Cup masks have different shapes.")
            return False
        if disc_mask.ndim != 2 or cup_mask.ndim != 2:
            print("Error: Masks must be 2D.")
            return False
        return True

    def _get_main_region_props(self, mask: np.ndarray) -> Optional[Any]:
        """Finds properties of the largest connected region in a binary mask."""
        props_list = regionprops(mask)
        if not props_list:
            return None
        # Return props of the largest region if multiple exist
        return max(props_list, key=lambda p: p.area)

    def extract_metrics(self, disc_mask: np.ndarray, cup_mask: np.ndarray) -> Dict[str, Any]:
        """
        Extracts comprehensive metrics from disc and cup masks.

        Args:
            disc_mask: Binary mask of the optic disc segmentation (HxW).
            cup_mask: Binary mask of the optic cup segmentation (HxW).

        Returns:
            A dictionary containing calculated metrics (areas, ratios, ISNT, etc.),
            or an empty dictionary if essential components (disc/cup) cannot be processed.
            Metrics are returned as floats or booleans.
        """
        if not self._validate_masks(disc_mask, cup_mask):
            return {}
        h_img, w_img = disc_mask.shape

        # Preprocessing: Ensure binary, fill holes
        disc_mask = binary_fill_holes((disc_mask > 0)).astype(np.uint8)
        cup_mask = binary_fill_holes((cup_mask > 0)).astype(np.uint8)

        # Get region properties for disc and cup
        disc_props = self._get_main_region_props(disc_mask)
        cup_props = self._get_main_region_props(cup_mask)

        if not disc_props:
            # print("Warning: No valid disc region found.") # Reduce verbosity
            return {}
        if not cup_props:
            # print("Warning: No valid cup region found.") # Reduce verbosity
            return {}

        # --- Basic Area Metrics ---
        disc_area = float(disc_props.area)
        cup_area = float(cup_props.area)

        # Clamp cup area if segmentation is inconsistent
        if cup_area > disc_area:
            # print(f"Warning: Cup area ({cup_area:.0f}) > Disc area ({disc_area:.0f}). Clamping cup area.")
            cup_area = disc_area
        rim_area = max(0.0, disc_area - cup_area) # Ensure non-negative
        area_ratio = cup_area / disc_area if disc_area > 0 else 0.0
        area_ratio = min(1.0, area_ratio) # Ensure ratio <= 1

        # --- Contour-based Metrics (VCDR/HCDR) ---
        metrics: Dict[str, Any] = {}
        disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if disc_contours and cup_contours:
            disc_contour = max(disc_contours, key=cv2.contourArea)
            cup_contour = max(cup_contours, key=cv2.contourArea)
            _, _, disc_w_cnt, disc_h_cnt = cv2.boundingRect(disc_contour)
            _, _, cup_w_cnt, cup_h_cnt = cv2.boundingRect(cup_contour)

            vcdr = float(cup_h_cnt) / float(disc_h_cnt) if disc_h_cnt > 0 else 0.0
            hcdr = float(cup_w_cnt) / float(disc_w_cnt) if disc_w_cnt > 0 else 0.0
            vcdr = min(1.0, vcdr) # Clamp ratios
            hcdr = min(1.0, hcdr)

            # --- Centroid, Displacement, Laterality ---
            disc_center = disc_props.centroid # (row, col) i.e. (y, x)
            cup_center = cup_props.centroid
            displacement = np.linalg.norm(np.array(cup_center) - np.array(disc_center))
            y_center, x_center = int(disc_center[0]), int(disc_center[1])

            # Infer laterality based on disc centroid horizontal position
            image_center_x = w_img / 2.0
            inferred_laterality = 'OD' if x_center < image_center_x else 'OS'

            # --- ISNT Quadrant Analysis ---
            superior_mask = np.zeros((h_img, w_img), dtype=np.uint8); superior_mask[:y_center, :] = 1
            inferior_mask = np.zeros((h_img, w_img), dtype=np.uint8); inferior_mask[y_center:, :] = 1
            nasal_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            temporal_mask = np.zeros((h_img, w_img), dtype=np.uint8)

            if inferred_laterality == 'OD': # Right Eye: Nasal left, Temporal right
                nasal_mask[:, :x_center] = 1; temporal_mask[:, x_center:] = 1
            else: # Left Eye: Nasal right, Temporal left
                nasal_mask[:, x_center:] = 1; temporal_mask[:, :x_center] = 1

            # Rim mask calculation (ensure non-negative)
            rim_mask = disc_mask.astype(np.int16) - cup_mask.astype(np.int16)
            rim_mask.clip(0, out=rim_mask); rim_mask = rim_mask.astype(np.uint8)

            # Calculate rim and disc area per quadrant
            superior_rim = np.sum(rim_mask * superior_mask)
            inferior_rim = np.sum(rim_mask * inferior_mask)
            nasal_rim = np.sum(rim_mask * nasal_mask)
            temporal_rim = np.sum(rim_mask * temporal_mask)

            superior_disc = np.sum(disc_mask * superior_mask)
            inferior_disc = np.sum(disc_mask * inferior_mask)
            nasal_disc = np.sum(disc_mask * nasal_mask)
            temporal_disc = np.sum(disc_mask * temporal_mask)

            # Calculate rim-to-disc ratio per quadrant
            superior_ratio = float(superior_rim) / float(superior_disc) if superior_disc > 0 else 0.0
            inferior_ratio = float(inferior_rim) / float(inferior_disc) if inferior_disc > 0 else 0.0
            nasal_ratio = float(nasal_rim) / float(nasal_disc) if nasal_disc > 0 else 0.0
            temporal_ratio = float(temporal_rim) / float(temporal_disc) if temporal_disc > 0 else 0.0

            # Clamp ratios and check ISNT rule (I >= S >= N >= T)
            superior_ratio = min(1.0, superior_ratio)
            inferior_ratio = min(1.0, inferior_ratio)
            nasal_ratio = min(1.0, nasal_ratio)
            temporal_ratio = min(1.0, temporal_ratio)
            isnt_violation = not (inferior_ratio >= superior_ratio >= nasal_ratio >= temporal_ratio)

            # --- Compile Metrics ---
            metrics = {
                'disc_area': disc_area, 'cup_area': cup_area, 'rim_area': rim_area,
                'area_ratio': area_ratio, 'vcdr': vcdr, 'hcdr': hcdr,
                'cup_displacement': float(displacement),
                'superior_rim_ratio': superior_ratio, 'inferior_rim_ratio': inferior_ratio,
                'nasal_rim_ratio': nasal_ratio, 'temporal_rim_ratio': temporal_ratio,
                'isnt_violation': bool(isnt_violation),
                # 'inferred_laterality': inferred_laterality # Optional: include if needed
            }
        else:
            # Fallback: Only return area metrics if contours failed
            metrics = { 'disc_area': disc_area, 'cup_area': cup_area,
                        'rim_area': rim_area, 'area_ratio': area_ratio }

        # Add real-world measurements if ratio provided
        if self.pixel_to_mm_ratio is not None:
            ratio_sq = self.pixel_to_mm_ratio ** 2
            metrics['disc_area_mm2'] = metrics.get('disc_area', 0) * ratio_sq
            metrics['cup_area_mm2'] = metrics.get('cup_area', 0) * ratio_sq
            metrics['rim_area_mm2'] = metrics.get('rim_area', 0) * ratio_sq
            if 'cup_displacement' in metrics:
                metrics['cup_displacement_mm'] = metrics['cup_displacement'] * self.pixel_to_mm_ratio

        return metrics

    def analyze_progression(self, metrics_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes a time series of metrics for evidence of progression.

        Requires at least two valid metrics dictionaries in the series.

        Args:
            metrics_series: A list of metrics dictionaries, chronologically ordered.

        Returns:
            A dictionary containing progression indicators (changes, rates, flags),
            or a dictionary with an 'error' key if analysis cannot be performed.
        """
        if not isinstance(metrics_series, list):
             return {"error": "Input must be a list of metric dictionaries."}

        # Filter out potential None or non-dict entries
        valid_metrics = [m for m in metrics_series if isinstance(m, dict) and m]
        if len(valid_metrics) < 2:
            return {"error": f"Need >= 2 valid metric dicts (found {len(valid_metrics)})"}

        first, last = valid_metrics[0], valid_metrics[-1]

        # Check required keys are present in first and last valid dicts
        required_keys = ['vcdr', 'rim_area', 'area_ratio', 'superior_rim_ratio',
                         'inferior_rim_ratio', 'isnt_violation', 'nasal_rim_ratio',
                         'temporal_rim_ratio']
        if not all(k in first and k in last for k in required_keys):
            missing = [k for k in required_keys if k not in first or k not in last]
            return {"error": f"Missing required keys for progression: {missing}"}

        # --- Calculate Change Metrics ---
        def safe_change(key: str) -> float:
            try: return float(last[key]) - float(first[key])
            except (TypeError, ValueError, KeyError): return 0.0

        def safe_pct_change(key: str) -> float:
            try:
                l, f = float(last[key]), float(first[key])
                if f == 0: return 0.0 if l == 0 else np.inf # Handle division by zero
                return ((l - f) / f) * 100.0
            except (TypeError, ValueError, KeyError): return 0.0

        change_keys = ['vcdr', 'hcdr', 'rim_area', 'area_ratio', 'superior_rim_ratio',
                       'inferior_rim_ratio', 'nasal_rim_ratio', 'temporal_rim_ratio',
                       'cup_displacement']
        percent_change_keys = ['vcdr', 'rim_area'] # Add others if needed

        indicators = {f'{k}_change': safe_change(k) for k in change_keys if k in first and k in last}
        indicators.update({f'{k}_percent_change': safe_pct_change(k) for k in percent_change_keys if k in first and k in last})

        # --- Calculate Rate of Change ---
        intervals = len(valid_metrics) - 1
        if intervals > 0:
            # Create a new dictionary for rates
            rate_indicators = {}
            for key, value in indicators.items():
                if '_change' in key and '_percent' not in key:
                    rate_indicators[f'{key}_rate'] = value / intervals
            # Update the original dictionary with rates after iteration
            indicators.update(rate_indicators)
        else: # Should not happen due to checks, but set rate to 0 defensively
            rate_indicators = {
                f'{key}_rate': 0.0 
                for key in indicators 
                if '_change' in key and '_percent' not in key
            }
            indicators.update(rate_indicators)

        # --- ISNT Violation Change ---
        indicators['isnt_violated_now'] = bool(last['isnt_violation'])
        indicators['isnt_newly_violated'] = (bool(last['isnt_violation']) and
                                            not bool(first['isnt_violation']))

        # --- Progression Likelihood Flag (Example) ---
        vcdr_change = indicators.get('vcdr_change', 0)
        rim_pct_change = indicators.get('rim_area_percent_change', 0)
        is_newly_violated = indicators['isnt_newly_violated']

        indicators['progression_likely'] = (
            (vcdr_change > self.VCDR_PROGRESSION_THRESHOLD) or
            (rim_pct_change < self.RIM_AREA_PERCENT_PROGRESSION_THRESHOLD) or
            is_newly_violated
        )
        # --- Store number of visits used for analysis ---
        indicators['num_valid_visits'] = len(valid_metrics)

        # Replace any infinities from percent change calc if needed
        for k, v in indicators.items():
             if isinstance(v, float) and np.isinf(v):
                  # print(f"Warning: Replacing infinite value for {k} with large number.")
                  indicators[k] = np.sign(v) * 1e9 # Or np.nan

        return indicators

    def visualize_ratio_measurements(self, image: np.ndarray, disc_mask: np.ndarray,
                                     cup_mask: np.ndarray) -> np.ndarray:
        """
        Creates a visualization overlay showing segmentation contours and
        lines indicating VCDR/HCDR measurements based on bounding boxes.

        Args:
            image: The original fundus image (RGB or Grayscale).
            disc_mask: Binary mask of the optic disc.
            cup_mask: Binary mask of the optic cup.

        Returns:
            The original image blended with the visualization overlay (RGB).
            Returns the original image if contours cannot be found.
        """
        # Ensure masks are binary
        disc_mask = (disc_mask > 0).astype(np.uint8)
        cup_mask = (cup_mask > 0).astype(np.uint8)

        # Convert image to RGB
        if image.ndim == 2: image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3: image_rgb = image.copy()
        elif image.shape[2] == 4: image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else: raise ValueError(f"Unsupported image shape: {image.shape}")

        # Find contours
        disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not disc_contours or not cup_contours:
            print("Warning (visualize): No disc or cup contours found.")
            return image_rgb # Return original image

        overlay = image_rgb.copy()
        disc_contour = max(disc_contours, key=cv2.contourArea)
        cup_contour = max(cup_contours, key=cv2.contourArea)

        # Draw contours
        cv2.drawContours(overlay, [disc_contour], -1, (0, 255, 0), 2)  # Green disc
        cv2.drawContours(overlay, [cup_contour], -1, (255, 0, 0), 2)   # Red cup

        # Get bounding boxes for VCDR/HCDR lines
        disc_x, disc_y, disc_w, disc_h = cv2.boundingRect(disc_contour)
        cup_x, cup_y, cup_w, cup_h = cv2.boundingRect(cup_contour)

        # Draw VCDR/HCDR measurement lines
        disc_cx, disc_cy = disc_x + disc_w // 2, disc_y + disc_h // 2
        cup_cx, cup_cy = cup_x + cup_w // 2, cup_y + cup_h // 2
        cv2.line(overlay, (disc_cx, disc_y), (disc_cx, disc_y + disc_h), (0, 255, 255), 2) # Yellow Disc V
        cv2.line(overlay, (cup_cx, cup_y), (cup_cx, cup_y + cup_h), (255, 255, 0), 2)     # Cyan Cup V
        cv2.line(overlay, (disc_x, disc_cy), (disc_x + disc_w, disc_cy), (0, 255, 255), 2) # Yellow Disc H
        cv2.line(overlay, (cup_x, cup_cy), (cup_x + cup_w, cup_cy), (255, 255, 0), 2)     # Cyan Cup H

        # Calculate and display ratios
        vcdr = float(cup_h) / float(disc_h) if disc_h > 0 else 0.0
        hcdr = float(cup_w) / float(disc_w) if disc_w > 0 else 0.0
        vcdr_disp, hcdr_disp = min(vcdr, 1.0), min(hcdr, 1.0) # Clamp for display

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"VCDR: {vcdr_disp:.2f}", (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"HCDR: {hcdr_disp:.2f}", (10, 60), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend overlay
        alpha = 0.6
        return cv2.addWeighted(image_rgb, 1.0 - alpha, overlay, alpha, 0)

    def plot_metrics_over_time(self, metrics_series: List[Dict[str, Any]],
                               visit_numbers: Optional[List[Union[int, float]]] = None
                               ) -> Optional[plt.Figure]:
        """
        Plots key glaucoma metrics over time to visualize progression.
        Adds padding to the upper y-axis limit for better visibility.

        Args:
            metrics_series: List of metrics dictionaries, chronologically ordered.
            visit_numbers: Optional list of visit identifiers corresponding to metrics_series.

        Returns:
            A matplotlib Figure object containing the plots, or None if plotting is not possible.
        """
        if not isinstance(metrics_series, list): print("Plotting Error: metrics_series must be a list."); return None
        num_total_visits = len(metrics_series)
        if visit_numbers is None: visit_numbers = list(range(1, num_total_visits + 1))
        elif len(visit_numbers) != num_total_visits: visit_numbers = list(range(1, num_total_visits + 1))

        plot_data, valid_visit_nums = [], []
        for i, metrics in enumerate(metrics_series):
            if isinstance(metrics, dict) and metrics:
                metrics_copy = metrics.copy(); metrics_copy['visit'] = visit_numbers[i]; plot_data.append(metrics_copy); valid_visit_nums.append(visit_numbers[i])

        if len(plot_data) < 2: print(f"Plotting Info: Need >= 2 valid data points (found {len(plot_data)})."); return None
        df = pd.DataFrame(plot_data).sort_values('visit'); plot_visits_unique = df['visit'].unique()

        # --- Create Plots ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True) # Adjusted figsize slightly
        fig.set_facecolor('white') 

        # Plot 1: VCDR and HCDR
        axs[0, 0].plot(df['visit'], df.get('vcdr', np.nan), 'o-', label='VCDR', color='dodgerblue', markersize=5)
        axs[0, 0].plot(df['visit'], df.get('hcdr', np.nan), 's--', label='HCDR', color='lightcoral', markersize=5)
        axs[0, 0].set_ylabel('Cup-to-Disc Ratio'); axs[0, 0].set_title('Cup-to-Disc Ratios (CDR)'); axs[0, 0].legend()

        # Plot 2: Rim Area
        axs[0, 1].plot(df['visit'], df.get('rim_area', np.nan), 'o-', color='forestgreen', markersize=5)
        axs[0, 1].set_ylabel('Rim Area (pixels)'); axs[0, 1].set_title('Neuroretinal Rim Area')

        # Plot 3: ISNT Quadrant Ratios
        axs[1, 0].plot(df['visit'], df.get('inferior_rim_ratio', np.nan), 's-', label='Inferior', color='purple', markersize=5)
        axs[1, 0].plot(df['visit'], df.get('superior_rim_ratio', np.nan), 'o-', label='Superior', color='orange', markersize=5)
        axs[1, 0].plot(df['visit'], df.get('nasal_rim_ratio', np.nan), '^-', label='Nasal', color='teal', markersize=5)
        axs[1, 0].plot(df['visit'], df.get('temporal_rim_ratio', np.nan), 'd-', label='Temporal', color='firebrick', markersize=5)
        axs[1, 0].set_ylabel('Rim-to-Disc Ratio'); axs[1, 0].set_title('ISNT Quadrant Rim Ratios'); axs[1, 0].legend()

        # Plot 4: Area Ratio
        axs[1, 1].plot(df['visit'], df.get('area_ratio', np.nan), 'o-', color='darkgoldenrod', markersize=5)
        axs[1, 1].set_ylabel('Cup-to-Disc Area Ratio'); axs[1, 1].set_title('Cup/Disc Area Ratio')

        # --- Formatting and Y-axis Adjustment ---
        for ax in axs.flat:
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('Visit Number')
            ax.set_xticks(plot_visits_unique)
            ax.tick_params(axis='x', rotation=45)

            # --- Adjust Y-axis Upper Limit ---
            current_bottom, current_top = ax.get_ylim()
            if current_top <= 0: # Handle cases where max value is 0 or less
                 new_top = 0.1 # Set a small default top limit
            else:
                 # Add 10-15% padding to the top
                 padding = (current_top - current_bottom) * 0.15 # Calculate padding based on current range
                 new_top = current_top + max(padding, 0.05) # Add padding, ensure at least a small gap

            ax.set_ylim(bottom=0, top=new_top) # Keep bottom at 0, adjust top
            # ------------------------------------

        title_visits = f"Visits {min(plot_visits_unique)} to {max(plot_visits_unique)}"
        fig.suptitle(f'Glaucoma Metrics Progression ({title_visits})', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # tight_layout usually works well here
        plt.show() # Display the plot

        return fig