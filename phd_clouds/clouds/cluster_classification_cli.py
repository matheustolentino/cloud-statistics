"""
Config-driven, CLI-friendly wrapper around the cluster-based classification
implemented in ``phd_clouds/clouds.py`` (the ``CloudProcessing`` class).

This is a **copy**, not an edit, of the driver logic that ties ``CloudProcessing``
together for a single day -- that logic previously only existed as
``process_single_date`` in ``notebooks/cluster_classification.ipynb`` (cell 4).
``CloudProcessing`` itself is not messy enough to warrant copying wholesale (it
is already a fairly clean, reusable class with a parametrized
``classify_clusters_user_defined_thresholds(min_cloudp, min_rainp,
min_liquid_percentage, min_ice_percentage)`` method -- exactly the "percentage
of hydrometeors" knobs this CLI exposes), so it is imported and reused as-is.

Two things from the original notebook driver were fixed here rather than
carried over verbatim, without touching ``phd_clouds/clouds.py``:
  * ``CloudProcessing.get_filenames_from_datastr`` builds the categorize glob
    pattern as ``f"{datastr}_{site}_'categorize'.nc"`` (stray literal quotes
    around ``categorize`` -- phd_clouds/clouds.py:512) which never matches a
    real downloaded file. Fixed here via a small subclass override.
  * ``path_microphys`` was read from a notebook-global closure variable
    instead of being one of ``process_single_date``'s parameters. Here it is
    an explicit argument.

Also dropped: the notebook's per-variable sensitivity sweep (running the same
day through many threshold values to build a sensitivity dataset). This CLI
runs one set of thresholds per invocation, which is what "pass a config file
with percentage-of-hydrometeors thresholds" means; sweeping is a separate,
research-specific concern and not reimplemented here.
"""
from __future__ import annotations

import datetime
import glob
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from phd_clouds.clouds import CloudProcessing

logger = logging.getLogger(__name__)


class CloudProcessingFixed(CloudProcessing):
    """``CloudProcessing`` with the categorize-filename bug fixed (see module docstring)."""

    def get_filenames_from_datastr(self, datastr: str) -> None:
        dic_files = {"classification": None, "lwc": None, "iwc": None, "der": None,
                     "ier": None, "categorize": None, "radar": None, "mwr": None}
        if self.path_classification is not None:
            dic_files["classification"] = glob.glob(
                os.path.join(self.path_classification, f"{datastr}_{self.site}_classification.nc")
            )
        if self.path_microphys is not None:
            dic_files["lwc"] = glob.glob(
                os.path.join(self.path_microphys, f"{datastr}_{self.site}_lwc-scaled-adiabatic.nc")
            )
            dic_files["iwc"] = glob.glob(
                os.path.join(self.path_microphys, f"{datastr}_{self.site}_iwc-Z-T-method.nc")
            )
        if self.path_categorize is not None:
            dic_files["categorize"] = glob.glob(
                os.path.join(self.path_categorize, f"{datastr}_{self.site}_categorize.nc")
            )
        if self.path_radar is not None:
            dic_files["radar"] = glob.glob(
                os.path.join(self.path_radar, f"{datastr}_{self.site}_rpg-fmcw-94*.nc")
            )
        if self.path_mwr is not None:
            dic_files["mwr"] = glob.glob(
                os.path.join(self.path_mwr, f"{datastr}_{self.site}_hatpro*.nc")
            )
        self.filenames = dic_files


# ---------------------------------------------------------------------------
# Default "percentage of hydrometeors" thresholds == the values already used
# as defaults by CloudProcessing.classify_clusters_user_defined_thresholds,
# plus the analyze_clouds/attenuation/fit-parameter knobs process_single_date
# hardcoded. Dumped verbatim to
# hydrometeor_configs/cluster_hydrometeor_percentages.example.json.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "min_cloud_pixels": 100,
    "min_rain_pixels": 10,
    "min_liquid_percentage": 70,
    "min_ice_percentage": 90,
    "filter_abl_ice_clouds": True,
    "thick_threshold": 700,
    "base_threshold": 4000,
    "fit_params": {"small_than": 0.2, "larger_than": 0.8},
    "attenuation": {
        "time_roll": "10T",
        "lwp_threshold": 0.8,
        "corr_threshold": -0.5,
        "lwp2_threshold": 1,
    },
    "products_to_save": {
        "cloud_occurrence": True,
        "cloud_props": True,
        "cloud_type": True,
        "fit_params": False,
        "count_verification": False,
        "time_duration_cloud_type": True,
    },
}


@dataclass
class ClusterConfig:
    min_cloud_pixels: int
    min_rain_pixels: int
    min_liquid_percentage: float
    min_ice_percentage: float
    filter_abl_ice_clouds: bool
    thick_threshold: float
    base_threshold: float
    fit_params: Dict[str, float]
    attenuation: Dict[str, Any]
    products_to_save: Dict[str, bool]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ClusterConfig":
        return cls(
            min_cloud_pixels=raw["min_cloud_pixels"],
            min_rain_pixels=raw["min_rain_pixels"],
            min_liquid_percentage=raw["min_liquid_percentage"],
            min_ice_percentage=raw["min_ice_percentage"],
            filter_abl_ice_clouds=raw["filter_abl_ice_clouds"],
            thick_threshold=raw["thick_threshold"],
            base_threshold=raw["base_threshold"],
            fit_params=raw["fit_params"],
            attenuation=raw["attenuation"],
            products_to_save=raw["products_to_save"],
        )

    @classmethod
    def load(cls, path: Optional[str]) -> "ClusterConfig":
        if path is None:
            return cls.from_dict(DEFAULT_CONFIG)
        with open(path) as f:
            raw = json.load(f)
        return cls.from_dict(raw)


@dataclass
class DayPaths:
    classification: str
    categorize: str
    radar: str
    mwr: str
    microphys: str  # folder containing both lwc-scaled-adiabatic and iwc-Z-T-method files


@dataclass
class RunOptions:
    output_dir: Optional[str] = None
    save_products: bool = False
    make_plots: bool = False


def run_single_date(datastr: str, site: str, paths: DayPaths, config: ClusterConfig,
                     options: RunOptions) -> Dict[str, Any]:
    """Config-driven equivalent of the notebook's ``process_single_date``, for one day."""
    cloud_processing = CloudProcessingFixed(
        path_classification=paths.classification,
        path_microphys=paths.microphys,
        path_categorize=paths.categorize,
        path_radar=paths.radar,
        path_mwr=paths.mwr,
        site=site,
        min_cloud_pixels=config.min_cloud_pixels,
        min_rain_pixels=config.min_rain_pixels,
    )

    cloud_processing.get_filenames_from_datastr(datastr)
    if not cloud_processing.filenames["classification"]:
        raise FileNotFoundError(f"No classification file found for {datastr} in {paths.classification}")

    cloud_processing.load_classification()
    cloud_processing.load_categorize()
    cloud_processing.load_lwc_scaled_adiabatic()
    cloud_processing.load_iwc_Z_T_method()
    cloud_processing.load_radar()
    cloud_processing.load_mwr()
    cloud_processing.initialize_time()
    cloud_processing.initialize_datasets()
    cloud_processing.add_radar_lwp()
    cloud_processing.add_mwr_lwp()
    cloud_processing.calculate_fit_parameters(**config.fit_params)
    cloud_processing.create_count_dataset_for_verification()
    cloud_processing.plot_linear_fit_lwp(make_plot=options.make_plots)
    cloud_processing.generate_cloud_mask()

    cloud_processing.classify_clusters_user_defined_thresholds(
        min_cloudp=config.min_cloud_pixels,
        min_rainp=config.min_rain_pixels,
        min_liquid_percentage=config.min_liquid_percentage,
        min_ice_percentage=config.min_ice_percentage,
    )
    cloud_processing.analyze_clouds(
        filter_abl_ice_clouds=config.filter_abl_ice_clouds,
        thick_threshold=config.thick_threshold,
        base_threshold=config.base_threshold,
    )
    cloud_processing.create_cluster_classification_product()

    import matplotlib.pyplot as plt
    from phd_clouds.constants import CLOUD_COLORS
    cloud_cmap = plt.cm.colors.ListedColormap(CLOUD_COLORS)
    cloud_processing.create_attenuation_mask(
        cloud_cmap,
        time_roll=config.attenuation["time_roll"],
        lwp_treshold=config.attenuation["lwp_threshold"],
        corr_treshold=config.attenuation["corr_threshold"],
        lwp2_treshold=config.attenuation["lwp2_threshold"],
        make_plot=options.make_plots,
    )
    cloud_processing.add_attenuation_to_clouds()
    cloud_processing.compute_cloud_lwp_iwp(make_plot=options.make_plots)

    if options.save_products and options.output_dir:
        cloud_processing.save_processed_data(options.output_dir, config.products_to_save)

    return {
        "datastr": datastr,
        "cloud_type": cloud_processing.cloud_type,
        "cloud_occurrence": getattr(cloud_processing, "cloud_occurrence", None),
        "cloud_props": getattr(cloud_processing, "cloud_props", None),
    }


def run(date_list: Sequence[str], site: str, paths: DayPaths, config: ClusterConfig,
        options: RunOptions) -> List[Dict[str, Any]]:
    """Process a list of ``YYYYMMDD`` date strings with the cluster-based method.
    Returns a list of ``{"date": ..., "status": "ok"|"failed", "error": Optional[str]}``."""
    results = []
    for datastr in date_list:
        try:
            run_single_date(datastr, site, paths, config, options)
            results.append({"date": datastr, "status": "ok", "error": None})
            logger.info("Cluster-based classification OK for %s", datastr)
        except Exception as exc:  # noqa: BLE001 - one bad day must not stop the batch
            logger.error("Cluster-based classification failed for %s: %s", datastr, exc)
            results.append({"date": datastr, "status": "failed", "error": str(exc)})
    return results
