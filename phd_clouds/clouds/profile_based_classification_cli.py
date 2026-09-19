"""
Config-driven, CLI-friendly rewrite of ``profile_based_classification.py``.

This is a **copy**, not an edit, of the original research script. The original
is left untouched; this module extracts its actually-used logic (the
``CloudProcess`` sequence-detection algorithm and the per-day processing
pipeline) and makes the parts that were hardcoded module-level constants
(``CLOUD_TYPES``, ``TARG_BET_CLOUD``, ``TARG_TO_FILTER``, ``TARG_TO_GET_BELLOW``,
``HYDRO_TYPES``, the ``NBINS_*`` thresholds, ...) loadable from a JSON
"hydrometeor sequence" config file instead.

What was intentionally dropped from the original when copying it here:
  * the unconditional module-level driver script (it ran on import!) —
    replaced by the explicit ``run()`` function below.
  * dead code never actually called in the default pipeline: the four large
    ``plot_cloud_*``/``plot_variable_inside_cloud`` figure functions (only
    ``plot_cloud_mask_2`` and the inline summary plot in ``process_cloud_data``
    were ever invoked), ``mask_target``, ``count_sublists_larger_than_1``,
    ``flatten_list_with_itertools``, ``group_by_month``, ``rechunk_dataset``,
    ``get_total_folder_size``, ``break_into_sequences``, ``plot_chirp_intervals``.
  * the ``process_for_specific_analysis`` branch (regenerating cloudnet
    products with custom DER parameters and comparing against an
    institution-specific NAS path for MWR data). That is a one-off research
    comparison, not part of the classification itself; it is not exposed here.
  * the interactive ``breakpoint()`` call left in the original after the
    summary plot.

Everything else -- the ``CloudProcess`` class, ``groupSequence``,
``compare_radar_chirp_configurations``, the file-loading/merging steps -- is
carried over with the same behaviour, only reorganised into callable
functions driven by a config object instead of module globals.
"""
from __future__ import annotations

import datetime
import glob
import json
import logging
import operator
import os
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

from phd_clouds.constants import (
    AERO_NO_CLOUD,
    AERO_WITH_INSECT_NO_CLOUD,
    CLASSIFICATION_TICK_LABELS,
    CLEAR_SKY,
    CLOUD_LIQUID,
    DRIZZLE_OR_RAIN,
    DRIZZLE_OR_RAIN_LIQUID_DROPLETS,
    GRANADA_ALTITUDE,
    ICE_PARTICLES,
    ICE_WITH_SUP_WATER,
    INSECT_NO_CLOUD,
    MELTING_ICE,
    MELTING_ICE_LIQUID_DROPLETS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw Cloudnet target-classification codes. These are fixed by the Cloudnet
# product itself, not something a "hydrometeor sequence" config should
# redefine -- the config instead says how to *combine* these fixed codes.
# ---------------------------------------------------------------------------
CLASSIFICATION_CODES: Dict[str, int] = {
    "CLEAR_SKY": CLEAR_SKY,
    "CLOUD_LIQUID": CLOUD_LIQUID,
    "DRIZZLE_OR_RAIN": DRIZZLE_OR_RAIN,
    "DRIZZLE_OR_RAIN_LIQUID_DROPLETS": DRIZZLE_OR_RAIN_LIQUID_DROPLETS,
    "ICE_PARTICLES": ICE_PARTICLES,
    "ICE_WITH_SUP_WATER": ICE_WITH_SUP_WATER,
    "MELTING_ICE": MELTING_ICE,
    "MELTING_ICE_LIQUID_DROPLETS": MELTING_ICE_LIQUID_DROPLETS,
    "AERO_NO_CLOUD": AERO_NO_CLOUD,
    "INSECT_NO_CLOUD": INSECT_NO_CLOUD,
    "AERO_WITH_INSECT_NO_CLOUD": AERO_WITH_INSECT_NO_CLOUD,
}


def _codes(names: Sequence[str]) -> List[int]:
    try:
        return [CLASSIFICATION_CODES[name] for name in names]
    except KeyError as exc:
        raise ValueError(
            f"Unknown classification code name {exc}. Valid names: {sorted(CLASSIFICATION_CODES)}"
        ) from None


# ---------------------------------------------------------------------------
# Default hydrometeor-sequence definition == exactly the behaviour that was
# hardcoded in profile_based_classification.py (CLOUD_TYPES, TARG_BET_CLOUD,
# TARG_TO_FILTER, TARG_TO_GET_BELLOW, CLOUD_PHASE, HYDRO_TYPES, TARG_BET_HYDRO,
# NBINS_*, THRESHOLD_LWP). Dumped verbatim to
# hydrometeor_configs/profile_hydrometeor_sequences.example.json.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "nbins_between_cloud": 5,
    "nbins_between_hydro": 1,
    "nbins_cloud": 3,
    "nbins_get_rain": 20,
    "threshold_lwp_g_m2": 5000,
    "no_hydrometeor_codes": ["CLEAR_SKY", "AERO_NO_CLOUD", "INSECT_NO_CLOUD", "AERO_WITH_INSECT_NO_CLOUD"],
    "cloud_types": {
        "Liquid": {
            "codes": ["CLOUD_LIQUID", "DRIZZLE_OR_RAIN_LIQUID_DROPLETS"],
            "phase": "single_phase",
            "bridge_codes": ["CLEAR_SKY", "AERO_NO_CLOUD", "INSECT_NO_CLOUD", "AERO_WITH_INSECT_NO_CLOUD"],
            "filter_species": [
                {"code": "DRIZZLE_OR_RAIN", "gap_below_bins": 10000, "gap_above_bins": 200},
                {"code": "ICE_PARTICLES", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "ICE_WITH_SUP_WATER", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "MELTING_ICE", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "MELTING_ICE_LIQUID_DROPLETS", "gap_below_bins": 5, "gap_above_bins": 5},
            ],
            "extend_base_through": None,
        },
        "Ice": {
            "codes": ["ICE_PARTICLES"],
            "phase": "single_phase",
            "bridge_codes": ["CLEAR_SKY", "CLOUD_LIQUID", "AERO_NO_CLOUD", "INSECT_NO_CLOUD", "AERO_WITH_INSECT_NO_CLOUD"],
            "filter_species": [
                {"code": "CLOUD_LIQUID", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "DRIZZLE_OR_RAIN", "gap_below_bins": 10000, "gap_above_bins": 200},
                {"code": "DRIZZLE_OR_RAIN_LIQUID_DROPLETS", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "ICE_WITH_SUP_WATER", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "MELTING_ICE", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "MELTING_ICE_LIQUID_DROPLETS", "gap_below_bins": 5, "gap_above_bins": 5},
            ],
            "extend_base_through": None,
        },
        "Mixed_phase": {
            "codes": ["CLOUD_LIQUID", "ICE_PARTICLES", "ICE_WITH_SUP_WATER", "MELTING_ICE", "MELTING_ICE_LIQUID_DROPLETS"],
            "phase": "mixed_phase",
            "bridge_codes": ["CLEAR_SKY", "DRIZZLE_OR_RAIN", "AERO_NO_CLOUD", "INSECT_NO_CLOUD", "AERO_WITH_INSECT_NO_CLOUD"],
            "filter_species": [
                {"code": "DRIZZLE_OR_RAIN", "gap_below_bins": 10000, "gap_above_bins": 200},
                {"code": "DRIZZLE_OR_RAIN_LIQUID_DROPLETS", "gap_below_bins": 5, "gap_above_bins": 5},
            ],
            "extend_base_through": None,
        },
        "Pre_liquid": {
            "codes": ["CLOUD_LIQUID", "DRIZZLE_OR_RAIN_LIQUID_DROPLETS"],
            "phase": "single_phase",
            "bridge_codes": ["CLEAR_SKY", "AERO_NO_CLOUD", "INSECT_NO_CLOUD", "AERO_WITH_INSECT_NO_CLOUD"],
            "filter_species": [
                {"code": "ICE_PARTICLES", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "ICE_WITH_SUP_WATER", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "MELTING_ICE", "gap_below_bins": 5, "gap_above_bins": 5},
                {"code": "MELTING_ICE_LIQUID_DROPLETS", "gap_below_bins": 5, "gap_above_bins": 5},
            ],
            "extend_base_through": {"codes": ["DRIZZLE_OR_RAIN"], "max_gap_bins": 20},
        },
        "Pre_mixed_phase": {
            "codes": ["CLOUD_LIQUID", "ICE_PARTICLES", "ICE_WITH_SUP_WATER", "MELTING_ICE", "MELTING_ICE_LIQUID_DROPLETS"],
            "phase": "mixed_phase",
            "bridge_codes": ["CLEAR_SKY", "DRIZZLE_OR_RAIN", "AERO_NO_CLOUD", "INSECT_NO_CLOUD", "AERO_WITH_INSECT_NO_CLOUD"],
            "filter_species": [],
            "extend_base_through": {"codes": ["DRIZZLE_OR_RAIN", "DRIZZLE_OR_RAIN_LIQUID_DROPLETS"], "max_gap_bins": 20},
        },
    },
    "hydro_types": {
        "Liquid": ["CLOUD_LIQUID", "DRIZZLE_OR_RAIN", "DRIZZLE_OR_RAIN_LIQUID_DROPLETS"],
        "Ice": ["ICE_PARTICLES"],
        "Mixed_phase": ["ICE_WITH_SUP_WATER", "MELTING_ICE", "MELTING_ICE_LIQUID_DROPLETS"],
        "Total": [
            "CLOUD_LIQUID", "DRIZZLE_OR_RAIN", "DRIZZLE_OR_RAIN_LIQUID_DROPLETS",
            "ICE_PARTICLES", "ICE_WITH_SUP_WATER", "MELTING_ICE", "MELTING_ICE_LIQUID_DROPLETS",
        ],
    },
}


@dataclass
class CloudTypeSpec:
    name: str
    codes: List[int]
    phase: str
    bridge_codes: List[int]
    filter_species: List[Dict[str, int]]  # [{"code": int, "gap_below_bins": int, "gap_above_bins": int}]
    extend_base_through: Optional[Dict[str, Any]]


@dataclass
class HydrometeorSequenceConfig:
    nbins_between_cloud: int
    nbins_between_hydro: int
    nbins_cloud: int
    nbins_get_rain: int
    threshold_lwp_g_m2: float
    no_hydrometeor_codes: List[int]
    cloud_types: Dict[str, CloudTypeSpec]
    hydro_types: Dict[str, List[int]]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "HydrometeorSequenceConfig":
        cloud_types = {}
        for name, spec in raw["cloud_types"].items():
            extend = spec.get("extend_base_through")
            cloud_types[name] = CloudTypeSpec(
                name=name,
                codes=_codes(spec["codes"]),
                phase=spec["phase"],
                bridge_codes=_codes(spec["bridge_codes"]),
                filter_species=[
                    {
                        "code": CLASSIFICATION_CODES[fs["code"]],
                        "gap_below_bins": fs.get("gap_below_bins", raw["nbins_between_cloud"]),
                        "gap_above_bins": fs.get("gap_above_bins", raw["nbins_between_cloud"]),
                    }
                    for fs in spec.get("filter_species", [])
                ],
                extend_base_through=(
                    {"codes": _codes(extend["codes"]), "max_gap_bins": extend["max_gap_bins"]}
                    if extend
                    else None
                ),
            )
        hydro_types = {name: _codes(codes) for name, codes in raw["hydro_types"].items()}
        return cls(
            nbins_between_cloud=raw["nbins_between_cloud"],
            nbins_between_hydro=raw["nbins_between_hydro"],
            nbins_cloud=raw["nbins_cloud"],
            nbins_get_rain=raw["nbins_get_rain"],
            threshold_lwp_g_m2=raw["threshold_lwp_g_m2"],
            no_hydrometeor_codes=_codes(raw["no_hydrometeor_codes"]),
            cloud_types=cloud_types,
            hydro_types=hydro_types,
        )

    @classmethod
    def load(cls, path: Optional[str]) -> "HydrometeorSequenceConfig":
        if path is None:
            return cls.from_dict(DEFAULT_CONFIG)
        with open(path) as f:
            raw = json.load(f)
        return cls.from_dict(raw)


@dataclass
class RunOptions:
    output_dir: Optional[str] = None
    fig_dir: Optional[str] = None
    save_products: bool = False
    make_plots: bool = False


# ---------------------------------------------------------------------------
# Helpers carried over from the original file (unchanged behaviour)
# ---------------------------------------------------------------------------
def groupSequence(lst, cloud_bins=0):
    res = [[lst[0]]]
    for i in range(1, len(lst)):
        if lst[i - 1] + 1 == lst[i]:
            res[-1].append(lst[i])
        else:
            res.append([lst[i]])
    return [sublist for sublist in res if len(sublist) >= cloud_bins]


def round_datetimeindex_to_seconds(datetimeindex):
    return datetimeindex.round("1s")


def common_prefix_of_filenames(paths: Sequence[str], extension: str) -> pd.DatetimeIndex:
    filenames_per_path = []
    for path in paths:
        filenames = [file[:8] for file in os.listdir(path) if file.endswith(extension)]
        filenames_per_path.append(set(filenames))
    common_prefixes = set.intersection(*filenames_per_path) if filenames_per_path else set()
    datetime_prefixes = []
    for prefix in common_prefixes:
        try:
            datetime_prefixes.append(datetime.datetime.strptime(prefix, "%Y%m%d"))
        except ValueError:
            pass
    datetime_prefixes.sort()
    return pd.DatetimeIndex(datetime_prefixes)


def compare_radar_chirp_configurations(start_date, end_date, database_intersection, path_radar):
    """Group dates by radar chirp/range-resolution configuration (unchanged from original)."""
    valid_dates = [date for date in database_intersection if start_date <= date <= end_date]
    resolution_dict: Dict[Any, List[datetime.datetime]] = {}
    height_dict: Dict[Any, np.ndarray] = {}
    current_resolution = None

    for date in valid_dates:
        radar_file_pattern = date.strftime("%Y%m%d") + "_granada_rpg-fmcw-94*.nc"
        matches = glob.glob(os.path.join(path_radar, radar_file_pattern))
        if not matches:
            logger.warning("No radar file found for %s, skipping from chirp grouping", date)
            continue
        radar = nc.Dataset(matches[0])
        new_resolution = tuple(radar["range_resolution"][:])

        if current_resolution is None:
            current_resolution = new_resolution

        key = new_resolution if new_resolution != current_resolution else current_resolution
        if key not in resolution_dict:
            resolution_dict[key] = []
            height_dict[key] = radar["range"][:]
        resolution_dict[key].append(date)
        current_resolution = new_resolution

    return resolution_dict, height_dict


def handle_serialization(column):
    if isinstance(column, np.ma.MaskedArray):
        return column.filled(np.nan).tolist()
    return column


def check_time_resolution(classification, categorize, date):
    try:
        if (
            classification.dimensions["time"].size == categorize.dimensions["time"].size
            and sum(categorize["time"][:] == classification["time"][:]) == categorize.dimensions["time"].size
        ):
            time_auxiliary = [
                date.strftime("%Y%m%d") + " " + str(datetime.timedelta(hours=float(h)))
                for h in categorize["time"]
            ]
            return round_datetimeindex_to_seconds(pd.to_datetime(time_auxiliary))
        logger.warning("classification and categorize files with different time resolution - %s", date)
        return None
    except Exception as exc:  # noqa: BLE001 - mirrors original defensive behaviour
        logger.error("An error occurred while checking time resolution for %s: %s", date, exc)
        return None


def generate_cloudnet_products(date, path_cate, path_cloudnet_lwc=None, path_cloudnet_iwc=None,
                                path_cloudnet_der=None, path_cloudnet_ier=None, params_der=None):
    """Regenerate LWC/IWC/DER/IER products locally from categorize via cloudnetpy (optional, opt-in)."""
    from cloudnetpy.products import generate_der, generate_iwc, generate_lwc, generate_ier

    cate_path = os.path.join(path_cate, date.strftime("%Y%m%d") + "_granada_categorize.nc")
    if path_cloudnet_lwc is not None:
        generate_lwc(cate_path, os.path.join(path_cloudnet_lwc, date.strftime("%Y%m%d") + "_granada_lwc.nc"))
    if path_cloudnet_iwc is not None:
        generate_iwc(cate_path, os.path.join(path_cloudnet_iwc, date.strftime("%Y%m%d") + "_granada_iwc.nc"))
    if path_cloudnet_der is not None:
        out = os.path.join(path_cloudnet_der, date.strftime("%Y%m%d") + "_granada_der.nc")
        generate_der(cate_path, out, parameters=params_der) if params_der is not None else generate_der(cate_path, out)
    if path_cloudnet_ier is not None:
        generate_ier(cate_path, os.path.join(path_cloudnet_ier, date.strftime("%Y%m%d") + "_granada_ier.nc"))


# ---------------------------------------------------------------------------
# Core algorithm: unchanged from the original CloudProcess class.
# ---------------------------------------------------------------------------
class CloudProcess:
    def __init__(self, df_class, cloud_value, no_cloud_target_values, consecutive_bins,
                 cloud_type="single_phase", bins_cloud=0):
        self.classification = df_class
        self.height = df_class.columns
        self.time = df_class.index
        self.cloud = cloud_value
        self.cloud_type = cloud_type
        self.no_cloud_targ = no_cloud_target_values
        self.consecutive_bins = consecutive_bins
        self.nbins_cloud = bins_cloud

        self.cloud_indexes()
        self.cloud_boundaries()

    def cloud_indexes(self):
        conditions = [self.classification == elem for elem in self.cloud]
        merged_conditions = reduce(operator.or_, conditions)
        self.row_cloud, self.col_cloud = np.where(merged_conditions)

        self.row_cloud_unique = np.unique(self.row_cloud)
        self.col_cloud_boundaries = self.time.shape[0] * [np.nan]
        if self.cloud_type == "mixed_phase":
            remove_unique = []
            for i, i_time in enumerate(self.row_cloud_unique):
                sequence_cloud = groupSequence(self.col_cloud[self.row_cloud == i_time], self.nbins_cloud)
                sequence_cloud = self.new_sequence(i_time, self.no_cloud_targ, self.consecutive_bins, sequence_cloud)
                n_layer = len(sequence_cloud)
                count = 0
                remove_position = []
                for j, sublist in enumerate(sequence_cloud):
                    aux_classification = self.classification.iloc[i_time, sublist]
                    if self.are_all_elements_equal(aux_classification, ICE_PARTICLES) or \
                            self.are_all_elements_equal(aux_classification, CLOUD_LIQUID):
                        remove_position.append(j)
                        count += 1
                if count > 0:
                    self.remove_elements_using_pop(sequence_cloud, remove_position)
                if count < n_layer:
                    self.col_cloud_boundaries[i_time] = sequence_cloud
                if count == n_layer:
                    remove_unique.append(i)
            self.row_cloud_unique = np.delete(self.row_cloud_unique, remove_unique)
        else:
            for i, i_time in enumerate(self.row_cloud_unique):
                sequence_cloud = groupSequence(self.col_cloud[self.row_cloud == i_time], self.nbins_cloud)
                self.col_cloud_boundaries[i_time] = self.new_sequence(
                    i_time, self.no_cloud_targ, self.consecutive_bins, sequence_cloud
                )

    def new_sequence(self, time, targets, nbins, sequence_cloud):
        n_sequence = len(sequence_cloud)
        i = 0
        while i < n_sequence - 1 and n_sequence > 1:
            ini = sequence_cloud[i][-1]
            end = sequence_cloud[i + 1][0]
            hole = self.classification.iloc[time, ini + 1:end]
            seq_bins, ind_i, ind_f = self.count_consecutive(targets, hole)
            if seq_bins > nbins:
                i += 1
            else:
                new_layer = sorted(sequence_cloud[i] + sequence_cloud[i + 1])
                sequence_cloud.pop(i)
                sequence_cloud.pop(i)
                sequence_cloud.insert(i, new_layer)
                n_sequence -= 1
        return sequence_cloud

    def cloud_boundaries(self):
        self.cloud_base = []
        self.cloud_top = []
        for i_time in self.row_cloud_unique:
            self.cloud_base.append([sublist[0] for sublist in self.col_cloud_boundaries[i_time]])
            self.cloud_top.append([sublist[-1] for sublist in self.col_cloud_boundaries[i_time]])
        self.time_cbt = self.time[self.row_cloud_unique]

    def are_all_elements_equal(self, df, value):
        return (df == value).all().all()

    def remove_elements_using_pop(self, lst, positions):
        positions.sort(reverse=True)
        for pos in positions:
            lst.pop(pos)

    def cloud_mask(self):
        mask = pd.DataFrame(False, index=self.classification.index, columns=self.classification.columns)
        for i_time in self.row_cloud_unique:
            for sublist in self.col_cloud_boundaries[i_time]:
                mask.iloc[i_time, sublist] = True
        return mask

    def calculate_cloud_properties(self, height_cloud_base, height_cloud_top, height_cloud_mean,
                                    geometric_cloud_thickness, cloud_type: str) -> None:
        for i, time_cloud in enumerate(self.time_cbt):
            h_cb = self.height[self.cloud_base[i]]
            h_ct = self.height[self.cloud_top[i]]
            height_cloud_base.loc[time_cloud, cloud_type] = h_cb
            height_cloud_top.loc[time_cloud, cloud_type] = h_ct
            height_cloud_mean.loc[time_cloud, cloud_type] = (h_cb + h_ct) / 2
            geometric_cloud_thickness.loc[time_cloud, cloud_type] = (h_ct - h_cb)

    def filter_species(self, specie, dzb_max, dzt_max):
        row_specie, col_specie = np.where(self.classification == specie)
        intersection = np.intersect1d(self.row_cloud_unique, row_specie)

        if intersection.any():
            no_candidate = -1.0 * self.classification.shape[1]
            for i_time in intersection:
                sequence_cloud = self.col_cloud_boundaries[i_time]
                sequence_specie = groupSequence(col_specie[row_specie == i_time])

                for cloud_thickness in sequence_cloud:
                    dbl = no_candidate
                    dab = abs(no_candidate)
                    ibl = np.nan
                    iab = np.nan
                    for specie_thickness in sequence_specie:
                        dnew = specie_thickness[-1] - cloud_thickness[0]
                        if dnew < 0 and dnew > dbl:
                            dbl = dnew
                            ibl = specie_thickness[-1]
                        dnew = specie_thickness[0] - cloud_thickness[-1]
                        if dnew > 0 and dnew < dab:
                            dab = dnew
                            iab = specie_thickness[0]

                    if sum(self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1] + 1] == specie) > 0.0:
                        self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1] + 1] = np.nan

                    if not np.isnan(ibl):
                        if (cloud_thickness[0] - ibl) < dzb_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1] + 1] = np.nan
                    if not np.isnan(iab):
                        if (iab - cloud_thickness[-1]) < dzt_max:
                            self.classification.iloc[i_time, cloud_thickness[0]:cloud_thickness[-1] + 1] = np.nan

        self.cloud_indexes()
        self.cloud_boundaries()

    def get_specie_below(self, specie, max_bins):
        if np.size(specie) == 1:
            row_specie, col_specie = np.where(self.classification == specie[0])
        else:
            conditions = [self.classification == elem for elem in specie]
            merged_conditions = reduce(operator.or_, conditions)
            row_specie, col_specie = np.where(merged_conditions)

        intersection = np.intersect1d(self.row_cloud_unique, row_specie)
        mask = np.full(self.classification.shape[0], False)
        mask[intersection] = True
        self.classification.loc[~mask, :] = np.nan

        if intersection.any():
            for i_time in intersection:
                sequence_cloud = self.col_cloud_boundaries[i_time]
                sequence_specie = groupSequence(col_specie[row_specie == i_time])
                threshold = 0
                for cloud_thickness in sequence_cloud:
                    bin_cbase = cloud_thickness[0]
                    bin_ctop = cloud_thickness[-1]
                    closest_bin = 0
                    for specie_layer in sequence_specie:
                        if specie_layer[-1] < bin_cbase and specie_layer[-1] < threshold:
                            self.classification.iloc[i_time, threshold:bin_ctop + 1] = np.nan
                        elif specie_layer[-1] < bin_cbase:
                            closest_bin = specie_layer[-1]
                        else:
                            self.classification.iloc[i_time, threshold:bin_ctop + 1] = np.nan
                    if bin_cbase - closest_bin > max_bins:
                        self.classification.iloc[i_time, threshold + 1:bin_ctop + 1] = np.nan
                    threshold = bin_ctop

        self.cloud_indexes()
        self.cloud_boundaries()

    def count_consecutive(self, numbers_set, sample):
        mask = [1 if num in numbers_set else 0 for num in sample]
        current_count = 0
        max_count = 0
        start_index = 0
        end_index = 0
        for i, value in enumerate(mask):
            if value:
                if current_count == 0:
                    start_index = i
                current_count += 1
                if current_count > max_count:
                    max_count = current_count
                    end_index = i
            else:
                current_count = 0
        return max_count, start_index, end_index


# ---------------------------------------------------------------------------
# Per-day pipeline
# ---------------------------------------------------------------------------
CLOUD_COLUMNS = ["Liquid", "Ice", "Mixed_phase", "Pre_liquid", "Pre_mixed_phase"]


@dataclass
class DayPaths:
    classification: str
    categorize: str
    radar: str
    lwc: str
    iwc: str
    der: str
    ier: str


def load_single_day(date: datetime.datetime, paths: DayPaths, height: np.ndarray, chirp_res,
                     site: str = "granada"):
    """Open the raw Cloudnet files needed for one day. Raises FileNotFoundError with a
    clear message instead of the original's silent ``glob(...)[0]`` IndexError.

    ``height``/``chirp_res`` come from ``compare_radar_chirp_configurations`` (grouping
    dates by radar range-resolution, as the original ``process_cloud_data_parallel`` did)
    rather than being recomputed per day, since the radar range grid -- not
    ``categorize['height']`` -- is what the classification/reflectivity DataFrames are
    indexed by.
    """
    datestr = date.strftime("%Y%m%d")

    def _find_one(folder: str, pattern: str) -> str:
        matches = glob.glob(os.path.join(folder, pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching '{pattern}' found in '{folder}' for {datestr}")
        return matches[0]

    radar_path = _find_one(paths.radar, f"{datestr}_{site}_rpg-fmcw-94*.nc")
    categorize = nc.Dataset(_find_one(paths.categorize, f"{datestr}_{site}_categorize.nc"))
    classification = nc.Dataset(_find_one(paths.classification, f"{datestr}_{site}_classification.nc"))
    radar = xr.open_dataset(radar_path)

    time = check_time_resolution(classification, categorize, date)
    if time is None:
        raise ValueError(f"Could not align classification/categorize time for {datestr}")

    if not np.array_equal(radar["range_resolution"].values, chirp_res):
        logger.warning("Radar range resolution differs from the chirp group's resolution for %s", datestr)
    if categorize["height"][:].size != height.size:
        logger.warning("Radar and categorize files have different height-bin counts for %s: %s vs %s",
                        datestr, categorize["height"][:].size, height.size)

    cloudnet_lwc = xr.open_dataset(_find_one(paths.lwc, f"{datestr}_{site}_lwc-scaled-adiabatic.nc"))
    cloudnet_iwc = xr.open_dataset(_find_one(paths.iwc, f"{datestr}_{site}_iwc-Z-T-method.nc"))
    cloudnet_der = xr.open_dataset(_find_one(paths.der, f"{datestr}_{site}_der.nc"))
    cloudnet_ier = xr.open_dataset(_find_one(paths.ier, f"{datestr}_{site}_ier.nc"))

    cloud_products = xr.merge([cloudnet_lwc, cloudnet_iwc, cloudnet_der, cloudnet_ier],
                               compat="no_conflicts", join="exact")
    cloud_products = cloud_products.reindex(time=time, method="nearest", tolerance="5s")
    cloud_products.coords["height"] = height

    df_reflectivity = pd.DataFrame(data=categorize["Z"][:], index=time, columns=height)
    df_classification = pd.DataFrame(data=classification["target_classification"][:], index=time, columns=height)

    ds_integrated_variables = xr.Dataset(
        {
            "LWP": (["time"], categorize["lwp"][:]),
            "IWP": (["time"], np.trapz(cloud_products.iwc.values, height, axis=1)),
        },
        coords={"time": time},
    )
    cloud_physical_properties = xr.merge([cloud_products, ds_integrated_variables], join="exact")

    xr_radar_variables = xr.merge(
        [
            xr.DataArray(categorize["Z"][:], coords={"time": time, "range": height}, name="Z"),
            xr.DataArray(categorize["v"][:], coords={"time": time, "range": height}, name="v"),
        ],
        compat="no_conflicts",
        join="exact",
    )

    return {
        "date": date,
        "categorize": categorize,
        "df_reflectivity": df_reflectivity,
        "df_classification": df_classification,
        "cloud_physical_properties": cloud_physical_properties,
        "xr_radar_variables": xr_radar_variables,
        "time": time,
        "height": height,
    }


def classify_single_day(loaded: Dict[str, Any], config: HydrometeorSequenceConfig, options: RunOptions,
                         nchirp: int = 0) -> Dict[str, Any]:
    """Config-driven equivalent of the original ``process_cloud_data``."""
    date = loaded["date"]
    time = loaded["time"]
    height = loaded["height"]
    df_classification = loaded["df_classification"]
    cloud_physical_properties = loaded["cloud_physical_properties"]

    lwp = loaded["categorize"]["lwp"][:]
    if np.any(lwp > config.threshold_lwp_g_m2):
        lwp[lwp > config.threshold_lwp_g_m2] = np.nan

    cloud_int = {"No Cloud": 0, "Liquid": 1, "Pre_liquid": 2, "Ice": 3, "Mixed_phase": 4, "Pre_mixed_phase": 5}
    array_cloud = np.zeros((df_classification.index.shape[0], df_classification.columns.shape[0]))

    number_of_layers = pd.DataFrame(index=time, columns=CLOUD_COLUMNS)
    height_cloud_base = pd.DataFrame(index=time, columns=CLOUD_COLUMNS)
    height_cloud_top = pd.DataFrame(index=time, columns=CLOUD_COLUMNS)
    height_cloud_mean = pd.DataFrame(index=time, columns=CLOUD_COLUMNS)
    geometric_cloud_thickness = pd.DataFrame(index=time, columns=CLOUD_COLUMNS)

    all_cloud_base = pd.DataFrame(index=number_of_layers.index, columns=number_of_layers.columns)
    all_cloud_top = pd.DataFrame(index=number_of_layers.index, columns=number_of_layers.columns)
    total_cloud_mask = np.full(df_classification.shape, False)

    for cloud_name, spec in config.cloud_types.items():
        classification_filter = CloudProcess(
            df_classification.copy(), spec.codes, spec.bridge_codes, config.nbins_between_cloud,
            spec.phase, config.nbins_cloud,
        )
        for fs in spec.filter_species:
            classification_filter.filter_species(fs["code"], fs["gap_below_bins"], fs["gap_above_bins"])

        if spec.extend_base_through is not None:
            classification_filter.get_specie_below(
                spec.extend_base_through["codes"], spec.extend_base_through["max_gap_bins"]
            )

        classification_filter.calculate_cloud_properties(
            height_cloud_base, height_cloud_top, height_cloud_mean, geometric_cloud_thickness, cloud_name
        )

        number_of_layers.loc[classification_filter.time_cbt, cloud_name] = [
            len(sublist) for sublist in classification_filter.cloud_base
        ]
        total_cloud_mask = np.logical_or(total_cloud_mask, classification_filter.cloud_mask().to_numpy())
        array_cloud[classification_filter.cloud_mask().to_numpy()] = cloud_int[cloud_name]

        mask_cloud_base = number_of_layers.loc[classification_filter.time_cbt, cloud_name] == 1
        all_cloud_base.loc[classification_filter.time_cbt[mask_cloud_base], cloud_name] = \
            height_cloud_base.loc[classification_filter.time_cbt[mask_cloud_base], cloud_name].astype(float)
        all_cloud_top.loc[classification_filter.time_cbt[mask_cloud_base], cloud_name] = \
            height_cloud_top.loc[classification_filter.time_cbt[mask_cloud_base], cloud_name].astype(float)

    ds_cloud = xr.Dataset(data_vars={"cloud_classification": (["time", "height"], array_cloud)},
                           coords={"time": time, "height": height})

    if options.make_plots:
        _plot_summary(date, ds_cloud, all_cloud_base, all_cloud_top, options)

    ds_hydrometeor = xr.Dataset(coords={"time": time, "range": height})
    for hydro_name, codes in config.hydro_types.items():
        classification_filter = CloudProcess(df_classification.copy(), codes, config.no_hydrometeor_codes,
                                              config.nbins_between_hydro)
        ds_hydrometeor[hydro_name] = xr.DataArray(classification_filter.cloud_mask(), dims=("time", "range"),
                                                    coords={"time": time, "range": height})

    number_of_layers.index.name = "time"
    ds_layers = xr.Dataset.from_dataframe(number_of_layers)

    ds_mask = xr.Dataset({"cloud_mask": (["time", "range"], total_cloud_mask)},
                          coords={"time": df_classification.index, "range": df_classification.columns})
    xr_radar_variables = loaded["xr_radar_variables"].copy()
    xr_radar_variables["Z"] = xr_radar_variables["Z"].where(ds_mask.cloud_mask)
    xr_radar_variables["v"] = xr_radar_variables["v"].where(ds_mask.cloud_mask)

    result = {
        "ds_cloud_classification": ds_cloud,
        "ds_hydrometeor": ds_hydrometeor,
        "ds_layers": ds_layers,
        "xr_radar_variables": xr_radar_variables,
        "cloud_physical_properties": cloud_physical_properties,
        "height_cloud_base": height_cloud_base,
        "height_cloud_top": height_cloud_top,
        "height_cloud_mean": height_cloud_mean,
        "geometric_cloud_thickness": geometric_cloud_thickness,
    }

    if options.save_products and options.output_dir:
        _save_products(date, nchirp, result, options.output_dir)

    return result


def _plot_summary(date, ds_cloud, all_cloud_base, all_cloud_top, options: RunOptions) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    cloud_category = ["No Cloud or Removed for Class.", "Liquid", "Pre_liquid", "Ice", "Mixed_phase", "Pre_mixed_phase"]
    list_cloud_colors = ["#FFFFFF", "#007CFF", "blue", "cyan", "yellow", "orange"]
    cloud_cmap = plt.cm.colors.ListedColormap(list_cloud_colors)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.02], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    pc0 = ax.pcolormesh(ds_cloud["time"], (ds_cloud["height"] + GRANADA_ALTITUDE) / 1e3,
                         ds_cloud["cloud_classification"].T, cmap=cloud_cmap, vmin=0, vmax=len(cloud_category))
    for cloud in all_cloud_base.columns:
        ax.scatter(all_cloud_base.index, (all_cloud_base[cloud] + GRANADA_ALTITUDE) / 1e3, color="red", s=3)
        ax.scatter(all_cloud_top.index, (all_cloud_top[cloud] + GRANADA_ALTITUDE) / 1e3, color="black", s=3)

    ax.set_ylabel("Height (km) a.s.l")
    ax.set_xlabel("Time (UTC)")
    ax.grid(True, linestyle=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim([ds_cloud["time"].values[0], ds_cloud["time"].values[-1]])
    ax.set_ylim([0, (ds_cloud["height"].values[-1] + GRANADA_ALTITUDE) / 1e3])

    cax_scat = fig.add_subplot(gs[0, 1])
    cbar_scat = plt.colorbar(pc0, cax=cax_scat, ticks=[], orientation="vertical")
    for idx, (color, name) in enumerate(zip(cloud_cmap.colors, cloud_category)):
        rect = plt.Rectangle((0, idx), 1, 1, color=color)
        cbar_scat.ax.add_patch(rect)
        cbar_scat.ax.text(1.5, idx + 0.5, name, color="black", va="center", fontsize=14)

    if options.fig_dir:
        os.makedirs(options.fig_dir, exist_ok=True)
        fig.savefig(
            os.path.join(options.fig_dir, date.strftime("%Y%m%d") + "_cloud_classification_profile_based.png"),
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)


def _save_products(date, nchirp, result: Dict[str, Any], output_dir: str) -> None:
    name_folders_nc = ["radar_variables", "cloud_physical_properties", "hydrometeor", "number_of_layers"]
    datasets = [result["xr_radar_variables"], result["cloud_physical_properties"],
                result["ds_hydrometeor"], result["ds_layers"]]
    for ds, name in zip(datasets, name_folders_nc):
        folder = os.path.join(output_dir, f"chirp_{nchirp}", name)
        os.makedirs(folder, exist_ok=True)
        ds.to_netcdf(os.path.join(folder, f"{date.strftime('%Y%m%d')}_{name}.nc"))

    json_frames = {
        "height_cloud_base": result["height_cloud_base"],
        "height_cloud_top": result["height_cloud_top"],
        "height_cloud_mean": result["height_cloud_mean"],
        "geometric_cloud_thickness": result["geometric_cloud_thickness"],
    }
    for name, df in json_frames.items():
        df = df.copy()
        df.index.name = "time"
        folder = os.path.join(output_dir, f"chirp_{nchirp}", name)
        os.makedirs(folder, exist_ok=True)
        try:
            df_serializable = df.applymap(handle_serialization)
            df_serializable.to_json(os.path.join(folder, f"{date.strftime('%Y%m%d')}_{name}.json"), orient="index")
        except Exception as exc:  # noqa: BLE001 - mirrors original defensive behaviour
            logger.error("Error while saving %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------
def run(dates: Sequence[datetime.datetime], paths: DayPaths, config: HydrometeorSequenceConfig,
        options: RunOptions, site: str = "granada") -> List[Dict[str, Any]]:
    """Process a list of dates with the profile-based method. Returns a list of
    ``{"date": ..., "status": "ok"|"failed", "error": Optional[str]}``."""
    intervals_dic, height_dic = compare_radar_chirp_configurations(min(dates), max(dates), dates, paths.radar)

    results = []
    for nchirp, key_res in enumerate(intervals_dic):
        height = height_dic[key_res]
        for date in intervals_dic[key_res]:
            try:
                loaded = load_single_day(date, paths, height, key_res, site=site)
                classify_single_day(loaded, config, options, nchirp=nchirp)
                results.append({"date": date, "status": "ok", "error": None})
                logger.info("Profile-based classification OK for %s", date.strftime("%Y-%m-%d"))
            except Exception as exc:  # noqa: BLE001 - one bad day must not stop the batch
                logger.error("Profile-based classification failed for %s: %s", date, exc)
                results.append({"date": date, "status": "failed", "error": str(exc)})

    processed_dates = {r["date"] for r in results}
    for date in dates:
        if date not in processed_dates:
            results.append({"date": date, "status": "failed", "error": "No radar file found for this date"})
    return results
