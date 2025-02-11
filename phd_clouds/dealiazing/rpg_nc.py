from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any
from matplotlib import dates, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import xarray as xr
from gfatpy.lidar.plot.quicklook import apply_gap_size
from gfatpy.radar.plot import RADAR_PLOT_INFO 

from pdb import set_trace

from gfatpy.radar.plot.utils import circular_grid
from gfatpy.radar.utils import (
    check_is_netcdf,
    enhance_rpgpy_dataset,
    ppi_to_cartessian,
    rhi_to_cartessian,
)
from gfatpy.radar.retrieve.retrieve import (
    add_all_products_from_LV0,
    add_all_products_from_LV1,
    retrieve_dBZe,
)
from gfatpy.utils.plot import color_list
from gfatpy.utils.utils import parse_datetime

# from gfatpy.radar.utils_deprecated import ppi_to_cartessian, rhi_to_cartessian


class rpg:
    """Class to handle RPG netcdf files."""

    def __init__(self, path: str | Path):
        """Constructor for the rpg class.

        Args:

            - path (Path): Path to the RPG netcdf file.
        """
        self.path = check_is_netcdf(path)
        self.type = self.path.name.split(".")[0].split("_")[-1]
        self.level = int(self.path.name.split(".")[-2][-1])
        self.raw = enhance_rpgpy_dataset(self.path) #TODO: load data only when needed
        self._data = None
        self._band = None

    @property
    def data(self) -> xr.Dataset:
        """Property to return the data from the RPG netcdf file.

        Returns:

            - xr.Dataset: The RPG netcdf file as an xarray dataset.
        """
        if self._data is None:
            self._data = self.add_all_products()
        return self._data

    @property
    def band(self) -> str:
        if self._band is None:
            if self.raw["radar_frequency"].values > 75:
                self._band = "W"
            else:
                self._band = "Ka"
        return self._band

    def add_all_products(self) -> xr.Dataset:
        """Method to add all products to the RPG netcdf file.

        Raises:

            - ValueError: If the level is not 0 or 1.

        Returns:

            - xr.Dataset: The RPG netcdf file with all products added.
        """

        if self.level == 0:
            data = add_all_products_from_LV0(self.raw)
        elif self.level == 1:
            data = add_all_products_from_LV1(self.raw, self.band)
        else:
            raise ValueError("Level must be 0 or 1.")
        return data

    def plot_spectrum(
        self, target_time: datetime | np.datetime64, target_range: float, **kwargs
    ) -> tuple[Figure, Path | None]:
        """Generates a plot of the doppler spectrum at a specific time and range.

        Args:

            - target_time (datetime | np.datetime64): The time for the plot.
            - target_range (float):  The range for the plot.

        Returns:

            - tuple[Figure, Path | None]: The figure handle and the path to the saved file (None if `savefig = False`).
        """

        fig = kwargs.get("fig", None) 
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 7))
        else:            
            ax = fig.get_axes()[0]

        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)

        time_str = f"{target_time:%H%M%S}"

        range_str = f"{target_range:.2f}"
        label_type = kwargs.get("label_type", "both")
        if label_type == "both":
            label = f"{time_str} | {range_str} m"
        elif label_type == "range":
            label = f"{range_str} m"
        else:
            label = f"{time_str}"

        data = self.data
        chirp_number = int(
            data["chirp_number"].sel(range=target_range, method="nearest").values.item()
        )
        sdata = data.sel(time=target_time, range=target_range, method="nearest")
        velocity_vectors = sdata["velocity_vectors"].sel(chirp=chirp_number).values
        sdata = sdata.assign_coords(spectrum=velocity_vectors)

        # Convert to dBZe
        sdata["doppler_spectrum_dBZe"] = retrieve_dBZe(sdata["doppler_spectrum"], self.band)
        sdata["doppler_spectrum_dBZe"].attrs = {
            "long_name": "Power density",
            "units": "dB",
        }
        if not np.isnan(sdata["doppler_spectrum_dBZe"].values).all():
            sdata["doppler_spectrum_dBZe"].plot(
                ax=ax, color=kwargs.get("color", "black"), label=label
            ) #type: ignore

        if kwargs.get("velocity_limits", None) is not None:
            ax.set_xlim(*kwargs.get("velocity_limits"))
        else:
            nyquist_velocity = data["nyquist_velocity"].sel(chirp=chirp_number).values.item()
            ax.set_xlim(-nyquist_velocity, nyquist_velocity)

        ax.set_xlabel("Doppler velocity, [m/s]")
        ax.set_ylabel("Power density, [dB]")
        ax.set_title(f"Time: {str(target_time).split('.')[0]}, Range: {target_range}")

        # Add vertical lines at 0
        ax.axvline(x=0, color="black", linestyle="--")
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)

        filepath = None
        if fig is not None:
            fig.tight_layout()
            output_dir = kwargs.get("output_dir", None)
            if output_dir is not None:
                filepath = output_dir / self.path.name.replace(
                    ".nc",
                    f"_spectrum_{target_time:%Y%m%dT%H%M}_{target_range}.png",
                )                
                fig.savefig(filepath, dpi=300)
        return fig, filepath

    def plot_spectra_by_time(
        self,
        target_range: float,
        time_slice: tuple[datetime, datetime] | list[datetime],
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Generates a plot of the doppler spectra at a specific range and time slice.

        Args:

            - target_range (float): range to plot the spectra.
            - time_slice (tuple[datetime, datetime]): time slice to plot the spectra
            - kwargs: additional arguments to pass to the plot_spectrum method (e.g., savefig | output_dir).

        Returns:
            - tuple[Figure, Path | None]: The figure handle and the path to the saved file (None if `savefig = False`).
        """
        original_time_slice = time_slice        
        if all(isinstance(t, np.datetime64) for t in original_time_slice):
            original_time_slice = (parse_datetime(original_time_slice[0]), parse_datetime(original_time_slice[1]))

        if isinstance(time_slice, list):
            _time_list = time_slice
            time_list = np.unique(
                [
                    self.data.sel(time=time_, method="nearest").time.values
                    for time_ in _time_list
                ]
            )
            time_slice = (time_list[0], time_list[-1])
            # Find the time_list in the data.time and create a new time_list with the values found using .sel(method="nearest")
        else:
            time_slice = time_slice
            time_list = self.data.time.sel(time=slice(*time_slice)).values

        data = self.data.copy()
        data = data.sel(range=target_range, method="nearest")
        data = data.sel(time=slice(*time_slice))
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))
        colors = color_list(len(time_list))
        if len(data.time) == 0:
            raise ValueError("No data found for the given time slice.")
        for idx, time_ in enumerate(time_list):
            self.plot_spectrum(
                time_,
                target_range,
                **{"color": colors[idx]},
                **{
                    "fig": fig,
                    "savefig": False,
                    "velocity_limits": kwargs.get("velocity_limits", None),
                    "label_type": "time",
                },
            )
        
        ax.set_title(
            f"Range: {target_range} | Period: {original_time_slice[0]:%Y%m%d} {original_time_slice[0]:%H:%M:%S} - {original_time_slice[-1]:%H:%M:%S}"
        )
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)

        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir", None)
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            filepath = output_dir / self.path.name.replace(
                ".nc",
                f"_spectra_{original_time_slice[0]:%Y%m%dT%H%M}_{original_time_slice[-1]:%Y%m%dT%H%M}_{target_range}.png",
            )
            fig.savefig(filepath, dpi=300)
        return fig, filepath

    def plot_spectra_by_range(
        self,
        target_time: datetime | np.datetime64,
        range_slice: tuple[float, float] | list[float],
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Generates a plot of the doppler spectra at a specific time and time slice.

        Args:

            - target_time (float): time to plot the spectra.
            - range_slice (tuple[float, float]): time slice to plot the spectra
            - kwargs: additional arguments to pass to the plot_spectrum method (e.g., savefig | output_dir).

        Returns:
            - tuple[Figure, Path | None]: The figure handle and the path to the saved file (None if `savefig = False`).
        
        """
        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)

        original_range_slice = range_slice
        if isinstance(range_slice, list):
            _range_list = range_slice
            range_list = np.unique(
                [
                    self.data.sel(range=range_, method="nearest").range.values
                    for range_ in _range_list
                ]
            )
            range_slice = (range_list[0], range_list[-1])
        else:
            range_slice = range_slice
            range_list = self.data.range.sel(range=slice(*range_slice)).values

        data = self.data.copy()
        data = data.sel(time=target_time, method="nearest")
        data = data.sel(range=slice(*range_slice))
        
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))
        colors = color_list(len(range_list))
        if len(data.range) == 0:
            raise ValueError("No data found for the given range slice.")
        for idx, range_ in enumerate(range_list):
            self.plot_spectrum(
                target_time,
                range_,
                **{"color": colors[idx]},
                **{
                    "fig": fig,
                    "savefig": False,
                    "velocity_limits": kwargs.get("velocity_limits", None),
                    "label_type": "range",
                },
            )
        default_title = f"Time: {target_time:%Y%m%dT%H:%M:%S} | Range: [{range_slice[0]/1e3:.2f} - {range_slice[1]/1e3:.2f}] km"
        ax.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize_axis", 12))
        ax.set_title(kwargs.get("title", default_title), fontdict={"fontsize": kwargs.get("fontsize_title", 12)})
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=kwargs.get("fontsize_legend", 12))
        ax.set_xlabel("Doppler velocity, [m/s]", fontsize=kwargs.get("fontsize_labels", 12))
        ax.set_ylabel("Power density, [dB]", fontsize=kwargs.get("fontsize_labels", 12))

        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir", None)
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            filepath = output_dir / self.path.name.replace(
                ".nc",
                f"_spectra_{target_time:%Y%m%dT%H%M%S}_{original_range_slice[0]:.1f}_{original_range_slice[-1]:.1f}.png",
            )
            fig.savefig(filepath, dpi=300)
        return fig, filepath

    def plot_2D_spectrum(
        self,
        target_time: np.datetime64 | datetime,
        range_limits: tuple[float, float] | None = None,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Generates a 2D plot of the doppler spectrum at a specific time.

        Args:
            target_time (np.datetime64 | datetime): The time for the plot.
            range_limits (tuple[float, float], optional): Range limits for the plot. Defaults (None) to the minimum and maximum range in the dataset.
            vmin (float, optional): _description_. Defaults to 0.
            vmax (float, optional): _description_. Defaults to 1.

        Returns:
            tuple[Figure, Path | None]: _description_
        """

        """
        Generates a range-velocity plot based on the dataset and a specific time.

        Args:

            - dataset (pandas.DataFrame): The dataset containing the required data.
            - time_str (str): The time string for the desired plot time (e.g., '20210913T112500.0').
            - min_value (float, optional): The minimum value to consider for the doppler_spectrum_dBZe plot. Default is 0.01.
            - power_spectrum_limits (tuple, optional): The limits for the power spectrum plot. Default sets limits to autoscale.

        Returns:

            - None
        """
        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)
        
        data = self.data
        if range_limits is None:
            range_limits = (data.range.min().values, data.range.max().values)
        range_limits = (range_limits[0] / 1e3, range_limits[1] / 1e3)

        data = data.assign_coords(range=data["range_layers"].values)
        data["range"] = data["range"] / 1e3
        data = data.sel(time=target_time, method="nearest")
        data = data.sel(range=slice(*range_limits))
        chirps = np.sort(np.unique(data["chirp_number"].values).astype(int))
        number_of_chirps = len(chirps)
        data = data.sel(chirp=chirps)

        data["doppler_spectrum_dBZe"] = retrieve_dBZe(data["doppler_spectrum"], self.band)
        data["doppler_spectrum_dBZe"].attrs = {
            "long_name": "Power density",
            "units": "dB",
        }

        chirp_info = {"range_limits": {}, "height_ratio": {}}

        for chirp_ in chirps:
            min_range = data.range[data["chirp_number"].values == chirp_].min().item()
            max_range = data.range[data["chirp_number"].values == chirp_].max().item()
            chirp_info["range_limits"][chirp_] = (min_range, max_range)
            chirp_info["height_ratio"][chirp_] = (
                100 * (max_range - min_range) / (range_limits[1] - range_limits[0])
            )

        height_ratios = np.flip(
            [chirp_info["height_ratio"][chirp_] for chirp_ in chirps]
        )

        fig, axes = plt.subplots(
            number_of_chirps,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": height_ratios},
        )  # Adjust height_ratios if chirps change
        if isinstance(axes, Axes):
            axes = np.array([axes])

        axes = np.flip(
            axes
        )  # Reverse the order of the axes to plot the highest chirp first
        vmin = kwargs.get("vmin", RADAR_PLOT_INFO["limits"]["dBZe"][0])
        vmax = kwargs.get("vmax", RADAR_PLOT_INFO["limits"]["dBZe"][1])
        cm = []
        for idx, chirp_ in enumerate(data.chirp.values):
            x_vals = data["velocity_vectors"].sel(chirp=chirp_).values
            nyquist_velocity = data["nyquist_velocity"].sel(chirp=chirp_).values
            # Plot the doppler_spectrum
            cm_ = axes[idx].imshow(
                data["doppler_spectrum_dBZe"].sel(range=slice(*chirp_info["range_limits"][chirp_])),
                aspect="auto",
                extent=[x_vals[0], x_vals[-1], *chirp_info["range_limits"][chirp_]],
                cmap=kwargs.get("cmap", "jet"),
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            
            axes[idx].axvline(x=nyquist_velocity, color="gray", linestyle="--")
            axes[idx].axvline(x=-nyquist_velocity, color="gray", linestyle="--")
            cm.append(cm_)

        if kwargs.get("power_spectrum_limits", None) is not None:
            for cm_ in cm:
                cm_.set_clim(vmin, vmax)

        axes[0].set_xlabel("Doppler Velocity, [m/s]")
        nyquist_velocity_limits = (
            -data["nyquist_velocity"].max().values.item(),
            data["nyquist_velocity"].max().values.item(),
        )
        for ax_ in axes:
            ax_.set_facecolor("white")
            ax_.set_xlim(*nyquist_velocity_limits)
            ax_.set_ylabel("Height, [m]")
            ax_.axvline(x=0, color="black", linestyle="--")
            ax_.minorticks_on()
            ax_.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
            ax_.grid(
                which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5
            )

        for ax_ in axes[1:]:
            ax_.set_xticklabels([])

        fig.suptitle(
            f'2D Doppler spectrum of {self.band}-band at {str(target_time).split(".")[0]}',
            fontsize=16,
        )

        plt.subplots_adjust(hspace=0.05)
        cax = fig.add_axes(
            (0.85, 0.15, 0.04, 0.7)
        )  # Adjust the width value to make the colorbar wider
        if kwargs.get("colorbar_label", None) is not None:
            fig.colorbar(cm[0], cax=cax, label=kwargs.get("colorbar_label"))
        else:
            if isinstance(cm, list):
                fig.colorbar(cm[0], cax=cax, label=data["doppler_spectrum_dBZe"].attrs["units"])
            else:
                fig.colorbar(cm, cax=cax, label=data["doppler_spectrum_dBZe"].attrs["units"])

        fig.subplots_adjust(left=0.1, right=0.82, bottom=0.10, top=0.9, wspace=0.2)

        # Adjust the right value to accommodate the wider colorbar
        # fig.tight_layout()

        output_dir = kwargs.get("output_dir", None)
        filepath = None
        if output_dir is not None:            
            filepath = output_dir / self.path.name.replace(
                ".nc",
                f"_{self.band}_2d-spectrum_{target_time:%Y%m%dT%H%M}.png",
            )
            fig.savefig(filepath, dpi=300)
        return fig, filepath

    def quicklook(self, variable: str | list[str] | None = None, **kwargs):
        if self.type == "ZEN":
            self.plot_zen(variable, **kwargs)
        elif self.type == "PPI":
            self.plot_ppi(variable, **kwargs)
        elif self.type == "RHI":
            self.plot_rhi(variable, **kwargs)
        else:
            raise ValueError(f"Type {self.type} is not valid")

    def plot_zen(self, variable: str | list[str] | None = None, **kwargs) -> tuple[list[Figure], list[Path]]:
        """Generates a quicklook plot for zenith pointing scans.

        Args:
            variable (str | list[str] | None, optional): Radar variables. Defaults to None.

        Raises:
            ValueError: Variable not in the file.
            ValueError: Variable is not valid.

        Returns:
            tuple[list[Figure], list[Path]]: List of figures and list of paths to the saved files.
        """        
        range_limits = kwargs.get("range_limits", (0, 12.0))

        if variable is None:
            variables_to_plot = ["dBZe"]
        elif isinstance(variable, str):
            variables_to_plot = [variable]
        elif isinstance(variable, list):
            variables_to_plot = variable
        else:
            raise ValueError(f"Variable {variable} is not valid")

        data = self.data.copy()
        data["range"] = data["range"] / 1e3
        list_of_figs, list_of_paths = [], []
        for var in variables_to_plot:
            if var not in self.data:
                raise ValueError(f"Variable {var} is not in the file")

            # vmin, vmax = RADAR_PLOT_INFO["limits"][var]
            fig, ax = plt.subplots(
                figsize=kwargs.get("figsize", (10, 7))
            )  # subplot_kw=dict(projection='polar')
            if var == 'ldr_slanted':
                vmin, vmax = kwargs.get("vmin", 0), kwargs.get("vmax", 30)
            else:
                vmin, vmax = kwargs.get("vmin", RADAR_PLOT_INFO["limits"][var][0]), kwargs.get("vmax", RADAR_PLOT_INFO["limits"][var][1])
            
            cmap = kwargs.get("cmap", "jet")
            pcm = data[f"{var}"].plot(
                x="time",
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            ) # type: ignore

            apply_gap_size(ax, data_array=self.data)

            ax.set_xlabel("Time, [UTC]")
            ax.set_ylim(*range_limits)
            ax.set_ylabel("Range, [km]")

            # Get current colorbar and shrink it 0.7, and set 'units' as label

            if "units" in data[var].attrs:
                units = data[var].attrs["units"]
            else:
                units = "?"
            if "long_name" in data[var].attrs:
                long_name = data[var].attrs["long_name"]
            else:
                long_name = vars

            # cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
            pcm.colorbar.remove()  # Remove the automatically generated colorbar
            colorbar = plt.colorbar(pcm, ax=ax, shrink=1.0)

            if kwargs.get("colorbar_label", None) is not None:
                colorbar.set_label(kwargs.get("colorbar_label", None))
            else:
                colorbar.set_label(f"{self.band}-band {long_name}, [{units}]")

            ax.set_title(f"{str(data.time.values[0]).split('.')[0]}")

            fig.tight_layout()
            list_of_figs.append(fig)
            if kwargs.get("savefig", False):
                output_dir = kwargs.get("output_dir", Path.cwd())
                filepath = output_dir / self.path.name.replace(
                    ".nc", f"_{self.band}-{var}.png"
                )
                dpi = kwargs.get("dpi", 300)
                fig.savefig(filepath, dpi=dpi)
                list_of_paths.append(filepath)
        return list_of_figs, list_of_paths

    def plot_ppi(
        self, variable: str | list[str]  | None = None, **kwargs
    ) -> tuple[list[Figure], list[Path]]:
        if variable is None:
            variables_to_plot = ["dBZe"]
        elif isinstance(variable, str):
            variables_to_plot = [variable]
        elif isinstance(variable, list):
            variables_to_plot = variable
        else:
            raise ValueError(f"Variable {variable} is not valid")

        data = self.data.sortby("azimuth")
        x, y = ppi_to_cartessian(data["range"], data["azimuth"], data["elevation"])
        mdata = data.mean(dim="time")
        list_of_figs, list_of_paths = [], []
        for var in variables_to_plot:
            if var not in mdata:
                raise ValueError(f"Variable {var} is not in the file")
            vmin, vmax = RADAR_PLOT_INFO["limits"][var]
            fig, ax = plt.subplots(
                figsize=(10, 10)
            )  # subplot_kw=dict(projection='polar')
            pcm = ax.pcolormesh(
                x / 1e3,
                y / 1e3,
                data[var].values.T,
                vmin=vmin,
                vmax=vmax,
                shading=kwargs.get("shading", "nearest"),
            )
            # Include title with time and elevation angle
            ax.set_xlim(*RADAR_PLOT_INFO["limits"]["ppi"]["x"])
            ax.set_ylim(*RADAR_PLOT_INFO["limits"]["ppi"]["y"])
            ax.set_xlabel("East-West distance from radar [km]")
            ax.set_ylabel("North-South distance from radar [km]")
            circular_grid(ax, radius=ax.get_xticks())
            ax.set_aspect("equal")
            cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
            if "units" in data[var].attrs:
                units = data[var].attrs["units"]
            else:
                units = "?"
            cbar.set_label(f"{var}, [{units}]")
            ax.set_title(
                f"{data.time.values[0]}. Elevation: {data.elevation.values[0]}"
            )
            fig.tight_layout()
            list_of_figs.append(fig)
            if kwargs.get("savefig", False):
                output_dir = kwargs.get("output_dir", Path.cwd())
                filepath = output_dir / self.path.name.replace(".nc", f"_{self.band}-{var}.png")
                dpi = kwargs.get("dpi", 300)
                fig.savefig(filepath, dpi=dpi)
                plt.close(fig)
                list_of_paths.append(filepath)
        return list_of_figs, list_of_paths

    def plot_rhi(self, variable: str | list[str] | None = None, **kwargs):
        if variable is None:
            variables_to_plot = ["dBZe"]
        elif isinstance(variable, str):
            variables_to_plot = [variable]
        elif isinstance(variable, list):
            variables_to_plot = variable

        label_angle = kwargs.get("label_angle", None)
        circular_grid = kwargs.get("circular_grid", False)
        cmap = kwargs.get("cmap", "viridis")
        shading = kwargs.get("shading", "nearest")

        # sort data['azimuth'] as increasing and sort the rest of the data accordingly
        data = self.data.sortby("elevation")
        constante_azimuth_angle = data["azimuth"].values[0]
        x, y = rhi_to_cartessian(data["range"], data["azimuth"], data["elevation"])
        mdata = data.mean(dim="time")
        list_of_figs, list_of_paths = [], []
        for var in variables_to_plot:
            if var not in mdata:
                raise ValueError(f"Variable {var} is not in the file")
            vmin, vmax = RADAR_PLOT_INFO["limits"][var]
            fig, ax = plt.subplots(
                figsize=(10, 10)
            )  # subplot_kw=dict(projection='polar')
            pcm = ax.pcolormesh(
                x / 1e3,
                y / 1e3,
                data[var].values.T,
                vmax=vmin,
                vmin=vmax,
                shading=shading,
                cmap=cmap,
            )
            ax.set_title(
                f"{str(data.time.values[0]).split('.')[0]} | Azimuth: {constante_azimuth_angle:.1f}°"
            )
            ax.set_xlim(*RADAR_PLOT_INFO["limits"]["rhi"]["x"])
            ax.set_ylim(*RADAR_PLOT_INFO["limits"]["rhi"]["y"])
            ax.set_xlabel(f"Distance from radar, [km]")
            ax.set_ylabel("Height distance from radar, [km]")
            if circular_grid:
                if label_angle is None:
                    label_angle = np.min(self.data["elevation"].values) - 5
                circular_grid(ax, radius=ax.get_xticks(), label_angle=label_angle)
            ax.set_aspect("equal")
            cbar = plt.colorbar(pcm, ax=ax, shrink=0.7)
            if "units" in data[var].attrs:
                units = data[var].attrs["units"]
            else:
                units = "?"
            if "long_name" in data[var].attrs:
                long_name = data[var].attrs["long_name"]
            else:
                long_name = var
            cbar.set_label(f"{long_name}, [{units}]")
            fig.tight_layout()
            list_of_figs.append(fig)
            if kwargs.get("savefig", False):
                output_dir = kwargs.get("output_dir", Path.cwd())
                filepath = output_dir / self.path.name.replace(".nc", f"_{self.band}-{var}.png")
                dpi = kwargs.get("dpi", 300)
                fig.savefig(filepath, dpi=dpi)
                plt.close(fig)
                list_of_paths.append(filepath)
        return list_of_figs, list_of_paths

    def plot_profile(
        self,
        target_times: datetime | np.datetime64 | list[datetime] | tuple[datetime, datetime],
        range_limits: tuple[float, float],
        variable: str | None = None,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        
        if variable is None:
            variable = "dBZe"

        fig = kwargs.get("fig", None)
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 7)))
        else:            
            ax = fig.get_axes()[0]  
            fig.savefig('testka.png', dpi=300)

        original_target_times = target_times        

        if isinstance(target_times, np.datetime64) or isinstance(target_times, datetime):
            time_list = [self.data.sel(time=target_times, method="nearest").time.values]
            title_str = f"Time: {str(time_list[0]).split('.')[0]}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_profile_{original_target_times:%Y%m%dT%H%M}.png",
            )
        elif isinstance(target_times, tuple):            
            time_list = self.data.sel(time=slice(*target_times)).time.values

            title_str = f"Period: {target_times[0]:%Y-%m-%d} {target_times[0]:%H:%M:%S} - {target_times[-1]:%H:%M:%S}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_profile_{target_times[0]:%Y%m%dT%H%M}_{target_times[-1]:%Y%m%dT%H%M}.png",
            )
        elif isinstance(target_times, list):
            _time_list = target_times
            time_list = np.unique(
                [
                    self.data.sel(time=time_, method="nearest").time.values
                    for time_ in _time_list
                ]
            )
            title_str = f"Period:  {target_times[0]:%Y-%m-%d} {target_times[0]:%H:%M:%S} - {target_times[-1]:%H:%M:%S}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_profile_{target_times[0]:%Y%m%dT%H%M}_{target_times[-1]:%Y%m%dT%H%M}.png",
            )
        else:
            raise ValueError("target_times must be a datetime, np.datetime64, tuple or list.")
        
        data = self.data.copy()
        data = data.sel(range=slice(*range_limits))
        colors = kwargs.get('color_list', color_list(len(time_list)))
        data["range"] = data["range"] / 1e3
        for idx, time_ in enumerate(time_list):
            data[variable].sel(time=time_).plot(y='range', ax=ax, color=colors[idx], label=f"{parse_datetime(time_):%H:%M:%S}") #type: ignore
        if 'range_limits' in kwargs:
            range_limits = kwargs.get('range_limits', [0, 10000.])/1e3
            ax.set_ylim(range_limits)
        if 'variable_limits' in kwargs:            
            ax.set_xlim(kwargs.get('variable_limits'))
        
        ax.set_ylabel("Range, [km]")
        ax.set_xlabel(f"{data[variable].attrs['long_name']}, [{data[variable].attrs['units']}]")
        ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
        ax.grid(
            which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5
        )

        ax.set_title(title_str)
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)        
        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir", None)
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300)
        return fig, filepath

    def plot_timeseries(
        self,
        target_ranges: float | list[float] | np.ndarray[Any, np.dtype[np.float64]],
        time_limits: tuple[datetime | np.datetime64 , datetime | np.datetime64] | None = None,
        variable: str | None = None,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        
        if variable is None:
            variable = "dBZe"

        if time_limits is None:
            time_limits = (parse_datetime(self.data.time.min().values), parse_datetime(self.data.time.max().values))
        original_time_limits = time_limits

        fig = kwargs.get("fig", None)
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))
        else:            
            ax = fig.get_axes()[0]              
        
        if isinstance(target_ranges, tuple):            
            range_list = self.data.sel(range=slice(*target_ranges)).range.values/1e3

            title_str = f"Period: {time_limits[0]:%Y-%m-%d} {time_limits[0]:%H:%M:%S} - {time_limits[-1]:%H:%M:%S}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_timeseries_{original_time_limits[0]:%Y%m%dT%H%M}_{original_time_limits[-1]:%Y%m%dT%H%M}_{target_ranges[0]:.1f}_{target_ranges[-1]:.1f}.png",
            )
        elif isinstance(target_ranges, list):
            _range_list = target_ranges
            range_list = np.unique(
                [
                    self.data.sel(range=range_, method="nearest").range.values
                    for range_ in _range_list
                ]
            )/1e3
            title_str = f"Period: {time_limits[0]:%Y%m%d} {time_limits[0]:%H:%M:%S} - {time_limits[-1]:%H:%M:%S}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_timeseries_{time_limits[0]:%Y%m%dT%H%M}_{time_limits[-1]:%Y%m%dT%H%M}_{target_ranges[0]:.1f}_{target_ranges[-1]:.1f}.png",
            )
        else:
            range_list = [self.data.sel(range=target_ranges, method="nearest").range.values/1e3]
            title_str = f"Time: {str(time_limits[0]).split('.')[0]}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_timeseries_{time_limits[0]:%Y%m%dT%H%M}_{time_limits[-1]:%Y%m%dT%H%M}_{target_ranges:.1f}.png",
            )

        data = self.data.copy()        
        colors = kwargs.get('color_list', color_list(len(range_list)))
        data["range"] = data["range"] / 1e3            
        for idx, range_ in enumerate(range_list):
            data[variable].sel(range=range_).plot(x='time', ax=ax, color=colors[idx], label=f"{range_:.3f} km") #type: ignore
        
        if 'range_limits' in kwargs:
            range_limits = kwargs.get('range_limits', [0,10000.])/1e3
            ax.set_ylim(range_limits)
        if 'variable_limits' in kwargs:            
            ax.set_xlim(kwargs.get('variable_limits'))
        
        ax.set_title(title_str)
        ax.set_ylabel(f"{data[variable].attrs['long_name']}, [{data[variable].attrs['units']}]")
        ax.set_xlabel("Time, [UTC]")
        ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
        ax.minorticks_on()
        ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
        ax.grid(
            which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5
        )
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)        
        
        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir", None)
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300)
        return fig, filepath



    # def __str__(self) -> str:
    #     return super().__str__()
