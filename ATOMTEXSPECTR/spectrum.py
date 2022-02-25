"""
spectrum.py - fundamental class for file-spectrum parsing.
In the module, it is represented as a class of differential
energy distribution - the spectrum of ionizing radiation.
"""
import os
import warnings
import datetime
import numpy
from ATOMTEXSPECTR import plot
from ATOMTEXSPECTR.tools import bin_centers_from_edges, parsing_datetime, parsing_inaccuracy, machineEpsilon
import ATOMTEXSPECTR
from uncertainties import UFloat, unumpy
from ATOMTEXSPECTR.warn import spectrum_error, spectrum_warning, uncalibrated_error
from copy import deepcopy
import io


class spectrum:
    """

    """

    def __init__(self,

                 counts = None,
                 cps = None,
                 sp_start_count = None,
                 sp_stop_count = None,
                 measuretime = None,
                 actualtime = None,
                 inaccuracy = None,
                 channel_bin_edges = None,
                 energy_bin_edges = None,
                 **kwargs,
                 ):
        self.data = dict()
        # Checking i/o for an ordinate axis
        if not (counts is None) ^ (cps is None): # (not False) == True
            raise spectrum_error("It is necessary to specify "
                                 "what value is in the data: counts or cps!")
        self._counts = None
        self._cps = None
        self._channel_bin_edges = None
        self._energy_bin_edges = None
        self.energy_cal = None

        if counts is not None:
            if len(counts) == 0:
                raise spectrum_error("Empty spectrum counts.")
            if inaccuracy is None and numpy.any(numpy.asarray(counts) < 0):
                raise spectrum_error("The counts should not contain negative values!")
            self._counts = parsing_inaccuracy(counts, inaccuracy,
                                              lambda x: numpy.maximum(numpy.sqrt(x), 1)
                                              )
        else:
            if len(cps) == 0:
                raise spectrum_error("Spectrum without cps.")
            self._cps = parsing_inaccuracy(cps, inaccuracy, lambda x: numpy.nan)




        # conditional statements for parsing the abscissa axis
        # Package based on bin_edges from numpy.histogram()
        """
        |   |   |   |   |   |   |   |
        0   1   2   3   .   .   .   1023
        """
        if channel_bin_edges is None and not (counts is None and cps is None):
            channel_bin_edges = numpy.arange(len(self) + 1)

        self.channel_bin_edges = channel_bin_edges


        self.energy_bin_edges = energy_bin_edges


        # conditional statements for parsing time
        self.measuretime = None
        self.actualtime = None

        if measuretime is not None:
            self.measuretime = float(measuretime)
        if actualtime is not None: # NULL
            self.actualtime = float(actualtime)
            if measuretime is not None:
                if self.measuretime > self.actualtime:
                    raise ValueError(f"Time of measurements {measuretime} can't be "
                                     f"more actual time {actualtime}.")
        self.sp_start_count = parsing_datetime(sp_start_count, submit_None = True)
        self.sp_stop_count = parsing_datetime(sp_stop_count , submit_None = True)
        if (
            self.actualtime is not None and
            self.sp_start_count is not None and
            self.sp_stop_count is not None
        ):
            raise spectrum_error("No more than two out of three arguments must be specified for")
        elif (self.sp_start_count is not None and self.sp_stop_count is not None):
            if self.sp_start_count > self.sp_stop_count:
                raise ValueError(f"Time of start {sp_start_count} набора спектра "
                                 f"there should be less end time {sp_stop_count}!")
            self.actualtime = (self.sp_stop_count - self.sp_start_count).total_seconds()
        elif self.actualtime is not None and self.sp_start_count is not None:
            self.sp_stop_count = self.sp_start_count + datetime.timedelta(seconds = self.actualtime)
        elif self.actualtime is not None and self.sp_stop_count is not None:
            self.sp_start_count = self.sp_stop_count - datetime.timedelta(seconds = self.actualtime)
        # Filling out a new vocabulary in the class spectrum.
        for key in kwargs:
            self.data[key] = kwargs[key]
        # Calibration for energy radionuclides
        # todo добавить методы для применения калибровок
        self.calibration_energy = None
        # These two lines make sure operators between a Spectrum
        # and a numpy arrays are forbidden and cause a TypeError
        self.__array_ufunc__ = None
        self.__array_priority__ = 1

    def __len__(self) -> int:
        """
        The number of bins in the spectrum.
        """
        try:
            return len(self.counts)
        except spectrum_error:
            return len(self.cps)

    def __str__(self):
        """
        Output description file-spectrum.
        """
        import platform
        lines = ["ATOMTEXSPECTR"]
        ltups = []
        for index in ['sp_start_count', 'sp_stop_count', 'actualtime', 'measuretime', 'is_calibrated_for_energy']:
            ltups.append((index, getattr(self, index)))
        ltups.append(("Numbers of channels", len(self.bin_item)))
        if self._counts is None:
            # todo сделать ремарке по gross_counts
            ltups.append(("gross_counts", None))
        else:
            ltups.append(("Sum counts:", self.counts.sum()))
            try:
                ltups.append(("Spectrum cps:", self.cps.sum()))
            except spectrum_error:
                # todo сделать ремарке по gross_cps
                ltups.append(("gross_cps", None))
            if "filename" in self.data:
                if platform.system() == "Windows":
                    delim = "\\"
                else:
                    delim = "/"
                i = -1; name_file = list()
                while self.data["filename"][i] != delim:
                    name_file.insert(i, self.data["filename"][i])
                    i -= 1

                ltups.append(("Name spectrum:", "".join(name_file)))
                ltups.append(("Path spectrum:", self.data["filename"][0:-len(name_file)]))
            else:
                ltups.append(("filename", None))
            for lt in ltups:

                lines.append("    {:40} {}".format(f"{lt[0]}:", lt[1]))
            # header = ["{:>{}}".format("", max([len(i) for i in lines[0]]))]
            return "\n".join(lines)

# --------------- Property for counts --------------- #
# An important advantage of working through properties
# is that you can check input values before assigning them to attributes.

    @property
    def counts(self):
        """
        Counts in each bin, with uncertainty.
        """
        if self._counts is not None:
            return self._counts
        else:
            try:
                return self.cps * self.measuretime
            except TypeError:
                raise spectrum_error("Unknown measurement time; it is impossible to get the counting rate from the number of counts")
    @property
    def values_counts(self):
        """
        Counts in each bin, no inaccuracy (uncertainties).
        """
        return unumpy.nominal_values(self.counts) # numpy.array (float)

    @property
    def inaccuracy_counts(self):
        """
        Inaccuracy (uncertainties) on the counts in each bin.
        """
        return unumpy.std_devs(self.counts) # numpy.array (float)

# --------------- Property for cps --------------- #
    @property
    def cps(self):
        """
        Counts per second in each bin, with inaccuracy (uncertainties).
        """
        if self._cps is not None:
            return self._cps
        else:
            try:
                return self.counts / self.measuretime
            except TypeError:
                raise spectrum_error(
                    "Unknown meatime; cannot calculate CPS from counts"
                )

    @property
    def values_cps(self):
        """
        Counts per second in each bin, no inaccuracy (uncertainties).
        """
        return unumpy.nominal_values(self.cps) # numpy.array (float)

    @property
    def inaccuracy_cps(self):
        """
        Inaccuracy (uncertainties) on the counts per second in each bin.
        """
        return unumpy.std_devs(self.cps) # numpy.array (float)

# --------------- Property for cps per keV --------------- #
    @property
    def cpskev(self):
        """
        Counts per second per keV in each bin, with inaccuracy (uncertainty).
        """
        return self.cps / self.bin_widths_kev   #  an numpy.array of uncertainties.ufloats

    @property
    def values_cpskev(self):
        return unumpy.nominal_values(self.cpskev) # numpy.array (float)

    @property
    def inaccuracy_cpskev(self):
        return unumpy.std_devs(self.cpskev) # numpy.array (float)

# --------------- Property for channels --------------- #
    @property
    def bin_item(self):
        """
        Bin item
        """
        return numpy.arange(len(self), dtype = int) # numpy.array of int from 0 to (len(self.counts) - 1)

    @property
    def channels(self):
        """
        Alias for bin_item().
        """
        warnings.warn(  "Channels is deprecated terminology. Use bin_item instead.",
                        DeprecationWarning,
                        )
        return numpy.arange(len(self), dtype=int)

    @property
    def channel_bin_centers(self):
        """
        Convenience function for accessing the channel values of bin centers.
        """
        return bin_centers_from_edges(self.channel_bin_edges) # numpy.array of floats, same length as self.counts

    @property
    def channel_bin_widths(self):
        """
        The width of each bin, in channel values.
        """
        return numpy.diff(self.channel_bin_edges) # numpy.array of floats, same length as self.counts

    @property
    def is_calibrated_for_energy(self):
        return self.energy_bin_edges is not None

    @property
    def energy_bin_centers(self):
        """
        Convenience function for accessing the energies of bin centers.
        """
        if not self.is_calibrated_for_energy:
            raise uncalibrated_error("Spectrum not calibrated.")
        else:
            return bin_centers_from_edges(self.energy_bin_edges) # numpy.array of floats, same length as self.counts

    @property
    def energy_bin_edges(self):
        """
        Get the bin edge energies of a spectrum.
        """
        return self._energy_bin_edges # numpy.array of floats or None

    @energy_bin_edges.setter
    def energy_bin_edges(self, energy_bin_edges):
        """
        Set the bin edge energies of a spectrum.
        """
        if energy_bin_edges is None:
            self._energy_bin_edges = None
        elif len(energy_bin_edges) != len(self) + 1:
            raise spectrum_error("Bad length of bin edges vector")
        elif numpy.any(numpy.diff(energy_bin_edges) <= 0):
            raise ValueError("Bin edge energies must be strictly increasing")
        else:
            self._energy_bin_edges = numpy.array(energy_bin_edges, dtype = float)

    @property
    def bin_widths_kev(self):
        """
        The width of each bin, in keV.
        """
        warnings.warn(
            "bin_widths is deprecated and will be removed in a "
            "future release. Use energy_bin_edges (or channel_bin_widths) "
            "instead.",
            DeprecationWarning,
        )
        if not self.is_calibrated_for_energy:
            raise uncalibrated_error("Spectrum is not calibrated")
        else:
            return numpy.diff(self.energy_bin_edges)


    @property
    def channel_bin_edges(self):
        """
        Get the channel bin edges of a spectrum
        """
        return self._channel_bin_edges # numpy.array of floats or None

    @channel_bin_edges.setter
    def channel_bin_edges(self, channel_bin_edges):
        """
        Set the channel bin edges of a spectrum
        """
        if channel_bin_edges is None:
            self._channel_bin_edges = None
        elif len(channel_bin_edges) != len(self) + 1:
            raise spectrum_error("Bad length of bin edges vector")
        elif numpy.any(numpy.diff(channel_bin_edges) <= 0):
            raise ValueError("Channel bin edges must be strictly increasing")
        else:
            self._channel_bin_edges = numpy.array(channel_bin_edges, dtype=float)

    @classmethod
    def import_file(cls, filename, debugging = False):
        """
        Build a Spectrum object from a filename.
        """
        _, extension = os.path.splitext(filename)
        if extension.lower() == ".spe":
            data_from_filename, calibration = ATOMTEXSPECTR.read.spe.reading(filename, deb = debugging)
        elif extension.lower() == ".ats":
            data_from_filename, calibration = ATOMTEXSPECTR.read.ats.reading(filename, deb = debugging)
        else:
            raise NotImplementedError(f"Extension file-spectrum {extension} not reading.")
        output = cls(**data_from_filename)
        output.data["filename"] = filename
        if calibration is not None:
            output.apply_calibration(calibration)
        return output  # object spectrum
    # todo create write object file-spectrum
    @classmethod
    def import_from_list(cls, data_list, bins = None, calibration = False, xmin = None, xmax = None, **kwargs):
        """
        Build a Spectrum object from a listmode data.
        """
        assert len(data_list) > 0
        if xmin is None:
            xmin = 0
        if xmax is None:
            xmax = numpy.ceil(max(data_list))
        if bins is None:
            bins = numpy.arange(xmin, xmax + 1, dtype=int)
        assert xmin < xmax
        if isinstance(bins, int):
            assert bins > 0
        else:
            assert len(bins) > 1
        bin_counts, bin_edges = numpy.histogram(data_list, bins = bins, range = (xmin, xmax))
        kwargs["counts"] = bin_counts
        kwargs["energy_bin_edges" if calibration else "channel_bin_edges"] = bin_edges
        return cls(**kwargs)

    def copy(self):
        """
        Copy of this spectrum object.
        """
        from copy import deepcopy
        return deepcopy(self) #  a spectrum object identical to this one

    def parsing_abscissa(self, x):
        """
        Parse the axis abscissa mode to get the associated data and plot label.
        """
        if x == "energy":
            xedges = self.energy_bin_edges
            xlabel = "Energy [keV]"
        elif x == "channel":
            xedges = self.channel_bin_edges
            xlabel = "Channel"
        else:
            raise ValueError(f"Unsupported xmode: {x:s}")
        return xedges, xlabel

    def parsing_ordinate(self, ordinate):
        """
        Parse the axis ordinate mode to get the associated data and plot label.
        """
        if ordinate == "counts":
            ydata = self.values_counts
            yuncs = self.inaccuracy_counts
            ylabel = "Counts"
        elif ordinate == "cps":
            ydata = self.values_cps
            yuncs = self.inaccuracy_cps
            ylabel = "cps [1/s]"
        elif ordinate == "cpskev":
            ydata = self.values_cpskev
            yuncs = self.inaccuracy_cpskev
            ylabel = "cps [1/s/keV]"
        else:
            raise ValueError(f"Unsupported ymode: {ordinate:s}")
        return ydata, yuncs, ylabel

    def plot(self, *args, **kwargs):
        """
        Plot a spectrum with matplotlib's plot command.
        """
        emode = kwargs.pop("emode", "none")

        alpha = kwargs.get("alpha", 1)
        plotaxes = plot.plot_spectrum(self, *args, **kwargs)
        ax = plotaxes.plot()
        color = ax.get_lines()[-1].get_color()
        if emode == 'band':
            plotaxes.errorband(color = color, alpha = alpha * 0.5, label = "_nolegend_" )
        elif emode == "bars" or emode == "bar":
            plotaxes.errorbar(color = color, label = "_nolegend_")
        elif emode != "none":
            raise spectrum_error(f"Unknown error setting format {emode}, "
                                f"use 'bars' или 'band' ")
        return ax

    def fill_between(self, **kwargs):
        """
        Plot a spectrum with matplotlib's fill_between command

        """
        plotter = plot.plot_spectrum(self, **kwargs)
        return plotter.fill_between()

    def write(self, name):
        """Write the Spectrum to an hdf5 file.
        Parameters
        ----------
        name : str, h5py.File, h5py.Group
            The filename or an open h5py File or Group.
        """
        # build datasets dict
        dsets = {}
        # handle counts versus CPS data
        if self._cps is None:
            assert self._counts is not None
            # NOTE: integer character of counts has been destroyed
            dsets["counts"] = self.values_counts
            dsets["uncs"] = self.inaccuracy_counts
        if self._counts is None:
            assert self._cps is not None
            dsets["cps"] = self.values_cps
            dsets["uncs"] = self.inaccuracy_cps
        # handle other array data
        for key in ["bin_edges_raw", "bin_edges_kev"]:
            val = getattr(self, key)
            if val is not None:
                dsets.update({key: val})

        # build attributes dict
        attrs = deepcopy(self.data)
        # convert time attributes to strings
        for key in ["start_time", "stop_time"]:
            val = getattr(self, key)
            if val is not None:
                iso8601 = f"{val:%Y-%m-%dT%H:%M:%S.%f%z}"
                attrs.update({key: iso8601})
        for key in ["livetime", "realtime"]:
            val = getattr(self, key)
            if val is not None:
                attrs.update({key: val})
        # cannot specify all three of start, stop, and real time
        if "start_time" in attrs and "stop_time" in attrs and "realtime" in attrs:
            attrs.pop("realtime")

        # write all spectrum data to file
        io.write_h5(name, dsets, attrs)

        # write calibration to file
        if self.energy_cal is not None:
            try:
                with io.open_h5(name, "r+") as h5:
                    group = io.create_group("energy_cal")
                    self.energy_cal.write(group)
            except AttributeError:
                warnings.warn(
                    "Unable to write energy calibration data to file. "
                    "This may be caused by the use of "
                    "bq.EnergyCalBase classes, which are deprecated"
                    "and will be removed in a future release; "
                    "use bq.Calibration instead",
                    DeprecationWarning,
                )
# --------------- Operation from spectrum --------------- #

    def __add__(self, other):
        """
        Add spectra together.
        """

        self._add_sub_error_checking(other)
        if (self._counts is None) ^ (other._counts is None):
            raise spectrum_error(
                "Addition of counts-based and CPS-based spectra is "
                + "ambiguous, use Spectrum(counts=specA.counts+specB.counts) "
                + "or Spectrum(cps=specA.cps+specB.cps) instead."
            )

        if self._counts is not None and other._counts is not None:
            kwargs = {"counts": self.counts + other.counts}
            if self.measuretime and other.measuretime:
                kwargs["livetime"] = self.measuretime + other.measuretime
            else:
                warnings.warn(
                    "Addition of counts with missing livetimes, "
                    + "livetime was set to None.",
                    spectrum_warning,
                )
        else:
            kwargs = {"cps": self.cps + other.cps}

        if self.is_calibrated_for_energy and other.is_calibrated_for_energy:
            spectrum_object = spectrum(energy_bin_edges=self.energy_bin_edges, **kwargs)
        else:
            spectrum_object = spectrum(channel_bin_edges=self.channel_bin_edges, **kwargs)
        return spectrum_object

    def __sub__(self, other):
        """
        Normalize spectra (if possible) and subtract.
        """
        self._add_sub_error_checking(other)
        try:
            kwargs = {"cps": self.cps - other.cps}
            if (self._cps is None) or (other._cps is None):
                warnings.warn(
                    "Subtraction of counts-based specta, spectra "
                    + "have been converted to CPS",
                    spectrum_warning,
                )
        except spectrum_error:
            try:
                kwargs = {"counts": self.values_counts - other.values_counts}
                kwargs["uncs"] = [numpy.nan] * len(self)
                warnings.warn(
                    "Subtraction of counts-based spectra, "
                    + "livetimes have been ignored.",
                    spectrum_warning,
                )
            except spectrum_error:
                raise spectrum_error(
                    "Subtraction of counts and CPS-based spectra without"
                    + "livetimes not possible"
                )

        if self.is_calibrated_for_energy and other.is_calibrated_for_energy:
            spectrum_object = spectrum(energy_bin_edges=self.energy_bin_edges, **kwargs)
        else:
            spectrum_object = spectrum(channel_bin_edges=self.channel_bin_edges, **kwargs)
        return spectrum_object

    def _add_sub_error_checking(self, other):
        """
        Parsing errors for spectra addition or subtraction.
        """

        if not isinstance(other, spectrum):
            raise TypeError(
                "Spectrum addition/subtraction must involve a Spectrum object"
            )
        if len(self) != len(other):
            raise spectrum_error("Cannot add/subtract spectra of different lengths")
        if self.is_calibrated_for_energy ^ other.is_calibrated_for_energy:
            raise spectrum_error(
                "Cannot add/subtract uncalibrated spectrum to/from a "
                + "calibrated spectrum. If both have the same calibration, "
                + 'please use the "calibrate_like" method'
            )
        if self.is_calibrated_for_energy and other.is_calibrated_for_energy:
            if not numpy.all(self.energy_bin_edges == other.energy_bin_edges):
                raise NotImplementedError(
                    "Addition/subtraction for arbitrary calibrated spectra "
                    + "not implemented"
                )
                # TODO: if both spectra are calibrated but with different
                #   calibrations, should one be rebinned to match?
        if not self.is_calibrated_for_energy and not other.is_calibrated_for_energy:
            if not numpy.all(self.channel_bin_edges == other.channel_bin_edges):
                raise NotImplementedError(
                    "Addition/subtraction for arbitrary uncalibrated "
                    + "spectra not implemented"
                )

    def __mul__(self, other):
        """
        Return a new Spectrum object with counts (or CPS) scaled up.
        """
        return self._mul_div(other, div = False)

    __rmul__ = __mul__

    def __div__(self, other):
        """
        Return a new Spectrum object with counts (or CPS) scaled down.
        """
        return self._mul_div(other, div = True)

    __truediv__ = __div__

    def _mul_div(self, scaling_factor, div=False):
        """
        Multiply or divide a spectrum by a scalar. Handle errors.
        """

        if not isinstance(scaling_factor, UFloat):
            try:
                scaling_factor = float(scaling_factor)
            except (TypeError, ValueError):
                raise TypeError("Spectrum must be multiplied/divided by a scalar")
            if (
                    scaling_factor == 0
                    or numpy.isinf(scaling_factor)
                    or numpy.isnan(scaling_factor)
            ):
                raise ValueError("Scaling factor must be nonzero and finite")
        else:
            if (
                    scaling_factor.nominal_value == 0
                    or numpy.isinf(scaling_factor.nominal_value)
                    or numpy.isnan(scaling_factor.nominal_value)
            ):
                raise ValueError("Scaling factor must be nonzero and finite")
        if div:
            multiplier = 1 / scaling_factor
        else:
            multiplier = scaling_factor

        if self._counts is not None:
            data_arg = {"counts": self.counts * multiplier}
        else:
            data_arg = {"cps": self.cps * multiplier}

        if self.is_calibrated_for_energy:
            spect_obj = spectrum(energy_bin_edges=self.energy_bin_edges, **data_arg)
        else:
            spect_obj = spectrum(channel_bin_edges=self.channel_bin_edges, **data_arg)
        return spect_obj

    def downsample(self, f, parsing_measuretime = None):
        """
        Downsample counts and create a new spectrum.
        """

        if self._counts is None:
            raise spectrum_error("Cannot downsample from CPS")
        if f < 1:
            raise ValueError("Cannot upsample a spectrum; f must be > 1")

        if parsing_measuretime is None:
            new_measuretime = None
        elif parsing_measuretime.lower() == "preserve":
            new_measuretime = self.measuretime
        elif parsing_measuretime.lower() == "reduce":
            new_measuretime = self.measuretime / f
        else:
            raise ValueError(f"Illegal value for handle_livetime: {parsing_measuretime}")
        # TODO handle uncertainty?
        old_counts = self.values_counts.astype(int)
        new_counts = numpy.random.binomial(old_counts, 1.0 / f)

        if self.is_calibrated_for_energy:
            return spectrum(
                counts = new_counts,
                energy_bin_edges = self.energy_bin_edges,
                measuretime = new_measuretime,
            )
        else:
            return spectrum(
                counts=new_counts,
                channel_bin_edges = self.channel_bin_edges,
                measuretime = new_measuretime,
            )

    def has_uniform_bins(self, use_kev=None, rtol=None):
        # todo изменить данный метод
        """
        Test whether the Spectrum has uniform binning.
        """

        if rtol is None:
            rtol = 100 * machineEpsilon()
        if rtol < machineEpsilon():
            raise ValueError("Relative tolerance rtol cannot be < system EPS")

        if use_kev is None:
            use_kev = self.is_calibrated_for_energy

        if use_kev and not self.is_calibrated_for_energy:
            raise uncalibrated_error(
                "Cannot access energy bins with an uncalibrated Spectrum."
            )

        bin_widths = self.energy_bin_edges if use_kev else self.channel_bin_edges

        iterator = iter(bin_widths)
        x0 = next(iterator, None)
        for x in iterator:
            if abs(x / x0 - 1.0) > rtol:
                return False
        return True

    def find_bin_index(self, x, use_kev=None):
        """
        Find the Spectrum bin index or indices containing x-axis value(s) x.
        """

        if use_kev is None:
            use_kev = self.is_calibrated_for_energy

        if use_kev and not self.is_calibrated_for_energy:
            raise uncalibrated_error(
                "Cannot access energy bins with an uncalibrated Spectrum."
            )

        bin_edges, bin_widths, _ = self.get_bin_properties(use_kev)
        x = numpy.asarray(x)

        if numpy.any(x < bin_edges[0]):
            raise spectrum_error("requested x is < lowest bin edge")
        if numpy.any(x >= bin_edges[-1]):
            raise spectrum_error("requested x is >= highest bin edge")

        return numpy.searchsorted(bin_edges, x, "right") - 1

    def get_bin_properties(self, use_kev=None):
        """
        Convenience function to get bin properties: edges, widths, centers.
        """

        if use_kev is None:
            use_kev = self.is_calibrated_for_energy

        if use_kev:
            if not self.is_calibrated_for_energy:
                raise uncalibrated_error(
                    "Cannot access energy bins with an uncalibrated Spectrum."
                )
            return self.energy_bin_edges, self.energy_bin_edges, self.energy_bin_edges
        else:
            return self.channel_bin_edges, self.channel_bin_edges, self.channel_bin_edges

    def apply_calibration(self, cal):
        """
        Use a Calibration to generate bin edge energies for this spectrum.
        """

        try:
            self.energy_bin_edges = cal.ch2kev(self.channel_bin_edges)
            warnings.warn(
                "The use of EnergyCalBase classes is deprecated "
                "and will be removed in a future release; "
                "use bq.Calibration instead",
                DeprecationWarning,
            )
        except AttributeError:
            self.energy_bin_edges = cal(self.channel_bin_edges)
        self.energy_bin_edges = cal(self.channel_bin_edges)
        self.energy_cal = cal

    def calibrate_like(self, other):
        """
        Apply another Spectrum object's calibration (bin edges vector).
        Bin edges are copied, so the two spectra do not have the same object
        in memory.
        """

        if other.is_calibrated_for_energy:
            self.energy_bin_edges = other.energy_bin_edges.copy()
            self.energy_cal = other.energy_cal
        else:
            raise uncalibrated_error("Other spectrum is not calibrated")

    def rm_calibration(self):
        """Remove the calibration (if it exists) from this spectrum."""

        self.energy_bin_edges = None
        self.energy_cal = None

    def combine_bins(self, f):
        """
        Make a new Spectrum with counts combined into bigger bins.
        If f is not a factor of the number of bins, the counts from the first
        spectrum will be padded with zeros.
        """

        f = int(f)
        if self._counts is None:
            key = "cps"
        else:
            key = "counts"
        data = getattr(self, key)
        if len(self) % f == 0:
            padded_counts = numpy.zercopy(data)
        else:
            pad_len = f - len(self) % f
            pad_counts = unumpy.uarray(numpy.zeros(pad_len), numpy.zeros(pad_len))
            padded_counts = numpy.concatenate((data, pad_counts))
        padded_counts.resize(int(len(padded_counts) / f), f)
        combined_counts = numpy.sum(padded_counts, axis=1)
        if self.is_calibrated_for_energy:
            combined_bin_edges = self.energy_bin_edges[::f]
            if combined_bin_edges[-1] != self.energy_bin_edges[-1]:
                combined_bin_edges = numpy.append(
                    combined_bin_edges, self.energy_bin_edges[-1]
                )
        else:  # TODO: should be able to combine bins
            combined_bin_edges = None

        kwargs = {
            key: combined_counts,
            "energy_bin_edges": combined_bin_edges,
            "livetime": self.measuretime,
        }
        object = spectrum(**kwargs)
        return object

    # todo add rebin object