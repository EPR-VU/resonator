"""
This module contains models and fitters for resonators exhibiting Fano resonance lineshapes.

Fano resonance arises from quantum interference between a resonant pathway (through the resonator)
and a direct (non-resonant) pathway. The resulting lineshape is asymmetric and is characterized by
the Fano asymmetry parameter `fano_asymmetry` (often denoted q in the literature).

When `fano_asymmetry` is zero, all models reduce to their symmetric Lorentzian counterparts.

Note: For the shunt configuration, this is equivalent to the `LinearShunt` model in `shunt.py`,
which uses the parameter name `asymmetry` instead of `fano_asymmetry`.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import background, guess, linear
from .reflection import AbstractReflection
from .shunt import AbstractShunt
from .transmission import AbstractSymmetricTransmission


# Fano reflection models and fitters

class FanoReflection(AbstractReflection):
    """
    This class models a resonator operated in reflection with a Fano-type asymmetric lineshape.

    The Fano resonance arises from interference between the resonant path (through the resonator)
    and a direct non-resonant path. The `fano_asymmetry` parameter (real-valued) controls the
    degree and direction of the asymmetry. When `fano_asymmetry` is zero, this reduces exactly
    to the symmetric `LinearReflection` model.

    The model is:
        S11 = -1 + 2 * (1 + j * fano_asymmetry) / (1 + (internal_loss + 2j * detuning) / coupling_loss)

    where detuning = frequency / resonance_frequency - 1.
    """

    def __init__(self, *args, **kwds):
        """
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def fano_reflection(frequency, resonance_frequency, coupling_loss, internal_loss, fano_asymmetry):
            detuning = frequency / resonance_frequency - 1
            return -1 + (2 * (1 + 1j * fano_asymmetry)) / (1 + (internal_loss + 2j * detuning) / coupling_loss)

        super(FanoReflection, self).__init__(func=fano_reflection, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        resonance_frequency, coupling_loss, internal_loss = guess.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        params['fano_asymmetry'].set(value=0, min=-100, max=100)
        return params


class FanoReflectionFitter(linear.LinearResonatorFitter):
    """
    This class fits data from a resonator operated in reflection with a Fano-type asymmetric lineshape.

    The `fano_asymmetry` parameter (often written as q in the literature) characterizes the degree
    and direction of the asymmetry. A value of zero corresponds to a symmetric Lorentzian (identical
    to `LinearReflectionFitter`). Large magnitudes produce highly asymmetric lineshapes.
    """

    def __init__(self, frequency, data, background_model=None, errors=None, **kwds):
        """
        Fit the given data to a composite model that is the product of a background model and the FanoReflection model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data.
        :param background_model: an instance (not the class) of a model representing the background response without the
          resonator; the default of `background.MagnitudePhase()` assumes that this is modeled well by a single complex
          constant at all frequencies.
        :param errors: an array of complex numbers containing the standard errors of the mean of the data points.
        :param kwds: keyword arguments passed directly to `lmfit.model.Model.fit()`.
        """
        if background_model is None:
            background_model = background.MagnitudePhase()
        super(FanoReflectionFitter, self).__init__(frequency=frequency, data=data,
                                                   foreground_model=FanoReflection(),
                                                   background_model=background_model, errors=errors, **kwds)

    def invert(self, scattering_data):
        """
        Return the resonator detuning and internal_loss corresponding to the given normalized scattering data.

        Inverts: S11 = -1 + 2*(1 + j*q) / (1 + (internal_loss + 2j*detuning) / coupling_loss)

        :param scattering_data: normalized complex scattering data (S11 at the resonator plane).
        :return: detuning, internal_loss (both array[float])
        """
        z = self.coupling_loss * (2 * (1 + 1j * self.fano_asymmetry) / (1 + scattering_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss


# Fano shunt models and fitters

class FanoShunt(AbstractShunt):
    """
    This class models a resonator operated in the shunt-coupled configuration with a Fano-type
    asymmetric lineshape.

    In the shunt geometry, the direct transmission path naturally produces Fano interference with
    the resonant path. The `fano_asymmetry` parameter controls the degree of asymmetry.
    When `fano_asymmetry` is zero, this reduces exactly to the symmetric `LinearShunt` model.

    The model is:
        S21 = 1 - (1 + j * fano_asymmetry) / (1 + (internal_loss + 2j * detuning) / coupling_loss)

    where detuning = frequency / resonance_frequency - 1.

    Note: This is equivalent to `LinearShunt` with the `asymmetry` parameter renamed to
    `fano_asymmetry` for naming consistency with `FanoReflection`.
    """

    def __init__(self, *args, **kwds):
        """
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def fano_shunt(frequency, resonance_frequency, coupling_loss, internal_loss, fano_asymmetry):
            detuning = frequency / resonance_frequency - 1
            return 1 - (1 + 1j * fano_asymmetry) / (1 + (internal_loss + 2j * detuning) / coupling_loss)

        super(FanoShunt, self).__init__(func=fano_shunt, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        resonance_frequency, coupling_loss, internal_loss = guess.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        params['fano_asymmetry'].set(value=0, min=-100, max=100)
        return params


class FanoShuntFitter(linear.LinearResonatorFitter):
    """
    This class fits data from a shunt-coupled resonator with a Fano-type asymmetric lineshape.

    In the shunt geometry, the direct transmission path naturally produces Fano interference with
    the resonant path. The `fano_asymmetry` parameter controls the degree of asymmetry.

    Note: This is equivalent to `LinearShuntFitter` with the parameter renamed `fano_asymmetry`
    instead of `asymmetry`.
    """

    def __init__(self, frequency, data, background_model=None, errors=None, **kwds):
        """
        Fit the given data to a composite model that is the product of a background model and the FanoShunt model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data.
        :param background_model: an instance (not the class) of a model representing the background response without the
          resonator; the default of `background.MagnitudePhase()` assumes that this is modeled well by a single complex
          constant at all frequencies.
        :param errors: an array of complex numbers containing the standard errors of the mean of the data points.
        :param kwds: keyword arguments passed directly to `lmfit.model.Model.fit()`.
        """
        if background_model is None:
            background_model = background.MagnitudePhase()
        super(FanoShuntFitter, self).__init__(frequency=frequency, data=data,
                                              foreground_model=FanoShunt(),
                                              background_model=background_model, errors=errors, **kwds)

    def invert(self, scattering_data):
        """
        Return the resonator detuning and internal_loss corresponding to the given normalized scattering data.

        Inverts: S21 = 1 - (1 + j*q) / (1 + (internal_loss + 2j*detuning) / coupling_loss)

        :param scattering_data: normalized complex scattering data (S21 at the resonator plane).
        :return: detuning, internal_loss (both array[float])
        """
        z = self.coupling_loss * ((1 + 1j * self.fano_asymmetry) / (1 - scattering_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss


# Fano transmission models and fitters

class FanoSymmetricTransmission(AbstractSymmetricTransmission):
    """
    This class models a resonator operated in transmission with a Fano-type asymmetric lineshape,
    assuming equal coupling losses at both ports.

    The model is:
        S21 = (1 + j * fano_asymmetry) / (1 + (internal_loss + 2j * detuning) / coupling_loss)

    where detuning = frequency / resonance_frequency - 1.

    When `fano_asymmetry` is zero, this reduces exactly to `LinearSymmetricTransmission`.
    """

    def __init__(self, *args, **kwds):
        def fano_symmetric_transmission(frequency, resonance_frequency, coupling_loss, internal_loss, fano_asymmetry):
            detuning = frequency / resonance_frequency - 1
            return (1 + 1j * fano_asymmetry) / (1 + (internal_loss + 2j * detuning) / coupling_loss)

        super(FanoSymmetricTransmission, self).__init__(func=fano_symmetric_transmission, *args, **kwds)

    def guess(self, data, frequency=None, coupling_loss=None):
        params = self.make_params()
        smoothed_magnitude = guess.smooth(np.abs(data))
        peak_index = np.argmax(smoothed_magnitude)
        resonance_frequency_guess = frequency[peak_index]
        params['resonance_frequency'].set(value=resonance_frequency_guess, min=frequency.min(), max=frequency.max())
        power_minus_half_max = smoothed_magnitude ** 2 - smoothed_magnitude[peak_index] ** 2 / 2
        f1 = np.interp(0, power_minus_half_max[:peak_index], frequency[:peak_index])
        f2 = np.interp(0, -power_minus_half_max[peak_index:], frequency[peak_index:])
        linewidth = f2 - f1
        internal_plus_coupling = linewidth / resonance_frequency_guess
        internal_over_coupling = (1 / np.abs(data[peak_index]) - 1)
        if coupling_loss is None:
            params['coupling_loss'].set(value=internal_plus_coupling / (1 + internal_over_coupling),
                                        min=1e-12, max=1)
            params['internal_loss'].set(value=(internal_plus_coupling * internal_over_coupling /
                                               (1 + internal_over_coupling)),
                                        min=1e-12, max=1)
        else:
            params['coupling_loss'].set(value=coupling_loss, vary=False)
            params['internal_loss'].set(value=internal_plus_coupling - coupling_loss, min=1e-12, max=1)
        params['fano_asymmetry'].set(value=0, min=-100, max=100)
        return params


class FanoTransmissionFitter(linear.LinearResonatorFitter):
    """
    This class fits data from a resonator operated in transmission with a Fano-type asymmetric
    lineshape, assuming equal coupling losses at both ports.

    Because the off-resonance transmission goes to zero, the background magnitude must be provided
    to anchor the fit. The `fano_asymmetry` parameter controls the degree of lineshape asymmetry;
    when zero, this is equivalent to `CCxSTFitterKnownMagnitude`.
    """

    def __init__(self, frequency, data, background_magnitude, errors=None, **kwds):
        """
        Fit the given data to a composite model that is the product of a MagnitudePhase background
        and the FanoSymmetricTransmission model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data.
        :param background_magnitude: the transmission magnitude in the absence of the resonator (not in dB).
        :param errors: an array of complex numbers containing the standard errors of the mean of the data points.
        :param kwds: keyword arguments passed directly to `lmfit.model.Model.fit()`.
        """
        self.background_magnitude = background_magnitude
        super(FanoTransmissionFitter, self).__init__(frequency=frequency, data=data,
                                                     foreground_model=FanoSymmetricTransmission(),
                                                     background_model=background.MagnitudePhase(),
                                                     errors=errors, **kwds)

    def guess(self, frequency, data):
        phase_guess = np.angle(data[np.argmax(np.abs(data))])
        params = self.background_model.make_params(magnitude=self.background_magnitude, phase=phase_guess)
        params['magnitude'].vary = False
        background_values = self.background_model.eval(params=params, frequency=frequency)
        params.update(self.foreground_model.guess(data=data / background_values, frequency=frequency))
        return params

    def invert(self, scattering_data):
        """
        Return the resonator detuning and internal_loss corresponding to the given normalized scattering data.

        Inverts: S21 = (1 + j*q) / (1 + (internal_loss + 2j*detuning) / coupling_loss)

        :param scattering_data: normalized complex scattering data (S21 at the resonator plane).
        :return: detuning, internal_loss (both array[float])
        """
        z = self.coupling_loss * ((1 + 1j * self.fano_asymmetry) / scattering_data - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss
