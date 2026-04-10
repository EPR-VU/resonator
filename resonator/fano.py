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

from . import background, guess, linear
from .reflection import AbstractReflection
from .shunt import AbstractShunt


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
