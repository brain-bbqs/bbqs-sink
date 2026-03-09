import typing

import numpy
import scipy.signal


def parse_passband(passband: typing.Literal["delta", "theta", "spindles", "gamma", "ripples"]) -> numpy.ndarray:
    """
    Parse the passband argument for `filter_lfp.`

    This function allows users to specify either a custom passband or one of several canonical bands.

    Parameters
    ----------
    passband : one of the following strings
        (low, high) of bandpass filter or one of the following canonical bands.
            'delta':    (  0,   4)
            'theta':    (  4,  10)
            'spindles': ( 10,  20)
            'gamma':    ( 30,  80)
            'ripples':  (100, 250)
    """
    if passband == "delta":
        passband = numpy.array([0, 4])
    elif passband == "theta":
        passband = numpy.array([4, 10])
    elif passband == "spindles":
        passband = numpy.array([10, 20])
    elif passband == "gamma":
        passband = numpy.array([30, 80])
    elif passband == "ripples":
        passband = numpy.array([100, 250])

    return passband


def filter_lfp(
    lfp: numpy.ndarray,
    passband: typing.Literal["delta", "theta", "spindles", "gamma", "ripples"],
    sampling_rate: float = 1250.0,
    order: int = 4,
    filter: str = "butter",
) -> numpy.ndarray:
    """
    Apply a passband filter to a signal.

    `scipy.signal.butter` is implemented but other filters are not.

    Parameters
    ----------
    lfp: numpy.ndarray
        LFP signal to be filtered.
    sampling_rate: float, default: 1250.0
        Sampling rate of LFP signal in Hz.
        The default value was chosen from use in a previous group.
    passband: str
        The name of the canonical bandpass filter. See `parse_passband` for options.
    order: int, default: 4
        The order of the filter.
    filter: str, default: "butter"
        The type of filter to apply.
        Currently, only "butter" is implemented.

    Returns
    -------
    filt: numpy.ndarray
        The filtered signal.
    """
    if filter == "butter":
        passband = parse_passband(passband)
        b, a = scipy.signal.butter(order, passband / (sampling_rate / 2), "bandpass")
        filt = scipy.signal.filtfilt(b, a, lfp)
        return filt
    else:
        message = f"Filter type {filter} not implemented!"
        raise NotImplementedError(message)


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def hilbert_lfp(filt: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculate the phase and amplitude of a filtered signal.

    Parameters
    ----------
    filt : numpy.array
        Filtered lfp signal. Usually, this is the output of filter_lfp.

    Returns
    -------
    phase : np.ndarray
    amplitude : np.ndarray
    """
    # scipy.signal.hilbert runs way faster on a power of 2
    hilb = scipy.signal.hilbert(filt, _next_power_of_2(len(filt)))
    hilb = hilb[: len(filt)]

    amp = numpy.abs(hilb)
    phase = numpy.mod(numpy.angle(hilb), 2 * numpy.pi)

    return phase, amp
