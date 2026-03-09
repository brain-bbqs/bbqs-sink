import warnings

import numpy
import tqdm


def get_trials_info(
    trial_trace: numpy.ndarray,
    event_trace: numpy.ndarray,
    sampling_rate: float,
    voltage_range: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    A basic utility function to decode the trial information from the TTL channels in the SpikeGLX recordings.

    The trial information is encoded in a sequence of 4 hexadecimal digits, where each digit is represented by a
    voltage level on the 'event signal' channel.

    The 'trial onset signal' channel is used to identify the start and end of each trial.

    Parameters
    ----------
    trial_trace : numpy array
        The extracted voltage trace corresponding to the 'trial onset signal' (TTL channel) from the recording.
    event_trace : numpy array
        The extracted voltage trace corresponding to the 'event signal' (TTL channel) from the recording.
    sampling_rate : float
        The sampling rate of the recording, used to convert sample indices to time.
    voltage_range : float
        The voltage range of the TTL signals, used to scale the event trace to the appropriate range for decoding.
        Example values used for NIDQ boards might be around `4.5 * 1e6`.

    Returns
    -------
    trial_numbers: numpy array
        Array with decoded trial ID for each trial.
    trial_times: numpy array
        Array with `t_start` and `t_stop` timestamps for each trial.
    """
    hex_base = 16
    hex_dict = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
    }

    tenms_interval = int(0.01 * sampling_rate)

    scaled_tr_events = event_trace * (hex_base - 1) / voltage_range
    scaled_tr_events = (
        (scaled_tr_events - numpy.median(scaled_tr_events)) / numpy.max(scaled_tr_events) * (hex_base - 1)
    )

    tr_trial_bin = numpy.zeros(trial_trace.shape, dtype=int)
    tr_trial_bin[trial_trace > numpy.max(trial_trace) // 2] = 1

    t_start_idxs = numpy.where(numpy.diff(tr_trial_bin) > 0)[0]
    t_stop_idxs = numpy.where(numpy.diff(tr_trial_bin) < 0)[0]

    if len(t_start_idxs) == 0 or len(t_stop_idxs) == 0:
        message = "No trial start or stop events found in the provided traces!"
        raise ValueError(message)

    # discard first stop event if it comes before a start event
    if t_stop_idxs[0] < t_start_idxs[0]:
        warnings.warn(message="Discarding first trial", stacklevel=2)
        t_stop_idxs = t_stop_idxs[1:]

    # discard last start event if it comes after last stop event
    if t_start_idxs[-1] > t_stop_idxs[-1]:
        warnings.warn(message="Discarding last trial", stacklevel=2)
        t_start_idxs = t_start_idxs[:-1]

    trial_numbers = []
    trial_times = []

    for t in tqdm.tqdm(iterable=range(len(t_start_idxs)), desc="Parsing hex signals"):
        start_idx = t_start_idxs[t]
        stop_idx = t_stop_idxs[t]

        trial_times.append(numpy.array([start_idx, stop_idx]) / sampling_rate)

        i_start = start_idx
        trial_digits = ""

        for i in range(4):
            digit = numpy.median(scaled_tr_events[i_start + 10 : i_start + tenms_interval - 10])
            digit = int(round(digit))
            trial_digits += hex_dict[digit]
            i_start += tenms_interval
        trial_numbers.append(int(trial_digits, hex_base))

    trial_number_array = numpy.array(trial_numbers)
    trial_times_array = numpy.array(trial_times)
    return trial_number_array, trial_times_array
