# Decode TTL Sequence

TTL pulses can be sent as simple on/off trigger times, but they can also be sent as a sequence of pulses that encode information in the timing or amplitude of the pulses.
For example, a sequence of 3 pulses with varying voltage levels could encode 3 bits of information (`000`, `001`, `010`, `011`, `100`, `101`, `110`, `111`).

This utility provides a way that has been used to decode voltage-varying TTL sequences into integer values.
This specific method was used by the [Feldman lab](https://www.feldmanlab.org/) at UC Berkeley to encode the trial number during an experiment using Neuropixels probes with a NIDQ board.
The TTL pulses were scaled into an intermediate hex-code (16-bit) representation, which is mapped to the integer at the final step.

This code has been adapted from a previous [data conversion project](https://github.com/catalystneuro/feldman-lab-to-nwb/blob/6938040c2846ff6ca4f5a4530e9102960927e006/feldman_lab_to_nwb/utils.py#L1C1-L366C61).



### Installation

Using any Python setup, install the minimal dependencies with:

```bash
pip install numpy tqdm
```
