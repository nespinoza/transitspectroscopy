# Adding an observing mode

Everything `transitspectroscopy` needs to know about an instrument mode is one
{class}`~transitspectroscopy.instruments.modes.Mode` in the mode registry
(`transitspectroscopy.instruments`). The JWST stages look the mode up from the data's
metadata; they do not branch on mode names. Adding a mode therefore means registering one
`Mode`, plus, if the mode needs it, a Stage 2 plug-in.

## The built-in modes

```python
from transitspectroscopy import instruments

instruments.list_modes()
# ['miri/photometry', 'niriss/soss', 'nirspec/g395h', 'nirspec/g395m', 'nirspec/prism']

mode = instruments.get_mode('nirspec/g395h')
mode.detector_steps['tso_jump']          # {'window': 10, 'nsigma': 10}
mode.tracing_for('nrs2')['xstart']       # 5
mode.extraction_defaults['aperture_radius']
```

`ts.jwst.load(...).mode_definition` holds the mode of a loaded dataset.

## What a mode defines

| Field | Used by | Meaning |
|---|---|---|
| `match` | identification | Rules on `instrument`, `dispersive_element`, `filter`, `exp_type`, `detector` |
| `detector_steps` | `load.detector_calibration`, `stage1` | Reference-pixel treatment, calwebb_detector1 step order, TSO-jump window/nsigma, NaN-fill window |
| `tracing` | spectral `stage2` | Per-detector start/end columns, spline knots, cross-correlation kernel |
| `extraction_defaults` | `stage2` | Aperture, 1/f and background apertures, optimal-extraction settings |
| `stage2` | `stage2` | `'spectral'`, `'soss'`, `'imaging'`, or a callable plug-in |
| `background_strategy`, `trace_source`, `wavelength_source`, `known_limitations` | documentation and plug-ins | How the mode treats backgrounds, traces and wavelengths |

## A new spectroscopic mode

A single-trace spectroscopic mode reuses the spectral Stage 2 and only needs its numbers.
For example, a NIRSpec grating not yet supported:

```python
from transitspectroscopy import instruments
from transitspectroscopy.instruments import Mode

g235h = instruments.get_mode('nirspec/g395h')   # start from a similar mode
instruments.register(Mode(
    name='nirspec/g235h', observatory='JWST', instrument='nirspec', label='NIRSpec/G235H',
    kind='slit_spectroscopy',
    match=({'instrument': 'nirspec', 'dispersive_element': 'g235h'},),
    detectors=('nrs1', 'nrs2'),
    detector_steps=g235h.detector_steps,
    tracing={'nrs1': dict(g235h.tracing['nrs1'], xend=600),   # values to be validated!
             'nrs2': dict(g235h.tracing['nrs2'])},
    extraction_defaults=g235h.extraction_defaults,
    known_limitations='Tracing limits not yet validated on real data.',
))
```

After this, `ts.jwst.load(files)` recognizes G235H data and every stage uses the new values.
Validate the numbers on real data before relying on them, and contribute the mode by adding
it to `src/instruments/jwst_modes.py` with a synthetic test in `tests/test_instruments.py`.

## A mode with its own Stage 2

When a mode needs a different Stage 2 (several orders, imaging, a different background
treatment...), give it a callable. It receives the `stage2` inputs and a validated
{class}`~transitspectroscopy.config.ExtractionConfig`:

```python
def my_stage2(input_dictionary, config, outputfolder, suffix):
    rates = [model.data for model in input_dictionary['rampstep']]
    ...  # use transitspectroscopy.core for tracing, extraction or photometry
    return results

instruments.register(Mode(name='nircam/grism', observatory='JWST', instrument='nircam',
                          label='NIRCam/Grism', kind='slitless_spectroscopy',
                          match=({'instrument': 'nircam', 'exp_type': 'nrc_tsgrism'},),
                          stage2=my_stage2))
```

`ts.jwst.stage2(inputs, config=ExtractionConfig(...))` then dispatches to `my_stage2`.
The algorithms in `transitspectroscopy.core` work on plain arrays, so a plug-in can combine
them freely.
