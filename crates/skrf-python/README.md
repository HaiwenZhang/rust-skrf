# skrf-python

Python bindings for [skrf-core](../skrf-core) - RF/microwave engineering in Rust.

## Installation

```bash
pip install skrf_python
```

## Features

- High-performance Touchstone file parsing
- S/Z/Y/ABCD parameter transformations
- Network operations (cascade, de-embed, interpolation)
- Vector fitting for rational function approximation

## Quick Start

```python
import skrf_python

# Load a Touchstone file
network = skrf_python.Network.from_touchstone("device.s2p")

# Access S-parameters
s_params = network.s
freqs = network.f

# Perform vector fitting
vf = skrf_python.VectorFitting()
vf.fit(network, n_poles_real=4)
```

## Requirements

- Python >= 3.10
- NumPy >= 2.0

## License

BSD-3-Clause
