<h1 align="center">
  <a href="https://pytorchfi.github.io/"><img src="https://user-images.githubusercontent.com/7104017/75485879-22e79400-5971-11ea-9376-2d898034c23a.png" width="150"></a>
  <br/>
    PyTorchFI
  </br>
</h1>

<p align="center">
    <a href="https://pypi.org/project/pytorchfi/"><img src="https://img.shields.io/pypi/dm/pytorchfi?color=da67f7"></a>
    <a href="https://opensource.org/licenses/NCSA"><img src="https://img.shields.io/badge/license-NCSA-blue"></a>
</p>

<h4 align="center">A project by <a href="http://rsim.cs.uiuc.edu/" target="_blank">RSim Research Group</a> in collaboration with <a href="https://www.nvidia.com/en-us/research/" target="_blank">NVIDIA Research.</a></h4>

<p align="center">
  <a href="#background">Background</a> •
  <a href="#usage">Usage</a> •
  <a href="#technologies">Code</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#contributors">Contributors</a> •
  <a href="#license">License</a>
</p>

## Background

PyTorchFI is a runtime fault injector tool for PyTorch to simulate bit flips within the neural network. Check us out on PyPI [here](https://pypi.org/project/pytorchfi/).

The documentation can be found at [pytorchfi.github.io](https://pytorchfi.github.io/).

## Usage

### Installing

**From Pip**

Install using `pip install pytorchfi`

**From Source**

Download this repository into your project folder.

### Importing

Import the entire package:

```python
import pytorchfi
```

Import a specific module:

```python
from pytorchfi import core
```

## Code

### Structure

The main source code of PyTorchFI is held in `pytorchfi`, which carries both `core` and `util` implementations.

### Formatting

All python code is formatted with [black](https://black.readthedocs.io/en/stable/).

## Contributing

Before contributing, please refer to our [contributing guidelines](https://github.com/pytorchfi/pytorchfi/blob/master/CONTRIBUTING.md).

## Contributors

- [Sarita V. Adve](http://sadve.cs.illinois.edu/) (UIUC)
- [Neeraj Aggarwal](https://neerajaggarwal.com) (UIUC)
- [Christopher W. Fletcher](http://cwfletcher.net/) (UIUC)
- [Siva Kumar Sastry Hari](https://research.nvidia.com/person/siva-hari) (NVIDIA)
- [Abdulrahman Mahmoud](http://amahmou2.web.engr.illinois.edu/) (UIUC)
- [Alex Nobbe](https://github.com/Alexn99) (UIUC)

## License

[NCSA](https://opensource.org/licenses/NCSA) License. Copyright © 2020 [RSim Research Group](http://rsim.cs.uiuc.edu/).
