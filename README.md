<h1 align="center">
  <a href="https://pytorchfi.github.io/"><img src="https://user-images.githubusercontent.com/7104017/75485879-22e79400-5971-11ea-9376-2d898034c23a.png" width="150"></a>
  <br/>
    PyTorchFI
  </br>
</h1>

<p align="center">
    <img src="https://img.shields.io/circleci/build/github/pytorchfi/pytorchfi/master"></img>
    <a href="https://codecov.io/gh/pytorchfi/pytorchfi"><img src="https://codecov.io/gh/pytorchfi/pytorchfi/branch/master/graph/badge.svg"></a>
    <a href="https://pypi.org/project/pytorchfi/"><img src="https://img.shields.io/pypi/dm/pytorchfi?color=da67f7"></a>
    <a href="https://opensource.org/licenses/NCSA"><img src="https://img.shields.io/badge/license-NCSA-blue"></a>
</p>

<p align="center">
  <a href="#background">Background</a> •
  <a href="#usage">Usage</a> •
  <a href="#code">Code</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#contributors">Contributors</a> •
  <a href="#license">License</a>
</p>

## Background

PyTorchFI is a runtime perturbation tool for deep neural networks (DNNs), implemented for the popular PyTorch deep learning platform. PyTorchFI enables users to perform perturbation on weights or neurons of a DNN during runtime. It is extremely versatile for dependability and reliability research, with applications including resiliency analysis of classification networks, resiliency analysis of object detection networks, analysis of models robust to adversarial attacks, training resilient models, and for DNN interpertability.

For example, this is an object detection network before a fault injection

|                                              Golden Output                                               |                                       Output with Fault Injection                                        |
| :------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
| ![](https://user-images.githubusercontent.com/7104017/85642872-7fb93980-b647-11ea-8717-8d16cb1c35b3.jpg) | ![](https://user-images.githubusercontent.com/7104017/85642867-7def7600-b647-11ea-89b9-570278c22101.jpg) |

## Usage

Download on PyPI [here](https://pypi.org/project/pytorchfi/), or take a look at our documentation at [pytorchfi.github.io](https://pytorchfi.github.io/).

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

### Testing

```bash
pytest
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

## Funding Sources

This project was funded in part by the [Applications Driving Architectures (ADA) Research Center](https://adacenter.org/), a JUMP Center co-sponsored by SRC and DARPA, and in collaboration with NVIDIA Research.

## License

[NCSA](https://opensource.org/licenses/NCSA) License. Copyright © 2020 [RSim Research Group](http://rsim.cs.uiuc.edu/).
