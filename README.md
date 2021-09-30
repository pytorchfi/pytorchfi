<h1 align="center">
  <a href="https://pytorchfi.dev/"><img src="https://user-images.githubusercontent.com/7104017/75485879-22e79400-5971-11ea-9376-2d898034c23a.png" width="150"></a>
  <br/>
    PyTorchFI
  </br>
</h1>

<p align="center">
    <img src="https://img.shields.io/circleci/build/github/pytorchfi/pytorchfi/master"></img>
    <a href="https://codecov.io/gh/pytorchfi/pytorchfi"><img src="https://codecov.io/gh/pytorchfi/pytorchfi/branch/master/graph/badge.svg"></a>
    <a href="https://pypi.org/project/pytorchfi/"><img src="https://static.pepy.tech/personalized-badge/pytorchfi?period=total&units=international_system&left_color=grey&right_color=magenta&left_text=downloads"></a>
<a href="https://app.fossa.com/projects/git%2Bgithub.com%2Fpytorchfi%2Fpytorchfi?ref=badge_shield" alt="FOSSA Status"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Fpytorchfi%2Fpytorchfi.svg?type=shield"/></a>
    <a href="https://opensource.org/licenses/NCSA"><img src="https://img.shields.io/badge/license-NCSA-blue"></a>
</p>

<p align="center">
  <a href="#background">Background</a> •
  <a href="#usage">Usage</a> •
  <a href="#code">Code</a> •
  <a href="#contributors">Contributors</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>


[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fpytorchfi%2Fpytorchfi.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fpytorchfi%2Fpytorchfi?ref=badge_large)

## Background

PyTorchFI is a runtime perturbation tool for deep neural networks (DNNs), implemented for the popular PyTorch deep learning platform. PyTorchFI enables users to perform perturbation on weights or neurons of a DNN during runtime. It is extremely versatile for dependability and reliability research, with applications including resiliency analysis of classification networks, resiliency analysis of object detection networks, analysis of models robust to adversarial attacks, training resilient models, and for DNN interpertability.

An example of a use case for PyTorchFI is to simulate an error by performaing a fault-injection on an object recognition model.
|                                              Golden Output                                               |                                       Output with Fault Injection                                        |
| :------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
| ![](https://user-images.githubusercontent.com/7104017/85642872-7fb93980-b647-11ea-8717-8d16cb1c35b3.jpg) | ![](https://user-images.githubusercontent.com/7104017/85642867-7def7600-b647-11ea-89b9-570278c22101.jpg) |

## Usage

Download on PyPI [here](https://pypi.org/project/pytorchfi/), or take a look at our documentation at [pytorchfi.dev](https://pytorchfi.dev/).

You can also learn more with our [interactive demo](https://colab.research.google.com/drive/1BMB4LbsTU_K_YXUFzRyfIynpGu5Yhr1Y).


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

The main source code of PyTorchFI is held in `pytorchfi`, which carries both `core` and `error_models.py` implementations.

### Formatting

All python code is formatted with [black](https://black.readthedocs.io/en/stable/).

## Contributors

Before contributing, please refer to our [contributing guidelines](https://github.com/pytorchfi/pytorchfi/blob/master/CONTRIBUTING.md).

- [Sarita V. Adve](http://sadve.cs.illinois.edu/) (UIUC)
- [Neeraj Aggarwal](https://neerajaggarwal.com) (UIUC)
- [Christopher W. Fletcher](http://cwfletcher.net/) (UIUC)
- [Siva Kumar Sastry Hari](https://research.nvidia.com/person/siva-hari) (NVIDIA)
- [Abdulrahman Mahmoud](http://amahmou2.web.engr.illinois.edu/) (UIUC)
- [Alex Nobbe](https://github.com/Alexn99) (UIUC)
- [Jose Rodrigo Sanchez Vicarte](https://jose-sv.github.io/) (UIUC)

## Citation

View the [published paper](http://rsim.cs.illinois.edu/Pubs/20-DSML-PyTorchFI.pdf). If you use or reference PyTorchFI, please cite:

```
@INPROCEEDINGS{PytorchFIMahmoudAggarwalDSML20,
author={A. {Mahmoud} and N. {Aggarwal} and A. {Nobbe} and J. R. S. {Vicarte} and S. V. {Adve} and C. W. {Fletcher} and I. {Frosio} and S. K. S. {Hari}},
booktitle={2020 50th Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W)},
title={PyTorchFI: A Runtime Perturbation Tool for DNNs},
year={2020},
pages={25-31},
}
```

## Funding Sources

This project was funded in part by the [Applications Driving Architectures (ADA) Research Center](https://adacenter.org/), a JUMP Center co-sponsored by SRC and DARPA, and in collaboration with NVIDIA Research.

## License

[NCSA](https://opensource.org/licenses/NCSA) License. Copyright © 2021 [RSim Research Group](http://rsim.cs.uiuc.edu/).