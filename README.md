# PyTorchFI

![pypi shield](https://img.shields.io/pypi/dm/pytorchfi?color=da67f7)

PyTorchFI is a runtime fault injector tool for PyTorch to simulate bit flips within the neural network. Check us out on PyPI [here](https://pypi.org/project/pytorchfi/).

The documentation can be found at [pytorchfi.github.io](https://pytorchfi.github.io/).

### Installation

**From Pip**

Install using `pip install pytorchfi` Then in your project source files:

**From Source**

Download this repository into your project folder.

### Usage

Import the entire package:

```python
import pytorchfi
```

Import a specific module:

```python
from pytorchfi import core
```

### Code

#### Structure

The main source code of PyTorchFI is held in `pytorchfi`, which carries both `Core` and `Util` implementations.

#### Formatting

All python code is formatted with [black](https://black.readthedocs.io/en/stable/).

### Contributing

Before contributing, please refer to our [contributing guidelines](https://github.com/pytorchfi/pytorchfi/blob/master/CONTRIBUTING.md).

### Contributors

- [Sarita V. Adve](http://sadve.cs.illinois.edu/) (UIUC)
- [Neeraj Aggarwal](https://neerajaggarwal.com) (UIUC)
- [Christopher W. Fletcher](http://cwfletcher.net/) (UIUC)
- [Siva Kumar Sastry Hari](https://research.nvidia.com/person/siva-hari) (NVIDIA)
- [Abdulrahman Mahmoud](http://amahmou2.web.engr.illinois.edu/) (UIUC)
- [Alex Nobbe](https://github.com/Alexn99) (UIUC)

### License

[NCSA License](https://opensource.org/licenses/NCSA)