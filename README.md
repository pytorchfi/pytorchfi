# PyTorchFI

PyTorchFI is a runtime fault injector tool for PyTorch to simulate bit flips within the neural network. Check us out on PyPI [here](https://pypi.org/project/pytorchfi/).

## Installation

### Via Pip

Install using `pip install pytorchfi` Then in your project source files:

```python
import pytorchfi
```

### From Source

Download this repository into your project folder. Then in your project source files:

```python
from src import PyTorchFI_Core
```

## Documentation

The documentation can be found at [https://pytorchfi.github.io/docs/](https://pytorchfi.github.io/docs/).

## Code

### Structure

The main source code of PyTorchFI is held in `src`, which carries both `Core` and `Util` implementations.

### Formatting

All python code is formatted with [black](https://black.readthedocs.io/en/stable/).

## Contributors

- [Sarita V. Adve](http://sadve.cs.illinois.edu/) (UIUC)
- [Neeraj Aggarwal](https://neerajaggarwal.com) (UIUC)
- [Christopher W. Fletcher](http://cwfletcher.net/) (UIUC)
- [Siva Kumar Sastry Hari](https://research.nvidia.com/person/siva-hari) (NVIDIA)
- [Abdulrahman Mahmoud](http://amahmou2.web.engr.illinois.edu/) (UIUC)
- [Alex Nobbe](https://github.com/Alexn99) (UIUC)

## License

[NCSA License](https://opensource.org/licenses/NCSA)
