from setuptools import setup
import torch
from torch.utils.cpp_extension import CppExtension



extra_compile_args = ['-g']

ext_modules = [
    CppExtension('pygrid.grid_cpp', ['src/grid.cpp']),
]
cmdclass = {'build_ext': torch.utils.cpp_extension.BuildExtension}




__version__ = '1.0.0'

install_requires = ['scipy']
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='torch_extension',
    version=__version__,
    description=('PyTorch Extension'),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
