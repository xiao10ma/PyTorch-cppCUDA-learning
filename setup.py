from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    # package 的名称，比如叫pytorch，tensorflow,numpy 这样子
    name='cppcuda_tutorial',
    version='1.0',
    author='mzp',
    author_email='mazipei21@gmail.com',
    description='cppcuda example',
    long_description='cppcuda example',
    ext_modules=[
        CppExtension(
            name='cppcuda_tutorial',
            sources=['interpolation.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)