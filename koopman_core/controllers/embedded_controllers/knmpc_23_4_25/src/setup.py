from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import distutils.sysconfig as sysconfig
from platform import system
from glob import glob
import os
import shutil as sh
from subprocess import call


class build_ext_osqp(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


'''
Define macros
'''
# Pass EMBEDDED flag to cmake to generate osqp_configure.h
# and qdldl_types.h files
cmake_args = []
embedded_flag = 2
cmake_args += ['-DEMBEDDED:INT=%i' % embedded_flag]

# Pass Python flag to compile interface
define_macros = []
define_macros += [('PYTHON', None)]

# Generate glob_opts.h file by running cmake
current_dir = os.getcwd()
os.chdir('..')
if os.path.exists('build'):
    sh.rmtree('build')
os.makedirs('build')
os.chdir('build')
call(['cmake'] + cmake_args + ['..'], stdout=open(os.devnull, 'wb'))
os.chdir(current_dir)

'''
Define compiler flags
'''
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []

# Add additional libraries
libraries = []
if system() == 'Linux':
    libraries = ['rt']

'''
Include directory
'''
include_dirs = [os.path.join('..', 'include')]  # OSQP includes

'''
Source files
'''
sources_files = ['knmpc_23_4_25module.c']             # Python wrapper
sources_files += glob(os.path.join('osqp', '*.c'))      # OSQP files


knmpc_23_4_25 = Extension('knmpc_23_4_25',
                            define_macros=define_macros,
                            libraries=libraries,
                            include_dirs=include_dirs,
                            sources=sources_files,
                            extra_compile_args=compile_args)


setup(name='knmpc_23_4_25',
      version='0.6.2.post0',
      author='Bartolomeo Stellato, Goran Banjac',
      author_email='bartolomeo.stellato@gmail.com',
      description='This is the Python module for embedded OSQP: ' +
                  'Operator Splitting solver for Quadratic Programs.',
      setup_requires=["numpy >= 1.7"],
      install_requires=["numpy >= 1.7", "future"],
      license='Apache 2.0',
      cmdclass={'build_ext': build_ext_osqp},
      ext_modules=[knmpc_23_4_25])
