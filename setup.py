#import setuptools
from setuptools import setup, Extension
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('frat/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='dre',
    version=main_ns['__version__'],
    packages=['dre'],
    package_data={'dre': ['resources/*.*']},
    scripts=['bin/word_embedder', 'bin/frat_json2png'],
    license='GPLv3',
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/dre',
    description="Fast Rectangle Annotation Tool",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    keywords=["documents", "regex", "semantic"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"],
    install_requires=["torch", "sly", "nltk", "fargv", "tqdm", "numpy"],
)