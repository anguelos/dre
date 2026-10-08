from setuptools import setup, find_packages

version = {}
with open('src/dre/version.py') as f:
    exec(f.read(), version)

setup(
    name='dre',
    version=version['__version__'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    license='GPLv3',
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/dre',
    description='Differentiable probabilistic regex engine over NFA transition matrices',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    keywords=['regex', 'NFA', 'probabilistic', 'semantic', 'nlp'],
    python_requires='>=3.9',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
    ],
    install_requires=['numpy', 'sly', 'fargv', 'matplotlib', 'networkx', 'torch', 'tqdm', 'pylelemmatize'],
    entry_points={
        'console_scripts': [
            'phoc_train_semantic=phoc.train_semantic:main',
            'phoc_train_decoder=phoc.train_decoder:main',
        ],
    },
    extras_require={
        'jax': ['jax'],
        'nltk': ['nltk'],
        'test': ['pytest', 'pytest-cov', 'coverage'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'myst-parser'],
        'dev': ['ipython', 'graphviz'],
    },
)
