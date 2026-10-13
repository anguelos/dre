from setuptools import setup, find_packages

version = {}
with open('src/dre/version.py') as f:
    exec(f.read(), version)
optional = {}  # dre.optional.OPTIONAL lists every optional dependency; it imports nothing heavy
with open('src/dre/optional.py') as f:
    exec(f.read(), optional)

BARE_BONES = ['numpy', 'torch', 'sly', 'pylelemmatize', 'fargv', 'tqdm']
RECOMMENDED = sorted({pip_name for pip_name, _, _ in optional['OPTIONAL'].values()})  # apt needs: see dre/optional.py
TEST = ['pytest', 'pytest-cov', 'coverage'] + RECOMMENDED  # the tests exercise the optional features too
DOCS = ['sphinx', 'sphinx-rtd-theme', 'myst-parser']
DEV = sorted(set(TEST + DOCS + ['ipython', 'ruff', 'build']))

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
    python_requires='>=3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
    ],
    install_requires=BARE_BONES,
    entry_points={
        'console_scripts': [
            'phoc_train_semantic=phoc.train_semantic:main',
            'phoc_train_decoder=phoc.train_decoder:main',
        ],
    },
    extras_require={
        'recommended': RECOMMENDED,
        'test': TEST,
        'docs': DOCS,
        'dev': DEV,
    },
)
