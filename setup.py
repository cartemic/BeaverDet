from codecs import open
from os import path
import sys
from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'beaverdet', '_version.py')) as version_file:
    exec(version_file.read())

with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

desc = readme + '\n\n' + changelog
try:
    import pypandoc
    long_description = pypandoc.convert_text(desc, 'rst', format='md')
    with open(path.join(here, 'README.rst'), 'w') as rst_readme:
        rst_readme.write(long_description)
except (ImportError, OSError, IOError):
    long_description = desc

install_requires = [
    'python',
    'pandas',
    'numpy',
    'sympy',
    'pint',
    'cantera',
    'nptdms',
    'scipy'
]

tests_require = [
    'pytest',
    'pytest-cov',
    'mock'
]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(
    name='beaverdet',
    version=__version__,
    description='A collection of detonation tools',
    long_description=desc,
    author='Mick Carter',
    author_email='cartemic@oregonstate.edu',
    url='https://github.com/cartemic/BeaverDet',
    license='MIT',
    python_requires='>=3.6.*',
    packages=['beaverdet',
              'beaverdet.tests'],
    package_dir={'beaverdet': 'beaverdet'},
    package_data={'beaverdet': ['lookup_data/*', 'tests/test_data/*'],
                  'beaverdet.tests': ['lookup_data/*', 'tests/test_data/*']},
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires
    )
