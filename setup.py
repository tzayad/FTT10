from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ftt10',
    url='https://github.com/jladan/package_demo',
    author='John Ladan',
    author_email='jladan@uwaterloo.ca',
    # Needed to actually package something
    packages=['ftt10'],
    # Needed for dependencies
    install_requires=['pytrends', 'yfinance'],
    # *strongly* suggested for sharing
    version='10.01',
    # The license can be anything you like
    license='MITULIM',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
