from setuptools import setup

setup(name='vae',
      version='0.1',
      description='Variational Autoencoders',
      license='MIT',
      packages=['vae'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)