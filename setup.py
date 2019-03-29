from setuptools import setup, find_packages

with open('README.md') as f:
  readme = f.read()

setup(
  name='Tensorflow-YOLOv3',
  version='0.1',
  description=readme,
  author='kcosta',
  author_email='kcosta@student.42.fr',
  url='https://github.com/kcosta42/Tensorflow-YOLOv3',
  license='MIT',
  packages=find_packages(exclude=('tests', 'docs'))
)
