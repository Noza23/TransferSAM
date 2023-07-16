from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SAMTransfer',
    url='https://github.com/Noza23/TransferSAM',
    author='Giorgi Nozadze',
    author_email='giorginozadze23@yahoo.com',
    long_description=long_description,
    packages=find_packages(exclude="notebooks"),
    # Needed for dependencies
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'torch',
        'torchvision'
        'cv2',
        'nibabel',
        'monai'
    ],
    version='0.1',
    license='MIT',
    description='Fine-Tune, retrain SAM Mask-Decoder',
    python_requires='>=3.9'
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)