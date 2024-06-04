from setuptools import setup, find_packages

setup(
    name='lora_to_pdf',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',  
        'datasets',  
        'peft', 
        'transformers[torch]', 
        'pymupdf', 
    ],
    entry_points={
        'console_scripts': [
            'run=base:main'
        ]
    }
)