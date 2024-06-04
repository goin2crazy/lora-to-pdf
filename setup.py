from setuptools import setup, find_packages

setup(
    name='pdf_to_lora',
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
            'pdf_to_lora=.base:main'
        ]
    }
)
