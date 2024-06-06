# PDF to LoRA

`pdf_to_lora` is a Python package designed to handle PDF files and perform operations using machine learning models with the help of libraries such as `torch`, `transformers`, `datasets`, and `peft`.

## Installation

To install the package, you can use the following command:

```bash
pip install .
```

This will install all the required dependencies listed in the `setup.py` file:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `transformers[torch]`
- `pymupdf`

## Usage

The package provides various functionalities, including reading PDFs and training models. Below is an example of how to use the main functionalities of the package: 

### Reading PDFs

The `base.py` module is used to read PDF files. Here's an example:

```bash 
python base.py --doc_path "[PATH TO PDF DOCUMENT]" --mode "preview" --page 3
```

### Training Models

The package includes scripts for training models using the provided datasets and configurations. The main training script is located in `base.py`.

To start training, you can use the following command:

```bash
python base.py --doc_path "[PATH TO PDF DOCUMENT]" --mode "train" --extra_words "http://" --num_epoch 5 --batch_size 1 --chunk_size 170 --save_steps 500
```

### Configuration

Training configurations are defined in `nn/train_config.py`. You can modify this file to adjust the training parameters according to your needs.

## Entry Points

The package includes a console script entry point defined in `setup.py`. This allows you to run the main script directly from the command line:

```bash
pdf_to_lora
```

## Project Structure

The repository is structured as follows:

```
pdf_to_lora/
├── data
│   ├── __init__.py
│   ├── reader.py
│   ├── example
│   │   └── h.pdf
├── nn
│   ├── __init__.py
│   ├── dataset.py
│   ├── inference.py
│   ├── train.py
│   ├── train_config.py
│   ├── trainer.py
├── .gitignore
├── base.py
├── requirements.txt
├── setup.py
```

- `data/`: Contains modules related to data reading and processing.
- `nn/`: Contains modules and scripts for neural network training and inference.
- `base.py`: Base script for the package.
- `requirements.txt`: List of additional dependencies.
- `setup.py`: Setup script for installing the package.

## Dependencies

The package requires the following libraries:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `transformers[torch]`
- `pymupdf`

## Acknowledgments

This project uses open-source libraries and datasets. We acknowledge the contributions of the developers and the community for providing these valuable resources.

