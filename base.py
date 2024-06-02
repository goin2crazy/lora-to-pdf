import argparse
from .data import document
from .nn import train, TrainConfig

def data_(document_path,  preparation_fn): 
    doc = document(document_path, preparation_fn)

    return doc.read_all()

def train_nn(text_data, config: TrainConfig): 
    model = train(text_data, cfg = config)
    return model 

def run(doc_path, 
        preparation_fn = None, 
        config:TrainConfig = None): 

    if type(doc_path) == list:
        text_data = [data_(dp, preparation_fn) for dp in doc_path]
    else: 
        text_data = data_(doc_path, preparation_fn)

    default_config = TrainConfig() if (config == None) else config

    model = train_nn(text_data, default_config)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process document and train neural network")
    
    parser.add_argument('--doc_path', type=str, required=True, help='Path to the document')

    args = parser.parse_args()

    run(args.doc_path)
