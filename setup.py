from data.reader import document
from config import DOC_PATH, DATA_SAVE_PATH



def data_(document_path, document_save_path, preparation_fn): 
    doc = document(document_path, preparation_fn)

    save_path = document_save_path
    doc.to_txt(save_path, progressbar=True)

    return doc, save_path

def run(): 
    doc, path = data_()

if __name__ == "__main__": 
    run() 