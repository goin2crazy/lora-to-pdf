from data.reader import document
from config import DOC_PATH, DATA_SAVE_PATH

def data_(): 
    doc = document(DOC_PATH)

    save_path = DATA_SAVE_PATH
    doc.to_txt(save_path, progressbar=True)

    return doc, save_path

def run(): 
    doc, path = data_()

if __name__ == "__main__": 
    run() 