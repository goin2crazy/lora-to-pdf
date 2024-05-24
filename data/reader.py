import fitz 
from .filters import prepare

class document():   
    def __init__(self, path: str = "/") -> None : 
        self.path = path
        self.doc = fitz.open(path)

    def prepare_text(self, text) -> str: 
        return prepare(text)

    def __getitem__(self, idx) -> str: 
        try: 
            page = self.doc.load_page(idx)
            text = page.get_text() 
        except: 
            text = ''

        return self.prepare_text(text)

    def __len__(self) -> int: 
        return len(self.doc)
    
    def __str__(self) -> str:
        return f"Document from {self.path}, first page [{self[0]}]"
    
    def to_txt(self, save_path: str, progressbar =False):

        if progressbar == True: 
            from tqdm import tqdm

            text = ""
            for i in tqdm(range(len(self))): 
                text += self[i]
        else: 
            text = ""
            for i in range(self): 
                text += self[i]
        
        with open(save_path, 'w', encoding='utf-8') as f: 
            f.write(text)
            f.close()