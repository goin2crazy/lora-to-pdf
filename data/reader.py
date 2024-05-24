import fitz 
from filters import prepare

class document():   
    def __init__(self, path: str = "/") -> None : 
        self.path = path
        self.doc = fitz.open(path)

    def prepare_text(self, text) -> str: 
        return prepare(text)

    def __getitem__(self, idx) -> str: 
        page = self.doc.load_page(idx)
        text = page.get_text() 

        return self.prepare_text(text)

    def __len__(self) -> int: 
        return len(self.doc)
    
    def __str__(self) -> str:
        return f"Document from {self.path}, first page [{self[0]}]"