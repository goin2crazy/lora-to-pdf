# Load model directly
import datasets

def chunk_text(text, chunk_size):
    """
    Splits the input text into chunks of the specified size.

    Args:
    text (str): The input text to be chunked.
    chunk_size (int): The size of each chunk.

    Returns:
    list: A list of text chunks.
    """
    # Split the text into words
    words = text.split()

    # Create chunks
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))

    return chunks

def encode_text(t, tokenizer):
    tokenz =  tokenizer(t, return_tensors='pt', padding='max_length', truncation=True)
    tokenz['labels'] = tokenz['input_ids'].clone()
    return tokenz

def to_dataset(large_text, tokenizer, chunk_size = 150, verbose = False): 
    # chunking the text
    chunks = chunk_text(large_text, chunk_size)

    if verbose == True: 
        # Print the first few chunks to verify
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1}:\n{chunk}\n")

        print(f"[chunks count {len(chunks)}]")

    return datasets.Dataset.from_dict({"text": chunks}).map(
        lambda t: encode_text(t['text']), remove_columns=['text'])
    


