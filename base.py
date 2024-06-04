import argparse
from data import document
from nn import train, TrainConfig
from typing import Callable

extra_w_c = 0 
red_flag_c = 0 

def data_(document_path: str, prep_fn: Callable = None): 
    if document_path.endswith(".pdf"): 
        doc = document(document_path, prep_fn)
        return doc.read_all()

    elif document_path.endswith(".txt"): 
        with open(document_path, 'r', encoding='utf-8') as file: 
            doc: str = file.read() 
            file.close() 

        return prep_fn(doc)


def train_nn(text_data, config: TrainConfig): 
    model = train(text_data, cfg = config)
    return model 

def run(doc_path, 
        preparation_fn:Callable = None, 
        config:TrainConfig = None): 

    if type(doc_path) == list:
        text_data = " ".join([data_(dp, preparation_fn) for dp in doc_path])
    else: 
        text_data = data_(doc_path, preparation_fn)

    default_config = TrainConfig() if (config == None) else config

    global extra_w_c 
    global red_flag_c

    print(f"""
        deleted EXTRA_WORDS chars: {extra_w_c}
        found RED_FLAGS {red_flag_c}
    """)
    model = train_nn(text_data, default_config)
    return model


def build_prep_fn(extra_w, red_flag_w):

    def fn(t:str): 
        global extra_w_c
        global red_flag_c

        for w in red_flag_w: 
            if w in t: 
                t_new = ''
                red_flag_c = red_flag_c+1 
        for w in extra_w: 
            t_new = t.replace(w, '')
            extra_w_c = extra_w_c + (len(t) - len(t_new))

        return t_new

    return fn

def main(): 
    parser = argparse.ArgumentParser(description="Process document and train neural network")
    
    parser.add_argument('--doc_path', type=str, nargs='+', default=[], help='Paths to the document')
    parser.add_argument('--mode', type=str, default='train', help='"preview" [For check the document] or "train" For Train LoRA')
    parser.add_argument('--page', type=int, default=0, help='Page for preview document')

    
    parser.add_argument('--extra_words', type=str, nargs='+', default=[], help='Words to replace [useless Links, extra Names, .etc]')
    parser.add_argument('--red_flag_words', type=str, nargs='+', default=[], help='Words to replace page [For Example: "CONCLUSION"]')

    # LORA Params
    parser.add_argument('--rank', type=int, default=8, help='LORA rank')
    parser.add_argument('--l_alpha', type=int, default=24, help='LORA alpha')
    parser.add_argument('--l_dropout', type=float, default=0.1, help='LORA dropout')
    parser.add_argument('--l_modules', type=str, nargs='+', default=['lm_head', 'c_fc', 'c_proj', 'wte', 'wpe'], help='LORA target modules')

    # Train Params
    parser.add_argument('--chunk_size', type=int, default=150, help='Chunk size')
    parser.add_argument('--model_preset', type=str, default='0x7o/nanoFialka-v1', help='Model preset')
    parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per device')
    parser.add_argument('--save_steps', type=int, default=50, help='Save steps')
    parser.add_argument('--save_total_limit', type=int, default=1, help='Save total limit')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay')

    args = parser.parse_args()

    train_cfg = TrainConfig(
        chunk_size=args.chunk_size,
        model_preset=args.model_preset,
        rank=args.rank,
        l_alpha=args.l_alpha,
        l_dropout=args.l_dropout,
        l_modules=args.l_modules,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr=args.lr,
        wd=args.wd
    )

    if args.mode.lower().strip() == 'train': 
        run(doc_path=args.doc_path, 
            config=train_cfg, 
            preparation_fn=build_prep_fn(extra_w = args.extra_words, 
                                        red_flag_w = args.red_flag_words))
    elif args.mode.lower().strip() == 'preview': 
        for i in args.doc_path: 
            doc = document(i)
            print(f"""DOCUMENT EXAMPLES PAGE: 
                    {doc[args.page]}""")

if __name__ == "__main__":
    main()
# EXAMPLE: 
# !python base.py --chunk_size 200 --model_preset "custom_model" --num_epoch 20 --batch_size 4
