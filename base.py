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

    run(doc_path=args.doc_path, config=train_cfg)

# EXAMPLE: 
# !python base.py --chunk_size 200 --model_preset "custom_model" --num_epoch 20 --batch_size 4
