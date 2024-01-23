from typing import List, Optional
import fire
from nick_custom_handler import LlamaHandler


class SystemProperties : 
    def __init__(self, ckpt_dir, tokenizer_path, max_batch_size, max_seq_len) :
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

class SmapleContext : 
    def __init__(self, ckpt_dir, tokenizer_path, max_batch_size, max_seq_len) : 
        self.system_properties = SystemProperties(ckpt_dir, tokenizer_path, max_batch_size, max_seq_len)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    dialogs: List[Dialog] = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    ]

    context = SmapleContext(ckpt_dir, tokenizer_path, max_batch_size, max_seq_len)
    handler = LlamaHandler()
    handler.initialize(context)

    datas = [{"body" : {"dialog" : dialogs, "max_gen_len" : 128}}]
    dialog = handler.preprocess(datas)
    inf_result = handler.inference(dialog)
    results =  handler.postprocess(inf_result)

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)