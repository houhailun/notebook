import os
import torch
import time
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from pretrain_dataset import PretrainBinaryDataset, split_dataset
from transformer import GPTConfig, MiniGPT
from trainer import Trainer


def main():
    epochs = 10
    learning_rate = 5e-5
    batch_size = 8
    train_ratio = 0.9997
    weight_decay = 0.01
    
    last_checkpoint_path = "/data2/minigpt/models/20241210/checkpoint-450000.pth"
    dataset_path = "/data2/minigpt/dataset/pretrain/mobvoi_seq_monkey_general_open_corpus.bin"
    output_dir = "/data2/minigpt/models/20241210"

    # 模型分布式, autocast会自动将float32绽放为float16（autocast不支持bfloat16），这里不用指定数据类型
    config = GPTConfig(flash_attn=False)
    model = MiniGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 数据加载要设置分布式采样器
    ds = PretrainBinaryDataset(dataset_path, config.context_length)
    train_set, eval_set = split_dataset(ds[:], train_ratio)

    # 训练参数
    train_args = {
        "train_batch_size": batch_size,
        "eval_strategy": "step",
        "eval_steps": 1000,
        "warmup_steps": 1000,
        "save_strategy": "step",
        "save_steps": 30000,
        "num_train_epochs": epochs,
        "output_dir": output_dir,
        "last_checkpoint_path": last_checkpoint_path,
        "use_mixed_precision": True,
    }
    start_time = time.time()
    trainer = Trainer(model, optimizer, train_args, verbose=True)
    trainer.set_seed(123)
    trainer.set_dataset(train_set, eval_set)
    trainer.train()
    print(f"train use time: {(time.time()-start_time)/60:.2f}min") if trainer.verbose else None
    test_generate_text(trainer)


def test_generate_text(trainer):
    if not trainer.is_main_process:
        print(f"{trainer.device}:return for not main process.")
        return
    # input_text = "小丽: 你好，我是文毅斌，很高兴认识你。\n小美: 你好"
    input_text = "库里在第三节上篮时被防守球员犯规，但裁判并未理会"  
    tokenizer_path = "/data2/minigpt/models/tokenizer_v3"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    generated_text = trainer.predict(tokenizer, input_text, 100)
    print(f"{trainer.device}:generate text test result:{generated_text}")

# I/O
if __name__ == "__main__": 
    print("Current working directory:", os.getcwd())
    # generate()
    main()

