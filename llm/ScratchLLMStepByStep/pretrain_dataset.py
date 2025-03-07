import json
import numpy as np

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split, Subset



def read_text_dataset(data_path, max_size=100*1024*1024, context_key="text"):
    """读取文本数据集
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        current_size = 0
        current_texts = []
        while True:
            line = f.readline()
            if not line:
                if current_texts:
                    yield current_texts
                break
            
            data = json.loads(line)
            current_texts.append(data[context_key])
            current_size += len(data[context_key])
            if current_size > max_size:
                yield current_texts
                current_size = 0
                current_texts = []
                

class PretrainTextDataset(Dataset):
    """文本预训练数据集
    """
    def __init__(self, texts: list, tokenizer, max_length, stride=1):
        """初始化函数"""
        self.max_length = max_length
        self.stride = stride  # 移动窗口的步长
        self.tokenizer = tokenizer
        sep_token = self.tokenizer.unk_token
        # 注意这里使用的是sep_token，而不是cls_token
        token_ids = tokenizer.encode(sep_token.join(texts) + sep_token)
        
        self.input_set = []
        self.target_set = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i:i+max_length]
            target_ids = token_ids[i+1:i+max_length+1]
            self.input_set.append(torch.tensor(input_ids))
            self.target_set.append(torch.tensor(target_ids))
        
    def __len__(self):
        return len(self.input_set)
    
    def __getitem__(self, index):
        return self.input_set[index], self.target_set[index]
    
    
def create_dataloaders_from_texts(texts_data, train_ratio, tokenizer, max_length, batch_size):
    """从文本数据创建数据加载器
    """
    train_size = int(len(texts_data) * train_ratio)
    train_texts = texts_data[:train_size]
    eval_texts = texts_data[train_size:]
    
    train_dataset = PretrainTextDataset(train_texts, tokenizer, max_length)
    eval_dataset = PretrainTextDataset(eval_texts, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, eval_loader


def texts_to_bin(input_path, output_path, tokenizer, content_key="context"):
    """文本转二进制
    """
    bos_token = tokenizer.special_tokens_map["bos_token"]
    eos_token = tokenizer.special_tokens_map["eos_token"]
    max_buffered_length = 1 * 1024 * 1024
    
    with open(input_path, 'r', encoding='utf-8') as fr, open(output_path, 'wb') as fw:
        buffered_ids = []
        i = 0
        while True:
            line = fr.readline()
            if not line: break
            content = json.loads(line).get(content_key, "")
            if not content: continue
            
            # 将数据序列化为二进制格式
            tokenized = tokenizer.encode(bos_token + content + eos_token)
            buffered_ids += tokenized['input_ids']
            if len(buffered_ids) > max_buffered_length:
                arr = np.array(buffered_ids, dtype=np.uint16)
                fw.write(arr.tobytes())
                buffered_ids.clear()
                i += 1
                print(f"write {i}m bytes") if i % 100 == 0 else None
            
            # 处理最后一段不满max_buffer_length的token序列
            if len(buffered_ids) > 0:
                arr = np.array(buffered_ids, dtype=np.uint16)
                fw.write(arr.tobytes())
                print(f"write arr: {len(arr)}")
                

class PretrainBinaryDataset(Dataset):
    def __init__(self, data_path, max_tokens):
        with open(data_path) as f:
            f.seek(0, 2)
            self.total_tokens = f.tell() // np.dtype('uint16').itemsize
            print(f"total_size: {self.total_tokens}")
        
        self.data = np.memmap(data_path, dtype=np.uint16, 
                              shape=(self.total_tokens//max_tokens, max_tokens))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """读取样本"""
        if isinstance(index, int):
            return self._get_single_item(index)
        elif isinstance(index, slice):
            return self._get_slice_items(index)
        elif isinstance(index, list):
            return self._get_list_items(index)
        else:
            raise TypeError("index must be int, slice or list")
        
    def _get_single_item(self, index):
        """获得单个样本"""
        assert isinstance(index, int)
        item = self.data[index]
        # 在计算交叉熵损失时要求目标输出为长整型
        inputs = item[:-1].astype(np.int64)
        targets = item[1:].astype(np.int64)
        
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
    
    def _get_list_items(self, indexes):
        """获的多个样本"""
        assert isinstance(indexes, list)
        
        items = [self.data[i] for i in indexes]
        inputs = [item[:-1] for item in items]
        targets = [item[1:] for item in items]
        
        inputs_tensors = torch.tensor(inputs, dtype=torch.int64)
        targets_tensors = torch.tensor(targets, dtype=torch.int64)
        
        return inputs_tensors, targets_tensors
        
    def _get_slice_items(self, index):
        """获得切片样本"""
        # slice为内置切片对象，indices方法返回start, stop, step三个元素的元组, range(start, stop, step)则返回一个正常的索引迭代器
        return Subset(self, range(*index.indices(len(self))))


def split_dataset(data, train_ratio):
    """切分数据集
    """
    train_len = int(len(data) * train_ratio)
    eval_len = len(data) - train_len
    return random_split(data, [train_len, eval_len])


def create_dataloaders(ds, batch_size, train_ratio, local_rank=-1):
    """创建数据加载器
    """
    train_ds, eval_ds = split_dataset(ds, train_ratio)
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds, rank=local_rank) if local_rank >= 0 else None
    shuffle = True if sampler == None else False
    
    # num_workers用于epoch结束后不关闭workers，但实际测试，我们这个场景下用多进程num_workers加载，并不能提高速度。
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=0, drop_last=True, sampler=sampler)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    return train_loader, eval_loader