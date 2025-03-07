import os
import torch
from torch.nn import nn
from torch.nn import functional as f
from torch.utils.data import (
    DataLoader, 
    Dataset, 
    DistributedSampler, 
    random_split, 
    DistributedDataParallel
    )
from torch.distributed import init_process_group
import torch.distributed as dist



def init_distributed_mode(self):
    """初始化分布式训练模式
    """
    # 获取当前进程的全局进程号
    rank = int(os.environ.get('RANK', -1))
    if rank == -1:
        self.ddp = False             # 不使用分布式
        self.is_main_process = True  # 主进程
        return
    
    # 分布式模式
    os.environ['NCCL_DEBUG'] = 'WARN'
    world_size = int(os.environ['WORLD_SIZE'])                                  # 获取进程总数
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)   # 初始化进程组
    
    self.ddp = True
    self.rank = rank
    self.is_main_process = rank == 0                    # 判断当前进程是否为主进程
    self.local_rank = int(os.environ['LOCAL_RANK'])     # 获取当前进程的本地排名
    self.device = f'cuda:{self.local_rank}'             # 根据本地排名设置设备
    torch.cuda.set_device(self.local_rank)              # 设置当前进程使用的 CUDA 设备


# ------------------- 数据集分片处理 -------------------
# 数据并行需要将数据分割为多份，每份数据分给到一张GPU去处理
# 在DDP中，这个数据集切割和分配的功能由一个分布式采样器`DistributedSampler`来完成。
# 1.对原始数据集进行处理
# 2.使用DistributedSampler对数据集进行分片处理，每个gpu进程都会使用采样器对训练数据集进行切分采样。
# 3.动态分配：DDP支持在每轮训练前，使用set_epoch方法动态计算出数据集索引，确保每个epoch前可以随机打乱数据顺序。

def set_dataset(self, train_set, eval_set):
    """数据集分片处理
    
    """
    self.train_set = train_set
    self.eval_set = eval_set
    print(f'set trainset: {len(train_set)}, evalset: {len(eval_set)}') if self.verbose else None
    

def init_dataloader(self):
    """数据加载器初始化
    """
    assert self.train_set and self.eval_set, 'train_set and eval_set cannot be None'
    train_set, eval_set, batch_size = self.train_set, self.eval_set, self.batch_size
    
    sampler = DistributedSampler(train_set) if self.ddp else None
    # 训练集数据加载器
    ### 注意采样和shuffle是互斥的，如果采样器不为None，那么shuffle必须为False
    self.train_loader = DataLoader(train_set, 
                                   batch_size=batch_size, 
                                   sampler=sampler, 
                                   num_workers=0,  # 数据加载时使用的子进程数量为 0，意味着数据加载将在主进程中完成
                                   drop_last=True, # 如果最后一个批次的数据量不足 batch_size，则丢弃该批次。
                                   shuffle=(sampler is None))
    
    # 验证集数据加载器
    ### NOTE：评估集不需要分片处理
    self.eval_loader = DataLoader(eval_set, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=False)
    
    self.steps_per_epoch = len(self.train_loader)
    self.total_steps = self.steps_per_epoch * self.epochs
    print(f'steps_per_epoch: {self.steps_per_epoch}, total_steps: {self.total_steps}') if self.verbose else None
   
def train_epoch(self, epoch):
    """训练一个epoch
    """
    assert self.train_loader and self.eval_loader, 'train_loader and eval_loader cannot be None'
    
    self.train_loader.sampler.set_epoch(epoch)  # 设置采样器的 epoch
    for i, (X, Y) in enumerate(self.train_loader):
        pass
    
    
# --------------------------------- 模型复制 ---------------------------------
# DDP分布式训练中，每个GPU进程都需要复制一份完整的模型参数和优化器状态，因此需要在每个卡上加载一份模型，使用DDP封装
# 封装的目的是让模型在训练过程中能在各个进程间同步参数状态。
def wrap_model_with_ddp(model, local_rank):
    """模型复制
    """
    # 位置编码用的是复数，而nccl不支持复数，此变量并不要求在多进程中保持一致，所以暂时屏蔽对此变量的同步
    model._ddp_params_and_buffers_to_ignore = ["pos_cis"]
    model = DistributedDataParallel(model, device_ids=[local_rank])
    print(f"{cur_time()} packaged model with DDP in cuda:{local_rank}")
    return model


# --------------------------------- 模型评估 ---------------------------------
# 在分布式训练中，每张卡上的模型状态始终是保持同步的，所以对模型损失的评估只需要在主进程卡上进行即可
def check_and_evaluate(self, lr):
    """检测模型是否需要评估"""
    # 只有在满足设定的评估步数时才进行评估
    if (self.step + 1) % self.eval_steps != 0:
        return
    
    if self.is_main_process:  # 主进程
        train_loss = self.train_loss_acc / self.eval_steps
        eval_loss = self.evaluate()
        print(f"{self.cur_time()} lr={lr:.5f}, train_loss: {train_loss:.4f}, "
            + f"eval_loss: {eval_loss:.4f}, steps: {self.step}/{self.total_steps}"
        )
        self.train_loss_acc = 0

    # 引入同步屏障，起到多进程间训练状态同步的目的
    # 原因是evaluate()是一个耗时的操作，添加这句代码可以让其他进程等待主进程执行完模型评估后再统一进行下一步训练，避免多个进程中的模型状态不同步，出发nccl同步超时。
    dist.barrier() if self.ddp else None  #
        
        
def evaluate(self):
    """模型评估
    """
    # 注意：评估阶段不能多进程同步，必须用原始model
    model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
    model.eval()
    num_batches = len(self.eval_loader)
    total_loss = 0
    
    for (X, Y) in self.eval_loader:
        with torch.no_grad():
            logits = model(X)
        loss = f.cross_entropy(logits.flatten(0, 1), Y.flatten())
        total_loss += loss.item()
        
    model.train()
    return total_loss / num_batches


# --------------------------------- 模型保存与恢复 ---------------------------------
# 模型的状态保存也只需要在主进程上执行
def check_and_save_checkpoint(self, cur_epoch):
    """模型是否需要保存模型，并保存
    """
    # 只有在满足设定的保存步数时才进行保存
    if self.step % self.save_step != 0:
        return
    
    if self.is_main_process:
        checkpoint_path = f"{self.output_dir}/checkpoint-{self.step}.pth"
        self.save_model(checkpoint_path, cur_epoch)
        print(f"{self.cur_time()} device:{self.device}-save checkpoint: {checkpoint_path}")
    
    # 设置屏障，让所有进程等待主进程的checkpoint操作
    dist.barrier() if self.ddp else None
    
    
# --------------------------------- 训练流程改造 ---------------------------------
def train_epoch(self, cur_epoch):
    """训练一个epoch
        1. 分布式模式下，在单轮训练开始前用set_epoch函数打乱顺序； 
        2. 每次单步训练前，调用`adjust_lr`动态调整学习率；
        3. 评估验证改用封装后的`check_and_evaluate`方法，兼容单卡训练和多卡训练；
        4. 增加保存checkpoint环节的函数调用：`check_and_save_checkpoint`，也兼容单卡和多卡训练； 
    """
    assert self.train_loader and self.eval_loader, 'train_loader and eval_loader cannot be None'
    
    # 每个epoch开始前打乱数据顺序
    self.train_loader.sampler.set_epoch(cur_epoch) if self.ddp else None
    
    for i, (X, Y) in enumerate(self.train_loader):
        lr = self.adjust_lr()
        train_loss = self.train_step(X.to(self.device), Y.to(self.device))
        self.train_loss_acc += train_loss.item()
        self.step += 1
        self.check_and_evaluate(lr)
        self.check_and_save_checkpoint(cur_epoch)
        

def train(self):
    """主流程
        1. 添加对分布式环境的初始化； 
        2. 添加对恢复训练的支持，能从指定的checkpoint恢复训练； 
        3. 添加对DDP模型状态同步的支持； 
        4. 主循环基本不变，只添加了从上次epoch继续训练的支持； 
        5. 最后训练完后，清理和注销进程资源； 
    """
    # 初始化分布式模式
    self.init_distributed_mode()
    # 初始化数据加载器
    self.init_dataloader()
    # 将模型迁移到指定设备
    self.model.to(self.device)
    # 恢复训练状态
    last_epoch = 0
    if self.last_checkpoint_path:
        last_epoch = self.load_from_checkpoint()
        
    # 分布式训练需要使用ddp同步模型状态
    if self.ddp:
        self.model = self.warp_model_with_ddp(self.model, self.local_rank)
    # 打印模型所在的设备
    model_device = next(self.model.parameters()).device
    print("Model is on device: ", model_device)  
    # 训练主循环
    for epoch in range(last_epoch, self.num_epochs):
        self.train_epoch(epoch)
    # 注销分布式进程
    dist.destroy_process_group() if self.ddp else None