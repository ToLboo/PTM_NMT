# import各种所需的第三方包
import types
import argparse
import logging
from functools import partial
import json

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    BertGenerationConfig,
    BertGenerationEncoder,
    BertTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

# 自定义的util函数
from utils import TSVDataset, collect_fn, build_inputs_with_special_tokens
import sacrebleu
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


# 设置种子随机数，用于实验的可重复性
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.ngpu > 1:
        torch.cuda.manual_seed_all(args.seed)


# 收集所有并行计算设备的tensor并使用指定的reduce操作
def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    # 求和
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    # 除以当前进程组内的进程数
    rt /= torch.distributed.get_world_size()
    return rt


# 新建优化器和调度器
"""
优化器，预训练参数和初始化参数使用不同的学习率
:param args:
:param model:
:return:
"""
def get_optimizer_and_schedule(args, model: EncoderDecoderModel):
    # 是否多GPU训练
    if args.ngpu > 1:
        model = model.module

    # 未预训练的参数
    init_params_id = []
    for layer in model.decoder.transformer.h:
        # 通过map对参数进行标识符（id）映射
        init_params_id.extend(list(map(id, layer.crossattention.parameters())))
        init_params_id.extend(list(map(id, layer.ln_cross_attn.parameters())))
    # 预训练模型的参数
    pretrained_params = filter(
        lambda p: id(p) not in init_params_id, model.parameters()
    )
    initialized_params = filter(lambda p: id(p) in init_params_id, model.parameters())
    # 未训练和预训练参数采用不同的学习率
    params_setting = [
        {"params": initialized_params},
        {"params": pretrained_params, "lr": args.finetune_lr},
    ]
    optimizer = optim.Adam(params_setting, lr=args.lr)
    # 调度器使用基于cosine函数的动态学习率调整方法，并进行预热（warmup）
    schedule = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.num_training_steps,
        num_warmup_steps=args.num_warmup_steps,
    )
    return optimizer, schedule


# 获取模型和分词器
def get_model(args):
    """
    加载预训练模型，如果本地有模型，直接加载本地模型；否则远程拉取transformers的模型
    :param args:
    :return:
    """
    # 从model_path加载训练好的模型及分词器
    if args.model_path:
        model = EncoderDecoderModel.from_pretrained(args.model_path)
        src_tokenizer = BertTokenizer.from_pretrained(
            os.path.join(args.model_path, "src_tokenizer")
        )
        tgt_tokenizer = GPT2Tokenizer.from_pretrained(
            os.path.join(args.model_path, "tgt_tokenizer")
        )
        tgt_tokenizer.build_inputs_with_special_tokens = types.MethodType(
            build_inputs_with_special_tokens, tgt_tokenizer
        )
        # local_rank表示当前进程对应的GPU号
        if local_rank == 0 or local_rank == -1:
            print("model and tokenizer load from save success")
    else:
        # 加载预训练分词器
        # 加载encoder分词器
        # src_tokenizer = BertTokenizer.from_pretrained(args.src_pretrain_dataset_name)
        # src_tokenizer = BertTokenizer.from_pretrained(args.src_pretrain_dataset_name)
        # src_tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        src_tokenizer = os.path.join(os.getcwd(), 'bert-base-chinese')
        # 加载decoder分词器
        tgt_tokenizer = GPT2Tokenizer.from_pretrained(args.tgt_pretrain_dataset_name)
        # 定制化tgt_tokenizer
        tgt_tokenizer.add_special_tokens(
            {"bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]"}
        )
        tgt_tokenizer.build_inputs_with_special_tokens = types.MethodType(
            build_inputs_with_special_tokens, tgt_tokenizer
        )
        # 加载预训练模型，包括编码器和解码器
        encoder = BertGenerationEncoder.from_pretrained(args.src_pretrain_dataset_name)
        decoder = GPT2LMHeadModel.from_pretrained(
            args.tgt_pretrain_dataset_name, add_cross_attention=True, is_decoder=True
        )
        # 扩充解码器（tgt）的词表和token embedding
        decoder.resize_token_embeddings(len(tgt_tokenizer))  # 更新vocab的大小
        decoder.config.bos_token_id = tgt_tokenizer.bos_token_id  # 特殊符号的id更新
        decoder.config.eos_token_id = tgt_tokenizer.eos_token_id
        decoder.config.vocab_size = len(tgt_tokenizer)
        decoder.config.add_cross_attention = True
        decoder.config.is_decoder = True
        model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config
        ) # 加载seq2seq的配置
        # 将编码器和解码器放到一个模型中
        model = EncoderDecoderModel(
            encoder=encoder, decoder=decoder, config=model_config
        ) # 加载seq2seq
    # 将模型“发射”到GPU计算设备
    if local_rank != -1:
        model = model.to(device)
    # 多GPU并行计算
    if args.ngpu > 1:
        print("{}/{} GPU start".format(local_rank, torch.cuda.device_count()))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    optimizer, scheduler = get_optimizer_and_schedule(args, model)
    return model, src_tokenizer, tgt_tokenizer, optimizer, scheduler


# 用于保存训练好的模型和分词器
def save_model(
    args,
    model,
    optimizer,
    src_tokenizer: BertTokenizer,
    tgt_tokenizer: GPT2Tokenizer,
    nstep,
    nepoch,
    bleu,
    loss,
):
    # 记录整体训练评价结果
    train_metric_log_file = os.path.join(args.output_dir, "training_metric.tsv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(train_metric_log_file):
        with open(train_metric_log_file, "a", encoding="utf-8") as fa:
            fa.write("{}\t{}\t{}\t{}\n".format(nepoch, nstep, loss, bleu))
    else:
        with open(train_metric_log_file, "w", encoding="utf-8") as fw:
            fw.write("epoch\tstep\tloss\tbleu\n")
            fw.write("{}\t{}\t{}\t{}\n".format(nepoch, nstep, loss, bleu))

    # 保存模型
    model_save_path = os.path.join(
        args.output_dir, "epoch{}_step{}/".format(nepoch, nstep)
    )
    os.makedirs(model_save_path)
    model.save_pretrained(model_save_path)
    if local_rank == 0 or local_rank == -1:
        print(
            "epoch:{} step:{} loss:{} bleu:{} model save complete.".format(
                nepoch, nstep, round(loss, 4), round(bleu, 4)
            )
        )
    if args.save_optimizer:
        torch.save(optimizer, os.path.join(model_save_path, "optimizer.pt"))

    # 保存tokenizer
    src_tokenizer.save_pretrained(os.path.join(model_save_path, "src_tokenizer"))
    tgt_tokenizer.save_pretrained(os.path.join(model_save_path, "tgt_tokenizer"))


# 主函数
def main(args):
    if local_rank == 0 or local_rank == -1:
        print(vars(args))
    model, src_tokenizer, tgt_tokenizer, optimizer, scheduler = get_model(args) # 加载预训练模型
    # 是否进行预测
    if args.ispredict:
        while True:
            # 命令行读取要翻译的字符串
            input_str = input("src: ")
            output_str = predict(
                input_str,
                model,
                src_tokenizer,
                tgt_tokenizer,
                args.max_src_len,
                args.max_tgt_len,
            )
            # 打印翻译后的字符串
            print(output_str)
    else:
        # 有独立的验证集
        if args.eval_data_path:
            train_dataset = TSVDataset(data_path=args.train_data_path)
            eval_dataset = TSVDataset(data_path=args.eval_data_path)
            if local_rank == 0 or local_rank == -1:
                print(
                    "load train_dataset:{} and eval_dataset:{}".format(
                        len(train_dataset), len(eval_dataset)
                    )
                )
        # 如果没有验证集，就按比例从训练集中划分出训练集
        else:
            dataset = TSVDataset(data_path=args.train_data_path)
            train_size = int(args.train_dataset_ratio * len(dataset))
            eval_size = len(dataset) - train_size
            # 将训练集按照train_dataset_ratio划分出训练集和验证集
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [train_size, eval_size], generator=torch.Generator()
            )
            if local_rank == 0 or local_rank == -1:
                print(
                    "load dataset:{} split into train_dataset{} and eval_dataset:{}".format(
                        len(dataset), train_size, eval_size
                    )
                )
        # 定义batch数据采集函数，定义迭代器的输出生成
        collect_fn_ = partial(
            collect_fn,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        )
        # 训练集的数据加载器，训练集迭代器 ，每次取一个batch的数据进行训练
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            collate_fn=collect_fn_,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            sampler=DistributedSampler(train_dataset),
        )
        # 验证集的数据加载器，评测集迭代器，每次取一个batch的数据进行评测
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=collect_fn_,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
        )
        # 进行训练
        train(
            args=args,
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
        )


# 进行训练的函数
def train(
    args,
    model: EncoderDecoderModel,
    train_dataloader,
    eval_dataloader,
    optimizer,
    scheduler,
    src_tokenizer,
    tgt_tokenizer,
):
    eval_bleu = -1
    # epoch循环
    for epoch in range(args.nepoch):
        step = 0
        # 一个epoch的迭代（iteration）次数
        total_batch = train_dataloader.__len__()
        for data in train_dataloader:
            # 拆分batch
            (
                input_ids,
                attention_mask,
                decoder_input_ids,
                decoder_attention_mask,
                labels,
            ) = data
            # 网络前向推理，计算loss
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                decoder_input_ids=decoder_input_ids.to(device),
                decoder_attention_mask=decoder_attention_mask.to(device),
                labels=labels.to(device),
                return_dict=True,
            )
            # loss反向传播
            loss = outputs.loss
            loss.backward()
            # 参数更新
            optimizer.step()
            # 调度器更新学习率
            scheduler.step()
            step += 1
            # 梯度清零
            model.zero_grad()
            # 如果是多GPU训练，对多GPU的loss求平均
            if args.ngpu > 1:
                reduced_loss = reduce_tensor(loss).cpu().item()
            else:
                reduced_loss = loss.cpu().item()
            if local_rank == 0 or local_rank == -1:
                # 记录训练过程中的信息
                lr, finetune_lr = scheduler.get_lr()[0], scheduler.get_lr()[1]
                print(
                    "\rstep:{}/{}, bleu:{} loss:{}, lr:{}, ft.lr:{}".format(
                        step,
                        total_batch,
                        round(eval_bleu, 4),
                        round(reduced_loss, 4),
                        round(lr, 6),
                        round(finetune_lr, 6),
                    ),
                    end="",
                )
                writer.add_scalar("loss", reduced_loss, int(step * (1 + epoch)))
                writer.add_scalar("lr", lr, int(step * (1 + epoch)))
                writer.add_scalar("finetune_lr", finetune_lr, int(step * (1 + epoch)))
                # 每save_step步或者epoch结束，进行评估，保存模型
                # 达到指定的step或者所有batch迭代结束，评估一次评测集的结果
                if step % args.save_step == 0 or step % total_batch == 0:
                    eval_bleu = eval(
                        model, eval_dataloader, tgt_tokenizer, args.max_src_len
                    )
                    writer.add_scalar("bleu", eval_bleu, int(step * (1 + epoch)))
                    # 模型保存
                    model_to_save = model.module if hasattr(model, "module") else model
                    save_model(
                        args,
                        model_to_save,
                        optimizer,
                        src_tokenizer,
                        tgt_tokenizer,
                        step,
                        epoch,
                        eval_bleu,
                        reduced_loss,
                    )


# 进行评估的函数
def eval(
    model: EncoderDecoderModel,
    eval_dataloader,
    tgt_tokenizer,
    max_src_len,
    num_beams=6,
):
    """
    按batch进行模型预测，计算decoder输出的结果和原来的英文句子的bleu，作为模型的指标
    :param model:
    :param eval_dataloader: eval数据集迭代器
    :param tgt_tokenizer:   gpt2 分词器
    :param max_src_len:
    :param num_beams:    beam search 的k值
    :return:
    """
    hyp, ref = [], []
    # 评估的时候不用计算梯度
    with torch.no_grad():
        for data in tqdm(eval_dataloader):
            input_ids, attention_mask, decoder_input_ids, _, _ = data
            # 区分多GPU训练和单GPU训练
            if hasattr(model, "module"):
                generate = model.module.generate
            else:
                generate = model.generate
            # 网络前向传播
            outputs = generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=max_src_len,
                num_beams=num_beams,
                bos_token_id=tgt_tokenizer.bos_token_id,
                eos_token_id=tgt_tokenizer.eos_token_id,
            )# 模型输出
            # 对网络输出进行解码
            hypoth = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refer = tgt_tokenizer.batch_decode(
                decoder_input_ids, skip_special_tokens=True
            )
            # 预测值结果存储
            hyp.extend(hypoth) # 模型原来的id
            # 真实值结果存储
            ref.extend(refer) # 模型预测结果
    # 计算bleu
    bleu = sacrebleu.corpus_bleu(hyp, [ref])
    return bleu.score


# 用于预测的函数
def predict(
    input_str,
    model: EncoderDecoderModel,
    src_tokenizer,
    tgt_tokenizer,
    max_src_len,
    max_tgt_len,
    num_beam=6,
):
    # 对输入字符串进行分词
    inputs = src_tokenizer(
        [input_str],
        padding="max_length",
        truncation=True,
        max_length=max_src_len,
        return_tensors="pt",
    )
    # 将网络的输入数据“发射”到计算设备上
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    # 区分多GPU训练和单GPU训练
    if hasattr(model, "module"):
        generate = model.module.generate
    else:
        generate = model.generate
    # 网络前向传播
    outputs = generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_tgt_len,
        num_beams=num_beam,
        bos_token_id=tgt_tokenizer.bos_token_id,
        eos_token_id=tgt_tokenizer.eos_token_id,
    )
    # 对网络输出进行解码
    output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str


# 整个程序的入口函数
if __name__ == "__main__":
    # 参数解析器
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset_name", default="origin", type=str)
    parse.add_argument("--src_pretrain_dataset_name", default=None, type=str)
    parse.add_argument("--tgt_pretrain_dataset_name", default=None, type=str)
    parse.add_argument("--train_data_path", default=None, type=str)
    parse.add_argument("--eval_data_path", default=None, type=str)
    parse.add_argument("--run_path", default=None, type=str)
    parse.add_argument("--output_dir", default="../checkpoints/", type=str)
    parse.add_argument("--optimizer", default="adam", type=str)
    parse.add_argument("--lr", default=1e-7, type=float)
    parse.add_argument("--finetune_lr", default=1e-5, type=float)
    parse.add_argument("--ngpu", default=1, type=int)
    parse.add_argument("--seed", default=17, type=int)
    parse.add_argument("--max_src_len", default=128, type=int)
    parse.add_argument("--max_tgt_len", default=128, type=int)
    parse.add_argument("--save_step", default=100, type=int)
    parse.add_argument("--num_training_steps", default=100, type=int)
    parse.add_argument("--num_warmup_steps", default=100, type=int)
    parse.add_argument("--nepoch", default=1, type=int)
    parse.add_argument("--num_workers", default=16, type=int)
    parse.add_argument("--train_batch_size", default=32, type=int)
    parse.add_argument("--eval_batch_size", default=32, type=int)
    parse.add_argument("--drop_last", default=False, action="store_true")
    parse.add_argument("--ispredict", action="store_true", default=False)
    parse.add_argument("--save_optimizer", action="store_true", default=False)
    parse.add_argument("--train_dataset_ratio", default=0.999, type=float)
    parse.add_argument("--model_path", default=None, type=str)
    parse.add_argument("--local_rank", default=0, type=int)
    args = parse.parse_args()
    # 判断是否有GPU，以及有几个GPU
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = args.local_rank
        device = torch.device("cpu")
    # 主进程创建tensorboard的记录器
    if local_rank == 0 or local_rank == -1:
        sw_log_path = os.path.join(args.run_path, args.dataset_name)
        if not os.path.exists(sw_log_path):
            os.makedirs(sw_log_path)
        writer = SummaryWriter(sw_log_path)
    # 设置随机数种子
    set_seed(args)
    # 进入main的流程，训练模型并评估
    main(args)
