import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import sentencepiece as spm
from typing import Optional
from dataclasses import dataclass
import math
import os


@dataclass    
class ModelParams:
    emb_dim: int 
    use_cache: bool
    kv_num_heads: Optional[int]
    num_heads: int
    device: str
    max_seq_len: int
    max_batch_size: int
    ffn_hidden_dim: int
    theta: Optional[int]
    thresh:Optional[float]
    n_layers: int
    vocab_size: int
    div_batch: int
    k_tokens: Optional[int]

def train_tokenizer(vocab_size: int, dataset, model_prefix:str="tokenizer", model_type: str="bpe"):
    concatenated_texts = []
    for idx in range(len(dataset)):
        prompt = dataset['prompt'][idx]
        text = dataset['text'][idx]
        concatenated_text = f"Question: {prompt} Answer: {text}"
        concatenated_texts.append(concatenated_text)

    with open(f"{model_prefix}.txt", "w", encoding="utf-8") as f:
        for text in concatenated_texts:
            f.write(text + "\n")
    spm.SentencePieceTrainer.Train(
        f'--input={model_prefix}.txt --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type}')
   

class Data(Dataset):
    def __init__(self, dataset, tokenizer, params: ModelParams):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = params.max_seq_len
        self.device = params.device
        self.train = not params.use_cache

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt =  self.dataset['prompt'][idx]
        text = self.dataset['text'][idx]
        if self.train:
            encoded_prompt = torch.tensor(self.tokenizer.encode(prompt),dtype = torch.int32)
            encoded_text = torch.tensor(self.tokenizer.encode(text), dtype = torch.int32)
            return encoded_prompt[:self.max_seq_len],encoded_text[:self.max_seq_len]
        else:
            concat_text = f"Question: {prompt} Answer: {text}"
            tokenized_text = self.tokenizer.encode(concat_text)
        
            to_ret = torch.tensor(tokenized_text, dtype = torch.int32).to(self.device)
            return to_ret[:self.max_seq_len]

    def collate_fn(self, batch):
        if torch.is_tensor(batch):
            padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
            return padded_batch.to(self.device)
        else:
            batch = list(zip(*batch))
            padded_prompt = pad_sequence(batch[0], batch_first=True, padding_value=0)
            padded_text = pad_sequence(batch[1], batch_first=True, padding_value=0)
            
            return padded_prompt.to(self.device), padded_text.to(self.device)

class RotaryEmbeddings(nn.Module):
    def __init__(self, device:str, theta: int =10000):
        super().__init__()
        self.theta = theta
        self.device = device

    def forward(self, x: torch.Tensor, seq_len:Optional[int]=None, emb_dim:Optional[int]=None)->torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        assert emb_dim%2==0, "Embeddings dimension must be even"
        #Q_i=10000^(-2(i-1)/emb_dim)
        thetas = (1.0/self.theta**((2*torch.arange(0,emb_dim,2))//emb_dim)).to(self.device)
        thetas_repeated = thetas.unsqueeze(0).repeat(seq_len, 1)
        thetas_true = thetas_repeated * (torch.arange(seq_len, device = self.device)+1).unsqueeze(1)
        #calculate the rotation matrices using these thetas, apply them on the embeddings in  2D or complex space
        matrix_rot = torch.stack((torch.sin(thetas_true),torch.cos(thetas_true)),dim=-1).to(self.device)
        comp_matrix = torch.view_as_complex(matrix_rot).unsqueeze(0)
        x_reshaped = torch.view_as_complex(x.reshape(batch_size, seq_len, emb_dim//2, 2))
        rotated_x = torch.view_as_real(x_reshaped * comp_matrix).squeeze(-1).reshape(batch_size, seq_len, emb_dim).to(self.device)
        del x_reshaped, comp_matrix, matrix_rot, thetas_true, thetas_repeated, thetas
        torch.cuda.empty_cache()
        return rotated_x


class GQattention(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.use_cache = params.use_cache
        self.device = params.device
        self.pos_rotor = RotaryEmbeddings(self.device)

        self.num_heads = params.num_heads
        self.kv_num_heads = params.kv_num_heads if params.kv_num_heads is not None else params.num_heads
        assert params.emb_dim % self.num_heads==0, "Make the embedding dim divisible by num_heads"
        self.head_dim = params.emb_dim//self.num_heads
        self.kv_head_dim = params.emb_dim//self.kv_num_heads if self.kv_num_heads is not None else self.head_dim 

        self.wq = nn.Linear(params.emb_dim, self.num_heads*self.head_dim).to(self.device)
        self.wk = nn.Linear(params.emb_dim, self.kv_num_heads*self.kv_head_dim).to(self.device)
        self.wv = nn.Linear(params.emb_dim, self.kv_num_heads*self.kv_head_dim).to(self.device)
        self.wo = nn.Linear(params.emb_dim, self.num_heads*self.head_dim).to(self.device)
        if self.use_cache:
            self.c_v = torch.zeros((params.max_batch_size, params.max_seq_len, self.kv_num_heads, self.kv_head_dim))
            self.c_k = torch.zeros((params.max_batch_size, params.max_seq_len, self.kv_num_heads, self.kv_head_dim))

    def forward(self, x:torch.Tensor, cur_pos: Optional[int]=None)->torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        output = self.wo(x)
        
        xq = self.pos_rotor(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = self.pos_rotor(value).reshape(batch_size, seq_len, self.kv_num_heads, self.kv_head_dim)
        xk = key.reshape(batch_size, seq_len, self.kv_num_heads, self.kv_head_dim)
        
        if self.use_cache:
            self.c_v[:batch_size, cur_pos:cur_pos+seq_len]=xv
            self.c_k[:batch_size, cur_pos:cur_pos+seq_len]=xk
            
            keys = self.c_k[:batch_size, :cur_pos+seq_len]
            values = self.c_v[:batch_size, :cur_pos+seq_len]

            n_rep = self.num_heads//self.kv_num_heads

            keys = keys[:,:,:,None,:].expand(keys.shape[0], keys.shape[1],
                                           self.kv_num_heads, n_rep, self.kv_head_dim).reshape(keys.shape[0],
                                            keys.shape[1], self.kv_num_heads*n_rep, self.kv_head_dim)

            values = values[:,:,:,None,:].expand(values.shape[0], values.shape[1],
                                                 self.kv_num_heads, n_rep, self.kv_head_dim).reshape(values.shape[0],
                                                 values.shape[1], self.kv_num_heads*n_rep, self.kv_head_dim)

        else:
            keys = xq
            values = xv
        
        xq = xq.permute(0, 2, 1, 3).contiguous().to(self.device)
        keys = keys.permute(0, 2, 3, 1).contiguous().to(self.device)
        values = values.permute(0, 2, 1, 3).contiguous().to(self.device)
        
        query_key_score = torch.matmul(xq, keys)/math.sqrt(self.head_dim)
        attention_score = torch.matmul(query_key_score, values).transpose(1,2).contiguous().reshape(batch_size, seq_len, -1)
        output = self.wo(attention_score)

        del query_key_score, attention_score, xq, keys, values 
        torch.cuda.empty_cache()
        #make sure that the dimensions are correct and that the training and inferencing parts are compatible
        return output

class RMSnorm(nn.Module):
    def __init__(self, dim:int, device:str, thresh: float = 1e-4):
        super().__init__()
        self.params = nn.Parameter(torch.ones(dim))
        self.thresh = thresh
        self.device = device

    def forward(self, x:torch.Tensor)->torch.Tensor:
        denom = torch.sqrt(x.pow(2).mean(-1,keepdims=True)).to(self.device)
        res = ((x.to(self.device))*self.params.to(self.device))/denom
        del denom
        torch.cuda.empty_cache()
        return res

class SwiGLu_Forward(nn.Module):
    def __init__(self, params:ModelParams):
        super().__init__()
        self.hidden_dim = params.ffn_hidden_dim
        self.device = params.device
        self.w1 = nn.Linear(params.emb_dim, self.hidden_dim).to(self.device)
        self.w2 = nn.Linear(params.emb_dim, self.hidden_dim).to(self.device)
        self.w3 = nn.Linear(self.hidden_dim, params.emb_dim).to(self.device)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.w3(self.w2(x)*nn.functional.silu(self.w1(x)))



