from utils import *
import numpy as np

class Encoder(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.device = params.device
        self.emb_dim = params.emb_dim
        self.thresh = params.thresh
        self.norm = RMSnorm(self.emb_dim, self.device, self.thresh)
        self.attention = GQattention(params)
        self.ffn = SwiGLu_Forward(params)

    def forward(self, x:torch.Tensor, cur_pos: Optional[int])->torch.Tensor:
        first_layer = x + self.attention(self.norm(x),cur_pos) 
        second_layer = first_layer + self.ffn(self.norm(first_layer))

        del first_layer
        torch.cuda.empty_cache()
        return second_layer


class MODRouter(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.k_tokens = params.k_tokens if params.k_tokens is not None else params.max_seq_len
        assert self.k_tokens<=params.max_seq_len, "k should be lower"
        self.device = params.device
        self.n_layers = params.n_layers
        self.seq_len = params.max_seq_len
        self.emb_dim = params.emb_dim
        self.r_weight_layers = nn.ModuleList([nn.Linear(self.emb_dim, 1) for i in range(self.n_layers)]).to(self.device)
        self.encoders = nn.ModuleList([Encoder(params) for i in range(self.n_layers)]).to(self.device)
                   
    def forward(self, x:torch.tensor)->torch.Tensor:
        res = x.to(self.device)
        for layer in range(self.n_layers):
            route_weights = self.r_weight_layers[layer](res)
            flattened_weights = route_weights.squeeze(2)
            topk_indices = torch.topk(flattened_weights, k=self.k_tokens).indices
            sorted_top_k = torch.sort(topk_indices, dim=-1).values           
            chosen_tokens = res[torch.arange(res.size(0)).unsqueeze(1),sorted_top_k]
            
            all_indices = torch.arange(flattened_weights.shape[1]).repeat(flattened_weights.shape[0],1).to(self.device)
           
            #please find out a way to compute the following bs in pytorch
            np_all_indices = all_indices.to("cpu").numpy()
            np_sorted_top_k = sorted_top_k.to("cpu").numpy()
            fin_arr = []
            for idx in range(all_indices.shape[0]):
                diff = np.setdiff1d(np_all_indices[idx], np_sorted_top_k[idx])
                fin_arr.append(diff)

            botk_indices = torch.tensor(fin_arr, dtype = torch.int32).to(self.device)
            rejected_tokens = res[torch.arange(res.size(0)).unsqueeze(1),botk_indices]
            
            res = self.encoders[layer](chosen_tokens,None)
            if rejected_tokens.shape[1] < self.seq_len - self.k_tokens:
                rejected_tokens = torch.nn.functional.pad(rejected_tokens, (0,0,0,self.seq_len-self.k_tokens,0,0))
            res = torch.cat((res,rejected_tokens),dim=1)
            
            del route_weights,flattened_weights,topk_indices,sorted_top_k,chosen_tokens,all_indices,np_sorted_top_k,np_all_indices,botk_indices,rejected_tokens
            torch.cuda.empty_cache()
        return res


class Transformer(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.device = params.device
        self.vocab_size = params.vocab_size
        self.emb_dim = params.emb_dim
        self.thresh = params.thresh
        self.norm = RMSnorm(self.emb_dim, self.device, self.thresh)  
        self.div_batch = params.div_batch
        self.linear = nn.Linear(params.emb_dim, self.vocab_size).to(self.device)
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim).to(self.device)
        self.mod = MODRouter(params)

    def forward(self, x:torch.Tensor, cur_pos: Optional[int])->torch.Tensor:
        assert self.div_batch<=x.shape[0], "Batch serializer should not exceed tensor dimensions"

        res = self.embeddings(x)
  
        #layer_count = 0
        #stats = None
        #try:
        res = self.norm(self.mod(res))
        # layer_count+=1
           #stats = torch.cuda.memory_stats()
         #I'm only serializing the following linear layer because I'm poor af(PLEASE GIB A100)
        res = torch.chunk(res, self.div_batch, dim = 0)
       
        accumulated_output = None
        for chunk_idx in range(len(res)):
            out = self.linear(res[chunk_idx])
            if accumulated_output is None:
                accumulated_output = out
            else:
                # Concatenate the current output with the accumulated output along the specified dimension
                accumulated_output = torch.cat((accumulated_output, out), dim=0)
    
            del out
            torch.cuda.empty_cache()

        #print(f"Final Chunk shape: {np.array(res_chunks).shape}")
       
        probas = nn.functional.softmax(accumulated_output, dim = 1)
        del accumulated_output, res
        torch.cuda.empty_cache()
        return probas 
        #except:
        #   print(f"Stopped at {layer_count}")
        #    print(f"Memory stats for your convenience: {stats}")
        #    del res
        #   torch.cuda.empty_cache()
        #    return torch;.zeros(x.shape[0], self.vocab_size)


