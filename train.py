from tqdm import tqdm
from transformer import *
from utils import *

ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

torch.manual_seed(42)

data = ModelParams(
    emb_dim=128, 
    use_cache=False, 
    device=device, 
    num_heads=16, 
    kv_num_heads=None,
    max_batch_size=8, 
    max_seq_len=256, 
    ffn_hidden_dim=512, 
    theta=None, 
    thresh=None,
    n_layers=1, 
    vocab_size=tokenizer.vocab_size + 1, 
    div_batch=8, 
    k_tokens=128
)

dataloader = DataLoader(
    Data(ds, tokenizer, data), 
    batch_size=data.max_batch_size, 
    collate_fn=Data(ds, tokenizer, data).collate_fn
)
transformer = Transformer(data)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

if __name__ == "__main__":
    num_epochs = 10

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                inputs, targets = batch
                outputs = transformer(inputs, None)
                one_hot_encoded_targets = torch.eye(data.vocab_size)[target].to(device)
                loss = criterion(outputs, one_hot_encoded_targets.to(device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                pbar.set_postfix({'Loss': total_loss / len(dataloader)})
                del loss, outputs, inputs, targets
                torch.cuda.empty_cache()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    torch.save(transformer.state_dict(), "transformer_model.pth")
