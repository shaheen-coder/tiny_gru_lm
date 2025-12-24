import torch
import torch.nn.functional as F
from model import GruLM
import sentencepiece as spm

@torch.no_grad()
def generate(
    model: GruLM,
    input_ids: torch.Tensor,
    max_tokens: int = 20,
    temp: float = 1.0,
    top_k: int | None = None,
):
    model.eval()
    generated = input_ids

    for _ in range(max_tokens):
        logits, _ = model(generated)
        logits = logits[:, -1, :] / temp

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    return generated

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('tokenizer.model')
    vocab_size = tokenizer.get_piece_size()

    model = GruLM(
        vocab_size=vocab_size,
        emb_dim=100,
        hdn_dim=100,
        num_layers=2,
        dropout=0.1,
        tie_wght=True
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load('grulm.pt', map_location=device))
    
    print("Model loaded. Type 'q' to exit.")

    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'q': 
            break
        prompt = f"<prompt> {prompt} <ai>" 

        token_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

        output_ids = generate(model, input_tensor, max_tokens=50, top_k=50)
        result_str = tokenizer.decode(output_ids[0].tolist())
        
        print(f"Output: \n{result_str}\n")

