from src.model import TransformerLikeModel

import torch

SEQUENCE_LENGTH = 7
PREDICTION_LENGTH = 7
EMBED_SIZE = 4
ENCODER_SIZE = 1
DECODER_SIZE = 1
NUM_HEADS = 2
BATCH_SIZE = 2
EPOCHS = 500
DROPOUT = 0.05
TRAIN_PERCENTAGE = 0.75
SEED = 42
DELTA = False
EARLY_STOPPING = False

def main():

    model: TransformerLikeModel = TransformerLikeModel.load_model("ristoranti_model_dec.pth", TransformerLikeModel, embed_size=EMBED_SIZE,
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        num_head_dec_1=NUM_HEADS,
        num_head_dec_2=NUM_HEADS,
        num_head_enc=NUM_HEADS,
        output_len=PREDICTION_LENGTH,
        max_seq_length=SEQUENCE_LENGTH,
    )

    torch.set_printoptions(profile="full")
    print("Positional Encoding")
    print("-" * 30)
    print(model.pe.position_embeddings.data[0:7].T)
    print("\n")
    print("Attention Matrices")
    print("-" * 30)
    for i in range(NUM_HEADS):
        print(f"Head {i + 1}")
        layer = model.encoder[0].mha
        W_Q, W_K, W_V = layer.get_attention_matrix(i) # type: ignore
        print("\tQuery:")
        print(W_Q)
        print("\n\tKey:")
        print(W_K)
        print("\n\tValue:")
        print(W_V)
        print("\n\tOutput Matrix:")
        print(layer.W_O.weight) # type: ignore
    print("\n")
    print("Add & Norm")
    print("-" * 30)
    for i in range(2):
        print(f"Layer {i + 1}")
        layer = model.encoder[0].norm1 if i == 0 else model.encoder[0].norm2
        print(f"\tGamma {layer.weight}") # type: ignore
        print(f"\tBeta {layer.bias}") # type: ignore
    print("\n")
    print("FeedForward Layer")
    print("-" * 30)
    layer = model.encoder[0].ff
    print(f"\tWeight 1: {layer.fc1.weight}") # type: ignore
    print(f"\tBias 1: {layer.fc1.bias}") # type: ignore
    print(f"\tWeight 2: {layer.fc2.weight}") # type: ignore
    print(f"\tBias 2: {layer.fc2.bias}") # type: ignore
    print("CLS token")
    print("-" * 30)
    print(f"\tCLS token: {model.cls_token.data}")
    print("\nNumber of trainable parameters: ", sum(p.numel() for p in model.seca.parameters() if p.requires_grad))
    print(model.seca.decoder.weight)

    print("\n\n")
    print("Decoder")
    print("-" * 30)
    print("Number of trainable parameters: ", sum(p.numel() for p in model.decoder.parameters() if p.requires_grad))
    layer = model.decoder[0]
    print("Masked Multi-Head Attention")
    print("Number of trainable parameters: ", sum(p.numel() for p in layer.mha_1.parameters() if p.requires_grad)) # type: ignore
    print("\n")
    print("Cross Multi-Head Attention")
    print("Number of trainable parameters: ", sum(p.numel() for p in layer.mha_2.parameters() if p.requires_grad)) # type: ignore
    print("\n")
    print("FeedForward Layer")
    print("Number of trainable parameters: ", sum(p.numel() for p in layer.ff.parameters() if p.requires_grad)) # type: ignore
    print("\n")
    print("Add & Norm 1")
    print("Number of trainable parameters: ", sum(p.numel() for p in layer.norm1.parameters() if p.requires_grad)) # type: ignore
    print("\n")
    print("Add & Norm 2")
    print("Number of trainable parameters: ", sum(p.numel() for p in layer.norm2.parameters() if p.requires_grad)) # type: ignore
    print("\n")
    print("Add & Norm 3")
    print("Number of trainable parameters: ", sum(p.numel() for p in layer.norm3.parameters() if p.requires_grad)) # type: ignore
    print("\n")

if __name__ == "__main__":
    main()