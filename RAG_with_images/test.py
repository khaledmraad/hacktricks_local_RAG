from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model = HuggingFaceEmbedding(
    model_name="llamaindex/vdr-2b-multi-v1",
    device="cpu",  # "mps" for mac, "cuda" for nvidia GPUs
    trust_remote_code=True,
)

image_embedding = model.get_image_embedding("RAG_with_images/data_wiki/10.jpg")
print(image_embedding)
# query_embedding = model.get_query_embedding("some query")

