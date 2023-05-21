import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import tiktoken  # for counting tokens
import csv

GPT_MODEL = "gpt-3.5-turbo"
csv_file = "Context_Dataset_V2.csv"
openai.api_key = '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'

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# example
# n_tokens = num_tokens(str)

# Read the CSV file and retrieve the documents
documents = []
low_level_topics = []
with open(csv_file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        documents.append(row[0].strip())
        low_level_topics.append(row[1].strip())


''' # Checking that all strings in documents list are less than 1600 tokens!
doc_tokens = []
for i in range(len(documents)):
    tokens_in_doc = num_tokens(documents[i])
    doc_tokens.append(num_tokens(documents[i]))
    print(low_level_topics[i] + " = " + str(doc_tokens[i]))
'''

# calculate embeddings
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

embeddings = []
for batch_start in range(0, len(documents), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = documents[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": documents, "embedding": embeddings})

# save document chunks and embeddings
SAVE_PATH = "dementia_info.csv"

df.to_csv(SAVE_PATH, index=False)