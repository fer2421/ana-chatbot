import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from IPython.display import display
import csv

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

csv_file = "Context_Dataset_V2.csv"
openai.api_key = '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'
embeddings_path = "dementia_info.csv"
df = pd.read_csv(embeddings_path)

documents = []
low_level_topics = []
with open(csv_file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        documents.append(row[0].strip())
        low_level_topics.append(row[1].strip())

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float], list[int]]:
    """Returns a list of strings, relatednesses, and indices, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    embeddings = df['embedding'].to_list()
    relatednesses = [relatedness_fn(query_embedding, embedding) for embedding in embeddings]
    indices = sorted(range(len(relatednesses)), key=lambda k: relatednesses[k], reverse=True)
    strings = df['text'].iloc[indices[:top_n]].tolist()
    relatednesses = [relatednesses[i] for i in indices[:top_n]]
    return strings, relatednesses, indices[:top_n]

# examples
strings, relatednesses, indices = strings_ranked_by_relatedness("Mi mama se levanta en las noches para ir al baño, que puedo hacer al respecto?", df, top_n=5)
for string, relatedness, index in zip(strings, relatednesses, indices):
    print(f"{relatedness=:.3f}")
    print(f"Topic: {low_level_topics[index]}")
    display(string)
