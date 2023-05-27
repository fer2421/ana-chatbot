import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from IPython.display import display
import csv
import os
import time



# answer given to the user
answer = ""

# variables
strings = None
relatednesses = None
indices = None

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = os.environ["OPENAI_API_KEY"]
key = "OPENAI_API_KEY"


# Embeddings
embeddings_path_info = "dementia_info.csv"
df_info = pd.read_csv(embeddings_path_info)
# convert embeddings from CSV str type back to list type
df_info['embedding'] = df_info['embedding'].apply(ast.literal_eval)

# Embeddings
embeddings_path_q = "questions_embeddings.csv"
df_q = pd.read_csv(embeddings_path_q)
# convert embeddings from CSV str type back to list type
df_q['embedding'] = df_q['embedding'].apply(ast.literal_eval)

# Contexts Data
documents = []
low_level_topics = []
sources = []
with open("Context_Dataset_V2.csv", 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        documents.append(row[0].strip())
        low_level_topics.append(row[1].strip())
        sources.append(row[4].strip())

# Questions Data
questions = []
answers = []
low_level_topics_Q = []
with open("Question_Dataset_2.csv", 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        questions.append(row[0].strip())
        answers.append(row[1].strip())
        low_level_topics_Q.append(row[2].strip())

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

# number of tokens in a string function
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# Searches for text relevant to the query
# Stuffs that text into a message for GPT
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    #strings, relatednesses, indices = strings_ranked_by_relatedness(query, df, top_n=1)
    global strings
    global relatednesses
    global indices
    introduction = 'Use the information below about dementia and caregiving to answer the subsequent question in Spanish in less than 100 words. If the answer cannot be found in the text, write "No encontré la respuesta."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nInformation section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


# Sends the message to GPT
# Returns GPT's answer
def ask(
    query: str,
    df: pd.DataFrame = df_info,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    print("hi")
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You help dementia caregivers by answering questions about dementia."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.6,
        #max_tokens=512
    )
    print("bye")
    response_message = response["choices"][0]["message"]["content"]
    return response_message

''' Example:
answer = ask('A cuales centros de atencion podria llevar a mi papa con demencia?', print_message=True)
print(answer)
'''

def get_answer(user_question):

    # find the most related question
    question, relatednesses_q, indices_q = strings_ranked_by_relatedness(user_question, df_q, top_n=1)
    print(relatednesses_q[0])
    if float(relatednesses_q[0]) > 0.95:
        answer = str(answers[indices_q[0]])
    else:
        # find the most related doc
        global strings
        global relatednesses
        global indices
        strings, relatednesses, indices = strings_ranked_by_relatedness(user_question, df_info, top_n=1)
        print(relatednesses[0])
        # if the relatedness is above threshold then that topic is relevant for the answer to the user
        if float(relatednesses[0]) > 0.8:
            answer = ask(user_question) + f"\n Esta respuesta proviene de: {sources[indices[0]]}"
        else:
            answer = "Perdón! No encontré la respuesta."

    return answer


user_q = "Como puedo ayudar a la persona con demencia a ser menos apatica y a que se levante a hacer cosas?"
start = time.time()
text = get_answer(user_q)
done = time.time()
elapsed = done - start
print("Question: " + user_q)
print("Answer: " + text)
print(elapsed)