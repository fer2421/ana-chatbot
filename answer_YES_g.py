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

# answer given to the user
answer = ""

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
openai.api_key = os.environ["OPENAI_API_KEY"]

key = "OPENAI_API_KEY"
if key in os.environ:
    print(f"{key} is set")
    print(f"Value: {os.environ[key]}")
else:
    print(f"{key} is not set")

# Embeddings
embeddings_path = "dementia_info.csv"
df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

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

# function to find the most similar question in the approved db
def find_most_similar_question(user_question, csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Load the multilingual NLP model
    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

    # Encode the user question and all the questions from the CSV file
    question_embeddings = model.encode(df.iloc[:, 0].values.tolist() + [user_question])

    # Calculate the cosine similarity between the user question and all the questions
    similarities = cosine_similarity([question_embeddings[-1]], question_embeddings[:-1])[0]

    # Find the index of the most similar question
    most_similar_index = similarities.argmax()

    # Retrieve the cosine similarity value of the most similar question
    most_similar_value = similarities[most_similar_index]
    
    # Retrieve the most similar question from the DataFrame
    most_similar_question = df.iloc[most_similar_index, 0]

    return most_similar_question, most_similar_value, most_similar_index


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
    strings, relatednesses, indices = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the information below about dementia and caregiving to answer the subsequent question in Spanish. If the answer cannot be found in the text, write "No encontré la respuesta."'
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
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You help dementia caregivers by answering questions about dementia in Spanish in less than 100 words."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

''' Example:
answer = ask('A cuales centros de atencion podria llevar a mi papa con demencia?', print_message=True)
print(answer)
'''

def get_answer(user_question):

    most_similar_question, most_similar_value, most_similar_index = find_most_similar_question(user_question, "Question_Dataset_2.csv")

    if most_similar_value >= 0.8:
        answer = str(answers[most_similar_index + 1])
    else:
        # find the most related doc
        strings, relatednesses, indices = strings_ranked_by_relatedness(user_question, df, top_n=1)
        # if the relatedness is above threshold then that topic is relevant for the answer to the user
        if float(relatednesses[0]) > 0.8:
            answer = ask(user_question) + f"\n Esta respuesta proviene de: {sources[indices[0]]}"
        else:
            answer = "Perdón! No encontré la respuesta."

    return answer

user_q = "Como puedo ayudar a la persona con demencia a que sea menos apatica y se levante a hacer cosas?"
text = get_answer(user_q)

print("Question: " + user_q)
print("Answer: " + text)