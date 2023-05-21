import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv

csv_file = "Question_Dataset_2.csv"
questions = []
answers = []
low_level_topics_Q = []
with open(csv_file, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        questions.append(row[0].strip())
        answers.append(row[1].strip())
        low_level_topics_Q.append(row[2].strip())



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

'''
def find_most_similar_question(user_question, csv_file, batch_size=100):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Load the multilingual NLP model
    model = SentenceTransformer('distilbert-base-multilingual-cased')

    # Encode all the questions from the CSV file
    question_embeddings = model.encode(df.iloc[:, 0].values.tolist())

    # Encode the user question
    user_question_embedding = model.encode([user_question])[0]

    # Calculate the cosine similarity between the user question and all the questions in batches
    similarities = []
    for i in range(0, len(question_embeddings), batch_size):
        batch_embeddings = question_embeddings[i:i+batch_size]
        batch_similarities = cosine_similarity([user_question_embedding], batch_embeddings)[0]
        similarities.extend(batch_similarities)

    # Find the index of the most similar question
    most_similar_index = np.argmax(similarities)

    # Retrieve the most similar question from the DataFrame
    most_similar_question = df.iloc[most_similar_index, 0]

    # Retrieve the cosine similarity value of the most similar question
    most_similar_value = similarities[most_similar_index]

    return most_similar_question, most_similar_value
'''

# Example usage
answer = ""
question = ""
user_question = "¿Cómo puedo calmar a la persona con demencia cuando tiene alucinaciones?"
csv_file = "Question_Dataset_2.csv"
most_similar_question, most_similar_value, most_similar_index = find_most_similar_question(user_question, csv_file)


if most_similar_value >= 0.7:
    question = str(most_similar_question)
    answer = str(answers[most_similar_index + 1])
else:
    answer = "Perdón, no tengo suficiente cononicimiento para responder esa pregunta."
    question = "No hay pregunta similar."

print(question)
print(answer)
# print("Most similar question:", most_similar_question)
# print("Most similar value:", most_similar_value)

