import os
import sys
import pandas as pd
from openai import OpenAI
import numpy as np
import json
import faiss

sys.path.append(os.path.abspath(".."))
from Feedback_agent.rubric_and_sample import IELTS_rubrics as rubric

client = OpenAI()

feedback_path = "./sample_essay.csv"
df = pd.read_csv(feedback_path, encoding='utf-8')

topic = df["topic"]
essay = df["essay"]
feedback = df["feedback"]
predicted_grade = df["predicted"]
desired_grade = df["desired"]
sample_grade = df["sample_score"]

def generate_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding")
        return None
    
#index = faiss.read_index("../RAG/faiss_index_train.bin")
index = faiss.read_index("../RAG/faiss_index_train_topics.bin")
#with open("../RAG/embeddings_dataset_train.json", "r", encoding="utf-8") as f:
with open("../RAG/embeddings_dataset_train_topics.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
    
def search_cosine_similarity(query_text, top_k=3):
    
    query_embedding = generate_embedding(query_text)
    if query_embedding is None:
        return []
    
    
    query_embedding_np = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_embedding_np)
    
    
    distances, indices = index.search(query_embedding_np, top_k)
    
    
    results = []
    for i, idx in enumerate(indices[0]):
        result = metadata[idx]
        result["similarity"] = distances[0][i]  
        results.append(result)
    return results

label_map = {
    0: '<4',
    1: 4.0,
    2: 4.5,
    3: 5.0,
    4: 5.5,
    5: 6.0,
    6: 6.5,
    7: 7.0,
    8: 7.5,
    9: 8.0,
    10: 8.5,
    11: 9.0
}

def get_score_prompt_version_RAG(topic, essay):
    
    results = search_cosine_similarity(f"Prompt: {topic}\nEssay: {essay}", top_k=3)

    grader_prompt = f"""
    You are an IELTS writing section examiner, tasked with evaluating essays strictly based on the official IELTS writing rubric.

    **Writing Question**: {topic}
    **Student Essay**: {essay}
    \n
    **Reference IELTS Rubric**:
    {rubric.BASIC_RUBRIC}
    {rubric.CRITERIA}
    {rubric.BAND_SCORE}
    \n
    **Reference Essays for Consistency**:
    - Example 1: Writing Question: {results[0]['prompt']} Essay: {results[0]['essay']} Score: {label_map[float(results[0]['label'])]}
    \n
    - Example 2: Writing Question: {results[1]['prompt']} Essay: {results[1]['essay']} Score: {label_map[float(results[1]['label'])]}
    \n
    - Example 3: Writing Question: {results[2]['prompt']} Essay: {results[2]['essay']} Score: {label_map[float(results[2]['label'])]}
    \n
    **Guidelines for Scoring**:
    - Assign scores in 0.5 intervals from 0 to 9 based on the IELTS rubric.
    - If the essay content is irrelevant or off-topic, assign a score of 0.
    - Avoid generic scores like 5, 6, or 7 unless the essay fully justifies such a rating.
    - Use the provided example essays and their scores to guide your grading and ensure consistency.

    **Enhanced Scoring Process**:
    - Generate multiple scores for the same essay by slightly varying the context or examples provided (e.g., shuffle or modify reference essays where appropriate).
    - Compute the average of these scores to improve reliability.
    - If scores vary significantly, consider revisiting the rubric alignment for the given essay.

    **Final Output**:
    - Output only the final averaged score directly, only the score number, but if the score is smaller than 4, output only the '<4'. Don't be too strict on the score, be more flexible.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": grader_prompt}
        ]
    )
    return response.choices[0].message.content

def main():
    print("Enter the topic for the essay:")
    user_topic = input("> ")

    print("\nEnter the essay text:")
    user_essay = input("> ")

    #print("\nGenerating grade...")
    
    grade = get_score_prompt_version_RAG(user_topic, user_essay)
    
    if '<4' in grade:
        grade = '<4'
    elif float(grade) < 4:
        grade = '<4'
        
    print(f"\nPredicted Grade: {grade}")

if __name__ == "__main__":
    main()