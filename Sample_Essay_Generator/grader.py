import sys
from openai import OpenAI
import numpy as np
import json
import faiss
from pathlib import Path

current_file = Path(__file__).resolve()
rubric_path = current_file.parent.parent / "Feedback_agent"
sys.path.append(str(rubric_path))

# Now you can import the IELTS_rubrics module
from rubric_and_sample import IELTS_rubrics as rubric

client = OpenAI()

def generate_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding")
        return None

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

def search_combined_similarity(query_prompt, query_essay,  topic_weight, essay_weight, top_k):
    
    essay_index_path = current_file.parent.parent / "RAG" / "faiss_index_train_essay.bin"
    prompt_index_path = current_file.parent.parent / "RAG" / "faiss_index_train_topics.bin"
    essay_embeddings_json_path = current_file.parent.parent / "RAG" / "embeddings_dataset_train_essay.json"
    prompt_embeddings_json_path = current_file.parent.parent / "RAG" / "embeddings_dataset_train_topics.json"
    
    essay_index = faiss.read_index(str(essay_index_path))
    prompt_index = faiss.read_index(str(prompt_index_path))
    
    with open(str(essay_embeddings_json_path), "r", encoding="utf-8") as f:
        essay_metadata = json.load(f)
    with open(str(prompt_embeddings_json_path), "r", encoding="utf-8") as f:
        prompt_metadata = json.load(f)

    query_prompt_embedding = generate_embedding(query_prompt)
    query_essay_embedding = generate_embedding(query_essay)
    
    if query_prompt_embedding is None or query_essay_embedding is None:
        return []
    
    query_prompt_embedding_np = np.array([query_prompt_embedding], dtype=np.float32)
    query_essay_embedding_np = np.array([query_essay_embedding], dtype=np.float32)
    faiss.normalize_L2(query_prompt_embedding_np)
    faiss.normalize_L2(query_essay_embedding_np)
    
    prompt_distances, prompt_indices = prompt_index.search(query_prompt_embedding_np, len(prompt_metadata))
    essay_distances, essay_indices = essay_index.search(query_essay_embedding_np, len(essay_metadata))

    # resort according to the original indices 0,1,...
    prompt_sort_indices = np.argsort(prompt_indices[0])
    prompt_indices_sorted = prompt_indices[0][prompt_sort_indices]
    prompt_distances_sorted = prompt_distances[0][prompt_sort_indices]
    
    essay_sort_indices = np.argsort(essay_indices[0])
    essay_indices_sorted = essay_indices[0][essay_sort_indices]
    essay_distances_sorted = essay_distances[0][essay_sort_indices]

    #(index, pormpt_distance, essay_distance)
    prompt_essay_pair = list(zip(essay_indices_sorted,prompt_distances_sorted,essay_distances_sorted))

    sorted_prompt_essay_pair = sorted(prompt_essay_pair, key=lambda x: topic_weight * x[1] + essay_weight * x[2], reverse=True)[0:top_k]

    results = []
    for i, x in enumerate(sorted_prompt_essay_pair):
        idx = x[0]
        sim = topic_weight * x[1] + essay_weight * x[2]
        result = essay_metadata[idx]
        result["similarity"] = sim 
        results.append(result)
    return results
    

def get_score_prompt_version_RAG(topic, essay,topic_weight, essay_weight, top_k):
    
    results = search_combined_similarity(topic,essay, topic_weight=topic_weight, essay_weight=essay_weight,top_k=top_k)

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
    - Output only the final averaged score directly, only the score number. Don't be too strict on the score, be more flexible.
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

    print("\nGenerating grade...")
    
    grade = get_score_prompt_version_RAG(user_topic, user_essay, topic_weight=0.3, essay_weight=0.7, top_k=3)
    
    if '<4' in grade:
        grade = '<4'
    elif float(grade) < 4:
        grade = '<4'
        
    print(f"\nPredicted Grade: {grade}")

if __name__ == "__main__":
    main()