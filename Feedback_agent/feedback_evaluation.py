import feedback_eval_rubrics as er
import pandas as pd
from openai import OpenAI

def evaluate_feedback(file_path, CoT=False, few_shot=False):
    # read feedbacks from generator
    df = pd.read_csv(file_path, encoding='utf-8')

    prompt_list = df["prompt"]
    essay_list = df["essay"]
    feedback_list = df["feedback"]
    score_list = df["band"]

    evaluation_results = []
    client = OpenAI()

    for (prompt, essay, score, feedback) in zip(prompt_list, essay_list, score_list, feedback_list):
        # construct user content for evaluation
        user_content = f"Prompt: {prompt}\nEssay: {essay}\nBand_score: {score}\nFeedback: {feedback}"
        
        
        # system rubric for evaluator
        system_content = """
        You are good at evaluating the generated feedback for the IELTS writing response. 
        Given the following following writing question, student essay and the corresponding feedback, 
        evaluate the quality of the feedback based on authenticity and effectiveness. 
        Specifically, consider whether the feedback is constructive, 
        addresses key aspects of writing (such as grammar, coherence, vocabulary, and task completion), 
        and provides actionable insights. Rate the feedback on a scale from 1 (poor) to 5 (excellent), and briefly explain your reasoning. Be concise and brief.
        """


        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        )
        
        evaluation = completion.choices[0].message.content
        evaluation_results.append(evaluation)

    

    export_df = pd.DataFrame({
        "prompt+essay": [f"Prompt: {p}\nEssay: {e}" for p, e in zip(prompt_list, essay_list)],
        "band": score_list,
        "feedback": feedback_list,
        "evaluation": evaluation_results
    })


    export_path = "./evaluator_result/evaluation"

    if CoT:
        export_path += "_CoT"
    if few_shot:
        export_path += "_FewShot"
    
    export_df.to_csv(f"{export_path}.csv", index=False, encoding='utf-8', mode="w")
    return None


if __name__ == "__main__":
    evaluate_feedback("./generator_result/feedback.csv", CoT=False, few_shot=False)
    '''evaluate_feedback("./generator_result/feedback_CoT.csv", CoT=True, few_shot=False)
    evaluate_feedback("./generator_result/feedback_FewShot.csv", CoT=False, few_shot=True)
    evaluate_feedback("./generator_result/feedback_CoT_FewShot.csv", CoT=True, few_shot=True)'''