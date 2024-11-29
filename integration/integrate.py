import sys
from pathlib import Path

# Define paths to the external directories
feedback_agent_path = Path(__file__).resolve().parent.parent / "Feedback_agent" / "integration"
sample_agent_path = Path(__file__).resolve().parent.parent / "Sample_Essay_Generator"

# Add paths to sys.path for importing modules
sys.path.append(str(feedback_agent_path))
sys.path.append(str(sample_agent_path))

# Import external modules
import two_agent_Interaction_new_prompt as feedback_agent
import grader as grader_agent 
import sample_essay_generator as sample_agent

def process_essay(topic: str, essay: str, desired_score: float):
    """
    Process the essay based on the provided topic and desired score.

    Parameters:
    - topic (str): The essay topic.
    - essay (str): The user's essay.
    - desired_score (float): The desired band score.

    Returns:
    - dict: A dictionary containing predicted score, feedback, and a sample essay.
    """
    # Predicted score (from grader_agent)
    predicted_score = float(grader_agent.get_score_prompt_version_RAG(topic, essay, topic_weight=0.3, essay_weight=0.7, top_k=3))
    
    if predicted_score >= desired_score:
        
        if predicted_score == 9.0:
            feedback = "Great job! Your essay got full score. Keep up the good work!"
            sample_essay = "Great job! Your are really good at English. Keep up the good work!"  
            return {
                "predicted_score": predicted_score,
                "feedback": feedback,
                "sample_essay": sample_essay
            }
            
        else:
            if predicted_score == 8.5:
                desired_score = 9.0
            else:
                desired_score = predicted_score + 1.0  
            feedback = feedback_agent.generate_feedback(topic, essay, predicted_score, desired_score)
            if feedback == "Failed to generate 'good' feedback within the turn limit. Please try again.":
                feedback = f"Great job! Your essay is already meet the desired score you want. But I failed to provide you with 'good' feedback to reach a score of {desired_score} within the turn limit.\n\n"
                sample_essay = "Sorry, we cannot generate a sample essay without a feedback. Please try again"
            else:
                feedback_head = f"Great job! Your essay is already meet the desired score you want. But I still can try to provide you with feedback and a sample essay to reach a score of {desired_score}.\n\n"
                feedback = feedback_head + feedback
                for i in range(10):
                    sample_essay = sample_agent.generate_sample_essay(topic, essay, feedback, predicted_score, desired_score)
                    sample_score = float(grader_agent.get_score_prompt_version_RAG(topic, sample_essay, topic_weight=0.3, essay_weight=0.7, top_k=3))
                    if float(sample_score) >= float(desired_score):
                        break
                    if i == 9:
                        sample_essay = "Sorry, we cannot generate a sample essay that meets the desired score base on your essay. Please try again"
    
    else:
        feedback = feedback_agent.generate_feedback(topic, essay, predicted_score, desired_score)
        if feedback == "Failed to generate 'good' feedback within the turn limit. Please try again.":
            sample_essay = "Sorry, we cannot generate a sample essay without a feedback. Please try again"
        else:
            for i in range(10):
                sample_essay = sample_agent.generate_sample_essay(topic, essay, feedback, predicted_score, desired_score)
                sample_score = float(grader_agent.get_score_prompt_version_RAG(topic, sample_essay, topic_weight=0.3, essay_weight=0.7, top_k=3))
                if float(sample_score) >= float(desired_score):
                    break
                if i == 9:
                    sample_essay = "Sorry, we cannot generate a sample essay that meets the desired score base on your essay. Please try again"
        
    # Output grade
    if predicted_score < 4:
        output_score = '<4'
    else:
        output_score = predicted_score
        
    
    return {
        "predicted_score": output_score,
        "feedback": feedback,
        "sample_essay": sample_essay
    }

# Example for testing:
if __name__ == "__main__":
    # Test the function with predefined parameters
    test_topic = "Rich countries often give money to poorer countries, but it does not solve poverty. Therefore, developed countries should give other types of help to the poor countries rather than financial aid. To what extent do you agree or disagree?"
    test_essay = "Poverty represents a worldwide crisis. It is the ugliest epidemic in a region, which could infect countries in the most debilitating ways. To tackle this issue, rich countries need to help those in need and give a hand when possible. I agree that there are several ways of aiding poor countries other than financial aid, like providing countries in need with engineers, workers, and soldiers who would build infrastructure. Building universities, hospitals, and roadways. By having a solid infrastructure, poor countries would be able to monetise their profits and build a stronger and more profitable economy which would help them in the long term. Once unprivilged countries find their niche, the major hurdle would be passed and would definitely pave the way for much brighter future. However, I do disagree that financial aid does not solve poverty, it does if used properly and efficiently. The most determining factor if financial aid would be the way to go, is by identifying what type of poor countries' representative are dealing with. Some countries will have a responsible leader and some will not, with that being said, implementing a strategy, to distinguish responsible leaders from others, would tailor the type of aid rich countries could use. An example, A clear report and constant observation would be applied to track the progress and how this type of aid is being monetized. In summary, types of aid varies from country to another, and tailoring the type of aid is of paramount importance to solve this problem that had huge toll on poor countries."
    test_desired_score = 6.0

    results = process_essay(test_topic, test_essay, test_desired_score)
    
    # Output results
    print("\n--- Results ---")
    print(f"Predicted Score: {results['predicted_score']}")
    print(f"Feedback: {results['feedback']}")
    print(f"Sample Essay: {results['sample_essay']}")