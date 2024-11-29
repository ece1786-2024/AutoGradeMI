import gradio as gr
from pathlib import Path
import sys
import time

integrator_path = Path(__file__).resolve().parent / "integration"
sys.path.append(str(integrator_path))
import integrate as inter

def process_input(topic, essay, desired_score):
    # Validate the desired score
    try:
        desired_score = float(desired_score)
        if desired_score < 0 or desired_score > 9 or (desired_score * 10) % 5 != 0:
            return "Invalid Input", "Desired Score must be between 0 and 9 with 0.5 intervals!", ""
    except ValueError:
        return "Invalid Input", "Desired Score must be a number between 0 and 9 with 0.5 intervals!", ""
    
    # Validate essay length
    word_count = len(essay.split())
    if word_count > 400:
        return "Invalid Input", "The essay exceeds the maximum length of 400 words!", ""

    try:
        results = inter.process_essay(topic, essay, desired_score)
        predicted_score = f"Predicted Score: {results['predicted_score']}"
        feedback = results["feedback"]
        sample_essay = results["sample_essay"]
        return predicted_score, feedback, sample_essay
    except Exception as e:
        return "Error", f"An error occurred: {e}", ""

# Define the Gradio interface
my_theme = gr.Theme.from_hub('earneleh/paris')

with gr.Blocks(theme=my_theme) as demo:

    with gr.Row():
        with gr.Column(scale=8):  
            gr.Markdown("# IELTS Essay Evaluator")
            gr.Markdown(
                "Provide an essay topic, your essay, and the desired band score to receive predictions, feedback, and a sample essay."
            )
        with gr.Column(scale=1, min_width=150):  
            toggle_dark = gr.Button(value="Toggle Dark")          
             
    with gr.Row():
        topic_input = gr.Textbox(label="Essay Topic", placeholder="Enter your essay topic...")
    
    with gr.Row():
        essay_input = gr.Textbox(
            label="Essay",
            placeholder="Write your essay here...",
            lines=10,
            max_lines=15,
        )

    with gr.Row():
        score_input = gr.Slider(
            minimum=0,
            maximum=9,
            step=0.5,
            label="Desired Band Score",
            value=6.5,
        )

    with gr.Row():
        submit_btn = gr.Button(value="Evaluate Essay")

    with gr.Row():
        predicted_output = gr.Textbox(label="Predicted Score", interactive=False)
        feedback_output = gr.Textbox(label="Feedback", interactive=False, lines=5)
        sample_output = gr.Textbox(label="Sample Essay", interactive=False, lines=5)

    submit_btn.click(
        fn=process_input,
        inputs=[topic_input, essay_input, score_input],
        outputs=[predicted_output, feedback_output, sample_output],
    )
    toggle_dark.click(
        None,
        None,
        None,
        js="""
        () => {
            document.body.classList.toggle('dark');
        }
        """,
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
