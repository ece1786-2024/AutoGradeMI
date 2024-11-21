import os
import pandas as pd
import Feedback_agent.rubric_and_sample.IELTS_rubrics as rubric
from openai import OpenAI

# input: topic, essay, feedback, and desired grade
def generate_sample_essay(topic, essay, feedback, desired_grade):
    return None