import sys
from pathlib import Path
import pandas as pd

integration_path = Path(__file__).resolve().parent.parent / "integration"

sys.path.append(str(integration_path))

import integrate

def main():
    desired_score = 8.0
    progress = 1
    df = pd.read_json('../RAG/manual_embedding_dataset_test.json')
    question_list = df["prompt"]
    essay_list=df["essay"]
    label_list = df['label']

    pred_score_list = []
    feedback_list = []
    sample_list = []
    desired_score_list = []

    print(len(question_list), len(essay_list), len(label_list))

    for (q,e) in zip(question_list, essay_list):
        print(f"current progress: {progress}")
        res_dict = integrate.process_essay(q, e, desired_score)
        pred_score_list.append(res_dict["predicted_score"])
        feedback_list.append(res_dict["feedback"])
        sample_list.append(res_dict["sample_essay"])
        desired_score_list.append(desired_score)
        progress += 1

    export_dict = dict()
    export_dict["topic"] = question_list
    export_dict["essay"] = essay_list
    export_dict["label"] = label_list
    export_dict["predicted"] = pred_score_list
    export_dict["feedback"] = feedback_list
    export_dict["sample_essay"] = sample_list
    export_dict["desired"] = desired_score_list

    df = pd.DataFrame(export_dict)
    df.to_csv("./test_integration.csv", index=False, encoding='utf-8', mode='w')
    print("DONE!!!")
    return None


if __name__ == "__main__":
    main()
