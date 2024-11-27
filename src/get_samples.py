import os
from openai import OpenAI
from together import Together
from openai import AzureOpenAI
import numpy as np
import json 
import math
from src.helpers.non_conformity import r_score, annotate, highest_risk_graph
import numpy as np
from src.helpers.sayless import query_model

'''
i/p: risk scores
o/p: risk quantile
'''
def compute_quantile(scores, alpha):
    n = len(scores)
    scores = [-num for num in scores]
    scores.sort()

    index = ((math.ceil((n + 1) * (1 - alpha)) - 1))

    # in case sample size is too small, must simply omit every claim
    if index > (n-1):
        return -1000
    
    return -scores[index]
    
'''
i/p: questions; method: method[0] = graph search, method[1] = calib. anno; alphas; beta
o/p: claims retained, realized factuality for each alpha
'''
def store_qs(questions, method, alphas, beta = 0):

    method1 = method[0]
    method2 = method[1]

    correct_qs_1 = []
    correct_qs_2 = []
    incorrect_qs_1 = []
    incorrect_qs_2 = []

    seed = 42
    np.random.seed(seed)
    noise = np.random.normal(0, 1, len(questions))
    noise = noise.tolist()

    for alpha in alphas:

        for i in range(len(questions)):

            # 49-1 split
            calibration_set, validation_set = questions[:i] + questions[i+1:], questions[i]
            noise_calib = noise[:i] + noise[i+1:]

            # compute op 1
            scores_1 = [r_score(q, n, method1[0], method1[1], beta = beta) for q, n in zip(calibration_set, noise_calib)]
            quantile_1 = compute_quantile(scores_1, alpha)
            
            U_1 = highest_risk_graph(quantile_1, noise[i], validation_set, method1[0], beta = beta)
            subc_1 = [validation_set["claims"][j]["subclaim"] for j in U_1]

            # compute op 2
            scores_2 = [r_score(q, n, method2[0], method2[1], beta = beta) for q, n in zip(calibration_set, noise_calib)]
            quantile_2 = compute_quantile(scores_2, alpha)
            
            U_2 = highest_risk_graph(quantile_2, noise[i], validation_set, method2[0], beta = beta)
            subc_2 = [validation_set["claims"][k]["subclaim"] for k in U_2]

            # filtering methods vary and both give non-empty outputs
            if U_1 != U_2 and len(U_1) > 0 and len(U_2) > 0:
                if annotate(U_1, validation_set, "ind") == 1:
                    correct_qs_1.append([validation_set["prompt"], subc_1])
                elif annotate(U_1, validation_set, "ind") == 0:
                    incorrect_qs_1.append([validation_set["prompt"], subc_1])
                else:
                    print(f"Missing an annotation for question: {validation_set['prompt']}")

                if annotate(U_2, validation_set, "ind") == 1:
                    correct_qs_2.append([validation_set["prompt"], subc_2])
                elif annotate(U_2, validation_set, "ind") == 0:
                    incorrect_qs_2.append([validation_set["prompt"], subc_2])
                else:
                    print(f"Missing an annotation for question: {validation_set['prompt']}")

    # Create a dictionary to store the samples
    samples1 = {
        "correct": correct_qs_1,
        "incorrect": incorrect_qs_1
    }

    samples2 = {
        "correct": correct_qs_2,
        "incorrect": incorrect_qs_2
    }

    # Write the samples to a JSON file
    with open(f"/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/{method1[0]}_samples.json", "w") as f:
        json.dump(samples1, f, indent=4)
    with open(f"/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/{method2[0]}_samples.json", "w") as f:
        json.dump(samples2, f, indent=4)
    
    return

def comp_legibility(methods, model, open_source = False):
    if open_source:
        API_KEY = os.environ.get("TOGETHER_API_KEY")
        client = Together(api_key=API_KEY)

    else:
        # query default variables
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        OAI_KEY = os.environ.get("OAI_KEY")
        client = AzureOpenAI(
        api_version = "2024-08-01-preview",
        azure_endpoint = endpoint,
        api_key = OAI_KEY 
        )

    for method in methods:
        # Open the corresponding JSON file
        filepath = f"/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/{method}_samples.json"
        with open(filepath, "r") as file:
            data = json.load(file)

        correct_samples = data["correct"]
        incorrect_samples = data["incorrect"]

        # Initialize confusion matrix counts
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        # Function to prompt model
        # def query_model(client, prompt, model, max_tokens=1000, temperature=0, n_samples=1):        

        def check_subclaims(q_prompt, subclaims):
            subclaims_str = "\n".join(subclaims)
            leg_prompt = f"Given the prompt: {q_prompt}\n\nDo the following subclaims contain any errors?\n\n{subclaims_str}\n\nRespond with exactly one character: either 'Y' for 'there's an error' or 'N' for 'no error'."
            response = query_model(client, leg_prompt, model)
            return response.strip()

        # Check "correct" samples
        for item in correct_samples:
            q_prompt, subclaims = item
            result = check_subclaims(q_prompt, subclaims)
            if result == "Y":
                false_positive += 1
            elif result == "N":
                true_negative += 1
            else:
                print(f"Weird output: {result}")
                print()

        # Check "incorrect" samples
        for item in incorrect_samples:
            prompt, subclaims = item
            result = check_subclaims(prompt, subclaims)
            if result == "Y":
                true_positive += 1
            elif result == "N":
                false_negative += 1
            else:
                print(f"Weird output: {result}")
                print()

        # Print confusion matrix for the current method
        print(f"Confusion Matrix for method {method}:")
        print(f"True Positive: {true_positive}")
        print(f"True Negative: {true_negative}")
        print(f"False Positive: {false_positive}")
        print(f"False Negative: {false_negative}")
        print()
    
    return

if __name__ == "__main__":
    file_path = '/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/MATH_open_subclaims_with_scores.json'

    with open(file_path, 'r') as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    questions = data.get("data", [])
    
    # get questions
    store_qs(questions, [["ind", "ind"], ["simult", "graph"]],[0.1, 0.15, 0.2])

    print("Llama Grading")
    comp_legibility(["simult", "ind"], "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", open_source = True)
    print()
    print("GPT-4o Grading")
    comp_legibility(["simult", "ind"], "gpt-4o", open_source = False)
    
    # implement GPT later
    # comp_leg(["simult", "ind"], "client", "model", open = False)

    MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

