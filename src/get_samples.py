import sys
import copy
from openai import OpenAI
import numpy as np
import json 
import matplotlib.pyplot as plt
import math
from src.helpers.non_conformity import r_score, annotate, highest_risk_graph
import random
import numpy as np
from src.helpers.dependent_cp.greedy import greedy_search
from src.helpers.dependent_cp.simult import simult_search, to_valid_array

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
        print(f"Alpha: {alpha}")
        print()
        # keep track of num correct/incorrect ops
        correct_1 = 0
        incorrect_1 = 0
        correct_2 = 0
        incorrect_2 = 0

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
            subc_2 = [validation_set["claims"][j]["subclaim"] for j in U_2]

            if U_1 != U_2:
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

                
                correct_1 += annotate(U_1, validation_set, "ind")
                incorrect_1 += (1 - annotate(U_1, validation_set, "ind"))
                correct_2 += annotate(U_2, validation_set, "ind")
                incorrect_2 += (1 - annotate(U_2, validation_set, "ind"))

    

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

if __name__ == "__main__":
    file_path = '/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/MATH_subclaims_with_scores.json'

    with open(file_path, 'r') as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    questions = data.get("data", [])
    
    # get questions
    print("Alphas = 0.1, 0.15, 0.2")
    print()
    store_qs(questions, [["ind", "ind"], ["simult", "graph"]],[0.1, 0.15, 0.2])

