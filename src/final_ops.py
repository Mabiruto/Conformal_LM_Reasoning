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

import json
import numpy as np

def check_final_ops(questions, method, alphas, beta=0, math_key_file='/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/MATH_key.json'):
    total_correct = 0
    correct_w_answer = 0

    seed = 42
    np.random.seed(seed)
    noise = np.random.normal(0, 1, len(questions))
    noise = noise.tolist()

    with open(math_key_file, 'r') as f:
        math_key = json.load(f)
    
    problems = math_key['problem']
    solutions = math_key['solution']

    for alpha in alphas:
        for i in range(len(questions)):
            # 49-1 split
            calibration_set, validation_set = questions[:i] + questions[i+1:], questions[i]
            noise_calib = noise[:i] + noise[i+1:]

            # compute op
            scores = [r_score(q, n, method[0], method[1], beta=beta) for q, n in zip(calibration_set, noise_calib)]
            quantile = compute_quantile(scores, alpha)
            
            U_filt = highest_risk_graph(quantile, noise[i], validation_set, method[0], beta=beta)
            subclaims = [validation_set["claims"][j]["subclaim"] for j in U_filt]

            # check whether output factual
            if annotate(U_filt, validation_set, "ind") == 1:
                total_correct += 1

                prompt = validation_set["prompt"]
                try:
                    index = problems.index(prompt)
                    solution = solutions[index]
                    print(f"Prompt: {prompt}")
                    print()
                    print(f"Solution: {solution}")
                    print()
                    print("Subclaims:")
                    if len(subclaims) > 0:
                        for subclaim in subclaims:
                            print(f"> {subclaim}")
                        print()
                    else:
                        print("None")
                    annotation = input("Does the subclaim contain the correct final answer? (Y/N): ")
                    if annotation.upper() == 'Y':
                        correct_w_answer += 1
                except ValueError:
                    print(f"Prompt not found in problems: {prompt}")

    if total_correct > 0:
        proportion_correct = correct_w_answer / total_correct
        print(f"Proportion of correct final answers: {proportion_correct:.2f}")
    else:
        print("No valid annotations found.")
                
    
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

    # check final ops
    
    print("Our method:")
    check_final_ops(questions, ["simult", "graph"], [0.1])
    print()
    print("Independent:")
    check_final_ops(questions, ["simult", "graph"], [0.1])
    print()
    