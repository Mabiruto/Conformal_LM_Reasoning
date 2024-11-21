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
i/p: questions, method, alphas
o/p: claims retained, realized factuality for each alpha
'''
def validation(questions, method, alphas):
    n = len(questions)

    if method[2] == "oracle":
        
        most_true = {}

        for i in range(len(questions)):
            max_claims = 0
            for graph in questions[i]["graph_annotations"]["y"]:
                claims = 0 
                for item in graph:
                    claims += item
                if claims > max_claims:
                    max_claims = claims
            most_true[i] = max_claims/len(questions[i]["claims"])

        most_true_sorted = dict(sorted(most_true.items(), key=lambda item: item[1]))
        
        # vals to return
        claims_retained = []
        realized_fact = []

        for alpha in alphas:
            alpha_retained = []

            false_num = math.floor(alpha * n)  # number of items to select

            realized_fact.append((n - false_num)/n)

            idx = 0
            for key, value in most_true_sorted.items():
                idx += 1
                if idx <= false_num:
                    alpha_retained.append(1)
                else:
                    alpha_retained.append(value)

            claims_retained.append(sum(alpha_retained)/len(alpha_retained))
            
        return realized_fact, claims_retained, np.zeros(len(alphas))

    seed = 42
    np.random.seed(seed)
    noise = np.random.normal(0, 1, len(questions))
    noise = noise.tolist()
    
    realized_fact = []
    claims_retained = []
    std_err = []

    for alpha in alphas:
        
        alpha_error = []
        alpha_claims = []


        for i in range(len(questions)):
            
            # 49-1 split
            calibration_set, validation_set = questions[:i] + questions[i+1:], questions[i]
            noise_calib = noise[:i] + noise[i+1:]

            # compute quantile
            if len(method) == 6:
                mult = method[5]
            else: mult = 0
            calibration_scores = [r_score(q, n, method[2], method[3], beta = mult) for q, n in zip(calibration_set, noise_calib)]
            quantile = compute_quantile(calibration_scores, alpha)

            
            U_filt = highest_risk_graph(quantile, noise[i], validation_set, method[2], beta = mult)
            if annotate(U_filt, validation_set, method[4]) == 0:
                    alpha_error.append(1)
            else:
                alpha_error.append(0)
            
            alpha_claims.append(len(U_filt)/len(validation_set["claims"]))

        std_err.append(get_std_error(alpha_claims))
        realized_fact.append(1 - sum(alpha_error)/len(questions))
        claims_retained.append(sum(alpha_claims)/len(alpha_claims))

    return realized_fact, claims_retained, std_err

def get_plots(data, methods, alphas_valid, file_path):

    # store validation results for each method
    valid = {}

    # get data for each method
    i = 0
    for method in methods:    
        valid[method[0]] = validation(data[i], method, alphas_valid)

        i += 1

    '''
    Validation Plot
    '''

    plt.figure()

    target_fact_valid = [1 - alpha for alpha in alphas_valid]
    
    for method in methods:
        
        # color given in m
        line_color = method[1]

        name = method[0]

        realized_fact = valid[name][0]
        percent_claims = valid[name][1]

        err = valid[name][2]

        # plt.plot(target_fact_valid, percent_claims, label=name, linewidth = 1, color = line_color)
        # plt.errorbar(target_fact_valid, percent_claims, yerr=err, linewidth = 2, color = line_color)
        
        plt.plot(realized_fact, percent_claims, label=name, linewidth = 1, color = line_color)
        print(err)
        plt.errorbar(realized_fact, percent_claims, yerr=err, linewidth = 2, color = line_color)

    # Add labels and title
    # plt.xlabel('1 - Alpha')
    plt.xlabel('Coherent Factuality')
    plt.ylabel('Percent of Claims Retained')
    

    plt.title('Claims Retained vs. Coherent Factuality')
    # plt.title('Claims Retained vs. Target Factuality')

    # Add a legend
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=1)  # Adjust ncol for the number of columns

    # Show the plot
    plt.autoscale()
    out_file = f"{file_path}/valid_plot.png"
    plt.savefig(out_file, bbox_inches='tight')
    

def get_std_error(vals):
    return np.std(vals) * 1.96 / np.sqrt(len(vals))


if __name__ == "__main__":
    '''
    
    methods format: [ [label], [color], [filtering method], [calib. annos], [valid. annos], [beta (default = 0)] ]
    
    '''

    file_path_1 = '/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/MATH_subclaims_with_scores.json'


    # GPT graphs stored here
    with open(file_path_1, 'r') as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    questions1 = data.get("data", [])

    
    file_path_2 = '/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/MATH_subclaims_with_scores_gold.json'
    # Human graphs stored here
    with open(file_path_2, 'r') as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    questions2 = data.get("data", [])

    methods = []
    
    methods.append([f"GPT Graphs", "blue", "simult", "graph", "manual", 0])
    methods.append([f"Human (Ideal) Graphs", "purple", "simult", "graph", "manual", 0])

    valid_alphas = [0.1,0.2,0.3]

    questions = {}
    questions[0] = questions1[:10]
    questions[1] = questions2[:10]
    
    get_plots(questions, methods, valid_alphas, "/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/out")
