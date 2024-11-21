import json
import shutil
from pathlib import Path

def process_file(input_file, output_file):
    # Ensure the input file exists
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"The file {input_file} does not exist.")

    # Copy the input file to the output file
    shutil.copy(input_file, output_file)

    # Load the JSON data from the copied file
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Update dep_graph with gold_graph where applicable
    for question in data.get("data", []):
        if "gold_graph" in question:
            question["dep_graph"] = question["gold_graph"]

    # Save the updated data back to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# File paths
input_file = "/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/MATH_subclaims_with_scores.json"  # Replace with your actual input file path
output_file = "/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/data/MATH_subclaims_with_scores_gold.json"  # Replace with your desired output file path

# Process the file
process_file(input_file, output_file)
print(f"File has been processed and saved to {output_file}")
