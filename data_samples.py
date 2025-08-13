import csv
from collections import defaultdict
import random
import os

def create_balanced_sample(input_file, output_file, proportion):
    """
    Creates a balanced sample from the input TSV file that maintains:
    1. The specified proportion of the original data
    2. Balanced representation of both labels (1 and 0)
    3. Proportional representation from each language

    Args:
        input_file (str): Path to input TSV file
        output_file (str): Path to output TSV file
        proportion (float): Proportion of data to sample (0.0 to 1.0)
    """
  
    data_by_language_label = defaultdict(lambda: defaultdict(list))

    with open(input_file, "r", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        fieldnames = reader.fieldnames

        for row in reader:
            language = row['language']
            label = row['label']
            data_by_language_label[language][label].append(row)

    sampled_rows = []

    for language in data_by_language_label:
        for label in data_by_language_label[language]:
            group_data = data_by_language_label[language][label]
            sample_size = max(1, int(len(group_data) * proportion))
            sampled_rows.extend(random.sample(group_data, sample_size))

    with open(output_file, "w", encoding="utf-8", newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(sampled_rows)

if __name__=="__main__":

    folder_path = "result_files"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    input_file="datasets/cleaned_data.tsv"
    output_file=f"{folder_path}/sampled_data.tsv"
    proportion =0.5
    create_balanced_sample(input_file=input_file, output_file=output_file, proportion=proportion)