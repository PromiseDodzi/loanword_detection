import csv

def get_data_stats(input_data):
    total_data = 0
    concepts = set()
    words_per_language = {}
    label_counts = {0: 0, 1: 0}
    label_language_counts = {}

    with open(input_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            total_data += 1

            concept = row["concept"].lower()
            concepts.add(concept)

            language = row["language"].lower()
            words_per_language[language] = words_per_language.get(language, 0) + 1

            label = int(row["label"])
            label_counts[label] += 1

            if language not in label_language_counts:
                label_language_counts[language] = {0: 0, 1: 0}
            label_language_counts[language][label] += 1

    # Print summary statistics
    print(f"Total number of data: {total_data}")
    print(f"Total number of unique concepts: {len(concepts)}")
    print("Total number of words for each language:")
    for lang, count in words_per_language.items():
        print(f"  {lang}: {count}")
    print(f"Total number of label=0: {label_counts[0]}")
    print(f"Total number of label=1: {label_counts[1]}")
    print("Total number of label=0 for each language:")
    for lang, labels in label_language_counts.items():
        print(f"  {lang}: {labels[0]}")
    print("Total number of label=1 for each language:")
    for lang, labels in label_language_counts.items():
        print(f"  {lang}: {labels[1]}")

if __name__=="__main__":
    input_name="datasets\cleaned_data.tsv"
    get_data_stats(input_data=input_name)