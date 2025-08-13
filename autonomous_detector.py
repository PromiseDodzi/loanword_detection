import csv
from collections import defaultdict
from basic_model import BorrowingDetector

def run_detector(input_file,output_file, obtain_output="yes"):

    """
    Process data and run basic autonomous loanword detector.

    Args:
        input file: a tsv file containing preprocessed data
        output file: name  and folder of output file

    Returns:
        file: a tsv file that contains probabilities associated with each word, and also predictions.
    """

    language_data = defaultdict(list)  

    with open(input_file, "r", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            word = row["word"]
            ipa=row['ipa']
            pos = row["pos"]
            concept=row["concept"]
            label = int(row["label"])
            language = row["language"]
            language_data[language].append((word,ipa, pos, concept, label))

    predicted_prob={}
    predicted_status_map = {}

    for language, items in language_data.items():
        ipa = [w for _,w, _, _, _ in items]
        word_pos_dict = {w: pos for _, w, pos, _, _ in items}

        print(f"Processing language: {language}")

        # Run borrowing detector
        detector = BorrowingDetector(
            threshold=0.4, 
            max_iterations=20, 
            ngram_range=(2, 10)) #when ablated model that does not use n-gram is deployed, remember to eliminate this argument
        borrowed_words, probabilities = detector.detect_borrowed_words(ipa, word_pos_dict)
 
        for word in ipa:
            predicted_status_map[word] = 1 if word in borrowed_words else 0
            predicted_prob[word] = probabilities[word]


    if obtain_output=="yes":
        with open(output_file, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["word","ipa", "pos", "concept", "language",  "true_label", "predicted_proba", "basic_prediction"])
            for language, items in language_data.items():
                for word, ipa, pos, concept, true_label in items:
                    basic_prediction = predicted_status_map.get(ipa, 0)
                    predicted_proba = predicted_prob.get(ipa, 0)
                    writer.writerow([word, ipa, pos, concept, language, true_label, predicted_proba, basic_prediction])

    return language_data, predicted_status_map, predicted_prob

if __name__== "__main__":

    input_file="datasets/cleaned_data.tsv" #this can be the sampled_data from result_files to see relationship between proportion and results
    output_file="result_files/data_with_predictions_basic.tsv"
    _, _, _ = run_detector(input_file=input_file, output_file=output_file, obtain_output="yes")