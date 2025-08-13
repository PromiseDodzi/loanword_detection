import csv

def calculate_confusion_matrix(true_labels, predicted_labels):
    """Calculate TP, TN, FP, FN manually"""
    tp = tn = fp = fn = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    return tp, tn, fp, fn

def precision_score(tp, fp):
    """Precision = TP / (TP + FP)"""
    denominator = tp + fp
    return tp / denominator if denominator != 0 else 0

def recall_score(tp, fn):
    """Recall = TP / (TP + FN)"""
    denominator = tp + fn
    return tp / denominator if denominator != 0 else 0

def f1_score(precision, recall):
    """F1 = 2 * (precision * recall) / (precision + recall)"""
    denominator = precision + recall
    return 2 * (precision * recall) / denominator if denominator != 0 else 0

def evaluate_predictions(tsv_file_path, predictions):
    """
    Evaluate prediction performance against true labels.

    This function reads a tab-separated values (TSV) file containing true labels
    and predicted labels, computes common classification metrics (precision,
    recall, F1 score), and prints them along with the confusion matrix counts.

    Args:
        tsv_file_path (str): Path to the TSV file containing at least two columns:
            - "true_label": The ground truth binary labels (0 or 1).
            - A column matching the `predictions` argument: The predicted binary labels (0 or 1).
        predictions (str): The column name in the TSV file containing predicted labels.


    Outputs:
        Prints:
            - True Positives (TP)
            - True Negatives (TN)
            - False Positives (FP)
            - False Negatives (FN)
            - Precision (float, 4 decimal places)
            - Recall (float, 4 decimal places)
            - F1 Score (float, 4 decimal places)

    Information:
        - Requires `precision_score`, `recall_score`, `f1_score` and `calculate_confusion_matrix` functions.
        - All label values are expected to be integers (0 or 1).
    """
    true_labels = []
    predicted_labels = []

    with open(tsv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            true_labels.append(int(row["true_label"]))
            predicted_labels.append(int(row[predictions]))

    # Calculate metrics
    tp, tn, fp, fn = calculate_confusion_matrix(true_labels, predicted_labels)
    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    f1 = f1_score(precision, recall)

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")



def evaluate_predictions_by_language(tsv_file_path, predictions):
    """
    Evaluate classification predictions grouped by language from a TSV file.

    This function reads a tab-separated values (TSV) file containing true labels,
    predicted labels, and language information. For each language, it calculates
    and prints classification metrics (precision, recall, F1 score) as well as
    confusion matrix values (TP, TN, FP, FN). It also returns a dictionary of 
    results for further processing.

    Parameters
    ----------
    tsv_file_path : str
        Path to the TSV file containing predictions and true labels.
    predictions : str
        The column name in the TSV file that contains the predicted labels.

    Returns
    -------
    dict
        A dictionary where keys are language names and values are dictionaries
        containing:
            - 'num_samples': int, number of samples for the language
            - 'TP': int, true positives
            - 'TN': int, true negatives
            - 'FP': int, false positives
            - 'FN': int, false negatives
            - 'precision': float, precision score
            - 'recall': float, recall score
            - 'f1_score': float, F1 score

    """
  
    language_data = {}
    results_dict = {}  

    with open(tsv_file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            language = row.get("language", "unknown")
            if language not in language_data:
                language_data[language] = {
                    "true_labels": [],
                    "predicted_labels": []
                }
            language_data[language]["true_labels"].append(int(row["true_label"]))
            language_data[language]["predicted_labels"].append(int(row[predictions]))

    for language, labels in language_data.items():
        true_labels = labels["true_labels"]
        predicted_labels = labels["predicted_labels"]

        if not true_labels:
            continue

        print(f"\nLanguage: {language}")
        print(f"Number of samples: {len(true_labels)}")

        tp, tn, fp, fn = calculate_confusion_matrix(true_labels, predicted_labels)
        precision = precision_score(tp, fp)
        recall = recall_score(tp, fn)
        f1 = f1_score(precision, recall)

        results_dict[language] = {
            "num_samples": len(true_labels),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"True Positives (TP): {tp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return results_dict


if __name__=="__main__":
    
    print("Metrics relative to results of basic autonomous model")
    print("*"*40)
    tsv_file_path="result_files/data_with_predictions_basic.tsv"
    predictions="basic_prediction"
    evaluate_predictions(tsv_file_path, predictions=predictions)
    results = evaluate_predictions_by_language(tsv_file_path, predictions=predictions)
    print("*"*40)
    print("Metrics relative to results of scaled model")
    print("*"*40)
    predictions="scaled_prediction"
    tsv_file_path="result_files/data_with_predictions_scaled.tsv"
    evaluate_predictions(tsv_file_path, predictions=predictions)
    results = evaluate_predictions_by_language(tsv_file_path, predictions=predictions)