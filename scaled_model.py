from collections import defaultdict
import csv
from comparability_score import align_and_score_by_concept
from autonomous_detector import run_detector


from collections import defaultdict

def word_comparability(input_file, output_file):
    """
    Compute a comparability score for each word across languages based on a shared concept.

    This function processes linguistic data from the specified input file,
    grouping words by their conceptual meaning, then scoring their similarity
    across different languages. The result is an average comparability score 
    for each word, indicating how phonetically/structurally similar it is to 
    words for the same concept in other languages.

    Steps:
        1. Run an external detector to obtain language data.
        2. Group words by their shared concept across languages.
        3. For each concept, compute pairwise comparability scores between words.
        4. Store scores symmetrically for both (wordA, wordB) and (wordB, wordA).
        5. Compute each word’s average comparability score.

    Args:
        input_file (str): Path to the input file containing the raw language data.
        output_file (str): Path to the file where detector results may be written.

    Returns:
        dict[str, float]:
            A mapping from word (IPA form) to its average comparability score,
            where scores range from 0.0 (no similarity) to higher values indicating
            greater similarity to words of the same concept in other languages.

    Information:
        - Requires the following functions: 
            `run_detector` function 
             `align_and_score_by_concept` function 
        - The comparability score is symmetric: score(wordA, wordB) = score(wordB, wordA).
    """
    language_data, _, _ = run_detector(input_file=input_file, output_file=output_file, obtain_output="no")

    # Gather words grouped by their conceptual meaning
    concept_to_word_lang = defaultdict(list)
    for language, items in language_data.items():
        for word, ipa, pos, concept, label in items:
            concept_to_word_lang[concept].append((ipa, language))

    # Score each concept's word pairs
    word_pair_to_comparability_score = {}
    for concept, word_lang_list in concept_to_word_lang.items():
        scored = align_and_score_by_concept(word_lang_list)
        for (wA, wB), score in scored.items():
            word_pair_to_comparability_score[(wA, wB)] = score
            word_pair_to_comparability_score[(wB, wA)] = score  # symmetry

    # Compute average comparability score per word
    word_to_comparability = {}
    for concept, word_lang_list in concept_to_word_lang.items():
        words = [w for (w, _) in word_lang_list]
        for w1 in words:
            scores = [
                word_pair_to_comparability_score.get((w1, w2), 0.0)
                for w2 in words if w2 != w1
            ]
            word_to_comparability[w1] = sum(scores) / len(scores) if scores else 0.0

    return word_to_comparability


def add_crosslinguistic(input_file, output_file):
    """
    Generate a cross-linguistic lexical dataset with predicted loanword status and comparability scores.

    This function runs a loanword detection process on the given input lexical data to:
    - Predict the probability of cognate status for each word (using a detector model).
    - Compute a comparability score for each word (how cross-linguistically comparable it is).
    - Combine these metrics into a weighted composite score.
    - Apply a dynamic threshold to produce a final scaled binary prediction.

    The results are written to a tab-delimited output file with the following columns:
        word                — The lexical item.
        ipa                 — The word's phonetic transcription in IPA.
        pos                 — Part of speech.
        concept             — The semantic concept the word represents.
        language            — The language the word belongs to.
        predicted_proba     — Basic autonomous Model-predicted probability of loanword status (0–1).
        comparability_score — Adjusted comparability score (0–1, higher means more comparable).
        composite_score     — Weighted combination of predicted probability and comparability score.
        basic_prediction    — Raw binary prediction from the basic autonomous  model.
        scaled_prediction   — Final scaled binary prediction after threshold adjustment.
        true_label          — The gold-standard label for cognate status.

    Parameters
    ----------
    input_file : str
        Path to the input file containing lexical data.
    output_file : str
        Path to the output file where results will be saved.

    Notes
    -----
    - The composite score is computed with weights w1=0.75 (predicted probability) 
      and w2=0.25 (comparability score).
    - The decision threshold is dynamically adjusted based on the difference between
      comparability score and predicted probability.
    - Requires the `run_detector` and `word_comparability` helper functions.

    """
    language_data, predicted_status_map, predicted_prob = run_detector(
        input_file=input_file,
        output_file=output_file,
        obtain_output="no"
    )
    word_to_comparability = word_comparability(
        input_file=input_file,
        output_file=output_file
    )

    w1, w2 = 0.75, 0.25
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "word", "ipa", "pos", "concept", "language", "predicted_proba",
            "comparability_score", "composite_score", "basic_prediction", 
            "scaled_prediction", "true_label"
        ])
        for language, items in language_data.items():
            for word, ipa, pos, concept, true_label in items:
                basic_prediction = predicted_status_map.get(ipa, 0)
                predicted_proba = predicted_prob.get(ipa, 0)
                comparability_score = round(1 - word_to_comparability.get(word, 0.0), 2)
                composite_score = round((w1 * predicted_proba + w2 * comparability_score) / (w1 + w2), 2)
                threshold = 0.117 + 0.432 * (comparability_score - predicted_proba)
                scaled_prediction = 1 if composite_score >= threshold else 0
                writer.writerow([
                    word, ipa, pos, concept, language, round(predicted_proba, 2),
                    comparability_score, composite_score, basic_prediction,
                    scaled_prediction, true_label
                ])

                
if __name__=="__main__":
    input_file = "datasets/cleaned_data.tsv"
    output_file="result_files/data_with_predictions_scaled.tsv"
    add_crosslinguistic(input_file=input_file, output_file=output_file)