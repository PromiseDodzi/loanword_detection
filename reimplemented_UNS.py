import re
from collections import defaultdict
import math
import csv
from evaluator import evaluate_predictions,evaluate_predictions_by_language

#Original Paper: Used consonant-based "pseudo-syllables" (e.g., |pu|ra| + |ttha| in Malayalam)
#Current Code: Splits words into syllables (e.g., "happiness" → ["hap", "pi", "ness"])
#Purpose: Measures diversity of syllables following the initial "stem" (first stem_length syllables) to initialize nativeness scores.
#remember, in original paper, stem = first 2 syllables, and diversity is counted only after second syllable, for simplicity. We do same here

def syllable_split(word):
    """
    Split a given word into syllable-like segments based on vowel boundaries.
    """
    vowels = "aeiouyAEIOUY"
    syllables = []
    current_syllable = ""

    for i, char in enumerate(word):
        current_syllable += char
        if char.lower() in vowels:
            if i < len(word) - 1 and word[i + 1].lower() not in vowels:
                syllables.append(current_syllable)
                current_syllable = ""
    if current_syllable:
        syllables.append(current_syllable)

    if not syllables:
        return [word]

    return syllables


def initialize(self, words):
    """Initialize nativeness scores based on *unique following syllables* after the stem"""

    wn = dict()
    WD = set()
    stem_to_following_sequences = defaultdict(list)
    word_to_stem = {}

    # Gather following syllables for each stem
    for word in words:
        syllables = syllable_split(word)
        if len(syllables) > self.stem_length:
            stem = ''.join(syllables[:self.stem_length])
            following = syllables[self.stem_length:]
            stem_to_following_sequences[stem].append(following)
            word_to_stem[word] = stem

    # For each stem, get total number of unique following syllables
    stem_to_unique_syllables = dict()
    for stem, followings in stem_to_following_sequences.items():
        unique_sylls = set()
        for seq in followings:
            unique_sylls.update(seq)  
        stem_to_unique_syllables[stem] = unique_sylls

    #Initialize word scores based on diversity 
    for word in words:
        stem = word_to_stem.get(word, '')
        diversity = len(stem_to_unique_syllables.get(stem, set()))

        # Score from Eq. 2
        self.wn[word] = min(0.99, diversity / 10)

        if diversity > 5:
            self.WD.add(word)


class UNS:
    def __init__(self, stem_length=2, max_diversity=10, diverse_threshold=5,
                 lambda_param=1.0, ngram_size=1, max_iter=100, convergence_threshold=0.001):
        """
        Initialize UNS parameters:
        - stem_length: number of initial syllables to consider as "stem"
        - max_diversity: diversity threshold for initialization (μ in paper)
        - diverse_threshold: threshold for highly diverse words (ζ in paper)
        - lambda_param: relative weighting parameter (λ in paper)
        - ngram_size: size of character n-grams to use
        - max_iter: maximum number of iterations
        - convergence_threshold: threshold for convergence
        """
        self.stem_length = stem_length
        self.max_diversity = max_diversity
        self.diverse_threshold = diverse_threshold
        self.lambda_param = lambda_param
        self.ngram_size = ngram_size
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold

        # Placeholders for model parameters
        self.N = defaultdict(float)  # Native n-gram distribution
        self.L = defaultdict(float)  # Loanword n-gram distribution
        self.wn = dict()  # Nativeness scores
        self.WD = set()  # Highly diverse words

    def get_ngrams(self, word):
        """Generate character n-grams from a word"""
        return [word[i:i+self.ngram_size] for i in range(len(word)-self.ngram_size+1)]

    def initialize(self, words):
      """Initialize nativeness scores based on *unique following syllables* after the stem"""

      self.wn = dict()
      self.WD = set()

      stem_to_following_sequences = defaultdict(list)
      word_to_stem = {}

      for word in words:
          syllables = syllable_split(word)
          if len(syllables) > self.stem_length:
              stem = ''.join(syllables[:self.stem_length])
              following = syllables[self.stem_length:]
              stem_to_following_sequences[stem].append(following)
              word_to_stem[word] = stem

      stem_to_unique_syllables = dict()
      for stem, followings in stem_to_following_sequences.items():
          unique_sylls = set()
          for seq in followings:
              unique_sylls.update(seq)
          stem_to_unique_syllables[stem] = unique_sylls

      total_diversity = 0
      max_div = 0

      for word in words:
          stem = word_to_stem.get(word, '')
          diversity = len(stem_to_unique_syllables.get(stem, set()))
          total_diversity += diversity
          max_div = max(max_div, diversity)

      threshold = total_diversity / len(words) if words else 0  #using average diversity as threshold

      for word in words:
          stem = word_to_stem.get(word, '')
          diversity = len(stem_to_unique_syllables.get(stem, set()))
          div_score = diversity / max_div if max_div > 0 else 0  #using the maximum diversity as denominator instead of adhoc 10
          self.wn[word] = min(0.99, div_score)

          if diversity > threshold:
              self.WD.add(word)

    def update_native_distribution(self, words):
        """Update native n-gram distribution (Eq. 13)"""
        new_N = defaultdict(float)

        # First collect unnormalized updates
        for word in words:
            ngrams = self.get_ngrams(word)
            for c in ngrams:
                # Eq. 13 numerator
                numerator = self.wn[word] ** 2
                denominator = (self.wn[word] ** 2 +
                              (1 - self.wn[word]) ** 2 * self.L.get(c, 1e-10) / max(self.N.get(c, 1e-10), 1e-10))
                new_N[c] += numerator / denominator

        # Normalize (Eq. 14)
        total = sum(new_N.values())
        if total > 0:
            for c in new_N:
                new_N[c] /= total
        else:
            # If all zeros, set uniform distribution
            vocab_size = len(new_N)
            for c in new_N:
                new_N[c] = 1.0 / vocab_size

        self.N = new_N

    def update_loanword_distribution(self, words):
        """Update loanword n-gram distribution (Eq. 15)"""
        new_L = defaultdict(float)

        # First collect unnormalized updates
        for word in words:
            ngrams = self.get_ngrams(word)
            for c in ngrams:
                # Eq. 15 numerator
                numerator = (1 - self.wn[word]) ** 2
                denominator = ((1 - self.wn[word]) ** 2 +
                              self.wn[word] ** 2 * self.N.get(c, 1e-10) / max(self.L.get(c, 1e-10), 1e-10))
                new_L[c] += numerator / denominator

        # Normalize (Eq. 14 equivalent)
        total = sum(new_L.values())
        if total > 0:
            for c in new_L:
                new_L[c] /= total
        else:
            # If all zeros, set uniform distribution
            vocab_size = len(new_L)
            for c in new_L:
                new_L[c] = 1.0 / vocab_size

        self.L = new_L

    def update_nativeness_scores(self, words):
        """Update nativeness scores (Eq. 9)"""
        new_wn = {}
        changed = 0.0

        for word in words:
            ngrams = self.get_ngrams(word)

            # Compute numerator terms
            inertia_term = (self.lambda_param * (1 if word in self.WD else 0)) / (1 - self.wn[word] + 1e-10)

            sum_N = 0.0
            sum_N_plus_L = 0.0

            for c in ngrams:
                N_c = self.N.get(c, 1e-10)
                L_c = self.L.get(c, 1e-10)
                denominator = ((1 - self.wn[word]) ** 2 * N_c +
                               self.wn[word] ** 2 * L_c + 1e-10)

                sum_N += N_c / denominator
                sum_N_plus_L += (N_c + L_c) / denominator

            # Compute new score (Eq. 9)
            numerator = inertia_term + sum_N
            denominator = sum_N_plus_L
            new_score = numerator / (denominator + 1e-10)

            # Clip to [0, 1]
            new_score = max(0.0, min(1.0, new_score))

            new_wn[word] = new_score
            changed += abs(new_score - self.wn[word])

        # Update all scores
        self.wn.update(new_wn)

        # Return average change for convergence check
        return changed / len(words)

    def fit(self, words):
        """Run the full UNS algorithm (Algorithm 1 in paper)"""
        print("Initializing scores based on syllable diversity...")
        self.initialize(words)

        # Initial uniform distributions for N and L
        all_ngrams = set()
        for word in words:
            all_ngrams.update(self.get_ngrams(word))

        vocab_size = len(all_ngrams)
        for c in all_ngrams:
            self.N[c] = 1.0 / vocab_size
            self.L[c] = 1.0 / vocab_size

        # Iterative refinement
        print("Starting iterative refinement...")
        for iteration in range(self.max_iter):
            # Update distributions
            print(f"Iteration {iteration + 1}: Updating distributions...")
            self.update_native_distribution(words)
            self.update_loanword_distribution(words)

            # Update scores
            print(f"Iteration {iteration + 1}: Updating scores...")
            avg_change = self.update_nativeness_scores(words)

            print(f"Iteration {iteration + 1}: Average score change = {avg_change:.4f}")

            # Check convergence
            if avg_change < self.convergence_threshold:
                print(f"Converged after {iteration + 1} iterations.")
                break

    def predict(self, word):
        """Get nativeness score for a word"""
        return self.wn.get(word, 0.5)  # Default to neutral if word not seen
    
def run_UNS(input_link, output_link):
    """Run the UNS model on data for loanword detection"""
    language_data = defaultdict(list) 

    with open(input_link, "r", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            word = row["word"]
            ipa=row["ipa"]
            pos = row["pos"]
            concept=row["concept"]
            label = int(row["label"])
            language = row["language"]
            language_data[language].append((word, ipa, pos, concept, label))

 
    predicted_prob={}
    predicted_status_map = {}

    for language, items in language_data.items():
        words = [w for _,w, _, _, _ in items]
        word_pos_dict = {w: pos for _, w, pos, _, _ in items}

        print(f"Processing language: {language}")
        # Run borrowing detector
        uns = UNS(stem_length=2, max_diversity=5, diverse_threshold=3,
                lambda_param=1.0, ngram_size=2, max_iter=20)

        uns.fit(words)


        sorted_words = sorted(words, key=lambda x: uns.wn[x], reverse=True)
        for word in sorted_words:
            predicted_status_map[word] = 1 if uns.wn[word] > 0.5 else 0
            predicted_prob[word] = uns.wn[word]


    with open(output_link, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["word", "ipa", "pos", "concept", "language",  "true_label", "predicted_proba", "uns_prediction"])
        for language, items in language_data.items():
            for word, ipa, pos, concept, true_label in items:
                uns_prediction = predicted_status_map.get(ipa, 0)
                predicted_proba = predicted_prob.get(ipa, 0)
                writer.writerow([word, ipa, pos, concept, language, true_label, predicted_proba, uns_prediction])

if __name__=="__main__":
    input_link="datasets/cleaned_data.tsv"
    output_link="result_files/data_with_predictions_uns.tsv"
    run_UNS(input_link=input_link, output_link=output_link)

    #evaluate UNS performance
    print("*"*40)
    print("EVALUATING UNS PERFORMANCE")
    predictions="uns_prediction"
    evaluate_predictions(tsv_file_path=output_link, predictions=predictions)
    evaluate_predictions_by_language(tsv_file_path=output_link, predictions=predictions)
