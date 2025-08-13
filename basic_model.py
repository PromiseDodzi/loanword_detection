import math
import statistics
from collections import defaultdict
from tqdm import tqdm

class BorrowingDetector:
    """
    Detects potential loan words in a given list of words
    using statistical and pattern-based features such as n-gram rarity,
    character transition probabilities, and part-of-speech weighting.
    """

    def __init__(self, ngram_range=(2, 10), threshold=0.3, max_iterations=7,
                 convergence_threshold=0.01, feature_weights=None, pos_weights=None):
        """
        Initialize the BorrowingDetector with configurable parameters.

        Args:
            ngram_range (tuple): Min and max length of n-grams to analyze.
            threshold (float): Probability threshold for classifying a word as borrowed.
            max_iterations (int): Maximum refinement iterations for detection.
            convergence_threshold (float): Stopping criterion based on stability of borrowed set.
            feature_weights (dict): Optional custom weights for feature importance.
            pos_weights (dict): Optional custom weights for POS-based adjustments.
        """
        self.ngram_range = ngram_range
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.feature_weights = feature_weights if feature_weights is not None else {
            'rare_ngram_score': 0.40,
            'ngram_entropy': 0.20,
            'length_zscore': 0.11,
            'transition_entropy': 0.21,
            'rare_transition_score': 0.63,
            'avg_transition_prob': 0.17
        }

        self.pos_weights = pos_weights if pos_weights is not None else {
            'noun': 1.0,
            'adjective': 0.5,
            'verb': 0.3,
            'adverb': 0.2,
            'function': 0.05
        }

    def get_ngrams(self, word, n, pad=True):
        """
        Generate n-grams from a given word.

        Args:
            word (str): The word to process.
            n (int): Length of the n-grams.
            pad (bool): Whether to pad word boundaries with '#' markers.

        Returns:
            list[str]: List of n-grams.
        """
        if pad:
            word = '#' + word + '#'
        return [word[i:i + n] for i in range(len(word) - n + 1)]

    def calculate_entropy(self, frequencies):
        """
        Calculate the Shannon entropy of a set of frequencies.

        Args:
            frequencies (list[float]): Frequency counts.

        Returns:
            float: Entropy value.
        """
        total = sum(frequencies)
        if total == 0:
            return 0
        return -sum((f / total) * math.log2(f / total) for f in frequencies if f > 0)

    def extract_features(self, words):
        """
        Extract statistical features for each word to aid borrowing detection.

        Features include:
        - Rare n-gram score
        - N-gram entropy
        - Length z-score
        - Transition entropy
        - Rare transition score
        - Average transition probability

        Args:
            words (list[str]): Words to analyze.

        Returns:
            tuple:
                list[dict]: Feature dictionaries for each word.
                float: Average word length in the set.
        """
        if not words:
            return [], 0

        word_lengths = [len(word) for word in words]
        avg_length = statistics.mean(word_lengths) if word_lengths else 0
        length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0

        ngram_counts = defaultdict(lambda: 1)
        transition_counts = defaultdict(lambda: defaultdict(lambda: 1))
        total_transitions = 0

        for word in words:
            word_lower = word.lower()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for gram in self.get_ngrams(word_lower, n):
                    ngram_counts[gram] += 1
            for i in range(len(word_lower) - 1):
                current_char = word_lower[i]
                next_char = word_lower[i + 1]
                transition_counts[current_char][next_char] += 1
                total_transitions += 1

        total_ngrams = sum(ngram_counts.values())

        transition_probs = defaultdict(dict)
        for char in transition_counts:
            char_total = sum(transition_counts[char].values())
            for next_char in transition_counts[char]:
                transition_probs[char][next_char] = (
                    transition_counts[char][next_char] / char_total
                )

        word_features = []

        for word in words:
            features = {}
            word_len = len(word)
            word_lower = word.lower()

            rare_ngram_score = 0
            ngram_freqs = []

            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                grams = self.get_ngrams(word_lower, n)
                for gram in grams:
                    freq = ngram_counts[gram] / total_ngrams
                    ngram_freqs.append(freq)
                    if freq < 0.005:
                        rare_ngram_score += (0.005 - freq) * 200
                    elif freq < 0.02:
                        rare_ngram_score += (0.02 - freq) * 50

            features['rare_ngram_score'] = rare_ngram_score / (len(ngram_freqs) + 1e-10)
            features['ngram_entropy'] = self.calculate_entropy(
                [f * total_ngrams for f in ngram_freqs]
            )

            length_zscore = (
                (word_len - avg_length) / length_std if length_std > 0 else 0
            )
            features['length_zscore'] = length_zscore

            transition_scores = []
            rare_transition_score = 0
            for i in range(len(word_lower) - 1):
                current_char = word_lower[i]
                next_char = word_lower[i + 1]
                if current_char in transition_probs and next_char in transition_probs[current_char]:
                    prob = transition_probs[current_char][next_char]
                    transition_scores.append(prob)
                    if prob < 0.01:
                        rare_transition_score += (0.01 - prob) * 100
                    elif prob < 0.05:
                        rare_transition_score += (0.05 - prob) * 20

            features['transition_entropy'] = (
                self.calculate_entropy([p * total_transitions for p in transition_scores])
                if transition_scores else 0
            )
            features['rare_transition_score'] = rare_transition_score / (len(transition_scores) + 1e-10)
            features['avg_transition_prob'] = (
                sum(transition_scores) / len(transition_scores) if transition_scores else 0
            )

            word_features.append(features)

        return word_features, avg_length

    def compute_borrowing_probabilities(self, words, features_list, avg_length, word_pos_dict):
        """
        Compute the probability of each word being borrowed based on extracted features.

        Args:
            words (list[str]): Words to analyze.
            features_list (list[dict]): Extracted feature sets for each word.
            avg_length (float): Average word length in the corpus.
            word_pos_dict (dict): Mapping of words to part-of-speech tags.

        Returns:
            dict: Mapping of words to borrowing probabilities (0–1).
        """
        if not words or not features_list:
            return {}

        normalized_features = {}
        for feature_name in self.feature_weights.keys():
            values = [f[feature_name] for f in features_list]
            if not values:
                normalized_features[feature_name] = [0.0] * len(words)
                continue
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                normalized = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized = [0.0] * len(values)
            normalized_features[feature_name] = normalized

        probabilities = {}

        for i, word in enumerate(words):
            base_score = 0
            for feature_name, weight in self.feature_weights.items():
                base_score += weight * normalized_features[feature_name][i]

            length_zscore = features_list[i]['length_zscore']
            length_factor = 1 + (0.5 * (1 / (1 + math.exp(-3 * (abs(length_zscore) - 1.5)))) - 0.5)

            pos = word_pos_dict.get(word, 'noun')
            pos_weight = self.pos_weights.get(pos, 0.7)

            adjusted_score = base_score * length_factor * pos_weight

            prob_borrowed = 1 / (1 + math.exp(-10 * (adjusted_score - 0.5)))

            if features_list[i]['rare_ngram_score'] > 0.8:
                prob_borrowed = min(prob_borrowed * 1.3, 1.0)
            if features_list[i]['rare_transition_score'] > 0.7:
                prob_borrowed = min(prob_borrowed * 1.3, 1.0)
            if features_list[i]['avg_transition_prob'] < 0.1:
                prob_borrowed = min(prob_borrowed * 1.2, 1.0)

            probabilities[word] = max(0.0, min(1.0, prob_borrowed))

        return probabilities

    def detect_borrowed_words(self, words, word_pos_dict=None):
        """
        Detect borrowed words from a list of words using iterative refinement.

        Args:
            words (list[str]): Words to analyze.
            word_pos_dict (dict, optional): Mapping of words to POS tags. Defaults to 'noun'.

        Returns:
            tuple:
                list[str]: Borrowed words sorted by probability.
                dict: Mapping of words to borrowing probabilities.
        """
        if not words:
            return [], {}

        if word_pos_dict is None:
            word_pos_dict = {word: 'noun' for word in words}

        all_probabilities = {}
        previous_borrowed = set()
        current_words = words.copy()
        full_feature_cache = {}
        EPSILON = 1e-10

        for iteration in tqdm(range(self.max_iterations), desc="Detecting borrowed words"):
            features_list, avg_length = self.extract_features(current_words)
            for word, features in zip(current_words, features_list):
                full_feature_cache[word] = features

            probabilities = self.compute_borrowing_probabilities(
                current_words, features_list, avg_length, word_pos_dict
            )

            current_borrowed = {word for word, prob in probabilities.items() if prob >= self.threshold}
            current_native = set(current_words) - current_borrowed

            if iteration > 0 and current_native:
                native_patterns = self._build_pattern_database(current_native)
                borrowed_patterns = self._build_pattern_database(current_borrowed)

                for word in current_words:
                    features = full_feature_cache[word]
                    patterns = self._extract_word_patterns(word)
                    native_score = sum(
                        native_patterns[p] / (native_patterns[p] + borrowed_patterns.get(p, 0.1))
                        for p in patterns if p in native_patterns
                    )
                    pattern_score = native_score / (len(patterns) + EPSILON)

                    is_weird = (
                        features['rare_ngram_score'] > 0.8 or
                        features['rare_transition_score'] > 0.8 or
                        abs(features['length_zscore']) > 2.5
                    )

                    if not is_weird and pattern_score > 0.7:
                        probabilities[word] *= 0.5
                    elif is_weird and pattern_score < 0.3:
                        probabilities[word] = min(probabilities[word] * 1.5, 1.0)

            for word, prob in probabilities.items():
                if word in all_probabilities:
                    prev_avg = all_probabilities[word]
                    all_probabilities[word] = (prev_avg * iteration + prob) / (iteration + 1)
                else:
                    all_probabilities[word] = prob

            if previous_borrowed:
                diff = previous_borrowed.symmetric_difference(current_borrowed)
                if len(diff) < self.convergence_threshold * len(previous_borrowed):
                    break

            current_words = [word for word in current_words if word not in current_borrowed]
            previous_borrowed = current_borrowed.copy()

        borrowed_words = [
            word for word, prob in sorted(all_probabilities.items(), key=lambda x: -x[1])
            if prob >= self.threshold
        ]
        return borrowed_words, all_probabilities

    def _build_pattern_database(self, words):
        """
        Build a frequency database of character-based word patterns.

        Patterns include:
        - Prefixes and suffixes (2–3 chars)
        - 3-character internal segments

        Args:
            words (set[str]): Words to process.

        Returns:
            dict: Mapping of patterns to their counts.
        """
        patterns = defaultdict(int)
        for word in words:
            word_lower = word.lower()
            if len(word_lower) >= 2:
                patterns[f"pre_{word_lower[:2]}"] += 1
                patterns[f"suf_{word_lower[-2:]}"] += 1
            if len(word_lower) >= 3:
                patterns[f"pre_{word_lower[:3]}"] += 1
                patterns[f"suf_{word_lower[-3:]}"] += 1
                for i in range(len(word_lower) - 2):
                    patterns[f"seg_{word_lower[i:i + 3]}"] += 1
        return patterns

    def _extract_word_patterns(self, word):
        """
        Extract character-based patterns from a single word.

        Returns:
            set[str]: Patterns such as 2–3 char prefixes, suffixes, and internal segments.
        """
        patterns = set()
        word_lower = word.lower()
        if len(word_lower) >= 2:
            patterns.add(f"pre_{word_lower[:2]}")
            patterns.add(f"suf_{word_lower[-2:]}")
        if len(word_lower) >= 3:
            patterns.add(f"pre_{word_lower[:3]}")
            patterns.add(f"suf_{word_lower[-3:]}")
            for i in range(len(word_lower) - 2):
                patterns.add(f"seg_{word_lower[i:i + 3]}")
        return patterns
