import math
import statistics
from collections import defaultdict
from tqdm import tqdm

#---------WITHOUT N-GRAM FEATURES
class NoNgram:
    def __init__(self, threshold=0.3, max_iterations=7, convergence_threshold=0.01,
                 feature_weights=None, pos_weights=None):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

   
        self.feature_weights = feature_weights if feature_weights is not None else {
            'length_zscore': 0.1833,      
            'transition_entropy': 0.35,    
            'rare_transition_score': 1.05, 
            'avg_transition_prob': 0.2833  
        }

        self.pos_weights = pos_weights if pos_weights is not None else {
            'noun': 1.0,
            'adjective': 0.5,
            'verb': 0.3,
            'adverb': 0.2,
            'function': 0.05
        }

    def calculate_entropy(self, frequencies):
        total = sum(frequencies)
        if total == 0:
            return 0
        return -sum((f / total) * math.log2(f / total) for f in frequencies if f > 0)

    def extract_features(self, words):
        if not words:
            return [], 0

        # Calculate basic statistics
        word_lengths = [len(word) for word in words]
        avg_length = statistics.mean(word_lengths) if word_lengths else 0
        length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0

        # Initialize counts with smoothing
        transition_counts = defaultdict(lambda: defaultdict(lambda: 1))
        total_transitions = 0

        # collect statistics
        for word in words:
            word_lower = word.lower()

            # Character transition counts
            for i in range(len(word_lower)-1):
                current_char = word_lower[i]
                next_char = word_lower[i+1]
                transition_counts[current_char][next_char] += 1
                total_transitions += 1

        # Calculate transition probabilities
        transition_probs = defaultdict(dict)
        for char in transition_counts:
            char_total = sum(transition_counts[char].values())
            for next_char in transition_counts[char]:
                transition_probs[char][next_char] = transition_counts[char][next_char] / char_total

        word_features = []

        for word in words:
            features = {}
            word_len = len(word)
            word_lower = word.lower()

            # Length features
            if length_std > 0:
                length_zscore = (word_len - avg_length) / length_std
            else:
                length_zscore = 0
            features['length_zscore'] = length_zscore

            # Transition probability features
            transition_scores = []
            rare_transition_score = 0
            for i in range(len(word_lower)-1):
                current_char = word_lower[i]
                next_char = word_lower[i+1]
                if current_char in transition_probs and next_char in transition_probs[current_char]:
                    prob = transition_probs[current_char][next_char]
                    transition_scores.append(prob)
                    if prob < 0.01:
                        rare_transition_score += (0.01 - prob) * 100
                    elif prob < 0.05:
                        rare_transition_score += (0.05 - prob) * 20

            features['transition_entropy'] = self.calculate_entropy([p * total_transitions for p in transition_scores]) if transition_scores else 0
            features['rare_transition_score'] = rare_transition_score / (len(transition_scores) + 1e-10)
            features['avg_transition_prob'] = sum(transition_scores)/len(transition_scores) if transition_scores else 0

            word_features.append(features)

        return word_features, avg_length

    def compute_borrowing_probabilities(self, words, features_list, avg_length, word_pos_dict):
        if not words or not features_list:
            return {}

        # Normalize features
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
            # Calculate base score
            base_score = 0
            for feature_name, weight in self.feature_weights.items():
                base_score += weight * normalized_features[feature_name][i]

            # Apply length adjustment
            length_zscore = features_list[i]['length_zscore']
            length_factor = 1 + (0.5 * (1 / (1 + math.exp(-3 * (abs(length_zscore) - 1.5)))) - 0.5)

            # Apply POS adjustment
            pos = word_pos_dict.get(word, 'noun')
            pos_weight = self.pos_weights.get(pos, 0.7)

            # Combine factors
            adjusted_score = base_score * length_factor * pos_weight

            # Sigmoid function to get probability
            prob_borrowed = 1 / (1 + math.exp(-10 * (adjusted_score - 0.5)))

            # Post-processing rules
            if features_list[i]['rare_transition_score'] > 0.7:
                prob_borrowed = min(prob_borrowed * 1.3, 1.0)
            if features_list[i]['avg_transition_prob'] < 0.1:
                prob_borrowed = min(prob_borrowed * 1.2, 1.0)

            probabilities[word] = max(0.0, min(1.0, prob_borrowed))

        return probabilities

    def detect_borrowed_words(self, words, word_pos_dict=None):
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
            # Feature extraction
            features_list, avg_length = self.extract_features(current_words)
            for word, features in zip(current_words, features_list):
                full_feature_cache[word] = features

            # Compute borrowing probabilities
            probabilities = self.compute_borrowing_probabilities(
                current_words, features_list, avg_length, word_pos_dict
            )

            current_borrowed = {word for word, prob in probabilities.items() if prob >= self.threshold}
            current_native = set(current_words) - current_borrowed

            # Pattern-based refinement 
            if iteration > 0 and current_native:
                native_patterns = self._build_pattern_database(current_native)
                borrowed_patterns = self._build_pattern_database(current_borrowed)

                for word in current_words:
                    features = full_feature_cache[word]
                    original_prob = probabilities[word]

                    # Extract patterns and calculate native-likeness score
                    patterns = self._extract_word_patterns(word)
                    native_score = sum(
                        native_patterns[p] / (native_patterns[p] + borrowed_patterns.get(p, 0.1))
                        for p in patterns if p in native_patterns
                    )
                    pattern_score = native_score / (len(patterns) + EPSILON)

                    # Check "weirdness" based on remaining features
                    is_weird = (
                        features['rare_transition_score'] > 0.8 or
                        abs(features['length_zscore']) > 2.5
                    )

                    # Adjust probabilities based on pattern score and weirdness
                    if not is_weird and pattern_score > 0.7:
                        probabilities[word] *= 0.5
                    elif is_weird and pattern_score < 0.3:
                        probabilities[word] = min(probabilities[word] * 1.5, 1.0)

            # Update running average of probabilities
            for word, prob in probabilities.items():
                if word in all_probabilities:
                    prev_avg = all_probabilities[word]
                    all_probabilities[word] = (prev_avg * iteration + prob) / (iteration + 1)
                else:
                    all_probabilities[word] = prob

            # Convergence check
            if previous_borrowed:
                diff = previous_borrowed.symmetric_difference(current_borrowed)
                if len(diff) < self.convergence_threshold * len(previous_borrowed):
                    break

            # Remove current borrowed words
            current_words = [word for word in current_words if word not in current_borrowed]
            previous_borrowed = current_borrowed.copy()

        # Final borrowed words output
        borrowed_words = [
            word for word, prob in sorted(all_probabilities.items(), key=lambda x: -x[1])
            if prob >= self.threshold
        ]
        return borrowed_words, all_probabilities

    def _build_pattern_database(self, words):
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
    

    #---------WITHOUT TRANSITION PROBABILITIES
    class NoTransitions:
    def __init__(self, ngram_range=(2, 10), threshold=0.3, max_iterations=7, convergence_threshold=0.01,
                 feature_weights=None, pos_weights=None):
        self.ngram_range = ngram_range
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.feature_weights = feature_weights if feature_weights is not None else {
            'rare_ngram_score': 0.58,  
            'ngram_entropy': 0.29,      
            'length_zscore': 0.16,      
            # Removed transition features: transition_entropy, rare_transition_score, avg_transition_prob
        }

        self.pos_weights = pos_weights if pos_weights is not None else {
            'noun': 1.0,
            'adjective': 0.5,
            'verb': 0.3,
            'adverb': 0.2,
            'function': 0.05
        }

    def get_ngrams(self, word, n, pad=True):
        if pad:
            word = '#' + word + '#'
        return [word[i:i + n] for i in range(len(word) - n + 1)]

    def calculate_entropy(self, frequencies):
        total = sum(frequencies)
        if total == 0:
            return 0
        return -sum((f / total) * math.log2(f / total) for f in frequencies if f > 0)

    def extract_features(self, words):
        if not words:
            return [], 0

        # Calculate basic statistics
        word_lengths = [len(word) for word in words]
        avg_length = statistics.mean(word_lengths) if word_lengths else 0
        length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0

        # Initialize counts with smoothing
        ngram_counts = defaultdict(lambda: 1)
        total_ngrams = 0

        # collect statistics
        for word in words:
            word_lower = word.lower()

            # N-gram counts
            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
                for gram in self.get_ngrams(word_lower, n):
                    ngram_counts[gram] += 1
                    total_ngrams += 1

        word_features = []

        for word in words:
            features = {}
            word_len = len(word)
            word_lower = word.lower()

            # N-gram rarity features
            rare_ngram_score = 0
            ngram_freqs = []

            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
                grams = self.get_ngrams(word_lower, n)
                for gram in grams:
                    freq = ngram_counts[gram] / total_ngrams
                    ngram_freqs.append(freq)
                    if freq < 0.005:
                        rare_ngram_score += (0.005 - freq) * 200
                    elif freq < 0.02:
                        rare_ngram_score += (0.02 - freq) * 50

            features['rare_ngram_score'] = rare_ngram_score / (len(ngram_freqs) + 1e-10)
            features['ngram_entropy'] = self.calculate_entropy([f * total_ngrams for f in ngram_freqs])

            # Length features
            if length_std > 0:
                length_zscore = (word_len - avg_length) / length_std
            else:
                length_zscore = 0
            features['length_zscore'] = length_zscore

            word_features.append(features)

        return word_features, avg_length

    def compute_borrowing_probabilities(self, words, features_list, avg_length, word_pos_dict):
        if not words or not features_list:
            return {}

        # Normalize features
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
            # Calculate base score
            base_score = 0
            for feature_name, weight in self.feature_weights.items():
                base_score += weight * normalized_features[feature_name][i]

            # Apply length adjustment
            length_zscore = features_list[i]['length_zscore']
            length_factor = 1 + (0.5 * (1 / (1 + math.exp(-3 * (abs(length_zscore) - 1.5)))) - 0.5)

            # Apply POS adjustment
            pos = word_pos_dict.get(word, 'noun')
            pos_weight = self.pos_weights.get(pos, 0.7)

            # Combine factors
            adjusted_score = base_score * length_factor * pos_weight

            # Sigmoid function to get probability
            prob_borrowed = 1 / (1 + math.exp(-10 * (adjusted_score - 0.5)))

            # Post-processing rules (simplified)
            if features_list[i]['rare_ngram_score'] > 0.8:
                prob_borrowed = min(prob_borrowed * 1.3, 1.0)

            probabilities[word] = max(0.0, min(1.0, prob_borrowed))

        return probabilities

    def detect_borrowed_words(self, words, word_pos_dict=None):
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
            # Feature extraction
            features_list, avg_length = self.extract_features(current_words)
            for word, features in zip(current_words, features_list):
                full_feature_cache[word] = features

            # Compute borrowing probabilities
            probabilities = self.compute_borrowing_probabilities(
                current_words, features_list, avg_length, word_pos_dict
            )

            current_borrowed = {word for word, prob in probabilities.items() if prob >= self.threshold}
            current_native = set(current_words) - current_borrowed

            # Pattern-based refinement (skip on first iteration)
            if iteration > 0 and current_native:
                native_patterns = self._build_pattern_database(current_native)
                borrowed_patterns = self._build_pattern_database(current_borrowed)

                for word in current_words:
                    features = full_feature_cache[word]
                    original_prob = probabilities[word]

                    # Extract patterns and calculate native-likeness score
                    patterns = self._extract_word_patterns(word)
                    native_score = sum(
                        native_patterns[p] / (native_patterns[p] + borrowed_patterns.get(p, 0.1))
                        for p in patterns if p in native_patterns
                    )
                    pattern_score = native_score / (len(patterns) + EPSILON)

                    # Check "weirdness" based on remaining features
                    is_weird = (
                        features['rare_ngram_score'] > 0.8 or
                        abs(features['length_zscore']) > 2.5
                    )

                    # Adjust probabilities based on pattern score and weirdness
                    if not is_weird and pattern_score > 0.7:
                        probabilities[word] *= 0.5
                    elif is_weird and pattern_score < 0.3:
                        probabilities[word] = min(probabilities[word] * 1.5, 1.0)

            # Update running average of probabilities
            for word, prob in probabilities.items():
                if word in all_probabilities:
                    prev_avg = all_probabilities[word]
                    all_probabilities[word] = (prev_avg * iteration + prob) / (iteration + 1)
                else:
                    all_probabilities[word] = prob

            # Convergence check
            if previous_borrowed:
                diff = previous_borrowed.symmetric_difference(current_borrowed)
                if len(diff) < self.convergence_threshold * len(previous_borrowed):
                    break

            # Remove current borrowed words
            current_words = [word for word in current_words if word not in current_borrowed]
            previous_borrowed = current_borrowed.copy()

        # Final borrowed words output
        borrowed_words = [
            word for word, prob in sorted(all_probabilities.items(), key=lambda x: -x[1])
            if prob >= self.threshold
        ]
        return borrowed_words, all_probabilities

    def _build_pattern_database(self, words):
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
    
#----------ADDING FEATURES-------
class AdditionalFeatures:
    def __init__(self, ngram_range=(2, 10), threshold=0.3, max_iterations=7, convergence_threshold=0.01,
                 feature_weights=None, pos_weights=None):
        self.ngram_range = ngram_range
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.vowels = {
                      'a', 'e', 'i', 'o', 'u',
                      'á', 'à', 'â', 'ä', 'ã', 'å',
                      'é', 'è', 'ê', 'ë',
                      'í', 'ì', 'î', 'ï',
                      'ó', 'ò', 'ô', 'ö', 'õ',
                      'ú', 'ù', 'û', 'ü',
                      'æ','œ', 'ø','ß',
                      'ã', 'õ',
                      'ä', 'ö', 'ü',
                  }

        self.feature_weights = feature_weights if feature_weights is not None else {
            'rare_ngram_score': 0.20,
            'ngram_entropy': 0.10,
            'rare_cv_score': 0.15,
            'cv_entropy': 0.08,
            'char_diversity': 0.05,
            'char_anomaly': 0.08,
            'vowel_ratio': 0.04,
            'vowel_anomaly': 0.04,
            'consonant_cluster_score': 0.15,
            'cluster_count': 0.04,
            'length_zscore': 0.05,
            'norm_spectral_entropy': 0.08,
            'norm_spectral_irregularity': 0.08,
            'transition_entropy': 0.10,
            'rare_transition_score': 0.30,
            'avg_transition_prob': 0.08
        }

        self.pos_weights = pos_weights if pos_weights is not None else {
            'noun': 1.0,
            'adjective': 0.5,
            'verb': 0.3,
            'adverb': 0.2,
            'function': 0.05
        }


    def get_ngrams(self, word, n, pad=True):
        if pad:
            word = '#' + word + '#'
        return [word[i:i + n] for i in range(len(word) - n + 1)]

    def word_to_cv(self, word):
        return ''.join('V' if c.lower() in self.vowels else 'C' for c in word)


    def compute_dft(self, signal):
        N = len(signal)
        if N == 0:
            return []
        dft_result = []
        for k in range(N):
            real_part = sum(signal[n] * math.cos(2 * math.pi * k * n / N) for n in range(N))
            imag_part = -sum(signal[n] * math.sin(2 * math.pi * k * n / N) for n in range(N))
            dft_result.append(complex(real_part, imag_part))
        return dft_result

    def calculate_entropy(self, frequencies):
        total = sum(frequencies)
        if total == 0:
            return 0
        return -sum((f / total) * math.log2(f / total) for f in frequencies if f > 0)

    def extract_features(self, words):
        if not words:
            return [], 0

        # Calculate basic statistics
        word_lengths = [len(word) for word in words]
        avg_length = statistics.mean(word_lengths) if word_lengths else 0
        length_std = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0

        # Initialize counts with smoothing
        ngram_counts = defaultdict(lambda: 1)
        cv_ngram_counts = defaultdict(lambda: 1)
        char_counts = defaultdict(lambda: 1)
        vowel_counts = defaultdict(lambda: 1)
        transition_counts = defaultdict(lambda: defaultdict(lambda: 1))
        total_transitions = 0

        # collect statistics
        for word in words:
            word_lower = word.lower()

            # Character and vowel counts
            for char in word_lower:
                char_counts[char] += 1
                if char in self.vowels:
                    vowel_counts[char] += 1

            # N-gram counts
            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
                for gram in self.get_ngrams(word_lower, n):
                    ngram_counts[gram] += 1

                cv_word = self.word_to_cv(word_lower)
                for gram in self.get_ngrams(cv_word, n):
                    cv_ngram_counts[gram] += 1

            # Character transition counts
            for i in range(len(word_lower)-1):
                current_char = word_lower[i]
                next_char = word_lower[i+1]
                transition_counts[current_char][next_char] += 1
                total_transitions += 1

        # Calculate totals with smoothing
        total_ngrams = sum(ngram_counts.values())
        total_cv_ngrams = sum(cv_ngram_counts.values())
        total_chars = sum(char_counts.values())
        total_vowels = sum(vowel_counts.values())

        # Calculate transition probabilities
        transition_probs = defaultdict(dict)
        for char in transition_counts:
            char_total = sum(transition_counts[char].values())
            for next_char in transition_counts[char]:
                transition_probs[char][next_char] = transition_counts[char][next_char] / char_total

        word_features = []

        for word in words:
            features = {}
            word_len = len(word)
            word_lower = word.lower()

            # N-gram rarity features
            rare_ngram_score = 0
            ngram_freqs = []

            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
                grams = self.get_ngrams(word_lower, n)
                for gram in grams:
                    freq = ngram_counts[gram] / total_ngrams
                    ngram_freqs.append(freq)
                    if freq < 0.005:
                        rare_ngram_score += (0.005 - freq) * 200
                    elif freq < 0.02:
                        rare_ngram_score += (0.02 - freq) * 50

            features['rare_ngram_score'] = rare_ngram_score / (len(ngram_freqs) + 1e-10)
            features['ngram_entropy'] = self.calculate_entropy([f * total_ngrams for f in ngram_freqs])

            # CV pattern features
            cv_word = self.word_to_cv(word_lower)
            rare_cv_score = 0
            cv_freqs = []

            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
                grams = self.get_ngrams(cv_word, n)
                for gram in grams:
                    freq = cv_ngram_counts[gram] / total_cv_ngrams
                    cv_freqs.append(freq)
                    if freq < 0.01:
                        rare_cv_score += (0.01 - freq) * 100

            features['rare_cv_score'] = rare_cv_score / (len(cv_freqs) + 1e-10)
            features['cv_entropy'] = self.calculate_entropy([f * total_cv_ngrams for f in cv_freqs])

            # Character distribution features
            unique_chars = len(set(word_lower))
            features['char_diversity'] = unique_chars / word_len if word_len > 0 else 0

            # Character frequency anomalies
            char_anomaly = 0
            for char in word_lower:
                freq = char_counts[char] / total_chars
                if freq < 0.005:
                    char_anomaly += (0.005 - freq) * 200
            features['char_anomaly'] = char_anomaly / word_len if word_len > 0 else 0

            # Vowel features
            vowel_pattern = ''.join(c for c in word_lower if c in {'a','e','i','o','u'})
            vowel_ratio = len(vowel_pattern) / word_len if word_len > 0 else 0
            features['vowel_ratio'] = vowel_ratio

            # Vowel frequency anomalies
            vowel_anomaly = 0
            for char in vowel_pattern:
                freq = vowel_counts[char] / total_vowels
                if freq < 0.05:
                    vowel_anomaly += (0.05 - freq) * 20
            features['vowel_anomaly'] = vowel_anomaly / len(vowel_pattern) if vowel_pattern else 0

            # Consonant cluster features
            consonant_clusters = []
            current_cluster = ""
            vowels = set('aeiou')

            for char in word_lower:
                if char not in vowels:
                    current_cluster += char
                else:
                    if len(current_cluster) > 1:
                        consonant_clusters.append(current_cluster)
                    current_cluster = ""

            if len(current_cluster) > 1:
                consonant_clusters.append(current_cluster)

            cluster_score = sum(len(cluster)**1.5 for cluster in consonant_clusters)
            features['consonant_cluster_score'] = cluster_score
            features['cluster_count'] = len(consonant_clusters)

            # Length features
            if length_std > 0:
                length_zscore = (word_len - avg_length) / length_std
            else:
                length_zscore = 0
            features['length_zscore'] = length_zscore

            # Spectral features
            signal = []
            for n in range(2, 5):
                grams = self.get_ngrams(word_lower, n)
                signal.extend(ngram_counts[gram] for gram in grams)

            if signal:
                spectrum = self.compute_dft(signal)
                magnitudes = [abs(x) for x in spectrum]
                total_power = sum(magnitudes)

                if total_power > 0:
                    # Normalized spectral entropy
                    spectral_entropy = self.calculate_entropy(magnitudes)
                    max_entropy = math.log2(len(magnitudes)) if magnitudes else 1
                    features['norm_spectral_entropy'] = spectral_entropy / max_entropy if max_entropy > 0 else 0

                    # Spectral irregularity (normalized)
                    irregularity = sum(abs(magnitudes[i] - magnitudes[i-1])
                                     for i in range(1, len(magnitudes)))
                    features['norm_spectral_irregularity'] = irregularity / total_power if total_power > 0 else 0
                else:
                    features['norm_spectral_entropy'] = 0
                    features['norm_spectral_irregularity'] = 0
            else:
                features['norm_spectral_entropy'] = 0
                features['norm_spectral_irregularity'] = 0

            # Transition probability features
            transition_scores = []
            rare_transition_score = 0
            for i in range(len(word_lower)-1):
                current_char = word_lower[i]
                next_char = word_lower[i+1]
                if current_char in transition_probs and next_char in transition_probs[current_char]:
                    prob = transition_probs[current_char][next_char]
                    transition_scores.append(prob)
                    if prob < 0.01:
                        rare_transition_score += (0.01 - prob) * 100
                    elif prob < 0.05:
                        rare_transition_score += (0.05 - prob) * 20

            features['transition_entropy'] = self.calculate_entropy([p * total_transitions for p in transition_scores]) if transition_scores else 0
            features['rare_transition_score'] = rare_transition_score / (len(transition_scores) + 1e-10)
            features['avg_transition_prob'] = sum(transition_scores)/len(transition_scores) if transition_scores else 0

            word_features.append(features)

        return word_features, avg_length

    def compute_borrowing_probabilities(self, words, features_list, avg_length, word_pos_dict):
        if not words or not features_list:
            return {}

        # Normalize features
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
            # Calculate base score
            base_score = 0
            for feature_name, weight in self.feature_weights.items():
                base_score += weight * normalized_features[feature_name][i]

            # Apply length adjustment
            length_zscore = features_list[i]['length_zscore']
            length_factor = 1 + (0.5 * (1 / (1 + math.exp(-3 * (abs(length_zscore) - 1.5)))) - 0.5)

            # Apply POS adjustment
            pos = word_pos_dict.get(word, 'noun')
            pos_weight = self.pos_weights.get(pos, 0.7)

            # Combine factors
            adjusted_score = base_score * length_factor * pos_weight

            # Sigmoid function to get probability
            prob_borrowed = 1 / (1 + math.exp(-10 * (adjusted_score - 0.5)))

            # Post-processing rules
            if features_list[i]['rare_ngram_score'] > 0.8:
                prob_borrowed = min(prob_borrowed * 1.3, 1.0)
            if features_list[i]['consonant_cluster_score'] > 0.7:
                prob_borrowed = min(prob_borrowed * 1.2, 1.0)
            if features_list[i]['char_anomaly'] > 0.7:
                prob_borrowed = min(prob_borrowed * 1.1, 1.0)
            if features_list[i]['rare_transition_score'] > 0.7:
                prob_borrowed = min(prob_borrowed * 1.3, 1.0)
            if features_list[i]['avg_transition_prob'] < 0.1:
                prob_borrowed = min(prob_borrowed * 1.2, 1.0)

            probabilities[word] = max(0.0, min(1.0, prob_borrowed))

        return probabilities

    def detect_borrowed_words(self, words, word_pos_dict=None):
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
          # Feature extraction
          features_list, avg_length = self.extract_features(current_words)
          for word, features in zip(current_words, features_list):
              full_feature_cache[word] = features

          # Compute borrowing probabilities
          probabilities = self.compute_borrowing_probabilities(
              current_words, features_list, avg_length, word_pos_dict
          )

          current_borrowed = {word for word, prob in probabilities.items() if prob >= self.threshold}
          current_native = set(current_words) - current_borrowed

          #Pattern-based refinement (skip on first iteration)
          if iteration > 0 and current_native:
              native_patterns = self._build_pattern_database(current_native)
              borrowed_patterns = self._build_pattern_database(current_borrowed)

              for word in current_words:
                  features = full_feature_cache[word]
                  original_prob = probabilities[word]

                  # Extract patterns and calculate native-likeness score
                  patterns = self._extract_word_patterns(word)
                  native_score = sum(
                      native_patterns[p] / (native_patterns[p] + borrowed_patterns.get(p, 0.1))
                      for p in patterns if p in native_patterns
                  )
                  pattern_score = native_score / (len(patterns) + EPSILON)

                  # Check "weirdness" based on extracted features
                  is_weird = (
                      features['rare_ngram_score'] > 0.8 or
                      features['consonant_cluster_score'] > 1.5 or
                      features['char_anomaly'] > 0.7 or
                      features['rare_transition_score'] > 0.8 or
                      abs(features['length_zscore']) > 2.5
                  )

                  # Adjust probabilities based on pattern score and weirdness
                  if not is_weird and pattern_score > 0.7:
                      probabilities[word] *= 0.5
                  elif is_weird and pattern_score < 0.3:
                      probabilities[word] = min(probabilities[word] * 1.5, 1.0)

          #Update running average of probabilities
          for word, prob in probabilities.items():
              if word in all_probabilities:
                  prev_avg = all_probabilities[word]
                  all_probabilities[word] = (prev_avg * iteration + prob) / (iteration + 1)
              else:
                  all_probabilities[word] = prob

          # Convergence check
          if previous_borrowed:
              diff = previous_borrowed.symmetric_difference(current_borrowed)
              if len(diff) < self.convergence_threshold * len(previous_borrowed):
                  break

          # Remove current borrowed words
          current_words = [word for word in current_words if word not in current_borrowed]
          previous_borrowed = current_borrowed.copy()

          # Early stopping if too few words remain--> does not seem to affect results
          # if len(current_words) < max(len(words)/25, 0.1 * len(words)):
          #     break

      # Final borrowed words output
      borrowed_words = [
          word for word, prob in sorted(all_probabilities.items(), key=lambda x: -x[1])
          if prob >= self.threshold
      ]
      return borrowed_words, all_probabilities


    def _build_pattern_database(self, words):
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