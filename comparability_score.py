import math
from collections import defaultdict


def get_phonological_features(char):
    """
    Get phonological features for a character using articulatory features.

    Args:
        char: a character

    Returns:
        a dictionary that contains articulatory features
    """

    features = {}
    # Vowel features
    if char.lower() in 'aeiouəɨɯɪʊɛɔæɑɒʌɤøyɵœɶɐɘɜɞʉɯɪʏʊɨʉɘɜɤɚɾʐɭɫʎʝ':
        features['type'] = 'vowel'
        # Height
        if char.lower() in 'iɨɯɪʏʊeøyɵəɘɵ':
            features['height'] = 'high'
        elif char.lower() in 'eɛøœəɘɜɞ':
            features['height'] = 'mid'
        elif char.lower() in 'aæɑɒɐɶ':
            features['height'] = 'low'
        else:
            features['height'] = 'other'

        # Backness
        if char.lower() in 'iɪeɛæyʏøœɶ':
            features['backness'] = 'front'
        elif char.lower() in 'ɨəɘɜɞɵ':
            features['backness'] = 'central'
        elif char.lower() in 'ɯʊoɔɑɒʌɤɐ':
            features['backness'] = 'back'
        else:
            features['backness'] = 'other'

        # Roundedness
        if char.lower() in 'ouɔɒøyʏɵœʊ':
            features['rounded'] = 'rounded'
        else:
            features['rounded'] = 'unrounded'

        # Nasalization
        if char.lower() in 'ãẽĩõũ':
            features['nasalized'] = True
        else:
            features['nasalized'] = False

    # Consonant features
    else:
        features['type'] = 'consonant'

        # Voicing
        if char.lower() in 'bdgjlmnrvwzðʒɮʐɾɦɢʁɰʕʔɣɹɻʀʍ':
            features['voicing'] = 'voiced'
        else:
            features['voicing'] = 'voiceless'

        # Manner of articulation
        if char.lower() in 'bdptdkgqʔ':
            features['manner'] = 'stop'
        elif char.lower() in 'fvθðszʃʒxɣχʁħʕh':
            features['manner'] = 'fricative'
        elif char.lower() in 'mnŋɲɳɴ':
            features['manner'] = 'nasal'
        elif char.lower() in 'ʦʣʧʤ':
            features['manner'] = 'affricate'
        elif char.lower() in 'rlɹɾɽʀ':
            features['manner'] = 'liquid'
        elif char.lower() in 'jw':
            features['manner'] = 'glide'
        else:
            features['manner'] = 'other'

        # Place of articulation
        if char.lower() in 'bpmfvw':
            features['place'] = 'labial'
        elif char.lower() in 'tdnszlrθð':
            features['place'] = 'dental/alveolar'
        elif char.lower() in 'ʃʒʧʤ':
            features['place'] = 'postalveolar'
        elif char.lower() in 'jɲçʝ':
            features['place'] = 'palatal'
        elif char.lower() in 'kgŋxɣ':
            features['place'] = 'velar'
        elif char.lower() in 'qχʁ':
            features['place'] = 'uvular'
        elif char.lower() in 'ħʕʜʢ':
            features['place'] = 'pharyngeal'
        elif char.lower() in 'ʔh':
            features['place'] = 'glottal'
        else:
            features['place'] = 'other'

    return features

def feature_diff(char1, char2):
    """
    Calculate the feature difference between two characters.

    Args:
        char1: first character
        char2: second character
    Returns:
        a value between 0 (identical) and 1 (completely different).
    
    information:
        - Relies on these functions:
            `get_phonological_features` → returns a dictionary that contains articulatory features.
    """

    if char1 == char2:
        return 0.0

    # Get features for both characters
    f1 = get_phonological_features(char1)
    f2 = get_phonological_features(char2)

    if f1.get('type') != f2.get('type'):
        return 1.0

    common_keys = set(f1.keys()) & set(f2.keys())
    if not common_keys:
        return 1.0

    matches = sum(1 for k in common_keys if f1[k] == f2[k])
    feature_similarity = matches / len(common_keys)

    return 1.0 - feature_similarity


def feature_edit_distance(word1, word2):
    """
    Calculate the feature-weighted edit distance between two words.

    Args:
        word1: first word
        word2: second word

    Returns:
        A scalar that indicates the feature weighted edit distance between the two words
        
    information:
        Substitution cost is based on feature distance:
     
    """


    if not word1 and not word2:
        return 0.0

    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            # Substitution cost based on feature difference
            if word1[i-1] == word2[j-1]:
                subst_cost = 0
            else:
                subst_cost = feature_diff(word1[i-1], word2[j-1])

            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + subst_cost
            )

    return 1-dp[m][n]


def nw_align(s1, s2, match=1, mismatch=-1, gap=-1):
    """
    Needleman-Wunsch pairwise alignment.

    Args:
        s1: first word
        s2: second word
        match=match score
        mismatch: mismatch score

    Returns:
        aligned pairs of words
    """

    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0] = i*gap
    for j in range(1,m+1): dp[0][j] = j*gap

    for i in range(1,n+1):
        for j in range(1,m+1):
            score_diag = dp[i-1][j-1] + (match if s1[i-1] == s2[j-1] else mismatch)
            score_up = dp[i-1][j] + gap
            score_left = dp[i][j-1] + gap
            dp[i][j] = max(score_diag, score_up, score_left)

    i,j = n,m
    align1, align2 = [], []
    while i>0 or j>0:
        if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + (match if s1[i-1] == s2[j-1] else mismatch):
            align1.append(s1[i-1])
            align2.append(s2[j-1])
            i -= 1; j -= 1
        elif i>0 and dp[i][j] == dp[i-1][j] + gap:
            align1.append(s1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(s2[j-1])
            j -= 1
    return ''.join(reversed(align1)), ''.join(reversed(align2))


def extract_char_pairs_context(al1, al2):
    """
    Extract char-char pairs with neighbors in aligned words.

    Args:
        al1: first aligned word
        al2: second aligned word
        match=match score
        mismatch: mismatch score

    Returns:
        a list of phonemes and their phonological contexts
    """
    pairs = []
    length = len(al1)
    for i in range(length):
        c1 = al1[i]
        c2 = al2[i]
        left1 = al1[i-1] if i>0 else None
        right1 = al1[i+1] if i<(length-1) else None
        left2 = al2[i-1] if i>0 else None
        right2 = al2[i+1] if i<(length-1) else None
        pairs.append(((c1,c2), (left1,left2), (right1,right2)))
    return pairs


def normalize_counts(counts):
    """
    Normalize counts to probabilities with add-one smoothing.

    Args:
        al1: counts to be normalized

    Returns:
        a dictionary in which values are probabilities
    """
    
    total = sum(counts.values()) + len(counts)
    probs = {k: (v+1)/total for k,v in counts.items()}
    return probs


def align_and_score(words_nlang):
    """
    Aligns and scores word pairs across multiple languages using character-pair 
    context probabilities and feature-based edit distance.

    This function:
    1. Iterates over all possible language pairs from the given aligned word sets.
    2. Performs Needleman–Wunsch alignment on each word pair.
    3. Extracts contextual character pairs from the alignments and counts their occurrences.
    4. Converts counts into probabilities for each character context triplet.
    5. Scores each aligned pair by combining:
        - The log-probability of its character-pair contexts.
        - The feature-based edit distance between the aligned words.
    6. Normalizes the scores between 0 and 1 for comparability.
    7. Returns the scored pairs sorted in descending order of similarity.

    Parameters
    ----------
    words_nlang : list of tuple of str
        A list where each element is a tuple of words, one per language, 
        representing aligned words across all languages.
        Example for 3 languages: 
        [
            ("house", "haus", "casa"),
            ("cat", "katze", "gato"),
            ...
        ]

    Returns
    -------
    list of tuple
        A list of tuples in the form:
        [((word_langA, word_langB), normalized_score), ...]
        where `normalized_score` is between 0 (least similar) and 1 (most similar),
        sorted in descending order of similarity.

    information
    -----
    - Relies on these functions:
        `nw_align(wordA, wordB)` → returns aligned strings (alA, alB).
        `extract_char_pairs_context(alA, alB)` → yields triplets (center, left, right).
        `normalize_counts(counts)` → converts frequency counts into probabilities.
        `feature_edit_distance(alA, alB)` → computes a numeric edit distance score.
    - The log-probability is smoothed with a default probability of 1e-8 for unseen contexts.
    - The final score is the average of the log-probability and the edit distance score.
    """
    n_langs = len(words_nlang[0])
    languages = list(range(n_langs))

    pairs_indices = [(i, j) for i in range(n_langs) for j in range(i + 1, n_langs)]

    context_counts = defaultdict(int)
    aligned_pairs_storage = defaultdict(list)

    for wset in words_nlang:
        for (i, j) in pairs_indices:
            wA, wB = wset[i], wset[j]
            alA, alB = nw_align(wA, wB)
            aligned_pairs_storage[(i, j)].append((wA, wB, alA, alB))
            pairs = extract_char_pairs_context(alA, alB)
            for center, left, right in pairs:
                context_counts[(center, left, right)] += 1

    prob_map = normalize_counts(context_counts)

    scored_pairs = []
    for (i, j), aligned_list in aligned_pairs_storage.items():
        for (wA, wB, alA, alB) in aligned_list:
            pairs = extract_char_pairs_context(alA, alB)
            log_prob = 0.0
            for center, left, right in pairs:
                p = prob_map.get((center, left, right), 1e-8)
                log_prob += math.log(p)
            scored_pairs.append(((wA, wB), (log_prob + feature_edit_distance(alA, alB))/2))  

    log_probs = [score for (_, score) in scored_pairs]
    if not log_probs:
        return []

    min_log_prob = min(log_probs)
    max_log_prob = max(log_probs)

    if max_log_prob == min_log_prob:
        normalized_scored_pairs = [((wA, wB), 1.0) for ((wA, wB), _) in scored_pairs]
    else:
        normalized_scored_pairs = [
            ((wA, wB), (score - min_log_prob) / (max_log_prob - min_log_prob))
            for ((wA, wB), score) in scored_pairs
        ]

    normalized_scored_pairs.sort(key=lambda x: x[1], reverse=True)
    return normalized_scored_pairs


def align_and_score_by_concept(word_lang_list):
    """
    word_lang_list: list of (word, language) for a single concept
    Returns: dict of (wordA, wordB) -> normalized log_prob score
    Information:
        -relies on the `align_and_score` function above
    """
    from collections import defaultdict

    lang_to_word = {}
    for word, lang in word_lang_list:
        lang_to_word[lang] = word

    if len(lang_to_word) < 2:
        return {}

    sorted_langs = sorted(lang_to_word)
    wset = tuple(lang_to_word[lang] for lang in sorted_langs)

    scored_pairs = align_and_score([wset])

    return {pair: score for pair, score in scored_pairs}

