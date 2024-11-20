import spacy
from datasets import load_dataset
from collections import defaultdict
import math

# Load SpaCy model and dataset
nlp = spacy.load("en_core_web_sm")
text_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

# Initialize counts
unigram_counts = defaultdict(int)
bigram_counts = defaultdict(lambda: defaultdict(int))
total_unigrams = 0

# Train unigram and bigram models
for doc in text_data:
    # Process each line as a separate document
    processed_doc = nlp(doc['text'])
    lemmas = ['START']  # Begin each document with START for bigram

    # Collect lemmas, filtering out non-alpha tokens
    for token in processed_doc:
        if token.is_alpha:
            lemma = token.lemma_.lower()
            lemmas.append(lemma)
            unigram_counts[lemma] += 1
            total_unigrams += 1

    # Count bigrams in the document
    for i in range(1, len(lemmas)):
        bigram_counts[lemmas[i - 1]][lemmas[i]] += 1

# Filter out START from unigram_counts to avoid zero division
unigram_counts = {k: v for k, v in unigram_counts.items() if v > 0}

# Convert counts to log probabilities, avoiding zero division errors
unigram_probs = {word: math.log(count / total_unigrams) for word, count in unigram_counts.items()}
bigram_probs = {
    w1: {w2: math.log(count / unigram_counts[w1]) for w2, count in following.items() if unigram_counts[w1] > 0}
    for w1, following in bigram_counts.items()
}

# ---------------- Another Question ----------------
# Using the bigram model to continue the sentence "I have a house in ..." with the most probable word

# Initial prompt
prompt = ["i", "have", "a", "house", "in"]

# Convert the prompt to lemmas using SpaCy
processed_prompt = nlp(" ".join(prompt))
lemmas_prompt = [token.lemma_.lower() for token in processed_prompt if token.is_alpha]

# Get the last word of the prompt to find the most probable next word
last_word = lemmas_prompt[-1]

# Find the most probable next word based on the bigram model
if last_word in bigram_probs:
    next_word = max(bigram_probs[last_word], key=bigram_probs[last_word].get)
    print(f"The most probable next word after '{last_word}' is '{next_word}'")
else:
    print(f"No bigram data available for the word '{last_word}'.")
