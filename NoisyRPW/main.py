import random
import numpy as np
import hashlib
import hmac
import secrets

# Get the vocabulary of the GPT-3 model
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocabulary = tokenizer.get_vocab()


# Use hash function to simulate LLM
def language_model(prompt):
    # Calculate hash value of the prompt
    hash_value = hashlib.sha3_256(' '.join(prompt).encode()).digest()

    # Seed a random number generator with the hash value
    seed = int.from_bytes(hash_value, byteorder='big')
    random.seed(seed)

    # Generate probabilities for each word in the vocabulary
    total = 0
    probabilities = {}
    for word in vocabulary:
        probability = random.uniform(0, 1)
        probabilities[word] = probability
        total += probability

    # Normalize probabilities to get a valid distribution
    distribution = {word: probability / total for word, probability in probabilities.items()}

    return distribution


# Define the Noisy Robust Private Watermarking Algorithm function
def NoisyRPW(prompt, key, delta, h, n, sigma):

    # 1 Generate n tokens
    for t in range(n+1):
        # 2 Apply the language_model to obtain logit vector
        distribution = language_model(prompt)
        word_vector = list(distribution.keys())
        logit_vector = list(distribution.values())

        # 3 Sort the logit vector in descending order
        sorted_indices = np.argsort(logit_vector)[::-1]
        sorted_logit_vector = [logit_vector[i] for i in sorted_indices]
        sorted_word_vector = [word_vector[i] for i in sorted_indices]

        # 4 Initialize k the index of the most likely token

        k = 0
        while True:
            # 5 Temporarily set the new token s^(t) to the kth token of sorted_logit_vector
            token = sorted_word_vector[k]

            # 6 Compute H and i*
            # Compute sens
            PRF_vector = []
            for l in range(len(logit_vector)):
                F = hmac.new(key, digestmod=hashlib.sha3_256)
                F.update(token.encode())
                F.update(sorted_word_vector[l].encode())
                PRF_vector.append(int.from_bytes(F.digest(), byteorder='big'))


            # Compute H*
            H_star = float('inf')
            for i in range(1, h+1):
                H = hmac.new(key, digestmod=hashlib.sha3_256)
                H.update(token.encode())
                H.update(prompt[t - i].encode())
                H_i = int.from_bytes(H.digest(), byteorder='big')
                if H_i < H_star:
                    H_star = H_i

            # 7 Generate a random bit based on H_i*
            random.seed(H_star)
            random_bit = random.randint(0, 1) # 0: green, 1: red
            # Generate a random noise from a normal distribution with mean 0 and standard deviation 0.5
            noise = np.random.normal(loc=0, scale=0.5)
            noisy_random_bit = random_bit + noise
            noisy_random_bit = int(round(noisy_random_bit))
            noisy_random_bit = max(0, min(noisy_random_bit, 1))

            # 8-10 Decision making
            if noisy_random_bit == 0:
                prompt.append(token)  # Token is on the green list
                break
            elif noisy_random_bit == 1 and sorted_logit_vector[k+1] < sorted_logit_vector[0] - delta:
                prompt.append(sorted_word_vector[0])  # Token is on the red list
                break
            else:
                k += 1

    return prompt

K = secrets.token_bytes(32)

print(NoisyRPW(['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'], K, 0.3, 3, 15, 0.1))