import random
import math
import numpy as np
import hashlib
import hmac



# Get the vocabulary of the GPT-2 model
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
vocabulary = tokenizer.get_vocab()



# Use hash function to simulate LLM
def random_logit():
    sign = random.choice([-1, 1]) 
    exponent = random.uniform(-2, 2)
    mantissa = random.uniform(0, 1) 
    return sign * mantissa * math.pow(10, exponent)

def language_model(prompt):
    # Calculate hash value of the prompt
    hash_value = hashlib.sha3_256(' '.join(prompt).encode()).digest()

    # Seed a random number generator with the hash value
    seed = int.from_bytes(hash_value, byteorder='big')
    random.seed(seed)
    
    # Generate logit vector for each word in the vocabulary
    logits = {}
    for word in vocabulary:
        logits[word] = random_logit()

    return logits



# Define the Noisy Robust Private Watermarking Algorithm function
def NoisyRPW(prompt, key, delta, h, n, sigma, beta):
    N_p = len(prompt)
    prompt_result = prompt[:]

    # Generate n tokens
    for t in range(n):
        
        # Apply the language_model to obtain logit vector
        logits = language_model(prompt_result)
        word_vector = list(logits.keys())
        logit_vector = np.array(list(logits.values()))

        # Add noise to the logit vector
        noise = np.random.normal(loc=0, scale=delta*sigma, size=logit_vector.shape)
        noisy_logit_vector = logit_vector + noise
        
        # Sort the logit vector in descending order
        sorted_indices = np.argsort(noisy_logit_vector)[::-1]
        sorted_noisy_logit_vector = [noisy_logit_vector[i] for i in sorted_indices]
        sorted_word_vector = [word_vector[i] for i in sorted_indices]

        # Initialize k the index of the most likely token
        k = 0
        while True:
            # Temporarily set the new token s^(t) to the kth token of sorted_logit_vector
            token = sorted_word_vector[k]

            # Compute H and i*
            H_star = float('inf')
            for i in range(1, h+1):
                H = hmac.new(key, digestmod=hashlib.sha3_256)
                H.update(token.encode())
                new_varnew_var = H.update(prompt_result[N_p + t - i].encode())
                
                H_i = int.from_bytes(H.digest(), byteorder='big')
                if H_i < H_star:
                    H_star = H_i

            # Generate a random bit based on H_i*
            random.seed(H_star)
            rand = random.random()
            if rand < beta:
                random_bit = 1    # Token is on the green list
            else:
                random_bit = 0    # Token is on the red list

            # Decision making
            if random_bit == 1:
                prompt_result.append(token)
                break
            elif random_bit == 0 and sorted_noisy_logit_vector[k+1] < sorted_noisy_logit_vector[0] - delta:
                prompt_result.append(sorted_word_vector[0])
                break
            else:
                k += 1

    return prompt_result
