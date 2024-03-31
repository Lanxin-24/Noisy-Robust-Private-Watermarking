def is_green_list_word(word, key, beta):
    # Compute H and i* for the given word
    H_star = float('inf')
    for i in range(1, h+1):
        H = hmac.new(key, digestmod=hashlib.sha3_256)
        H.update(word.encode())
        H.update(prompt[t - i].encode())
        H_i = int.from_bytes(H.digest(), byteorder='big')
        if H_i < H_star:
            H_star = H_i

    # Generate a random bit based on H_i*
    random.seed(H_star)
    rand = random.random()
    return rand < beta


def count_green_list_words(prompt_result):
    green_list_count = 0
    
    for word in prompt_result:
        # Check if the word is a green list word
        if is_green_list_word(word):
            green_list_count += 1
            
    return green_list_count
