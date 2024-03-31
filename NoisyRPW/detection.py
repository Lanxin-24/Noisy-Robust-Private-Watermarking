def count_green_list_words(prompt, prompt_result, key, beta, h):
    green_list_count = 0
    N_p = len(prompt)

    # Check if the word t is a green list word and count
    for t in range(N_p, len(prompt_result)):    
        word = prompt_result[t]
        
        # Compute H and i* for the given word
        H_star = float('inf')
        for i in range(1, h+1):
        H = hmac.new(key, digestmod=hashlib.sha3_256)
        H.update(word.encode())
        H.update(prompt_result[t - i].encode())
        H_i = int.from_bytes(H.digest(), byteorder='big')
        if H_i < H_star:
            H_star = H_i

        # Generate a random bit based on H_i*
        random.seed(H_star)
        rand = random.random()
        if rand < beta
            green_list_count += 1
            
    return green_list_count
