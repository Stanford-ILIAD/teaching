"""
    Simple script for creating the data used to train CompILE on the Omniglot dataset.
 """
 
import digits_utils 
import pickle
import numpy as np
np.random.seed(0)

def generate_words(n=100, save_words=None):
    sixgrams = {"c-a-t-":0.05,
    "b-a-t-":0.2,
    "b-e-a-":0.1,
    "t-e-a-":0.1,
    "e-a-t-":0.1,
    "a-t-^-":0.1,
    "-e-a-^":0.1,
    "--b-a-":0.1,
    "-a--t-":0.1,
    "-e-^a-":0.05}
    bigrams = {"--":0.4,"^-":0.4,"c-":0.2}
    sixgrams_keys = list(sixgrams.keys())
    bigrams_keys = list(bigrams.keys())
    sixgrams_p = [sixgrams[k] for k in sixgrams_keys]
    bigrams_p = [bigrams[k] for k in bigrams_keys]
    words = []
    for i in range(n):
        words.append(np.random.choice(sixgrams_keys, p=sixgrams_p)+np.random.choice(bigrams_keys, p=bigrams_p))
        words.append(np.random.choice(bigrams_keys, p=bigrams_p)+np.random.choice(sixgrams_keys, p=sixgrams_p))
    if(save_words):
        pickle.dump(words, open(save_words,"wb"))
    return words


alphabet = digits_utils.generate_skill_dict() 
pickle.dump(alphabet, open("final_chars_dict", "wb")) # dict mapping chars to trajectories
generate_words(n=1000, save_words="1k_words_for_compile.pkl") # sample of trajectories to use for training compile