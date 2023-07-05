import random
from random import shuffle
import nltk 
import re 
from nltk.corpus import wordnet 


nltk.download('wordnet','/root/nltk_data')
random.seed(1)
import warnings 
warnings.filterwarnings('ignore')

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']
########################################################################
# word Level perturbations 
######################################################################## 


# common method 
def split_words(sentence):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)
    return words, num_words 

def concat(words):
    return " ".join(words)

import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

#! synonym replacement (SR)
def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

def word_synonym_replacement(sentence,alpha=0.1): 
    words,num_words = split_words(sentence)
    n = max(1, int(alpha*num_words))
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break
        
    return concat(new_words)

#! random_insertion (WR)
def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
 
def word_insertion(sentence,alpha=0.1):
    words,num_words = split_words(sentence)
    n = max(1, int(alpha*num_words))
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return concat(new_words)

#! random_swap (WS,word swap)
def _swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

def word_swap(sentence,alpha=0.1):
    words,num_words = split_words(sentence)
    n = max(1, int(alpha*num_words))
    new_words = words.copy()
    for _ in range(n):
        new_words = _swap_word(new_words)
    return concat(new_words)

#! random_deletion (WD, word deletion)
def word_deletion(sentence,alpha=0.1):
    words,num_words = split_words(sentence)
    
    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > alpha:
            new_words.append(word)
    
    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return concat(new_words)
    
#! insert puctuation (IP)
def word_insert_punctuation(sentence, punc_ratio=0.3):
    PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
    
    words = sentence.split(' ')
    new_line = []
    
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
            new_line.append(word)
        else:
            new_line.append(word)
    return concat(new_line)


########################################################################
# Character Level perturbations 
########################################################################

import nlpaug.augmenter.char as nac
def char_keyboard(sentence,alpha=0.1): #! Keyboard 
    aug = nac.KeyboardAug(aug_word_p=alpha)   
    new_sentence = aug.augment(sentence)
    return concat(new_sentence)

def char_ocr(sentence,alpha=0.1): #! OCR 
    aug = nac.OcrAug(aug_word_p=alpha)   
    new_sentence = aug.augment(sentence)
    return concat(new_sentence)

def char_insert(sentence,alpha=0.1): #! insert 
    aug = nac.RandomCharAug('insert',aug_word_p=alpha)   
    new_sentence = aug.augment(sentence)
    return concat(new_sentence)

def char_substitute(sentence,alpha=0.1): #! substitute
    aug = nac.RandomCharAug('substitute',aug_word_p=alpha)   
    new_sentence = aug.augment(sentence)
    return concat(new_sentence)

def char_swap(sentence,alpha=0.1): #! swap 
    aug = nac.RandomCharAug('swap',aug_word_p=alpha)   
    new_sentence = aug.augment(sentence)
    return concat(new_sentence)

def char_delete(sentence,alpha=0.1): #! delete 
    aug = nac.RandomCharAug('delete',aug_word_p=alpha)   
    new_sentence = aug.augment(sentence)
    return concat(new_sentence)

def backtrans(sentence):
    return sentence 

#method_chunk = [word_synonym_replacement,word_insertion,word_swap,word_deletion,word_insert_punctuation,
#                char_keyboard,char_ocr,char_insert,char_substitute,char_swap,char_delete,'style_former','style_casual','style_passive','style_active','backtrans']

method_chunk = [word_synonym_replacement,word_insertion,word_swap,word_deletion,word_insert_punctuation,
                char_keyboard,char_ocr,char_insert,char_substitute,char_swap,char_delete]
train_chunk = method_chunk 

def get_method_chunk():
    return method_chunk

def get_method_chunk_train():
    return train_chunk

#from styleformer import Styleformer
# class style_transfer:
#     def __init__(self,style,device):
#         self.style_dict = {'style_former':0,'style_casual':1,'style_passive':2,'style_active':3}
#         self.style = style 
#         self.style_value = self.style_dict[style]
#         self.sf = Styleformer(style=self.style_value)
#         self.device = device 
    
#     def __call__(self,sentence):
#         new_sentence = self.sf.transfer(sentence,inference_on=self.device)
#         return new_sentence 
 