"""
Relevance relevance_baselines:
1. KL div
2. Semantic similarity
3. RM3 score 
"""
from my_utils import utils_linux
import math
import sys
import timeit
import nltk
import string
import os

def calc_word_in_sen_prob(curr_word,sen,source):
    collection_stats_RT_pickle = utils_linux.read_pickle("tf_RT_dict") #!!!! 
    collection_stats_wiki_pickle = utils_linux.read_pickle("tf_wiki_dict")
    mu_wiki = 19
    mu_RT = 24 
    
    word_freq_sen = sen.count(curr_word)
    if source == "RT":
        try:
            if curr_word in collection_stats_RT_pickle.keys() :
                p_dir_word = float(float((word_freq_sen+mu_RT*collection_stats_RT_pickle[curr_word]))/float(len(sen)+mu_RT))
            elif curr_word.lower() in collection_stats_RT_pickle.keys():
                p_dir_word = float(float((word_freq_sen+mu_RT*collection_stats_RT_pickle[curr_word.lower()]))/float(len(sen)+mu_RT))
            elif curr_word.upper() in collection_stats_RT_pickle.keys():
                p_dir_word = float(float((word_freq_sen+mu_RT*collection_stats_RT_pickle[curr_word.upper()]))/float(len(sen)+mu_RT))
            else:
                print curr_word +" not in RT_tf_dict"
        except Exception as err:
            sys.stderr.write('problem in RT_tf_dict:' + curr_word)     
            print err.args      
            print err
    elif source == "wiki":
        try:
            if curr_word in collection_stats_wiki_pickle.keys() or curr_word.lower() in collection_stats_wiki_pickle.keys() or curr_word.upper() in collection_stats_wiki_pickle.keys():
                p_dir_word = float(float((word_freq_sen+mu_RT*collection_stats_wiki_pickle[curr_word]))/float(len(sen)+mu_wiki))
            else:
                print curr_word +" not in wiki_tf_dict"
        except Exception as err:
            sys.stderr.write('problem in wiki_tf_dict:' + curr_word)     
            print err.args      
            print err
    return p_dir_word
    
def calc_KL_div_between_clm_sen(collection):
    """
    KL (Pc||Qs) = sigma_claim_words(P(w|Mc)*log(P(w|Mc)/P(w|Ms)),
    when P(w|Ms) is Dirichlet smoothed
    Stages:
    1. According to setup -  unified collection or separated collections - RT and wiki
    2. For every claim and sentence pair, according to the corpus the sen comes from,
    3. Get the tf of each claim word in the corpus
    4. calc the the dist according to the sentence model
    5. calc dist according to claim model
    6. KL div  
    """
    if collection == "RT":
        curr_clm_sen_dict = utils_linux.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\model\clm_as_key_sen_support_score_val_RT")
    elif collection =="wiki":
        curr_clm_sen_dict = utils_linux.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\model\clm_as_key_sen_support_score_val_wiki")
    KL_div_clm_sen_dict_curr = {} #key is clm,sen val is the KL div
    
    
    for (clm,sen_list) in curr_clm_sen_dict.items():
        for word in clm.split():
            word_clm_prob = float(float(clm.count(word))/float(len(clm.split())))
            for sen in sen_list:
                word_sen_prob = calc_word_in_sen_prob(word,sen[0],collection) #the second entry is the true supportivness score 
                if (clm,sen[0]) in KL_div_clm_sen_dict_curr.keys():
                    KL_div_clm_sen_dict_curr[(clm,sen[0])] += float(word_clm_prob + float(math.log(float(word_clm_prob))/float(word_sen_prob)))
                else:
                    KL_div_clm_sen_dict_curr[(clm,sen[0])] = float(word_clm_prob + float(math.log(float(word_clm_prob))/float(word_sen_prob)))                                       
    if collection == "RT":
        utils_linux.save_pickle("KL_div_clm_sen_dict_RT", KL_div_clm_sen_dict_curr)
    elif collection == "wiki":
        utils_linux.save_pickle("KL_div_clm_sen_dict_wiki", KL_div_clm_sen_dict_curr)
        
def calc_lexical_similarity_based_on_KL_div(collection):
    """
    relevance baseline as:
    P(s belongs to Rel(c)| c ) ~ exp(-KL(P_(c ) ||P_(s ))
    """
    lexical_sim_relevance_bsln_curr = {}
    
    curr_KL_div_clm_sen_dict = utils_linux.read_pickle("KL_div_clm_sen_dict_"+collection)  
    for (clm,sen) in curr_KL_div_clm_sen_dict.keys():
        lexical_sim_relevance_bsln_curr[(clm,sen)] = math.exp(-curr_KL_div_clm_sen_dict[(clm,sen)])  
    utils_linux.save_pickle("lexical_sim_relevance_bsln_"+collection, lexical_sim_relevance_bsln_curr)
    
def create_relevance_clm_sen_score():
    """
    compare the results the baseline achieved to, with respect to the labels we have- relevance as a binary level.
    Transform to 
    """
    clm_sen_support_ranking_sorted = utils_linux.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\clm_sen_support_ranking_sorted_full")
    clm_sen_relevance_dict = {} #key is (clm, sen) , val is 1- relevant, 0 - not relevant 
    
    
    for (clm,sen,score) in clm_sen_support_ranking_sorted.keys():
        if score > 0:
            clm_sen_relevance_dict[(clm,sen)] = 1
        else:
            clm_sen_relevance_dict[(clm,sen)] = 0
    utils_linux.save_pickle("clm_sen_relevance_dict", clm_sen_relevance_dict)        
    
def evaluate_standard_retrieval_relevancy():
    """
    In comparison to a ranked list of sentence from the IR methods used in Indri,
    compare this to the relevance judgment we have
   Need to CV 3 paranms:
       the number of docs for the working set
       the weight of the doc title
       the weight between doc score and sen score
    """
   
    """---------------------------------------------    
    07.10.14 update
    for a retrieval process as relevance baseline (and also supportiveness):
    1. Build unigram Language Model with MLE estimates for claims as a mixture of the claim and the entity:
        LM_claim: a*claim+ (1-a)*entity
    2.  Build unigram LM with MLE estimates for a document as a mixture of the body (=sentences), the title and the corpus :
        LM_document: = b_1*body+ b_2*title+b_(3 )*corpus            ,  b_1+b_2+b_3=1, b-3 = 0.1 ----> so b_2 = 0.9 - b_1
        Tune b_1,b_2  (JM smoothing with corpus)     
    3.    Document retrieval according to:
        Score(d) = sim(LM_claim, LM_document)  using Cross Entropy
                   meaning claim represented according to LM_claim and represented according to LM_document  (No cut-off, retrieve all documents).
    4. Sentence retrieval :
            Interpolation of the similarity between claim as represented  by LM_claim and sentences_LM (smooth with corpus) and the document d score (of the sentence):
            scorefinal(s) = c *score(d) + (1-c) *sim( claim, sentence) 
                                 
    """
def represent_claim_with_claim_LM_prob_vector(claim_entity_list):
    claim_represented_by_clmLM_vector_for_alpha = {}
    term_stem_tf_idf_wiki = utils_linux.read_pickle("term_stem_tf_idf_wiki")
    
    for (curr_entity,curr_clm) in claim_entity_list:
        if "'s" in curr_entity:
            curr_entity = curr_entity.replace("'","")
        if "'s" in curr_clm:
            curr_clm = curr_clm.replace("'s","")
#         curr_clm_words = nltk.word_tokenize(curr_clm.replace("-"," ").replace(","," "))
        curr_clm_words = remove_excluded_chars_clean_line(curr_clm)
        curr_clm_words_set = set(word.lower() for word in curr_clm_words)
        curr_clm_words_set_list = [word for word in curr_clm_words_set]
#         curr_clm_words = [word for word in curr_clm_words_set]
        curr_entity_words = nltk.word_tokenize(curr_entity.replace("-"," ").replace(","," "))
        curr_entity_words = [word.lower() for word in curr_entity_words]
        curr_clm_stems = []
        curr_entity_stems = [] 
            
        for word in curr_clm_words_set_list:
            try:
                curr_clm_stems.append(term_stem_tf_idf_wiki[word][0])
            except Exception as err: 
                sys.stderr.write('problem in create_claim_LM_prob_vector in word ' + word)     
                print err.args      
                print err 
        
        for word in curr_entity_words:
            try:
                curr_entity_stems.append(term_stem_tf_idf_wiki[word][0])
            except Exception as err: 
                    sys.stderr.write('problem in create_claim_LM_prob_vector in word ' + word)     
                    print err.args      
                    print err
        for alpha_int in range(0,11,1):
            alpha_float = float(float(alpha_int)/float(10))
            
                        
            clm_vector = [0]*len(curr_clm_words_set_list) #the p vector
            for word in curr_clm_words_set_list:
                clm_vector[curr_clm_words_set_list.index(word)] = alpha_float*(float((float(curr_clm_stems.count(term_stem_tf_idf_wiki[word][0]))/float(len(curr_clm_words)))))
                if word in curr_entity_words:
                    clm_vector[curr_clm_words_set_list.index(word)] += (1-alpha_float)*(float((float(curr_entity_stems.count(term_stem_tf_idf_wiki[word][0]))/float(len(curr_entity_words)))))
            
            #prob check
            if str(sum(clm_vector)) !="1.0":
                print "clm_vector not sum to 1:" +curr_clm +str(sum(clm_vector))
            
            if curr_clm in claim_represented_by_clmLM_vector_for_alpha.keys():
                claim_represented_by_clmLM_vector_for_alpha[curr_clm].extend([clm_vector])
            else:
                claim_represented_by_clmLM_vector_for_alpha[curr_clm] = [clm_vector]
    return claim_represented_by_clmLM_vector_for_alpha 


def remove_and_split_on_non_ascii_chars(all_sen_words):
    to_remove = []
    to_add = []
    for word in all_sen_words:
        non_asci_char = [c for c in word if not 0 < ord(c) < 127]
        non_asci_char_word = ''.join(non_asci_char)
        if len(non_asci_char_word) >0:
            orig_word = word
            for c in non_asci_char:
                word= word.replace(c, " ")
#                 new_word = word.replace(non_asci_char_word, " ")
            to_remove.append(orig_word)
            new_words = word.split()
            for word in new_words:
                to_add.append(word)
    for word in to_remove:
        all_sen_words.remove(word)
    for word in to_add:
        all_sen_words.append(word)
    return all_sen_words

def remove_excluded_chars_clean_line(line):
    exclude = set(string.punctuation)
    
    for s in exclude:
            if s in line:
                line = line.replace(s," ")
#         all_sen_words = nltk.word_tokenize(line.replace("-"," ").replace(","," ").replace(".", " "))
    all_sen_words = nltk.word_tokenize(line)
    all_sen_words = remove_and_split_on_non_ascii_chars(all_sen_words)
    for word in all_sen_words:
        if word.count("/") >=2 or "www." in word or ".com" in word or "http" in word:
            try:
                all_sen_words.remove(word)
                continue
            except Exception as err: 
                sys.stderr.write('problem in word.count("/"): ' +word)     
                print err.args      
                print err
        if word.count("/") == 1:
            try:
                all_sen_words.append(word.split("/")[0])
                all_sen_words.append(word.split("/")[1])
                all_sen_words.remove(word)
            except Exception as err: 
                sys.stderr.write('problem in word.count("/"): ' +word)     
                print err.args      
                print err  
    corrected_words = []
    for word in all_sen_words:
        try:
            if any(s in word for s in exclude):
                    corrected_words.append((word,''.join(c for c in word if c not in exclude)))
        except Exception as err: 
                sys.stderr.write('problem in exclude: ' +s)     
                print err.args      
                print err
    for (bad_word,correct_word) in corrected_words:
        try:
            all_sen_words.remove(bad_word)
            if len(correct_word) >0:
                all_sen_words.append(correct_word)
        except Exception as err: 
            sys.stderr.write('problem in all_sen_words: ' + correct_word +" in bad_word: "+bad_word)     
            print err.args      
            print err  
    
    return all_sen_words
    
    

def represent_claim_with_doc_LM_prob_vector(claim_entity_list):#     claim_represented_by_docLM_vector_for_beta = {}
    print "enter  represent_claim_with_doc_LM_prob_vector"
    term_stem_tf_idf_wiki = utils_linux.read_pickle("term_stem_tf_idf_wiki")
    print " after read term_stem_tf_idf_wiki enter represent_claim_with_doc_LM_prob_vector"
    wiki_docs_path = r"/lv_local/home/liorab/support/corpuses/wikipedia_movie_articles_0614"
    
    doc_error = open("bad_docs.txt","wb")
    progress_log = open("progress.log","wb")
    error_log = open("errors.log","wb")
    #go over the documents, and according to b_1*body + (1-b_1)*title +0.1*corpus  
    corpus_beta = 0.1
    non_sentence_lines= ['<DOC>','<DOCNO>','<TEXT>','<title>','<senti>','</DOC>','</DOCNO>','</TEXT>','</title>','</senti>']
      
    for (curr_entity,curr_clm) in claim_entity_list:
        claim_represented_by_docLM_vector_for_beta = {}
        clm_counter = 1
        start = timeit.default_timer()
        print("################  STARTED CLAIM " +curr_clm +" IN " +str(start) +" TIME")
        progress_log.write("################  STARTED CLAIM " +curr_clm +" IN " +str(start) +" TIME")
        if "'s" in curr_clm:
            curr_clm = curr_clm.replace("'s","")
        curr_clm_words = nltk.word_tokenize(curr_clm.replace("-"," ").replace(","," "))
        curr_clm_words_set = set(word.lower() for word in curr_clm_words)
        curr_clm_words_set_list = [word for word in curr_clm_words_set]
        for filename in os.listdir(wiki_docs_path):
            try:
                curr_body_stems = []
                curr_title_stems = []
                clm_vectors_for_curr_doc = []
                exclude = set(string.punctuation)
                remove_also = ["''"]
                for s in remove_also:
                    exclude.add(s)
                if ".txt" in filename:
                    curr_doc_title = filename.split(".txt")[0]
                    curr_title_words = remove_excluded_chars_clean_line(curr_doc_title)
                    curr_title_words = [word.lower() for word in curr_title_words if not word in exclude and len(word) >0 ]
                    with open(wiki_docs_path+"/"+filename, 'r') as f:
                        doc_lines=f.read().strip()
                        curr_body_words = []
                        for i, line in enumerate(doc_lines.split('\n')):
                            if any(sign in line for sign in non_sentence_lines):
                                continue
                            curr_line_words = remove_excluded_chars_clean_line(line) 
                            curr_body_words.extend([word.lower() for word in curr_line_words if not word in exclude and len(word)>0])
                        for word in curr_body_words:
                            try:
                                curr_body_stems.append(term_stem_tf_idf_wiki[word][0])
                            except Exception as err: 
#                                 sys.stderr.write('problem in finding stem for word in body in word: ' + word +" in document: "+curr_doc_title)     
#                                 print err.args      
#                                 print err
                                error_log.write('problem in finding stem for word in body in word: ' + word +" in document: "+curr_doc_title +'\n')
#                                 error_log.write(err.args +'\n')
                                error_log.write(str(err) +'\n')
                        for word in curr_title_words:
                            try:
                                curr_title_stems.append(term_stem_tf_idf_wiki[word][0])
                            except Exception as err: 
#                                 sys.stderr.write('problem in finding stem for title in word ' + word +" in document: "+curr_doc_title)   
#                                 print err.args      
#                                 print err
                                error_log.write('problem in finding stem for title in word: ' + word +" in document: "+curr_doc_title +'\n')
#                                 error_log.write(err.args +'\n')
                                error_log.write(str(err) +'\n')
                    for beta_int in range(0,10,1):
                        beta_float = float(float(beta_int)/float(10))
                        clm_vector = [0]*len(curr_clm_words_set_list) #the q vector
                        for word in curr_clm_words_set_list:
                            clm_vector[curr_clm_words_set_list.index(word)] = beta_float*(float((float(curr_body_stems.count(term_stem_tf_idf_wiki[word][0]))/float(len(curr_body_words)))))         
                            if word in curr_title_words:
                                try:
                                    clm_vector[curr_clm_words_set_list.index(word)] += (1-beta_float)*(float((float(curr_title_stems.count(term_stem_tf_idf_wiki[word][0]))/float(len(curr_title_words)))))
                                except Exception as err: 
#                                     sys.stderr.write('problem in clm_vector[curr_clm_words.index(word)] ' + word +" in document: "+curr_doc_title)   
#                                     print err.args      
#                                     print err
                                    error_log.write('problem in clm_vector[curr_clm_words.index(word)] in word: ' + word +" in document: "+curr_doc_title +'\n')
#                                     error_log.write(err.args +'\n')
                                    error_log.write(str(err) +'\n')  
                            clm_vector[curr_clm_words_set_list.index(word)] += corpus_beta*float(term_stem_tf_idf_wiki[word][1])
                        clm_vectors_for_curr_doc.append(clm_vector)   
                    if (curr_clm) in claim_represented_by_docLM_vector_for_beta.keys():
                        claim_represented_by_docLM_vector_for_beta[(curr_clm)].extend([(curr_doc_title,clm_vector)])
                    else:
                        claim_represented_by_docLM_vector_for_beta[(curr_clm)] = [(curr_doc_title,clm_vector)]   
            except Exception as err: 
#                     sys.stderr.write('problem in represent_claim_with_doc_LM_prob_vector in doc ' + curr_doc_title)     
#                     print err.args      
#                     print err
                    error_log.write('problem in represent_claim_with_doc_LM_prob_vector in document: '+curr_doc_title +'\n')
#                     error_log.write(err.args +'\n')
                    error_log.write(str(err) +'\n')
        stop = timeit.default_timer()
        print "################  FINISHED CLAIM " +curr_clm +" IN " +str(stop - start ) +" TIME"
        progress_log.write("################  FINISHED CLAIM " +curr_clm +" IN " +str(stop - start ) +" TIME")
        utils_linux.save_pickle("claim_"+str(clm_counter)+"_represented_by_docLM_vector_for_beta", claim_represented_by_docLM_vector_for_beta)
        clm_counter += 1  
    progress_log.close()
    error_log.close()
#     return claim_represented_by_docLM_vector_for_beta                

def create_term_stem_tf_idf_dict():
    term_stem_file_path_wiki = r"./wiki_terms_stem"
    term_tf_file_path_wiki = r"./wiki_tf"
    term_idf_file_path_wiki = r"./wiki_idf"
     
    term_stem_tf_idf_wiki = {} #key is a term, val is a tuple - stem, tf and idf values
    
    term_stem_file_wiki =  open(term_stem_file_path_wiki,"rb").read().strip()
    for i, line in enumerate(term_stem_file_wiki.split('\n')):
        term_stem_tf_idf_wiki[line.split("|")[0]] = [line.split("|")[1]]
    
    term_tf_file_wiki = open(term_tf_file_path_wiki,"rb").read().strip()
    for i, line in enumerate(term_tf_file_wiki.split('\n')):  
        if len(term_stem_tf_idf_wiki[line.split("|")[0]]) == 2: #already got the tf value for this term
            continue
        else:
            term_stem_tf_idf_wiki[line.split("|")[0]].append(line.split("|")[1])
    
    term_idf_file_wiki = open(term_idf_file_path_wiki,"rb").read().strip() 
    for i, line in enumerate(term_idf_file_wiki.split('\n')):
        if len(term_stem_tf_idf_wiki[line.split("|")[0]]) == 3: #already got the idf  value for this term
            continue
        else:
            term_stem_tf_idf_wiki[line.split("|")[0]].append(line.split("|")[1])
    
     
    utils_linux.save_pickle("term_stem_tf_idf_wiki", term_stem_tf_idf_wiki)
    
    
     
def calc_Cross_Entropy_between_clm_vector_doc_vector():
    claim_represented_by_clmLM_vector_for_alpha = utils_linux.read_pickle("claim_represented_by_clmLM_vector_for_alpha")  
    claim_represented_by_docLM_vector_for_beta = utils_linux.read_pickle("claim_represented_by_docLM_vector_for_beta")    
    claim_doc_title_CE_result = {} #key is a claim and doc title, value is a list of lists:
                                    #for each alpha value (11 total), have a list of size 10 - for each beta value (10 values total)
                                    #in each entry - the CE result
    #for each alpha value, and for each beta value, calc the CE between
    for (clm,list_of_p_vectors) in claim_represented_by_clmLM_vector_for_alpha.items():
        for (doc_title,list_of_q_vectors) in claim_represented_by_docLM_vector_for_beta[clm]:
            for p_vector in list_of_p_vectors:
                curr_CE_res = []
                for q_vector in list_of_q_vectors:
                    curr_CE_res.append(utils_linux.calc_CE(p_vector, q_vector))
                if (clm,doc_title) in claim_doc_title_CE_result.keys():
                    claim_doc_title_CE_result[clm,doc_title].append(curr_CE_res)
                else:
                    claim_doc_title_CE_result[clm,doc_title] = curr_CE_res
def rank_docs():
    """
    for each alpha and beta value - rank the documents++ -
    .
    """

 
def retrieve_documents():
    """
    1. For every claim and document in the corpus we have (wiki/RT/IMDB/combined) -  calc the CE between them based on:
        1.1. the claim LM and the document LM -  tune alpha, beta
        1.2 for a given claim, alpha , beta value, have a list of the document, the p and q prob vectors, and the CE result 
    """
    claims_entity_doc_path = r"./claims_support_test.txt"
    wiki_docs_path = r"/lv_local/home/liorab/support/corpuses/wikipedia_movie_articles_0614"   
    claim_entity_list = [] #list of tuple -claim and entity from raw claims file
    
    claim_represented_by_clmLM_vector_for_alpha = {} #key is a claim, value is a list of 11 p vectors-one for each alpha value
    claim_represented_by_docLM_vector_for_beta = {}  #key is a claim, value is a list of 10 tuples -  doc title and the q vector -one for each beta value between 0 and 0.9
    
    claims_entity_doc = open(claims_entity_doc_path,"rb").read().strip()
    for i, line in enumerate (claims_entity_doc.split('\n')):
        claim_entity_list.append((line.split("|")[0],(line.split("|")[1])))
    
#     represent a claim with the claim_LM, 
#     claim_represented_by_clmLM_vector_for_alpha = represent_claim_with_claim_LM_prob_vector(claim_entity_list)
#     utils_linux.save_pickle("claim_represented_by_clmLM_vector_for_alpha", claim_represented_by_clmLM_vector_for_alpha)                                                      
#     claim_represented_by_docLM_vector_for_beta = represent_claim_with_doc_LM_prob_vector(claim_entity_list)
    represent_claim_with_doc_LM_prob_vector(claim_entity_list)
    utils_linux.save_pickle("claim_represented_by_docLM_vector_for_beta", claim_represented_by_docLM_vector_for_beta)   
    
    
def main():
#     collections = ["RT","wiki"]
#     for collection in collections:
#         calc_KL_div_between_clm_sen(collection)
#         calc_lexical_similarity_based_on_KL_div(collection)
#     create_relevance_clm_sen_score()
    create_term_stem_tf_idf_dict()
#     retrieve_documents()
    
if __name__ == '__main__':
    main() 