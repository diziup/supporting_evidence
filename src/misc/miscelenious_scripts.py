# -*- coding: utf-8 -*-
'''
@author: liorab
'''
import scipy.stats
import gensim
from my_utils import utils
from stemming.porter2 import stem
import sys
import csv
import pandas as pd
import string 
import os
import nltk
import math


def calc_pearson():
#     IMDB_raters=[20178,49160,30105,198401,74072,449888,214170]
    IMDB_raters_kappa_order=[96128,214170,29616,49160,449888,74072]
    kappa=[0.193,0.198,0.254,0.231,0.224,0.185]
#     support_percentage=[0,0.059,0.067,0.1,0.27,0.289,0.294]
    pearson_res=scipy.stats.pearsonr(IMDB_raters_kappa_order,kappa)
    print pearson_res
 
def test_gensim():
    word_rep_file_word2vec = r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity\word_rep\GoogleNews-vectors-negative300.bin.gz"
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format(word_rep_file_word2vec, binary=True)
    result = model.most_similar(['sad','good'],[] , 10)
    print result


def create_clm_and_sentences_file_for_BerkleyLM():
    """
    from all the clm_claimnum_clm_text_sen_text_dict files, for all the claim, create a file of sentences, so that 
    the Berkley LM will be able to read it and create a score for each movie star LM
    """
    claim_num_and_text=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict_pickle')
    for (clm_num,clm_text) in claim_num_and_text.items():
            curr_clm_and_sens=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\clm_'+clm_num+'_clm_text_sen_text_dict')
            with open("clm_"+str(clm_num)+"_all_sen_text.txt", 'wb') as clm_sen_text_f:
                for clm_or_sen in curr_clm_and_sens.values()[1:61]:
                    clm_sen_text_f.write(clm_or_sen+"\n")
                

def create_claim_dict_and_sen_dict(): 
    dir_path=r"C:\study\technion\MSc\Thesis\Y!\support_test\input_crowdflower_second_trial"
    claim_dict = {} #key is the claim number, value is the claim text
    for claim_reviews_file in os.listdir(dir_path):
        with open (dir_path+"\\"+claim_reviews_file, 'r') as f:
            claim_num=claim_reviews_file.split('supp_claim_')[1].split(".")[0]
            data = pd.read_csv(f)
            claim_text = data['claim']
            sen = data['sen']
            is_gold = data['_golden']
#             claim_dict = {} #key is the claim number, value is the claim text
            claim_sen_dict = {}#key is a tuple of claim_num, sen_num and value is the review text
#             with open ("clm_"+str(claim_num),'wb') as c_file:
#                         c_file.write(claim_text[1])
            for sen_num in range(0,len(data)):
                if is_gold[sen_num] !=1:
#                     with open ("clm_"+str(claim_num)+"_sen_"+str(sen_num),'wb') as rev_file:
#                         rev_file.write(sen[sen_num])
            #insert to dict - key is a claims and sentence pair, and val is currently empty. will be field by the sentiment similarity 
                    claim_dict[claim_num]=claim_text[1]
                    claim_sen_dict[(claim_num,sen_num)]=sen[sen_num] #for instance -  KEY: (4,1)  VAL: Film ,2004 ,Zombies ,Dawn of the Dead ,Individuals try and survive a zombie outbreak by securing a shopping mall
#                 self.save_pickle("claim_dict_pickle","claim_dict")
#                 self.save_pickle("claim_sen_dict_pickle","claim_sen_dict")
    utils.save_pickle("claim_dict_pickle", claim_dict)
    utils.save_pickle("claim_sen_dict_pickle", claim_sen_dict)
                
def test_dict_clm_sen_support_ranking_clm_sen_key_supp_score_value():
    clm_sen_support_ranking = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\model\clm_sen_support_ranking_clm_sen_key_supp_score_value")
    claim_num_and_text=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict_pickle')
    for (clm_num,clm_text) in claim_num_and_text.items():
        curr_clm_and_sens=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\clm_'+clm_num+'_clm_text_sen_text_dict') 
        del curr_clm_and_sens[0] 
        for sen in curr_clm_and_sens.values():
            try:
                print (clm_sen_support_ranking[(clm_text,sen)])
            
            except Exception as err: 
                    sys.stderr.write('problem in test_dict_clm_sen_support_ranking_clm_sen_key_supp_score_value with:'+ clm_text+ " and sen"+ sen)     
                    print err.args      
                    print err    
                
def test_stemming():
    sen="Stiller scores his funniest screen performance yet"    
    for word in sen.split():
        print stem(word)
        
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

def create_terms_file_from_corpuses_and_claims():
    """
    for the retrieval process relevance baseline, need to have the stems of all the the claims, the corpuses and the entities
    for each document in the corpuses (wiki/RT/IMDB?)
    """
    wiki_single_doc = open (r"C:\study\supporting_evidence\wikipedia_movie_articles_0614_with_body\wiki_single_doc_body", 'rb').read().strip()
    claim_and_entity_file =open ( r"C:\study\technion\MSc\Thesis\Y!\rawClaim_SW.txt", 'rb').read().strip()
#     claim_and_entity_file = open ( r"C:\study\technion\MSc\Thesis\Y!\toy_story_claim.txt", 'rb').read().strip()
    terms_set = set()
    
    exclude = set(string.punctuation)
#     remove_also = ["''"]
#     for s in remove_also:
#         exclude.add(s)
     
    for i, line in enumerate(claim_and_entity_file.split('\n')):
        all_sen_words = nltk.word_tokenize(line.split("|")[1].replace("-"," ").replace(","," "))
        all_sen_words = remove_and_split_on_non_ascii_chars(all_sen_words)
        for word in all_sen_words:
            if not word in exclude:
                terms_set.add(word.lower()) 
            else:
                print "exclude " +word
         
    for i, line in enumerate(wiki_single_doc.split('\n')):
#         line = line.split("|")[1]
        orig_line = line
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
        for word in all_sen_words:   
            if not word in exclude:  
                terms_set.add(word.lower()) 
                if word.lower() =="deliveryperson":
                    print orig_line
            else:
                print "exclude " +word
    #merge with wiki vocab from Indri
    wiki_vocab = open("wiki_vocab_body","rb").read().strip()
    for i, line in enumerate(wiki_vocab.split('\n')):
        word = line.split()[0] 
        terms_set.add(word.lower()) 
    
    with open("wiki_claims_entities_terms_body.txt", 'wb') as csvfile:
        terms_file = csv.writer(csvfile)
        for term in terms_set:
            terms_file.writerow([term])

def create_claim_and_sentences_dict_from_retrieval_files(setup, input_files_path):
        print "creating claim and sens dict..."
        
        claim_sentences_dict = {}
        for f in os.listdir(input_files_path): 
            curr_file = open(input_files_path+"\\"+f)
            sen = curr_file.read().strip() # score, sentence
            for i, line in enumerate(sen.split('\n')):                   
                if i%2 == 0: # a metadata line
                    data = line.split(' ')
                    curr_claim = int(data[0])
                else:
                    if "When they smooch in a greenhouse she says he's a good kisser" in line:
                        print "here" 
                    if claim_sentences_dict.has_key(curr_claim):
                        if not line in claim_sentences_dict[curr_claim]:
                            claim_sentences_dict[curr_claim].append(line)
                        else:
                            print line +" already in " +str(curr_claim)
                    else:
                        claim_sentences_dict[curr_claim] = [line]
        sum_sens = 0
        for claim_num in claim_sentences_dict.keys():
            print claim_num, len(claim_sentences_dict[claim_num])
            sum_sens += len(claim_sentences_dict[claim_num])
        print "total sen: ", sum_sens
        utils.save_pickle(setup + "_claim_sentences", claim_sentences_dict)    

def create_terms_file_from_claim_and_sentences_dict(setup):
    """
    12.01.15' - for the different baselines, create a dict from a retrieval file - 
    for use in the features module 
    """    
    print "creating terms from files..."
    terms_set = set()
    claim_sentences_dict = utils.read_pickle(setup + "_claim_sentences")
    claims_dict = utils.read_pickle("claim_dict")
    for (clm_num,sentences_list) in claim_sentences_dict.items():
        claim_text = claims_dict[str(clm_num)]
        curr_words = claim_text
        for sen in sentences_list:
            print "in clm",clm_num," sen" , sentences_list.index(sen)
            curr_words += " "+ sen
            all_sen_words = nltk.word_tokenize(curr_words.replace("-"," ").replace(","," "))
            exclude = set(string.punctuation)
            remove_also = ['"',"'","--",'"""',"===",".","'",]
            for s in remove_also:
                exclude.add(s)
            for word in all_sen_words:
#                 if "Terror,2004" in word:
                    if not word in exclude:
                        terms_set.add(word) 
                    else:
                        print "no " +word
                        
    with open(setup+"_clm_sen_terms.txt", 'wb') as csvfile:
        terms_file = csv.writer(csvfile)
        for term in terms_set:
            terms_file.writerow([term])

def calc_idf(vocab_filepath,source):
    """
    from the vocabulary data file of an index,
    create a dict of a term (key) and idf (value)
    """
    idf_dict = {}
    N = 0
    f = open(vocab_filepath,"r").read().strip() 
    for i, line in enumerate(f.split('\n')):
        if i == 0:
            N = float(line.split()[2])
        else:
            term = line.split()[0]
            df = float(line.split()[2])
            idf = float(math.log(N/df))
            idf_dict[term] = idf
    utils.save_pickle(source+"_term_idf", idf_dict)
    
def create_terms_file_from_claim_and_sentences():
    """
    for the semantic similarity using the max sim between every word, 
    need to calculate the idf of each term.
    The resulting file is the input to the c++ program in Indri - calcIdfForClaimSentenceTerms
    """
#     claim_num_and_text=utils_linux.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict_pickle')
#    
#     for (clm_num,clm_text) in claim_num_and_text.items():
    terms_set = set()
    clm_sen_support_ranking_RT_sorted = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\clm_sen_support_ranking_RT_sorted")
    clm_sen_support_ranking_wiki_sorted = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\clm_sen_support_ranking_wiki_sorted")
    clm_sen_support_ranking_list = [clm_sen_support_ranking_RT_sorted,clm_sen_support_ranking_wiki_sorted]
    for curr_dict in clm_sen_support_ranking_list:
        for (clm,sen) in curr_dict.keys():
            curr_words = clm + " "+ sen
    #         words_dict  = utils_linux.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\clm_'+clm_num+'_tokenized_sen_VSM')
            all_sen_words = nltk.word_tokenize(curr_words.replace("-"," ").replace(","," "))
            
#             all_sen_words = words_dict.values()
            exclude = set(string.punctuation)
            remove_also = ['"',"'","--",'"""',"===",".","'",]
            for s in remove_also:
                exclude.add(s)
            for word in all_sen_words:
#                 if "Terror,2004" in word:
                    if not word in exclude:
                        terms_set.add(word) 
                    else:
                        print "no " +word
                        
    with open("clm_sen_terms.txt", 'wb') as csvfile:
        terms_file = csv.writer(csvfile)
        for term in terms_set:
            terms_file.writerow([term])
            
#     terms_list = []
#     terms_set = set()
#     for (clm_num,clm_text) in claim_num_and_text.items():
#         curr_clm_and_sens=utils_linux.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\clm_'+clm_num+'_clm_text_sen_text_dict') 
#         for curr_clm_or_sen in curr_clm_and_sens.values():
#             exclude = set(string.punctuation)
#             remove_also = ['"',"'","--"]
#             for s in remove_also:
#                 exclude.add(s)
#             s = ''.join(ch for ch in curr_clm_or_sen if ch not in exclude)
# #             terms_list.extend(curr_clm_or_sen.split())
#             for w in s.split():
#                 terms_set.add(w)
#     with open("clm_sen_terms_3.txt", 'wb') as csvfile:
#         terms_file = csv.writer(csvfile)
#         for term in terms_set:
#             terms_file.writerow([term])
     
def create_terms_files_from_vocab_files():
    """
    from the vocablary files - of the RT and wikipedia indexed, create a file of just the terms- one per line (without the document id...)s
    this will be the input to findIdf_liora.cpp in Indri
    """
    wiki_vocab_file = open(r"C:\study\technion\MSc\Thesis\Y!\support_test\claim_sentence_similarity_VSM\vocab_wiki",'rb').read().strip()
    RT_vocab_file = open(r"C:\study\technion\MSc\Thesis\Y!\support_test\claim_sentence_similarity_VSM\vocab_RT",'rb').read().strip() 
    wiki_terms_file = open(r"C:\study\technion\MSc\Thesis\Y!\support_test\claim_sentence_similarity_VSM\wiki_terms_file.txt",'wb')
    RT_terms_file = open(r"C:\study\technion\MSc\Thesis\Y!\support_test\claim_sentence_similarity_VSM\RTs_terms_file.txt",'wb')
    
    for i, line in enumerate(wiki_vocab_file.split('\n')):
        wiki_terms_file.write(line.split()[0]+"\n")
    for i, line in enumerate(RT_vocab_file.split('\n')):
        RT_terms_file.write(line.split()[0]+"\n") 
    
def read_idf_count(wiki_RT_unified):
    """
    From the findIdf_liora in Indri, read the file of terms and their idf, according to whether if its the sepearte collections or the unified one
    """         
    term_idf_files_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\claim_sentence_similarity_VSM"
    term_idf_dict= {} #key is a term, val is the idf
    
    idf_file = open(term_idf_files_path+r"\idf_res_"+wiki_RT_unified+".txt", 'rb').read().strip()
    for i, line in enumerate(idf_file.split('\n')):
        term_idf_dict[line.split("|")[0]] = float(line.split("|")[1])

    utils.save_pickle(wiki_RT_unified+"_term_idf_dict", term_idf_dict) 
 
def concatinate_documents_to_single_doc():
    """to try and remove un-reguilar sentences from the wikipedia dump documents
    """ 
#     docs_path = r"C:\supporting_evidence\wikipedia_movie_articles_0614"
    docs_path = r"C:\study\supporting_evidence\wikipedia_movie_articles_0614_with_body"
    wiki_single_doc = open (docs_path + r"\\wiki_single_doc_body", 'wb')
    lines_to_write = ""
    try:
        for filename in os.listdir(docs_path):
            all_lines_for_single_doc = ""
            """
            11.10.14 update - add the title field to the lines to write
            """
            skip = ['<DOC>','<DOCNO>','<TEXT>','<senti>','</DOC>','</DOCNO>','</TEXT>','</senti>'] 
            try:
                with open(docs_path+"\\"+filename, 'r') as f:
                    doc_lines = f.read().strip()
                    for i, line in enumerate(doc_lines.split('\n')):
                        to_remove_line = 0
                        for s in skip:
                            if s in line or line =="":
                                to_remove_line = 1
                                continue
                        try:
                            if to_remove_line == 0: #sentence- get it but remove the <s>, </s>
                                if "<s>" in line:
                                    lines_to_write += line.split('<s>')[1].split('</s>')[0] +"\n"
                                elif "<title>" in line:
                                    lines_to_write += line.split('<title>')[1].split('</title>')[0] +"\n"
                        except Exception as err: 
                            sys.stderr.write('problem in :'+line)     
                            print err.args      
                            print err
                    print "finished " +filename
            except Exception as err: 
                            sys.stderr.write('problem in :'+filename)     
                            print err.args      
                            print err
    except Exception as err: 
                            sys.stderr.write('problem in :'+filename)     
                            print err.args      
                            print err
    wiki_single_doc.write(lines_to_write)  
    wiki_single_doc.close()            
#     with open("wiki_single_doc.txt",'wb') as csvfile:
#         f=csv.writer(csvfile)
#         f.writerow([all_lines_for_single_doc])

def term_freq_dict_from_file():
    """
    from the tf of the terms in the claims and sentences, alculated using the script findTf_liora in Indri
    Create a dict for use in the relevance baseline of the relevance using KL div
    """
   
    tf_wiki_dict = {}
    tf_RT_dict = {}
    
    wiki_f = open(r"C:\study\technion\MSc\Thesis\Y!\support_test\relevance_baselines\tf_terms_wiki.txt",'rb').read().strip()
    for i, line in enumerate(wiki_f.split('\n')):
        tf_wiki_dict[line.split("|")[0]] = float(line.split("|")[1])

    RT_f = open(r"C:\study\technion\MSc\Thesis\Y!\support_test\relevance_baselines\tf_terms_RT.txt",'rb').read().strip() 
    for i, line in enumerate(RT_f.split('\n')):
        tf_RT_dict[line.split("|")[0]] = float(line.split("|")[1])
    
    utils.save_pickle("tf_RT_dict", tf_RT_dict)
    utils.save_pickle("tf_wiki_dict", tf_wiki_dict)
  
def add_body_to_doc_datasets():
    """
    according to the update in 10.14
    """  
    wiki_path = r"C:\study\supporting_evidence\wikipedia_movie_articles_0614"
    new_wiki_with_body = r"C:\study\supporting_evidence\wikipedia_movie_articles_0614_with_body"
    errors = open("errog.log","wb")
    try:
        for filename in os.listdir(wiki_path):
            try:
                with open(wiki_path+"\\"+filename, 'r') as f:
#                     print "in "+filename
                    try:
                        if filename == "Aankhon Ke Saamne.txt":
                            print "here"
                        new_doc_str = ""
                        line = f.readline()
#                         if "<title>" in line:
                            
                        while not "<s>" in line and "</senti>" not in line :
                            if "<title>" in line:
                                new_doc_str += line.split("\n")[0] +"</title>\n"
                                line = f.readline() 
                            if "<senti>" in line or "</senti>" in line:
                                new_doc_str += "<body>\n"
                                line = f.readline() 
                            new_doc_str += line 
                            line = f.readline()
                        
                        while "<s>" in line:
                            new_doc_str += line 
                            line = f.readline()
                        new_doc_str += "</body>\n"
                        line = f.readline()
                        if "</title>" in line:
                            line = f.readline()
                        while line !="":
                            new_doc_str += line 
                            line = f.readline()
                    except Exception as err: 
                        sys.stderr.write('problem in add_body_to_doc_datasets: ' + filename)
                        print err.args      
                        print err
                        errors.write(filename+"\n")   
            except Exception as err: 
                sys.stderr.write('problem in add_body_to_doc_datasets: ' + filename)
                print err.args      
                print err
                errors.write(filename+"\n")          
            f.close()
            try:
                new_wiki_doc = open(new_wiki_with_body+"\\"+filename, "wb")
                new_wiki_doc.write(new_doc_str)
                new_wiki_doc.close()
                print "finished "+ filename    
            except Exception as err: 
                sys.stderr.write('problem in add_body_to_doc_datasets')     
                print err.args      
                print err
                errors.write(filename+"\n")      
           
            
    except Exception as err: 
        sys.stderr.write('problem in add_body_to_doc_datasets')     
        print err.args      
        print err   
        errors.write(filename+"\n")      
    errors.close()
               
def map_doc_title_docno():
    """
    for the sentence retrivela using my ra
    create a map  - key is document title, value is the docno (wikipedia/RT/IMDB..collections)
    """
    
    wiki_docs_path =r"C:\study\supporting_evidence\wikipedia_movie_articles_0614"
    doc_title_docno_dict = {} 
    for filename in os.listdir(wiki_docs_path):
        try:
            if ".txt" in filename:
                title = filename.split(".txt")[0]
                with open(wiki_docs_path+"\\"+filename, 'r') as f:
                        doc_lines=f.read().strip()
                        for i, line in enumerate(doc_lines.split('\n')):
                            if "<DOCNO>" in line:
                                docno = line.split("<DOCNO>")[1].split("</DOCNO>")[0]
                                doc_title_docno_dict[title] = docno
        except Exception as err: 
            sys.stderr.write('problem in map_doc_title_docno')     
            print err.args      
            print err                       
    utils.save_pickle("doc_title_docno_dict",doc_title_docno_dict)

def test_asci():
    
    exclude = set(string.punctuation)
    remove_also = ["''"]
    remove_asci = [u"—",u" –",u"”",u" —",u"’",u"“",u"‘"]
    terms_set = set()
    for s in remove_also:
        exclude.add(s)
#     for s in remove_asci:
#         exclude.add(s.encode("utf-8"))
    
    f = open("Mexico Trilogy.txt","rb").read().strip()
    for i, line in enumerate(f.split("\n")):
#         line.replace(re.sub('[ -~]', '', line)," ")
        all_sen_words = nltk.word_tokenize(line.replace("-"," ").replace(","," "))
#         all_sen_words = [word.decode("utf-8","ignore") for word in all_sen_words ]
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
#                 if any(s in word for s in exclude):
                for s in remove_asci:
                    if s in word:
                        print "out"
                        corrected_words.append((word,''.join(c for c in word if c not in exclude)))
            except Exception as err: 
                    sys.stderr.write('problem in exclude: ' +s.encode("utf-8"))     
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
        for word in all_sen_words:
#             if word.count("/") == 1:
#                     terms_set.add(word.split("/")[0].lower())
#                     terms_set.add(word.split("/")[1].lower()) 
#                     continue    
            if not word in exclude:  
                terms_set.add(word.lower()) 
            else:
                print "exclude " +word
    

    with open("bla.txt", 'wb') as csvfile:
        terms_file = csv.writer(csvfile)
        for term in terms_set:
            terms_file.writerow([term])   


def find_wiki_problematioc_docs():
    wiki_docs_path = r"C:\supporting_evidence\wikipedia_movie_articles_0614"
    problematic_doc = open("problematic_docs.txt","wb")
    for filename in os.listdir(wiki_docs_path):
            try:
                if ".txt" in filename:
                    curr_doc_title = filename.split(".txt")[0]
                    with open(wiki_docs_path+"\\"+filename, 'r') as f:
                        print "opened" +curr_doc_title
            except Exception as err: 
                sys.stderr.write('problem in represent_claim_with_doc_LM_prob_vector in doc ' + curr_doc_title)
                problematic_doc.write(curr_doc_title +"\n")     
                print err.args      
                print err
    problematic_doc.close()

def main():
#     calc_pearson()
#     create_terms_file_from_claim_and_sentences()
#     create_terms_files_from_vocab_files()
#     read_idf_count("RT")
#     create_claim_dict_and_sen_dict()
#     concatinate_documents_to_single_doc()
#     term_freq_dict_from_file()
#     create_terms_file_from_corpuses_and_claims()
#     find_wiki_problematioc_docs()
#     add_body_to_doc_datasets()
#     map_doc_title_docno()
#     create_claim_and_sentences_dict_from_retrieval_files("support_basline",r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\claimEntity_sen_output")
#     create_terms_file_from_claim_and_sentences_dict("support_basline")
    d = utils.read_pickle("support_basline_claim_sentences")
    print "lior"
#         calc_idf(r"C:\study\technion\MSc\Thesis\Y!\support_test\data\wikiWithBody_vocab","wikiWithBody")
if __name__ == '__main__':
    main()