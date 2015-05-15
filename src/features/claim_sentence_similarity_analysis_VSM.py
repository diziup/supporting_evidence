'''
29.07.2014
Use VSM - using turian and word2vec to measure the similarity between the claim and sentence pair
create a vector for each claim and sentence,
and measure the similarity by cosine.
check if there is a connection - accuracy  to the supportiveness annotation
'''

import numpy as np
import sys
import parse_tree
from my_utils import utils_linux
import gensim
import nltk
import csv
from nltk.corpus import stopwords
import collections
import os
import string
import pickle
from munkres import Munkres

class clm_sen_similarity_VSM():
    dim = 300
    word_rep_file_tur = r"Tur_neu_dim"+str(dim)+".txt"
    word_rep_file_word2vec = "wikipediaModel_0315.bin"#r"GoogleNews-vectors-negative300.bin.gz"
    # 15.5.'15 update - add the entity vector, using the pre-trained free-base entity vectors. 
    entity_rep_file_word2vec = "freebase-vectors-skipgram1000-en.bin.gz"
    word_rep_dict = {}
    representation = "word2vec"
    claim_sens_files_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\stanford_contituent_parser\input"
    claims_parser_result_file = r""
    sen_parser_result_file = r""
    parser_res_files_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\stanford_contituent_parser\output"
    tokenized_clm_and_sen_dict = {}
    nodes_cnt = 0
    clm_sen_cosine_sim_res = {} #key is a claim and sentence text, and val is the cosine res between their vectors
    clm_text_sen_text_dict = {} #key is number, value is claim text or sentence text
    all_clm_sen_sim_res = {} #across all claims and sentences, and not for a particular pair, for the analyse_sentence_support module
    compositional_func = "max_words_similarity"
    wiki_rt_setup = "separate" #whether the collection setup is separated collections, or a unified collection of RT and wiki
    max_sim_words_clm_and_sen = {} #key is clm, sen and value is the word and sim score that is the max
    features = "VSM"
    input_data = "dict" #or files... dict is when the claim and sentences are processed already, file
            # is when they are not...
    setup = "support_baseline"
    
    
    def __init__(self,setup):
        self.output_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\clm_sentence_sim_VSM"
        self.clm_sen_input_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\stanford_contituent_parser\input"
        self.input_data = "dict"
        self.setup = setup
        self.claim_sentence_list_dict = {} #key is a claim_num, value is a list of its sentences
        self.claim_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,54,55,57,58,59,60,61,66]
         
    def create_clm_sim_dict(self,input_file):
        """
        create a dict of key is number and value is claim or its corresponding sentence
        """   
        file_name='clm_'+input_file.split('_')[1]
        clm_sen_file = open(self.claim_sens_files_path+"\\"+input_file,'r')
        clm_sen_doc = clm_sen_file.read().strip() 
        for i, line in enumerate(clm_sen_doc.splitlines()):
            self.clm_text_sen_text_dict[i]=line
        utils_linux.save_pickle(file_name+"_clm_text_sen_text_dict",self.clm_text_sen_text_dict)    
     
    def save_pickle(self,file_name):
        with open(file_name, 'wb') as handle:
            if file_name == "word_rep_"+str(self.dim):
                pickle.dump(self.word_rep_dict, handle)
            elif "clm_text_sen_text_dict" in file_name:
                pickle.dump(self.clm_text_sen_text_dict, handle) 
            elif '_tokenized_sen_' in file_name:
                pickle.dump(self.tokenized_clm_and_sen_dict,handle)          
            elif file_name == "all_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim):
                pickle.dump(self.all_clm_sen_sim_res,handle) 
            elif "clm_sen_cosine_sim_res" in file_name:
                pickle.dump(self.clm_sen_cosine_sim_res,handle)   
                    
    def read_pickle(self,file_name):
        if "tokenized_sen" in file_name:
            with open(file_name, 'rb') as handle:
                self.tokenized_clm_and_sen_dict = pickle.loads(handle.read())                                 
        elif file_name == "word_rep_"+str(self.dim):
            with open(file_name, 'rb') as handle:
                self.word_rep_dict = pickle.loads(handle.read())
        elif "clm_text_sen_text_dict" in file_name:
            with open(file_name, 'rb') as handle:
                self.clm_text_sen_text_dict = pickle.loads(handle.read())
        elif file_name == "all_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim):
            with open(file_name, 'rb') as handle:
                self.all_clm_sen_sim_res=pickle.loads(handle.read())
        elif "_clm_sen_cosine_sim_res_" in file_name:
            with open(file_name, 'rb') as handle:
                self.clm_sen_cosine_sim_res = pickle.loads(handle.read())
       
    def read_word_rep_file(self):
        try:
            rep_file = open(self.word_rep_file_tur,'r')
            rep_doc = rep_file.read().strip()
#             rep_doc = rep_file.read()
            for line in (rep_doc.split('\n')):
                self.word_rep_dict[line.split()[0]]=line.split()[1:self.dim+1] 
            utils_linux.save_pickle("word_rep_"+str(self.dim),self.word_rep_dict)
            print "finished read_word_rep_file"
        except Exception as err: 
            sys.stderr.write('problem in read_word_rep_file:')     
            print err.args      
            print err

    def  read_sen_word2vec(self):
        #read the available words from google news
        print "reading word2vec"
        self.model_words = gensim.models.word2vec.Word2Vec.load_word2vec_format(self.word_rep_file_word2vec, binary=True)
        self.model_entity = gensim.models.word2vec.Word2Vec.load_word2vec_format(self.entity_rep_file_word2vec, binary=False)
        print "finished reading word2vec"
    
    def tokenize_sentences_from_dict(self):
        try:
            """
            08.01.15 - tokenize sentences when the input sentences are already in a claim->sentence list dict
            24.01.15 - tokenize per claim, save in separate dict for each claim for the runs on the servers.
            """
            self.claim_sentence_list_dict = utils_linux.read_pickle(self.setup+"_claim_sentences")
            print "claim_sentence_list_dict len: " +str(len(self.claim_sentence_list_dict.keys()))
            claim_dict = utils_linux.read_pickle("claim_dict")
            stopWords = stopwords.words('english')
            exclude = set(string.punctuation)
            exclude.add('--')          
            
            #for (clm,sentences_list) in self.claim_sentence_list_dict.items():
            for clm in self.claim_list:
                sen_cnt = 0
                sentences_list = self.claim_sentence_list_dict[clm]
#                     for i, line_with_query in enumerate(sen_doc.splitlines()):
                print "\t tokenizing " +str(len(sentences_list)) +" from dict in clm ", clm
                self.tokenized_clm_and_sen_dict = {}
                clm_text = claim_dict[str(clm)]
                #15.05.'15 update- tokenize the claim with the claim's entity
#                         line=line_with_query.split("|")[1]
                # Split up expression with "-"!! 27.09 update!!!!
#                 line.replace("-"," ")
                #tokenize the claim first and insert as the first element to the tokenized dict
                tempWords = nltk.word_tokenize(clm_text.replace("-"," "))
                if self.features is "BOW":
                        self.tokenized_clm_and_sen_dict[sen_cnt] = [w for w in tempWords if (w.lower() not in stopWords and len(w) > 1 and not w.isdigit())  ]
                elif self.features is "VSM":
                    self.tokenized_clm_and_sen_dict[sen_cnt] = [self.check_word_in_rep_dict(w,self.representation) for w in tempWords if ( self.check_word_in_rep_dict(w,self.representation) is not "no") and  w not in exclude ]
                sen_cnt += 1
                
                for sen in sentences_list:
                    try:
                        tempWords = nltk.word_tokenize(sen.replace("-"," "))
                    except UnicodeError:
                        print "handling unicode error"
                        words_unicode = unicode(sen.replace("-"," "), errors='ignore')
                        tempWords = nltk.word_tokenize(words_unicode)
                        tempWords = [word.encode("utf-8") for word in tempWords] 
                    if self.features is "BOW":
                        self.tokenized_clm_and_sen_dict[sen_cnt] = [w for w in tempWords if (w.lower() not in stopWords and len(w) > 1 and not w.isdigit())  ]
                    elif self.features is "VSM":
                        self.tokenized_clm_and_sen_dict[sen_cnt] = [self.check_word_in_rep_dict(w,self.representation) for w in tempWords if ( self.check_word_in_rep_dict(w,self.representation) is not "no") and  w not in exclude ]
                    if len(self.tokenized_clm_and_sen_dict[sen_cnt]) == 0:
                        print "\t", sen_cnt , "is empty"
                    sen_cnt += 1
                
                utils_linux.save_pickle(self.setup+'_clm_'+str(clm)+'_tokenized_clm_and_sen_dict_'+self.features,self.tokenized_clm_and_sen_dict)
            
#             tokenized_sen=csv.writer(open(file_name+'_tokenized_sen_'+self.features+".csv", "wb"))
#             for sen in self.tokenized_clm_and_sen_dict.values():
#                 tokenized_sen.writerow(sen)
                           
        except Exception as err: 
            sys.stderr.write('problem in tokenize sentences:')     
            print err.args      
            print err 
    
    def tokenize_sentences_from_files(self, input_file):
        try:
            """
            tokenize sentences to remove 's and stuff like that.
            27.09.14 update - add another split according to "-" so that expression like eager-to-please, that do not appear in the word2vec vocab, 
            will not get removed from the sen
            """
            
            sen_file = open(self.claim_sens_files_path+"\\"+input_file,'r')
            sen_doc = sen_file.read().strip() 
            self.tokenized_clm_and_sen_dict = {} 
            file_name = 'clm_'+input_file.split('_')[1]
            stopWords = stopwords.words('english')
            exclude = set(string.punctuation)
            exclude.add('--')
            print ('in '+file_name)
#                     for i, line_with_query in enumerate(sen_doc.split(r'(?:\n|\r)')):
            for i, line in enumerate(sen_doc.splitlines()):
#                     for i, line_with_query in enumerate(sen_doc.splitlines()):
                print "in line ", i
#                         line=line_with_query.split("|")[1]
                # Split up expression with "-"!! 27.09 update!!!!
#                 line.replace("-"," ")
                tempWords = nltk.word_tokenize(line.replace("-"," "))
                
                if self.features is "BOW":
                    self.tokenized_clm_and_sen_dict[i] = [w for w in tempWords if (w.lower() not in stopWords and len(w) > 1 and not w.isdigit())  ]
                elif self.features is "VSM":
                    line = line.split("|")[0]
                    self.tokenized_clm_and_sen_dict[i] = [self.check_word_in_rep_dict(w,self.representation) for w in tempWords if ( self.check_word_in_rep_dict(w,self.representation) is not "no") and  w not in exclude ]
                    """08.06 liora update - tokenized sen with stop words, bacause we may use a parse tree and this is necessary for this.
                    """
            utils_linux.save_pickle(file_name+'_tokenized_sen_'+self.features,self.tokenized_clm_and_sen_dict)
            
            tokenized_sen = csv.writer(open(file_name+'_tokenized_sen_'+self.features+".csv", "wb"))
            for sen in self.tokenized_clm_and_sen_dict.values():
                tokenized_sen.writerow(sen)
                           
        except Exception as err: 
            sys.stderr.write('problem in tokenize sentences:')     
            print err.args      
            print err  

    def check_word_in_rep_dict(self,word,representation):
        if representation is "turian":
            if word in self.word_rep_dict.keys():
                return word
            elif word.lower() in self.word_rep_dict.keys():
                return word.lower()
            elif word.upper() in self.word_rep_dict.keys():
                return word.upper()
            else: 
                return "no"
        elif representation is "word2vec":
            if self.model_words.__contains__(word):
                return word
            if self.model_words.__contains__(word.lower()):
                return word.lower()
            else: 
                return "no"
    
    
    def represent_clm_and_sen_as_VSM(self,input_file):
        print "enetered represent_clm_and_sen_as_VSM"
        try:
            """
            1. for every sentence in tokenized_sentences, represent it as one vector, depending on the model - additive (summing up vectors),
            averaging, weighted additive, multiplicative. A more advanced option is to use a parse tree, and according to this sum up word vectors. 
            """
            clm_sen_VSM_matrix = np.zeros((len(self.tokenized_clm_and_sen_dict),self.dim))
            if self.compositional_func is "additive":
                for (key,sen) in self.tokenized_clm_and_sen_dict.items():
                    curr_clm_sen_matrix = np.zeros((1,self.dim))
                    for word in sen:
                        if self.representation is "turian":
                            curr_clm_sen_matrix = np.vstack((curr_clm_sen_matrix, self.word_rep_dict[word]))
                        elif self.representation is "word2vec":
                            if self.model_words.__contains__(word):
                                curr_clm_sen_matrix = np.vstack((curr_clm_sen_matrix, self.model_words.__getitem__(word)))
                    curr_clm_sen_matrix = np.delete(curr_clm_sen_matrix,(0),axis=0) #delete the first zero row
                    curr_clm_sen_matrix = curr_clm_sen_matrix.astype(np.float)
                    clm_sen_VSM_matrix[self.tokenized_clm_and_sen_dict.keys().index(key)] = np.sum(curr_clm_sen_matrix,0)/len(sen) #sum up all the word vectors
            if self.compositional_func is "parse_tree":
                    try:
                        parse_tree_nodes = parse_tree.main(self.parser_res_files_path+"\\"+input_file)
                        sen_num = 0
                        for r_node in parse_tree_nodes: #every r_node =root node is a sentence, and put the result in the clm_sen_VSM_matrix 
                            self.nodes_cnt = 0
                            sen_vector = self.calc_sentence_vector_based_on_parse_tree(r_node)
                            clm_sen_VSM_matrix[sen_num] = sen_vector[0]/float(self.nodes_cnt)#save the -dim- dimensional vector 
                            sen_num += 1 
                    
                    except Exception as err: 
                        sys.stderr.write('problem in represent_sen_as_NN parse_tree in filename : '+input_file+ ' and sen:' +sen_num)     
                        print err.args      
                        print err
            file_name='clm_'+input_file.split("_")[1]
            np.save(file_name+"_clm_sen_VSM_matrix_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim)+".npy", clm_sen_VSM_matrix)
            
        except Exception as err: 
                        sys.stderr.write('problem in represent_clm_and_sen_as_VSM:')     
                        print err.args      
                        print err  
    
    def calc_sentence_vector_based_on_parse_tree(self,parse_tree_node): 
        #termination condition
        try:
            if parse_tree_node.word != "":
                #<find the word vector>
                if self.representation is "turian":
                    if not self.check_word_in_rep_dict(parse_tree_node.word,self.representation) =="no": 
                        word_vector = self.word_rep_dict[parse_tree_node.word]
                        word_vector = map(float, word_vector)
                    else:
                        word_vector=np.zeros((1,self.dim))
                elif self.representation is "word2vec":
                    if self.model_words.__contains__(parse_tree_node.word):
                        word_vector = self.model_words.__getitem__(parse_tree_node.word)
                        word_vector = map(float, word_vector)
                    else:
                        print parse_tree_node.word +"not in google words"
                        word_vector=np.zeros((1,self.dim))
                self.nodes_cnt+=1
                return word_vector
                   
            vector_list = []
            for child in parse_tree_node.children:
                res=self.calc_sentence_vector_based_on_parse_tree(child)
                if res is not None:
                    vector_list.append(res)
                else:
                    break
                    
            if len(vector_list) > 1: 
                idx=0
                new_node = ""
                           
                curr_vector = vector_list[idx] #the current word from the leaves
                curr_word_mat = np.zeros((1,self.dim))
                curr_word_mat = np.vstack((curr_word_mat,curr_vector))
                curr_word_mat = np.delete(curr_word_mat,(0),axis=0) #delete the first zero row
                curr_word_mat = curr_word_mat.astype(np.float)
                if parse_tree_node.children[idx].label in self.POS_weights.keys():
                    curr_word_mat = self.POS_weights[parse_tree_node.children[idx].label]*curr_word_mat
                else:
                    curr_word_mat = 1*curr_word_mat
                
                next_vector=vector_list[idx+1] #next word from the leaf
                next_word_mat=np.zeros((1,self.dim))
                next_word_mat=np.vstack((next_word_mat,next_vector))
                next_word_mat=np.delete(next_word_mat,(0),axis=0) #delete the first zero row
                next_word_mat = next_word_mat.astype(np.float)
                if parse_tree_node.children[idx+1].label in self.POS_weights.keys():
                    next_word_mat = self.POS_weights[parse_tree_node.children[idx+1].label]*next_word_mat
                else:
                    next_word_mat = 1*next_word_mat
                
                new_resulting_vector=np.add(curr_word_mat, next_word_mat)
             
                new_node=new_resulting_vector
                next_idx=idx+1
                while next_idx+1 < len(vector_list):
                    next_vector=vector_list[next_idx+1]
                    next_word_mat = np.zeros((1,self.dim))
                    next_word_mat = np.vstack((next_word_mat,next_vector))
                    next_word_mat = np.delete(next_word_mat,(0),axis=0)
                    next_word_mat = next_word_mat.astype(np.float)
                    if parse_tree_node.children[next_idx+1].label in self.POS_weights.keys():
                        next_word_mat = self.POS_weights[parse_tree_node.children[next_idx+1].label]*next_word_mat
                    else:
                        next_word_mat=1*next_word_mat
                    new_resulting_vector=np.add(new_node, next_word_mat)
    #                 new_node_list.append((new_node_list[idx][0],next_wlv[0],new_node_list[idx][2]+self.POS_weights[next_wlv[1]]*next_wlv[2]))
                    new_node=new_resulting_vector
                    next_idx+=1
                return new_node
            
            else: #only one child but still need to take the POS tag of the upper node..
                    new_node=(vector_list[0]) #return the word and the vector, without the label 
                    return new_node
        except Exception as err:
                sys.stderr.write('problem in calc_sentence_vector_based_on_parse_tree in child:'+parse_tree_node.word +" , "+parse_tree_node.label)     
                print err.args      
                print err
    
    def calc_clm_sen_similarity(self,input_file_name):
        """
            for each clm_sen_matrix_VSM, calculate the cosine sim between the claim and each sentence
            """
        print "enetered calc_clm_sen_similarity"  
        try:   
            self.clm_sen_cosine_sim_res={}
            
            clm_sen_VSM_array=np.load(input_file_name+"_clm_sen_VSM_matrix_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim)+".npy")
            claim_vector=clm_sen_VSM_array[0]
            rows,col=clm_sen_VSM_array.shape
            for sen_row in range(1,rows):
                curr_clm_sen_sim = utils_linux.cosine_measure(claim_vector, clm_sen_VSM_array[sen_row])
                self.clm_sen_cosine_sim_res[(self.clm_text_sen_text_dict[0],self.clm_text_sen_text_dict[sen_row])]=curr_clm_sen_sim
#                 self.all_clm_sen_sim_res[(self.clm_text_sen_text_dict[0],self.clm_text_sen_text_dict[sen_row])]=curr_clm_sen_sim
            utils_linux.save_pickle(input_file_name+"_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim),self.clm_sen_cosine_sim_res)
        except Exception as err:
            sys.stderr.write('problem in calc_clm_sen_similarity:')     
            print err.args      
            print err 
   
    def calc_clm_sen_sim_based_on_max_sim_between_word(self,input_file_name):
        """
        According to Idan's proposal:
        compute the similarity between every word in the claim and every word in the sentence, 
        choose the most similar word for each match.
        #update 22.09 -  add the idf of words -  accoridng to wehther the wiki_rt_setup is in the unified collection or if it is in the seprate wiki and RT collections
        if the wiki_rt_setup is separate, i need to know if the current sentence is from the wiki or RT.
        This can be determined according to the index in the tokenized dict
        """       
        #read the idf dict
        print "calculating max sim for claim", input_file_name
        
        curr_claim_num = int(input_file_name.split("_")[1])
        if self.wiki_rt_setup == "separate":
#             RT_term_idf_dict = utils_linux.read_pickle("RT_term_idf_dict")
#             wiki_term_idf_dict = utils_linux.read_pickle("wiki_term_idf_dict")
            wiki_term_idf_dict = utils_linux.read_pickle("wikiWithBody_term_idf")
        elif self.wiki_rt_setup == "unified":
            unified_term_idf_dict = utils_linux.read_pickle("unified_term_idf_dict")
            
        curr_claim_words = self.tokenized_clm_and_sen_dict.values()[0]
        curr_claim = ' '. join(curr_claim_words)
        self.clm_sen_cosine_sim_res = {}
        print "==== in claim: "+ curr_claim
        curr_clm_word_vec = np.zeros((1,self.dim))
        try:
            print str(len(self.tokenized_clm_and_sen_dict.keys()))+" sentences"
            for sen_idx in range(1,len(self.tokenized_clm_and_sen_dict.keys())): #the even numbers are the RT sentence, the odd -  from wiki
                #go over the clm's sen
                #if isinstance((sen_idx/1000), int):
                #    print "in sen index:" +str(sen_idx) 
                curr_sen_word_vec = np.zeros((1,self.dim))
                max_sim = 0
                curr_clm_sen_sim = 0
                curr_sen_words = self.tokenized_clm_and_sen_dict.values()[sen_idx]
                if len(curr_sen_words) > 0:
                    curr_sen = ' '. join(curr_sen_words)
#                     print "in sen words: "+curr_sen
                    not_found_words = []
                    ####go over claims words
                    for claim_word in curr_claim_words:  
                        if self.representation is "turian":
                            curr_clm_word_vec = self.word_rep_dict[claim_word]
                        elif self.representation is "word2vec" :
                            if self.model_words.__contains__(claim_word):
                                curr_clm_word_vec = self.model_words.__getitem__(claim_word)
                               
                            else:
                                not_found_words.append(claim_word)
                                print "did not find in word2vec claim word: "+claim_word
                                continue
                        ####go over sen words
                        for sen_word in curr_sen_words:
                            if self.representation is "turian":
                                curr_sen_word_vec = self.word_rep_dict[sen_word]
                            elif self.representation is "word2vec":
                                if self.model_words.__contains__(sen_word):
                                    curr_sen_word_vec = self.model_words.__getitem__(sen_word)
                                else:
                                    not_found_words.append(sen_word)
                                    continue
                            if sen_word == claim_word or sen_word.lower() == claim_word.lower():
                                curr_words_sim = 0
                            else:
                                curr_words_sim = utils_linux.cosine_measure(curr_clm_word_vec, curr_sen_word_vec)
                            
                            try:
        #                         if sen_idx%2 == 1: #odd sen index - so it is from wiki
                                weighted_with_idf_curr_words_sim = 0
                                if wiki_term_idf_dict.has_key(sen_word) and wiki_term_idf_dict.has_key(claim_word):
                                    weighted_with_idf_curr_words_sim = curr_words_sim*wiki_term_idf_dict[sen_word]*wiki_term_idf_dict[claim_word]
                                elif wiki_term_idf_dict.has_key(sen_word.lower()) and wiki_term_idf_dict.has_key(claim_word.lower()):
                                    weighted_with_idf_curr_words_sim = curr_words_sim*wiki_term_idf_dict[sen_word.lower()]*wiki_term_idf_dict[claim_word.lower()]
                                        
        #                         else:
        #                             weighted_with_idf_curr_words_sim = curr_words_sim*RT_term_idf_dict[sen_word]*RT_term_idf_dict[claim_word]
                            except Exception as err: 
                                sys.stderr.write('problem in find idf:'+sen_word +" ")     
                                print err.args      
                                print err 
                            if weighted_with_idf_curr_words_sim > max_sim:
                                max_sim = weighted_with_idf_curr_words_sim
                                max_sim_sen_word = sen_word
                                max_sim_clm_word = claim_word
                                
                        #finished sen for one claim word. write to logger
                        curr_clm_sen_sim = max_sim #finished with the curr sen and a particular clm word
                    try:
                        curr_clm_sen_sim = float(curr_clm_sen_sim/float(len(curr_claim_words)*len(curr_sen_words)))
                        #03/15 update -  keep the clm num and sen num
                        #self.clm_sen_cosine_sim_res[(curr_claim,curr_sen)] = curr_clm_sen_sim
                        self.clm_sen_cosine_sim_res[(curr_claim_num,sen_idx)] = curr_clm_sen_sim
                        
#                         print "entered to dict sen idx: ",sen_idx
                        self.max_sim_words_clm_and_sen[(curr_claim,curr_sen)] = (max_sim_clm_word,max_sim_sen_word,str(curr_clm_sen_sim))
#                         self.clm_sen_cosine_sim_res[(self.clm_text_sen_text_dict[0],self.clm_text_sen_text_dict[sen_idx])]=curr_clm_sen_sim
#                         self.max_sim_words_clm_and_sen[(self.clm_text_sen_text_dict[0],self.clm_text_sen_text_dict[sen_idx])] = (max_sim_clm_word,max_sim_sen_word,str(curr_clm_sen_sim))
                    except Exception as err: 
                            sys.stderr.write('problem in calc_clm_sen_sim_based_on_max_sim_between_word:'+str(sen_idx) +" ")     
                            print err.args      
                            print err
                else:
                    print "sen " +str(sen_idx) +" is empty"
        except Exception as err: 
                        sys.stderr.write('problem in sen_idx'+str(sen_idx) +" ")     
                        print err.args      
                        print err 
        utils_linux.save_pickle(input_file_name+"_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim), self.clm_sen_cosine_sim_res)
#         self.save_to_csv_file(self.clm_sen_cosine_sim_res,input_file_name+"_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim)+".csv")
        print "finished calculating max sim for claim", input_file_name
    
    def get_word_vec_from_word_rep_dict(self,word):
        try:
            if self.representation is "turian":
                if self.word_rep_dict.has_key(word):
                    curr_vec = self.word_rep_dict[word]
                    return curr_vec
                else:
                    print "word not found in turian", word
                    return np.zeros((1,self.dim))
            elif self.representation is "word2vec" :
                if self.model_words.__contains__(word):
                    curr_vec = self.model_words.__getitem__(word)
                    return curr_vec
                elif self.model_words.__contains__(word.lower()):
                    curr_vec = self.model_words.__getitem__(word.lower())
                    return curr_vec
                else:
                    print "word not found in word2vec", word
                    return np.zeros((self.dim))
        except Exception as err: 
            sys.stderr.write('problem in get_word_vec_from_word_rep_dict'+word)     
            print err.args      
            print err    
    
    def find_words(self,words_list):
        try:
            curr_len = 0
            curr_found_words = []
            for word in words_list:
                curr_clm_word_vec = self.get_word_vec_from_word_rep_dict(word) 
                if np.array_equal(curr_clm_word_vec,np.zeros((curr_clm_word_vec.shape[0])))== True:
                    continue
                else:
                    curr_len += 1
                    curr_found_words.append(word)
            return (curr_len,curr_found_words)
        except Exception as err: 
            sys.stderr.write('problem in find_words',words_list)     
            print err.args      
            print err    
         
    def calc_clm_sen_sim_based_on_max_sim_full_coverage_claim(self,input_file_name):
        """
        As an extension of max sim between claim word and sen words,
        for each claim word (excluding stop words and perhaps excluding the entity in the claim),
        find a match - using the hungarian algorithm:
        for each claim and sentence:
            have an n*m matrix, n = words in claim;m - words in sentence (if not squared, pad it)
            entry i,j is the cosine sim between word i and word j, scaled with their idf.
            The profit is the total similarity between clain and sentence
        """
        print "calculating max_sim_full_coverage_claim", input_file_name
        
        curr_claim_num = int(input_file_name.split("_")[1])  
        curr_claim_words = self.tokenized_clm_and_sen_dict.values()[0]
        curr_claim_len = len(curr_claim_words)
        curr_claim = ' '. join(curr_claim_words)
        self.clm_sen_cosine_sim_res = {}
        wiki_term_idf_dict = utils_linux.read_pickle("wikiWithBody_term_idf")
        print "\t in claim: "+ curr_claim
        
#         curr_clm_word_vec = np.zeros((1,self.dim))
        print "\t with",str(len(self.tokenized_clm_and_sen_dict.keys()))+" sentences"
        curr_claim_len,curr_claim_found_words = self.find_words(curr_claim_words)
        curr_claim_words_matrix = np.zeros((1,self.dim))              
        for claim_word in curr_claim_found_words:
            claim_word_idx = curr_claim_found_words.index(claim_word)
            curr_clm_word_vec = self.get_word_vec_from_word_rep_dict(claim_word)
            if wiki_term_idf_dict.has_key(claim_word):
                curr_clm_word_vec = curr_clm_word_vec*wiki_term_idf_dict[claim_word]
            elif wiki_term_idf_dict.has_key(claim_word.lower()):
                curr_clm_word_vec = curr_clm_word_vec*wiki_term_idf_dict[claim_word.lower()]
            curr_claim_words_matrix = np.vstack((curr_claim_words_matrix,curr_clm_word_vec))
        curr_claim_words_matrix = np.delete(curr_claim_words_matrix,(0),axis=0)
        for sen_idx in range(1,len(self.tokenized_clm_and_sen_dict.keys())):
            #    print "in sen index:" +str(sen_idx) 
            try:
                curr_sen_words_matrix = np.zeros((1,self.dim))
                curr_sen_word_vec = np.zeros((1,self.dim))
                curr_clm_sen_sim = 0
                curr_sen_words = self.tokenized_clm_and_sen_dict.values()[sen_idx]
                if len(curr_sen_words) > 0:
                    negativity_flag = 0
                    curr_sen = ' '. join(curr_sen_words)
                    curr_sen_len,curr_sen_found_words = self.find_words(curr_sen_words)
                    ####go over sen words
                    curr_claim_sen_words_sim_matrix = np.zeros((curr_claim_len,curr_sen_len))
                    for sen_word in curr_sen_found_words:
                        sen_word_idx = curr_sen_found_words.index(sen_word)
                        curr_sen_word_vec = self.get_word_vec_from_word_rep_dict(sen_word)
                        if wiki_term_idf_dict.has_key(sen_word):
                            curr_sen_word_vec = curr_sen_word_vec*wiki_term_idf_dict[sen_word]
                        elif wiki_term_idf_dict.has_key(sen_word.lower()):
                            curr_sen_word_vec = curr_sen_word_vec*wiki_term_idf_dict[sen_word.lower()]
                        curr_sen_words_matrix = np.vstack((curr_sen_words_matrix,curr_sen_word_vec))
                    curr_sen_words_matrix = np.delete(curr_sen_words_matrix, (0),axis=0)
                    for claim_word_idx in range(0,curr_claim_words_matrix.shape[0]):
                        for sen_word_idx in range(0,curr_sen_words_matrix.shape[0]):
                            curr_words_sim = utils_linux.cosine_measure(curr_claim_words_matrix[claim_word_idx], curr_sen_words_matrix[sen_word_idx])
                            if curr_words_sim < 0:
                                negativity_flag = 1
                            curr_claim_sen_words_sim_matrix[claim_word_idx][sen_word_idx] = curr_words_sim
                    #put large value in the zero values items
                    zero_row_ind,zero_col_ind = np.where(curr_claim_sen_words_sim_matrix==0) 
                    for ind in range(0,len(zero_row_ind)):
                        curr_claim_sen_words_sim_matrix[zero_row_ind[ind],zero_col_ind[ind]] = sys.maxsize 
                    # if the matrix is not square, pad it with minus inf rows
                    while curr_claim_sen_words_sim_matrix.shape[0] < curr_claim_sen_words_sim_matrix.shape[1]:
                        curr_claim_sen_words_sim_matrix = np.vstack((curr_claim_sen_words_sim_matrix,curr_sen_len*[sys.maxsize]))
                    while curr_claim_sen_words_sim_matrix.shape[0] > curr_claim_sen_words_sim_matrix.shape[1]:
                        #TODO
                        # need to decide with the others what to do when the claim is longer than the sentence
                        a = np.vstack((curr_claim_sen_words_sim_matrix.T,curr_claim_len*[sys.maxsize]))
                        curr_claim_sen_words_sim_matrix = a.T
                        
                    # update 21/04/15 -  because the hungarian algo' requires non-negative values,
                    # so add a large constant to the matrix, and on this matrix apply the algo' to get the assignment,
                    # the value will be determined according to the original matrix
                    curr_clm_sen_sim = self.apply_hungarian_algo(curr_claim_sen_words_sim_matrix, curr_claim_len,negativity_flag)
                    self.clm_sen_cosine_sim_res[(curr_claim_num,sen_idx)] = curr_clm_sen_sim
                else:
                    print "\t sen " +str(sen_idx) +" is empty"
                    self.clm_sen_cosine_sim_res[(curr_claim_num,sen_idx)] = 0                
            except Exception as err: 
                sys.stderr.write('\t problem in sen_idx'+str(sen_idx) +" ")     
                print err.args      
                print err                
        # convert the sim profit matrix to a cost matrix
        utils_linux.save_pickle(input_file_name+"_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim), self.clm_sen_cosine_sim_res)
#         self.save_to_csv_file(self.clm_sen_cosine_sim_res,input_file_name+"_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim)+".csv")
        print "\t finished calc_clm_sen_sim_based_on_max_sim_full_coverage_claim", input_file_name
    
    def apply_hungarian_algo(self,matrix,claim_len, negativity_flag):
#       print "applying hungrarin algo..."
        #construct the Munkres() object
        try:
            munkres = Munkres()
            cost_sim_matrix = []
            cost_sim_matrix_orig = []
            C = 1000
            converted_matrix = matrix.copy()
            if negativity_flag == 1:
                converted_matrix += C 
            #convert the similarity - as I want a maximun profit (sim) rather than a minmum cost
            for row in converted_matrix:
                cost_row = map(lambda x: -1*x ,row)
                cost_sim_matrix += [cost_row]
            indexes = munkres.compute(converted_matrix)
            for row in matrix:
                cost_row = map(lambda x: -1*x ,row)
                cost_sim_matrix_orig += [cost_row]
                
    #                     print_matrix(curr_claim_sen_words_sim_matrix, msg='Highest profit through this matrix:')
            curr_clm_sen_sim = 0
            for row, column in indexes:
                if row < claim_len:
                    value = cost_sim_matrix_orig[row][column]
                    curr_clm_sen_sim += value
    #                 print '(%f, %f) -> %f' % (row, column, value)
                else:
    #                 print "covered claim words"
                    break
    #         print 'total profit as words sim=%f' % curr_clm_sen_sim
            return curr_clm_sen_sim
        except Exception as err: 
                sys.stderr.write('problem in apply_hungarian_algo')     
                print err.args      
                print err  
        
    def save_to_csv_file(self,d,file_name):      #save to file
        with open(file_name, 'wb') as csvfile:
            w = csv.writer(csvfile)
            for sen in d.items():
                w.writerow([sen]) 
                 
    def combine_all_clm_sen_sim(self,features):
        try:
            print "in combine_all_clm_sen_sim..."
            claim_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,54,55,57,58,59,60,61,66]
            for clm_num in claim_list:
                self.read_pickle("clm_"+str(clm_num)+"_clm_sen_cosine_sim_res_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim))             
                print "clm_num:" ,clm_num, len(self.clm_sen_cosine_sim_res)
                self.all_clm_sen_sim_res.update(self.clm_sen_cosine_sim_res)
            print "finished combine_all_clm_sen_sim"   
        except Exception as err: 
                sys.stderr.write('problem in combine_all_clm_sen_sim:')     
                print err.args      
                print err        
    
    def remove_entity_title_from_clm_and_sen_and_retokenize(self):
        """
        Another version of the semantic similarity , when the entity tiltle is removed from the clm and sen
        1. Read claim and corres' sen clm_sen_doc_title_entity_presence_flag dict
        2. remove from clm and sen
        3. re-tokenize
        """
        clm_sen_doc_title_entity_presence_flag = utils_linux.read_pickle("clm_sen_doc_title_entity_presence_flag") #key is a claim and sen, value is the doc title and a flag of whether the doc title is in the sen
        clm_sen_no_entity_name = {} #key is orig claim and orig sen, value is the edited clm and edited sen
        clm_sen_no_entity_name_tokenized = {} #key is a tokenized clm, value is a list of tokenized sen
        clm_sen_support_ranking_wiki_sorted = utils_linux.read_pickle("clm_sen_support_ranking_wiki_sorted")
        clm_sen_support_ranking_RT_sorted = utils_linux.read_pickle("clm_sen_support_ranking_RT_sorted")
        clm_text_movie_name_dict = utils_linux.read_pickle("clm_text_movie_name_dict")
        exclude = set(string.punctuation)
        exclude.add('--')
         
        for ((clm,sen),(doc_tit,flag)) in clm_sen_doc_title_entity_presence_flag.items():
            if flag ==1: 
                new_clm = clm.replace(clm_text_movie_name_dict[clm],"")
                new_sen = sen.replace(clm_text_movie_name_dict[clm],"")
                clm_sen_no_entity_name_tokenized[(clm,sen)] = ([self.check_word_in_rep_dict(w,self.representation) for w in nltk.word_tokenize(new_clm.replace("-"," ")) if ( self.check_word_in_rep_dict(w,self.representation) is not "no") and  w not in exclude] 
                                                            ,[self.check_word_in_rep_dict(w,self.representation) for w in nltk.word_tokenize(new_sen.replace("-"," ")) if ( self.check_word_in_rep_dict(w,self.representation) is not "no") and  w not in exclude])
            else:                                               
                clm_sen_no_entity_name_tokenized[(clm,sen)]= ([self.check_word_in_rep_dict(w,self.representation) for w in nltk.word_tokenize(clm.replace("-"," ")) if ( self.check_word_in_rep_dict(w,self.representation) is not "no") and  w not in exclude]
                                                            ,[self.check_word_in_rep_dict(w,self.representation) for w in nltk.word_tokenize(sen.replace("-"," ")) if ( self.check_word_in_rep_dict(w,self.representation) is not "no") and  w not in exclude])
        
        not_found_words = []
        if self.wiki_rt_setup == "separate":
            RT_term_idf_dict = utils_linux.read_pickle("RT_term_idf_dict")
            wiki_term_idf_dict = utils_linux.read_pickle("wikiWithBody_term_idf")
        for ((clm,sen),(clm_tokens,sen_tokens)) in clm_sen_no_entity_name_tokenized.items():
            try:
                max_sim = 0
                curr_clm_sen_sim = 0
                for claim_word in clm_tokens:  
                    if self.representation is "turian":
                        curr_clm_word_vec = self.word_rep_dict[claim_word]
                    else:
                        if self.model_words.__contains__(claim_word):
                            curr_clm_word_vec = self.model_words.__getitem__(claim_word)
                        else:
                            not_found_words.append(claim_word)
                            continue
                    ####go over sen words
                    for sen_word in sen_tokens:
                        if self.representation is "turian":
                            curr_sen_word_vec = self.word_rep_dict[sen_word]
                        elif self.representation is "word2vec":
                            if self.model_words.__contains__(sen_word):
                                curr_sen_word_vec = self.model_words.__getitem__(sen_word)
                            else:
                                not_found_words.append(sen_word)
                                continue
                        if sen_word == claim_word or sen_word.lower() == claim_word.lower():
                            curr_words_sim = 0
                        else:
                            curr_words_sim = utils_linux.cosine_measure(curr_clm_word_vec, curr_sen_word_vec)
                        try:
                            if (clm,sen) in clm_sen_support_ranking_wiki_sorted.keys(): #odd sen index - so it is from wiki
                                weighted_with_idf_curr_words_sim = curr_words_sim*wiki_term_idf_dict[sen_word]*wiki_term_idf_dict[claim_word]
                            elif (clm,sen) in clm_sen_support_ranking_RT_sorted.keys():
                                weighted_with_idf_curr_words_sim = curr_words_sim*RT_term_idf_dict[sen_word]*RT_term_idf_dict[claim_word]
                        except Exception as err: 
                            sys.stderr.write('problem in find idf:'+sen_word +" ")     
                            print err.args      
                            print err 
                        if weighted_with_idf_curr_words_sim > max_sim:
                            max_sim = weighted_with_idf_curr_words_sim
                            max_sim_sen_word = sen_word
                            max_sim_clm_word = claim_word
                            
                    #finished sen for one claim word. write to logger
                    curr_clm_sen_sim = max_sim #finished with the curr sen and a particular clm word
                curr_clm_sen_sim=float(curr_clm_sen_sim/float(len(clm.split())*len(sen.split())))
                self.clm_sen_cosine_sim_res[(clm,sen)] = curr_clm_sen_sim
                self.max_sim_words_clm_and_sen[(clm,sen)] = (max_sim_clm_word,max_sim_sen_word,curr_clm_sen_sim)
            except Exception as err: 
                    sys.stderr.write('problem in remove_entity_title_from_clm_and_sen_and_retokenize:clm '+clm +" ; sen: "+sen)     
                    print err.args      
                    print err 
        
        all_clm_sen_sim_res_sorted = collections.OrderedDict(sorted(self.self.clm_sen_cosine_sim_res.items(),key=lambda x: (x[0][0],float(x[1])), reverse=True))
        all_clm_sen_sim_res_sorted_with_words = collections.OrderedDict(sorted(self.self.clm_sen_cosine_sim_res.items(),key=lambda x: (x[0][0],float(x[1][2])), reverse=True))
        utils_linux.save_pickle(self.setup+"_claim_sen_VSM_similarity_sorted_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim)+"_entity_removed_with_words",all_clm_sen_sim_res_sorted_with_words)
        utils_linux.save_pickle(self.setup+"_claim_sen_VSM_similarity_sorted_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim)+"_entity_remove",all_clm_sen_sim_res_sorted)
        with open(self.setup+"_claim_sen_VSM_similarity_sorted_"+self.representation+"_"+self.compositional_func+"_"+str(self.dim)+"_entity_remove.csv",'wb') as csvfile:  
            w = csv.writer(csvfile)
            for ((clm,sen),sim_score) in all_clm_sen_sim_res_sorted_with_words.items():
                w.writerow([clm,sen,str(sim_score)])      
        utils_linux.save_pickle(self.setup+"_clm_sen_no_entity_name_tokenized", clm_sen_no_entity_name_tokenized)
     
def main():
    try:
        
        setup = "support_baseline"
        clm_sen_sim = clm_sen_similarity_VSM(setup)
        clm_sen_sim.features = "VSM"
        clm_sen_sim.compositional_func = "max_words_similarity"
        entity_presence = "allow" #if "remove" the entity name from the clm and sen
       
        clm_sen_sim.combine_all_clm_sen_sim(clm_sen_sim.features)    
        utils_linux.save_pickle(clm_sen_sim.setup+"_all_clm_sen_cosine_sim_res_"+clm_sen_sim.representation+"_"+clm_sen_sim.compositional_func+"_"+str(clm_sen_sim.dim),clm_sen_sim.all_clm_sen_sim_res) 
        
        if clm_sen_sim.representation == "turian":
#             clm_sen_sim.read_word_rep_file()
            clm_sen_sim.read_pickle("word_rep_"+str(clm_sen_sim.dim))
        elif clm_sen_sim.representation =="word2vec":
            clm_sen_sim.read_sen_word2vec()
        
        #if want to remove entity name from clm and sen
        if entity_presence == "remove":
            clm_sen_sim.remove_entity_title_from_clm_and_sen_and_retokenize()
        else: 
            #for each claim and sentence file, activate the program
            if clm_sen_sim.compositional_func is "max_words_similarity":
                if clm_sen_sim.input_data == "files":
                    for filename in os.listdir(clm_sen_sim.clm_sen_input_path):
                        #create a dict of the claims and sentences
                        file_clm_name ='clm_' + filename.split('_')[1]
        #                 if "clm_21" in file_clm_name : 
            #             clm_sen_sim.create_clm_sim_dict(filename)
                        clm_sen_sim.read_pickle(file_clm_name + "_clm_text_sen_text_dict")
                        clm_sen_sim.tokenize_sentences_from_files(filename, clm_sen_sim.features)
                        clm_sen_sim.read_pickle(file_clm_name+'_tokenized_sen_'+clm_sen_sim.features)
                        clm_sen_sim.calc_clm_sen_sim_based_on_max_sim_between_word(file_clm_name)
                        clm_sen_sim.save_to_csv_file(clm_sen_sim.max_sim_words_clm_and_sen, clm_sen_sim.setup+"_max_sim_words_clm_and_sen_"+file_clm_name+".csv")
#                         clm_sen_sim.tokenize_sentences(filename,clm_sen_sim.features);  
                elif clm_sen_sim.input_data == "dict":
#                     clm_sen_sim.tokenize_sentences_from_dict()
                    clm_sen_sim.claim_sentence_list_dict = utils_linux.read_pickle(clm_sen_sim.setup+"_claim_sentences")
                    #clm_sen_sim.tokenized_clm_and_sen_dict = utils_linux.read_pickle(clm_sen_sim.setup+'_tokenized_clm_and_sen_dict_'+clm_sen_sim.features)
#                     for clm in clm_sen_sim.claim_sentence_list_dict.keys():
                    for clm in clm_sen_sim.claim_list:
                        file_clm_name = 'clm_' + str(clm)
                        clm_sen_sim.tokenized_clm_and_sen_dict = utils_linux.read_pickle(clm_sen_sim.setup+'_clm_'+str(clm)+'_tokenized_clm_and_sen_dict_'+clm_sen_sim.features)
                        clm_sen_sim.read_pickle(file_clm_name + "_clm_text_sen_text_dict")
#                         clm_sen_sim.calc_clm_sen_sim_based_on_max_sim_between_word(file_clm_name)
#                         clm_sen_sim.calc_clm_sen_sim_based_on_max_sim_full_coverage_claim(file_clm_name)
#                         clm_sen_sim.save_to_csv_file(clm_sen_sim.max_sim_words_clm_and_sen, clm_sen_sim.setup+"_clm_"+str(clm)+"_max_sim_words_clm_and_sen.csv")     
        #             
            else:
                clm_sen_sim.read_pickle(file_clm_name+'_tokenized_sen_'+clm_sen_sim.features)
                if clm_sen_sim.compositional_func is "parse_tree":
                    clm_sen_sim.instantiate_POS_weight()
                clm_sen_sim.represent_clm_and_sen_as_VSM(filename)
                clm_sen_sim.calc_clm_sen_similarity(file_clm_name)
              
        
#             clm_sen_sim.combine_all_clm_sen_sim(clm_sen_sim.features)    
#             clm_sen_sim.save_pickle(clm_sen_sim.setup+"_all_clm_sen_cosine_sim_res_"+clm_sen_sim.representation+"_"+clm_sen_sim.compositional_func+"_"+str(clm_sen_sim.dim)) 
    #         
            #read back the pickle and sort for each claim by most similar
            clm_sen_sim.read_pickle(clm_sen_sim.setup+"all_clm_sen_cosine_sim_res_" + clm_sen_sim.representation+"_"+clm_sen_sim.compositional_func+"_"+str(clm_sen_sim.dim))
            all_clm_sen_sim_res_sorted = collections.OrderedDict(sorted(clm_sen_sim.all_clm_sen_sim_res.items(),key=lambda x: (x[0][0],float(x[1])), reverse=True))
            utils_linux.save_pickle(clm_sen_sim.setup+"_claim_sen_VSM_similarity_sorted_"+clm_sen_sim.representation+"_"+clm_sen_sim.compositional_func+"_"+str(clm_sen_sim.dim),all_clm_sen_sim_res_sorted)
            with open(clm_sen_sim.setup + "_claim_sen_VSM_similarity_sorted_" + clm_sen_sim.representation + "_" + clm_sen_sim.compositional_func+"_"+str(clm_sen_sim.dim)+".csv",'wb') as csvfile:  
                w = csv.writer(csvfile)
                for ((clm,sen),sim_score) in all_clm_sen_sim_res_sorted.items():
                    w.writerow([clm,sen,str(sim_score)])         
            
    except Exception as err: 
        sys.stderr.write('problem in main:')     
        print err.args      
        print err 
     

if __name__ == '__main__':
    main()
