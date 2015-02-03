import sys
try:
    import cPickle as pickle
except:
    import pickle
import os.path
import collections
import pandas as pd
import math
import string
import csv
from my_utils import  utils
import numpy as np
# base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\baseline_clmLMdocLM"
linux_base_path = r"/home/liorab/softwares/indri-5.5/retrieval_baselines"
# base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\SVM_zero_two_scale"
"""
support models (done on the wikipedia corpus only)

"""
class max_min_CE_scores_keeper():
    max_CE_claim_title = 0.0
    max_CE_claim_body = 0.0
    max_CE_entity_title = 0.0
    max_CE_entity_body = 0.0
    min_CE_claim_title = 0.0
    min_CE_claim_body = 0.0
    min_CE_entity_title = 0.0
    min_CE_entity_body = 0.0
    
    max_CE_claim_sentence = 0.0 
    max_CE_entity_sentence = 0.0
    min_CE_claim_sentence = 0.0 
    min_CE_entity_sentence = 0.0
    
    def __init__(self):
        max_CE_claim_title = 0.0
        max_CE_claim_body = 0.0
        max_CE_entity_title = 0.0
        max_CE_entity_body = 0.0
        min_CE_claim_title = 0.0
        min_CE_claim_body = 0.0
        min_CE_entity_title = 0.0
        min_CE_entity_body = 0.0
        
        max_CE_claim_sentence = 0.0 
        max_CE_entity_sentence = 0.0
        min_CE_claim_sentence = 0.0 
        min_CE_entity_sentence = 0.0

class max_min_sentiment_score_keeper():
    max_sentiment_score = 0.0 #for the sentiment similarity!!
    min_sentiment_score = 0.0#for the sentiment similarity!!
    max_claim_sentiment_label = 0.0 
    min_claim_sentiment_label = 0.0
    max_sentence_sentiment_label = 0.0 
    min_sentence_sentiment_label = 0.0
    max_claim_sentiment_entropy = 0.0
    min_claim_sentiment_entropy = 0.0
    max_sentence_sentiment_entropy = 0.0
    min_sentence_sentiment_entropy = 0.0
    
    def init(self):
        max_sentiment_score = 0.0
        min_sentiment_score = 0.0
        max_sentiment_label = 0.0
        min_sentiment_label = 0.0
        max_sentiment_entropy = 0.0
        min_sentiment_entropy = 0.0


class max_min_semantic_score_keeper():
    max_semantic_score = 0.0
    min_semantic_score = 0.0
          
    def init(self):
        max_semantic_score = 0.0
        min_semantic_score = 0.0
          
class CEScores():
    CE_claim_title = 0.0
    CE_claim_body = 0.0
    CE_entity_title = 0.0
    CE_entity_body = 0.0
    CE_claim_sentence = 0.0
    CE_entity_sentence = 0.0
    
    def  __init__(self):
        CE_claim_title = 0.0
        CE_claim_body = 0.0
        CE_entity_title = 0.0
        CE_entity_body = 0.0
        CE_claim_sentence = 0.0
        CE_entity_sentence = 0.0

class support_baseline():
    
    def  __init__(self,kernel,features_setup):
        self.relevance_linux_base_path = r"/home/liorab/softwares/indri-5.5/retrieval_baselines"
#         self.relevance_base_path = r"/home/liorab/softwares/indri-5.5/retrieval_baselines"
#         self.support_linux_base_path = r"/lv_local/home/liorab/softwares/indri-5.5/retrieval_baselines_support"
        self.features_setup = features_setup
        self.support_linux_base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines"
        self.SVM_path = self.support_linux_base_path+r"\SVM_zero_two_scale"+"\\" + self.features_setup +r"_features"
#         self.SVM_path = self.support_linux_base_path+ "/SVM_zero_two_scale/"+ self.features_setup+r"_features"
        self.train_path = self.SVM_path+r"\train"+"\\"
        self.test_path = self.SVM_path +r"\test"+"\\"
        self.model_path = self.SVM_path +r"\model"+"\\"
        self.prediction_path = self.SVM_path +r"/prediction/"
        self.docid_set_per_claim = {}
        self.top_k_docs = 50
        self.claim_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
        self.claim_entity_sentence_sen_output_path = self.support_linux_base_path+r"/claimEntity_sen_output/"
        self.claim_entity_body_title_output_path = self.support_linux_base_path+r"/claimEntity_bodyTitle_output/"
        self.claim_entity_doc_CE_scores_dict = {} #key is a claim, value is a key of :doc -> list of CE scores
        self.claim_entity_sen_CE_scores_dict = {} #key is a claim -> docid-> sen-> [2 CE scores]
        self.CE_claim_body_sum = 0
        self.CE_claim_title_sum = 0
        self.CE_entity_body_sum = 0
        self.CE_entity_title_sum = 0  #for the noramlization
        self.CE_claim_sentence_sum = 0
        self.CE_entity_sentence_sum = 0
        self.clm_sen_sentiment_similarity_dict = {}
        self.clm_sen_semantic_similarity_dict = {}       
        self.kernel = kernel
        self.claim_sentiment_vector_entropy = {}
        self.sentence_sentiment_vector_entropy = {}
        self.claim_sentiment_vector_and_label_dict = {}
        self.claim_sen_sentiment_vector_and_label_dict = {}
        
    def read_pickle(self,file_name):
        d = {}
        with open(file_name, 'rb') as handle:
            d = pickle.loads(handle.read()) 
        return d
    
    def save_pickle(self,file_name,d):
        with open(file_name, 'wb') as handle:
            pickle.dump(d, handle)
        handle.close()
    
    def create_set_of_docs_per_claim(self):
        print " creating set of docs per claim..."
        doc_res_dicts_path = self.relevance_linux_base_path+"/docs_norm_scores_dicts"
        for claim_num in self.claim_list:
            print "    in claim",claim_num
            curr_set = set()
            for filename in os.listdir(doc_res_dicts_path):
                if not "clm_key_ranked_list_of_docs_" in filename:
                    curr_dict = self.read_pickle(doc_res_dicts_path+"/"+filename)
                    curr_claim_num = filename.split("_clm_")[1].split("_dict_sorted")[0]
#                     print "    curr claim num",curr_claim_num
                    if str(claim_num) == curr_claim_num:
                        print "    found file, opening..."
                        top_k_docs = [key[1] for key in curr_dict.keys()][0:self.top_k_docs]
                        for docid in top_k_docs:
                            curr_set.add(docid)
                        print "curr_set len", len(curr_set)
            self.docid_set_per_claim[claim_num] = curr_set
            print "len docid_set_per_claim[clain_num]" ,len(self.docid_set_per_claim[claim_num])      
        print "finished create_set_of_docs_per_claim"
        
    def create_sen_ret_input_file(self):
        print "creating sentence ret files..."
        try:
            sen_ret_input_path = self.support_linux_base_path+"/sentence_ret_input"
            claims_no_SW_dict = self.read_pickle("claims_no_SW_dict")
            for claim_num in self.claim_list:
                print "    in claim", claim_num
                sen_ret_docno_file = open(sen_ret_input_path+"/sen_ret_top_k_docs_"+str(self.top_k_docs)+"_clm_"+str(claim_num),"wb")
                sen_ret_docno_file.write("<parameters>\n")
                sen_ret_docno_file.write("<query><number>"+str(claim_num)+"</number><text>"+claims_no_SW_dict[str(claim_num)][0].strip()+"|"+claims_no_SW_dict[str(claim_num)][1].strip()+"</text>")
                for workingDoc in self.docid_set_per_claim[claim_num]:
                    sen_ret_docno_file.write("<workingSetDocno>"+workingDoc+"</workingSetDocno>")
                sen_ret_docno_file.write("</query>\n")
                sen_ret_docno_file.write("</parameters>")
                sen_ret_docno_file.close()
        except Exception as err: 
            sys.stderr.write("problem in create_sen_ret_input_file")     
            print err.args      
            print err
    
    def map_claim_doc_to_CE_scores(self):
        # for the 4 CE scores sim(claim,body), sim(entity,body), sim(claim,title), sim(entity,title)
        #that are related to the document
        print " calc doc CE scores..."
        
        for claim_entity_body_title_file in os.listdir(self.claim_entity_body_title_output_path):
            f = open(self.claim_entity_body_title_output_path+claim_entity_body_title_file, "rb")
            data = pd.read_csv(f," ")
            curr_claim = data["q_number"][0]
            if not self.claim_entity_doc_CE_scores_dict.has_key(curr_claim):
                self.claim_entity_doc_CE_scores_dict[curr_claim] = {} 
            docid = data["documentName"]  
            body_score = data["bodyScore"]
            title_score = data["titleScore"]    
            results_num = len(docid)
            for row_index in range(0,results_num) :
                if "claim_body" in claim_entity_body_title_file:          
                    if self.claim_entity_doc_CE_scores_dict.has_key(curr_claim):
                        if self.claim_entity_doc_CE_scores_dict[curr_claim].has_key(docid[row_index]):
                            curr_CE_scores = self.claim_entity_doc_CE_scores_dict[curr_claim][docid[row_index]]
                            curr_CE_scores.CE_claim_body = math.exp(body_score[row_index])
                            self.CE_claim_body_sum += curr_CE_scores.CE_claim_body
                            curr_CE_scores.CE_claim_title = math.exp(title_score[row_index])   
                            self.CE_claim_title_sum += curr_CE_scores.CE_claim_title                       
                        else:
                            curr_CE_scores = CEScores() #create a new scores object 
                            curr_CE_scores.CE_claim_body = math.exp(body_score[row_index])
                            self.CE_claim_body_sum += curr_CE_scores.CE_claim_body
                            curr_CE_scores.CE_claim_title = math.exp(title_score[row_index])
                            self.CE_claim_title_sum += curr_CE_scores.CE_claim_title              
                        self.claim_entity_doc_CE_scores_dict[curr_claim][docid[row_index]] = curr_CE_scores
                else: #entity file
                    if self.claim_entity_doc_CE_scores_dict.has_key(curr_claim):
                        if self.claim_entity_doc_CE_scores_dict[curr_claim].has_key(docid[row_index]):
                            curr_CE_scores = self.claim_entity_doc_CE_scores_dict[curr_claim][docid[row_index]]
                            curr_CE_scores.CE_entity_body = math.exp(body_score[row_index])
                            self.CE_entity_body_sum += curr_CE_scores.CE_entity_body 
                            curr_CE_scores.CE_entity_title = math.exp(title_score[row_index])
                            self.CE_entity_title_sum += curr_CE_scores.CE_entity_title 
                            
                        else:
                                curr_CE_scores = CEScores() #create a new scores object 
                                curr_CE_scores.CE_entity_body = math.exp(body_score[row_index])
                                self.CE_entity_body_sum += curr_CE_scores.CE_entity_body 
                                curr_CE_scores.CE_entity_title = math.exp(title_score[row_index])
                                self.CE_entity_title_sum += curr_CE_scores.CE_entity_title 
                        self.claim_entity_doc_CE_scores_dict[curr_claim][docid[row_index]] = curr_CE_scores
        self.save_pickle("support_model_claim_entity_doc_CE_scores_dict",self.claim_entity_doc_CE_scores_dict)
        print "finished calc doc CE scores"
    
    def normalize_doc_CE_scores(self):
        self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict")
        for curr_claim in self.claim_list:
            for (docid,CE_scores) in self.claim_entity_doc_CE_scores_dict[curr_claim].items():
                CE_scores.CE_claim_title = float(CE_scores.CE_claim_title/self.CE_claim_title_sum)
                CE_scores.CE_claim_body = float(CE_scores.CE_claim_body/self.CE_claim_body_sum)
                CE_scores.CE_entity_title = float(CE_scores.CE_entity_title/self.CE_entity_title_sum)
                CE_scores.CE_entity_body =  float(CE_scores.CE_entity_body/self.CE_entity_body_sum)
        self.save_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized", self.claim_entity_doc_CE_scores_dict)
       
    def map_claim_sen_to_CE_scores(self):           
        print " calc sen CE scores..."
        for claim_entity_sen_file in os.listdir(self.claim_entity_sentence_sen_output_path):
            sen_file = open(self.claim_entity_sentence_sen_output_path+ claim_entity_sen_file)
            sen = sen_file.read().strip() # score, sentence
            for i, line in enumerate(sen.split('\n')):                   
                if i%2 == 0: # a metadata line
                    data = line.split(' ')
                    curr_claim =int(data[0])
                    docid = data[2]
                    sen_score = data[4]
                    if not self.claim_entity_sen_CE_scores_dict.has_key(curr_claim):
                        self.claim_entity_sen_CE_scores_dict[curr_claim] = {} 
#                     if not self.claim_entity_sen_CE_scores_dict[curr_claim].has_key(docid):
#                         self.claim_entity_sen_CE_scores_dict[curr_claim][docid] = {}
                else:
                    if self.claim_entity_sen_CE_scores_dict.has_key(curr_claim):
                        if self.claim_entity_sen_CE_scores_dict[curr_claim].has_key(docid):
                            if self.claim_entity_sen_CE_scores_dict[curr_claim][docid].has_key(line):
                                curr_CE_scores = self.claim_entity_sen_CE_scores_dict[curr_claim][docid][line]
                                if "claim" in claim_entity_sen_file:
                                    curr_CE_scores.CE_claim_sentence = math.exp(float(sen_score))
                                    self.CE_claim_sentence_sum += curr_CE_scores.CE_claim_sentence 
                                else:
                                    curr_CE_scores.CE_entity_sentence = math.exp(float(sen_score))
                                    self.CE_entity_sentence_sum += curr_CE_scores.CE_entity_sentence
                            else:
                                curr_CE_scores = CEScores()
                            if "claim" in claim_entity_sen_file:
                                curr_CE_scores.CE_claim_sentence = math.exp(float(sen_score))
                                self.CE_claim_sentence_sum += curr_CE_scores.CE_claim_sentence 
                            elif "entity" in claim_entity_sen_file:
                                curr_CE_scores.CE_entity_sentence = math.exp(float(sen_score))
                                self.CE_entity_sentence_sum += curr_CE_scores.CE_entity_sentence 
                        else:
                            curr_CE_scores = CEScores()
                            if "claim" in claim_entity_sen_file:
                                curr_CE_scores.CE_claim_sentence = math.exp(float(sen_score))
                                self.CE_claim_sentence_sum += curr_CE_scores.CE_claim_sentence 
                            elif "entity" in claim_entity_sen_file:
                                curr_CE_scores.CE_entity_sentence = math.exp(float(sen_score))
                                self.CE_entity_sentence_sum += curr_CE_scores.CE_entity_sentence
                    if not self.claim_entity_sen_CE_scores_dict[curr_claim].has_key(docid):
                        self.claim_entity_sen_CE_scores_dict[curr_claim][docid] = {}
                    self.claim_entity_sen_CE_scores_dict[curr_claim][docid][line] = curr_CE_scores
        self.save_pickle("support_model_claim_entity_sen_CE_scores_dict",self.claim_entity_sen_CE_scores_dict)             
        print "finished calc sen CE scores"
    
    def normalize_sen_CE_scores(self):
        self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict")
        for curr_claim in self.claim_list:
                for docid in self.claim_entity_sen_CE_scores_dict[curr_claim]:
                    for (sentences,CE_scores) in self.claim_entity_sen_CE_scores_dict[curr_claim][docid].items():
                        CE_scores.CE_claim_sentence = float(CE_scores.CE_claim_sentence/self.CE_claim_sentence_sum)
                        CE_scores.CE_entity_sentence = float(CE_scores.CE_entity_sentence/self.CE_entity_sentence_sum)
        self.save_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized", self.claim_entity_sen_CE_scores_dict)
    
    def convert_sentiment_dict(self):
        """
        Instead of having (clm_text, sentence) -> sentiment score dict form, 
        convert to clm_text -> [list of (sentence, score)]
        """
        self.clm_sen_sentiment_similarity_dict = self.read_pickle("support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted")
        self.sentence_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sen_sentiment_vector_entropy")
        self.claim_sen_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sen_sentiment_vector_and_label_dict")
        claim_dict = self.read_pickle("claim_dict")
        converted_sentiment_sim_dict = {}
        converted_sentence_sentiment_entropy = {}
        converted_sentence_vector_and_label = {}
        
        for ((clm_text,sen),score)  in self.clm_sen_sentiment_similarity_dict.items():
            if converted_sentiment_sim_dict.has_key(clm_text):
                converted_sentiment_sim_dict[clm_text].append((sen,score))
            else:
                converted_sentiment_sim_dict[clm_text] = [(sen,score)]
        for (clm_num,sen_num) in self.sentence_sentiment_vector_entropy.keys():
            if converted_sentence_sentiment_entropy.has_key(claim_dict[clm_num]):
                converted_sentence_sentiment_entropy[claim_dict[clm_num]].append((sen_num,self.sentence_sentiment_vector_entropy[clm_num,sen_num]))
            else:
                converted_sentence_sentiment_entropy[claim_dict[clm_num]] = [(sen_num,self.sentence_sentiment_vector_entropy[clm_num,sen_num])]
            
            if converted_sentence_vector_and_label.has_key(claim_dict[clm_num]):
                converted_sentence_vector_and_label[claim_dict[clm_num]].append(self.claim_sen_sentiment_vector_and_label_dict[clm_num,sen_num])
            else:
                converted_sentence_vector_and_label[claim_dict[clm_num]] = [self.claim_sen_sentiment_vector_and_label_dict[clm_num,sen_num]]
        sum = 0 
        for claim in converted_sentiment_sim_dict.keys():
            sum += len(converted_sentiment_sim_dict[claim])
            print claim, len(converted_sentiment_sim_dict[claim])
        print sum
        self.save_pickle("converted_support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted", converted_sentiment_sim_dict)
        self.save_pickle("converted_support_baseline_sentence_sentiment_entropy", converted_sentence_sentiment_entropy)
        self.save_pickle("converted_sentence_vector_and_label", converted_sentence_vector_and_label)
        
    def convert_semantic_dict(self):
        """
        Instead of having (clm_text, sentence) -> semantic score dict form, 
        convert to clm_text -> [list of (sentence, score)]
        """
        print "start to convert semantic dict...."
        self.clm_sen_semantic_similarity_dict = self.read_pickle("all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300")
        converted_semantic_dict = {}
        for ((clm_text,sen),score) in self.clm_sen_semantic_similarity_dict.items():
            if sen == "When they smooch in a greenhouse she says he's a good kisser":
                print clm_text
            if converted_semantic_dict.has_key(clm_text):
                converted_semantic_dict[clm_text].append((sen,score))
            else:
                converted_semantic_dict[clm_text] = [(sen,score)]
        self.save_pickle("converted_support_baseline_all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300", converted_semantic_dict)
        print "finished to convert semantic dict"
        
    def  sentiment_feature_max_min_normalization(self):
        #sum up on each claim's 
        left_out_max_min_features = {} 
        claim_dict = self.read_pickle("claim_dict")
        if "sentiment_sim" in self.features_setup:
            self.clm_sen_sentiment_similarity_dict = self.read_pickle("converted_support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted")
        if "label" in self.features_setup:
            self.claim_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sentiment_vector_and_label_dict")
            self.claim_sen_sentiment_vector_and_label_dict = self.read_pickle("converted_sentence_vector_and_label")
        if "entropy" in self.features_setup:
            self.claim_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sentiment_vector_entropy")
            self.sentence_sentiment_vector_entropy = self.read_pickle("converted_support_baseline_sentence_sentiment_entropy")
        print "calc max-min doc features..."   
        for left_out_claim in self.claim_list:
            max_min_sentiment_score = max_min_sentiment_score_keeper()
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            
            if "label" in self.features_setup:
                #find the max and min sentiment label of the claims
                temp_claim_sentiment_vector_and_label_dict = dict(self.claim_sentiment_vector_and_label_dict)
                del temp_claim_sentiment_vector_and_label_dict[str(left_out_claim)]
                sentiment_label = [sentiment_label for _,sentiment_label in temp_claim_sentiment_vector_and_label_dict.values()]
                #find the max/min across the claims of the train set
                max_min_sentiment_score.max_claim_sentiment_label = max(sentiment_label)
                max_min_sentiment_score.min_claim_sentiment_label = min(sentiment_label)
            #find the max and min entropy of the claim's sentiment vector
            if "entropy" in self.features_setup:
                temp_claim_entropy = dict(self.claim_sentiment_vector_entropy)
                del temp_claim_entropy[str(left_out_claim)]
                max_min_sentiment_score.max_claim_sentiment_entropy = max(temp_claim_entropy.values())
                max_min_sentiment_score.min_claim_sentiment_entropy = min(temp_claim_entropy.values())
#                 if max_sentiment_entropy > max_min_sentiment_score.max_claim_sentiment_entropy:
#                     max_min_sentiment_score.max_claim_sentiment_entropy = max_sentiment_entropy
#                 if min_sentiment_entropy < max_min_sentiment_score.min_claim_sentiment_entropy:
#                     max_min_sentiment_score.min_claim_sentiment_entropy = min_sentiment_entropy 
            #and now for the sentences normalization
            for curr_claim in curr_train_claims:
                curr_claim_text = claim_dict[str(curr_claim)]
                if "sentiment_sim" in self.features_setup:
                    sen_scores_list = [sen_score for _,sen_score in self.clm_sen_sentiment_similarity_dict[curr_claim_text]]
                    max_sentiment_sim = max(sen_scores_list)
                    min_sentiment_sim = min(sen_scores_list)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_sentiment_score.max_sentiment_score = max_sentiment_sim
                        max_min_sentiment_score.min_sentiment_score = min_sentiment_sim
                    else:
                        if max_sentiment_sim > max_min_sentiment_score.max_sentiment_score:
                            max_min_sentiment_score.max_sentiment_score = max_sentiment_sim
                        if min_sentiment_sim < max_min_sentiment_score.min_sentiment_score:
                            max_min_sentiment_score.min_sentiment_score = min_sentiment_sim
                if "label" in self.features_setup:
                    sen_label_list = [label for _ ,label in self.claim_sen_sentiment_vector_and_label_dict[claim_dict[str(curr_claim)]] ]
                    max_sentence_sentiment_label = max(sen_label_list)
                    min_sentence_sentiment_label = min(sen_label_list)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_sentiment_score.max_sentence_sentiment_label = max_sentence_sentiment_label
                        max_min_sentiment_score.min_sentence_sentiment_label = min_sentence_sentiment_label
                    else:
                        if max_sentence_sentiment_label > max_min_sentiment_score.max_sentence_sentiment_label:
                            max_min_sentiment_score.max_sentence_sentiment_label = max_sentence_sentiment_label
                        if min_sentence_sentiment_label < max_min_sentiment_score.min_sentence_sentiment_label:
                            max_min_sentiment_score.min_sentence_sentiment_label = min(sen_label_list) 
                if "entropy" in self.features_setup:               
                    sen_entropy_list = [sen_entropy for _, sen_entropy in self.sentence_sentiment_vector_entropy[claim_dict[str(curr_claim)]]]
                    max_sen_entropy = max(sen_entropy_list)
                    min_sen_entropy = min(sen_entropy_list)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_sentiment_score.max_sentence_sentiment_entropy = max_sen_entropy
                        max_min_sentiment_score.min_sentence_sentiment_entropy = min_sen_entropy
                    else:
                        if max_sen_entropy > max_min_sentiment_score.max_sentence_sentiment_entropy:
                            max_min_sentiment_score.max_sentence_sentiment_entropy = max_sen_entropy
                        if min_sen_entropy < max_min_sentiment_score.min_sentence_sentiment_entropy:
                            max_min_sentiment_score.min_sentence_sentiment_entropy = min_sen_entropy
#                      
            left_out_max_min_features[left_out_claim] = max_min_sentiment_score
                              
        self.save_pickle("left_out_max_min_support_sentiment_feature", left_out_max_min_features)              
            
    def semantic_features_max_min_normalization(self):
        """
        read all the claims cosine similarity dicts, and keep the max, min values for each left-out claim
        """
        left_out_max_min_features = {} 
        self.clm_sen_semantic_similarity_dict = self.read_pickle("converted_support_baseline_all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300")
        claim_dict = self.read_pickle("claim_dict")
        for left_out_claim in self.claim_list:
            max_min_semantic_score = max_min_semantic_score_keeper()
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            for curr_claim in curr_train_claims:
                curr_claim_text = claim_dict[str(curr_claim)]
                semantic_sim_list = [sem_sim_score for _,sem_sim_score in self.clm_sen_semantic_similarity_dict[curr_claim_text]]
                max_semantic_score = max(semantic_sim_list)
                min_semantic_score = min(semantic_sim_list)
                if curr_train_claims.index(curr_claim) == 0:
                    max_min_semantic_score.max_semantic_score = max_semantic_score
                    max_min_semantic_score.min_semantic_score = min_semantic_score
                else:
                    if max_semantic_score > max_min_semantic_score.max_semantic_score :
                        max_min_semantic_score.max_semantic_score = max_semantic_score
                    if min_semantic_score < max_min_semantic_score.min_semantic_score:
                        max_min_semantic_score.min_semantic_score = min_semantic_score
            left_out_max_min_features[left_out_claim] = max_min_semantic_score                   
        self.save_pickle("left_out_max_min_support_semantic_feature", left_out_max_min_features)
                                    
    def combine_semantic_dicts_with_original_full_claim_and_sentences(self):
        """
        before changing the semantic module to work with sentences id's,
        the claims and sentences that are saved in the clm_sen_cosine dicts are after tokenization. 
        and so need to chnage to save the original and full claims.
        the claims are taken from the claim_dict
        the sentences: find the index of the snetence in the tokenized_dict, which is created in allignment with claim_sentences dict, 
        that stores the original full sentence
        """
        
        semantic_sim_cosine_all_clms_dict = {}
        claim_dict = self.read_pickle("claim_dict")
        claim_sentences = self.read_pickle("support_baseline_claim_sentences")
        for clm in self.claim_list:
            tokenized_sen_dict = self.read_pickle("support_baseline_clm_"+str(clm)+"_tokenized_clm_and_sen_dict_VSM")
            temp = self.read_pickle("clm_"+str(clm)+"_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300")
            #change the claim text to the full one -  as the one that comes from the SVM programm is without words that 
            # do not appear in the VSM -  for example for, as...
            for (clm_text,sen) in temp.keys():
                #find the sentence in the tokenized dict
                sen_index = 0
                for tokenized_sens in tokenized_sen_dict.values():
                    curr_token_sen = ' '. join(tokenized_sens)
                    if curr_token_sen == sen:
                        sen_index = tokenized_sen_dict.values().index(tokenized_sens)
                        break
                if sen_index != 0:
                    full_sen = claim_sentences[clm][sen_index-1]
                else:
                    print "did not find this sentence in the tokenized dict"
                temp[(claim_dict[str(clm)],full_sen)] = temp[(clm_text,sen)]
                del temp[(clm_text,sen)]
            print "clm " +str(clm) +" " +str(len(temp)) +" sentences"
            semantic_sim_cosine_all_clms_dict.update(temp)
        self.save_pickle("all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300", semantic_sim_cosine_all_clms_dict)               
    
    def normalize_features(self):
        if "CE" in self.features_setup:
            self.CE_features_max_min_normalization()
#             
        if "sentiment" in self.features_setup:
#             self.convert_sentiment_dict()
            self.sentiment_feature_max_min_normalization()
        if "semantic" in self.features_setup:
#             self.combine_semantic_dicts_with_original_full_claim_and_sentences()
#             self.convert_semantic_dict()
            self.semantic_features_max_min_normalization()
            
    def CE_features_max_min_normalization(self):
        """
        for each claim and its sentences, and for each feature - CE score,
        normalize each pair of clm and sen in the max-min value of that feature.
        Since I am performing a LOOCV, I will find the max and min for each LOO setup, and normalize the test with the corresponding value.
        This is opposed to finding the max-min across all the data and normalize with it since I am not supposed to know anything about the test data after training...
        stages:
        1. Perform LOO process
        2. save for each left_out_clm, the max-min.
        """
        print "calc CE max-min doc features..."
        left_out_max_min_features = {} #key is a left-out claim number, value is the maximum and minimum for each of the 6 features
        self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized")
        self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized")
        for left_out_claim in self.claim_list:
            max_min_CE_scores = max_min_CE_scores_keeper()
            features_value_list = [[],[],[],[],[],[]]
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            for curr_claim in curr_train_claims:
                for (docid,CE_scores) in self.claim_entity_doc_CE_scores_dict[curr_claim].items():
                    features_value_list[0].append(CE_scores.CE_claim_title)
                    features_value_list[1].append(CE_scores.CE_claim_body)
                    features_value_list[2].append(CE_scores.CE_entity_title)
                    features_value_list[3].append(CE_scores.CE_entity_body)
            max_min_CE_scores.max_CE_claim_title = max(features_value_list[0])
            max_min_CE_scores.min_CE_claim_title = min(features_value_list[0])
            max_min_CE_scores.max_CE_claim_body  = max(features_value_list[1])
            max_min_CE_scores.min_CE_claim_body  = min(features_value_list[1])
            max_min_CE_scores.max_CE_entity_title = max(features_value_list[2])
            max_min_CE_scores.min_CE_entity_title = min(features_value_list[2])
            max_min_CE_scores.max_CE_entity_body = max(features_value_list[3])
            max_min_CE_scores.min_CE_entity_body = min(features_value_list[3])
        
            left_out_max_min_features[left_out_claim] = max_min_CE_scores 
        
        print "calc CE max-min sen features..."
        for left_out_claim in self.claim_list:
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            for curr_claim in curr_train_claims:
                for docid in self.claim_entity_sen_CE_scores_dict[curr_claim]:
                    for (sentences,CE_scores) in self.claim_entity_sen_CE_scores_dict[curr_claim][docid].items():
                        features_value_list[4].append(CE_scores.CE_claim_sentence)
                        features_value_list[5].append(CE_scores.CE_entity_sentence)
            curr_max_min_CE_scores_obj = left_out_max_min_features[left_out_claim]
            curr_max_min_CE_scores_obj.max_CE_claim_sentence = max(features_value_list[4])
            curr_max_min_CE_scores_obj.min_CE_claim_sentence = min(features_value_list[4])
            curr_max_min_CE_scores_obj.max_CE_entity_sentence = max(features_value_list[5])
            curr_max_min_CE_scores_obj.min_CE_entity_sentence = min(features_value_list[5]) 
                   
        #finished, save to pickle
        self.save_pickle("left_out_max_min_support_CE_features", left_out_max_min_features) 
            
    def convert_support_scores_dict_to_chars_as_sen(self):
        """
        as there is a difference between the sens in the support claim dict and the sens as they were retrieved from the baseline,
        convert the sen to only the chars.
        """
        exclude = set(string.punctuation)
        support_scores_dict = self.read_pickle("clm_sen_support_ranking_zero_to_two_clm_sen_key_supp_score_value")
        support_scores_dict_no_punct = {}
        for ((claim,sen),supp_score) in support_scores_dict.items():
            sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
            sen_no_space = sen_no_punct.replace(" ","")
            support_scores_dict_no_punct[(claim,sen_no_space)] = supp_score
        return support_scores_dict_no_punct
          
    def write_train_test_files_SVM(self):
        #write the files with the max-min normalized scores
        claim_sentences_dict = self.read_pickle("support_baseline_claim_sentences")
        self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized")
        self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized")
#         if "CE" in self.features_setup:
        left_out_max_min_CE_features = self.read_pickle("left_out_max_min_support_CE_features")
        if "sentiment" in self.features_setup:
            left_out_max_min_sentiment_features = self.read_pickle("left_out_max_min_support_sentiment_feature")
            if "sentiment_sim" in self.features_setup:
                self.clm_sen_sentiment_similarity_dict = self.read_pickle("support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted")
            if "entropy" in self.features_setup:
                self.claim_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sentiment_vector_entropy")
                self.sentence_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sen_sentiment_vector_entropy")
            if "label" in self.features_setup:
                self.claim_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sentiment_vector_and_label_dict")
                self.claim_sen_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sen_sentiment_vector_and_label_dict")
        if "semantic" in self.features_setup :
            left_out_max_min_semantic_features = self.read_pickle("left_out_max_min_support_semantic_feature")
            self.clm_sen_semantic_similarity_dict = self.read_pickle("all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300")
        support_scores_dict = self.convert_support_scores_dict_to_chars_as_sen()
        claim_dict = self.read_pickle("claim_dict")
        exclude = set(string.punctuation)
        
        for left_out_claim in self.claim_list: 
            print "writing SVM files for claim " +str(left_out_claim)
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            #line = ""
            curr_test_LOOCV = open (self.test_path+r"test_clm_num_"+str(left_out_claim)+"_CV", 'wb')
            curr_train_LOOCV = open (self.train_path+r"train_left_out_"+str(left_out_claim)+"_CV", 'wb')  
            for curr_claim in curr_train_claims:                   
                print "    curr train claim:" +str(curr_claim)
                sentences_set = set()
                dups_cnt = 0
                for docid in self.claim_entity_doc_CE_scores_dict[curr_claim]:
                    for sen in self.claim_entity_sen_CE_scores_dict[curr_claim][docid].keys():
                        if not sen in sentences_set:
                            try:
                                curr_features_vec = []
                                line = ""
                                sentences_set.add(sen)
                                if "CE" in self.features_setup:
                                    if "claim_title" in self.features_setup or "all" in self.features_setup :
                                        curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_claim_title)/\
                                                             (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_title-left_out_max_min_CE_features[left_out_claim].min_CE_claim_title)))
                                    if "claim_body" in self.features_setup or "all" in self.features_setup:
                                        curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_claim_body)/\
                                                                       (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_body-left_out_max_min_CE_features[left_out_claim].min_CE_claim_body)))
                                    if "entity_title" in self.features_setup or "all" in self.features_setup: 
                                        curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_entity_title)/\
                                                                 (float(left_out_max_min_CE_features[left_out_claim].max_CE_entity_title-left_out_max_min_CE_features[left_out_claim].min_CE_entity_title)))
                                    if "entity_body" in self.features_setup or "all" in self.features_setup:
                                        curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_entity_body)/\
                                                                (float(left_out_max_min_CE_features[left_out_claim].max_CE_entity_body-left_out_max_min_CE_features[left_out_claim].min_CE_entity_body)))
                                    if "claim_sentence" in self.features_setup or "all" in self.features_setup:
                                        curr_features_vec.append(float(self.claim_entity_sen_CE_scores_dict[curr_claim][docid][sen].CE_claim_sentence)/\
                                                                (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_sentence-left_out_max_min_CE_features[left_out_claim].min_CE_claim_sentence)))
                                    if "entity_sentence" in self.features_setup or "all" in self.features_setup:
                                        curr_features_vec.append(float(self.claim_entity_sen_CE_scores_dict[curr_claim][docid][sen].CE_entity_sentence)/\
                                                            (float(left_out_max_min_CE_features[left_out_claim].max_CE_entity_sentence-left_out_max_min_CE_features[left_out_claim].min_CE_entity_sentence)))
                                if "sentiment" in self.features_setup:
                                    if "sentiment_sim" in self.features_setup:
                                        if self.clm_sen_sentiment_similarity_dict.has_key((claim_dict[str(curr_claim)],sen)):
                                            curr_features_vec.append(float(self.clm_sen_sentiment_similarity_dict[(claim_dict[str(curr_claim)],sen)]/\
                                                                           left_out_max_min_sentiment_features[left_out_claim].max_sentiment_score-\
                                                                         left_out_max_min_sentiment_features[left_out_claim].min_sentiment_score))
                                        else:
                                            curr_features_vec.append(0)
                                    if "label" in self.features_setup:
                                        if self.claim_sentiment_vector_and_label_dict.has_key(str(curr_claim)):
                                            curr_features_vec.append(float(self.claim_sentiment_vector_and_label_dict[str(curr_claim)][1]/\
                                                                     (left_out_max_min_sentiment_features[left_out_claim].max_claim_sentiment_label-\
                                                                     left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_label)))
                                        else:
                                            curr_features_vec.append(0)
                                        if self.claim_sen_sentiment_vector_and_label_dict.has_key((str(curr_claim),str(claim_sentences_dict[curr_claim].index(sen)))):
                                            curr_features_vec.append(float(self.claim_sen_sentiment_vector_and_label_dict[str(curr_claim),str(claim_sentences_dict[curr_claim].index(sen))][1]/\
                                            (left_out_max_min_sentiment_features[left_out_claim].max_sentence_sentiment_label-\
                                            left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_label)))
                                        else:
                                            curr_features_vec.append(0)
                                    if "entropy" in self.features_setup:
                                        if self.claim_sentiment_vector_entropy.has_key(str(curr_claim)):
                                            curr_features_vec.append(float(self.claim_sentiment_vector_entropy[str(curr_claim)]/\
                                                                     (left_out_max_min_sentiment_features[left_out_claim].max_claim_sentiment_entropy-\
                                                                     left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_entropy)))
                                        else:
                                            curr_features_vec.append(0)
                                        if self.sentence_sentiment_vector_entropy.has_key((str(curr_claim),str(claim_sentences_dict[curr_claim].index(sen)))):
                                            curr_features_vec.append(float(self.sentence_sentiment_vector_entropy[(str(curr_claim),str(claim_sentences_dict[curr_claim].index(sen)))]/\
                                                                     (left_out_max_min_sentiment_features[left_out_claim].max_sentence_sentiment_entropy-\
                                                                     left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_entropy)))
                                        else:
                                            curr_features_vec.append(0)
                                if "semantic" in self.features_setup :
                                    if self.clm_sen_semantic_similarity_dict.has_key((claim_dict[str(curr_claim)],sen)):
                                        curr_features_vec.append((float(self.clm_sen_semantic_similarity_dict[(claim_dict[str(curr_claim)],sen)]/\
                                                                       left_out_max_min_semantic_features[left_out_claim].max_semantic_score-\
                                                                       left_out_max_min_semantic_features[left_out_claim].min_semantic_score)))
                                    else:
                                        curr_features_vec.append(0)
                                sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                                sen_no_space = sen_no_punct.replace(" ","")
                                if support_scores_dict.has_key((claim_dict[str(curr_claim)], sen_no_space)):
                                    curr_support_score = support_scores_dict[(claim_dict[str(curr_claim)],sen_no_space)]
    #                                 print " found " + claim_dict[str(curr_claim)] +" "+ sen +" supp score:"+ str(curr_support_score)
                                else:
                                    curr_support_score = 0
                                line += str(curr_support_score) + " qid:"+ str(curr_claim)+" "
                                for feature_idx in range(0,len(curr_features_vec)):
                                        line += str(feature_idx+1)+":"+str(curr_features_vec[feature_idx])+" "
    #                             line += "\n"
                                line += "#"+claim_dict[str(curr_claim)] +"|"+ sen +"\n" 
                                curr_train_LOOCV.write(line)
                            except Exception as err: 
                                sys.stderr.write("problem in write_train_test_files_SVM")     
                                print err.args      
                                print err
                        else:
                            dups_cnt += 1
                print curr_claim, "dups", dups_cnt ,len(sentences_set)," sentences:"
            curr_train_LOOCV.close()
            curr_claim = ""
            for docid in self.claim_entity_doc_CE_scores_dict[left_out_claim]:
                for sen in self.claim_entity_sen_CE_scores_dict[left_out_claim][docid].keys():
                    curr_features_vec = []
                    line = ""
                    if "CE" in self.features_setup:
                        if "claim_title" in self.features_setup or "all" in self.features_setup :
                            curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_claim_title)/\
                                                 (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_title-left_out_max_min_CE_features[left_out_claim].min_CE_claim_title)))
                        if "claim_body" in self.features_setup or "all" in self.features_setup:
                            curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_claim_body)/\
                                                           (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_body-left_out_max_min_CE_features[left_out_claim].min_CE_claim_body)))
                        if "entity_title" in self.features_setup or "all" in self.features_setup: 
                            curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_entity_title)/\
                                                     (float(left_out_max_min_CE_features[left_out_claim].max_CE_entity_title-left_out_max_min_CE_features[left_out_claim].min_CE_entity_title)))
                        if "entity_body" in self.features_setup or "all" in self.features_setup:
                            curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_entity_body)/\
                                                    (float(left_out_max_min_CE_features[left_out_claim].max_CE_entity_body-left_out_max_min_CE_features[left_out_claim].min_CE_entity_body)))
                        if "claim_sentence" in self.features_setup or "all" in self.features_setup:
                            curr_features_vec.append(float(self.claim_entity_sen_CE_scores_dict[left_out_claim][docid][sen].CE_claim_sentence)/\
                                                    (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_sentence-left_out_max_min_CE_features[left_out_claim].min_CE_claim_sentence)))
                        if "entity_sentence" in self.features_setup or "all" in self.features_setup:
                            curr_features_vec.append(float(self.claim_entity_sen_CE_scores_dict[left_out_claim][docid][sen].CE_entity_sentence)/\
                                                (float(left_out_max_min_CE_features[left_out_claim].max_CE_entity_sentence-left_out_max_min_CE_features[left_out_claim].min_CE_entity_sentence)))
#                                         
#                         curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_claim_title)/\
#                                                  (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_title-left_out_max_min_CE_features[left_out_claim].min_CE_claim_title)))
#                         curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_claim_body/\
#                                                        (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_body-left_out_max_min_CE_features[left_out_claim].min_CE_claim_body))))
#                         curr_features_vec.append(float(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_entity_title)/\
#                                                  (float(left_out_max_min_CE_features[left_out_claim].max_CE_claim_body-left_out_max_min_CE_features[left_out_claim].min_CE_claim_body)))
#                         curr_features_vec.append(self.claim_entity_doc_CE_scores_dict[left_out_claim][docid].CE_entity_body)
#                         curr_features_vec.append(self.claim_entity_sen_CE_scores_dict[left_out_claim][docid][sen].CE_claim_sentence)
#                         curr_features_vec.append(self.claim_entity_sen_CE_scores_dict[left_out_claim][docid][sen].CE_entity_sentence)
                    if "sentiment" in self.features_setup :
                        if "sentiment_sim" in self.features_setup:
                            if self.clm_sen_sentiment_similarity_dict.has_key((claim_dict[str(left_out_claim)],sen)):
                                        curr_features_vec.append(float(self.clm_sen_sentiment_similarity_dict[(claim_dict[str(left_out_claim)],sen)]/\
                                                                       left_out_max_min_sentiment_features[left_out_claim].max_sentiment_score-\
                                                                       left_out_max_min_sentiment_features[left_out_claim].min_sentiment_score))
                            else:
                                curr_features_vec.append(0)
                        if "label" in self.features_setup:
                            if self.claim_sentiment_vector_and_label_dict.has_key(str(left_out_claim)):
                                curr_features_vec.append(float(self.claim_sentiment_vector_and_label_dict[str(left_out_claim)][1]/\
                                                         (left_out_max_min_sentiment_features[left_out_claim].max_claim_sentiment_label-\
                                                         left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_label)))
                            else:
                                curr_features_vec.append(0)
                            if self.claim_sen_sentiment_vector_and_label_dict.has_key((str(left_out_claim),str(claim_sentences_dict[left_out_claim].index(sen)))):
                                curr_features_vec.append(float(self.claim_sen_sentiment_vector_and_label_dict[str(left_out_claim),str(claim_sentences_dict[left_out_claim].index(sen))][1]/\
                                                    (left_out_max_min_sentiment_features[left_out_claim].max_sentence_sentiment_label-\
                                                    left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_label))) 
                            else:
                                curr_features_vec.append(0)
                        if "entropy" in self.features_setup:
                            if self.claim_sentiment_vector_entropy.has_key(str(left_out_claim)):
                                curr_features_vec.append(float(self.claim_sentiment_vector_entropy[str(left_out_claim)]/\
                                                         (left_out_max_min_sentiment_features[left_out_claim].max_claim_sentiment_entropy-\
                                                         left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_entropy))) 
                            else:
                                curr_features_vec.append(0)
                            if self.sentence_sentiment_vector_entropy.has_key((str(left_out_claim),str(claim_sentences_dict[left_out_claim].index(sen)))):
                                curr_features_vec.append(float(self.sentence_sentiment_vector_entropy[(str(left_out_claim),str(claim_sentences_dict[left_out_claim].index(sen)))]/\
                                                         (left_out_max_min_sentiment_features[left_out_claim].max_sentence_sentiment_entropy-\
                                                         left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_entropy)))
                            else:
                                curr_features_vec.append(0)
                    if "semantic" in self.features_setup:
                        if self.clm_sen_semantic_similarity_dict.has_key((claim_dict[str(left_out_claim)],sen)):
                                    curr_features_vec.append((float(self.clm_sen_semantic_similarity_dict[(claim_dict[str(left_out_claim)],sen)]/\
                                                                   left_out_max_min_semantic_features[left_out_claim].max_semantic_score-\
                                                                   left_out_max_min_semantic_features[left_out_claim].min_semantic_score)))
                        else:
                            curr_features_vec.append(0)
                    sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                    sen_no_space = sen_no_punct.replace(" ","")
                    if support_scores_dict.has_key((claim_dict[str(left_out_claim)],sen_no_space)):
                        curr_support_score = support_scores_dict[claim_dict[str(left_out_claim)],sen_no_space]
                        print" found"
                    else:
                        curr_support_score = 0 
                    line += str(curr_support_score) + " qid:" + str(left_out_claim) +" "
                    for feature_idx in range(0,len(curr_features_vec)):
                            line += str(feature_idx+1)+":"+str(curr_features_vec[feature_idx]) +" "
                    line += "#" + claim_dict[str(left_out_claim)] +"|"+ sen +"\n"
#                     line += "\n"
                    curr_test_LOOCV.write(line)
            curr_test_LOOCV.close()    
            print "finished writing files for claim", left_out_claim
    
    def calc_num_of_support_sentences_in_data(self):
        """
        for each claim, calc the number of sentences in the data
        that are labeled
        """
        exclude = set(string.punctuation)
        support_scores_dict = self.convert_support_scores_dict_to_chars_as_sen()
#         self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized")
#         self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized")
        claim_sentences_dict = self.read_pickle("support_baseline_claim_sentences")
        claim_dict = self.read_pickle("claim_dict")
        claim_num_supp_sentences = {} #key- claim num, value - number of supporting sentences labled
        
        for curr_claim in self.claim_list:
            labeled_sen_cnt = 0.0
            supp_sen_cnt = 0.0
            print "    curr train claim:" +str(curr_claim)
            for sen in claim_sentences_dict[curr_claim]:
                sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                sen_no_space = sen_no_punct.replace(" ","")
                if support_scores_dict.has_key((claim_dict[str(curr_claim)],sen_no_space)):
                    labeled_sen_cnt += 1
                    if support_scores_dict[(claim_dict[str(curr_claim)],sen_no_space)] != 0:
                        supp_sen_cnt += 1  
            labeled_sen_percent = float(labeled_sen_cnt/float(len(claim_sentences_dict[curr_claim])))
            supp_sen_percent = float(supp_sen_cnt/float(len(claim_sentences_dict[curr_claim])))
            claim_num_supp_sentences[curr_claim] = (labeled_sen_cnt, labeled_sen_percent, supp_sen_cnt, supp_sen_percent)
        
        for claim in claim_num_supp_sentences.keys():
            print claim, claim_num_supp_sentences[claim]
        
    def read_predicted_support_score(self):
        clm_sen_predicition_score_dict_sorted = {}
#         claim_num_and_text = self.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict_pickle')
                                                                                                                                                        #flag of whether the entity is the doc title or in the sen itself
    #prediction score
        prediction_score_dict = {}
        
        for clm_num in self.claim_list:
            curr_pred_file = open(self.prediction_path+"\\"+str(clm_num)+"_prediction", 'r').read().strip()
            curr_test_file = open(self.test_path+"\\"+"test_clm_num_"+str(clm_num)+"_CV", 'r').read().strip()
#                 prediction_score_dict = {}
            sen_dict = {} #key is a line number from the file, val is a sen
            for i, line in enumerate(curr_pred_file.split('\n')):
#                     prediction_score_dict[i] = float(line)
                prediction_score_dict[i] = float(line)
            for i, line in enumerate(curr_test_file.split('\n')):
                #need to check if # is also in the sen itself, meaning if there are m
                if line.count("#") >1:
                    sen = line.split("#",1)[1].split("|")[1]
                else:
                    sen = line.split("#")[1].split("|")[1]
                sen_dict[i] = sen
            clm = line.split("#")[1].split("|")[0]
#                 sen_predicted_score_sorted = sorted(zip(sen_dict.values(), prediction_score_dict.values()),key=lambda x: (float(x[1])), reverse=True)
            sen_predicted_score_sorted = sorted(zip(sen_dict.values(), prediction_score_dict.values()),key=lambda x: (float(x[1])), reverse=True)
            clm_sen_predicition_score_dict_sorted[clm] = (sen_predicted_score_sorted) #key is clm , value is a list of sen and the predicted score
            
        with open("sort_sen_per_clm_pred_"+self.kernel+"_"+".csv", 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,sen_predicted_score_list) in clm_sen_predicition_score_dict_sorted.items():
                    for (sen, score) in sen_predicted_score_list:
                        w.writerow([clm,sen,str(score)])
        self.save_pickle("clm_as_key_sen_predicted_support_score_val_"+self.kernel, clm_sen_predicition_score_dict_sorted)               

    def process_SVM_rank_prediction_results(self,p):
        """
        the SVM rank output are scores for each claim,
        This function creates a ranking according to these scores.
        For each line in the test file, for which there is a claim and a sentence:
        """
        claim_dict = self.read_pickle("claim_dict")
        #SVM rank support score -read each predicition file, sort the 60 sentences, and calc ndcg
        clm_sen_predicition_score_dict_wiki = self.read_pickle("clm_as_key_sen_predicted_support_score_val_"+self.kernel)
    #         clm_as_key_sen_support_score_val_RT = utils.read_pickle("clm_as_key_sen_support_score_val_RT")
        clm_as_key_sen_support_score_val_wiki = self.read_pickle("clm_as_key_sen_support_score_val_wiki")    
    #     separated_list = [(clm_sen_predicition_score_dict_RT,clm_as_key_sen_support_score_val_RT,"RT"),(clm_sen_predicition_score_dict_wiki,clm_as_key_sen_support_score_val_wiki,"wiki")]
        separated_list = [(clm_sen_predicition_score_dict_wiki,clm_as_key_sen_support_score_val_wiki,"wiki")]
        for (curr_pred_supp_dict,curr_true_supp_dict,curr_source) in separated_list:
            NDCG_all_claims = {} #key is a claim, value is the nDCG
            AP_all_claims= {} 
            prec_at_5_all_claims = {}
            prec_at_10_all_claims = {}
            
            for clm in curr_true_supp_dict.keys():
                try:
                    if claim_dict[str(clm)]  in curr_pred_supp_dict.keys():
                        NDCG_all_claims[clm] = utils.calc_emp_NDCG(curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm],p)
                        AP_all_claims[clm] = utils.calc_AP_support(curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm],p)
                        prec_at_5_all_claims[clm] = utils.calc_precision_at_k(5, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                        prec_at_10_all_claims[clm] = utils.calc_precision_at_k(10, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                    
                except Exception as err: 
                    sys.stderr.write('problem in calc measures: in source: '+ curr_source+' in clm '+ claim_dict[str(clm)])     
                    print err.args      
                    print err
                    
            average_NDCG = float(float(sum(NDCG_all_claims.values()))/float(len(NDCG_all_claims)))
            MAP = float(float(sum(AP_all_claims.values()))/float(len(AP_all_claims)))
            average_prec_at_5 = float(float(sum(prec_at_5_all_claims.values()))/float(len(prec_at_5_all_claims)))
            average_prec_at_10 = float(float(sum(prec_at_10_all_claims.values()))/float(len(prec_at_10_all_claims)))
            std_NDCG = np.std(NDCG_all_claims.values())
            std_MAP = np.std(AP_all_claims.values())
            std_prec_at_5 = np.std(prec_at_5_all_claims.values())
            std_prec_at_10 = np.std(prec_at_10_all_claims  .values())
            
            print curr_source+ ": in average_NDCG: " +str(average_NDCG) +" std: "+str(std_NDCG) +" MAP :" +str(MAP) + " std:"+str(std_MAP)+ " average_prec_at_5:"+ str(average_prec_at_5) +" std:"+ str(std_prec_at_5)+" average_prec_at_10:"+str(average_prec_at_10) + " std:"+ str(std_prec_at_10)
            all_claims_sorted_by_NDCG_feature = collections.OrderedDict(sorted(NDCG_all_claims.items(),key=lambda x: (float(x[1])), reverse=True))
             
            with open('SVM_rank_nDCG@'+str(p)+"_"+self.kernel+".csv", 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,ndcg_score) in all_claims_sorted_by_NDCG_feature.items():
                    w.writerow([clm,ndcg_score])
                w.writerow(['average NDCG:'+str(average_NDCG)])
            
            all_claims_sorted_by_prec_at_5_feature = collections.OrderedDict(sorted(prec_at_5_all_claims.items(),key=lambda x: (float(x[1])), reverse=True))
             
            with open('SVM_rank_nDCG@'+str(p)+"_"+self.kernel+".csv", 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,prec_at_5) in all_claims_sorted_by_prec_at_5_feature.items():
                    w.writerow([clm,prec_at_5])
                w.writerow(['average NDCG:'+str(average_prec_at_5)])
          
def get_top_k_docs_id():    
    
    try:
        support_baseline_process = support_baseline()
        support_baseline_process.create_set_of_docs_per_claim()
        support_baseline_process.create_sen_ret_input_file()
        print "finished get_top_k_docs_id "
    
    except Exception as err: 
        sys.stderr.write("problem in get_top_k_docs_id")     
        print err.args      
        print err
      
def create_input_docs_for_SVM():
    features = "CE_all_sentiment_sim_label_entropy_semantic"
    support_baseline_process = support_baseline("none",features)
#     support_baseline_process.map_claim_doc_to_CE_scores()
#     support_baseline_process.map_claim_sen_to_CE_scores()
#     support_baseline_process.normalize_doc_CE_scores()
#     support_baseline_process.normalize_sen_CE_scores()
#     support_baseline_process.normalize_features()
    
    support_baseline_process.write_train_test_files_SVM()
 
def analyze_SVM_results():
    """
    possible features:
    for CE -  all the 6 composnents
    for sentiment -  sim (the sentiment similarity), label (the claim sentiment lable
            and the sentence sentiment label, and entropy - the claim sentiment vector entropy 
            and sentence sentiment vector entropy ( 
    for semantic -  semantic simply (the cosine similarity)::
        CE_all
        CE_all_sentiment_sim 
        CE_all_semantic
        CE_all_sentiment_sim_label_entropy #adding 4 features: the claim and sentence sentiment label, and 
                                                            the entropy of each sentiment vector (the confidence)
        CE_all_sentiment_sim_semantic
        CE_claim_title_claim_body_entity_title_entity_body
    """
    features = "CE_all_sentiment"
    support_baseline_process = support_baseline("linear",features)
    support_baseline_process.read_predicted_support_score()
    support_baseline_process.process_SVM_rank_prediction_results(10)
    
#     support_baseline_process.calc_num_of_support_sentences_in_data()
    return

 
def main():
    create_input_docs_for_SVM()
#     analyze_SVM_results()
    
    
if __name__ == '__main__':
    main()
