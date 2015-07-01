import sys
try:
    import cPickle as pickle
except:
    import pickle
import collections
import pandas as pd
import math
import string
from my_utils import  utils
import numpy as np
import svm_2_weight
import os
import csv

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
    max_claim_sentence_sentiment_label_diff = 0.0
    min_claim_sentence_sentiment_label_diff = 0.0
    max_claim_pos_words_ratio = 0.0
    max_claim_neg_words_ratio = 0.0
    min_claim_pos_words_ratio = 0.0
    min_claim_neg_words_ratio = 0.0 
    max_sen_pos_words_ratio = 0.0
    max_sen_neg_words_ratio = 0.0
    min_sen_pos_words_ratio = 0.0
    min_sen_neg_words_ratio = 0.0
    
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

class max_min_NLP_scores_keeper():
    max_sen_len = 0.0
    min_sen_len = 0.0
    max_stop_words_ratio = 0.0
    min_stop_words_ratio = 0.0
    max_dep_type_ratio = 0.0
    min_dep_type_ratio = 0.0
    max_NER_ratio = 0.0
    min_NER_ratio = 0.0
    
    def init(self):
        max_sen_len = 0.0
        min_sen_len = 0.0
        max_stop_words_ratio = 0.0
        min_stop_words_ratio = 0.0
        max_dep_type_ratio = 0.0
        min_dep_type_ratio = 0.0
        max_NER_ratio = 0.0
        min_NER_ratio = 0.0

class max_min_objLM_scores_keeper():
    max_1_star_prob = 0.0
    max_2_star_prob = 0.0
    max_3_star_prob = 0.0
    max_4_star_prob = 0.0
    max_5_star_prob = 0.0
    min_1_star_prob = 0.0
    min_2_star_prob = 0.0
    min_3_star_prob = 0.0
    min_4_star_prob = 0.0
    min_5_star_prob = 0.0
    
    
    def init(self):
        max_1_star_prob = 0.0
        max_2_star_prob = 0.0
        max_3_star_prob = 0.0
        max_4_star_prob = 0.0
        max_5_star_prob = 0.0
        mim_1_star_prob = 0.0
        min_2_star_prob = 0.0
        min_3_star_prob = 0.0
        min_4_star_prob = 0.0
        min_5_star_prob = 0.0

class max_min_MRF_scores_keeper():
    max_doc_MRF_score = 0.0
    min_doc_MRF_score = 0.0
    max_sen_MRF_score = 0.0
    min_sen_MRF_score = 0.0
    
    def init(self):
        max_doc_MRF_score = 0.0
        min_doc_MRF_score = 0.0
        max_sen_MRF_score = 0.0
        min_sen_MRF_score = 0.0
    
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
    
    def  __init__(self,kernel, features_setup, target, sentences_num):
#         self.relevance_linux_base_path = r"/lv_local/home/liorab/softwares/indri-5.5/retrieval_baselines"
        self.relevance_linux_base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\relevance_baselines\\"
#         self.relevance_base_path = r"/home/liorab/softwares/indri-5.5/retrieval_baselines"
        self.support_linux_base_path = r"/lv_local/home/liorab/softwares/indri-5.5/retrieval_baselines_support"
#         self.support_linux_base_path = r"/lv_local/home/liorab/support_test/"
#         self.support_linux_base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\\"
        self.features_setup = features_setup
#         self.support_linux_base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines"
        self.relevance_comparison_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\relevance_comparison"
        self.contradict_comparison_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\contradict_comparison"
        self.sentences_num = sentences_num 
        """#two setups - one in allSentences, using all the sentences
         retrieved from the top 50 documents across all alpha, beta values
         this is a one-phase for relevance/support
         second option -top1000Sentences, where we take from the top 50 documents the 
         1000 top sentences as retrived from the alpha, beta and lamda values that
         gave the highest MAP in the relevance baseline
         this is as a two-phase model - first the relevance, and then on that learn
         the features with respect to respect 
        """
        self.SVM_path = self.support_linux_base_path+r"/SVM_zero_two_scale_"+self.sentences_num+"\\" + self.features_setup +r"_features"
#         self.SVM_path = self.support_linux_base_path+ "/SVM_zero_two_scale/"+ self.features_setup+r"_features"       
        self.SVM_path_relevance = self.relevance_comparison_path +"\SVM_"+self.sentences_num+ "\\"+ self.features_setup +r"_features"
        self.SVM_path_contra = self.contradict_comparison_path+"\SVM_zero_two_scale_"+self.sentences_num + "\\"+ self.features_setup +r"_features"
        self.train_path = self.SVM_path+r"\train"+"\\"
        self.test_path = self.SVM_path +r"\test"+"\\"
        self.model_path = self.SVM_path +r"\model"+"\\"
        self.prediction_path = self.SVM_path +r"\prediction"+"\\"
        self.train_path_relevance = self.SVM_path_relevance+ r"\train"+"\\"
        self.test_path_relevance = self.SVM_path_relevance+ r"\test"+"\\"
        self.model_path_relevance = self.SVM_path_relevance+ r"\model"+"\\"
        self.prediction_path_relevance = self.SVM_path_relevance+ r"\prediction"+"\\"
        self.train_path_contra = self.SVM_path_contra + r"\train"+"\\"
        self.test_path_contra = self.SVM_path_contra + r"\test"+"\\"
        self.model_path_contra = self.SVM_path_contra + r"\model"+"\\"
        self.prediction_path_contra = self.SVM_path_contra + r"\prediction"+"\\"
        
        self.docid_set_per_claim = {}
        self.top_k_docs = 1000
#         self.claim_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,54,55,57,58,59,60,61,66]
        self.claim_list_dict = { 
                   "movies":[4,7,17,21,29,32,36,37,39,40,41,42,45,46,47,54,55,57,58,59,60,61,66,81,83,85,98,101,103,104,105,106,107,108,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126],
                    "sports":[127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176]
                  }
        self.claim_entity_sentence_sen_output_path = self.support_linux_base_path+r"\claimEntity_sen_output\\"
        self.claim_entity_body_title_output_path = self.support_linux_base_path+r"\claimEntity_bodyTitle_output\\" #ieir14
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
        self.claim_sen_sentiment_pos_words_ratio = {}
        self.claim_sen_sentiment_neg_words_ratio = {}
        self.claim_sentiment_pos_words_ratio = {}
        self.claim_sentiment_neg_words_ratio = {}
        self.sen_length = {}
        self.sw_ratio = {}
        self.entailment_res = {}
        self.obj_LM_dist = {}
        self.claim_doc_MRF_scores = {}
        self.claim_sen_MRF_scores = {}
        self.NER_ratio = {}
        self.typedDep_bin = {}
        self.typedDep_num = 47 
        self.target = target #the target to learn/report on -  supportivness or relevance, 
                            # and metric divergence - learn on the support, report on the relevance and vice versa
        self.left_out_max_min_CE_features = {}
        self.left_out_max_min_sentiment_features = {}
        self.left_out_max_min_semantic_features = {}
        self.left_out_max_min_NLP_features = {}
        self.left_out_max_min_objLM_features = {}
        self.left_out_max_min_MRF_features = {}
        self.claim_dict = {}
        self.claim_sentences_dict = {}
        self.claim_num_sentences_num_sentences_text_dict = {} # key is claim_num, another key is sentence num, and value is sentence
        self.claim_num_sentences_text_sentences_num_dict = {} # key is claim_num, another key is sentence text, and value is sentence num
        self.claim_num_sentences_no_punct_sentences_num_dict = {} # key is claim_num, another key is sentence text with no punctuation, and value is sentence num
        self.claim_sentences_docid_mapping = {}
        self.docid_doctitle_dict = {}
        
        self.corpus_beta = 0.1
        self.corpus_beta_int = int(10*self.corpus_beta)
        
        self.domain = "movies"
        
    def read_pickle(self,file_name):
        d = {}
        with open(file_name, 'rb') as handle:
            d = pickle.loads(handle.read())
        return d
#         d_to_return = {}
#         for d_claim_num in d.keys():
#             d_to_return[d_claim_num] = {}
#             for d_key in d[d_claim_num].keys():
#                 ces_scores = CEScores()
#                 for ces_key in d[d_claim_num][d_key].keys():
#                     setattr(ces_scores, ces_key, d[d_claim_num][d_key][ces_key])
#                 d_to_return[d_claim_num][d_key] = ces_scores
#         return d_to_return
    
    def save_pickle(self,file_name,d):
        with open(file_name, 'wb') as handle:
            pickle.dump(d, handle)
#         handle.close()
#         d_to_save = {}
#         ces_keys = ["CE_claim_title","CE_claim_body", "CE_entity_title", "CE_entity_body", "CE_claim_sentence", "CE_entity_sentence"]
#         for d_claim_num in d.keys():
#             d_to_save[d_claim_num] = {}
#             for d_key in d[d_claim_num].keys():
#                 d_to_save[d_claim_num][d_key] = {}
#                 ces_obj = d[d_claim_num][d_key]
#                 for ces_key in ces_keys:
#                     ces_val = getattr(ces_obj,ces_key)
#                     d_to_save[d_claim_num][d_key][ces_key] = ces_val 
            
#         with open(file_name, 'wb') as handle:
#             pickle.dump(d_to_save, handle)
#         handle.close()
    
    def read_old_pickle_testing(self,file_name):
        d = {}
        with open(file_name, 'rb') as handle:
            d = pickle.loads(handle.read())
        return d
    
    def create_set_of_docs_per_claim(self):
        """
        from each alpha, beta value from the relevance baseline,
        take the top k (50) docs, 
        that will serve as the workingDoc for the support sentence retrieval
        """
        print " creating set of docs per claim..."
        #02.06.15 update - 
#         doc_res_dicts_path = self.relevance_linux_base_path+"/docs_norm_scores_dicts" #ieir31
        doc_res_dicts_path =  self.relevance_linux_base_path+"claimLM_docLM_doc_ret_output\\" #in ieir60 for instance
        for claim_num in self.claim_list:
            
            print "\tin claim",claim_num
            self.docid_set_per_claim = {}
            curr_set = set()
#             for filename in os.listdir(doc_res_dicts_path):
            for alpha in range(0,11,1):
                for beta in range(0,11-self.corpus_beta_int,1):
                    (alpha_f,beta_f) = self.turn_to_float([alpha,beta])
#                 if not "clm_key_ranked_list_of_docs_" in filename:
                    filename = "relevance_baseline_claim_document_retrieval_normalized_score_dict_sorted_alpha_"+str(alpha_f)+"_beta_"+\
                        str(beta_f)+"_clm_"+str(claim_num)
                    curr_dict = self.read_pickle(doc_res_dicts_path+filename)
#                     curr_claim_num = filename.split("_clm_")[1].split("_dict_sorted")[0]
#                     curr_claim_num = filename.split("_clm_")[1]
#                     print "    curr claim num",curr_claim_num
#                     if str(claim_num) == curr_claim_num:
                    print "\tfound file, opening..."
                    #02.06.15 update
#                         top_k_docs = [key[1] for key in curr_dict.keys()][0:self.top_k_docs]
                    top_k_docs = [key for key in curr_dict.keys()][0:self.top_k_docs]
                    for docid in top_k_docs:
                        curr_set.add(docid)
                    print "\tcurr_set len", len(curr_set)
            self.docid_set_per_claim[claim_num] = curr_set
            print "len docid_set_per_claim[clain_num]" ,len(self.docid_set_per_claim[claim_num])      
        utils.save_pickle("support_baseline_docid_set_per_claim", self.docid_set_per_claim)
        print "finished create_set_of_docs_per_claim"
     
    def turn_to_float(self,my_input):
        output = []
        if len(my_input)>1:
            for number in my_input:
                if number == 0:
                    number_f = number
                elif number == 10:
                    number_f = 1
                else:
                    number_f = float(float(number)/float(10))
                output.append(number_f)
            return output
        else:
            my_input_f = 0
            if my_input[0] == 0:
                    my_input_f = my_input[0]
            elif my_input[0] == 10:
                    my_input_f = 1
            else:
                my_input_f = float(float(my_input[0])/float(10))
            return my_input_f
       
    def map_claim_and_sentences_num(self):
        self.claim_sentences_dict = utils.read_pickle("support_baseline_claim_sentences")
        self.claims_dict = utils.read_pickle("claim_dict")
        exclude = set(string.punctuation)
        for (clm_num, sentences_list) in self.claim_sentences_dict.items():
            self.claim_num_sentences_num_sentences_text_dict[clm_num] = {}
            self.claim_num_sentences_text_sentences_num_dict[clm_num] = {}
            self.claim_num_sentences_no_punct_sentences_num_dict[clm_num] = {}
            
            print "in claim", clm_num,  " sentences" ,len(sentences_list)
             
            for sen_num in range(0,len(sentences_list)):
                self.claim_num_sentences_num_sentences_text_dict[clm_num][sen_num] = sentences_list[sen_num].strip()
                if self.claim_num_sentences_text_sentences_num_dict[clm_num].has_key(sentences_list[sen_num].strip()):
                    self.claim_num_sentences_text_sentences_num_dict[clm_num][sentences_list[sen_num].strip()].append(sen_num)
                else:
                    self.claim_num_sentences_text_sentences_num_dict[clm_num][sentences_list[sen_num].strip()] = [sen_num]
                sen_no_punct = ''.join(ch for ch in sentences_list[sen_num].strip() if ch not in exclude)
                sen_no_space = sen_no_punct.replace(" ","") 
                self.claim_num_sentences_no_punct_sentences_num_dict[clm_num][sen_no_space] = sen_num
        self.save_pickle("support_baseline_claim_num_sentences_num_sentences_text_dict_allSentences", self.claim_num_sentences_num_sentences_text_dict)
        self.save_pickle("support_baseline_claim_num_sentences_text_sentences_num_dict_allSentences", self.claim_num_sentences_text_sentences_num_dict)
        self.save_pickle("support_baseline_claim_num_sentences_no_punct_sentences_num_dict_allSentences", self.claim_num_sentences_no_punct_sentences_num_dict)
    
    def create_sen_ret_input_file(self):
        print "creating sentence ret files..."
        try:
            sen_ret_input_path = self.support_linux_base_path+"sentence_ret_input"
            claims_no_SW_dict = self.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\claims\claims_"+self.domain+"_no_SW_dict")
            #28.06.15 change -  per domain
            MRF_ret_docids = self.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\retrieveal_process\MRF_doc_res_"+self.domain)  # structure: doc_res_dict[claim_num][docid] = score
#             for claim_num in self.claim_list:
            for claim_num in self.claim_list_dict[self.domain]:
                print "    in claim", claim_num
                # update 28.06.15 -  according to the fact that we are now evaluating the top 100 MRF sentenced only
#                 self.docid_set_per_claim[claim_num] = utils.read_pickle("support_baseline_docid_set_per_claim_"+str(claim_num))
                self.docid_set_per_claim[claim_num] = MRF_ret_docids[claim_num].keys()
                sen_ret_docno_file = open(sen_ret_input_path+"\sen_ret_top_k_docs_"+str(self.top_k_docs)+"_clm_"+str(claim_num),"wb")
                sen_ret_docno_file.write("<parameters>\n")
                sen_ret_docno_file.write("<query><number>"+str(claim_num)+"</number><text>"+claims_no_SW_dict[claim_num][0].strip()+"|"+claims_no_SW_dict[claim_num][1].strip()+"</text>")
                for workingDoc in self.docid_set_per_claim[claim_num]:
                    sen_ret_docno_file.write("<workingSetDocno>"+workingDoc+"</workingSetDocno>")
                sen_ret_docno_file.write("</query>\n")
                sen_ret_docno_file.write("</parameters>")
                sen_ret_docno_file.close()
        except Exception as err: 
            sys.stderr.write("problem in create_sen_ret_input_file")     
            print err.args      
            print err
    
    def create_held_out_claim_best_configuration_mapping(self):
        """
        for the top sentences setup,
        have a dict :key- held out claim_num, value is the alpha,beta,lambda configuration that 
        gave the highest AP in the relevance baseline process, based on the retrieval of 50 top docs
        """
        held_out_claim_best_configuration_relevance_baseline = {4:(0.5,0,0.8),7:(0.5,0,0.8),17:(0.5,0,0.8),21:(0.5,0,0.8),  
        36:(0.5,0,0.8),37:(0.5,0,0.8),39:(0.5,0,0.8),40:(0.5,0,0.8),41:(0.5,0,0.8),42:(0.5,0,0.8),
        45:(0.5,0,0.8),46:(0.5,0,0.8), 47:(0.5,0,0.8), 50:(0.5,0,0.8),51:(0.5,0,0.8), 53:(0.8,0,0.7),                        
        54:(0.5,0,0.8),55:(0.5,0,0.8),57:(0.5,0,0.8),58:(0.5,0,0.8),59:(0.5,0,0.8),60:(0.5,0,0.8),
        61:(0.5,0,0.8),62:(0.5,0,0.8),66:(0.5,0,0.8),69:(0.5,0,0.8),70:(0.5,0.3,0.7),79:(0.5,0,0.8),
        80:(0.5,0,0.8)}
        self.save_pickle("held_out_claim_best_configuration_relevance_baseline", held_out_claim_best_configuration_relevance_baseline)
                
    def get_top_sentences_as_two_stage_process(self):

        """
        For the second setup - using the top 1000 sentences:  first stage is based on the relevance baseline,
        second step is the supportivness features learning.
        The relevance comparison - top 1000 sentences: for each claim, get its top 1000 sentences according to the 
        alpha, beta, lambda configuration that gave the highest AP for it. 
        For every held-out claim,  get the top 1000 sentences for the each of the other train claims,
        as taken from the config' that gave the highest AP on them.
        """
            #top_sentences_dict = self.read_pickle(self.support_linux_base_path+"\\SVM_zero_two_scale_topSentences\clm_num_key_final_ranked_list_sen_alpha_0.8_beta_0_top_k_docs_50_lambda_0.8_sorted")
        self.create_held_out_claim_best_configuration_mapping()
        held_out_claim_best_configuration_relevance_baseline = self.read_pickle("held_out_claim_best_configuration_relevance_baseline")
        relevance_baseline_topSentences = {} # #key is held out claim num, value is a another key value, key is a train claim num, value is a list of sens
        sentence_num = 1000
        for left_out_claim in self.claim_list:
            relevance_baseline_topSentences[left_out_claim] = {}
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            alpha,beta,lamda = held_out_claim_best_configuration_relevance_baseline[left_out_claim]
            curr_sentences_file = self.read_pickle(self.support_linux_base_path+"\\SVM_zero_two_scale_topSentences\clm_num_key_final_ranked_list_sen_alpha_"+str(alpha)+"_beta_"+str(beta)+"_top_k_docs_50_lambda_"+str(lamda)+"_sorted")
            for curr_train_claim in curr_train_claims:
                relevance_baseline_topSentences[left_out_claim][curr_train_claim] = []
                curr_sen_list = curr_sentences_file[str(curr_train_claim)]
                for (sen, retrieval_score) in curr_sen_list:            
                    if len(relevance_baseline_topSentences[left_out_claim][curr_train_claim]) < sentence_num:
                        relevance_baseline_topSentences[left_out_claim][curr_train_claim].append((sen,retrieval_score))
                    else:
                        print "finished getting 1000 top sentences for left_out_claim "+str(left_out_claim)+" and train claim "+str(curr_train_claim)
                
#         for (clm,sen_list) in top_sentences_dict.items():
#             for (sen,retrieval_score) in sen_list:
#                 if relevance_baseline_topSentences.has_key(clm):
#                     if len(relevance_baseline_topSentences[clm]) < sentence_num:
#                         relevance_baseline_topSentences[clm].append((sen,retrieval_score))
#                 else:
#                     relevance_baseline_topSentences[clm] = [(sen,retrieval_score)]
        self.save_pickle("relevance_baseline_topSentences", relevance_baseline_topSentences)
        print "finished get_top_sentences_as_two_stage_process" 
        
    def map_sentences_to_their_docid(self):
        """
        for the two-phase setup, need to find for each sentence in the clm_num_key_final_ranked_list_sen_alpha_0.8_beta_0_top_k_docs_50_lambda_0.8_sorted
        dict,
        the docid of a sentence, to know each sentence doc component CE scores
        """
        print "in map_sentences_to_their_docid"
        relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
        held_out_claim_best_configuration_relevance_baseline = self.read_pickle("held_out_claim_best_configuration_relevance_baseline")
        relevance_baseline_topSentences_sens_docs_association = {} # key is a left_out_clm_num, then another value as key sentence,value is its docid
        relevance_baseline_topSentences_docs_association = {}  # key is a left_out_claim, value is key of train clm_num,value is a list of doc ids
#         sen_ret_path = "/home/liorab/baseline_ret/sen_ret_corpus_smoothing_res/claimLM_senLM_copus_smoothing_sen_ret_output_corpus_beta_0.1/"
        sen_ret_path = self.support_linux_base_path+r"\SVM_zero_two_scale_"+self.sentences_num+"\sen_res_alpha_beta_top_k_docs_50\\"
#         clm_num_list = '7' 
        for (left_out_clm_num) in relevance_baseline_topSentences.keys():
            print "in left out claim", left_out_clm_num
            left_out_clm_num = int(left_out_clm_num)
            relevance_baseline_topSentences_sens_docs_association[left_out_clm_num] = {}
            relevance_baseline_topSentences_docs_association[left_out_clm_num] = {}
            alpha,beta,lamda = held_out_claim_best_configuration_relevance_baseline[left_out_clm_num]
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_clm_num)
            for curr_train_claim in curr_train_claims:
                print "    in curr train claim", curr_train_claim
                curr_train_claim = int(curr_train_claim)
                sen_retrieval_file = open(sen_ret_path+"sen_res_alpha_"+str(alpha)+"_beta_"+str(beta)+"_top_k_docs_50_clm_"+str(curr_train_claim)).read().strip()
#             for curr_train_claim in curr_train_claims:
                relevance_baseline_topSentences_sens_docs_association[left_out_clm_num][curr_train_claim] = {}
                relevance_baseline_topSentences_docs_association[left_out_clm_num][curr_train_claim] = {}
            #open the clm_sen retrieval file
                curr_sen_ret_dict = {}
                #turn the file to dict -  key is sen, value is the docid
                for i, line in enumerate(sen_retrieval_file.split('\n')):
                    if i%2 == 0: # a metadata line
                        data = line.split(' ')
                        qId = data[0]
                        doc_id = data[2]
                    else:
                        curr_sen_ret_dict[line] = doc_id
                        qId = int(qId)
#                         if qId in curr_train_claims: 
#                         if qId in relevance_baseline_topSentences_docs_association[left_out_clm_num].keys():
                        if len(relevance_baseline_topSentences_docs_association[left_out_clm_num][qId]) != 0:
                            relevance_baseline_topSentences_docs_association[left_out_clm_num][qId].append((doc_id))
                        else:
                            relevance_baseline_topSentences_docs_association[left_out_clm_num][qId] = [doc_id]  
                for claim_num in relevance_baseline_topSentences[left_out_clm_num].keys():
                    for (sen, sen_score) in relevance_baseline_topSentences[left_out_clm_num][claim_num]:
                        if sen in curr_sen_ret_dict.keys():
                            relevance_baseline_topSentences_sens_docs_association[left_out_clm_num][curr_train_claim][sen] = curr_sen_ret_dict[sen]
                        elif sen.strip() in curr_sen_ret_dict.keys():
                            relevance_baseline_topSentences_sens_docs_association[left_out_clm_num][curr_train_claim][sen] = curr_sen_ret_dict[sen.strip()]
                        elif sen+" " in curr_sen_ret_dict.keys():
                            relevance_baseline_topSentences_sens_docs_association[left_out_clm_num][curr_train_claim][sen] = curr_sen_ret_dict[sen+" "]
                    
        self.save_pickle("relevance_baseline_topSentences_sens_docs_association", relevance_baseline_topSentences_sens_docs_association)
        self.save_pickle("relevance_baseline_topSentences_docs_association", relevance_baseline_topSentences_docs_association)        
    
    def map_claim_doc_to_CE_scores(self):
        # for the 4 CE scores sim(claim,body), sim(entity,body), sim(claim,title), sim(entity,title)
        #that are related to the document
        print " calc doc CE scores..."
        #1.07.2015 update - calc only for the docid that were retrived in the MRF retrieval
        MRF_retrieval_path = r"C:\Users\liorab\workspace\supporting_evidence\src\retrieveal_process\\"
        MRF_doc_res = self.read_pickle(MRF_retrieval_path+"MRF_doc_res_"+self.domain)
        for claim_entity_body_title_file in os.listdir(self.claim_entity_body_title_output_path):
            f = open(self.claim_entity_body_title_output_path+claim_entity_body_title_file, "rb")
            data = pd.read_csv(f," ")
            curr_claim = data["q_number"][0]
            if not self.claim_entity_doc_CE_scores_dict.has_key(curr_claim):
                self.claim_entity_doc_CE_scores_dict[curr_claim] = {} 
            docid = data["documentName"]  
            #1.07.2015 update - calc only for the docid that were retrived in the MRF retrieval
            if docid in MRF_doc_res[curr_claim].keys():
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
        
    def get_sen_num(self,clm_num,line):
        sen_num = -1
        if self.claim_num_sentences_text_sentences_num_dict[clm_num].has_key(line):
#             if len(self.claim_num_sentences_text_sentences_num_dict[clm_num][line]) == 1:
            sen_num = self.claim_num_sentences_text_sentences_num_dict[clm_num][line.strip()]
            # if there are several documents with 
#             elif len(self.claim_num_sentences_text_sentences_num_dict[clm_num][line]) > 1:
        return sen_num 
                   
    def map_claim_sen_to_CE_scores(self):           
        print " map_claim_sen_to_CE_scores..."
        #26/02/15 update - move to sen_num keeping instead of sentences
        #1.07.2015 update  -  move to handling the top 100 sentences from the MRF retrieval
#         self.claim_num_sentences_text_sentences_num_dict = self.read_pickle("support_baseline_claim_num_sentences_text_sentences_num_dict_allSentences")
        MRF_retrieval_path = r"C:\Users\liorab\workspace\supporting_evidence\src\retrieveal_process\\"
        MRF_doc_res = self.read_pickle(MRF_retrieval_path+"MRF_doc_res_"+self.domain)
        self.claim_num_sentences_text_sentences_num_dict = self.read_pickle(MRF_retrieval_path+"MRF_sen_text_sen_num_dict_"+self.domain) 
        
        for claim_entity_sen_file in os.listdir(self.claim_entity_sentence_sen_output_path):
            sen_file = open(self.claim_entity_sentence_sen_output_path + claim_entity_sen_file)
            sen = sen_file.read().strip() # score, sentence
            for i, line in enumerate(sen.split('\n')):                   
                if i%2 == 0: # a metadata line
                    data = line.split(' ')
                    curr_claim = int(data[0])
                    docid = data[2]
                    if not docid in MRF_doc_res[curr_claim].keys():
                        continue
                    sen_score = data[4]
                    if not self.claim_entity_sen_CE_scores_dict.has_key(curr_claim):
                        self.claim_entity_sen_CE_scores_dict[curr_claim] = {} 
#                     if not self.claim_entity_sen_CE_scores_dict[curr_claim].has_key(docid):
#                         self.claim_entity_sen_CE_scores_dict[curr_claim][docid] = {}
                else: #a sentence line
                    if self.claim_entity_sen_CE_scores_dict.has_key(curr_claim):
                        sen_num = -1
                        if self.claim_entity_sen_CE_scores_dict[curr_claim].has_key(docid):
                            # get the sentence num and keep it
#                             if self.claim_num_sentences_text_sentences_num_dict[curr_claim].has_key(line):
#                                 sen_num = self.claim_num_sentences_text_sentences_num_dict[curr_claim][line.strip()]
#                             elif self.claim_num_sentences_text_sentences_num_dict[curr_claim].has_key(line.strip()):
#                                 sen_num = self.claim_num_sentences_text_sentences_num_dict[curr_claim][line.strip()]
                            
                            sen_num = self.get_sen_num(curr_claim, line)
                            if sen_num == -1:
                                sen_num = self.get_sen_num(curr_claim, line.strip())
#                             if self.claim_entity_sen_CE_scores_dict[curr_claim][docid].has_key(line.strip()):
#                                 curr_CE_scores = self.claim_entity_sen_CE_scores_dict[curr_claim][docid][line.strip()]
                            if self.claim_entity_sen_CE_scores_dict[curr_claim][docid].has_key(sen_num):
                                curr_CE_scores = self.claim_entity_sen_CE_scores_dict[curr_claim][docid][sen_num]
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
#                     self.claim_entity_sen_CE_scores_dict[curr_claim][docid][line.strip()] = curr_CE_scores
                    sen_num = self.claim_num_sentences_text_sentences_num_dict[curr_claim][line.strip()]
                    self.claim_entity_sen_CE_scores_dict[curr_claim][docid][sen_num] = curr_CE_scores
        self.save_pickle("support_model_claim_entity_sen_CE_scores_dict",self.claim_entity_sen_CE_scores_dict)             
        print "finished map_claim_sen_to_CE_scores"
    
    def normalize_sen_CE_scores(self):
        self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict")
        for curr_claim in self.claim_list:
                for docid in self.claim_entity_sen_CE_scores_dict[curr_claim]:
                    for (sentence,CE_scores) in self.claim_entity_sen_CE_scores_dict[curr_claim][docid].items():
                        CE_scores.CE_claim_sentence = float(CE_scores.CE_claim_sentence/self.CE_claim_sentence_sum)
                        CE_scores.CE_entity_sentence = float(CE_scores.CE_entity_sentence/self.CE_entity_sentence_sum)
        self.save_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized", self.claim_entity_sen_CE_scores_dict)
    
    def convert_sentiment_dict(self):
        """
        Instead of having (clm_text, sentence) -> sentiment score dict form, 
        convert to clm_text -> [list of (sentence, score)]
        """
        print "converting sentiment dict.."
        self.clm_sen_sentiment_similarity_dict = self.read_pickle("support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted")
        self.sentence_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sen_sentiment_vector_entropy")
        self.claim_sen_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sen_sentiment_vector_and_label_dict")
        self.claim_num_sentences_text_sentences_num_dict = self.read_pickle("support_baseline_claim_num_sentences_text_sentences_num_dict_allSentences")
            
        claim_dict = self.read_pickle("claim_dict")
        converted_sentiment_sim_dict = {}
        converted_sentence_sentiment_entropy = {} 
        converted_sentence_vector_and_label = {}
        if self.sentences_num == "topSentences":
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")    
        if self.sentences_num == "allSentences":       
#             for ((clm_text,sen),score) in self.clm_sen_sentiment_similarity_dict.items():
            # 27/02/15 update -  move to clm_num,sen_num dict
            for ((clm_num,sen_num),score) in self.clm_sen_sentiment_similarity_dict.items():
                #TODO
                #get the curr claim num check if it is in the self.claim_num_sentences_text_sentences_num_dict dict 
                sen_num = self.claim_num_sentences_text_sentences_num_dict
                if converted_sentiment_sim_dict.has_key(clm_num):
                    converted_sentiment_sim_dict[clm_num].append((sen_num,score))
                else:
                    converted_sentiment_sim_dict[clm_num] = [(sen_num,score)]
    
            for (clm_num,sen_num) in self.sentence_sentiment_vector_entropy.keys():
                #if converted_sentence_sentiment_entropy.has_key(claim_dict[clm_num]):
                if converted_sentence_sentiment_entropy.has_key(clm_num):
                    converted_sentence_sentiment_entropy[clm_num].append((sen_num,self.sentence_sentiment_vector_entropy[clm_num,sen_num]))
                else:
                    converted_sentence_sentiment_entropy[clm_num] = [(sen_num,self.sentence_sentiment_vector_entropy[clm_num,sen_num])]
                
#                 if converted_sentence_vector_and_label.has_key(claim_dict[clm_num]):
                if converted_sentence_vector_and_label.has_key(clm_num):
                    converted_sentence_vector_and_label[clm_num].append(self.claim_sen_sentiment_vector_and_label_dict[clm_num,sen_num])
                else:
                    converted_sentence_vector_and_label[clm_num] = [self.claim_sen_sentiment_vector_and_label_dict[clm_num,sen_num]]
        
        elif self.sentences_num == "topSentences":
        #read the dict that holds the mapping between the sens and the sen numbers
            for left_out_claim in self.claim_list:
                converted_sentiment_sim_dict[left_out_claim] = {} 
                converted_sentence_sentiment_entropy[left_out_claim] = {}  
                converted_sentence_vector_and_label[left_out_claim] = {}
                for clm_num,sentence_list in relevance_baseline_topSentences[left_out_claim].items():
                    for sentence_text,score in sentence_list: 
                        if sentence_text in self.claim_num_sentences_text_sentences_num_dict[clm_num]:
                            sen_num = self.claim_num_sentences_text_sentences_num_dict[clm_num][sentence_text]
                        elif sentence_text.strip() in self.claim_num_sentences_text_sentences_num_dict[clm_num]:
                            sen_num = self.claim_num_sentences_text_sentences_num_dict[clm_num][sentence_text.strip()]
                            if self.clm_sen_sentiment_similarity_dict.has_key((clm_num,sen_num)):
                                score = self.clm_sen_sentiment_similarity_dict[clm_num,sen_num]                     
#                             elif self.clm_sen_sentiment_similarity_dict.has_key(clm_num,sentence_text.strip())):
#                                 sen, score = sentence_text.strip(), self.clm_sen_sentiment_similarity_dict[claim_dict[str(clm_num)],sentence_text.strip()]
                        if converted_sentiment_sim_dict.has_key(clm_num):
                            converted_sentiment_sim_dict[left_out_claim][clm_num].append((sen_num,score))
                        else:
                            converted_sentiment_sim_dict[left_out_claim][clm_num] = [(sen_num,score)]
                        if self.sentence_sentiment_vector_entropy.has_key(clm_num,sen_num):
                            if converted_sentence_sentiment_entropy[left_out_claim].has_key(clm_num):      
                                converted_sentence_sentiment_entropy[left_out_claim][clm_num].append((sen_num,self.sentence_sentiment_vector_entropy[clm_num,sen_num]))
                            else:
                                converted_sentence_sentiment_entropy[left_out_claim][clm_num] = [(sen_num,self.sentence_sentiment_vector_entropy[(clm_num,sen_num)])]
                        else:
                            print str(clm_num) ,str(sen_num) +" not in sentence_sentiment_vector_entropy"
                        if self.claim_sen_sentiment_vector_and_label_dict.has_key((clm_num,sen_num)):
                            if converted_sentence_vector_and_label[left_out_claim].has_key(clm_num):
                                converted_sentence_vector_and_label[left_out_claim][clm_num].append(self.claim_sen_sentiment_vector_and_label_dict[clm_num,sen_num])   
                            else:
                                converted_sentence_vector_and_label[left_out_claim][clm_num] = [self.claim_sen_sentiment_vector_and_label_dict[clm_num,sen_num]]
                        else:
                            print clm_num,str(sen_num) +" not in claim_sen_sentiment_vector_and_label_dict"
#         sum = 0 
#         for claim in converted_sentiment_sim_dict.keys():
#             sum += len(converted_sentiment_sim_dict[claim])
#             print claim, len(converted_sentiment_sim_dict[claim])
#         print sum
        self.save_pickle("converted_support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted_"+self.sentences_num, converted_sentiment_sim_dict)
        self.save_pickle("converted_support_baseline_sentence_sentiment_entropy_"+self.sentences_num, converted_sentence_sentiment_entropy)
        self.save_pickle("converted_sentence_vector_and_label_"+self.sentences_num, converted_sentence_vector_and_label)
        
    def convert_semantic_dict(self):
        """
        Instead of having (clm_text, sentence) -> semantic score dict form, 
        convert to clm_text -> [list of (sentence, score)]
        """
        print "start to convert semantic dict...."
        self.clm_sen_semantic_similarity_dict = self.read_pickle("support_baseline_all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300")
        self.claim_dict = self.read_pickle("claim_dict")
        converted_semantic_dict = {}
        if self.sentences_num == "allSentences": #change to sen_num 
            for ((clm_num,sen_num),score) in self.clm_sen_semantic_similarity_dict.items():
                if converted_semantic_dict.has_key(clm_num):
                    converted_semantic_dict[clm_num].append((sen_num,score))
                else:
                    converted_semantic_dict[clm_num] = [(sen_num,score)]
        elif self.sentences_num == "topSentences":
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
            #update 19.02 - change the dict to key- left out claim, value is key of train claim 
            # and then the value if a sentences list
            for left_out_claim in self.claim_list:
                converted_semantic_dict[left_out_claim] = {}
                for train_claim_num,sentences_list in relevance_baseline_topSentences[left_out_claim].items():
                    for sentence,ret_score in sentences_list:
                        if self.clm_sen_semantic_similarity_dict.has_key((train_claim_num,sentence)):
                            sen = sentence
                            score = self.clm_sen_semantic_similarity_dict[train_claim_num,sentence]
                        elif self.clm_sen_semantic_similarity_dict.has_key((train_claim_num,sentence.strip())):
                            sen = sentence.strip()
                            score = self.clm_sen_semantic_similarity_dict[train_claim_num,sentence.strip()]
                        if converted_semantic_dict[left_out_claim].has_key(train_claim_num):
                            converted_semantic_dict[left_out_claim][train_claim_num].append((sen,score))
                        else:
                            converted_semantic_dict[left_out_claim][train_claim_num] = [(sen,score)]
                                              
        self.save_pickle("converted_support_baseline_all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300_"+self.sentences_num, converted_semantic_dict)
        print "finished to convert semantic dict"
        
    def sentiment_feature_max_min_normalization(self):
        #sum up on each claim's 
        print "max min norm sentiment feature"
        left_out_max_min_features = {} 
        self.claim_dict = self.read_pickle("claim_dict")
        if self.sentences_num == "topSentences": 
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
        if "sentiment_sim" in self.features_setup:
            self.clm_sen_sentiment_similarity_dict = self.read_pickle("converted_support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted_"+self.sentences_num)
        if "label" in self.features_setup or "diff" in self.features_setup:
            self.claim_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sentiment_vector_and_label_dict")
            self.claim_sen_sentiment_vector_and_label_dict = self.read_pickle("converted_sentence_vector_and_label_"+self.sentences_num)
        if "entropy" in self.features_setup:
            self.claim_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sentiment_vector_entropy")
            self.sentence_sentiment_vector_entropy = self.read_pickle("converted_support_baseline_sentence_sentiment_entropy_"+self.sentences_num)
        if "lexicon" in self.features_setup:
            self.claim_sen_sentiment_pos_words_ratio = self.read_pickle("support_baseline_claims_sentences_positive_words_ratio_dict")
            self.claim_sen_sentiment_neg_words_ratio = self.read_pickle("support_baseline_claims_sentences_negative_words_ratio_dict")
            self.claim_sentiment_pos_words_ratio = self.read_pickle("support_baseline_claims_positive_words_ratio_dict")
            self.claim_sentiment_neg_words_ratio = self.read_pickle("support_baseline_claims_negative_words_ratio_dict")
        print "    calc max-min sentiment doc features..."
        for left_out_claim in self.claim_list:
            max_min_sentiment_score = max_min_sentiment_score_keeper()
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            
            if "label" in self.features_setup:
                #find the max and min sentiment label of the claims
                temp_claim_sentiment_vector_and_label_dict = dict(self.claim_sentiment_vector_and_label_dict)
                del temp_claim_sentiment_vector_and_label_dict[left_out_claim]
                sentiment_label = [sentiment_label for _,sentiment_label in temp_claim_sentiment_vector_and_label_dict.values()]
                #find the max/min across the claims of the train set
                max_min_sentiment_score.max_claim_sentiment_label = max(sentiment_label)
                max_min_sentiment_score.min_claim_sentiment_label = min(sentiment_label)
            #find the max/min diff in the sentiment label
            if "diff" in self.features_setup:
                temp_claim_sentiment_vector_and_label_dict = dict(self.claim_sentiment_vector_and_label_dict)
                del temp_claim_sentiment_vector_and_label_dict[left_out_claim]
                claim_sentiment_label = [sentiment_label for _,sentiment_label in temp_claim_sentiment_vector_and_label_dict.values()]
            #find the max and min entropy of the claim's sentiment vector
            if "entropy" in self.features_setup:
                temp_claim_entropy = dict(self.claim_sentiment_vector_entropy)
                del temp_claim_entropy[str(left_out_claim)]
                max_min_sentiment_score.max_claim_sentiment_entropy = max(temp_claim_entropy.values())
                max_min_sentiment_score.min_claim_sentiment_entropy = min(temp_claim_entropy.values())
            if "lexicon" in self.features_setup:
                temp_pos_lexicon_claim = dict(self.claim_sentiment_pos_words_ratio)
                temp_neg_lexicon_claim = dict(self.claim_sentiment_neg_words_ratio)
                del temp_pos_lexicon_claim[left_out_claim]
                del temp_neg_lexicon_claim[left_out_claim]
                max_min_sentiment_score.max_claim_pos_words_ratio = max(temp_pos_lexicon_claim.values())
                max_min_sentiment_score.min_claim_pos_words_ratio = min(temp_pos_lexicon_claim.values())
                max_min_sentiment_score.max_claim_neg_words_ratio = max(temp_neg_lexicon_claim.values())
                max_min_sentiment_score.min_claim_neg_words_ratio = min(temp_neg_lexicon_claim.values())
            #and now for the sentences normalization -  
            #update 20/2/15 -  for each left-out claim, with its sentences
            for curr_claim in curr_train_claims:
#               #27/02/15 update -  move to claim_num and sen_num  
                #curr_claim_text = self.claim_dict[str(curr_claim)]
                if "sentiment_sim" in self.features_setup:
                    if self.sentences_num == "allSentences":
                        sen_scores_list = [sen_score for _,sen_score in self.clm_sen_sentiment_similarity_dict[curr_claim]]
                    elif self.sentences_num == "topSentences": 
                        curr_top_sen_set = [sen for sen,sen_score in relevance_baseline_topSentences[left_out_claim][curr_claim]]
                        sen_scores_list = [sen_score for sen,sen_score in self.clm_sen_sentiment_similarity_dict[left_out_claim][curr_claim] if sen in curr_top_sen_set ]
                        #find the intersection between the topSentences
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
                    if self.sentences_num == "allSentences": 
#                         sen_label_list = [label for _ ,label in self.claim_sen_sentiment_vector_and_label_dict[self.claim_dict[str(curr_claim)]] ]
                        sen_label_list = [label for _ ,label in self.claim_sen_sentiment_vector_and_label_dict[curr_claim] ]
                    elif self.sentences_num == "topSentences":
                        curr_top_sen_set = [sen for sen,sen_score in relevance_baseline_topSentences[left_out_claim][curr_claim]]
#                         sen_label_list = [label for sen,label in self.claim_sen_sentiment_vector_and_label_dict[left_out_claim][self.claim_dict[str(curr_claim)]] ]
                        sen_label_list = [label for sen,label in self.claim_sen_sentiment_vector_and_label_dict[left_out_claim][curr_claim] ]
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
                if "diff" in self.features_setup:
                    if self.sentences_num == "allSentences": 
                        sen_label_list = [label for _ ,label in self.claim_sen_sentiment_vector_and_label_dict[curr_claim] ] 
                    elif self.sentences_num == "topSentences":
                        sen_label_list = [label for _ ,label in self.claim_sen_sentiment_vector_and_label_dict[left_out_claim][curr_claim] ] 
                    diff_sentiment_label = [sen_label_list[i] - claim_sentiment_label[curr_train_claims.index(curr_claim)] for i in range(0,len(sen_label_list))]
                    max_diff_sentiment_label = max(diff_sentiment_label)
                    min_diff_sentiment_label = min(diff_sentiment_label)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_sentiment_score.max_claim_sentence_sentiment_label_diff = max_diff_sentiment_label
                        max_min_sentiment_score.min_claim_sentence_sentiment_label_diff = min_diff_sentiment_label
                    else:
                        if max_diff_sentiment_label > max_min_sentiment_score.max_claim_sentence_sentiment_label_diff:
                            max_min_sentiment_score.max_claim_sentence_sentiment_label_diff = max_diff_sentiment_label
                        if min_diff_sentiment_label < max_min_sentiment_score.min_claim_sentence_sentiment_label_diff:
                            max_min_sentiment_score.min_claim_sentence_sentiment_label_diff = min_diff_sentiment_label
                if "entropy" in self.features_setup:               
                    if self.sentences_num == "allSentences":
                        sen_entropy_list = [sen_entropy for sen_num, sen_entropy in self.sentence_sentiment_vector_entropy[curr_claim]]
                    elif self.sentences_num == "topSentences":
                        sen_entropy_list = [sen_entropy for _, sen_entropy in self.sentence_sentiment_vector_entropy[left_out_claim][curr_claim]]
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
                if "lexicon" in self.features_setup:
                    if self.sentences_num == "allSentences":
                        sen_pos_words_ratio_list = [pos_words_ratio for sen_num, pos_words_ratio in self.claim_sen_sentiment_pos_words_ratio[curr_claim].items()]
                        sen_neg_words_ratio_list = [neg_words_ratio for sen_num, neg_words_ratio in self.claim_sen_sentiment_neg_words_ratio[curr_claim].items()]
                    max_sen_pos_words_ratio = max(sen_pos_words_ratio_list)
                    min_sen_pos_words_ratio = min(sen_pos_words_ratio_list)
                    max_sen_neg_words_ratio = max(sen_neg_words_ratio_list)
                    min_sen_neg_words_ratio = min(sen_neg_words_ratio_list)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_sentiment_score.max_sen_pos_words_ratio = max_sen_pos_words_ratio
                        max_min_sentiment_score.min_sen_pos_words_ratio = min_sen_pos_words_ratio
                        max_min_sentiment_score.max_sen_neg_words_ratio = max_sen_neg_words_ratio
                        max_min_sentiment_score.min_sen_neg_words_ratio = min_sen_neg_words_ratio
                    else:
                        if max_sen_pos_words_ratio > max_min_sentiment_score.max_sen_pos_words_ratio:
                            max_min_sentiment_score.max_sen_pos_words_ratio = max_sen_pos_words_ratio
                        if min_sen_pos_words_ratio < max_min_sentiment_score.min_sen_pos_words_ratio:
                            max_min_sentiment_score.min_sen_pos_words_ratio = min_sen_pos_words_ratio
                        if max_sen_neg_words_ratio > max_min_sentiment_score.max_sen_neg_words_ratio:
                            max_min_sentiment_score.max_sen_neg_words_ratio = max_sen_neg_words_ratio
                        if min_sen_neg_words_ratio < max_min_sentiment_score.min_sen_neg_words_ratio:
                            max_min_sentiment_score.min_sen_neg_words_ratio = max_min_sentiment_score
#`                  
            left_out_max_min_features[left_out_claim] = max_min_sentiment_score
                     
        self.save_pickle("left_out_max_min_support_sentiment_feature_"+self.sentences_num, left_out_max_min_features)              
            
    def semantic_features_max_min_normalization(self):
        """
        read all the claims cosine similarity dicts, and keep the max, min values for each left-out claim
        """
        print "max min norm semantic feature"
        left_out_max_min_features = {} 
        self.clm_sen_semantic_similarity_dict = self.read_pickle("converted_support_baseline_all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300_"+self.sentences_num)
        self.claim_dict = self.read_pickle("claim_dict")
        if self.sentences_num == "topSentences": 
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
        for left_out_claim in self.claim_list:
            max_min_semantic_score = max_min_semantic_score_keeper()
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            for curr_claim in curr_train_claims:
                curr_claim_text = self.claim_dict[str(curr_claim)]
                if self.sentences_num == "allSentences":
                    semantic_sim_list = [sem_sim_score for _,sem_sim_score in self.clm_sen_semantic_similarity_dict[curr_claim]]
                elif self.sentences_num == "topSentences":
                    curr_top_sen_set = [sen for sen,sen_score in relevance_baseline_topSentences[left_out_claim][curr_claim]]
                    semantic_sim_list = [sem_sim_score for sen,sem_sim_score in self.clm_sen_semantic_similarity_dict[left_out_claim][curr_claim] if sen in curr_top_sen_set ]
#                 semantic_sim_list = [sem_sim_score for _,sem_sim_score in self.clm_sen_semantic_similarity_dict[left_out_claim][curr_claim_text]]
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
        self.save_pickle("left_out_max_min_support_semantic_feature_"+self.sentences_num, left_out_max_min_features)
                                    
    def NLP_feature_max_min_normalization(self):
        print "max min norm NLP feature"
        left_out_max_min_NLP_features = {} 
        self.claim_dict = self.read_pickle("claim_dict")
        if self.sentences_num == "topSentences": 
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
        if "sen_len" in self.features_setup:
            self.sen_length = self.read_pickle("support_baseline_claim_num_sentences_num_sen_length")
        if "sw_ratio"  in self.features_setup:
            self.sw_ratio = self.read_pickle("support_baseline_claim_sen_POS_ratio")
        if "NER_ratio" in self.features_setup:
            self.NER_ratio = self.read_pickle("support_baseline_NER_sen_count")
        
        for left_out_claim in self.claim_list:
            max_min_NLP_scores = max_min_NLP_scores_keeper()
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            
            for curr_claim in curr_train_claims:
                if "sen_len" in self.features_setup:
                    #find the max and min sen len
                    if self.sentences_num == "allSentences":
                        sen_len_list = [sen_len for _,sen_len in self.sen_length[curr_claim].items()]
                    elif self.sentences_num == "topSentences":
                        curr_top_sen_set = [sen for sen,sen_score in relevance_baseline_topSentences[left_out_claim][curr_claim]]
                        sen_len_list = [sen_len for sen,sen_len in self.clm_sen_sentiment_similarity_dict[left_out_claim][curr_claim] if sen in curr_top_sen_set ]
#                     elif self.sentences_num == "topSentences": 
#                         curr_top_sen_set = [sen for sen,sen_score in relevance_baseline_topSentences[left_out_claim][curr_claim]]
#                         sen_scores_list = [sen_score for sen,sen_score in self.clm_sen_sentiment_similarity_dict[left_out_claim][curr_claim] if sen in curr_top_sen_set ]
                        #find the intersection between the topSentences
                    max_sen_len = max(sen_len_list)
                    min_sen_len = min(sen_len_list)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_NLP_scores.max_sen_len = max_sen_len
                        max_min_NLP_scores.min_sen_len = min_sen_len
                    else:
                        if max_sen_len > max_min_NLP_scores.max_sen_len:
                            max_min_NLP_scores.max_sen_len = max_sen_len
                        if min_sen_len < max_min_NLP_scores.min_sen_len:
                            max_min_NLP_scores.min_sen_len = min_sen_len
                            
                    #find the max/min across the claims of the train set
                if "sw_ratio" in self.features_setup:
                    if self.sentences_num == "allSentences":
                        sw_ratio = [sw_ratio for _,sw_ratio in self.sw_ratio[curr_claim].items()]
                    #find the max/min across the claims of the train set
                    max_sw_ratio = max(sw_ratio)
                    min_sw_ratio = min(sw_ratio)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_NLP_scores.max_stop_words_ratio = max_sw_ratio
                        max_min_NLP_scores.min_stop_words_ratio = min_sw_ratio
                    else:
                        if max_sw_ratio > max_min_NLP_scores.max_stop_words_ratio:
                            max_min_NLP_scores.max_stop_words_ratio = max_sw_ratio
                        if min_sw_ratio < max_min_NLP_scores.min_stop_words_ratio:
                            max_min_NLP_scores.min_stop_words_ratio = min_sw_ratio
                if "NER_ratio" in self.features_setup:
                    if self.sentences_num == "allSentences":
                        NER_ratio = [ner_ratio for _,ner_ratio in self.NER_ratio[curr_claim].items()]
                    max_NER_ratio = max(NER_ratio)
                    min_NER_ratio = min(NER_ratio)
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_NLP_scores.max_NER_ratio = max_NER_ratio
                        max_min_NLP_scores.min_NER_ratio = min_NER_ratio
                    else:
                        if max_NER_ratio > max_min_NLP_scores.max_NER_ratio:
                            max_min_NLP_scores.max_NER_ratio = max_NER_ratio
                        if min_NER_ratio < max_min_NLP_scores.min_NER_ratio:
                            max_min_NLP_scores.min_NER_ratio = min_NER_ratio
            
            left_out_max_min_NLP_features[left_out_claim] = max_min_NLP_scores
        utils.save_pickle("support_baseline_left_out_max_min_NLP_features", left_out_max_min_NLP_features)
    
    def objLM_feautres_max_min_normalization(self):
        print "max min norm objLM features"
        self.claim_dict = self.read_pickle("claim_dict")
        self.obj_LM_dist = self.read_pickle("support_baseline_claim_sentences_objLM_dist")
        if self.sentences_num == "topSentences": 
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
        
        for left_out_claim in self.claim_list:
            max_min_objLM_scorer = max_min_objLM_scores_keeper()
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            
            for curr_claim in curr_train_claims:
                if self.sentences_num == "allSentences":
                    sen_1_star_prob_list = [probs[0] for sen_num,probs in self.obj_LM_dist[curr_claim].items()]
                    sen_2_star_prob_list = [probs[1] for sen_num,probs in self.obj_LM_dist[curr_claim].items()]
                    sen_3_star_prob_list = [probs[2] for sen_num,probs in self.obj_LM_dist[curr_claim].items()]
                    sen_4_star_prob_list = [probs[3] for sen_num,probs in self.obj_LM_dist[curr_claim].items()]
                    sen_5_star_prob_list = [probs[4] for sen_num,probs in self.obj_LM_dist[curr_claim].items()]
                    
                    sen_1_star_prob_max = max(sen_1_star_prob_list)
                    sen_1_star_prob_min = min(sen_1_star_prob_list)
                    sen_2_star_prob_max = max(sen_2_star_prob_list)
                    sen_2_star_prob_min = min(sen_2_star_prob_list)
                    sen_3_star_prob_max = max(sen_3_star_prob_list)
                    sen_3_star_prob_min = min(sen_3_star_prob_list)
                    sen_4_star_prob_max = max(sen_4_star_prob_list)
                    sen_4_star_prob_min = min(sen_4_star_prob_list)
                    sen_5_star_prob_max = max(sen_5_star_prob_list)
                    sen_5_star_prob_min = min(sen_5_star_prob_list)
                    
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_objLM_scorer.max_1_star_prob = sen_1_star_prob_max
                        max_min_objLM_scorer.mim_1_star_prob= sen_1_star_prob_min
                        max_min_objLM_scorer.max_2_star_prob = sen_2_star_prob_max
                        max_min_objLM_scorer.mim_2_star_prob= sen_2_star_prob_min
                        max_min_objLM_scorer.max_3_star_prob = sen_3_star_prob_max
                        max_min_objLM_scorer.mim_3_star_prob= sen_3_star_prob_min
                        max_min_objLM_scorer.max_4_star_prob = sen_4_star_prob_max
                        max_min_objLM_scorer.mim_4_star_prob= sen_4_star_prob_min
                        max_min_objLM_scorer.max_5_star_prob = sen_5_star_prob_max
                        max_min_objLM_scorer.mim_5_star_prob= sen_5_star_prob_min
                        
                    else:
                        if sen_1_star_prob_max > max_min_objLM_scorer.max_1_star_prob:
                            max_min_objLM_scorer.max_1_star_prob = sen_1_star_prob_max
                        if sen_2_star_prob_max > max_min_objLM_scorer.max_2_star_prob:
                            max_min_objLM_scorer.max_2_star_prob = sen_2_star_prob_max
                        if sen_3_star_prob_max > max_min_objLM_scorer.max_3_star_prob:
                            max_min_objLM_scorer.max_3_star_prob = sen_3_star_prob_max
                        if sen_4_star_prob_max > max_min_objLM_scorer.max_4_star_prob:
                            max_min_objLM_scorer.max_4_star_prob = sen_4_star_prob_max
                        if sen_5_star_prob_max > max_min_objLM_scorer.max_5_star_prob:
                            max_min_objLM_scorer.max_5_star_prob = sen_5_star_prob_max
                       
                            
                        if sen_1_star_prob_min < max_min_objLM_scorer.min_1_star_prob:
                            max_min_objLM_scorer.min_1_star_prob = sen_1_star_prob_min
                        if sen_2_star_prob_min < max_min_objLM_scorer.min_2_star_prob:
                            max_min_objLM_scorer.min_2_star_prob = sen_2_star_prob_min
                        if sen_3_star_prob_min < max_min_objLM_scorer.min_3_star_prob:
                            max_min_objLM_scorer.min_3_star_prob = sen_3_star_prob_min
                        if sen_4_star_prob_min < max_min_objLM_scorer.min_4_star_prob:
                            max_min_objLM_scorer.min_4_star_prob = sen_4_star_prob_min
                        if sen_5_star_prob_min < max_min_objLM_scorer.min_5_star_prob:
                            max_min_objLM_scorer.min_5_star_prob = sen_5_star_prob_min
             
            self.left_out_max_min_objLM_features[left_out_claim] = max_min_objLM_scorer
        utils.save_pickle("support_baseline_left_out_max_min_objLM_features", self.left_out_max_min_objLM_features)
                    
    def MRF_features_max_min_normalization(self):
        print "max min norm MRF features"
        self.claim_dict = self.read_pickle("claim_dict")
        self.claim_doc_MRF_scores = self.read_pickle("support_baseline_claim_doc_MRF_scores") #key is a claim, another value of key of docid, his value is the ret score
        self.claim_sen_MRF_scores = self.read_pickle("support_baseline_claim_sen_MRF_scores")# key is a claim, another value of key of sen_num, his value is tuple: docid and ret score
        
        if self.sentences_num == "topSentences": 
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
            
        for left_out_claim in self.claim_list:
            max_min_MRF_scorer = max_min_MRF_scores_keeper()
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            
            for curr_claim in curr_train_claims:
                if self.sentences_num == "allSentences":
                    claim_doc_MRF_scores_list = [score for docid,score in self.claim_doc_MRF_scores[curr_claim].items()]
                    claim_sen_MRF_scores_list = [docid_score[1] for sen_num,docid_score in  self.claim_sen_MRF_scores[curr_claim].items()]
                    max_doc_MRF_score = max(claim_doc_MRF_scores_list)
                    min_doc_MRF_score = min(claim_doc_MRF_scores_list)
                    max_sen_MRF_score = max(claim_sen_MRF_scores_list)
                    min_sen_MRF_score = min(claim_sen_MRF_scores_list)
                    
                    if curr_train_claims.index(curr_claim) == 0:
                        max_min_MRF_scorer.max_doc_MRF_score = max_doc_MRF_score
                        max_min_MRF_scorer.min_doc_MRF_score = min_doc_MRF_score
                        max_min_MRF_scorer.max_sen_MRF_score = max_sen_MRF_score
                        max_min_MRF_scorer.min_sen_MRF_score = min_sen_MRF_score
                    else:
                        if max_doc_MRF_score > max_min_MRF_scorer.max_doc_MRF_score:
                            max_min_MRF_scorer.max_doc_MRF_score = max_doc_MRF_score
                        if max_sen_MRF_score > max_min_MRF_scorer.max_sen_MRF_score:
                            max_min_MRF_scorer.max_sen_MRF_score = max_sen_MRF_score
                        if min_doc_MRF_score < max_min_MRF_scorer.min_doc_MRF_score:
                            max_min_MRF_scorer.min_doc_MRF_score = min_doc_MRF_score
                        if min_sen_MRF_score < max_min_MRF_scorer.min_sen_MRF_score:
                            max_min_MRF_scorer.min_sen_MRF_score = min_sen_MRF_score
            
            self.left_out_max_min_MRF_features[left_out_claim] = max_min_MRF_scorer
        utils.save_pickle("support_baseline_left_out_max_min_MRF_features",self.left_out_max_min_MRF_features)   
                                                
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
            self.convert_sentiment_dict()
            self.sentiment_feature_max_min_normalization()
        if "NLP" in self.features_setup:
            self.NLP_feature_max_min_normalization()
        if "objLM" in self.features_setup:
            self.objLM_feautres_max_min_normalization()
        if "MRF" in self.features_setup:
            self.MRF_features_max_min_normalization()
        if "semantic" in self.features_setup:
# #             self.combine_semantic_dicts_with_original_full_claim_and_sentences()
            self.convert_semantic_dict()
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
        if self.sentences_num == "topSentences":
            relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")
            relevance_baseline_topSentences_docs_association = self.read_pickle("relevance_baseline_topSentences_docs_association") 
        for left_out_claim in self.claim_list:
            max_min_CE_scores = max_min_CE_scores_keeper()
            features_value_list = [[],[],[],[],[],[]]
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            for curr_claim in curr_train_claims:
                for (docid,CE_scores) in self.claim_entity_doc_CE_scores_dict[curr_claim].items():
                    #12.02.15 update - for the topSentences setup,
                    # go over only on the topSentences and their respective docid 
                    if self.sentences_num == "topSentences":
                        if docid in relevance_baseline_topSentences_docs_association[left_out_claim][curr_claim]:
                            features_value_list[0].append(CE_scores.CE_claim_title)
                            features_value_list[1].append(CE_scores.CE_claim_body)
                            features_value_list[2].append(CE_scores.CE_entity_title)
                            features_value_list[3].append(CE_scores.CE_entity_body)
                    elif self.sentences_num == "allSentences":
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
                    if self.sentences_num == "topSentences":
                        if docid in relevance_baseline_topSentences_docs_association[left_out_claim][curr_claim]:
                            for (sentences,CE_scores) in self.claim_entity_sen_CE_scores_dict[curr_claim][docid].items():
                                features_value_list[4].append(CE_scores.CE_claim_sentence)
                                features_value_list[5].append(CE_scores.CE_entity_sentence)
                    elif self.sentences_num == "allSentences":
                        for (sentences,CE_scores) in self.claim_entity_sen_CE_scores_dict[curr_claim][docid].items():
                            features_value_list[4].append(CE_scores.CE_claim_sentence)
                            features_value_list[5].append(CE_scores.CE_entity_sentence)
            curr_max_min_CE_scores_obj = left_out_max_min_features[left_out_claim]
            curr_max_min_CE_scores_obj.max_CE_claim_sentence = max(features_value_list[4])
            curr_max_min_CE_scores_obj.min_CE_claim_sentence = min(features_value_list[4])
            curr_max_min_CE_scores_obj.max_CE_entity_sentence = max(features_value_list[5])
            curr_max_min_CE_scores_obj.min_CE_entity_sentence = min(features_value_list[5]) 
                   
        #finished, save to pickle
        self.save_pickle("left_out_max_min_support_CE_features_"+self.sentences_num, left_out_max_min_features) 
            
    def convert_target_scores_dict_to_chars_as_sen(self,target):
        """
        as there is a difference between the sens in the support claim dict and the sens as they were retrieved from the baseline,
        convert the sen to only the chars.
        """
        exclude = set(string.punctuation)
        if target == "support":
            target_scores_dict = self.read_pickle("clm_sen_support_ranking_zero_to_two_clm_sen_key_supp_score_value")
        elif target == "relevance":
            target_scores_dict = self.read_pickle("clm_sen_relevance_dict")
            #target_scores_dict = self.read_pickle("claim_sen_relevance_dict_wiki")
        elif target == "contra":
            target_scores_dict = self.read_pickle("clm_as_key_sen_contradict_score_val_zero_to_two")
        target_scores_dict_no_punct = {}
        for ((claim,sen),supp_score) in target_scores_dict.items():
            sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
            sen_no_space = sen_no_punct.replace(" ","")
            target_scores_dict_no_punct[(claim,sen_no_space)] = supp_score
        return target_scores_dict_no_punct
    
    def add_claim_title_feature(self,curr_claim,docid,left_out_claim):
        if self.claim_entity_doc_CE_scores_dict[curr_claim].has_key(docid):
            return float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_claim_title-self.left_out_max_min_CE_features[left_out_claim].min_CE_claim_title)/\
                     (float(self.left_out_max_min_CE_features[left_out_claim].max_CE_claim_title-self.left_out_max_min_CE_features[left_out_claim].min_CE_claim_title))
        else:
            print docid +" not in add_claim_title_feature"
            
    def add_claim_body_feature(self,curr_claim,docid,left_out_claim):
        if self.claim_entity_doc_CE_scores_dict[curr_claim].has_key(docid):
            return float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_claim_body-self.left_out_max_min_CE_features[left_out_claim].min_CE_claim_body)/\
                     (float(self.left_out_max_min_CE_features[left_out_claim].max_CE_claim_body-self.left_out_max_min_CE_features[left_out_claim].min_CE_claim_body))
        else:
            print docid +" not in add_claim_body_feature"
            
    def add_entity_title_feature(self,curr_claim,docid,left_out_claim):
        if self.claim_entity_doc_CE_scores_dict[curr_claim].has_key(docid):
            return float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_entity_title-self.left_out_max_min_CE_features[left_out_claim].min_CE_entity_title)/\
                     (float(self.left_out_max_min_CE_features[left_out_claim].max_CE_entity_title-self.left_out_max_min_CE_features[left_out_claim].min_CE_entity_title))
        else:
            print docid +" not in add_entity_title_feature"
            
    def add_entity_body_feature(self,curr_claim,docid,left_out_claim):
        if self.claim_entity_doc_CE_scores_dict[curr_claim].has_key(docid):
            return float(self.claim_entity_doc_CE_scores_dict[curr_claim][docid].CE_entity_body-self.left_out_max_min_CE_features[left_out_claim].min_CE_entity_body)/\
                    (float(self.left_out_max_min_CE_features[left_out_claim].max_CE_entity_body-self.left_out_max_min_CE_features[left_out_claim].min_CE_entity_body))
        else:
            print docid +" not in add_entity_body_feature"
            
    def add_claim_sentence_feature(self,curr_claim,docid,left_out_claim,sen_num):
        if self.claim_entity_sen_CE_scores_dict[curr_claim][docid].has_key(sen_num):
            return float(self.claim_entity_sen_CE_scores_dict[curr_claim][docid][sen_num].CE_claim_sentence-self.left_out_max_min_CE_features[left_out_claim].min_CE_claim_sentence)/\
                            (float(self.left_out_max_min_CE_features[left_out_claim].max_CE_claim_sentence-self.left_out_max_min_CE_features[left_out_claim].min_CE_claim_sentence))
        else:
            print sen_num +" not in add_claim_sentence_feature, for clm " +str(curr_claim)
            
    def add_entity_sentence_feature(self,curr_claim,docid,left_out_claim,sen_num):
        if self.claim_entity_sen_CE_scores_dict[curr_claim][docid].has_key(sen_num): 
            return float(self.claim_entity_sen_CE_scores_dict[curr_claim][docid][sen_num].CE_entity_sentence-self.left_out_max_min_CE_features[left_out_claim].min_CE_entity_sentence)/\
                        (float(self.left_out_max_min_CE_features[left_out_claim].max_CE_entity_sentence-self.left_out_max_min_CE_features[left_out_claim].min_CE_entity_sentence))
        else:
            print sen_num +" not in add_entity_sentence_feature, for clm " +str(curr_claim)
    #TODO:
    #add the other features "add" functions
        
    def add_sentiment_sim_feature(self,curr_claim,sen_num,left_out_claim):
        return float((self.clm_sen_sentiment_similarity_dict[curr_claim,sen_num]-self.left_out_max_min_sentiment_features[left_out_claim].min_sentiment_score)/\
                                                           (self.left_out_max_min_sentiment_features[left_out_claim].max_sentiment_score-\
                                                         self.left_out_max_min_sentiment_features[left_out_claim].min_sentiment_score))

    def add_claim_sentiment_label_feature(self,curr_claim,left_out_claim):
        return float((self.claim_sentiment_vector_and_label_dict[str(curr_claim)][1]-self.left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_label)/\
                                                 (self.left_out_max_min_sentiment_features[left_out_claim].max_claim_sentiment_label-\
                                                 self.left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_label))

    def add_sen_sentiment_label_feature(self,curr_claim,sen_num,left_out_claim):
        return float((self.claim_sen_sentiment_vector_and_label_dict[curr_claim,sen_num][1]-self.left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_label)/\
                                            (self.left_out_max_min_sentiment_features[left_out_claim].max_sentence_sentiment_label-\
                                            self.left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_label))
    
    def add_claim_sentences_sentiment_label_diff_feature(self,curr_claim,sen,left_out_claim):
        return float((self.claim_sentiment_vector_and_label_dict[str(curr_claim)][1] - self.claim_sen_sentiment_vector_and_label_dict[str(curr_claim),str(self.claim_sentences_dict[curr_claim].index(sen))][1]\
                      - self.left_out_max_min_sentiment_features[left_out_claim].min_claim_sentence_sentiment_label_diff)/\
                                            (self.left_out_max_min_sentiment_features[left_out_claim].max_claim_sentence_sentiment_label_diff-\
                                            self.left_out_max_min_sentiment_features[left_out_claim].min_claim_sentence_sentiment_label_diff))
    
    def add_claim_sentiment_entropy_feature(self,curr_claim,left_out_claim):
        return float((self.claim_sentiment_vector_entropy[str(curr_claim)]-self.left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_entropy)/\
                                                                     (self.left_out_max_min_sentiment_features[left_out_claim].max_claim_sentiment_entropy-\
                                                                    self.left_out_max_min_sentiment_features[left_out_claim].min_claim_sentiment_entropy))
    
    def add_sen_sentiment_entropy_feature(self,curr_claim, sen_num, left_out_claim):
        return float((self.sentence_sentiment_vector_entropy[curr_claim,sen_num]-self.left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_entropy)/\
                                                                     (self.left_out_max_min_sentiment_features[left_out_claim].max_sentence_sentiment_entropy-\
                                                                     self.left_out_max_min_sentiment_features[left_out_claim].min_sentence_sentiment_entropy))
    
    def add_claim_sentiment_lexicon_pos_words_ratio_feature(self,curr_claim, sen_num, left_out_claim):
        return float((self.claim_sentiment_pos_words_ratio[curr_claim]-self.left_out_max_min_sentiment_features[left_out_claim].min_claim_pos_words_ratio)/\
                     (self.left_out_max_min_sentiment_features[left_out_claim].max_claim_pos_words_ratio-\
                      self.left_out_max_min_sentiment_features[left_out_claim].min_claim_pos_words_ratio)) 
    
    def add_claim_sentiment_lexicon_neg_words_ratio_feature(self,curr_claim, sen_num, left_out_claim):
        return float((self.claim_sentiment_neg_words_ratio[curr_claim]-self.left_out_max_min_sentiment_features[left_out_claim].min_claim_neg_words_ratio)/\
                     (self.left_out_max_min_sentiment_features[left_out_claim].max_claim_neg_words_ratio-\
                      self.left_out_max_min_sentiment_features[left_out_claim].min_claim_neg_words_ratio)) 
    
    def add_sen_sentiment_lexicon_pos_words_ratio_feature(self,curr_claim, sen_num, left_out_claim):
        return float((self.claim_sen_sentiment_pos_words_ratio[curr_claim][sen_num]-self.left_out_max_min_sentiment_features[left_out_claim].min_sen_pos_words_ratio)/\
                     (self.left_out_max_min_sentiment_features[left_out_claim].max_sen_pos_words_ratio-\
                      self.left_out_max_min_sentiment_features[left_out_claim].min_sen_pos_words_ratio)) 
    
    def add_sen_sentiment_lexicon_neg_words_ratio_feature(self,curr_claim, sen_num, left_out_claim):
        return float((self.claim_sen_sentiment_neg_words_ratio[curr_claim][sen_num]-self.left_out_max_min_sentiment_features[left_out_claim].min_sen_neg_words_ratio)/\
                     (self.left_out_max_min_sentiment_features[left_out_claim].max_sen_neg_words_ratio-\
                      self.left_out_max_min_sentiment_features[left_out_claim].min_sen_neg_words_ratio))
    
    def add_semantic_feature(self,curr_claim, sen_num, left_out_claim):
        return ((float(self.clm_sen_semantic_similarity_dict[curr_claim,sen_num]-self.left_out_max_min_semantic_features[left_out_claim].min_semantic_score)/\
                               (self.left_out_max_min_semantic_features[left_out_claim].max_semantic_score-\
                               self.left_out_max_min_semantic_features[left_out_claim].min_semantic_score)))

    def add_sen_len_feature(self,curr_claim, sen_num, left_out_claim):
        return (float(float(self.sen_length[curr_claim][sen_num]- self.left_out_max_min_NLP_features[left_out_claim].min_sen_len)/\
                               (float(self.left_out_max_min_NLP_features[left_out_claim].max_sen_len-\
                               self.left_out_max_min_NLP_features[left_out_claim].min_sen_len))))

    def add_sen_sw_ratio_feature(self,curr_claim, sen_num, left_out_claim):
        return (float((self.sw_ratio[curr_claim][sen_num]-self.left_out_max_min_NLP_features[left_out_claim].min_stop_words_ratio)/\
                           (self.left_out_max_min_NLP_features[left_out_claim].max_stop_words_ratio-\
                           self.left_out_max_min_NLP_features[left_out_claim].min_stop_words_ratio)))
    
    def add_sen_NER_ratio_feature(self,curr_claim, sen_num, left_out_claim):
        try:
            return (float((self.NER_ratio[curr_claim][sen_num]-self.left_out_max_min_NLP_features[left_out_claim].min_NER_ratio)/\
                               (self.left_out_max_min_NLP_features[left_out_claim].max_NER_ratio-\
                               self.left_out_max_min_NLP_features[left_out_claim].min_NER_ratio)))
        except Exception as err: 
            sys.stderr.write('problem in add_sen_NER_ratio_feature:' ,curr_claim, sen_num, left_out_claim)     
            print err.args      
            print err
            
    def add_typedDep_bin_feature(self,curr_claim, sen_num, left_out_claim):
        res_list = [] 
        try:
            if self.typedDep_bin[curr_claim].has_key(sen_num):
                res_list = self.typedDep_bin[curr_claim][sen_num]
            else:
                print "for clm ",curr_claim, "sen num", sen_num ,"not in typedDep_bin"
                res_list = [0]*self.typedDep_num
            return res_list
        
        except Exception as err: 
            sys.stderr.write('problem in add_typedDep_bin_feature:' ,curr_claim, sen_num, left_out_claim)     
            print err.args      
            print err
                    
    def add_entailemt_binary_feature(self,curr_claim, sen_num, left_out_claim):
        if self.entailment_res[curr_claim][sen_num][0] == "Entailment":
            return 1
        elif self.entailment_res[curr_claim][sen_num][0] == "NonEntailment":
            return 0
        #another option is to add the confidence... 
    
    def add_objLM_dist_feature(self,curr_claim, sen_num, left_out_claim):
        norm_obj_dist = []
        curr_obj_dist = self.obj_LM_dist[curr_claim][sen_num]
        norm_obj_dist.append(float((curr_obj_dist[0]-self.left_out_max_min_objLM_features[left_out_claim].min_1_star_prob)\
                                   /(self.left_out_max_min_objLM_features[left_out_claim].max_1_star_prob- \
                                    self.left_out_max_min_objLM_features[left_out_claim].min_1_star_prob)))
        norm_obj_dist.append(float((curr_obj_dist[1]-self.left_out_max_min_objLM_features[left_out_claim].min_1_star_prob)\
                                   /(self.left_out_max_min_objLM_features[left_out_claim].max_2_star_prob- \
                                   self.left_out_max_min_objLM_features[left_out_claim].min_1_star_prob)))
        norm_obj_dist.append(float((curr_obj_dist[2]-self.left_out_max_min_objLM_features[left_out_claim].min_3_star_prob)\
                                   /(self.left_out_max_min_objLM_features[left_out_claim].max_3_star_prob- \
                                   self.left_out_max_min_objLM_features[left_out_claim].min_3_star_prob)))
        norm_obj_dist.append(float((curr_obj_dist[3]-self.left_out_max_min_objLM_features[left_out_claim].min_4_star_prob)\
                                   /(self.left_out_max_min_objLM_features[left_out_claim].max_4_star_prob- \
                                    self.left_out_max_min_objLM_features[left_out_claim].min_4_star_prob)))
        norm_obj_dist.append(float((curr_obj_dist[4]-self.left_out_max_min_objLM_features[left_out_claim].min_5_star_prob)\
                                   /(self.left_out_max_min_objLM_features[left_out_claim].max_5_star_prob- \
                                    self.left_out_max_min_objLM_features[left_out_claim].min_5_star_prob)))
        return norm_obj_dist
    
    def add_MRF_scores_feature(self,curr_claim, sen_num, left_out_claim):
        try:
            MRF_doc_sen_scores = []
            curr_sen_docid = ""
            curr_sen_score = 0.0
            curr_doc_score = 0.0
            if self.claim_sen_MRF_scores[curr_claim].has_key(sen_num):
                curr_sen_docid = self.claim_sen_MRF_scores[curr_claim][sen_num][0] #tuple of docid and sen retrieval score
                curr_sen_score = self.claim_sen_MRF_scores[curr_claim][sen_num][1]
            if self.claim_doc_MRF_scores[curr_claim].has_key(curr_sen_docid):
                curr_doc_score = self.claim_doc_MRF_scores[curr_claim][curr_sen_docid]
            if curr_doc_score != 0.0:
                MRF_doc_sen_scores.append(float((curr_doc_score-self.left_out_max_min_MRF_features[left_out_claim].min_doc_MRF_score)\
                                            /(self.left_out_max_min_MRF_features[left_out_claim].max_doc_MRF_score- \
                                            self.left_out_max_min_MRF_features[left_out_claim].min_doc_MRF_score)))
            else:
                MRF_doc_sen_scores.append(curr_doc_score)
            if curr_sen_score != 0.0:
                MRF_doc_sen_scores.append(float((curr_sen_score-self.left_out_max_min_MRF_features[left_out_claim].min_sen_MRF_score)\
                                            /(self.left_out_max_min_MRF_features[left_out_claim].max_sen_MRF_score- \
                                            self.left_out_max_min_MRF_features[left_out_claim].min_sen_MRF_score)))
            else:
                MRF_doc_sen_scores.append(curr_sen_score)
            return MRF_doc_sen_scores
        except Exception as err: 
                    sys.stderr.write('problem in add_MRF_scores:' ,curr_claim, sen_num, left_out_claim)     
                    print err.args      
                    print err  
          
    def read_feature_normalization_and_data_dicts_for_writing_train_test_files_SVM(self):
        self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized")
        self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized")
        self.left_out_max_min_CE_features = self.read_pickle("left_out_max_min_support_CE_features_"+self.sentences_num)
        if "sentiment" in self.features_setup:
            self.left_out_max_min_sentiment_features = self.read_pickle("left_out_max_min_support_sentiment_feature_"+str(self.sentences_num))
            if "sentiment_sim" in self.features_setup:
                self.clm_sen_sentiment_similarity_dict = self.read_pickle("support_baseline_claim_sen_sentiment_JSD_similarity_socher_sorted")
            if "entropy" in self.features_setup:
                self.claim_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sentiment_vector_entropy")
                self.sentence_sentiment_vector_entropy = self.read_pickle("support_baseline_claim_sen_sentiment_vector_entropy")
            if "label" in self.features_setup or "diff" in self.features_setup:
                self.claim_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sentiment_vector_and_label_dict")
                self.claim_sen_sentiment_vector_and_label_dict = self.read_pickle("support_baseline_claim_sen_sentiment_vector_and_label_dict")
            if "lexicon" in self.features_setup:
                self.claim_sentiment_pos_words_ratio = self.read_pickle("support_baseline_claims_positive_words_ratio_dict")
                self.claim_sentiment_neg_words_ratio = self.read_pickle("support_baseline_claims_negative_words_ratio_dict")
                self.claim_sen_sentiment_pos_words_ratio = self.read_pickle("support_baseline_claims_sentences_positive_words_ratio_dict")
                self.claim_sen_sentiment_neg_words_ratio = self.read_pickle("support_baseline_claims_sentences_negative_words_ratio_dict")
        if "semantic" in self.features_setup :
            self.left_out_max_min_semantic_features = self.read_pickle("left_out_max_min_support_semantic_feature_"+str(self.sentences_num))
            self.clm_sen_semantic_similarity_dict = self.read_pickle("support_baseline_all_clm_sen_cosine_sim_res_word2vec_max_words_similarity_300")
        self.claim_dict = self.read_pickle("claim_dict")
        if "NLP" in self.features_setup:
            self.left_out_max_min_NLP_features = self.read_pickle("support_baseline_left_out_max_min_NLP_features")
            if "sen_len" in self.features_setup:
                self.sen_length = self.read_pickle("support_baseline_claim_num_sentences_num_sen_length")
            if "sw_ratio" in self.features_setup:
                self.sw_ratio = self.read_pickle("support_baseline_claim_sen_POS_ratio")
            if "NER_ratio" in self.features_setup:
                self.NER_ratio = self.read_pickle("support_baseline_NER_sen_count")
            if "typedDep_bin" in self.features_setup:
                self.typedDep_bin = self.read_pickle("support_baseline_sentences_typedDep_bool")
        if "entailment" in self.features_setup:
            self.entailment_res = self.read_pickle("support_baseline_claim_sentences_entailemtn_res")
        if "objLM" in self.features_setup:
            self.obj_LM_dist = self.read_pickle("support_baseline_claim_sentences_objLM_dist")
            self.left_out_max_min_objLM_features = self.read_pickle("support_baseline_left_out_max_min_objLM_features")
        if "MRF" in self.features_setup:
            self.claim_doc_MRF_scores = self.read_pickle("support_baseline_claim_doc_MRF_scores")
            self.claim_sen_MRF_scores = self.read_pickle("support_baseline_claim_sen_MRF_scores")
            self.left_out_max_min_MRF_features = self.read_pickle("support_baseline_left_out_max_min_MRF_features")
                        
    def get_features_for_SVM_train_test_files(self,curr_claim,docid,left_out_claim,sen_num):
        try:
            curr_features_vec = []
            if "CE" in self.features_setup:
                try:
                    if "claim_title" in self.features_setup or "all" in self.features_setup :
                        curr_features_vec.append(self.add_claim_title_feature(curr_claim,docid,left_out_claim))
                    if "claim_body" in self.features_setup or "all" in self.features_setup:
                        curr_features_vec.append(self.add_claim_body_feature(curr_claim, docid, left_out_claim))
                    if "entity_title" in self.features_setup or "all" in self.features_setup: 
                        curr_features_vec.append(self.add_entity_title_feature(curr_claim, docid, left_out_claim))
                    if "entity_body" in self.features_setup or "all" in self.features_setup:
                        curr_features_vec.append(self.add_entity_body_feature(curr_claim, docid, left_out_claim))
                    if "claim_sentence" in self.features_setup or "all" in self.features_setup:
                        curr_features_vec.append(self.add_claim_sentence_feature(curr_claim, docid, left_out_claim,sen_num))#sen.strip()))
                    if "entity_sentence" in self.features_setup or "all" in self.features_setup:
                        curr_features_vec.append(self.add_entity_sentence_feature(curr_claim, docid, left_out_claim,sen_num))#sen.strip()))
                except Exception as err: 
                    sys.stderr.write('problem in CE features:')     
                    print err.args      
                    print err
            if "sentiment" in self.features_setup:
                if "sentiment_sim" in self.features_setup:
                    if self.clm_sen_sentiment_similarity_dict.has_key((curr_claim,sen_num)):
                        curr_features_vec.append(self.add_sentiment_sim_feature(curr_claim,sen_num,left_out_claim))
#                     elif self.clm_sen_sentiment_similarity_dict.has_key((self.claim_dict[str(curr_claim)],sen+" ")):
#                         curr_features_vec.append(self.add_sentiment_sim_feature(curr_claim,sen+" ",left_out_claim))
                    else:
                        curr_features_vec.append(0)
                if "label" in self.features_setup:
                    if self.claim_sentiment_vector_and_label_dict.has_key(curr_claim):
                        curr_features_vec.append(self.add_claim_sentiment_label_feature(curr_claim,left_out_claim))
                    else:
                        curr_features_vec.append(0)
                    if self.claim_sen_sentiment_vector_and_label_dict.has_key(curr_claim,sen_num):
                        curr_features_vec.append(self.add_sen_sentiment_label_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0) 
#                     correct_sen = ""
#                     if sen in self.claim_sentences_dict[curr_claim]:
#                         correct_sen = sen
#                     elif sen.strip() in self.claim_sentences_dict[curr_claim]:
#                         correct_sen = sen.strip()
#                     if self.claim_sen_sentiment_vector_and_label_dict.has_key((str(curr_claim),str(self.claim_sentences_dict[curr_claim].index(correct_sen)))):
#                         curr_features_vec.append(self.add_sen_sentiment_label_feature(curr_claim, correct_sen, left_out_claim))
#                     else:
#                         curr_features_vec.append(0)
                if "diff" in self.features_setup:
                    if self.claim_sentiment_vector_and_label_dict.has_key(curr_claim) and self.claim_sen_sentiment_vector_and_label_dict.has_key(curr_claim,sen_num):
                        curr_features_vec.append(self.add_claim_sentences_sentiment_label_diff_feature(curr_claim, sen_num, left_out_claim))
#                     correct_sen = ""
#                     if sen in self.claim_sentences_dict[curr_claim]:
#                         correct_sen = sen
#                     elif sen.strip() in self.claim_sentences_dict[curr_claim]:
#                         correct_sen = sen.strip()
#                     if self.claim_sentiment_vector_and_label_dict.has_key(str(curr_claim)) and self.claim_sen_sentiment_vector_and_label_dict.has_key((str(curr_claim),str(self.claim_sentences_dict[curr_claim].index(correct_sen)))):
#                         curr_features_vec.append(self.add_claim_sentences_sentiment_label_diff_feature(curr_claim, correct_sen, left_out_claim))
                if "entropy" in self.features_setup:
                    if self.claim_sentiment_vector_entropy.has_key(curr_claim):
                        curr_features_vec.append(self.add_claim_sentiment_entropy_feature(curr_claim, left_out_claim))
                    else:
                        curr_features_vec.append(0)
                    if self.sentence_sentiment_vector_entropy.has_key(curr_claim,sen_num):
                        curr_features_vec.append(self.add_sen_sentiment_entropy_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
#                     correct_sen = ""
#                     if sen in self.claim_sentences_dict[curr_claim]:
#                         correct_sen = sen
#                     elif sen.strip() in self.claim_sentences_dict[curr_claim]:
#                         correct_sen = sen.strip()
#                     if self.sentence_sentiment_vector_entropy.has_key((str(curr_claim),str(self.claim_sentences_dict[curr_claim].index(correct_sen)))):
#                         curr_features_vec.append(self.add_sen_sentiment_entropy_feature(curr_claim, correct_sen, left_out_claim))
#                     else:
#                         curr_features_vec.append(0)
                if "lexicon" in self.features_setup:
                    if self.claim_sentiment_pos_words_ratio.has_key(curr_claim):
                        curr_features_vec.append(self.add_claim_sentiment_lexicon_pos_words_ratio_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
                    if self.claim_sentiment_neg_words_ratio.has_key(curr_claim):
                        curr_features_vec.append(self.add_claim_sentiment_lexicon_neg_words_ratio_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
                    if self.claim_sen_sentiment_pos_words_ratio.has_key(curr_claim):
                        curr_features_vec.append(self.add_sen_sentiment_lexicon_pos_words_ratio_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
                    if self.claim_sen_sentiment_neg_words_ratio.has_key(curr_claim):
                        curr_features_vec.append(self.add_sen_sentiment_lexicon_neg_words_ratio_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
            if "semantic" in self.features_setup :
                if self.clm_sen_semantic_similarity_dict.has_key((curr_claim,sen_num+1)):
                    curr_features_vec.append(self.add_semantic_feature(curr_claim, sen_num+1, left_out_claim)) #add 1 to sen_num cus in the claim_sentence_VSM_sim the counter starts fro 1..
#                 elif self.clm_sen_semantic_similarity_dict.has_key(curr_claim,sen_num):
#                     curr_features_vec.append(self.add_semantic_feature(curr_claim, sen.strip(), left_out_claim))
#                 elif self.clm_sen_semantic_similarity_dict.has_key((self.claim_dict[str(curr_claim)],sen+" ")):
#                     curr_features_vec.append(self.add_semantic_feature(curr_claim, sen+" ", left_out_claim))
                else:
                    curr_features_vec.append(0)
            if "NLP" in self.features_setup:
                if "sen_len" in self.features_setup:
                    if self.sen_length.has_key(curr_claim):
                        curr_features_vec.append(self.add_sen_len_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
                if "sw_ratio" in self.features_setup:
                    if self.sw_ratio.has_key(curr_claim):
                        curr_features_vec.append(self.add_sen_sw_ratio_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
                if "NER_ratio" in self.features_setup:
                    if self.NER_ratio.has_key(curr_claim):
                        curr_features_vec.append(self.add_sen_NER_ratio_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.append(0)
                if "typedDep_bin" in self.features_setup:
                    if self.typedDep_bin.has_key(curr_claim):
                        curr_features_vec.extend(self.add_typedDep_bin_feature(curr_claim, sen_num, left_out_claim))
                    else:
                        curr_features_vec.extend(0*[self.typedDep_num])
                     
            if "entailment_bin" in self.features_setup:
                if self.entailment_res.has_key(curr_claim):
                    curr_features_vec.append(self.add_entailemt_binary_feature(curr_claim, sen_num, left_out_claim))
                else:
                    curr_features_vec.append(0)
            if "objLM" in self.features_setup:
                if self.obj_LM_dist.has_key(curr_claim):
                    curr_features_vec.extend(self.add_objLM_dist_feature(curr_claim, sen_num, left_out_claim)) #this is a list, thus using extend !
            if "MRF" in self.features_setup:
                if self.claim_doc_MRF_scores.has_key(curr_claim) and self.claim_sen_MRF_scores.has_key(curr_claim):
                    curr_features_vec.extend(self.add_MRF_scores_feature(curr_claim, sen_num, left_out_claim))
        
        except Exception as err: 
                sys.stderr.write('problem in get_features_for_SVM_train_test_files:')     
                print err.args      
                print err
        return curr_features_vec
                                    
    def write_train_test_files_SVM_topSentences(self):
        #write the files with the max-min normalized scores
        self.claim_sentences_dict = self.read_pickle("support_baseline_claim_sentences")
        relevance_baseline_topSentences = self.read_pickle("relevance_baseline_topSentences")  # #key is held out claim num, value is a another key value, key is a train claim num, value is a list of sens
        relevance_baseline_topSentences_sens_docs_association = self.read_pickle("relevance_baseline_topSentences_sens_docs_association")
        exclude = set(string.punctuation)
        target_scores_dict = self.convert_target_scores_dict_to_chars_as_sen(self.target)
        self.read_feature_normalization_and_data_dicts_for_writing_train_test_files_SVM()
    
        for left_out_claim in self.claim_list: 
            print "writing SVM files for claim " +str(left_out_claim) +" for features: " +self.features_setup 
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            if self.target == "support":
                curr_test_LOOCV = open (self.test_path+r"test_clm_num_"+str(left_out_claim)+"_CV", 'wb')
                curr_train_LOOCV = open (self.train_path+r"train_left_out_"+str(left_out_claim)+"_CV", 'wb')
            elif self.target == "relevance":
                curr_test_LOOCV = open(self.test_path_relevance+r"test_clm_num_"+str(left_out_claim)+"_CV", 'wb')
                curr_train_LOOCV = open(self.train_path_relevance+r"train_left_out_"+str(left_out_claim)+"_CV", 'wb')
            elif self.target == "contra":
                curr_test_LOOCV = open (self.test_path_contra + r"test_clm_num_"+str(left_out_claim)+"_CV", 'wb')
                curr_train_LOOCV = open(self.train_path_contra + r"train_left_out_"+str(left_out_claim)+"_CV", 'wb')
            for curr_claim in curr_train_claims:                   
                print "curr train claim:" +str(curr_claim)
                sentences_set = set()
                dups_cnt = 0
                top_sentences_list = relevance_baseline_topSentences[left_out_claim][curr_claim]
                for (sen,ret_score) in top_sentences_list:
                    if not sen in sentences_set:
                        try:
                            curr_features_vec = []
                            line = ""
                            sentences_set.add(sen)
                            docid = relevance_baseline_topSentences_sens_docs_association[left_out_claim][curr_claim][sen]
                            curr_features_vec = self.get_features_for_SVM_train_test_files(curr_claim, docid, left_out_claim, sen)
                            sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                            sen_no_space = sen_no_punct.replace(" ","")
                            if target_scores_dict.has_key((self.claim_dict[str(curr_claim)], sen_no_space)):
                                curr_support_score = target_scores_dict[(self.claim_dict[str(curr_claim)],sen_no_space)]
                            else:
                                curr_support_score = 0
                            line += str(curr_support_score) + " qid:"+ str(curr_claim)+" "
                            for feature_idx in range(0,len(curr_features_vec)):
                                line += str(feature_idx+1)+":"+str(curr_features_vec[feature_idx])+" "
#                             line += "\n"
                            line += "#"+self.claim_dict[str(curr_claim)] +"|"+ sen +"\n" 
                            curr_train_LOOCV.write(line)
                        except Exception as err: 
                            sys.stderr.write("problem in write_train_test_files_SVM in left-out-claim"+str(left_out_claim)+" in train claim "+str(curr_claim))     
                            print err.args      
                            print err
                    else:
                        dups_cnt += 1
#                 print curr_claim, "dups", dups_cnt ,len(sentences_set)," sentences:"
            curr_train_LOOCV.close()
            curr_claim = ""
            for docid in self.claim_entity_doc_CE_scores_dict[left_out_claim]:
                for sen in self.claim_entity_sen_CE_scores_dict[left_out_claim][docid].keys():
                    curr_features_vec = []
                    line = ""
                    curr_features_vec = self.get_features_for_SVM_train_test_files(left_out_claim, docid, left_out_claim, sen)
                    sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                    sen_no_space = sen_no_punct.replace(" ","")
                    if target_scores_dict.has_key((self.claim_dict[str(left_out_claim)],sen_no_space)):
                        curr_support_score = target_scores_dict[self.claim_dict[str(left_out_claim)],sen_no_space]
#                         print" found"
                    else:
                        curr_support_score = 0 
                    line += str(curr_support_score) + " qid:" + str(left_out_claim) +" "
                    for feature_idx in range(0,len(curr_features_vec)):
                            line += str(feature_idx+1)+":"+str(curr_features_vec[feature_idx]) +" "
                    line += "#" + self.claim_dict[str(left_out_claim)] +"|"+ sen +"\n"
#                     line += "\n"
                    curr_test_LOOCV.write(line)
            curr_test_LOOCV.close()    
            print "finished writing files for claim", left_out_claim  
        print "finished writing files for features ", self.features_setup
        
    def filter_sentences_by_sw_ratio(self,curr_claim, sen_num, left_out_claim):
        """
        if the sw ratio is smaller than a threshold, don't write the sentence for train/test
        """
        try:
            sw_ratio_threshold = 0.1
            if len(self.sw_ratio) == 0:
                self.sw_ratio = self.read_pickle("support_baseline_claim_sen_POS_ratio")
            if len(self.left_out_max_min_NLP_features) == 0:
                self.left_out_max_min_NLP_features = self.read_pickle("support_baseline_left_out_max_min_NLP_features")
            if self.add_sen_sw_ratio_feature(curr_claim, sen_num, left_out_claim) < sw_ratio_threshold:
                return 0
            else:
                return 1
        except Exception as err: 
                sys.stderr.write('problem in filter_sentences_by_sw_ratio:')     
                print err.args      
                print err
               
    def write_train_test_files_SVM_allSentences(self):
        #write the files with the max-min normalized scores
        self.claim_sentences_dict = self.read_pickle("support_baseline_claim_sentences")
        self.claim_num_sentences_num_sentences_text_dict = self.read_pickle("support_baseline_claim_num_sentences_num_sentences_text_dict_allSentences")
        self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized")
        self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized")
#         if "CE" in self.features_setup:
        target_scores_dict = self.convert_target_scores_dict_to_chars_as_sen(self.target)
        self.read_feature_normalization_and_data_dicts_for_writing_train_test_files_SVM()
        exclude = set(string.punctuation)
        
        for left_out_claim in self.claim_list: 
            print "writing SVM files for claim " +str(left_out_claim) +" for features: " +self.features_setup 
            curr_train_claims = self.claim_list[:]
            curr_train_claims.remove(left_out_claim)
            #line = ""
            if self.target == "support":
                curr_test_LOOCV = open (self.test_path+r"test_clm_num_"+str(left_out_claim)+"_CV", 'wb')
                curr_train_LOOCV = open (self.train_path+r"train_left_out_"+str(left_out_claim)+"_CV", 'wb')
            elif self.target == "relevance":
                curr_test_LOOCV = open(self.test_path_relevance+r"test_clm_num_"+str(left_out_claim)+"_CV", 'wb')
                curr_train_LOOCV = open(self.train_path_relevance+r"train_left_out_"+str(left_out_claim)+"_CV", 'wb')
            elif self.target == "contra":
                curr_test_LOOCV = open (self.test_path_contra + r"test_clm_num_"+str(left_out_claim)+"_CV", 'wb')
                curr_train_LOOCV = open(self.train_path_contra + r"train_left_out_"+str(left_out_claim)+"_CV", 'wb')
            for curr_claim in curr_train_claims:                   
                sentences_num_to_write_curr_claim = 0
                print "    curr train claim:" +str(curr_claim)
                sentences_set = set()
                dups_cnt = 0
                for docid in self.claim_entity_doc_CE_scores_dict[curr_claim]:
                    #03/15 update - move to keep sen_num
                    for sen_num in self.claim_entity_sen_CE_scores_dict[curr_claim][docid].keys():
                        sentences_num_to_write_curr_claim += len(self.claim_entity_sen_CE_scores_dict[curr_claim][docid].keys())
                        sen = self.claim_num_sentences_num_sentences_text_dict[curr_claim][sen_num]
                        sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                        sen_no_space = sen_no_punct.replace(" ","")
                        if not sen_no_space in sentences_set:
                            try:
                                curr_features_vec = []
                                line = ""
                                sentences_set.add(sen_no_space)
#                                 if self.filter_sentences_by_sw_ratio(curr_claim, sen_num, left_out_claim) == 0:
#                                     if target_scores_dict.has_key((self.claim_dict[str(curr_claim)], sen_no_space)):
#                                         curr_support_score = target_scores_dict[(self.claim_dict[str(curr_claim)],sen_no_space)]
#                                         if curr_support_score == 1 or  curr_support_score == 2: 
#                                             print sen,"with support , skipping cus of low sw ratio"
#                                     continue
                                curr_features_vec = self.get_features_for_SVM_train_test_files(curr_claim, docid, left_out_claim, sen_num)
#                                 sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
#                                 sen_no_space = sen_no_punct.replace(" ","")
                                if target_scores_dict.has_key((self.claim_dict[str(curr_claim)], sen_no_space)):
                                    curr_support_score = target_scores_dict[(self.claim_dict[str(curr_claim)],sen_no_space)]
    #                                 print " found " + claim_dict[str(curr_claim)] +" "+ sen +" supp score:"+ str(curr_support_score)
                                else:
                                    curr_support_score = 0
                                line += str(curr_support_score) + " qid:"+ str(curr_claim)+" "
                                for feature_idx in range(0,len(curr_features_vec)):
                                        line += str(feature_idx+1)+":"+str(curr_features_vec[feature_idx])+" "
    #                             line += "\n"
                                line += "#"+self.claim_dict[str(curr_claim)] +"|"+ sen +"\n" 
                                curr_train_LOOCV.write(line)
                            except Exception as err: 
                                sys.stderr.write("problem in write_train_test_files_SVM")     
                                print err.args      
                                print err
                        else:
                            dups_cnt += 1
                print sentences_num_to_write_curr_claim , "sentences written for claim",curr_claim
#                 print curr_claim, "dups", dups_cnt ,len(sentences_set)," sentences:"
            curr_train_LOOCV.close()
            curr_claim = ""
            for docid in self.claim_entity_doc_CE_scores_dict[left_out_claim]:
                sentences_set = set() 
                for sen_num in self.claim_entity_sen_CE_scores_dict[left_out_claim][docid].keys():
                    curr_features_vec = []
                    line = ""
                    sen_text = self.claim_num_sentences_num_sentences_text_dict[left_out_claim][sen_num]
                    curr_features_vec = self.get_features_for_SVM_train_test_files(left_out_claim, docid, left_out_claim, sen_num)
                    sen_no_punct = ''.join(ch for ch in sen_text if ch not in exclude)
                    sen_no_space = sen_no_punct.replace(" ","")
                    if not sen_no_space in sentences_set:
                        sentences_set.add(sen_no_space)
#                         if self.filter_sentences_by_sw_ratio(curr_claim, sen_num, left_out_claim) == 0:
#                             if target_scores_dict.has_key((self.claim_dict[str(curr_claim)], sen_no_space)):
#                                 curr_support_score = target_scores_dict[(self.claim_dict[str(curr_claim)],sen_no_space)]
#                                 if curr_support_score == 1 or  curr_support_score == 2: 
#                                     print sen,"with support , skipping cus of low sw ratio"
#                             continue
                        if target_scores_dict.has_key((self.claim_dict[str(left_out_claim)],sen_no_space)):
                            curr_support_score = target_scores_dict[self.claim_dict[str(left_out_claim)],sen_no_space]
                        else:
                            curr_support_score = 0 
                        line += str(curr_support_score) + " qid:" + str(left_out_claim) +" "
                        for feature_idx in range(0,len(curr_features_vec)):
                            line += str(feature_idx+1)+":"+str(curr_features_vec[feature_idx]) +" "
                        line += "#" + self.claim_dict[str(left_out_claim)] +"|"+ sen_text +"\n"
                        curr_test_LOOCV.write(line)
            curr_test_LOOCV.close()    
            print "finished writing files for claim", left_out_claim
    
    def calc_num_of_support_sentences_in_data(self):
        if self.sentences_num == "allSentences":
            self.calc_num_of_support_sentences_in_data_allSentences()
        elif self.sentences_num == "topSentences":
            self.calc_num_of_support_sentences_in_data_topSentences()
   
    def create_true_num_of_support_sens_per_claim_dict(self):
        true_support_per_claim_dict = {4:3,7:4,17:8,21:2,36:2,37:5,39:5,40:2,41:1,42:2,45:2,46:5,47:2,50:0,51:0,53:0,54:1,55:3,57:9,58:4,59:5,60:11,61:3,62:0,66:2,69:0,70:0,79:0,80:0}
        self.save_pickle("true_support_per_claim_dict",true_support_per_claim_dict)
    
    def create_true_num_of_rel_sens_per_claim_dict(self):
        true_rel_per_claim_dict = {4:5,7:8,17:15,21:4,36:5,37:15,39:24,40:10,41:12,42:8,45:22,46:15,47:8,50:10,51:6,53:3,54:7,55:14,57:22,58:9,59:18,60:19,61:11,62:12,66:4,69:7,70:1,79:3,80:1}
        self.save_pickle("true_rel_per_claim_dict",true_rel_per_claim_dict)
        
    def calc_num_of_support_sentences_in_data_topSentences(self):
        """
        for each claim, calc the number of sentences in the data
        that are labeled, and that are labeled as supp
        """
        print "in calc_num_of_support_sentences_in_data"
        exclude = set(string.punctuation)
        support_scores_dict = self.convert_target_scores_dict_to_chars_as_sen("support")
        self.create_true_num_of_support_sens_per_claim_dict()
        true_support_per_claim_dict = self.read_pickle("true_support_per_claim_dict")
#         self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized")
#         self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized")
        claim_sentences_dict = self.read_pickle("relevance_baseline_topSentences")
        claim_dict = self.read_pickle("claim_dict")
        claim_num_supp_sentences = {} #key- claim num, and value is number of supporting sentences as labeled
        labeled_sen_percent_sum = 0
        support_sen_percent_sum = 0 
        sentences_num = {} ##key- claim num, value - number of sentences in the train
        for curr_claim in self.claim_list:
            curr_sen_list = []
            print "curr claim:" +str(curr_claim)
            for train_claim in claim_sentences_dict[curr_claim].keys():
#                     claim_num_supp_sentences[train_claim] = {}
                    labeled_sen_cnt = 0.0
                    supp_sen_cnt = 0.0
                    print "\t curr train claim:" +str(train_claim)
                    curr_sen_list = claim_sentences_dict[curr_claim][train_claim]
                    if sentences_num.has_key(curr_claim):
                        sentences_num[curr_claim] += len(claim_sentences_dict[curr_claim][train_claim])
                    else:
                        sentences_num[curr_claim] = len(claim_sentences_dict[curr_claim][train_claim])
                    for sen in curr_sen_list:
                        sen = sen[0] 
                        sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                        sen_no_space = sen_no_punct.replace(" ","")
                        if support_scores_dict.has_key((claim_dict[str(train_claim)],sen_no_space)):
                            labeled_sen_cnt += 1
                            if support_scores_dict[(claim_dict[str(train_claim)],sen_no_space)] != 0:
                                supp_sen_cnt += 1  
                    labeled_sen_percent = 100*float(labeled_sen_cnt/float(len(claim_sentences_dict[curr_claim][train_claim])))
                    labeled_sen_percent_sum += labeled_sen_percent
#                     supp_sen_percent = 100*float(supp_sen_cnt/float(float(len(claim_sentences_dict[curr_claim][train_claim]))))
                    if true_support_per_claim_dict[train_claim] != 0:
                        supp_sen_percent = 100*float(supp_sen_cnt/true_support_per_claim_dict[train_claim])
                    else:
                        supp_sen_percent = 0
                    support_sen_percent_sum += supp_sen_percent
                    if claim_num_supp_sentences.has_key(train_claim):
                        claim_num_supp_sentences[train_claim].append((labeled_sen_cnt, labeled_sen_percent, supp_sen_cnt, supp_sen_percent))
                    else:
                        claim_num_supp_sentences[train_claim] = [(labeled_sen_cnt, labeled_sen_percent, supp_sen_cnt, supp_sen_percent)]  
            
        #now average across all the train claims- per each in itself
        average_all_claims_labeled_percent = 0
        average_all_claims_support_percent = 0
        
        for claim in claim_num_supp_sentences.keys():
            sum_labeled_sen_cnt = 0
            sum_labeled_sen_percent = 0
            sum_supp_sen_cnt = 0
            sum_supp_sen_percent = 0
            for labeled_sen_cnt,labeled_sen_percent,supp_sen_cnt,supp_sen_percent in claim_num_supp_sentences[claim]:
                sum_labeled_sen_cnt += labeled_sen_cnt
                sum_labeled_sen_percent += labeled_sen_percent
                sum_supp_sen_cnt += supp_sen_cnt
                sum_supp_sen_percent += supp_sen_percent
#             avg_labeled_sen_cnt = float(sum_labeled_sen_cnt/(len(claim_num_supp_sentences.keys())))
            avg_labeled_sen_percent = float(sum_labeled_sen_percent/(len(claim_num_supp_sentences.keys())))
#             avg_supp_sen_cnt = float(sum_supp_sen_cnt/(len(claim_num_supp_sentences.keys())))
            avg_supp_sen_percent = float(sum_supp_sen_percent/(len(claim_num_supp_sentences.keys())))
            print str(claim) +" & " +'%.3f'%avg_labeled_sen_percent + "& " +'%.3f'%avg_supp_sen_percent
            average_all_claims_labeled_percent += avg_labeled_sen_percent
            average_all_claims_support_percent += avg_supp_sen_percent
        print "labeled sen percent avg", float(average_all_claims_labeled_percent/float(len(claim_num_supp_sentences.keys())))
        print "support sen percent avg", float(average_all_claims_support_percent/float(len(claim_num_supp_sentences.keys())))
        
    def calc_num_of_support_sentences_in_data_allSentences(self):
        """
        for each claim, calc the number of sentences in the data
        that are labeled, and that are labeled as supp
        """
        print "in calc_num_of_support_sentences_in_data"
        exclude = set(string.punctuation)
        support_scores_dict = self.convert_target_scores_dict_to_chars_as_sen("support")
        claim_sentences_dict = self.read_pickle("support_baseline_claim_sentences")
        claim_dict = self.read_pickle("claim_dict")
        claim_num_supp_sentences = {} #key- claim num, value - number of supporting sentences labled
        labeled_sen_percent_sum = 0
        for curr_claim in self.claim_list:
            labeled_sen_cnt = 0.0
            supp_sen_cnt = 0.0
            print "\t curr claim:" +str(curr_claim)
#             if self.sentences_num == "topSentences":
# #                 curr_claim = i(curr_claim)
#                 for train_claim in claim_sentences_dict[curr_claim].keys():
#                     print "\t curr train claim:" +str(curr_claim)
#                     curr_sen_list = claim_sentences_dict[curr_claim][train_claim]
#             elif self.sentences_num == "allSentences":
#                 curr_sen_list = claim_sentences_dict[curr_claim]       
            for sen in claim_sentences_dict[curr_claim] :
#                 if self.sentences_num == "topSentences":
#                     sen = sen[0] 
                sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                sen_no_space = sen_no_punct.replace(" ","")
                if support_scores_dict.has_key((claim_dict[str(curr_claim)],sen_no_space)):
                    labeled_sen_cnt += 1
                    if support_scores_dict[(claim_dict[str(curr_claim)],sen_no_space)] != 0:
                        supp_sen_cnt += 1  
            labeled_sen_percent = 100*float(labeled_sen_cnt/float(len(claim_sentences_dict[curr_claim])))
            labeled_sen_percent_sum += labeled_sen_percent
            supp_sen_percent = float(supp_sen_cnt/float(len(claim_sentences_dict[curr_claim])))
            claim_num_supp_sentences[curr_claim] = (labeled_sen_cnt, labeled_sen_percent, supp_sen_cnt, supp_sen_percent)  
        
        for claim in claim_num_supp_sentences.keys():
            print claim, claim_num_supp_sentences[claim]      
        print "labeled sen percent avg", float(labeled_sen_percent_sum/float(len(claim_num_supp_sentences.keys())))
         
    def calc_num_of_relevant_sentences_in_data(self):
        if self.sentences_num == "allSentences":
            self.calc_num_of_relevant_sentences_in_data_allSentences()
        elif self.sentences_num == "topSentences":
            self.calc_num_of_relevant_sentences_in_data_topSentences()
            
    def calc_num_of_relevant_sentences_in_data_allSentences(self):
        """
        for each claim, calc the number of sentences in the data
        that are labeled, and that are labeled as supp
        """
        exclude = set(string.punctuation)
        rel_scores_dict = self.convert_target_scores_dict_to_chars_as_sen("relevance")
#         self.claim_entity_doc_CE_scores_dict = self.read_pickle("support_model_claim_entity_doc_CE_scores_dict_normalized")
#         self.claim_entity_sen_CE_scores_dict = self.read_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized")
        claim_sentences_dict = self.read_pickle("support_baseline_claim_sentences")
        claim_dict = self.read_pickle("claim_dict")
        claim_num_rel_sentences = {} #key- claim num, value - number of supporting sentences labled
        
        for curr_claim in self.claim_list:
            rel_sen_cnt = 0.0
            print "curr train claim:" +str(curr_claim)
            for sen in claim_sentences_dict[curr_claim]:
                sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                sen_no_space = sen_no_punct.replace(" ","")
                if rel_scores_dict.has_key((claim_dict[str(curr_claim)],sen_no_space)):
                    if rel_scores_dict[(claim_dict[str(curr_claim)],sen_no_space)] != 0:
                        rel_sen_cnt += 1  
            rel_sen_percent = float(rel_sen_cnt/float(len(claim_sentences_dict[curr_claim])))
            claim_num_rel_sentences[curr_claim] = (rel_sen_cnt, rel_sen_percent)
        
        for claim in claim_num_rel_sentences.keys():
            print claim, claim_num_rel_sentences[claim]   
    
    def calc_num_of_relevant_sentences_in_data_topSentences(self):
        """
        for each claim, calc the number of sentences in the data
        that are labeled, and that are labeled as supp
        """
        exclude = set(string.punctuation)
        rel_scores_dict = self.convert_target_scores_dict_to_chars_as_sen("relevance")
        claim_sentences_dict = self.read_pickle("relevance_baseline_topSentences")
        claim_dict = self.read_pickle("claim_dict")
        self.create_true_num_of_rel_sens_per_claim_dict()
        true_rel_per_claim_dict = self.read_pickle("true_rel_per_claim_dict")
        sentences_num = {} ##key- claim num, value - number of sentences in the train
        rel_sen_percent_sum = 0
        claim_num_rel_sentences = {}
        for curr_claim in self.claim_list:
            curr_sen_list = []
            print "curr claim:" +str(curr_claim)
            for train_claim in claim_sentences_dict[curr_claim].keys():
                    rel_sen_cnt = 0.0
                    print "\t curr train claim:" +str(train_claim)
                    rel_sen_percent_sum = 0
                    curr_sen_list = claim_sentences_dict[curr_claim][train_claim]
                    if sentences_num.has_key(curr_claim):
                        sentences_num[curr_claim] += len(claim_sentences_dict[curr_claim][train_claim])
                    else:
                        sentences_num[curr_claim] = len(claim_sentences_dict[curr_claim][train_claim])            
                    for sen in curr_sen_list:
                        sen = sen[0] 
                        sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                        sen_no_space = sen_no_punct.replace(" ","")
                        if rel_scores_dict.has_key((claim_dict[str(train_claim)],sen_no_space)):
                            if rel_scores_dict[(claim_dict[str(train_claim)],sen_no_space)] != 0:
                                rel_sen_cnt += 1
                    rel_sen_percent = 100*float(rel_sen_cnt/true_rel_per_claim_dict[train_claim])
                    rel_sen_percent_sum += rel_sen_percent
                    if claim_num_rel_sentences.has_key(train_claim):
                        claim_num_rel_sentences[train_claim].append((rel_sen_cnt, rel_sen_percent))
                    else:
                        claim_num_rel_sentences[train_claim] = [( rel_sen_cnt, rel_sen_percent)]  
        
        average_all_claims_rel_percent = 0
        
        for claim in claim_num_rel_sentences.keys():
            sum_rel_sen_cnt = 0
            sum_rel_sen_percent = 0
            for rel_sen_cnt,rel_sen_percent in claim_num_rel_sentences[claim]:
                sum_rel_sen_cnt += rel_sen_cnt
                sum_rel_sen_percent += rel_sen_percent
#             avg_labeled_sen_cnt = float(sum_labeled_sen_cnt/(len(claim_num_supp_sentences.keys())))
#             avg_supp_sen_cnt = float(sum_supp_sen_cnt/(len(claim_num_supp_sentences.keys())))
            avg_rel_sen_percent = float(sum_rel_sen_percent/(len(claim_num_rel_sentences.keys())))
            print str(claim) +" & " +'%.3f'%avg_rel_sen_percent +"\\ \hline "
            average_all_claims_rel_percent += avg_rel_sen_percent
        print "rel sen percent avg", float(average_all_claims_rel_percent/float(len(claim_num_rel_sentences.keys())))
    
    def calc_num_of_supporting_judgments(self):
        exclude = set(string.punctuation)
        clm_as_key_sen_target_score_val_wiki = self.read_pickle("clm_as_key_sen_support_score_val_wiki")  # from the annotation       
        support_sentences_judgments_num_dict = {} #key is a claim num, value is the number of supporting sentence it has in the gold data
        
        for (clm_num,sen_supp_score_list) in clm_as_key_sen_target_score_val_wiki.items():
            for sen,supp_score in sen_supp_score_list:
                if supp_score == 1 or supp_score == 2:
                    if support_sentences_judgments_num_dict.has_key(clm_num):
                        support_sentences_judgments_num_dict[clm_num] += 1
                    else:
                        support_sentences_judgments_num_dict[clm_num] = 1
        self.save_pickle("support_sentences_judgments_num_dict", support_sentences_judgments_num_dict)
             
    def calc_supporting_sentences_recall_in_ranked_list(self,clm_num,pred_list,num_sentences):
        """
        for each claim, calc the number of sentences in the prediction ranked list
        that are labeled, and that are labeled as supp
        """
        print "in calc_supporting_sentences_recall_in_ranked_list..."
        exclude = set(string.punctuation)
        support_scores_dict = self.convert_target_scores_dict_to_chars_as_sen("support")
        claim_dict = self.read_pickle("claim_dict")
        support_sentences_judgments_num_dict = self.read_pickle("support_sentences_judgments_num_dict")
        num_supp_judged_sen = support_sentences_judgments_num_dict[clm_num]
        sen_no_space_set = set()
        labeled_sen_cnt = 0.0
        supp_sen_cnt = 0.0
        
        for sen,ret_score in pred_list[0:num_sentences]:
            sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
            sen_no_space = sen_no_punct.replace(" ","")
            if sen_no_space not in sen_no_space_set:
                sen_no_space_set.add(sen_no_space)
            elif sen in sen_no_space_set:
                print "sen", sen, "already in set"
                continue
            if support_scores_dict.has_key((claim_dict[str(clm_num)],sen_no_space)):
                labeled_sen_cnt += 1
                if support_scores_dict[(claim_dict[str(clm_num)],sen_no_space)] != 0:
                    supp_sen_cnt += 1  
        recall = float(supp_sen_cnt/float(num_supp_judged_sen))
        print "\t clm" , clm_num, "recall:" ,recall
        return recall
    
    def read_predicted_support_score(self):
        clm_sen_predicition_score_dict_sorted = {}                                                                                                                                                       #flag of whether the entity is the doc title or in the sen itself
    #prediction score
        prediction_score_dict = {}
        for clm_num in self.claim_list:
            if self.target == "support" or self.target == "metric_div_learn_supp_rep_rel" :
                curr_pred_file = open(self.prediction_path+"\\"+str(clm_num)+"_prediction", 'r').read().strip()
                curr_test_file = open(self.test_path+"\\"+"test_clm_num_"+str(clm_num)+"_CV", 'r').read().strip()
            elif self.target == "relevance" or self.target == "metric_div_learn_rel_rep_supp":
                curr_pred_file = open(self.prediction_path_relevance +"\\"+str(clm_num)+"_prediction", 'r').read().strip()
                curr_test_file = open(self.test_path_relevance+"\\"+"test_clm_num_"+str(clm_num)+"_CV", 'r').read().strip()
            elif self.target == "contra" :
                curr_pred_file = open(self.prediction_path_contra+"\\"+str(clm_num)+"_prediction", 'r').read().strip()
                curr_test_file = open(self.test_path_contra+"\\"+"test_clm_num_"+str(clm_num)+"_CV", 'r').read().strip()
            sen_dict = {} #key is a line number from the file, val is a sen
            for i, line in enumerate(curr_pred_file.split('\n')):
                prediction_score_dict[i] = float(line)
            try:
                for i, line in enumerate(curr_test_file.split('\n')):
                        #need to check if # is also in the sen itself, meaning if there are m
                        if line.count("#") >1:
                            sen = line.split("#",1)[1].split("|")[1]
                        else:
                            sen = line.split("#")[1].split("|")[1]
                        sen_dict[i] = sen
                clm = line.split("#")[1].split("|")[0]
            except Exception as err: 
                    sys.stderr.write('problem in calc measures: in test_file in clm '+str(clm_num))     
                    print err.args      
                    print err
            sen_predicted_score_sorted = sorted(zip(sen_dict.values(), prediction_score_dict.values()),key=lambda x: (float(x[1])), reverse=True)
            clm_sen_predicition_score_dict_sorted[clm] = (sen_predicted_score_sorted) #key is clm , value is a list of sen and the predicted score
        
        self.save_pickle("clm_as_key_sen_predicted_"+self.target+"_score_val_"+self.kernel, clm_sen_predicition_score_dict_sorted)           
#         with open("sort_sen_per_clm_pred_"+self.kernel+"_"+".csv", 'wb') as csvfile:
#                 w = csv.writer(csvfile)
#                 for (clm,sen_predicted_score_list) in clm_sen_predicition_score_dict_sorted.items():
#                     for (sen, score) in sen_predicted_score_list:
#                         w.writerow([clm,sen,str(score)])
    
    def write_top_sentences_from_prediction(self,top_sentences):
        claim_dict = self.read_pickle("claim_dict")
        clm_sen_predicition_score_dict_wiki = self.read_pickle("clm_as_key_sen_predicted_"+self.target+"_score_val_"+self.kernel)
        self.claim_sentences_docid_mapping = self.read_pickle("support_baseline_claim_sentences_docid_mapping")
        self.claim_num_sentences_text_sentences_num_dict = self.read_pickle("support_baseline_claim_num_sentences_text_sentences_num_dict_allSentences")
        self.docid_doctitle_dict = self.read_pickle("movies_docno_doctitle_dict")
        top_relevance_sentences_f = open(str(top_sentences)+"top_sentences_"+self.target+"_"+self.features_setup+"_"+self.kernel,"wb")
        
        for clm_num in self.claim_list:
            top_relevance_sentences_f.write("###"+str(clm_num)+","+claim_dict[str(clm_num)]+"\n")
            for sen_text,ret_score in clm_sen_predicition_score_dict_wiki[claim_dict[str(clm_num)]][0:top_sentences]:
                if self.claim_num_sentences_text_sentences_num_dict[clm_num].has_key(sen_text):
                    sen_num = self.claim_num_sentences_text_sentences_num_dict[clm_num][sen_text][0]
                    doctitle = self.get_sentence_document_title(clm_num, sen_num)
                top_relevance_sentences_f.write(sen_text+"|"+doctitle+"\n")
        
        top_relevance_sentences_f.close()
          
    def error_analysis(self,clm,pred_list,gold_list):
        """
        based on the gold sentences per claim, 
        find which supporting sentences were retrieved and in which rank
        """
        try:
            exclude = set(string.punctuation)
            pred_list_sen_no_punct_no_space = []
            retrieved_gold_supp_sen = []
            retrieved_gold_not_supp_sen = []
            not_retrieved_gold_sen = []
            error_analysis_f = open('error_analysis_'+self.features_setup+"_clm_"+str(clm)+".csv", 'wb')
             
            for sen,ret_score in pred_list:
                sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                sen_no_space = sen_no_punct.replace(" ","")
                pred_list_sen_no_punct_no_space.append(sen_no_space)
                
            for sen,supp_score in gold_list:
                sen_no_punct = ''.join(ch for ch in sen if ch not in exclude)
                sen_no_space = sen_no_punct.replace(" ","")
                if sen_no_space in pred_list_sen_no_punct_no_space:
                    if supp_score == 1 or supp_score == 2:
                        retrieved_gold_supp_sen.append((sen,pred_list_sen_no_punct_no_space.index(sen_no_space)))
                        error_analysis_f.write("claim "+str(clm)+" supp sen and retrieved:"+sen+" ,pred rank: "+str(pred_list_sen_no_punct_no_space.index(sen_no_space))+"\n")
                    elif supp_score == 0:
                        retrieved_gold_not_supp_sen.append((sen,pred_list_sen_no_punct_no_space.index(sen_no_space)))
                else:
                    if supp_score == 1 or supp_score == 2:   
                        not_retrieved_gold_sen.append(sen)         
            error_analysis_f.write("------------\n")
            for sen in not_retrieved_gold_sen:
                error_analysis_f.write("supp sen "+ sen +" not retrived \n")
        except Exception as err: 
            sys.stderr.write('problem in error_analysis: in clm: '+ str(clm))     
            print err.args      
            print err    
                    
    def get_sentence_document_title(self,clm_num,sen_num):
        doc_title = ""
        if self.claim_sentences_docid_mapping[clm_num].has_key(sen_num): 
            docid = self.claim_sentences_docid_mapping[clm_num][sen_num][0]
            if self.docid_doctitle_dict.has_key(docid):
                doc_title = self.docid_doctitle_dict[docid]
            else:
                print "docid not in docid_doctitle_dict "
        return doc_title
        
    def top_ranks_prediction_analysis(self,clm,pred_list):
        try:
            top_ranks_f = open('top_ranks_'+self.features_setup+"_"+self.target+"_clm_"+str(clm)+".csv", 'wb')
            top_ranks_f.write( "sen & support & doc title \n")
            self.claim_dict = self.read_pickle("claim_dict")    
            self.claim_sentences_docid_mapping = self.read_pickle("support_baseline_claim_sentences_docid_mapping")
            self.claim_num_sentences_text_sentences_num_dict = self.read_pickle("support_baseline_claim_num_sentences_text_sentences_num_dict_allSentences")
            self.docid_doctitle_dict = self.read_pickle("movies_docno_doctitle_dict")
            exclude = set(string.punctuation) 
            target_scores_dict = self.convert_target_scores_dict_to_chars_as_sen(self.target)
            p = 10
            for sen_text,pred_score in pred_list[0:p]:
                sen_no_punct = ''.join(ch for ch in sen_text if ch not in exclude)
                sen_no_space = sen_no_punct.replace(" ","")
                if target_scores_dict.has_key((self.claim_dict[str(clm)], sen_no_space)):
                    curr_support_score = target_scores_dict[(self.claim_dict[str(clm)],sen_no_space)]
                else:
                    curr_support_score = "#"
                sen_num = self.claim_num_sentences_text_sentences_num_dict[clm][sen_text]
                doc_title = self.get_sentence_document_title(clm,sen_num)
                top_ranks_f.write(sen_text+" & "+str(curr_support_score)+" & "+doc_title+"\n") 
        except Exception as err: 
                    sys.stderr.write('problem in top_ranks_prediction_analysis: in clm: '+ str(clm))     
                    print err.args      
                    print err
                    
    def process_SVM_rank_prediction_results(self,p):
        """
        the SVM rank output are scores for each claim,
        This function creates a ranking according to these scores.
        For each line in the test file, for which there is a claim and a sentence:
        """
        claim_dict = self.read_pickle("claim_dict")
        #SVM rank support score -read each predicition file, sort the 60 sentences, and calc ndcg
        clm_sen_predicition_score_dict_wiki = self.read_pickle("clm_as_key_sen_predicted_"+self.target+"_score_val_"+self.kernel)
        if self.target == "support" or self.target == "metric_div_learn_rel_rep_supp":
            clm_as_key_sen_target_score_val_wiki = self.read_pickle("clm_as_key_sen_support_score_val_wiki")    
        elif self.target == "relevance" or self.target == "metric_div_learn_supp_rep_rel":
            clm_as_key_sen_target_score_val_wiki = self.read_pickle("claim_sen_relevance_dict_wiki")
        elif self.target == "contra":
            clm_as_key_sen_target_score_val_wiki = self.read_pickle("clm_as_key_sen_true_contra_score_val_list_sorted_zero_to_two") 
        separated_list = [(clm_sen_predicition_score_dict_wiki,clm_as_key_sen_target_score_val_wiki,"wiki")]
        for (curr_pred_supp_dict,curr_true_supp_dict,curr_source) in separated_list:
            NDCG_all_claims = {} #key is a claim, value is the nDCG
            AP_all_claims= {} 
            prec_at_5_all_claims = {}
            prec_at_10_all_claims = {}
            recall = 0.0
            top_relevant_sen = 100
            for clm in self.claim_list:
                try:
                    if self.target == "support" or self.target == "metric_div_learn_rel_rep_supp" :
                        if claim_dict[str(clm)] in curr_pred_supp_dict.keys():
                            self.error_analysis(clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                            self.top_ranks_prediction_analysis(clm,curr_pred_supp_dict[claim_dict[str(clm)]])
                            NDCG_all_claims[clm] = utils.calc_emp_NDCG(curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm],p)
                            AP_all_claims[clm] = utils.calc_AP_support(curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm],p)
                            prec_at_5_all_claims[clm] = utils.calc_precision_at_k(5, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                        prec_at_10_all_claims[clm] = utils.calc_precision_at_k(10, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                    elif self.target == "contra":
                        if clm in curr_pred_supp_dict.keys():
                            NDCG_all_claims[clm] = utils.calc_emp_NDCG(curr_source,clm,curr_pred_supp_dict[clm],curr_true_supp_dict[clm],p)
                            AP_all_claims[clm] = utils.calc_AP_support(curr_source,clm,curr_pred_supp_dict[clm],curr_true_supp_dict[clm],p)                            
                            prec_at_5_all_claims[clm] = utils.calc_precision_at_k(5, curr_pred_supp_dict[clm],curr_true_supp_dict[clm])
                            prec_at_10_all_claims[clm] = utils.calc_precision_at_k(10, curr_pred_supp_dict[clm],curr_true_supp_dict[clm])
                    elif self.target == "relevance" or self.target == "metric_div_learn_supp_rep_rel":
                        if claim_dict[str(clm)] in curr_pred_supp_dict.keys():
                            self.top_ranks_prediction_analysis(clm,curr_pred_supp_dict[claim_dict[str(clm)]])
                            recall += self.calc_supporting_sentences_recall_in_ranked_list(clm,curr_pred_supp_dict[claim_dict[str(clm)]],top_relevant_sen)
#                             self.error_analysis(clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                            NDCG_all_claims[clm] = utils.calc_emp_NDCG(curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[claim_dict[str(clm)]],p)
                            AP_all_claims[clm] = utils.calc_AP_relevance(1000,curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[claim_dict[str(clm)]])                            
                            prec_at_5_all_claims[clm] = utils.calc_precision_at_k(5, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[claim_dict[str(clm)]])
                            prec_at_10_all_claims[clm] = utils.calc_precision_at_k(10, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[claim_dict[str(clm)]])
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
            
            print curr_source+ " target: "+self.target+ self.features_setup +" : in average_NDCG: " +str(average_NDCG) +" std: "+str(std_NDCG) +" MAP :" +str(MAP) + " std:"+str(std_MAP)+ " average_prec_at_5:"+ str(average_prec_at_5) +" std:"+ str(std_prec_at_5)+" average_prec_at_10:"+str(average_prec_at_10) + " std:"+ str(std_prec_at_10)
            all_claims_sorted_by_NDCG_feature = collections.OrderedDict(sorted(NDCG_all_claims.items(),key=lambda x: (float(x[1])), reverse=True))
            print "relevance recall from" ,top_relevant_sen, ":" ,float(recall/float(len(self.claim_list)))
            
            with open('SVM_rank_nDCG@'+str(p)+"_"+self.kernel+".csv", 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,ndcg_score) in all_claims_sorted_by_NDCG_feature.items():
                    w.writerow([clm,ndcg_score])
                w.writerow(['average NDCG:'+str(average_NDCG)])
            
            all_claims_sorted_by_prec_at_5_feature = collections.OrderedDict(sorted(prec_at_5_all_claims.items(),key=lambda x: (float(x[1])), reverse=True))
             
            with open('SVM_rank_prec_at_5@'+str(p)+"_"+self.kernel+".csv", 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,prec_at_5) in all_claims_sorted_by_prec_at_5_feature.items():
                    w.writerow([clm,prec_at_5])
                w.writerow(['average NDCG:'+str(average_prec_at_5)])
     
    def apply_svm_2_weight(self,model_path):
        svm_2_weight.main(model_path)
          
def get_top_k_docs_id():    
    
    try:
        features = "NLP_sen_len_sw_ratio_MRF"
        target = "support" #whether to learn with respect to the support score or relevance score
        sentences_num = "allSentences"
        support_baseline_process = support_baseline("none",features,target,sentences_num)
#         support_baseline_process.create_set_of_docs_per_claim()
        support_baseline_process.create_sen_ret_input_file()
        print "finished get_top_k_docs_id "
    
    except Exception as err: 
        sys.stderr.write("problem in get_top_k_docs_id")     
        print err.args      
        print err
      
def create_input_docs_for_SVM():
    features = "CE_all"
    target = "relevance" #whether to learn with respect to the support score or relevance score
    sentences_num = "top100MRFsentences"
    hyperEdge = "global" 
    support_baseline_process = support_baseline("none",features,target,sentences_num)
    support_baseline_process.map_claim_and_sentences_num()
    support_baseline_process.map_claim_doc_to_CE_scores()
    support_baseline_process.map_claim_sen_to_CE_scores()
    support_baseline_process.normalize_doc_CE_scores()
    support_baseline_process.normalize_sen_CE_scores()
#     if support_baseline_process.sentences_num == "topSentences":
#         support_baseline_process.map_sentences_to_their_docid()
    support_baseline_process.normalize_features()
    if support_baseline_process.sentences_num == "allSentences":
        support_baseline_process.write_train_test_files_SVM_allSentences()
#     elif support_baseline_process.sentences_num == "topSentences":
# # #         support_baseline_process.get_top_sentences_as_two_stage_process()
# # #         support_baseline_process.map_sentences_to_their_docid()
#         support_baseline_process.write_train_test_files_SVM_topSentences()
#      
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
    features = "CE_all_NLP_sen_len_MRF"
#     target = "relevance"
    target = "relevance"
    num_of_sentences = "allSentences" 
    support_baseline_process = support_baseline("linear",features,target,num_of_sentences)
    support_baseline_process.read_predicted_support_score()
    support_baseline_process.calc_num_of_supporting_judgments()
    support_baseline_process.process_SVM_rank_prediction_results(10)
    support_baseline_process.write_top_sentences_from_prediction(100)
#     support_baseline_process.apply_svm_2_weight(support_baseline_process.model_path)
#     support_baseline_process.calc_num_of_relevant_sentences_in_data()
#     support_baseline_process.calc_num_of_support_sentences_in_data()
    return

 
def main():
    get_top_k_docs_id()
#     create_input_docs_for_SVM()
#     analyze_SVM_results()
#     features = "CE_all_sentiment"
#     support_baseline_process = support_baseline("linear",features)
#     d = support_baseline_process.read_old_pickle_testing("support_model_claim_entity_sen_CE_scores_dict_normalized")
#     support_baseline_process.save_pickle("support_model_claim_entity_sen_CE_scores_dict_normalized_new", d)
if __name__ == '__main__':
    main()
