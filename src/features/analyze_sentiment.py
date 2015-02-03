'''
24.04.14
As part of the sentiment similarity task
Input- Sentences file
using Turian NN feature word vectors
Output - similarity between the sentences, based on words vector
@author: Liora
'''
from __future__ import division
import sys
import csv
import math
import pickle 
import numpy as np
import os
import pandas as pd
import subprocess 
import shlex
import collections
from my_utils import utils
# num_sen=30
tokenized_sen_file=r"RT_tokenized_sentences.csv"
is_tab = '\t'.__eq__
parser_result_file=r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity\stanford_parser_tree\parsed_sen_for_parser_M_turk_batch1"
re_written_parsed_sen=r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity\stanford_parser_tree\re_written_parsed_sen"


class sentences_similarity:
    tokenized_clm_and_sen_dict={}
    dim=300
    word_rep_file_tur= r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity\word_rep\Tur_neu_dim"+str(dim)+".txt"
    word_rep_file_word2vec = r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity\word_rep\GoogleNews-vectors-negative300.bin.gz"
    output_path= r"C:\study\technion\MSc\Thesis\Y!\sentimentSimilarity"
    word_rep_dict={}
    word_words_sim_dict={} # keeps the full cosine between all words and a word j as part of a key -  to see what are the max,min values.
    word_words_sim_dict_sorted={}
    sen_sim_res_dict={}
    sen_sim_res_dict_sorted={}
    coll_stats_dict={}
    coll_stats_dict_sorted={}
    top_K_freq_words=[]
    rating_dict={}
    RT_sen_set_dict={}
    RT_sentiment_bear_sen={}
    RT_non_sentiment_bear_sen={}
    POS_weights={}
    sentiment_bearing_tags=[]
    regular_tags=[]
    phrase_level_tags=[]
    clause_level_tags=[]
    regrssion_res_dict={}
    train_annotation_res_dict={}
    test_annotation_res_dict={}
    error_res_dict={}
    rating_hist_train={}
    rating_hist_test={}
    rating_hist_model={}
    nodes_cnt=0
    representation="word2vec"
    model=""
    claim_dict={} #key is the claim number, value is the claim text
    claim_sentiment_vector_and_label_dict={} #key is claim num, value is the sentiment vector and label
    claim_sen_sentiment_vector_and_label_dict={} #key is claim num and sen num, value is the sen sentiment vector and label
    claim_sen_dict={} #key is a tuple of claim_num, sen_num and value is the review text
    claim_sen_similarty_dict={}
    claim_sen_average_support_score={} #key is a claim text and sen text tuple, and value is the average support score for the crowdflower, calculateed in 
                                        #create_annotation_file, in process_res function
    claim_sen_sentiment_cos_simialrity_socher={} #key is claim text and sen text, and val is the similairty based on the cosine between the vectors on the 
                                            #label itself given
    claim_sen_sentiment_JSD_simialrity_socher={} #key is claim text and sen text, and val is the similairty based on the Jansen-Shannen div between the vectors                                      
    files_for_sentiment_anaysis_input_path=r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity\sentiment_analysis_socher\input_clm_sen"
    files_for_sentiment_anaysis_output_path=r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity\sentiment_analysis_socher\output_clm_sen_retrained_model"
    
#     representation="turian"
#     sen_sim_res_mat=numpy.empty((num_sen, num_sen,)).reshape((num_sen, num_sen))
    
    def __init__(self):
        self.output_path=r"C:\study\technion\MSc\Thesis\Y!\sentiment_similarity"
    
    def create_sentiment_socher_input_files(self):
           
            #read the claim and reviews files:
            try:
                test="RT_sen_sim"
                if test is "supp_clm_sen":
                    dir_path=r"C:\study\technion\MSc\Thesis\Y!\support_test\input_crowdflower_second_trial"
                    self.claim_reviews_dict = {}
                    for claim_reviews_file in os.listdir(dir_path):
                        with open (dir_path+"\\"+claim_reviews_file, 'r') as f:
                            claim_num = claim_reviews_file.split('supp_claim_')[1].split(".")[0]
                            data = pd.read_csv(f)
                            claim_text = data['claim']
                            sen = data['sen']
                            is_gold = data['_golden']
                            
                            with open ("clm_"+str(claim_num),'wb') as c_file:
                                        c_file.write(claim_text[1])
                            for sen_num in range(0,len(data)):
                                if is_gold[sen_num] !=1:
                                    with open ("clm_"+str(claim_num)+"_sen_"+str(sen_num),'wb') as rev_file:
                                        rev_file.write(sen[sen_num])
                            #insert to dict - key is a claims and sentence pair, and val is currently empty. will be field by the sentiment similarity 
                                    self.claim_dict[claim_num]=claim_text[1]
                                    self.claim_sen_dict[(claim_num,sen_num)]=sen[sen_num] #for instance -  KEY: (4,1)  VAL: Film ,2004 ,Zombies ,Dawn of the Dead ,Individuals try and survive a zombie outbreak by securing a shopping mall
    #                 self.save_pickle("claim_dict_pickle","claim_dict")
    #                 self.save_pickle("claim_sen_dict_pickle","claim_sen_dict")
                    utils.save_pickle("claim_dict_pickle", self.claim_dict)
                    utils.save_pickle("claim_sen_dict_pickle", self.claim_sen_dict)
      
            except Exception as err: 
                        sys.stderr.write('problem in create_sentiment_socher_input_files:')     
                        print err.args      
                        print err
    
    def apply_socher_sentiment_analysis_tool(self,orig_retrinaed_model):                    
        print "enter apply_socher_sentiment_analysis_tool in model: "+orig_retrinaed_model
        try:
            for filename in os.listdir(self.files_for_sentiment_anaysis_input_path):
                print "filename:"+ filename    
                ##first check with the original model from the corenlp##
                if orig_retrinaed_model is "orig":    
                    input_f=r'C:\\study\\technion\\MSc\\Thesis\\Y!\\sentiment_similarity\\sentiment_analysis_socher\\input_clm_sen\\'+filename
                    command = 'java -cp \"*\" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -output probabilities,root -file %s' % input_f
                elif orig_retrinaed_model is "retrained":
                    input_f=r'C:\\study\\technion\\MSc\\Thesis\\Y!\\sentiment_similarity\\sentiment_analysis_socher\\input_clm_sen\\'+filename
                    command = 'java -cp \"*\" edu.stanford.nlp.sentiment.SentimentPipeline -sentimentModel sentiment_model_rebuild.ser.gz -output probabilities,root -stdin < %s' % input_f
                proc = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,shell=True,stderr=subprocess.PIPE,
                                      cwd=r'C:\softwares\stanford-corenlp-full-2014-06-16\stanford-corenlp-full-2014-06-16')
                stdout, stderr = proc.communicate()
                retcode = proc.wait()
                if retcode < 0:
                    print >>sys.stderr, "error in command line", retcode
                curr_stdout_file = open(filename+"_"+orig_retrinaed_model+"_model_res.txt", "w")
                curr_stdout_file.write(stdout)
                curr_stdout_file.close()

        except Exception as err: 
            sys.stderr.write('problem in call_socher_sentiment_analysis_tool:')     
            print err.args      
            print err
    
    def process_sentiment_tool_result(self,orig_retrinaed_model):
        print "===enter process_sentiment_tool_result in model: "+orig_retrinaed_model+"==="
        try:
            self.claim_dict = utils.read_pickle("claim_dict")
            self.claim_sen_dict = utils.read_pickle("claim_sen_dict")
           
            sentiment_options = ["Very negative","Negative","Neutral","Positive","Very positive"]
            for filename in os.listdir(self.files_for_sentiment_anaysis_output_path):  # a curr_file for each claim and for each sentence
                claim_num = ""
                sen_num = ""
                sentiment_label = []
                sentiment_label_index = []
                sentiment_matrix = np.zeros(shape=((1,len(sentiment_options))))
                curr_file = open(self.files_for_sentiment_anaysis_output_path+"\\"+filename,'r').read().strip()
                #need to go over the lines to see if there is another sentence that got analyzed
                for i, line in enumerate((curr_file).split('\n')):
                    if line.startswith("  0:  "):
                        sentiment_vector_str = line.split("  0:  ")[1].split()
                        num = map(float,sentiment_vector_str)
                        sentiment_matrix = np.vstack((sentiment_matrix,num))
                        continue
                    if len(line.split()) == 1 or len(line.split()) == 2: #a label line:
                        sentiment_label.append(line.strip())
                        continue
#                     print line  
                
                #convert the sentiment label to numbers
                for l in sentiment_label:
                    for i in range(0,len(sentiment_options)):
                        if l == sentiment_options[i]:
                            sentiment_label_index.append(i+1)
                #determine for each claim and each sentence the overall sentiment vector and label             
                claim_num = filename.split("_")[1]
                if "sen" in filename:
                    sen_num =  filename.split("_")[3]
                
                #if it is more than one sentence, create a single label as a mean of the labels
                sentiment_label_index = float(sum(sentiment_label_index)/float(len(sentiment_label_index)))
                sentiment_matrix = np.delete(sentiment_matrix, 0, 0)
                sentiment_matrix = np.mean(sentiment_matrix, axis=0) #the mean of the rows in case there is more than 1 sentence
#                     
                if sen_num is "": #only a claim    
                    self.claim_sentiment_vector_and_label_dict[claim_num] = (sentiment_matrix,sentiment_label_index)
                else: #sentence of a claim
                    self.claim_sen_sentiment_vector_and_label_dict[(claim_num,sen_num)] = (sentiment_matrix,sentiment_label_index)         
            #save to file
#             with open ("claim_sentiment_socher.csv","wb") as csvfile: orig version of the model - from corenlp
            with open ("claim_sentiment_socher_"+orig_retrinaed_model+".csv","wb") as csvfile:
                clm_sen_socher = csv.writer(csvfile)
                for (clm,senti) in self.claim_sentiment_vector_and_label_dict.items():
                    clm_sen_socher.writerow([self.claim_dict[str(clm)]+" | "+','.join(map(str, senti[0]))+" | "+str(senti[1])])
            with open ("sen_sentiment_socher_"+orig_retrinaed_model+".csv","wb") as csvfile:
                sen_sen_socher = csv.writer(csvfile)
                for ((clm,sen),senti) in self.claim_sen_sentiment_vector_and_label_dict.items():
                    sen_sen_socher.writerow([self.claim_dict[str(clm)]+" | "+self.claim_sen_dict[clm,int(sen)]+"|"+','.join(map(str, senti[0]))+" | "+str(senti[1])])
            #pickle it
            utils.save_pickle("claim_sentiment_vector_and_label_dict_"+orig_retrinaed_model+"_model",self.claim_sentiment_vector_and_label_dict)
            utils.save_pickle("claim_sen_sentiment_vector_and_label_dict_"+orig_retrinaed_model+"_model",self.claim_sen_sentiment_vector_and_label_dict)
            
        except Exception as err: 
                    sys.stderr.write('problem in process_sentiment_tool_result:')     
                    print err.args      
                    print err

    def calc_sentiment_similarity_socher_tool(self,orig_retrinaed_model):
        """
        given the claim sentiment dict- key is claim num and val is sentiment vector and label as given
        by Socher's tool,
        calculate the sentiment similarity between the claim and its sentences
        """
        print "===enter calc_sentiment_similarity_socher_tool model:"+orig_retrinaed_model+"==="
        
        #original model in corenlp
        if orig_retrinaed_model is "orig":
            self.claim_sentiment_vector_and_label_dict = utils.read_pickle("claim_sentiment_vector_and_label_dict")
            self.claim_sen_sentiment_vector_and_label_dict = utils.read_pickle("claim_sen_sentiment_vector_and_label_dict")
        elif orig_retrinaed_model is "retrained":
            self.claim_sentiment_vector_and_label_dict = utils.read_pickle("claim_sentiment_vector_and_label_dict_retrained_model")
            self.claim_sen_sentiment_vector_and_label_dict = utils.read_pickle("claim_sen_sentiment_vector_and_label_dict_retrained_model")
            
        self.claim_dict = utils.read_pickle("claim_dict")
        self.claim_sen_dict = utils.read_pickle("claim_sen_dict")
#         self.read_pickle("claim_sentiment_vector_and_label_dict","claim_sentiment_vector_and_label_dict")
#         self.read_pickle("claim_sen_sentiment_vector_and_label_dict", "claim_sen_sentiment_vector_and_label_dict")
#         self.read_pickle("claim_dict", "claim_dict")
#         self.read_pickle("claim_sen_dict", "claim_sen_dict")
        #compute the similarity based on the label- a binary similarity
        for claim_num in self.claim_sentiment_vector_and_label_dict.keys():
            for (clm,sen) in self.claim_sen_sentiment_vector_and_label_dict.keys():
                if claim_num == clm:
                    #17.09.14 update - removed the label sim, not interesting for now!
#                     if not self.claim_sen_sentiment_vector_and_label_dict[clm,sen][1] == 3.0: 
#                         sen_sim_based_on_label = math.fabs(self.claim_sentiment_vector_and_label_dict[claim_num][1]-self.claim_sen_sentiment_vector_and_label_dict[clm,sen][1])#e.g Very Posirive- Positive = 5-4=1
#                     else:
#                         sen_sim_based_on_label=10
                    sen_sim_based_on_cosine = utils.cosine_measure(self.claim_sentiment_vector_and_label_dict[claim_num][0], self.claim_sen_sentiment_vector_and_label_dict[clm,sen][0])
                    #17.09.2014 edit  -  add similarity based on Jensen-Shannon div
                    sen_sim_based_on_JSD = utils.jsd(self.claim_sentiment_vector_and_label_dict[claim_num][0], self.claim_sen_sentiment_vector_and_label_dict[clm,sen][0])
#                     if sen_sim == 1 or sen_sim == 0:
                    self.claim_sen_similarty_dict[claim_num,sen]=[sen_sim_based_on_JSD,sen_sim_based_on_cosine] #key is claim num and sen num, val is the 
                                                                                #difference in the labels of the claim and sen sentiment - only cases of 1/0 matters 
                                                                                #(on 1-5 scale as Socher's output ands so 5-4, 4-4, 2 
                       
        #sort the claim sentence similarity dict by claim, and then by the sen_sim, in increarsing order
#         claim_sen_similarty_dict_based_on_label_sorted = collections.OrderedDict(sorted(self.claim_sen_similarty_dict.items(),key=lambda x: (-int(x[0][0]),-int(x[1][0])), reverse=True))
        claim_sen_similarty_dict_based_on_JSD_sorted = collections.OrderedDict(sorted(self.claim_sen_similarty_dict.items(),key=lambda x: (-int(x[0][0]),-float(x[1][0])), reverse=True)) #- float cus the smaller the JSD is, the more similar the clm and sen 
        claim_sen_similarty_dict_based_on_cosine_sorted = collections.OrderedDict(sorted(self.claim_sen_similarty_dict.items(),key=lambda x: (-int(x[0][0]),float(x[1][1])), reverse=True))           
        
        #save to file:
#         with open ("claim_sen_sentiment_similarity_based_on_label.csv","wb") as csvfile:
#             clm_sen_sim = csv.writer(csvfile)
#             for ((clm,sen),sim) in claim_sen_similarty_dict_based_on_label_sorted.items():
#                 clm_sen_sim.writerow([self.claim_dict[clm]+" | "+self.claim_sen_dict[clm,int(sen)]+" | "+str(sim[0])])
#                 self.claim_sen_sentiment_cos_simialrity_socher[(self.claim_dict[clm],self.claim_sen_dict[clm,int(sen)])]=[sim[0]]
        with open ("claim_sen_sentiment_similarity_based_on_cosine_"+orig_retrinaed_model+".csv","wb") as csvfile:
            clm_sen_sim = csv.writer(csvfile)
            for ((clm,sen),sim) in claim_sen_similarty_dict_based_on_cosine_sorted.items():
                clm_sen_sim.writerow([self.claim_dict[clm]+" | "+self.claim_sen_dict[clm,int(sen)]+" | "+str(sim[1])])
#                 self.claim_sen_sentiment_cos_simialrity_socher[(self.claim_dict[clm],self.claim_sen_dict[clm,int(sen)])].append(sim[1])
                self.claim_sen_sentiment_cos_simialrity_socher[(self.claim_dict[clm],self.claim_sen_dict[clm,int(sen)])]=sim[1]
                
        with open ("claim_sen_sentiment_similarity_based_on_JSD_"+orig_retrinaed_model+".csv","wb") as csvfile:
            clm_sen_sim = csv.writer(csvfile)
            for ((clm,sen),sim) in claim_sen_similarty_dict_based_on_JSD_sorted.items():
                clm_sen_sim.writerow([self.claim_dict[clm]+" | "+self.claim_sen_dict[clm,int(sen)]+" | "+str(sim[0])])     
                self.claim_sen_sentiment_JSD_simialrity_socher[(self.claim_dict[clm],self.claim_sen_dict[clm,int(sen)])]=sim[0]
        #save to pickle
#         utils_linux.save_pickle("claim_sen_sentiment_cos_simialrity_socher_"+orig_retrinaed_model, self.claim_sen_sentiment_cos_simialrity_socher)
#         utils_linux.save_pickle("claim_sen_sentiment_JSD_simialrity_socher_"+orig_retrinaed_model, self.claim_sen_sentiment_JSD_simialrity_socher)
#         self.save_pickle("claim_sen_sentiment_cos_simialrity_socher", "claim_sen_sentiment_cos_simialrity_socher")
        #sort the results according to the cosine/JSD sim, from the most similar to the least similar -for the ranking
        claim_sen_sentiment_cos_simialrity_socher_sorted = collections.OrderedDict(sorted(self.claim_sen_sentiment_cos_simialrity_socher.items(),key=lambda x: (x[0][0],float(x[1])), reverse=True))
        claim_sen_sentiment_JSD_simialrity_socher_sorted = collections.OrderedDict(sorted(self.claim_sen_sentiment_JSD_simialrity_socher.items(),key=lambda x: (x[0][0],-float(x[1])), reverse=True))
        utils.save_pickle("claim_sen_sentiment_cos_simialrity_socher_"+orig_retrinaed_model+"_sorted",claim_sen_sentiment_cos_simialrity_socher_sorted)
        utils.save_pickle("claim_sen_sentiment_JSD_simialrity_socher_"+orig_retrinaed_model+"_sorted",claim_sen_sentiment_JSD_simialrity_socher_sorted)
        
    def compare_sentiment_sim_support_intersection(self):
        """
        compare the dicts - the one with a claim and sentence, and the cosine sim/label sim between them
        and the second dict is the supportiveness of the sentence of a claim, as an average score of all the annotators.
        --- support scores :1 - not rel, 2- Strong contra, 3 - somewhat contra 4- neutral  5- somewhat supp, 6- strong supp ----
        Calc the "recall ": two versions
                version 1: out of all the supporting sentences (avg of the score is 5 and above) -  how many have the same sentiment as the claim
                version 2: Out of the sentences who got the majority vote of support.
        calc "precision":
                out of the same sentiment pairs (both versions), how much are suppotive
        """
        #read the avg support score of a sentence to a claim
        print "===enter compare_sentiment_sim_support_intersection==="
        claim_sen_average_support_score={}
        with open("claim_sen_average_support_score", 'rb') as handle:
                claim_sen_average_support_score = pickle.loads(handle.read()) #key is claim text and sen text , and val is the avg supp score
        #read the sentiment similarity dict
        self.claim_sen_sentiment_cos_simialrity_socher= utils.read_pickle("claim_sen_sentiment_cos_simialrity_socher")         
        support_version="majority_vote" #consider a suppportive sentence in two setups: wither received an average score of >=5, or was counted so by at least 3/4 of 5 annotators
        #calc precision and recall - as sets of the supporting pairs, and then the sen_sim pairs, and then the intersection between them 
        
        supporting_pairs_set=set()
        contradict_pairs_set=set()
        sen_sim_pairs_based_on_cosine_set=set()
        sen_sim_pairs_based_on_label_set=set()
        sen_diff_paires_based_on_cosine=set()
        if support_version is "average_score":
            for ((claim,sen),supp_score) in claim_sen_average_support_score.items():
                if supp_score >=5:
                    supporting_pairs_set.add((claim,sen))
        elif support_version is "majority_vote":
            d=utils.read_pickle("claim_support_sen_by_majority_dict").keys()
        for (claim,sen) in d:
            supporting_pairs_set.add((claim,sen))   
        
        for ((claim,sen),sen_sim) in self.claim_sen_sentiment_cos_simialrity_socher.items():
            if sen_sim[0] == 0 or sen_sim[0] == 1:
                sen_sim_pairs_based_on_label_set.add((claim,sen))
            if sen_sim[1] >0.75:
                sen_sim_pairs_based_on_cosine_set.add((claim,sen))
        
        recall_based_on_label=float(len(supporting_pairs_set.intersection(sen_sim_pairs_based_on_label_set)))/float(len(supporting_pairs_set))
        recall_based_on_cosine=float(len(supporting_pairs_set.intersection(sen_sim_pairs_based_on_cosine_set)))/float(len(supporting_pairs_set))
        precision_based_on_label=float(len(sen_sim_pairs_based_on_label_set.intersection(supporting_pairs_set)))/float(len(sen_sim_pairs_based_on_label_set))
        precision_based_on_cosine=float(len(sen_sim_pairs_based_on_cosine_set.intersection(supporting_pairs_set)))/float(len(sen_sim_pairs_based_on_cosine_set))
        #update from 29.07 - calculate the accuracy: agreed- number of sentences that are supp and sen sim and num of sen that are cont and sen_diff. 
        #vs. disagreed- number of sentences that are sen_sim and contradict, and num of sentences that are sen_diff and support
#         agreed_supp_sen_sim=supporting_pairs_set.intersection(sen_sim_pairs_based_on_cosine_set)
#         agreed_contra_diff_sen=self.contradict_set.intersection(self.diff_senti_set)
#         agreed=agreed_supp_sen_sim.union(agreed_contra_diff_sen)
#         disagreed_contra_sen_sim=self.contradict_set.intersection(self.same_senti_set)
#         disagreed_supp_diff_sen=self.supp_set.intersection(self.diff_senti_set)
#         disagreed=disagreed_contra_sen_sim.union(disagreed_supp_diff_sen)
#         
#         accuracy=float(float(len(agreed))/float(len(agreed)+len(disagreed)))
# #         print accuracy
#         accuracy_sen_sim_supp=float(len(agreed_supp_sen_sim))/float((len(disagreed)+len(agreed_supp_sen_sim)))
#         accuracy_diff_sen_contra=float(len(agreed_contra_diff_sen))/float((len(agreed_contra_diff_sen)+len(disagreed)))
#         print ('accuracy: '+str(accuracy)+ " accuracy_sen_sim_supp: "+str(accuracy_sen_sim_supp) +" accuracy_diff_sen_contra: "+str(accuracy_diff_sen_contra) )

        
        print "precision_based_on_label " +str(precision_based_on_label)
        print "precision_based_on_cosine "+ str(precision_based_on_cosine)
        print "recall_based_on_label " +str(recall_based_on_label)
        print "recall_based_on_cosine "+ str(recall_based_on_cosine)
        print "F1 measure using cosine" +str(float(2*(precision_based_on_cosine*recall_based_on_cosine)/float(precision_based_on_cosine+recall_based_on_cosine)))
        
        supp_sen_sim_intersection=supporting_pairs_set.intersection(sen_sim_pairs_based_on_cosine_set)
        #check the sen_sim sentences but not supportive
        sen_sim_not_supp = sen_sim_pairs_based_on_cosine_set - supporting_pairs_set
        supp_not_sen_sim = supporting_pairs_set - sen_sim_pairs_based_on_cosine_set
        #convert to dict for pickling and saving to text
        supp_sen_sim_intersection_dict=dict.fromkeys(supp_sen_sim_intersection, 0)
        sen_sim_not_supp_dict=dict.fromkeys(sen_sim_not_supp, 0)
        supp_not_sen_sim_dict=dict.fromkeys(supp_not_sen_sim, 0)
        
        dict_list=[(supp_sen_sim_intersection_dict,"supp_sen_sim_intersection"),(sen_sim_not_supp_dict,"sen_sim_not_supp_dict"),
                   (supp_not_sen_sim_dict,"supp_not_sen_sim_dict")]
        for (d,d_name) in dict_list:
        #write to file
            try:   
                with open (d_name+".csv","wb") as csvfile:
                    d_w = csv.writer(csvfile)
                    for (clm,sen) in d.keys():
                        d_w.writerow([clm+" | "+sen +"|supp score"+str(claim_sen_average_support_score[(clm,sen)])+" | sen_sim"+
                                        ' '.join(str(self.claim_sen_sentiment_cos_simialrity_socher[(clm,sen)]))])
            except Exception as err: 
                            sys.stderr.write('problem in compare_sentiment_sim_suppoer_intersection:'+clm +" "+ sen)     
                            print err.args      
                            print err


def main():
    try: 
        sen_sim = sentences_similarity()
        orig_retrinaed_model = "retrained"
#         sen_sim.create_sentiment_socher_input_files()
#         sen_sim.apply_socher_sentiment_analysis_tool(orig_retrinaed_model)
#         sen_sim.process_sentiment_tool_result(orig_retrinaed_model)
        sen_sim.calc_sentiment_similarity_socher_tool(orig_retrinaed_model)
#         sen_sim.compare_sentiment_sim_support_intersection()
        print "finished main"
            
    except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err


if __name__ == '__main__':
    main() 