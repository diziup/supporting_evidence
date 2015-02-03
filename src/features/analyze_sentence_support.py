'''
04.07
Model for supportvness of sentences, 
after the annotation test for support sentences,
'''
import sys
import numpy as np
import os
import pandas as pd
from fileinput import filename
import pickle
import math
import csv
import collections
from my_utils import utils
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import scipy.stats
import random
import copy
import string

class supportive_sentence():
#     support_annotated_files_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\output_crowdflower_second_trial"
    support_annotated_files_path=r"C:\study\technion\MSc\Thesis\Y!\support_test\output_crowdflower_second_trial"
    sentence_supp_score_and_new_rel_score={} #key sentence, val -  a tuple of support annotation score, and new relevance score
    shallow_pool_dict_wiki={}
    shallow_pool_dict_RT={}
    
    relevant_sen_set=set()
    not_rel_sen_set=set()
    supp_set=set()
    strong_supp_set=set()
    contradict_set=set()
    strong_contradict_set=set()
    #sentiment
    same_senti_set=set()
    diff_senti_set=set()
    #claim_sentence sim
    clm_sen_sim_res={}
    similar_claim_sen_set=set()
    diff_claim_sen_set=set()
    
    neutral_set=set()
    undecided_set=set()
    
    claim_num_list=[]
    claim_sen_sentiment_cos_simialrity_socher={}
    clm_sen_cosine_sim_res={}
    
    claim_dict={} #key is a claim, val is num of sentences in the intersection between suppotivneess and another featiure - sentiment sim, SVM_sim...
    
    clm_sen_doc_title_dict={} #key is a claim and sentece pair,  and value is the doc title 
    
    claim_num_and_text={} #key is claim text value is the claim number
    
    def __init__(self):
        relevant_sen_set=set()
        not_rel_sen_set=set()
        supp_set=set()
        contradict_set=set()
        neutral_set=set()
        same_senti_set=set()
        diff_senti_set=set()
        claim_num_list=[]
        

    def divide_sen_to_sets_support(self):
        try:
            """
            1.read the sentences from the annotation files,
            2.seprate to support,contradict etc
            3. calculate the avg number of support, contradict ...sentecnes across claims.
            4. calculate the std of this categorization.
            5. claims that are 1/2 std from the avg - are "outlier"
    #         """
            full_answers_possibilities=["Not relevant","Strong  Contradiction","Moderate Contradiction","Neutral","Moderate  Support","Strong Support"]
            reduced_answers_possibilities=["Not relevant","Contradiction","Neutral","Support"]
            check="full"
            files_num=0
            exclude = set(string.punctuation)
             
            if check is "full":
                answers_possibilities=full_answers_possibilities
            else:
                answers_possibilities=reduced_answers_possibilities
            total_mat=np.zeros((1, len(answers_possibilities))) 
            claim_text_supp_sen={} #claim and sen that were found to be supp by the majoruty (3/4 out of 5)
            relevant_majority_dict={}
            not_rel_majority_dict={}
            support_majority_dict={}
            contra_majority_dict={}
            neutral_majority_dict={}
            undecided_majority_dict={}
            not_rel_majority_dict={}
            
            claim_text_number_of_sentences_in_categorzation={} #key is claim text, val is a 4 entries vector -  for each category the number of sentences in it, for std calculation
                                                                #in the following order - not relevant, relevant,contra, neutra, supp, indecided
            #update from '11.14- have a dict of the sen and the doc title, for the document retrieval baseline comparison
            sen_doc_title_dict = {} #key is a sen, value is the doc title.
            curr_doc_title = ""
            
            self.relevant_sen_set=set()
            self.supp_set=set()
            self.contradict_set=set()
            self.neutral_set=set()
            self.undecided_set=set() 
            #for the NDCG calculation:
            curr_supp_with_score_set=set()
            curr_strong_supp_with_score_supp_set=set()              
            curr_contradict_with_score_set=set()
            curr_strong_contradict_with_score_set=set()
            curr_neutral_with_score_set=set()        
            curr_not_rel_with_score_set=set()        
            curr_undecided_with_score_set=set()        
            
            
            for filename in os.listdir(self.support_annotated_files_path):
    #             curr_mat=np.zeros((60, len(answers_possibilities))) # 10 sentences
                if filename.split("_")[0]=="f":
    #             if filename == "f_"+str(claim_num)+".csv":
                    
                    files_num+=1               
                    relevant_dict={}
                    support_dict={}
                    contra_dict={}
                    neutral_dict={}
                    undecided_dict={}
                    curr_relevant_sen_set=set()
                    curr_supp_set=set()
                    curr_strong_supp_set=set()              
                    curr_contradict_set=set()
                    curr_strong_contradict_set=set()
                    curr_neutral_set=set()
                    curr_undecided_set=set()
                    curr_not_rel_set=set()
                   
                    
                    claim_num=filename.split("_")[1].split(".")[0]
                    with open(self.support_annotated_files_path+"\\"+filename, 'r') as f:
                        data = pd.read_csv(f)
                        answer=data['to_what_extent_does_the_sentence_support_the_claim_or_contradict_it']
                        sentence=data['sen']
                        claim_text = data['claim'][1]
                        doc_title=data['tit']
                        is_gold=data['orig__golden'] 
                        sentence_lines_dict={}# a dict key is sentences, value is list lines it is in
    #                     trick_sen_file=open(self.trick_question_singel_path,'r').read().strip()
                        self.claim_num_and_text[claim_text]=claim_num 
                        for line_num in range(0,len(data)):
                            if is_gold[line_num] != 1 :
                                if sentence[line_num] in sentence_lines_dict.keys():
                                    sentence_lines_dict[sentence[line_num]].append(line_num)
                                else:
                                    sentence_lines_dict[sentence[line_num]]=[line_num]
                        
                        for sen in range(0,len(sentence_lines_dict)):  #go over all the sentences we have and their corresponding line nums- keys
                            curr_sen_rating=[]
                            for line_num in sentence_lines_dict.values()[sen][0:5]:#limit to 5 annotators
                                if check is "reduced":
                                    ans=answer[line_num]
                                    curr_doc_title=doc_title[line_num]
                                    if ans == "Not relevant":
    #                                     curr_mat[sen][0]+=1
                                        curr_sen_rating.append(1)   
                                    elif ans == "Strong  Contradiction" or ans == "Moderate Contradiction":
    #                                     curr_mat[sen][1]+=1
                                        curr_sen_rating.append(2)  
                                    elif ans == "Neutral":
    #                                     curr_mat[sen][2]+=1
                                        curr_sen_rating.append(3) 
                                    elif ans == "Strong Support" or ans == "Moderate  Support":
    #                                     curr_mat[sen][3]+=1
                                        curr_sen_rating.append(4) 
                                   
                                elif check is "full":
                                    ans=answer[line_num]
                                    for ans_poss in range(0,len(answers_possibilities)):#21.06
                                        if ans == answers_possibilities[ans_poss]:
    #                                         curr_mat[sen][ans_poss]+=1
                                            curr_sen_rating.append(ans_poss+1) 
                                            curr_doc_title=doc_title[line_num] 
                                            break
                            sen_no_punc = ''.join(ch for ch in sentence_lines_dict.keys()[sen] if ch not in exclude)
                            sen_no_space = sen_no_punc.replace(" ","")
                            sen_doc_title_dict[sen_no_space] = curr_doc_title
                            
                            #29.07.14 update- change from sentence set to claim and sentences pair set
                            if check is "reduced":
                                if curr_sen_rating.count(1) < 3:  #max 2 people marked it an not "Not Rel"
                                    self.relevant_sen_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_relevant_sen_set.add((claim_text,sentence_lines_dict.keys()[sen])) #the temporal for each claim
                                else:
                                    self.not_rel_sen_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_not_rel_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    continue
                                if curr_sen_rating.count(4) > 2: #a supportive sentence by the majority
                                    self.supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                elif curr_sen_rating.count(2) > 2: #contra sen
                                    self.contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                elif curr_sen_rating.count(3) > 2: #neutral sen
                                    self.neutral_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_neutral_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                else:
                                    self.undecided_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_undecided_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                            #rating scale for nDCG calculation - not rel and undecided=0, strong contra=1, contra=2, neut=3, supp=4. strong supp =5
                            elif check is "full":
                                if curr_sen_rating.count(1) < 3:  #max 2 people marked it an not "Not Rel"
                                    self.relevant_sen_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_relevant_sen_set.add((claim_text,sentence_lines_dict.keys()[sen])) #the temporal for each claim
                                else:
                                    self.not_rel_sen_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_not_rel_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_not_rel_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],0))
                                    #07.09 update -  get a score of -1 to non relevant and a score or 0 to undecided
                                    continue
                                if curr_sen_rating.count(5) > 2: #a moderate supportive sentence by the majority
                                    self.supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_supp_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],4))
                                elif curr_sen_rating.count(6) > 2: #a strong supportive sentence by the majority
                                    self.strong_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_strong_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_strong_supp_with_score_supp_set.add((claim_text,sentence_lines_dict.keys()[sen],5))
                                elif curr_sen_rating.count(2) > 2: #strong contra sen
                                    self.strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_strong_contradict_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],1))
                                elif curr_sen_rating.count(3) > 2: #moderate contra sen
                                    self.contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_contradict_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],2))
                                elif curr_sen_rating.count(4) > 2: #neutral sen
                                    self.neutral_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_neutral_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_neutral_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],3))
                                #else: 07.09 update -  add all the sentences - the none-relevant and the undecided too, when the undecided sentences will be randomized chosen to some category.
                                #update 10.09.14 -  if there are two people that said strong supp, and 2 that said support-  randomly choose either category...
                                #so that this will not be in the not-decided category
                                elif curr_sen_rating.count(5)==2 and curr_sen_rating.count(6)==2:
                                    rand_num = random.randrange(0,1)
                                    if rand_num == 0:
                                        self.strong_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_supp_with_score_supp_set.add((claim_text,sentence_lines_dict.keys()[sen],5)) 
                                    else:
                                        self.supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_supp_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],4))
                                elif curr_sen_rating.count(5)== 2 and curr_sen_rating.count(6)==1: #2 mod support and 1 strong...
                                    self.supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_supp_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],4))
                                elif curr_sen_rating.count(5)== 1 and curr_sen_rating.count(6)== 2: #1 mod support and 2 strong...
                                    self.supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_supp_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                    curr_supp_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],5))
                                elif curr_sen_rating.count(2)==2 and curr_sen_rating.count(3)==2 :
                                    rand_num = random.randrange(0,1)
                                    if rand_num == 0:
                                        self.strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_contradict_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],1))
                                    else:
                                        self.contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_contradict_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],2))
                                elif curr_sen_rating.count(2)==2 and curr_sen_rating.count(3)==1 :#2 strong cont and 1 mod cont
                                        self.strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_contradict_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],1))
                                elif curr_sen_rating.count(2)==1 and curr_sen_rating.count(3)==2 :#1 strong cont and 2 mod cont
                                        self.strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_contradict_set.add((claim_text,sentence_lines_dict.keys()[sen]))
                                        curr_strong_contradict_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],2))
                                else: #undecided pair
                                    self.undecided_set.add((claim_text,sentence_lines_dict.keys()[sen])) 
                                    curr_undecided_set.add((claim_text,sentence_lines_dict.keys()[sen])) 
                                    curr_undecided_with_score_set.add((claim_text,sentence_lines_dict.keys()[sen],0))
                                
                                                         
    #                         if curr_sen_rating.count(1) < 3:  #max 2 people marked it an not "Not Rel"
    #                             self.relevant_sen_set.add(sentence_lines_dict.keys()[sen])
    #                         else:
    #                             self.not_rel_sen_set.add(sentence_lines_dict.keys()[sen])
    #                             continue
    #                         if curr_sen_rating.count(4) > 2: #a supportive sentence by the majority
    #                             self.supp_set.add(sentence_lines_dict.keys()[sen])
    #                         elif curr_sen_rating.count(2) > 2: #contra sen
    #                             self.contradict_set.add(sentence_lines_dict.keys()[sen])
    #                         elif curr_sen_rating.count(3) > 2: #neutral sen
    #                             self.neutral_set.add(sentence_lines_dict.keys()[sen])
    #                         else:
    #                             self.undecided_set.add(sentence_lines_dict.keys()[sen])
                        
    #                 relevant_dict=dict.fromkeys(self.relevant_sen_set,0)
    #                 support_dict=dict.fromkeys(self.supp_set,0)
    #                 contra_dict=dict.fromkeys(self.contradict_set,0)
    #                 neutral_dict=dict.fromkeys(self.neutral_set,0)
    #                 undecided_dict=dict.fromkeys(self.undecided_set,0)
                    #08/08/14 create a ranked list of clm and sen- from the most support to the least support. for the NDCG calc
                    #for every set, group by the claim, as the NDCG is calculated per claim.
                    
                    if check is "reduced":
                        claim_text_number_of_sentences_in_categorzation[claim_text]=[len(curr_not_rel_set),len(curr_relevant_sen_set),len(curr_contradict_set),
                                                                                     len(curr_neutral_set),len(curr_supp_set),len(curr_undecided_set)]
                    else:
                        claim_text_number_of_sentences_in_categorzation[claim_text]=[len(curr_not_rel_set),len(curr_relevant_sen_set),len(curr_strong_contradict_set),
                                                                                     len(curr_contradict_set),len(curr_neutral_set),
                                                                                     len(curr_supp_set),len(curr_strong_supp_set),len(curr_undecided_set)]
#                     print("for claim:" +str(claim_num)+" out of "+str(len(curr_relevant_sen_set)) +" relevant, "+str(len(curr_supp_set))+" support,"  
#                             +str(len(curr_contradict_set))+" contrdict, "+str(len(curr_neutral_set))+" neutral, "+str(len(curr_undecided_set))+ " undecided," +str(len(curr_not_rel_set))+" not relevant")                
    #                 self.save_to_csv_file(relevant_dict, 'relevant_dict_'+str(claim_num))
    #                 self.save_to_csv_file(support_dict, 'support_dict_'+str(claim_num))
    #                 self.save_to_csv_file(contra_dict, 'contra_dict_'+str(claim_num))
    #                 self.save_to_csv_file(neutral_dict, 'neutral_dict_'+str(claim_num))
    #                 self.save_to_csv_file(undecided_dict,'undecided_dict_'+str(claim_num))                    
            
            clm_sen_support_ranking={}
            neutral_with_score_dict=dict.fromkeys(curr_neutral_with_score_set)
            supp_with_score_dict=dict.fromkeys(curr_supp_with_score_set)
            strong_supp_with_score_dict=dict.fromkeys(curr_strong_supp_with_score_supp_set)
            contra_with_score_dict=dict.fromkeys(curr_contradict_with_score_set)
            strong_contra_with_score_dict=dict.fromkeys(curr_strong_contradict_with_score_set)
            not_rel_with_score_dict=dict.fromkeys(curr_not_rel_with_score_set)
            undecided_with_score_dict=dict.fromkeys(curr_undecided_with_score_set)
            # 07.09 update - add the undecided and not relevant sentences to the set, for the all the sentences NDCG calculations -  remove if want to go back to previos
        
#             neutral_with_score_group_by_clm_dict=collections.OrderedDict(sorted(neutral_with_score_dict.items(),key=lambda x: (str(x[0][0])), reverse=True))
#             supp_with_score_group_by_clm_dict=collections.OrderedDict(sorted(supp_with_score_dict.items(),key=lambda x: (str(x[0][0])), reverse=True))
#             strong_supp_with_score_dict_group_by_clm_dict=collections.OrderedDict(sorted(strong_supp_with_score_dict.items(),key=lambda x: (str(x[0][0])), reverse=True))
#             contra_with_score_group_by_clm_dict=collections.OrderedDict(sorted(contra_with_score_dict.items(),key=lambda x: (str(x[0][0])), reverse=True))
#             strong_contra_with_score_group_by_clm_dict=collections.OrderedDict(sorted(strong_contra_with_score_dict.items(),key=lambda x: (str(x[0][0])), reverse=True))
            
            not_rel_and_undecided_dict = {}
            not_rel_and_undecided_dict.update((undecided_with_score_dict))
            not_rel_and_undecided_dict.update((not_rel_with_score_dict))
            self.calc_variance_on_all_data([not_rel_and_undecided_dict,strong_contra_with_score_dict,contra_with_score_dict,neutral_with_score_dict,
                                            supp_with_score_dict,strong_supp_with_score_dict])
                
            clm_sen_support_ranking.update(neutral_with_score_dict)
            clm_sen_support_ranking.update(supp_with_score_dict)
            clm_sen_support_ranking.update(strong_supp_with_score_dict)
            clm_sen_support_ranking.update(contra_with_score_dict)
            clm_sen_support_ranking.update(strong_contra_with_score_dict)
            #07.09 update- removie if want to go back to previous version
            clm_sen_support_ranking.update(not_rel_with_score_dict)
            clm_sen_support_ranking.update(undecided_with_score_dict)
            
            #group by claim and then sort for the supportiveness
            clm_sen_support_ranking_sorted=collections.OrderedDict(sorted(clm_sen_support_ranking.items(),key=lambda x: (str(x[0][0]),int(x[0][2])), reverse=True))
            #save to a pickle
            utils.save_pickle("clm_sen_support_ranking_sorted_full", clm_sen_support_ranking_sorted)
            self.save_to_csv_file(clm_sen_support_ranking_sorted, "clm_sen_support_ranking_sorted_full.csv")
            #across all claims, save 
            utils.save_pickle("claim_text_number_of_sentences_in_categorzation", claim_text_number_of_sentences_in_categorzation)
            relevant_majority_dict=dict.fromkeys(self.relevant_sen_set,0)
            support_majority_dict=dict.fromkeys(self.supp_set,0)
            contra_majority_dict=dict.fromkeys(self.contradict_set,0)
            neutral_majority_dict=dict.fromkeys(self.neutral_set,0)
            undecided_majority_dict=dict.fromkeys(self.undecided_set,0)
            not_rel_majority_dict=dict.fromkeys(self.not_rel_sen_set,0)      
                  
    #        add doc title
            major_dict_list=[(relevant_majority_dict,"relevant_majority_dict"),(support_majority_dict,"support_majority_dict"),
                       (contra_majority_dict,"contra_majority_dict"),(neutral_majority_dict,"neutral_majority_dict"),
                       (undecided_majority_dict,"undecided_majority_dict"),(not_rel_majority_dict,"not_rel_majority_dict")]
            for (d,d_name) in major_dict_list:
                new_dict=self.add_doc_title_to_clm_sen_pair(d,d_name)           
    #             self.save_to_csv_file(new_dict,d_name+'.csv')
    
    #         self.save_to_csv_file(self.relevant_majority_dict, 'relevant_majority_dict.csv')
    #         self.save_to_csv_file(self.support_majority_dict, 'support_majority_dict.csv')
    #         self.save_to_csv_file(self.contra_majority_dict, 'contra_majority_dict.csv')
    #         self.save_to_csv_file(self.neutral_majority_dict, 'neutral_majority_dict.csv')
    #         self.save_to_csv_file(self.undecided_majority_dict,'undecided_majority_dict.csv')                    
            utils.save_pickle("sen_doc_title_dict",sen_doc_title_dict)
            #save to pickle
    #         utils_linux.save_pickle("claim_support_sen_by_majority_dict",claim_text_supp_sen)
            utils.save_pickle("relevant_majority_dict", relevant_majority_dict)
            utils.save_pickle("support_majority_dict", support_majority_dict)
            utils.save_pickle("contra_majority_dict", contra_majority_dict)
            utils.save_pickle("neutral_majority_dict", neutral_majority_dict)
            utils.save_pickle("undecided_majority_dict", undecided_majority_dict)
            utils.save_pickle("not_rel_majority_dict", not_rel_majority_dict)
            
    #         print ("relevant avg "+ str(float(float(majority_rel_tot)/float(files_num))))
    #         print ("support avg "+ str(float(float(majority_supp_tot)/float(files_num))))
    #         print ("contradict avg "+ str(float(float(majority_cont_tot)/float(files_num)))) 
    #         print ("neutral avg "+ str(float(float(majority_neu_tot)/float(files_num))))
    #         print ("undecided avg "+ str(float(float(majority_undecided_tot)/float(files_num))))
            relevant_avg=float(float(len(relevant_majority_dict))/float(files_num))
            support_avg=float(float(len(support_majority_dict))/float(files_num))
            contradict_avg=float(float(len(contra_majority_dict))/float(files_num))
            neutral_avg=float(float(len(neutral_majority_dict))/float(files_num))
            undecided_avg=float(float(len(undecided_majority_dict))/float(files_num))
            not_rel_avg= float(float(len(not_rel_majority_dict))/float(files_num))
            print ("not rel avg "+ str(not_rel_avg))
            print ("relevant avg "+ str(relevant_avg))
            print ("contradict avg "+ str(contradict_avg)) 
            print ("neutral avg "+ str(neutral_avg))
            print ("support avg "+ str(support_avg)) 
            print ("undecided avg "+ str(undecided_avg))
            
            with open("claim_text_number_of_sentences_in_categorzation.csv", 'wb') as csvfile:
                    w = csv.writer(csvfile)
                    w.writerow(['not rel| rel|--|-|0 |+|++|undecided'])     
                    for item in claim_text_number_of_sentences_in_categorzation.items():
                        w.writerow([item])              
        
#             calc_variance_categorization(claim_text_number_of_sentences_in_categorzation,not_rel_avg,relevant_avg,contradict_avg,neutral_avg,support_avg,undecided_avg)
        except Exception as err: 
            sys.stderr.write('problem in categorize_support:')     
            print err.args      
            print err 
            
    
#     def calc_variance_on_all_data(self,not_rel_with_score_dict,supp_with_score_dict,strong_supp_with_score_dict,contra_with_score_dict
#                                   ,strong_contra_with_score_dict):
    def calc_variance_on_all_data(self,list_of_categprization_dict): #order according to score...0  is the not relevant , 1 in the strong contra...
#         total_num_of_pairs = len(not_rel_with_score_dict) + len(supp_with_score_dict)+ len(strong_supp_with_score_dict)+ len(contra_with_score_dict) + len(strong_contra_with_score_dict)
        total_num_of_pairs = 0
        total_score_avgerage = 0
        total_var = 0
        for d in list_of_categprization_dict:
            total_num_of_pairs += len(d)       
        
        for d_idx in range(0,len(list_of_categprization_dict)): 
            total_score_avgerage += d_idx*float(len(list_of_categprization_dict[d_idx])/float(total_num_of_pairs)) 
        #calc variance
        for d_idx in range(0,len(list_of_categprization_dict)): 
            total_var += len(list_of_categprization_dict[d_idx])*((d_idx-total_score_avgerage)**2)
        total_var = float(total_var/total_num_of_pairs)
        print total_var
            
    def calc_variance_categorization(self,claim_text_number_of_sentences_in_categorzation,not_rel_avg,relevant_avg,contradict_avg,neutral_avg,support_avg,undecided_avg):
        #calc variance of the categorization - for every claim 1/n*(x_i-avg)^2
            relevant_var=0
            supp_var=0
            contradict_var=0
            neutral_var=0
            undecided_var=0
            not_rel_var=0
            var_list=[not_rel_var,relevant_var,contradict_var,neutral_var,supp_var,undecided_var]
            avg_list=[not_rel_avg,relevant_avg,contradict_avg,neutral_avg,support_avg,undecided_avg]
            #not relevant, relevant,contra, neutra, supp, indecided
            for claim_text in claim_text_number_of_sentences_in_categorzation.keys():
                for var_idx in range(0,len(var_list)):
                    var_list[var_idx]+=(claim_text_number_of_sentences_in_categorzation[claim_text][var_idx]-avg_list[var_idx])**2
            
            var_list= [float(var/len(claim_text_number_of_sentences_in_categorzation.keys())) for var in var_list]
            std_list=[math.sqrt(curr_var) for curr_var in var_list]
            print ("not rel std "+ str(std_list[0]))
            print ("relevant std "+ str(std_list[1]))
            print ("contradict std "+ str(std_list[2])) 
            print ("neutral std "+ str(std_list[3]))
            print ("support std "+ str(std_list[4])) 
            print ("undecided std "+ str(std_list[5]))
            
            #find the claims that their number of relevant/ supportive sentence is more than 1/2 std from the average of each category
            category_idx_and_name_mapping={0:'not relevant',1:"relevant",2:"contradict",3:"neutral",4:"support",5:"undecided"}
            outlier_claim=[]
            std=1 
            for (claim_text,categories_num_of_sentences) in claim_text_number_of_sentences_in_categorzation.items():
                for category_idx in range(0,len(categories_num_of_sentences)):
                    if math.fabs(categories_num_of_sentences[category_idx]-avg_list[category_idx]) > std*std_list[category_idx]:
                        outlier_claim.append((claim_text,category_idx_and_name_mapping[category_idx],
                                               claim_text_number_of_sentences_in_categorzation[claim_text][category_idx],avg_list[category_idx]))
            with open("outlier_claims_categories_"+str(std)+".csv", 'wb') as csvfile:
                w = csv.writer(csvfile)
                for item in outlier_claim:
                    w.writerow([item])
            
            #plot a bar chart for all the claims, in each category  
            #plot histograms from the entire data: first for relevant and not relevant category
            #second for support categories
        
    #             self.plot_averaged_categories_bar_chart(not_rel_avg,relevant_avg,support_avg,contradict_avg,neutral_avg,undecided_avg,std_list)
    #             self.plot_all_claims_categories_bar_chart(claim_text_number_of_sentences_in_categorzation)=     
    
    def plot_all_claims_categories_bar_chart(self,claim_text_number_of_sentences_in_categorzation):
        #according to http://matplotlib.org/1.3.1/examples/api/barchart_demo.html
#         claims_num=[self.claim_num_and_text[claim_text] for claim_text in claim_text_number_of_sentences_in_categorzation.keys()]
        claims_text=[claim_text for claim_text in claim_text_number_of_sentences_in_categorzation.keys()]
        claims_text_n=[]
        #add \n between the claims text
        for claim in claims_text:
            words=claim.split()
            n_claim=""
            for word in words:
                n_claim += word + "\n"
            claims_text_n.append(n_claim)
        
        fig = pl.figure(figsize=(150, 150))
        #order of categories: not relevant, relevant,contra, neutra, supp, indecided 
        categories_data_list=[[claim_data[0],claim_data[1],claim_data[2],claim_data[3],claim_data[4],claim_data[5]] for claim_data  in claim_text_number_of_sentences_in_categorzation.values() ]
        not_rel_data=zip(*categories_data_list)[0]
        not_rel_percentage=[float(float(not_rel_d)/float(60)) for not_rel_d in not_rel_data]
        rel_data=zip(*categories_data_list)[1]
        rel_percentage = [float(float(rel_d)/float(60)) for rel_d in rel_data]
        #sort the rel and not rel
        rel_and_claims_zippped=zip(rel_percentage,not_rel_percentage,claims_text_n)
        rel_and_claims_zippped_sorted=sorted(rel_and_claims_zippped)
        rel_percentage_sorted,not_rel_percentage_sorted,claims_text_sorted=zip(*rel_and_claims_zippped_sorted)
    
        #plot bar graph    
        font = {'family' : 'normal',
                'weight':'normal',
                 'size'   : 9}
    
        plt.rc('font', **font)  
        num_of_claims = len(claim_text_number_of_sentences_in_categorzation.keys())
        ind = np.arange(num_of_claims)  # the x locations for the groups
        width = 0.15       # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, rel_percentage_sorted, width, color='b')
        rects2 = ax.bar(ind+width, not_rel_percentage_sorted, width, color='r')
        ax.set_ylabel('percentage',fontsize=15)
        ax.set_title('relevant and not relevant percentage across all claims',fontsize=15)
        ax.set_xticks(ind+width)
        ax.set_xticklabels(claims_text_sorted)
        ax.legend( (rects1[0], rects2[0]), ('relevant','not relevant') )
    
        def autolabel(rects):
        # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.3f'%float(height),
                        ha='left', va='bottom',fontsize=10,weight='normal')
        
        autolabel(rects1)
        autolabel(rects2)
#         plt.show()
#         savefig('rel_not_rel_per_claim.png')
        
        #other categories
        contra_data = zip(*categories_data_list)[2]
        #percentage out of the relevant
        contra_data_percent = [float(float(contra_data[i])/float(rel_data[i])) for i in range(0,len(rel_data))]
        neutral_data=zip(*categories_data_list)[3]
        neutral_data_percent = [float(float(neutral_data[i])/float(rel_data[i])) for i in range(0,len(rel_data))]
        supp_data=zip(*categories_data_list)[4]
        supp_data_percent = [float(float(supp_data[i])/float(rel_data[i])) for i in range(0,len(rel_data))]
        undecided_data=zip(*categories_data_list)[5]
        undecided_data_percent = [float(float(undecided_data[i])/float(rel_data[i])) for i in range(0,len(rel_data))]
        #percentage out of all the 60 claims
#         contra_data_percent = [float(float(contra_data[i])/float(60)) for i in range(0,len(rel_data))]
#         neutral_data=zip(*categories_data_list)[3]
#         neutral_data_percent = [float(float(neutral_data[i])/float(60)) for i in range(0,len(rel_data))]
#         supp_data=zip(*categories_data_list)[4]
#         supp_data_percent = [float(float(supp_data[i])/float(60)) for i in range(0,len(rel_data))]
#         undecided_data=zip(*categories_data_list)[5]
#         undecided_data_percent = [float(float(undecided_data[i])/float(60)) for i in range(0,len(rel_data))]
            
        # sort according to supportivness
        categories_and_claim_zipped=zip(supp_data_percent,neutral_data_percent,contra_data_percent,undecided_data_percent,claims_text_n)
        categories_and_claim_zipped_sorted=sorted(categories_and_claim_zipped)
#         supp_data_percent_sorted,neutral_data_percent_sorted,contra_data_percent_sorted,undecided_data_percent_sorted,not_rel_percentage_sorted,claims_text_sorted_supp = zip(*categories_and_claim_zipped_sorted)
        supp_data_percent_sorted,neutral_data_percent_sorted,contra_data_percent_sorted,undecided_data_percent_sorted,claims_text_sorted_supp = zip(*categories_and_claim_zipped_sorted)  
  
        
        fig = pl.figure(figsize=(100, 100))
        #order of categories: not relevant, relevant,contra, neutra, supp, indecided 
        num_of_claims = len(claim_text_number_of_sentences_in_categorzation.keys())
        ind = np.arange(num_of_claims)  # the x locations for the groups
        fig, ax = plt.subplots()
        rects3 = ax.bar(ind, supp_data_percent_sorted, width, color='b')
        rects4 = ax.bar(ind+width, neutral_data_percent_sorted, width, color='g')
        rects5 = ax.bar(ind+2*width, contra_data_percent_sorted, width, color='r')
        rects6 = ax.bar(ind+3*width, undecided_data_percent_sorted, width, color='y')
#         rects7 = ax.bar(ind+4*width, not_rel_percentage_sorted, width, color='m')
#         plt.rc('xtick', labelsize=10) 
        # add some
        ax.set_ylabel('percentage',fontsize=15)
        ax.set_title('support categories percentage',fontsize=15)
        ax.set_xticks(ind+width)
        ax.set_xticklabels(claims_text_sorted_supp)
        ax.legend( (rects3[0],rects4[0],rects5[0],rects6[0]), ('support','neutral','contradict','undecided') )
        
        autolabel(rects3)
        autolabel(rects4)
        autolabel(rects5)
        autolabel(rects6)
#         autolabel(rects7)
        plt.show()
        plt.close()
#         savefig('supp_contra_neu_per_claim.png')
    
    def plot_averaged_categories_bar_chart(self,not_rel_avg,relevant_avg,support_avg,contradict_avg,neutral_avg,undecided_avg,std_list):
        
        not_rel_majority_dict=utils.read_pickle("not_rel_majority_dict")
        relevant_majority_dict=utils.read_pickle("relevant_majority_dict")
        support_majority_dict=utils.read_pickle("support_majority_dict")
        contra_majority_dict=utils.read_pickle("contra_majority_dict")
        neutral_majority_dict=utils.read_pickle("neutral_majority_dict")
        undecided_majority_dict=utils.read_pickle("undecided_majority_dict")
        
        fig = pl.figure()
#         not_rel_data=(len(not_rel_majority_dict),not_rel_avg,std_list[0])
        not_rel_data=(not_rel_avg,std_list[0])
        rel_data=(relevant_avg,std_list[1])
#         rel_data=(len(relevant_majority_dict),relevant_avg,std_list[1])
        N = len(not_rel_data)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.20       # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, not_rel_data, width, color='r')
        rects2 = ax.bar(ind+width, rel_data, width, color='b')
        # add some
        ax.set_ylabel('frequency')
        ax.set_title('total count, avg and std')
        ax.set_xticks(ind+width)
        ax.set_xticklabels( ('total count', 'avg', 'std') )
        
        ax.legend( (rects1[0], rects2[0]), ('not relevant', 'relevant') )
        def autolabel(rects):
        # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%float(height),
                        ha='left', va='bottom')
        autolabel(rects1)
        autolabel(rects2)
#             plt.show()
        savefig('rel_not_rel_bars.png')
        
        fig = pl.figure()
        support_data=(len(support_majority_dict),support_avg,std_list[4])
        contradict_data=(len(contra_majority_dict),contradict_avg,std_list[2])
        neutral_data=(len(neutral_majority_dict),neutral_avg,std_list[3])
        undecided_data=(len(undecided_majority_dict),undecided_avg,std_list[5])
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, support_data, width, color='g')
        rects2 = ax.bar(ind+width, neutral_data, width, color='y')
        rects3 = ax.bar(ind+2*width, contradict_data, width, color='r')
        rects4 = ax.bar(ind+3*width, undecided_data, width, color='b')
        # add some
        ax.set_ylabel('frequency')
        ax.set_title('total count, avg and std')
        ax.set_xticks(ind+width)
        ax.set_xticklabels( ('total count', 'avg', 'std') )
        
        ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0]), ('support', 'neutral','contradict','undecided') )
   
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        autolabel(rects4)
#             plt.show()
        savefig('supp')
            
    def check_categories_correlation_with_extenal_info(self):
        """
        per claim, check if the number of supporting sentences, contradict etc... has a correlation with extrenal data:
        the external data is given in a file and includes: the number of Search Engine results (google) of the claim,
        the number of users on IMDB that voted for the movie's title.
        This is in order to try and classify the claims to tough ones and easy ones, 
        """            
        external_data_file=r"C:\study\technion\MSc\Thesis\Y!\support_test\categorization_analysis_on_test_results\SE_results_IMDB_num_raters.txt"
        claim_text_number_of_sentences_in_categorzation=utils.read_pickle("claim_text_number_of_sentences_in_categorzation")
        #order of categories: not relevant, relevant,contra, neutra, supp, indecided   
        claim_text_and_external_data_dict={} #key is claim text and value is a tuple (SE results, IMDB raters)
        
        extrernal_f = open (external_data_file,'r').read().strip()
        for i, line in enumerate(extrernal_f.split('\n')):
            claim_text=line.split("|")[0]
            num_SE_result=int(line.split("|")[1])
            num_IMDB_raters=int(line.split("|")[2])
            claim_text_and_external_data_dict[claim_text]=(num_SE_result,num_IMDB_raters)
        #evaluate pearson correlation betwenen the claim num of setntence for each category, with the two extrenal data:
        #compare with number of IMDB raters, according to the following formula: (x_i-X_bar)*(y_i-Y_bar) when x_i is the current claim number of supporting sentenes (for instance),
        #and X_bar is the average across all claims, y_i is the current claims number of raters and Ybar is the average
        [not_rel,rel,contra,neu,supp,undecide]=zip(*claim_text_number_of_sentences_in_categorzation.values())
        [SE_res,IMDB_raters]=zip(*claim_text_and_external_data_dict.values())
        not_rel_category_IMDB_raters_pearson=scipy.stats.pearsonr(not_rel,IMDB_raters)
        rel_category_IMDB_raters_pearson=scipy.stats.pearsonr(rel,IMDB_raters)
        contra_category_IMDB_raters_pearson=scipy.stats.pearsonr(contra,IMDB_raters)
        neu_category_IMDB_raters_pearson=scipy.stats.pearsonr(neu,IMDB_raters)
        supp_category_IMDB_raters_pearson=scipy.stats.pearsonr(supp,IMDB_raters)
        undecide_category_IMDB_raters_pearson=scipy.stats.pearsonr(undecide,IMDB_raters)
        
        print "not_rel_category_IMDB_raters_pearson: "+str(not_rel_category_IMDB_raters_pearson)
        print "rel_category_IMDB_raters_pearson: "+str(rel_category_IMDB_raters_pearson)
        print "contra_category_IMDB_raters_pearson: "+str(contra_category_IMDB_raters_pearson)
        print "neu_category_IMDB_raters_pearson: "+str(neu_category_IMDB_raters_pearson)
        print "supp_category_IMDB_raters_pearson: "+str(supp_category_IMDB_raters_pearson)
        print "undecide_category_IMDB_raters_pearson: "+str(undecide_category_IMDB_raters_pearson)
        
        not_rel_category_SE_results_pearson=scipy.stats.pearsonr(not_rel,SE_res)
        rel_category_SE_results_pearson=scipy.stats.pearsonr(rel,SE_res)
        contra_category_SE_results_pearson=scipy.stats.pearsonr(contra,SE_res)
        neu_category_SE_results_pearson=scipy.stats.pearsonr(neu,SE_res)
        supp_category_SE_results_pearson=scipy.stats.pearsonr(supp,SE_res)
        undecide_category_SE_results_pearson=scipy.stats.pearsonr(undecide,SE_res)
        
        print "not_rel_category_SE_results_pearson: "+str(not_rel_category_SE_results_pearson)
        print "rel_category_SE_results_pearson: "+str(rel_category_SE_results_pearson)
        print "contra_category_SE_results_pearson: "+str(contra_category_SE_results_pearson)
        print "neu_category_SE_results_pearson: "+str(neu_category_SE_results_pearson)
        print "supp_category_SE_results_pearson: "+str(supp_category_SE_results_pearson)
        print "undecide_category_SE_results_pearson: "+str(undecide_category_SE_results_pearson)
        
        #check spearman correlation for each category, with two ranks : SE resulus, and num of raters results
        # 
                
   
    def divide_sen_to_sets_sentiment_sim(self):
        "based on my annotations or on Socher's tool"
        sentiment_annotation="socher"
        claim_num=4
        
        if sentiment_annotation is "myself":
            my_sentiment= open(r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\sanity_check_sen_sim\claim_"+str(claim_num)+".txt",'r').read().strip() 
            for i, line in enumerate(my_sentiment.split('\n')):
                sen=line.split("|")[0]
                sen_sim_indicator=line.split("|")[1]
                if sen_sim_indicator =='1':
                    self.same_senti_set.add(sen)
                else: #not same sentiment
                    self.diff_senti_set.add(sen)
        elif sentiment_annotation is "socher":
            self.read_pickle("claim_sen_sentiment_simialrity_socher")
            for ((claim,sen),sen_sim) in self.claim_sen_sentiment_cos_simialrity_socher.items():
#             if sen_sim[0] == 0 or sen_sim[0] == 1:
#                 sen_sim_pairs_based_on_label_set.add((claim,sen))
                if sen_sim[1] >0.75:
                    self.same_senti_set.add((claim,sen))
                else:
                    self.diff_senti_set.add((claim,sen))
    
    def divide_to_set_by_clm_sen_sim(self,representation,composition_func,dim):
        "based on results of claim_sentence_similarity_analysis_VSM"

        self.read_pickle("all_clm_sen_cosine_sim_res_"+representation+"_"+composition_func+"_"+str(dim))
        for ((claim,sen),sim) in self.clm_sen_cosine_sim_res.items():
#             if sen_sim[0] == 0 or sen_sim[0] == 1:
#                 sen_sim_pairs_based_on_label_set.add((claim,sen))
            if sim >0.70:
                self.similar_claim_sen_set.add((claim,sen))
            else:
                self.diff_claim_sen_set.add((claim,sen))
            print ("") 
             
                        
    def calc_agreement_on_sentiment(self):
        same_senti_and_relevant_pairs=set()
        save_to_file=0
        same_senti_and_relevant_pairs=self.same_senti_set.intersection(self.relevant_sen_set)
        diff_senti_and_relevant_pairs=self.diff_senti_set.intersection(self.relevant_sen_set)
        agreed_supp_sen_sim=self.supp_set.intersection(self.same_senti_set)
#         agreed_supp_sen_sim=self.supp_set.intersection(same_senti_and_relevant_pairs)
        agreed_contra_diff_sen=self.contradict_set.intersection(self.diff_senti_set)
#         agreed_contra_diff_sen=self.contradict_set.intersection(diff_senti_and_relevant_pairs)
        agreed=agreed_supp_sen_sim.union(agreed_contra_diff_sen)
        
        disagreed_contra_sen_sim=self.contradict_set.intersection(self.same_senti_set)
#         disagreed_contra_sen_sim=self.contradict_set.intersection(same_senti_and_relevant_pairs)
        disagreed_supp_diff_sen=self.supp_set.intersection(self.diff_senti_set)
#         disagreed_supp_diff_sen=self.supp_set.intersection(diff_senti_and_relevant_pairs)
        disagreed=disagreed_contra_sen_sim.union(disagreed_supp_diff_sen)
        
        accuracy=float(float(len(agreed))/float(len(agreed)+len(disagreed)))
#         print accuracy
        accuracy_sen_sim_supp=float(len(agreed_supp_sen_sim))/float((len(disagreed)+len(agreed)))
        accuracy_diff_sen_contra=float(len(agreed_contra_diff_sen))/float((len(agreed)+len(disagreed)))
        print ('accuracy: '+str(accuracy)+ " accuracy_sen_sim_supp: "+str(accuracy_sen_sim_supp) +" accuracy_diff_sen_contra: "+str(accuracy_diff_sen_contra) )
        
        miss=float(float(len(disagreed))/float(len(agreed)+len(disagreed)))
        miss_disagreed_contra_sen_sim=float(float(len(disagreed_contra_sen_sim))/float(len(disagreed)))
        miss_disagreed_supp_diff_sen=float(float(len(disagreed_supp_diff_sen))/float(len(disagreed)))
        print ('miss : '+str(miss)+ " miss_miss_disagreed_contra_sen_sim: "+str(miss_disagreed_contra_sen_sim) +" miss_miss_disagreed_supp_diff_sen: "+str(miss_disagreed_supp_diff_sen) )
        
        #precision and recall
        recall_based_on_cosine=float(len(agreed_supp_sen_sim))/float(len(self.supp_set))
        precision_based_on_cosine=float(len(agreed_supp_sen_sim))/float(len(self.same_senti_set))
        print "precision_based_on_cosine "+ str(precision_based_on_cosine)
        print "recall_based_on_cosine "+ str(recall_based_on_cosine)
        print "F1 measure using cosine" +str(float(2*(precision_based_on_cosine*recall_based_on_cosine)/float(precision_based_on_cosine+recall_based_on_cosine)))
            
        
        
        sets_list=[(agreed_supp_sen_sim,"agreed_supp_sen_sim"),(agreed_contra_diff_sen,"agreed_contra_diff_sen"),
                   (disagreed_contra_sen_sim,"disagreed_contra_sen_sim"),(disagreed_supp_diff_sen,"disagreed_supp_diff_sen")]
        for (s,set_name) in sets_list:
            utils.save_pickle(set_name, s)
        
        relevant_dict = dict.fromkeys(self.relevant_sen_set,0)
        relevant_dict_sorted = collections.OrderedDict(sorted(relevant_dict.items(),key=lambda x: (str(x[0][0])), reverse=True))     
        agreed_dict = dict.fromkeys(agreed, 0)
        agreed_supp_sen_sim_dict=dict.fromkeys(agreed_supp_sen_sim,0)
        agreed_contra_diff_sen_dict =  dict.fromkeys(agreed_contra_diff_sen,0)
        disagreed_dict=dict.fromkeys(disagreed,0)
        disagreed_contra_sen_sim_dict=dict.fromkeys(disagreed_contra_sen_sim,0)
        disagreed_supp_diff_sen_dict=dict.fromkeys(disagreed_supp_diff_sen,0)
        
        dict_list=[(agreed_dict,"total_supp_sen_sim_agreed_dict"),(agreed_supp_sen_sim_dict,"agreed_supp_sen_sim_dict"),
                       (agreed_contra_diff_sen_dict,"agreed_contra_diff_sen_dict"),(disagreed_dict,"disagreed_sen_sim_dict"),
                       (disagreed_contra_sen_sim_dict,"disagreed_contra_sen_sim_dict"),(disagreed_supp_diff_sen_dict,"disagreed_supp_diff_sen_dict"),
                       ]
        for (d,d_name) in dict_list:
            new_dict={}
            new_dict=self.add_doc_title_to_clm_sen_pair(d,d_name)           
            self.save_to_csv_file(new_dict,d_name+'.csv') 
            
            #avg of sentences per claim that is in the curr dict
            
            #for specific claim
#         self.save_to_csv_file(agreed_dict, 'agreed_dict_'+str(claim_num)+'.csv')
#         self.save_to_csv_file(agreed_supp_sen_sim_dict, 'agreed_supp_sen_sim_dict_'+str(claim_num)+'.csv')
#         self.save_to_csv_file(agreed_contra_diff_sen_dict, 'agreed_contra_diff_sen_dict_'+str(claim_num)+'.csv')
#         self.save_to_csv_file(disagreed_dict, 'disagreed'+str(claim_num)+'.csv')
#         self.save_to_csv_file(disagreed_contra_sen_sim_dict, 'disagreed_contra_sen_sim_'+str(claim_num)+'.csv')
#         self.save_to_csv_file(disagreed_supp_diff_sen_dict, 'disagreed_supp_diff_sen_'+str(claim_num)+'.csv')
#         self.save_to_csv_file(neutral_sen_dict, 'neutral_sen_dict_'+str(claim_num)+'.csv')
#         self.save_to_csv_file(sen_sim_but_not_supp_dict, 'sen_sim_but_not_supp_dict'+str(claim_num)+'.csv') 
        
    
    def calc_agreement_on_clm_sen_similarity(self,representation,comp_func,dim):
        """
        using the claim and sentence similarity, by cosine between the VSM vectors of each claim and sentence (claim_sentence_similarity_analysis_VSM)
        calculate the accuracy between this and supportivness of each pair.
        """  
        try:              
            agreed_supp_clm_sen_sim=self.supp_set.intersection(self.similar_claim_sen_set)
            agreed_contra_clm_sen_diff=self.contradict_set.intersection(self.diff_claim_sen_set)
            agreed=agreed_supp_clm_sen_sim.union(agreed_contra_clm_sen_diff)
            
            disagreed_contra_clm_sen_sim=self.contradict_set.intersection(self.similar_claim_sen_set)
            disagreed_supp_clm_sen_diff=self.supp_set.intersection(self.diff_claim_sen_set)
            disagreed=disagreed_contra_clm_sen_sim.union(disagreed_supp_clm_sen_diff)
            
            accuracy=float(float(len(agreed))/float(len(agreed)+len(disagreed)))
            accuracy_clm_sen_sim_supp=float(len(agreed_supp_clm_sen_sim))/float(len(agreed))
            accuracy_clm_sen_diff_contra=float(len(agreed_contra_clm_sen_diff))/float(len(agreed))
            
            miss=float(float(len(disagreed))/float(len(agreed)+len(disagreed)))
            miss_disagreed_contra_clm_sen_sim=float(float(len(disagreed_contra_clm_sen_sim))/float(len(disagreed)))
            miss_disagreed_supp_clm_sen_diff=float(float(len(disagreed_supp_clm_sen_diff))/float(len(disagreed)))
            
            sets_list=[(agreed_supp_clm_sen_sim,"agreed_supp_clm_sen_sim"),(agreed_contra_clm_sen_diff,"agreed_contra_clm_sen_diff"),
                   (disagreed_contra_clm_sen_sim,"disagreed_contra_clm_sen_sim"),(disagreed_supp_clm_sen_diff,"disagreed_supp_clm_sen_diff")]
            for (s,set_name) in sets_list:
                utils.save_pickle(set_name, s)
            
            print ('accuracy: '+str(accuracy)+ " accuracy_sen_sim_supp: "+str(accuracy_clm_sen_sim_supp) +" accuracy_diff_sen_contra: "+str(accuracy_clm_sen_diff_contra) )
            print ('miss ' +str(miss) +" miss_disagreed_contra_clm_sen_sim: "+ str(miss_disagreed_contra_clm_sen_sim)+ " miss_disagreed_supp_clm_sen_diff: "+str(miss_disagreed_supp_clm_sen_diff))
            
            agreed_dict = dict.fromkeys(agreed, 0)
            agreed_supp_clm_sen_sim_dict=dict.fromkeys(agreed_supp_clm_sen_sim,0)
            agreed_contra_clm_sen_diff_dict=dict.fromkeys(agreed_contra_clm_sen_diff,0)
            disagreed_dict=dict.fromkeys(disagreed,0)
            disagreed_contra_clm_sen_sim_dict=dict.fromkeys(disagreed_contra_clm_sen_sim,0)
            disagreed_supp_clm_sen_diff_dict=dict.fromkeys(disagreed_supp_clm_sen_diff,0)
#             clm_sen_sim_but_not_supp_dict=dict.fromkeys(self.similar_claim_sen_set-self.supp_set,0)
#             clm_sen_sim_but_not_supp_dict =collections.OrderedDict(sorted(clm_sen_sim_but_not_supp_dict.items(),key=lambda x: (x[0][0]), reverse=True))  
#             supp_clm_sen_diff_dict=dict.fromkeys(self.supp_set.intersection(self.diff_claim_sen_set),0)
#             supp_clm_sen_diff_dict =collections.OrderedDict(sorted(supp_clm_sen_diff_dict.items(),key=lambda x: (x[0][0]), reverse=True))  
                        
            recall_based_on_cosine=float(len(agreed_supp_clm_sen_sim))/float(len(self.supp_set))
            precision_based_on_cosine=float(len(agreed_supp_clm_sen_sim))/float(len(self.similar_claim_sen_set))
            print "precision_based_on_cosine "+ str(precision_based_on_cosine)
            print "recall_based_on_cosine "+ str(recall_based_on_cosine)
            print "F1 measure using cosine" +str(float(2*(precision_based_on_cosine*recall_based_on_cosine)/float(precision_based_on_cosine+recall_based_on_cosine)))
            
            dict_list=[(agreed_dict,"total_supp_clm_sen_sim_agreed_dict"),(agreed_supp_clm_sen_sim_dict,"agreed_supp_clm_sen_sim_dict"),
                       (agreed_contra_clm_sen_diff_dict,"agreed_contra_clm_sen_diff_dict"),(disagreed_dict,"disagreed_clm_sen_sim_dict"),
                       (disagreed_contra_clm_sen_sim_dict,"disagreed_contra_clm_sen_sim_dict"),(disagreed_supp_clm_sen_diff_dict,"disagreed_supp_clm_sen_diff_dict"),
                       ]
            for (d,d_name) in dict_list:
                new_dict={}
                new_dict=self.add_doc_title_to_clm_sen_pair(d,d_name)
                self.save_to_csv_file(new_dict,d_name+'.csv')
                utils.save_pickle(d_name, d)
        
        except Exception as err: 
                sys.stderr.write('problem in calc_agreement_on_clm_sen_similarity:')     
                print err.args      
                print err
        
    def calc_supp_sen_sim_per_claim(self):
        try:
            """
            I define precision as ||supp AND sen_sim/sen_sim||- calculate per claims, and then avg
            """
            self.claim_sen_sentiment_cos_simialrity_socher=utils.read_pickle("claim_sen_sentiment_simialrity_socher")
            self.relevant_majority_dict=utils.read_pickle("relevant_majority_dict")
            self.support_majority_dict=utils.read_pickle("support_majority_dict")
            self.contra_majority_dict=utils.read_pickle("contra_majority_dict")
            self.neutral_majority_dict=utils.read_pickle("neutral_majority_dict")
            self.undecided_majority_dict=utils.read_pickle("undecided_majority_dict")
            no_agreed_claims=[]
            total_accuracy=0
            files_num=0
            #go over the files to get the claims:
            for filename in os.listdir(self.support_annotated_files_path):
                if filename.split("_")[0]=="f" :
                    files_num+=1
                    curr_supp_pairs_set=set()
                    curr_contra_pairs_set=set()
                    curr_agreed_supp_sen_sim_pairs=set()
                    curr_agreed_contra_sen_diff_pairs=set()
                    curr_disagreed_supp_sen_diff_pairs=set()
                    curr_disagreed_contra_sen_sim_pairs=set()
                    accuracy=0
                    claim_num=filename.split("_")[1]
#                     if claim_num == "70.csv":
                    with open(self.support_annotated_files_path+"\\"+filename, 'r') as f:
                        data = pd.read_csv(f)
                        claim_text = data['claim'][1]
                    #go over the dicts to get pairs that contain this claim
                        for (claim_key,sen_key) in self.support_majority_dict.keys():
                            if claim_key == claim_text:
                                curr_supp_pairs_set.add((claim_key,sen_key))
                        for (claim_key,sen_key) in self.contra_majority_dict.keys():
                            if claim_key == claim_text:
                                curr_contra_pairs_set.add((claim_key,sen_key))
                        #calc accuracy
                        curr_agreed_supp_sen_sim_pairs=self.same_senti_set.intersection(curr_supp_pairs_set)
                        curr_agreed_contra_sen_diff_pairs=self.diff_senti_set.intersection(curr_contra_pairs_set)
                        agreed=curr_agreed_supp_sen_sim_pairs.union(curr_agreed_contra_sen_diff_pairs)
                        curr_disagreed_contra_sen_sim_pairs=self.same_senti_set.intersection(curr_contra_pairs_set)
                        curr_disagreed_supp_sen_diff_pairs=self.diff_senti_set.intersection(curr_supp_pairs_set)
                        disagreed=curr_disagreed_contra_sen_sim_pairs.union(curr_disagreed_supp_sen_diff_pairs)
                        
                        if len(agreed) ==0 :
                            no_agreed_claims.append(claim_text)
                            continue  
                        accuracy=float(float(len(agreed))/float((len(agreed)+len(disagreed))))
                        total_accuracy+=accuracy
                        accuracy_supp_sen_sim=float(float(len(curr_agreed_supp_sen_sim_pairs))/float((len(agreed)+len(disagreed))))
                        accuracy_contra_sen_diff=float(float(len(curr_agreed_contra_sen_diff_pairs))/float((len(agreed)+len(disagreed))))
                        print ("accuracy for claim num: " +str(claim_num)+ ' : "'  +claim_text +'" : '+str(accuracy) +'" and accuracy_supp_sen_sim: '
                               +str(accuracy_supp_sen_sim) +" accuracy_contra_sen_diff: " +str(accuracy_contra_sen_diff))
                        
            print ("no agreed pairs for claims: "+','.join(no_agreed_claims))
            avg_accuracy=float(total_accuracy/float(files_num))
            print ("avg_accuracy: "+str(avg_accuracy))
                                   
        except Exception as err: 
            sys.stderr.write('problem in calc_supp_sen_sim_per_claim:')     
            print err.args      
            print err
    
    def save_to_csv_file(self,dict,file_name):      #save to file
        with open(file_name, 'wb') as csvfile:
            w = csv.writer(csvfile)
            for sen in dict.keys():
                w.writerow([sen])

    def read_pickle(self,file_name):
        if file_name is 'shallow_pool_dict_wiki':
            with open(file_name, 'rb') as handle:
                    self.shallow_pool_dict_wiki = pickle.loads(handle.read())
        if file_name is 'shallow_pool_dict_RT':
            with open(file_name, 'rb') as handle:
                    self.shallow_pool_dict_RT = pickle.loads(handle.read())
        if file_name is "claim_sen_sentiment_simialrity_socher":
            with open(file_name, 'rb') as handle:
                    self.claim_sen_sentiment_cos_simialrity_socher = pickle.loads(handle.read())
        if "all_clm_sen_cosine_sim_res_" in file_name:
            with open(file_name, 'rb') as handle:
                self.clm_sen_cosine_sim_res= pickle.loads(handle.read())
    
    def find_doc_title(self):
        try:
            input_files_path=r"C:\study\technion\MSc\Thesis\Y!\support_test\input_crowdflower_second_trial"
            for filename in os.listdir(input_files_path):  
                claim_num=filename.split("_")[2].split(".")[0]
                with open(input_files_path+"\\"+filename, 'r') as f:
                    data = pd.read_csv(f)
                    sentence=data['sen']
                    claim_text = data['claim'][1]
                    doc_title=data['tit']
                    is_gold=data['_golden']    
                    for sen_num in range(0,len(sentence)):
                        if is_gold[sen_num] !=1:
                            self.clm_sen_doc_title_dict[(claim_text,sentence[sen_num])]=doc_title[sen_num]
             
            utils.save_pickle("clm_sen_doc_title_dict",self.clm_sen_doc_title_dict)               
                            
        except Exception as err: 
                sys.stderr.write('problem in find_doc_title:')     
                print err.args      
                print err
    
    def add_doc_title_to_clm_sen_pair(self,d,d_name):
        """
        add the doc title to the dict of claim and sentence pair in the given dict
        """    
        new_dict={}
        self.claim_dict={}
        for ((clm,sen)) in d.keys():
            try:
                new_dict[(clm,sen,self.clm_sen_doc_title_dict[(clm,sen)])]=0
                if clm in self.claim_dict.keys():
                    self.claim_dict[clm]+=1
                else:
                    self.claim_dict[clm]=1
            except Exception as err: 
                    sys.stderr.write('problem in add_doc_title_to_clm_sen_pair in clm and sen pair :'+clm +" "+sen)     
                    print err.args      
                    print err
        new_dict=collections.OrderedDict(sorted(new_dict.items(),key=lambda x: (x[0][0]), reverse=True))
        avg_sen_per_claim=float(sum(self.claim_dict.values())/len(self.claim_dict.keys()))
        print ("in "+d_name+" avg_sen_per_claim: "+str(avg_sen_per_claim))
        return new_dict
    
    def compare_features_result(self,representation,dim,composition_func):
        """
        analyse which sentences are "found" by different features:
        for instance - supportive and sentiment diff, but supportive and VSM similar...
        and also the inetersection between them (to know which one rules the other)
        """
        try:
            agreed_supp_clm_sen_sim_set=set()
            agreed_contra_clm_sen_diff_set=set()
            disagreed_contra_clm_sen_sim_set=set()
            disagreed_supp_clm_sen_diff_set=set()
            agreed_supp_sen_sim_set=set()
            agreed_contra_diff_sen_set=set()
            disagreed_contra_sen_sim_set=set()
            disagreed_supp_diff_sen_set=set()
            
            agreed_supp_clm_sen_sim_set=utils.read_pickle_set("agreed_supp_clm_sen_sim")
            agreed_contra_clm_sen_diff_set=utils.read_pickle_set("agreed_contra_clm_sen_diff")
            disagreed_contra_clm_sen_sim_set=utils.read_pickle_set("disagreed_contra_clm_sen_sim")
            disagreed_supp_clm_sen_diff_set=utils.read_pickle_set("disagreed_supp_clm_sen_diff")
            agreed_supp_sen_sim_set=utils.read_pickle_set("agreed_supp_sen_sim")
            agreed_contra_diff_sen_set=utils.read_pickle_set("agreed_contra_diff_sen")
            disagreed_contra_sen_sim_set=utils.read_pickle_set("disagreed_contra_sen_sim")
            disagreed_supp_diff_sen_set=utils.read_pickle_set("disagreed_supp_diff_sen")
            #calc the intersections....
            VSM_found_sen_sim_not=disagreed_supp_diff_sen_set.intersection(agreed_supp_clm_sen_sim_set)
            VSM_found_sen_sim_not_dict=dict.fromkeys(VSM_found_sen_sim_not,0)
            
            sen_sim_found_VSM_not=disagreed_supp_clm_sen_diff_set.intersection(agreed_supp_sen_sim_set)
            sen_sim_found_VSM_not_dict=dict.fromkeys(sen_sim_found_VSM_not,0)
            
            supp_and_found_by_VSM_and_sen_sim=agreed_supp_sen_sim_set.intersection(agreed_supp_clm_sen_sim_set)
            supp_and_found_by_VSM_and_sen_sim_dict=dict.fromkeys(supp_and_found_by_VSM_and_sen_sim,0)
            
            
            
            comp_dict_list=[(sen_sim_found_VSM_not_dict,"sen_sim_found_VSM_not_dict"),
                            (VSM_found_sen_sim_not_dict,"VSM_found_sen_sim_not_dict"),
                            (supp_and_found_by_VSM_and_sen_sim_dict,"supp_and_found_by_VSM_and_sen_sim_dict")]
            for (d,d_name) in comp_dict_list:
                new_dict=self.add_doc_title_to_clm_sen_pair(d, d_name)
                self.save_to_csv_file(new_dict, d_name+"_"+representation+"_"+composition_func+"_"+str(dim)+".csv")
            
            
        except Exception as err: 
            sys.stderr.write('problem in compare_features_result:')     
            print err.args      
            print err    
            
    def calc_ranking_measures(self,analysis_on_feature,representation,composition_func,dim):
#         clm_sen_support_ranking=utils_linux.read_pickle("clm_sen_support_ranking_sorted") #grouped by claim and sorted from most supp to least
        clm_sen_support_ranking=utils.read_pickle("clm_sen_support_ranking_sorted_full") #for taking all the 60 claims, and not just the ones that are relevant by the annotators
        if analysis_on_feature is "sentiment_similarity":
            current_feature_dict_score=utils.read_pickle("claim_sen_sentiment_simialrity_socher_sorted") #grouped by claim and sorted from most similar to least
        elif analysis_on_feature is "clm_sen_similarity_VSM":
            current_feature_dict_score=utils.read_pickle("claim_sen_VSM_similarity_sorted_"+representation+"_"+composition_func+"_"+str(dim))
        elif analysis_on_feature is "entailment":
            current_feature_dict_score=utils.read_pickle("entailment_clm_sen_pair_sorted")
        clm_sen_support_set=set()
        clm_sen_feature_set=set()
        clm_as_key_sen_feature_score_val={} #key is a claim, val is a list of (sen,sentiment_score) tuple
        clm_as_key_sen_support_score_val={} #the same only with supp score
        NDCG_all_claims={} #key is claim, val is NDCG
        NDCG_all_claims_randomized_rank={} # same as above only for a randomized ranking
        
        # in the sentiment sorted list, I need only the list of the same sentences per claim! meaning sentences that are supp, contra, neutral etc.
        #meaning I need to create a sorted list of the sentences that are in the current claim support ordered list, and this list re-rank by similarity.
        for (clm,sen,score) in clm_sen_support_ranking.keys():
            clm_sen_support_set.add((clm,sen))
        for ((clm,sen),sim) in current_feature_dict_score.items():
            clm_sen_feature_set.add((clm,sen))
        #the intersection between the two sets:
        supportivness_sen_sim_intersection=clm_sen_support_set.intersection(clm_sen_feature_set)
        #go over the sentiment ranking dict and leave only those in the intersectiom
        for ((clm,sen),sim) in current_feature_dict_score.items():
            if not (clm,sen) in supportivness_sen_sim_intersection:
                del current_feature_dict_score[(clm,sen)]
        #now we are left with the semtiment ranking dict that contains only sentences that are supp, contra, neu...
        #create a dict of key is the claim, value if a sorted(!!!) list of tuple (sentence and supp score or sentiment score)
        for ((clm,sen),sim) in current_feature_dict_score.items():
            if clm in clm_as_key_sen_feature_score_val.keys():
                clm_as_key_sen_feature_score_val[clm].append((sen,sim))
            else:
                clm_as_key_sen_feature_score_val[clm]=[(sen,sim)]
        
        for (clm,sen,score) in clm_sen_support_ranking.keys():
            if clm in clm_as_key_sen_support_score_val.keys():
                clm_as_key_sen_support_score_val[clm].append((sen,score))
            else:
                clm_as_key_sen_support_score_val[clm]=[(sen,score)]
        utils.save_pickle("clm_as_key_sorted_sen_support_score_val", clm_as_key_sen_support_score_val)
        #sort them again by the scores...
#         clm_as_key_sen_support_score_vals_sorted=collections.OrderedDict(sorted(clm_as_key_sen_support_score_val.items(),key=lambda x: (str(x[1][1])), reverse=True))
        
        p=10
         
        for clm in clm_as_key_sen_feature_score_val.keys():
            if len(clm_as_key_sen_feature_score_val[clm]) >= p: #if there is 1 relevant sentence but p is higher...
                NDCG_all_claims[clm]=utils.calc_NDCG(clm_as_key_sen_feature_score_val[clm],clm_as_key_sen_support_score_val[clm],p)
                random.shuffle(clm_as_key_sen_feature_score_val[clm])
                NDCG_all_claims_randomized_rank[clm]=utils.calc_NDCG(clm_as_key_sen_feature_score_val[clm],clm_as_key_sen_support_score_val[clm],p)
            else:
                continue
        # rank the claims acording to the highest NDCG- as much as the NDCG is closet to 1- it is ideal..!
        all_claims_sorted_by_NDCG_feature=collections.OrderedDict(sorted(NDCG_all_claims.items(),key=lambda x: (float(x[1])), reverse=True))
        all_claims_sorted_by_NDCG_sentimet_random=collections.OrderedDict(sorted(NDCG_all_claims_randomized_rank.items(),key=lambda x: (float(x[1])), reverse=True))
        average_NDCG=float(float(sum(NDCG_all_claims.values()))/float(len(NDCG_all_claims)))
        average_NDCG_random=float(float(sum(NDCG_all_claims_randomized_rank.values()))/float(len(NDCG_all_claims_randomized_rank)))
        
        print "random_nDCG@p:"+str(p)+": "+str( average_NDCG_random)
        print "nDCG@p:"+str(p)+": "+str( average_NDCG)
        #save NDCG result to file
        if analysis_on_feature is "sentiment_similarity":
            with open('NDCG_all_claims_sorted_by_NDCG_@'+str(p)+"_"+analysis_on_feature+'_'+str(p)+'.csv', 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,ndcg_score) in all_claims_sorted_by_NDCG_feature.items():
                    w.writerow([clm,ndcg_score])
                w.writerow(['average NDCG:'+str(average_NDCG)])
        elif analysis_on_feature is "clm_sen_similarity_VSM":
            with open('NDCG_all_claims_sorted_by_NDCG_@'+str(p)+"_"+analysis_on_feature+'_'+representation+'_'+composition_func+'_'+str(dim)+'_'+str(p)+'.csv', 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,ndcg_score) in all_claims_sorted_by_NDCG_feature.items():
                    w.writerow([clm,ndcg_score])
                w.writerow(['average NDCG:'+str(average_NDCG)])
        elif analysis_on_feature is "entailment":
            with open('NDCG_all_claims_sorted_by_NDCG_@'+str(p)+"_"+analysis_on_feature+'.csv', 'wb') as csvfile:
                w = csv.writer(csvfile)
                for (clm,ndcg_score) in all_claims_sorted_by_NDCG_feature.items():
                    w.writerow([clm,ndcg_score])
                w.writerow(['average NDCG:'+str(average_NDCG)])
        
        with open("sorted_sen_according_to_feature_"+analysis_on_feature+'_'+str(p)+".csv", 'wb') as csvfile:
            w = csv.writer(csvfile)
            for ((clm,sen),feature_score) in current_feature_dict_score.items():
                w.writerow([clm,sen,feature_score])
#             for (clm,sen_score_list) in clm_as_key_sen_feature_score_val.items():
#                 for (sen,feature_score) in sen_score_list:
#                     w.writerow([clm,sen])
  

        
    def calc_randomized_nDCG_on_true_support(self):
        clm_as_key_sorted_sen_support_score_val=utils.read_pickle("clm_as_key_sorted_sen_support_score_val")
        NDCG_all_claims_randomized_rank={}
        p=10
        clm_as_key_sorted_sen_support_score_val_random_res  = copy.deepcopy(clm_as_key_sorted_sen_support_score_val)
        num_iter = 100
        nDCG_per_iter =[]
        for iter in range(0,num_iter):
            for clm in clm_as_key_sorted_sen_support_score_val.keys():
                random.shuffle(clm_as_key_sorted_sen_support_score_val_random_res[clm])
                NDCG_all_claims_randomized_rank[clm]=utils.calc_NDCG(clm_as_key_sorted_sen_support_score_val_random_res[clm],clm_as_key_sorted_sen_support_score_val[clm],p)
            average_NDCG_random=float(float(sum(NDCG_all_claims_randomized_rank.values()))/float(len(NDCG_all_claims_randomized_rank)))
            nDCG_per_iter.append(average_NDCG_random)
        
        average_random_nDCG_across_iter = float(float(sum(nDCG_per_iter))/float(num_iter))
        std_nDCG_across_iter = 0
        for avg_NDCG in nDCG_per_iter:
            std_nDCG_across_iter += (avg_NDCG-average_random_nDCG_across_iter)**2
        std_nDCG_across_iter = math.sqrt(float(std_nDCG_across_iter/num_iter))
        
        print ("average random nDCG@"+str(p)+" across "+str(num_iter)+" iterations: "+str(average_random_nDCG_across_iter)+", std: "+str(std_nDCG_across_iter))   
        
def main():
        try:      
            supp_sen=supportive_sentence()
            mode="create_features"
            representation="word2vec"
            dim=300                                            
            composition_func="additive"
            supp_sen.clm_sen_doc_title_dict=utils.read_pickle("clm_sen_doc_title_dict")
#             supp_sen.calc_randomized_nDCG_on_true_support()
            
            if mode is "create_features":
                analysis_on_feature="clm_sen_similarity_VSM"
                print('########'+ analysis_on_feature+ ' #######')
#                 supp_sen.find_doc_title()

                supp_sen.divide_sen_to_sets_support()
                
                if analysis_on_feature is "sentiment_similarity":
#                     supp_sen.divide_sen_to_sets_sentiment_sim()
#                     supp_sen.calc_agreement_on_sentiment()
                    supp_sen.calc_ranking_measures(analysis_on_feature,representation,composition_func,dim)
    #                 supp_sen.calc_supp_sen_sim_per_claim()
                elif analysis_on_feature is "clm_sen_similarity_VSM":
                    print ('######## '+ representation+' , '+composition_func+' , '+str(dim)+' #######')
                    supp_sen.calc_ranking_measures(analysis_on_feature,representation,composition_func,dim)
#                     supp_sen.divide_to_set_by_clm_sen_sim(representation,composition_func,dim)
#                     supp_sen.calc_agreement_on_clm_sen_similarity(representation,composition_func,dim)
                    
            elif mode is "compare_features":
                supp_sen.compare_features_result(representation,dim,composition_func)
            
            elif mode is "calc_support_result_statistics":
                supp_sen.check_categories_correlation_with_extenal_info()
                
#             supp_sen.calc_new_support_score()
            
        except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err
        
if __name__ == '__main__':
    main() 
    