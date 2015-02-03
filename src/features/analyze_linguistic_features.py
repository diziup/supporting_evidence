'''
28/07/14
According to the categorization of the claim and sentences, to Strong Supp, Moderate...Strong Contra...
Analyze different linguistic features:
1. The distribution/statistics of different POS tags
2. the sentiment predicted - or (average) sentiment as a mean of the vector output 
@author: Liora
'''
import sys
import nltk 
from my_utils import utils_linux
import numpy as np
import csv
import collections


class linguistic_analysis():
    strong_supp_dict={} #hold the categorized clm and sen from the process_annotation_res_files module, key is clm, sen, and doc title, and val is number of annotatos
    moderate_supp_dict={}                                                       #that rated the triple as such
    neutral_dict={}
    moderate_cont_dict={}
    strong_cont_dict={}
    not_rel_dict={}    
    
    strong_supp_POS_dist_dict={} #key is clm and sen, and value is a POS (sentiment-bearing) distribution/stats
    moderate_supp_POS_dist_dict={}
    neutral_POS_dist_dict={}
    moderate_cont_POS_dist_dict={}
    strong_cont_POS_dist_dict={}
    not_rel_POS_dist_dict={}
    word_level_POS_tags=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH'
                        ,'VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','ADV',',',':',"''",'.','$','#'] #according to http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
    POS_tags_order=[]
    total_num_words_claims=0 #keep the number of words in the total claims, to normalize the POS cnt in all claims
    total_num_words_sens=0 #same only for sentences
    
    
    
    def __init__(self):
        categorized_sen_res_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\categorization_result"
    
    def POS_tag_on_clm_sen(self,d,d_name): #activate the POS tag on each dict
        try:
            
                
            for ((clm,sen,doc_title),anno_cnt) in d.items():
#                 #initialize the POS tag counter dict for the current categorized claims and sentence to zero
                curr_clm_POS_cnt_dict={}
                curr_sen_POS_cnt_dict={}
                for POS_tag in self.word_level_POS_tags:
                    curr_clm_POS_cnt_dict[POS_tag]=0
                    curr_sen_POS_cnt_dict[POS_tag]=0
                tokenized_clm=nltk.word_tokenize(clm)
                tokenized_sen=nltk.word_tokenize(sen)
                POS_tag_curr_clm=nltk.pos_tag(tokenized_clm)
                POS_tag_curr_sen=nltk.pos_tag(tokenized_sen)
                clm_words, clm_tags = zip(*POS_tag_curr_clm)
                sen_words, sen_tags = zip(*POS_tag_curr_sen)
#                 tags_set_clm=set(clm_tags) 
#                 tags_set_sen=set(sen_tags) 
                #update the POS tags counter dicts for both claim and sentence
                self.total_num_words_claims+=len(clm_words)
                self.total_num_words_sens+=len(sen_words)
                
                for pos_tag_clm in clm_tags:
                    if pos_tag_clm in self.word_level_POS_tags:
                        curr_clm_POS_cnt_dict[pos_tag_clm]+=1
                    else:
                        print pos_tag_clm +" not in POS_tag list...add!"
                for POS_tag_sen in sen_tags:
                    if POS_tag_sen in self.word_level_POS_tags:
                        curr_sen_POS_cnt_dict[POS_tag_sen]+=1
                    else:
                        print POS_tag_sen +" not in POS_tag list...add!"
                #divide by the sen length to get a distribution
#                 curr_clm_POS_cnt_dict=curr_clm_POS_cnt_dict/float(len(clm_words))
#                 curr_sen_POS_cnt_dict=curr_sen_POS_cnt_dict/float(len(sen_words))
                
                if d_name is "strong_supp":        
                    self.strong_supp_POS_dist_dict[(clm,sen)]=((np.array((curr_clm_POS_cnt_dict.values()))/float(len(clm_words)),
                                                               np.array((curr_sen_POS_cnt_dict).values())/float(len(sen_words))))
                elif d_name is "moderate_supp":
                    self.moderate_supp_POS_dist_dict[(clm,sen)]=((np.array((curr_clm_POS_cnt_dict.values()))/float(len(clm_words)),
                                                               np.array((curr_sen_POS_cnt_dict).values())/float(len(sen_words))))
                elif d_name is "neu":
                    self.neutral_POS_dist_dict[(clm,sen)]=((np.array((curr_clm_POS_cnt_dict.values()))/float(len(clm_words)),
                                                               np.array((curr_sen_POS_cnt_dict).values())/float(len(sen_words))))
                elif d_name is "moderate_cont":
                    self.moderate_cont_POS_dist_dict[(clm,sen)]=((np.array((curr_clm_POS_cnt_dict.values()))/float(len(clm_words)),
                                                               np.array((curr_sen_POS_cnt_dict).values())/float(len(sen_words))))
                elif d_name is "strong_cont":
                    self.strong_cont_POS_dist_dict[(clm,sen)]=((np.array((curr_clm_POS_cnt_dict.values()))/float(len(clm_words)),
                                                               np.array((curr_sen_POS_cnt_dict).values())/float(len(sen_words))))
                elif d_name is "not_rel":
                    self.not_rel_POS_dist_dict[(clm,sen)]=((np.array((curr_clm_POS_cnt_dict.values()))/float(len(clm_words)),
                                                               np.array((curr_sen_POS_cnt_dict).values())/float(len(sen_words))))    
            if d_name is "strong_supp":
                utils_linux.save_pickle(d_name+"_POS_tag_cnt", self.strong_supp_POS_dist_dict)  
            elif d_name is "moderate_supp":
                utils_linux.save_pickle(d_name+"_POS_tag_cnt", self.moderate_supp_POS_dist_dict)                       
            elif d_name is "neu":
                utils_linux.save_pickle(d_name+"_POS_tag_cnt", self.moderate_supp_POS_dist_dict)                 
            elif d_name is "moderate_cont":
                utils_linux.save_pickle(d_name+"_POS_tag_cnt", self.moderate_cont_POS_dist_dict)              
            elif d_name is "strong_cont":
                utils_linux.save_pickle(d_name+"_POS_tag_cnt", self.strong_cont_POS_dist_dict)               
            elif d_name is "not_rel":
                utils_linux.save_pickle(d_name+"_POS_tag_cnt", self.not_rel_POS_dist_dict)        
            
            POS_tag_order_dict=dict.fromkeys(curr_clm_POS_cnt_dict.keys())
            utils_linux.save_pickle("POS_tag_order_dict",POS_tag_order_dict)
            
            
#             return curr_clm_POS_cnt_dict.keys()
            
        except Exception as err: 
            sys.stderr.write('problem in POS_tag_on_clm_sen:')     
            print err.args      
            print err
    
    def analyze_POS_tags_stats(self):
        try:
            strong_supp_dict=utils_linux.read_pickle("Strong_Support_annotated_sen_sorted")
            moderate_supp_dict=utils_linux.read_pickle("Moderate_Support_annotated_sen_sorted")
            neutral_dict=utils_linux.read_pickle("Neutral_annotated_sen_sorted")
            moderate_cont_dict=utils_linux.read_pickle("Moderate_Contradict_annotated_sen_sorted")
            strong_cont_dict=utils_linux.read_pickle("Strong_Contradict_annotated_sen_sorted")
            not_rel_dict=utils_linux.read_pickle("Non_relevant_annotated_sen_sorted")
            
            category_dict_list=[(strong_supp_dict,"strong_supp"),(moderate_supp_dict,"moderate_supp"),(neutral_dict,"neu"),(moderate_cont_dict,"moderate_cont"),
                       (strong_cont_dict,"strong_cont"),(not_rel_dict,"not_rel")]
            
            #how many POS tags in each sen and claim
            for (d,d_name) in category_dict_list:
                self.POS_tag_on_clm_sen(d,d_name)
                #read pickles
#                 if d_name is "strong_supp":
#                     self.strong_supp_POS_dist_dict=utils_linux.read_pickle(d_name+"_POS_tag_cnt")
#                 elif d_name is "moderate_supp":
#                     self.moderate_supp_POS_dist_dict=utils_linux.read_pickle(d_name+"_POS_tag_cnt")
#                 elif d_name is "neu":
#                     self.neutral_POS_dist_dict=utils_linux.read_pickle(d_name+"_POS_tag_cnt")
#                 elif d_name is "moderate_cont":
#                     self.moderate_cont_POS_dist_dict=utils_linux.read_pickle(d_name+"_POS_tag_cnt")
#                 elif d_name is "strong_cont":
#                     self.strong_cont_POS_dist_dict=utils_linux.read_pickle(d_name+"_POS_tag_cnt")
#                 elif d_name is "not_rel":
#                     self.not_rel_POS_dist_dict=utils_linux.read_pickle(d_name+"_POS_tag_cnt")
            
            self.POS_tags_order=utils_linux.read_pickle("POS_tag_order_dict").keys() 
            category_POS_tag_dict_list=[(self.strong_supp_POS_dist_dict,"strong_supp"),(self.moderate_supp_POS_dist_dict,
                                    "moderate_supp"),(self.neutral_POS_dist_dict,"neu"),(self.moderate_cont_POS_dist_dict,
                                    "moderate_cont"),(self.strong_cont_POS_dist_dict,"strong_cont"),
                                    (self.not_rel_POS_dist_dict,"not_rel")]   
              
            #now for every category dict, we have the stats on the POS tags in each clm and sen pair -> analyze it
            #for every dict, turn to a matrix, and them sum/avg...etc
            for (d,d_name) in category_POS_tag_dict_list:
                POS_tag_clm_cnt_matrix=np.zeros((1,len(self.word_level_POS_tags))) #for the dist in sen
                POS_tag_sen_cnt_matrix=np.zeros((1,len(self.word_level_POS_tags))) # for the dist in clm
                for ((clm,sen),(clm_POS_dist,sen_POS_dist)) in d.items():
#                     curr_clm_vector=np.zeros((1,len(self.word_level_POS_tags)))
                    POS_tag_clm_cnt_matrix=np.vstack((POS_tag_clm_cnt_matrix,clm_POS_dist))
#                     curr_sen_vector=np.zeros((1,len(self.word_level_POS_tags)))
                    POS_tag_sen_cnt_matrix=np.vstack((POS_tag_sen_cnt_matrix,sen_POS_dist))
                POS_tag_clm_cnt_matrix=np.delete(POS_tag_clm_cnt_matrix,(0),axis=0)    
                POS_tag_sen_cnt_matrix=np.delete(POS_tag_sen_cnt_matrix,(0),axis=0)
                POS_tag_clm_avg=np.mean(POS_tag_clm_cnt_matrix, axis=0) 
                POS_tag_sen_avg=np.mean(POS_tag_sen_cnt_matrix, axis=0)
                
                #transform to a tuple of (POS tag and its avg count, and then sort from max to min 
                clm_POS_tag_and_avg={}
                sen_POS_tag_and_avg={}
                for idx in range(0,len(POS_tag_clm_avg)):
                    clm_POS_tag_and_avg[self.POS_tags_order[idx]]=POS_tag_clm_avg[idx]
                for idx in range(0,len(POS_tag_sen_avg)):
                    sen_POS_tag_and_avg[self.POS_tags_order[idx]]=POS_tag_sen_avg[idx]    
                #sort
                clm_POS_tag_and_avg_sorted=collections.OrderedDict(sorted(clm_POS_tag_and_avg.items(),key=lambda x: (float(x[1])), reverse=True))
                sen_POS_tag_and_avg_sorted=collections.OrderedDict(sorted(sen_POS_tag_and_avg.items(),key=lambda x: (float(x[1])), reverse=True))
                #save to file
                with open (d_name+"_POS_tag_avg_sorted.csv","wb") as csvfile:
                    POS_tag_avg = csv.writer(csvfile)
                    POS_tag_avg.writerow(["clm_POS_tag_and_avg"])
                    for i in range(0,len(clm_POS_tag_and_avg_sorted)):
                        POS_tag_avg.writerow([" | "+clm_POS_tag_and_avg_sorted.keys()[i]+ " | "+str(clm_POS_tag_and_avg_sorted.values()[i])])
                    POS_tag_avg.writerow(["sen_POS_tag_and_avg"])  
                    for i in range(0,len(sen_POS_tag_and_avg_sorted)):
                        POS_tag_avg.writerow([" | "+sen_POS_tag_and_avg_sorted.keys()[i]+ " | "+str(sen_POS_tag_and_avg_sorted.values()[i])])
                          
#                 with open (d_name+"_POS_tag_avg.csv","wb") as csvfile:
#                     POS_tag_avg = csv.writer(csvfile)
#                     POS_tag_avg.writerow(["POS_tag_clm_avg | "+','.join(map(str, POS_tag_clm_avg))])
#                     POS_tag_avg.writerow(["POS_tag_sen_avg | "+','.join(map(str, POS_tag_sen_avg))])
                
#                 for ((clm,sen),(clm_POS_dict,sen_POS_dict)) in d.items():
# #                     curr_clm_vector=np.zeros((1,len(self.word_level_POS_tags)))
#                     POS_tag_clm_cnt_matrix=np.vstack((POS_tag_clm_cnt_matrix,clm_POS_dict.values()))
# #                     curr_sen_vector=np.zeros((1,len(self.word_level_POS_tags)))
#                     POS_tag_sen_cnt_matrix=np.vstack((POS_tag_sen_cnt_matrix,sen_POS_dict.values()))
#                 POS_tag_clm_cnt_matrix=np.delete(POS_tag_clm_cnt_matrix,(0),axis=0)    
#                 POS_tag_sen_cnt_matrix=np.delete(POS_tag_sen_cnt_matrix,(0),axis=0)
#                 POS_tag_clm_avg=np.mean(POS_tag_clm_cnt_matrix, axis=0) 
#                 POS_tag_sen_avg=np.mean(POS_tag_sen_cnt_matrix, axis=0)
                print (d_name +" POS_tag_clm_avg sum: "+str(sum(clm_POS_tag_and_avg_sorted.values())))
                print (d_name +" POS_tag_sen_avg: "+str(sum(sen_POS_tag_and_avg_sorted.values())))                                 
        except Exception as err: 
            sys.stderr.write('problem in analyze_POS_tags_stats:')     
            print err.args      
            print err 
        
    
def main():
        try:
            setup="single_sen"
            site="M-turk"
            claim_sen_lin_anal=linguistic_analysis()
            
            claim_sen_lin_anal.analyze_POS_tags_stats()
            
        except Exception as err: 
                sys.stderr.write('problem in main:')     
                print err.args      
                print err             
            
            
if __name__ == '__main__':
  main()   