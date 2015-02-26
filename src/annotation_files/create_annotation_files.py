'''
01.06.14
Script to create claims and corresponding sentence csv files for M-TURK
@author: Liora

1. create 10  source batches- for queries 1-10, 11-20 etc.
2. for each source batch, take a query and sentence up to 10 pairs for batch.
3. if there are left-overs, concatinate them.
'''
import sys
import csv
import pickle
import collections
from itertools import islice
import random

# from pandas.sandbox.qtpandas import Form

wiki_RT="wiki"
claim_batch_dict={}
RT_claim_key_sen_value_dict={}
wiki_sen_dict={}
                    

class support_sen_files:
    final_sen_set_sorted_wiki={}
    final_sen_set_sorted_RT={}
    claim_batch_dict={}
    claim_num_of_sen={}
    claim_num_of_sen_sorted={}
    claim_text_dict={}
    wiki_claim_sen_as_key_dict={}
    RT_claim_sen_as_key_dict={}
    trick_sen=[]
    docID_title_wiki_dict={}
    docID_title_RT_dict={}
    claim_kappa_dict={}
    
    def __init__(self):
            final_sen_set_sorted={}
            claim_batch_dict={}          
            claim_num_of_sen_sorted={}       
            
    def create_claim_batches_dict(self):
        claim_batch_dict={}
    #     for qID in range(1,101):
        for i in range(1,11):
            for j in range(1,11):
                self.claim_batch_dict[i]=([range((i-1)*10+1,(i-1)*10+(j+1))])
    
    def convert_claim_sen_dict(self):
        """
        make the RT/wiki dict be in the form of [claim_nun]-> [list of sen]
        instead of [claim,sen]->1
        """
        for (claim,sen) in self.RT_claim_sen_as_key_dict.keys():
            if claim not in self.RT_claim_key_sen_value_dict.keys():
                self.RT_claim_key_sen_value_dict[claim]=[sen]
            else:
                self.RT_claim_key_sen_value_dict[claim].extend([sen])    
    
    def create_2_claim_batches_dict(self):
            for i in range(1,51):
                for j in range(1,11):
                    claim_batch_dict[i]=([range((i-1)*2+1,(i-1)*2+3)])
    
    def create_10_claim_batches_dict(self):
        for i in range(1,11):
            for j in range(1,11):
                claim_batch_dict[i]=(range((i-1)*10+1,(i-1)*10+(j+1)))
    
    def read_pickle(self,file_name):
        if file_name is "sen_set_wiki_pickle" or file_name is "shallow_pool_dict_wiki":
            with open(file_name, 'rb') as handle:
                        self.final_sen_set_sorted_wiki = pickle.loads(handle.read())
        elif file_name is "sen_set_RT_pickle" or file_name is "shallow_pool_dict_RT":
            with open(file_name, 'rb') as handle:
                        self.final_sen_set_sorted_RT = pickle.loads(handle.read())
        elif file_name is "sen_set_wiki_with_score_pickle":
            with open(file_name, 'rb') as handle:
                        self.final_sen_set_sorted_wiki = pickle.loads(handle.read())
        elif file_name is "sen_set_RT_with_score_pickle":
            with open(file_name, 'rb') as handle:
                        self.final_sen_set_sorted_RT = pickle.loads(handle.read())
        elif file_name is "sen_set_stats_wiki_pickle" or file_name is "sen_set_stats_RT_pickle":
            with open(file_name, 'rb') as handle:
                        self.claim_num_of_sen= pickle.loads(handle.read())
        elif file_name is "filtered_wiki_sen_set_pickle":
            with open(file_name, 'rb') as handle:
                        self.final_sen_set_sorted_wiki= pickle.loads(handle.read())
        elif file_name is "claim_key_sen_value_wiki_dict":
            with open(file_name, 'rb') as handle:
                        self.wiki_claim_sen_as_key_dict= pickle.loads(handle.read())
        elif file_name is "claim_key_sen_value_RT_dict":
            with open(file_name, 'rb') as handle:
                        self.RT_claim_sen_as_key_dict= pickle.loads(handle.read())
        elif file_name is "dicID_title_mapping_wiki_pickle":
            with open(file_name, 'rb') as handle:
                        self.docID_title_wiki_dict= pickle.loads(handle.read())
        elif file_name is "dicID_title_mapping_RT_pickle":
            with open(file_name, 'rb') as handle:
                        self.docID_title_RT_dict= pickle.loads(handle.read())
                        
    def save_pickle(self,file_name):
        if file_name is "filtered_wiki_sen_set_pickle":
            with open(file_name, 'wb') as handle:
                    pickle.dump(self.final_sen_set_sorted_wiki, handle)       
        elif file_name is "claim_key_sen_value_wiki_dict":
            with open(file_name, 'wb') as handle:
                    pickle.dump(self.wiki_claim_sen_as_key_dict, handle)  
        elif file_name is "claim_key_sen_value_RT_dict":
            with open(file_name, 'wb') as handle:
                    pickle.dump(self.RT_claim_sen_as_key_dict, handle) 
        
    def compute_stats_claims_sen(self):
#         for item in self.claim_num_of_sen.items():
        #for each batch of the 
        self.claim_num_of_sen_sorted= collections.OrderedDict(sorted(self.claim_num_of_sen.items(), key=lambda item: (-int(item[1])), reverse=True))   
        
        w=csv.writer(open("claim_num_of_sen_sorted_"+wiki_RT+"_fin.csv" , "wb"))
        for (qID,doc_num) in self.claim_num_of_sen_sorted.items():
                    w.writerow([str(qID)+"|"+str(doc_num)])
        """
        CONTINUE HEAR according to Anna's, Oren's and Roi's answers
        """
        
    def take(self,start, stop,iterable):
        return list(islice(iterable, start,stop))
    
    def read_claims_text_file(self):
        claim_text_file=open(r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\rawClaim_SW.txt",'r').read().strip()        
        for i, line in enumerate(claim_text_file.split('\n')): 
            self.claim_text_dict[i+1]=line.split("|")[1]
    
    def read_trick_supp_file(self):
        trick_sen_file=open(r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\trick_questions.txt",'r').read().strip()
        for i,line in enumerate(trick_sen_file.split('\n')):
            self.trick_sen.extend([(line.split("|")[0],line.split("|")[1])])    

    def filter_wiki_sen_set(self):
        try:
            """
            with sorting by score version
            """
            for (qID_sen,score,docID) in self.final_sen_set_sorted_wiki.keys():
                if "http" in qID_sen[1] or "cite news" in qID_sen[1] or "accessed" in qID_sen[1] or "retrieved" in qID_sen[1] or "Retrieved" in qID_sen[1] or "class=" in qID_sen[1] or "scope=""row" in qID_sen[1] or len(qID_sen[1].split()) < 3 :
                    del self.final_sen_set_sorted_wiki[(qID_sen,score,docID)]
                
                """ the original version
                for (qID,sen) in self.final_sen_set_sorted_wiki.keys():
                    if "http" in sen or "cite news" in sen or "accessed" in sen or "retrieved" in sen or "class=" in sen or "scope=""row" in sen or len(sen) < 3 :
                        del self.final_sen_set_sorted_wiki[(qID,sen)]
                """
            self.save_pickle("filtered_wiki_sen_set_pickle")            
        except Exception as err: 
                    sys.stderr.write('problem in filtered_wiki_sen_set_pickle:')     
                    print err.args      
                    print err
                        
    def convert_sen_claim_dict(self):
    
        #for the wiki dict
        for item in self.final_sen_set_sorted_wiki.items(): #key is (claim,sen) val is list of tuple of  {score,docId)
                claim=item[0][0]
                sen=item[0][1]
                doc_id=item[1][0][1] 
                if claim not in self.wiki_claim_sen_as_key_dict.keys():
                        self.wiki_claim_sen_as_key_dict[claim]=[(sen,doc_id)]    
                else:         
                    self.wiki_claim_sen_as_key_dict[claim].extend([(sen,doc_id)])
        self.save_pickle("claim_key_sen_value_wiki_dict")
        
        for item in self.final_sen_set_sorted_RT.items():
                claim=item[0][0]
                sen=item[0][1]
                doc_id=item[1][0][1] 
                if claim not in self.RT_claim_sen_as_key_dict.keys():
                        self.RT_claim_sen_as_key_dict[claim]=[(sen,doc_id)]
                else:
                        self.RT_claim_sen_as_key_dict[claim].extend([(sen,doc_id)])
        self.save_pickle("claim_key_sen_value_RT_dict")
#             for key in self.final_sen_set_sorted_wiki.keys():
# #                 claim=key[0][0]
# #                 sen=key[0][1]
#                 claim=key[0]
#                 sen=key[1]
#                 doc_id=key[2]
#                 if claim not in self.wiki_claim_sen_as_key_dict.keys():
#                         self.wiki_claim_sen_as_key_dict[claim]=[(sen,doc_id)]    
#                 else:         
#                     self.wiki_claim_sen_as_key_dict[claim].extend([(sen,doc_id)])
#             self.save_pickle("claim_key_sen_value_wiki_dict")
        #for the RT dict
#             for key in self.final_sen_set_sorted_RT.keys():
#                 claim=key[0][0]
#                 sen=key[0][1]
#                 doc_id=key[2]
#                 if claim not in self.RT_claim_sen_as_key_dict.keys():
#                         self.RT_claim_sen_as_key_dict[claim]=[(sen,doc_id)]
#                 else:
#                         self.RT_claim_sen_as_key_dict[claim].extend([(sen,doc_id)])
#             self.save_pickle("claim_key_sen_value_RT_dict")
    """ original version
        #for the wiki dict
            for (claim,sen) in self.final_sen_set_sorted_wiki.keys():
                    if claim not in self.wiki_claim_sen_as_key_dict.keys():
                        self.wiki_claim_sen_as_key_dict[claim]=[(sen,self.final_sen_set_sorted_wiki[(claim,sen)])]
                    else:
#                         self.wiki_claim_sen_as_key_dict[claim].extend([sen])
                        self.wiki_claim_sen_as_key_dict[claim].extend([(sen,self.final_sen_set_sorted_wiki[(claim,sen)])])
            self.save_pickle("claim_key_sen_value_wiki_dict")
        #for the RT dict
            for (claim,sen) in self.final_sen_set_sorted_RT.keys():
                    if claim not in self.RT_claim_sen_as_key_dict.keys():
                        self.RT_claim_sen_as_key_dict[claim]=[(sen,self.final_sen_set_sorted_RT[(claim,sen)])]
                    else:
                        self.RT_claim_sen_as_key_dict[claim].extend([(sen,self.final_sen_set_sorted_RT[(claim,sen)])])
            self.save_pickle("claim_key_sen_value_RT_dict")
        """
           
    def create_support_files(self):

        """
        1. in batches of 10 claims per file,
            take a sentence for each claim. 
            the csv file format will be "claim1","sentence1",claim2,sentence2, etc 
        2.Two setups -  taking sentences randomely or sorting the dict by the sentence's score and taking from each DS top 25
        """
#         self.filter_wiki_sen_set()
#         self.read_pickle("filtered_wiki_sen_set_pickle")
        self.convert_sen_claim_dict()
        self.read_pickle("claim_key_sen_value_RT_dict")
        self.read_pickle("claim_key_sen_value_wiki_dict")
        self.read_pickle("dicID_title_mapping_wiki_pickle")
        self.read_pickle("dicID_title_mapping_RT_pickle")
        self.read_claims_text_file()
        self.read_trick_supp_file()
        site="crowdflower"
        choose_sentence_method="shallow_pool"
        
        for claim_num in [1]:#[4,21,7,17,37]:#,36,39,40,41]:
#         for claim_num in [4]:
            claim_text=self.claim_text_dict[claim_num]
            if choose_sentence_method is "random": #choose sentences randomly
                rand_sen_index_wiki=[]
                while len(rand_sen_index_wiki) <25:
                    curr_rand_index=random.randint(1,len(self.wiki_claim_sen_as_key_dict[str(claim_num)])-1)
                    if curr_rand_index not in rand_sen_index_wiki:
                            rand_sen_index_wiki.append(curr_rand_index)
                rand_sen_index_RT=[]
                while len(rand_sen_index_RT) <25:
                    curr_rand_index=random.randint(1,len(self.RT_claim_sen_as_key_dict[str(claim_num)])-1)
                    if curr_rand_index not in rand_sen_index_RT:
                            rand_sen_index_RT.append(curr_rand_index)                              
                for i in range(0,5):
                    if site is "crowdflower":
                        trick_index_1=random.randint(1,len(self.trick_sen)-1)
                        trick_index_2=random.randint(1,len(self.trick_sen)-1)
                        while trick_index_2 ==trick_index_1 :
                                    trick_index_2=random.randint(1,len(self.trick_sen)-1)
                        with open("supp_claim_"+str(claim_num)+"_sen_"+str(5*i)+"_"+str(5*(i+1))+".csv", "wb") as csvfile:
        #                         trick_index=random.randint(1,len(self.trick_sen)-1)
                                csv.register_dialect('escaped', escapechar="'", quotechar='"',doublequote=False)
                                w_supp_sen= csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC,dialect='escaped')
                                
        #                         w_supp_sen.writerow( ['claim1','sen1','tit1','claim1','sen2','tit2','claim1','sen3','tit3','claim1','sen4','tit4','claim1','sen5','tit5','claim1','sen6','tit6','claim1','sen7','tit7','claim1','sen8','tit8','claim1','sen9','tit9','claim1','sen10','tit10','claim1','sen11','tit11'] )
                                w_supp_sen.writerow( ['claim1','sen1','tit1','_golden','cf_support_gold','cf_support_gold_reason'])
                                w_supp_sen.writerow((claim_text,self.trick_sen[trick_index_1][0],self.trick_sen[trick_index_1][1],"TRUE","Not relevant","The sentence is not related to the entity in the claim at all"))
                                w_supp_sen.writerow((claim_text,self.trick_sen[trick_index_2][0],self.trick_sen[trick_index_2][1],"TRUE","Not relevant","The sentence is not related to the entity in the claim at all"))
                                w_supp_sen.writerow((claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[0+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[0+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[0+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[0+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[1+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[1+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[1+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[1+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[2+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[2+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[2+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[2+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[3+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[3+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[3+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[3+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[4+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[4+i*5]][1]]))
                                w_supp_sen.writerow((claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[4+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[3+i*5]][1]]))
                    elif site is "turk":
                            trick_index=random.randint(1,len(self.trick_sen)-1)
                            with open("supp_claim_"+str(claim_num)+"_sen_"+str(5*i)+"_"+str(5*(i+1))+".csv", "wb") as csvfile:
                                
        #                         trick_index=random.randint(1,len(self.trick_sen)-1)
                                csv.register_dialect('escaped', escapechar="'", quotechar='"',doublequote=False)
                                w_supp_sen= csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC,dialect='escaped')
                                w_supp_sen.writerow( ['claim1','sen1','tit1','claim1','sen2','tit2','claim1','sen3','tit3','claim1','sen4','tit4','claim1','sen5','tit5','claim1','sen6','tit6','claim1','sen7','tit7','claim1','sen8','tit8','claim1','sen9','tit9','claim1','sen10','tit10','claim1','sen11','tit11'] )
                                w_supp_sen.writerow((claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[0+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[0+i*5]][1]],
                                claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[0+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[0+i*5]][1]],
                                claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[1+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[1+i*5]][1]],
                                claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[1+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[1+i*5]][1]],
                                claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[2+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[2+i*5]][1]],
                                claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[2+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[2+i*5]][1]],
                                claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[3+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[3+i*5]][1]],
                                claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[3+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[3+i*5]][1]],
                                claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[4+i*5]][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_wiki[4+i*5]][1]],
                                claim_text,self.trick_sen[trick_index][0],self.trick_sen[trick_index][1],
                                claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[4+i*5]][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][rand_sen_index_RT[4+i*5]][1]],
                                  ))                       

            elif choose_sentence_method is "25_top":#choose from RT and wiki the top 25 sentences
                for i in range(0,5):
                    if site is "turk":
                        trick_index=random.randint(1,len(self.trick_sen)-1)
                        with open("supp_claim_"+str(claim_num)+"_top_25_sen_"+str(5*i)+"_"+str(5*(i+1))+".csv", "wb") as csvfile:         
    #                         trick_index=random.randint(1,len(self.trick_sen)-1)
                            csv.register_dialect('escaped', escapechar="'", quotechar='"',doublequote=False)
                            w_supp_sen= csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC,dialect='escaped')
                            w_supp_sen.writerow( ['claim1','sen1','tit1','claim1','sen2','tit2','claim1','sen3','tit3','claim1','sen4','tit4','claim1','sen5','tit5','claim1','sen6','tit6','claim1','sen7','tit7','claim1','sen8','tit8','claim1','sen9','tit9','claim1','sen10','tit10','claim1','sen11','tit11'] )
                            w_supp_sen.writerow((claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][0+i*5][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][0+i*5][1]],
                            claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][0+i*5][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][0+i*5][1]],
                            claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][1+i*5][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][1+i*5][1]],
                            claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][1+i*5][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][1+i*5][1]],
                            claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][2+i*5][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][2+i*5][1]],
                            claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][2+i*5][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][2+i*5][1]],
                            claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][3+i*5][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][3+i*5][1]],
                            claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][3+i*5][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][3+i*5][1]],
                            claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][4+i*5][0],self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][4+i*5][1]],
                            claim_text,self.trick_sen[trick_index][0],self.trick_sen[trick_index][1],
                            claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][4+i*5][0],self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][4+i*5][1]],
                              ))         
            
            elif choose_sentence_method is "shallow_pool":
                """for each claim, i took 4 sentences from every ret method, and so each claim has ~ 32 sentences from each ds
                so total of 64 sentences, have 4 batch with 16 sentences in each"""
               
                if site is "crowdflower":
                    num_of_files=min(len(self.wiki_claim_sen_as_key_dict[str(claim_num)])/10,len(self.RT_claim_sen_as_key_dict[str(claim_num)])/10,3)
                    with open("supp_claim_"+str(claim_num)+".csv", "wb") as csvfile:
                        csv.register_dialect('escaped', escapechar="'", quotechar='"',doublequote=False)
                        w_supp_sen= csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC,dialect='escaped')
                        w_supp_sen.writerow( ['claim','sen','tit','_golden','cf_support_gold','cf_support_gold_reason'])

#                         trick_index_1=random.randint(1,len(self.trick_sen)-1)
#                         trick_index_2=random.randint(1,len(self.trick_sen)-1)
#                         while trick_index_2 ==trick_index_1 :
#                             trick_index_2=random.randint(1,len(self.trick_sen)-1)
                        w_supp_sen.writerow((claim_text,self.trick_sen[0][0],self.trick_sen[0][1],"TRUE","Not relevant","The sentence does not mention or relate to the entity in the claim at all"))
                        w_supp_sen.writerow((claim_text,self.trick_sen[1][0],self.trick_sen[1][1],"TRUE","Not relevant","The sentence does not mention or relate to the entity in the claim at all"))
                        w_supp_sen.writerow((claim_text,self.trick_sen[2][0],self.trick_sen[2][1],"TRUE","Not relevant","The sentence does not mention or relate to the entity in the claim at all"))
                        w_supp_sen.writerow((claim_text,self.trick_sen[3][0],self.trick_sen[3][1],"TRUE","Not relevant","The sentence does not mention or relate to the entity in the claim at all"))          
                        for file_num in range(0,num_of_files):
                            for i in range(0,10):
                                try:
                                    w_supp_sen.writerow([claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][i+file_num*10][0].decode('utf-8', 'ignore'),#for the sentence itself remove the [0] if want the tuiple of sentence and docid
                                                         self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][i+file_num*10][1]]])
#                                                            self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][i+file_num*16][1]]])    
                                    w_supp_sen.writerow([claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][i+file_num*10][0],
                                                         self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][i+file_num*10][1]]])  
                                                         
#                                                            self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][i+file_num*16][1]]])
#                                 w_supp_sen.writerow( ['_id,_pct_missed,_judgments,_gold_case,_difficulty,_hidden,_contention,_pct_contested,_gold_pool,cf_support_gold,cf_support_gold_reason,claim,sen,tit,_golden'])
#                                 
#                                     w_supp_sen.writerow([',,,,,,,,,,,',claim_text,self.wiki_claim_sen_as_key_dict[str(claim_num)][i+file_num*16],
#                                                            self.docID_title_wiki_dict[self.wiki_claim_sen_as_key_dict[str(claim_num)][i+file_num*16][1]]])
#                                     w_supp_sen.writerow([',,,,,,,,,,,',claim_text,self.RT_claim_sen_as_key_dict[str(claim_num)][i+file_num*16],
#                                                            self.docID_title_RT_dict[self.RT_claim_sen_as_key_dict[str(claim_num)][i+file_num*16][1]]])                     
#                                 
                                except Exception as err: 
                                            sys.stderr.write('problem in create_support_file in file:' +str(file_num)+ " claim:" +str(claim_num))
                                            print err.args      
                                            print err

def main():
        try:
            site="crowdflower"
            wiki_RT="wiki"
#             claim_sen_file.create_claim_batches_dict()
            claim_sen_file=support_sen_files()
            """
            for regular set sen pickle - chnage on 20.06 to check if the top 25 yield better sentences
            claim_sen_file.read_pickle("sen_set_wiki_pickle")
            claim_sen_file.read_pickle("sen_set_RT_pickle")
            """
            """before shallow pool
            claim_sen_file.read_pickle("sen_set_wiki_with_score_pickle")
            claim_sen_file.read_pickle("sen_set_RT_with_score_pickle")
             """
            claim_sen_file.read_pickle("shallow_pool_dict_wiki")
#             claim_sen_file.read_pickle("shallow_pool_dict_RT")
#                 claim_sen_file.save_pickle(wiki_RT,"claim_batch_"+wiki_RT+"_pickle",claim_sen_file.claim_batch_dict)
#                 claim_sen_file.read_pickle("sen_set_stats_"+wiki_RT+"_pickle")
#                 claim_sen_file.compute_stats_claims_sen()
#             sen_per_q.statistics_sen_set(wiki_RT)
            claim_sen_file.create_support_files()
                    

        except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err 
            

if __name__ == '__main__':
    main()
