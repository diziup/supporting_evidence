'''
30.05.14
After all the rel method - regular and with the relevance model (exp queries) have been done, 
this script creates a set of sentences, from all the rel methods,
because we don't want duplicates.

@author: Liora
'''

import sys
import pickle
from test.test_set import cube
from sets import Set
import csv
import collections
from os import listdir
import linecache

class sentences_set_per_query:
    
    LM_dict={}
    MRF_dict={}
    RM_dict={}
    OKAPI_dict={}
    MANUAL_dict={}
    LMExp_dict={}
    MANUALExp_dict={}
    LM_no_senti_dict={}
    dicID_title_dict={}

    
    def __init__(self):
            LM_dict={}
            MRF_dict={}
            RM_dict={}
            OKAPI_dict={}
            MANUAL_dict={}
            LMExp_dict={}
            MANUALExp_dict={}
            LM_no_senti_dict={}
            
            dicID_title_dict={}
            

        
    """
    1. sets from all the relevance models
    2. create a set for each query as union of the above
    """  
    
    def read_pickle(self,wiki_RT,model):
        try: 
            if wiki_RT is "wiki" and not model is "LM_exp" and not model is "MANUAL_exp":  
                with open('topSen_'+wiki_RT+'_new_'+model+'_fin_pickle.txt', 'rb') as handle:
                    if model is "LM":
                        self.LM_dict = pickle.loads(handle.read())
                    elif model is "MRF":
                        self.MRF_dict = pickle.loads(handle.read()) 
                    elif model is "OKAPI":
                        self.OKAPI_dict = pickle.loads(handle.read())     
                    elif model is "RM":
                        self.RM_dict = pickle.loads(handle.read())    
                    elif model is "MANUAL":
                        self.MANUAL_dict = pickle.loads(handle.read()) 
                    elif model is "LM_no_senti":
                        self.LM_no_senti_dict = pickle.loads(handle.read())
                    
            elif not model is "LM_exp" and not model is "MANUAL_exp":
                with open('topSen_'+wiki_RT+'_'+model+'_fin_pickle.txt', 'rb') as handle:
                    if model is "LM":
                        self.LM_dict = pickle.loads(handle.read())
                    elif model is "MRF":
                        self.MRF_dict = pickle.loads(handle.read()) 
                    elif model is "OKAPI":
                        self.OKAPI_dict = pickle.loads(handle.read())     
                    elif model is "RM":
                        self.RM_dict = pickle.loads(handle.read())    
                    elif model is "MANUAL":
                        self.MANUAL_dict = pickle.loads(handle.read()) 
                    elif model is "LM_no_senti":
                        self.LM_no_senti_dict = pickle.loads(handle.read()) 
            if model is "LM_exp" :
                with open('topSen_'+wiki_RT+'_'+model+'_exp_fin_pickle.txt', 'rb') as handle:
                    self.LMExp_dict = pickle.loads(handle.read()) 
            elif model is "MANUAL":
                with open('topSen_'+wiki_RT+'_'+model+'_exp_fin_pickle.txt', 'rb') as handle:
                    self.MANUALExp_dict = pickle.loads(handle.read())
          
        #as of 26.06 decided to not use this becuase of the new wiki collection
#             if model is "LM" :
#                 with open('topSen_'+wiki_RT+'_'+model+'_exp_fin_pickle.txt', 'rb') as handle:
#                     self.LMExp_dict = pickle.loads(handle.read()) 
#             if model is "MANUAL":
#                 with open('topSen_'+wiki_RT+'_'+model+'_exp_fin_pickle.txt', 'rb') as handle:
#                     self.MANUALExp_dict = pickle.loads(handle.read())
        except Exception as err: 
                    sys.stderr.write('problem in read_pickle:' ,wiki_RT,model)     
                    print err.args      
                    print err 
#     
    def create_set_per_query(self,wiki_RT):
        try:
                        
            #go over all the models dict to get 
            dict_list=[self.LM_dict,self.MRF_dict,self.RM_dict,self.OKAPI_dict,self.MANUAL_dict,self.LMExp_dict,self.MANUALExp_dict,self.LM_no_senti_dict]
#             dict_list=[self.LM_dict]
            curr_set=set()
            qID_sen_docid_dict={}
    #         temp_sen_for_q_dict={}
            for qID in range(1,101):
                for curr_dict in dict_list:
                    curr_keys=curr_dict.keys()
                    for key in curr_keys:
                        if qID is int(key[0]):
                            curr_set.update([(key,curr_dict[key][0],curr_dict[key][1])])# 20.06. add the sentence score and docID, instead of just the [key]=qID and sen
                                                                                        #and so : claim_num, sen,score,docId
                            if key not in qID_sen_docid_dict.keys(): #save the qID,sen docID.
                                qID_sen_docid_dict[key]=curr_dict[key][1]
#                                 
            final_sen_from_set_dict= dict.fromkeys(curr_set,0)
            final_sen_from_set_sorted = collections.OrderedDict(sorted(final_sen_from_set_dict.items(), key=lambda item: (-int(item[0][0][0]), float(item[0][1])), reverse=True))
                                                                                                        
            with open("sen_set_no_docID"+wiki_RT+"_with_score_pickle", 'wb') as handle:
                    pickle.dump(final_sen_from_set_sorted, handle)
           
            with open("qID_sen_docid_dict"+wiki_RT+"_pickle", 'wb') as handle:
                    pickle.dump(qID_sen_docid_dict, handle)
            
            #match the qID,sen with docID
            """20.06.- no need
            for (qID_sen,val) in final_sen_from_set_sorted.items():
                final_sen_from_set_sorted[qID_sen]=qID_sen_docid_dict[qID_sen]
            """
            #save or read to csv file and pickle
                
            with open("sen_set_"+wiki_RT+"_with_score_pickle", 'wb') as handle:
                    pickle.dump(final_sen_from_set_sorted, handle)
           
#             with open("sen_set_"+wiki_RT+"_pickle", 'rb') as handle:
#                     final_sen_from_set_sorted = pickle.loads(handle.read())
                    
            w = csv.writer(open("sen_set_"+wiki_RT+"_with_score_fin.csv" , "wb"))
            for (qID_sen,score,docID) in final_sen_from_set_sorted.keys():
                    l = []
                    l.append('%s|%s|%s|%s' % (str(qID_sen[0]), str(qID_sen[1]),self.dicID_title_dict[docID],str(score)))
                    w.writerow(l)
                    
#             w = csv.writer(open("sen_set_"+wiki_RT+"with_score_fin.csv" , "wb"))
#             for (qID_sen,docID) in final_sen_from_set_sorted.keys():
#                     l = []
#                     l.append('%s|%s|%s' % (str(qID_sen[0]), str(qID_sen[1]),self.dicID_title_dict[docID]))
#                     w.writerow(l)
#              
        except Exception as err: 
                    sys.stderr.write('problem in create_set_per_query:')     
                    print err.args      
                    print err 
    
    def create_shallow_pool(self,wiki_RT):
        """
        take from every ret model (=LM, MRF, RM,OKAPI,MANUAL, LM_Exp, MAN_exp and no_sentiment_word) the top 7  sentences ~ for 
            a 50 sentence total pool for every claim
        """
        try:
            dict_list=[self.LM_dict,self.MRF_dict,self.RM_dict,self.OKAPI_dict,self.MANUAL_dict,self.LM_no_senti_dict] #no exp dicts, removed it in the end cus of the new wiki collection
            shallow_pool_dict={}
            qID_sen_docid_dict={}
    #         temp_sen_for_q_dict={}
            for qID in range(1,101):
                for curr_dict in dict_list:
                    try:
                        sen_cnt=0
        #                 while sen_cnt < 9:   
                        for key in curr_dict.keys():  
                            if sen_cnt < 9:     #30.06 -take more than needed to make sure that there will be 30 sentences for each claim 
                                claim_num=key[0]
                                if qID is int(claim_num):  
                                    sen=key[1]
                                    if not (claim_num,sen) in shallow_pool_dict.keys() and not "http" in sen and not "cite news" in sen and not "accessed" in sen and not "retrieved" in sen and not "Retrieved" in sen and not "class=" in sen  and not "scope=" in sen  and "*" not in sen and "**" not in sen and not len(sen.split()) <3:
                                        shallow_pool_dict[claim_num,sen]=[curr_dict[(key)]]
                                        sen_cnt+=1
                                        if key not in qID_sen_docid_dict.keys(): #save the qID,sen docID.
                                            qID_sen_docid_dict[key]=curr_dict[key][1]
                                            continue  
                                    else:
                                        continue
                                else:              
                                    continue        
                            else:
                                sen_cnt=0
                                break
                    except Exception as err: 
                        sys.stderr.write('problem in create_shallow_pool qid:' +claim_num)     
                        print err.args      
                        print err            
    #                         shallow_pool_dict[(claim_num,sen)]=
    #                         curr_set.update([(key,curr_dict[key][0],curr_dict[key][1])])# 20.06. add the sentence score and docID, instead of just the [key]=qID and sen
            #turn the dict from a key=(clain_num,sen) and val (score,dic_id) to all this tuple in the key, for create_annotation compatibilty 
          
            
            shallow_pool_dict_sorted=collections.OrderedDict(sorted(shallow_pool_dict.items(), key=lambda item: (-int(item[0][0])), reverse=True)) #sort by claim_num and score - score added 30.06
            
            with open("shallow_pool_dict_"+wiki_RT, 'wb') as handle:
                        pickle.dump(shallow_pool_dict_sorted, handle)               
                               
            w = csv.writer(open("shallow_pool_dict_"+wiki_RT+".csv" , "wb"))
            for item in shallow_pool_dict_sorted.items():
                    l = []
                    l.append('%s|%s|%s|%s' % (str(item[0][0]), str(item[0][1]),item[1][0][0],str(item[1][0][1]))) #claim_num,sen,score,doc id
                    w.writerow(l)        
        
        except Exception as err: 
                    sys.stderr.write('problem in create_shallow_pool:')     
                    print err.args      
                    print err     
                               
    def find_docID_title(self,wiki_RT):
        try:
      
#             wiki_RT="wiki"
            wiki_path=r"C:\study\technion\MSc\Thesis\Y!\datasets\wikipedia_movie_articles_0614"
            RT_path=r"C:\study\technion\MSc\Thesis\Y!\datasets\RT"
            if wiki_RT is "wiki":
                list_of_files=listdir(wiki_path)
                curr_path=wiki_path
            else :
                list_of_files=listdir(RT_path)
                curr_path=RT_path
            for filename in list_of_files:
                try:
                    if filename.endswith(".txt"):
                        curr_file=open(curr_path+"\\"+filename,'r').read().strip()
                        for i, line in enumerate(curr_file.split('\n')):
                            if i is 1:
                                docID_temp=line.split("<DOCNO>")[1]
                                docID=docID_temp.split("</DOCNO>")[0]
                            if i is 3:
                                title_temp=line.split("<title>")[1]
                                title=title_temp.split("__")[0]
                                self.dicID_title_dict[docID]=title
                except Exception as err: 
                        sys.stderr.write('problem in find_docID_title:'+filename)                
                        print err.args                 
            #save to pickle
            with open("dicID_title_mapping_"+wiki_RT+"_pickle", 'wb') as handle:
                    pickle.dump(self.dicID_title_dict, handle)   
           
            with open("dicID_title_mapping_"+wiki_RT+"_pickle", 'rb') as handle:
                    self.dicID_title_dict = pickle.loads(handle.read())        
             
        except Exception as err: 
                    sys.stderr.write('problem in find_docID_title:')     
                    print err.args      
                    print err 
                        
    def statistics_sen_set(self,wiki_RT):
        #get stats on number of sen per query in final sen set
        try:
            final_sen_from_set_sorted={}
            qID_sen_stats={}
            with open("sen_set_"+wiki_RT+"_pickle", 'rb') as handle:
                final_sen_from_set_sorted = pickle.loads(handle.read()) 
            for (qID_sen,docID) in final_sen_from_set_sorted.items():
                if qID_sen[0] in qID_sen_stats.keys():
                    qID_sen_stats[qID_sen[0]]=qID_sen_stats[qID_sen[0]]+1
                else:
                    qID_sen_stats[qID_sen[0]]=1
            
            qID_sen_stats_sorted=collections.OrderedDict(sorted(qID_sen_stats.items(), key=lambda item: (-int(item[0])), reverse=True))        
            with open("sen_set_stats"+wiki_RT+"_pickle", 'wb') as handle:
                    pickle.dump(qID_sen_stats_sorted, handle)
            w = csv.writer(open("sen_set_stats"+wiki_RT+"_fin.csv" , "wb"))
    #             for k, v in order_sen_qID_score.items():
            for (qID,doc_num) in qID_sen_stats_sorted.items():
                    w.writerow([str(qID)+"|"+str(doc_num)])
            
                
        except Exception as err: 
                    sys.stderr.write('problem in statistics_sen_set:')     
                    print err.args      
                    print err          
          
                
def main():
        try:
            wiki_RT="wiki"
            models=["LM","MRF","RM","OKAPI","MANUAL","LM_no_senti"]
            sen_per_q=sentences_set_per_query()
#             sen_per_q.find_docID_title(wiki_RT)
             
#             with open("dicID_title_mapping_"+wiki_RT+"_pickle", 'rb') as handle:
#                     sen_per_q.dicID_title_dict= pickle.loads(handle.read())
           
            for model in models:
                sen_per_q.read_pickle(wiki_RT, model)
#             create a set for every query
#             sen_per_q.create_set_per_query(wiki_RT)
            sen_per_q.create_shallow_pool(wiki_RT)
#             sen_per_q.statistics_sen_set(wiki_RT)
        except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err 
            

if __name__ == '__main__':
    main()