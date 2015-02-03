'''
27.05.'14
A standard (TRUE!) relevance model building class.
For each sentence, it will calculate a LM ,
and perform a smoothing with Dirichlet prior and mu = average sentence length.

(This is a fix to the previous rel model that conctatinated all the validate sentences 
and from that calculated the prob-  not good).


@author: Liora Braunstein

Stages:
1. parse the wiki and RT corpus separately.
2. collect corpus statistics for the smoothing - for each term its frequency in the index
3. for each sentence (!), calculate its prob vector.
4. combine all the prob vectors above, divide by total number of sentences (validated or total in the collection?)
5. take top k (5%?) most freq terms, keep their weights.
6. expand for the LM and MANUAL models.
'''
from __future__ import division
import sys
import nltk
import pickle
import collections
import csv
import math
import glob
import os
from nltk.corpus import stopwords

wiki_path=r"C:\study\technion\MSc\Thesis\Y!\datasets\wikipedia_dump010114_parsed"
RT_path=r"C:\study\technion\MSc\Thesis\Y!\datasets\RT"
coll_stats_f=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\standard_rel_model\collection_stats_"
# coll_stats=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\standard_rel_model\collection_stats_RT"


class standard_rel_model:
    
    average_sen_len=0
    coll_stats_dict={}
    sen_per_query_cnt_dict={}
    words_dict={}
    words_dict_sorted={}
    total_word_count_per_q_dict={}
    kTop_freq_words_dict={}
    kTop_freq_words_sorted_dict={}
    
    def __init__(self):
        coll_stats_dict={}
        sen_per_query_cnt_dict = {}
        words_dict={}
        words_dict_sorted={}
        total_word_count_per_q_dict={}
        kTop_freq_words_dict={}
        kTop_freq_words_sorted_dict={}
        
    def save_pickle(self,file_name,dict):
        with open(file_name, 'wb') as handle:
                pickle.dump(dict, handle)
                
    def read_pickle(self,file_name,mode):
        if mode is "coll_stats":
            with open(file_name, 'rb') as handle:
                self.coll_stats_dict = pickle.loads(handle.read())
        elif mode is "num_sen_per_q":
            with open(file_name, 'rb') as handle:
                self.sen_per_query_cnt_dict = pickle.loads(handle.read())
        elif mode is "sorted_words_prob":
            with open(file_name, 'rb') as handle:
                self.words_dict_sorted = pickle.loads(handle.read())
        elif mode is "raw_words_prob":
             with open(file_name, 'rb') as handle:
                self.words_dict = pickle.loads(handle.read())
        elif mode is "total_word_count_per_query":
            with open(file_name, 'rb') as handle:
                self.total_word_count_per_q_dict = pickle.loads(handle.read())
        elif mode is "top_words":
            with open(file_name, 'rb') as handle:
                self.kTop_freq_words_sorted_dict = pickle.loads(handle.read())
            

    def tokenize_sentences(self,input_file,model,wiki_RT):
                try:
                    sen_file = open(input_file,'r')
                    sen_doc = sen_file.read().strip() 
                    self.tokenized_words = {} 
                    stopWords = stopwords.words('english')
                    
                    for i, line in enumerate(sen_doc.split('\n')):
                        print "in line ", i
                        qID=line.split("|")[0]
                        tempWords = nltk.word_tokenize(line.split("|")[1])          
                        if not qID in self.tokenized_words:
                            self.tokenized_words[qID] = [w for w in tempWords if w not in stopWords]
                        else:
                            self.tokenized_words[qID].extend([w for w in tempWords if w not in stopWords])
                    with open("tokenized_sentence_noSW_"+model+"_"+wiki_RT, 'wb') as handle:
                        pickle.dump(self.tokenized_words, handle)            
                except Exception as err: 
                    sys.stderr.write('problem in tokenizeSentences:')     
                    print err.args      
                    print err
                    
    def calc_average_sen_len_wiki(self):
        try:
            """
            1. count number of sentences in every corpus
            2. count average sentence length for wiki collection
            3. have the number of word occurences from Indri index
            """
            total_sen_cnt=0
            total_sen_len=0
            os.chdir(wiki_path)
            for f in glob.glob("*.txt"):
#                 unicodedata.normalize("NFD", file)
                sen_file = open(f, mode='r').read().strip()
#                 sen_file=codecs.open(file,"r",encoding="utf8").read().strip()
#                 sen_file=codecs.open('unicode.rst', encoding='utf-8')
                for i, line in enumerate(sen_file.split('\n')):
                    if "<s>" in repr(line):
                        total_sen_cnt=total_sen_cnt+1
                        total_sen_len=total_sen_len+len(line.split(" "))
            
            self.average_sen_len=float(total_sen_len/total_sen_cnt)
            print self.average_sen_len    
                
        except Exception as err: 
                    sys.stderr.write('problem in collect_corpus_stats:')     
                    print err.args      
                    print err 
                      
    def calc_word_freq_based_on_sentences(self,wiki_RT,model,valid_stemmed_sen_path):
        """
        1.create a prob vector for each sentence in the validated senteces file 
        2.smooth with word's occurence in corpus using collection_stats - Dirichlet
        """
        stop_words = stopwords.words('english')
        avg_sen_len_RT=24; #in the entire corpus, with no regard to ret method
        avg_sen_len_wiki=19
        
        try:
            
            if wiki_RT is "RT":
                myu=avg_sen_len_RT
            else:
                myu=avg_sen_len_wiki
            
            ######read collection stats file for model
#             coll_stats_file=open(coll_stats_f+wiki_RT,'r').read().strip()
#             for i, line in enumerate(coll_stats_file.split('\n')):
#                 self.coll_stats_dict[line.split(":")[0]]=line.split(":")[1] #key is word, value is collection prob 
#             self.save_pickle("collection_stats"+"_"+wiki_RT+"_pickle", self.coll_stats_dict)
            self.read_pickle("collection_stats"+"_"+wiki_RT+"_pickle", "coll_stats")

        ##### count number of sentence per query
            valid_sen_file=open(valid_stemmed_sen_path+wiki_RT+"_"+model,'r').read().strip() 
            for i, line in enumerate(valid_sen_file.split('\n')):
                qID=line.split("|")[0]
                if str(qID) in self.sen_per_query_cnt_dict.keys():
                    self.sen_per_query_cnt_dict[qID]=self.sen_per_query_cnt_dict[qID]+1
                else:
                    self.sen_per_query_cnt_dict[qID]=1
            self.save_pickle("num_sen_per_q"+"_"+model+"_"+wiki_RT+"pickle",self.sen_per_query_cnt_dict)
#             self.read_pickle("num_sen_per_q"+"_"+model+"_"+wiki_RT+"_pickle","num_sen_per_q")
        ####read the stemmed sentences and calc freq   
#             """
            try:
                 
                for sen in valid_sen_file.split('\n'):
                    qID=sen.split("|")[0]
#                     print "qID is :", qID
                    sen_words=sen.split("|")[1].split()
                    denominator=len(sen_words)+myu  
                    for w in sen_words:
                        if w not in stop_words:
                            if w not in self.words_dict.keys():
                                if w in self.coll_stats_dict.keys():
                                    self.words_dict[qID,w]=float((float(sen_words.count(w))+float(self.coll_stats_dict[w]))/denominator)
                                else:
                                    print "did not find: ", w 
                                    self.words_dict[qID,w]=0
                            else:
                                self.words_dict[qID,w]=self.words_dict[w]+(sen_words.count(w)+self.coll_stats_dict[w]/denominator)
            except Exception as err: 
                sys.stderr.write('problem in words_prob:' ,sen,qID )     
                print err.args      
                print err                
            self.save_pickle("raw_words_prob_"+model+"_"+wiki_RT+"_pickle",self.words_dict)
#             """
#             self.read_pickle("raw_words_prob_"+model+"_"+wiki_RT+"_pickle","raw_words_prob")
#             """ 
            #divied the final prob in the num of sentences per query
            for (qID_word,prob) in self.words_dict.items():
                qID=qID_word[0]
                num_of_sen_per_qID=self.sen_per_query_cnt_dict[qID]
                prob=float(prob/num_of_sen_per_qID)
                self.words_dict[qID_word] =  prob
            #count num of words per query , for the top k choosing  
            for (qID_word,prob) in self.words_dict.items():    
                if qID_word[0] in self.total_word_count_per_q_dict.keys():
                    self.total_word_count_per_q_dict[qID_word[0]]=self.total_word_count_per_q_dict[qID_word[0]]+1
                else:
                    self.total_word_count_per_q_dict[qID_word[0]]=1
#             
# 
            self.words_dict_sorted=collections.OrderedDict(sorted(self.words_dict.items() ,key= lambda x: (-int(x[0][0]),float(x[1])),reverse=True))
#             w_top_freq_words=csv.writer(open("standard_freq_words_"+model+"_"+wiki_RT+".csv", "wb"))
#             
            self.save_pickle("sorted_words_prob_"+model+"_"+wiki_RT+"_pickle",self.words_dict_sorted)
            self.save_pickle("total_word_count_per_query_"+model+"_"+wiki_RT+"_pickle", self.total_word_count_per_q_dict)
#             """ 
#             self.words_dict_sorted=collections.OrderedDict(sorted(self.words_dict.items() ,key= lambda x: (-int(x[0][0]),float(x[1])),reverse=True))
            
            w_words_freq=csv.writer(open("standard_freq_words"+model+"_"+wiki_RT+".csv", "wb"))
            for item in self.words_dict_sorted:
                w_words_freq.writerow([item,self.words_dict_sorted[item]])
              
#            select the top 5% words from the prob
#             self.read_pickle("total_word_count_per_query_"+model+"_"+wiki_RT+"_pickle","total_word_count_per_query")
#             self.read_pickle("sorted_words_prob_"+model+"_"+wiki_RT+"_pickle","sorted_words_prob")
              
            for qID in range(1,101):
                if str(qID) in self.sen_per_query_cnt_dict.keys():
                    k=0.05
                    num_sen=max(20,math.ceil((k*self.total_word_count_per_q_dict[str(qID)])))
                    curr_result=[(key,val) for key,val in self.words_dict_sorted.items() if int(key[0]) == qID][0:int(num_sen)]
                    for (key,val) in curr_result:
                        self.kTop_freq_words_dict[key]=val #qID,term and value as prob           
#                     self.kTop_freq_words[key[0]]=int(k*self.total_word_count_per_q_dict[str(qID)])
            self.kTop_freq_words_sorted_dict=collections.OrderedDict(sorted(self.kTop_freq_words_dict.items() ,key= lambda x: (-int(x[0][0]),float(x[1])),reverse=True))
            self.save_pickle("top_word_count_per_query_"+model+"_"+wiki_RT+"_pickle", self.kTop_freq_words_sorted_dict)
#             self.read_pickle("top_word_count_per_query_"+model+"_"+wiki_RT+"_pickle","top_words")
            
            
            w_k_top_words_freq=csv.writer(open("standard_k_top_freq_words"+model+"_"+wiki_RT+".csv", "wb"))
            for item in self.kTop_freq_words_sorted_dict:
                w_k_top_words_freq.writerow([item,self.kTop_freq_words_sorted_dict[item]])
        
            
        except Exception as err: 
                    sys.stderr.write('problem in calc_word_freq_based_on_sentences:' )     
                    print err.args      
                    print err
                    
            #expand the claim with terms and their corresponding weight     
    def expand_claims(self,file,sen_or_doc,model,wiki_RT,q_orig_weight):
        try:
        #read claims file, and for each claim write an expanded claims file with top words in the rel model
            w_expnaded_claims=csv.writer(open("standard_expnaded_claims_"+sen_or_doc+"_"+model+"_"+wiki_RT+".csv", "wb"))
            claims_dict={}
            expanding_terms={}
            curr_terms=[]
#              w_top_freq_words.writerow([item,self.kTop_freq_words[item]])
            claims_file = open(file,'r').read().strip()
            for i, line in enumerate(claims_file.split('\n')):          
                claims_dict[i+1]=(line.split("|")[0],[word.lower() for word in line.split("|")[1].split()],line.split("|")[1]) #first is the movie title. second is the claim itself
            for qID in range(1,101):
                for (qIDword,prob) in self.kTop_freq_words_sorted_dict.items(): #key is qID and Term 
                        if int(qIDword[0]) is qID and qIDword[1] not in claims_dict[qID][1] and len(qIDword[1])> 1 : #if the word is not in the claim
                            if qIDword[0] in expanding_terms.keys():
                                expanding_terms[qIDword[0]].extend([(prob,qIDword[1])])
                            else:
                                expanding_terms[qIDword[0]]=[(prob,qIDword[1])]
#                             
                            

            w_expnaded_claims.writerow(["<parameters>"]) 
            for qID in range(1,101):
                    if sen_or_doc is "doc":
                            if str(qID) in expanding_terms.keys() and len(expanding_terms[str(qID)])> 0 :
                                w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#weight( 2.0 #1("+str(claims_dict[qID][0])+".title) 1.0 #weight ( "+str(1-q_orig_weight)+" #uw("+ str(claims_dict[qID][2])+") "+str(q_orig_weight)+" #combine( #weight( "+' '.join([ "%s %s" % x for x in expanding_terms[str(qID)]])+"))))</text></query>"])
                            else:
                                w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#weight( 2.0 #1("+str(claims_dict[qID][0])+".title) 1.0 #uw("+ str(claims_dict[qID][2])+" ))</text></query>"])         
    #                         w_expnaded_claims.writerow(["<query><number>]"+str([qID])+["</number><text>#weight( 2.0 #1)"]+str([claims_dict[qID][0]])+[".title) 1.0 #weight ( "]+str([1-q_orig_weight])+[" #uw("]+ str([claims_dict[qID][2]])+[") "]+str([q_orig_weight])+r" #combine("+' '.join(expanding_terms[qID])+r")))</text></query>")
                    if sen_or_doc is "sen":
                        if str(qID) in expanding_terms.keys() and len(expanding_terms[str(qID)])> 0:
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#combine[s]( #weight("+str(q_orig_weight)+" #combine("+str(claims_dict[qID][2])+")"+str(1-q_orig_weight) +" #combine( #weight(" +' '.join([ "%s %s" % x for x in expanding_terms[str(qID)]])+"))))</text>"])
                        else:
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#combine[s]( "+str(claims_dict[qID][2])+")</text>"])
            w_expnaded_claims.writerow(["</parameters>"])
                    
        except Exception as err: 
                sys.stderr.write('problem in expand_claims:')     
                print err.args      
                print err                 
                    
def main():
        try:
            model="MANUAL"
            wiki_RT="RT"
            valid_stemmed_sen_path = r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\buidingRM\MANUAL\stemmedSentences_"
            rm=standard_rel_model()
#             rm.calc_word_freq_based_on_sentences(wiki_RT,model,valid_stemmed_sen_path)
            rm.read_pickle("top_word_count_per_query_"+model+"_"+wiki_RT+"_pickle","top_words")
            rm.expand_claims(r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\rawClaim_SW.txt","sen",model,wiki_RT,0.5) 

#             rm.calc_average_sen_len_wiki()
        except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err 
            
if __name__ == '__main__':
    main()