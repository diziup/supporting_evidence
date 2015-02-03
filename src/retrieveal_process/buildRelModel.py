'''
Created on 1.05.14 
Module for building a relevance model from sentences
Input: file of stemmed sentences, for each model type (currently LM, MRF,RM,OKAPI and Manual)
       these are sentences that have stemmed (by the IndriRunQuery_liora program)
       and so every word is as it appears in the index
        for example -stemmedSentencesLM
output -  the MLE prob of every word in the sentences 
@author: Liora
'''
from __future__ import division
import sys
import nltk
import pickle
import collections
import csv
import math
from nltk.corpus import stopwords

class rel_model:
    words_dict={}
    words_count={}
    total_word_count={}
    kTop_freq_words={}
     
    def __init__(self):
        tokenized_words={}
        words_count={}
        total_word_count={}
        kTop_freq_words={}
#     def save_pickle(self,name):
#         with open(name, 'wb') as handle:
#             pickle.dump(self.tokenized_sen_dict, handle)
            
#     def read_pickle(self,name):
#         with open('tokenized_sentence.txt', 'rb') as handle:
#             self.tokenized_words = pickle.loads(handle.read()) 
    def write_top_words_res_file(self,model,wiki_RT):
        w_top_freq_words=csv.writer(open("k_top_freq_words_noSW_"+model+"_"+wiki_RT+".csv", "wb"))
        for item in self.kTop_freq_words:
            w_top_freq_words.writerow([item,self.kTop_freq_words[item]])
                         
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
                          
    def word_count(self,model,wiki_RT):
                try:
#                     with open('tokenized_sentence_LM', 'rb') as handle:
                    with open('tokenized_sentence_noSW_'+model+'_'+wiki_RT, 'rb') as handle:
                        self.tokenized_words = pickle.loads(handle.read())
                    for (qID,t_words) in self.tokenized_words.items():
                        for word in t_words:
                            if (qID,word) in self.words_count:
                                self.words_count[qID,word]+=1
                            else:
                                self.words_count[qID,word]=1
                    with open("words_count_noSW_"+model+"_"+wiki_RT, 'wb') as handle:
                        pickle.dump(self.words_count, handle)
                                 
                except Exception as err: 
                    sys.stderr.write('problem in word_count:')     
                    print err.args      
                    print err 
                    
    def calculate_tot_freq_word(self,model,wiki_RT):
        #calculate total words frequency for query
       
        with open('words_count_noSW_'+model+'_'+wiki_RT, 'rb') as handle:
                        self.words_count = pickle.loads(handle.read())
        for (qID,word) in self.words_count:
            if qID in self.total_word_count.keys():
                self.total_word_count[qID]+=self.words_count[(qID,word)]
            else:
                self.total_word_count[qID]=self.words_count[(qID,word)]
        with open("total_word_count_noSW_"+model+"_"+wiki_RT, 'wb') as handle:
                        pickle.dump(self.total_word_count, handle)
        
    def calculate_top_freq_word(self,model,wiki_RT):
        try:
            with open('words_count_noSW_'+model+'_'+wiki_RT, 'rb') as handle:
                self.words_count = pickle.loads(handle.read()) 
            with open('total_word_count_noSW_'+model+'_'+wiki_RT, 'rb') as handle:
                self.total_word_count = pickle.loads(handle.read())
#             try:   
            for qIDWord in self.words_count:
                self.words_count[qIDWord]=float(self.words_count[qIDWord]/self.total_word_count[qIDWord[0]])
#             except Exception as err:
#                     sys.stderr.write('problem in calculate_top_freq_word:', qIDWoed)   
            self.words_count=collections.OrderedDict(sorted(self.words_count.items(), key= lambda x: (-int(x[0][0]),float(x[1])),reverse=True))
            #choose the 5% most frequent words for each qID
            
            for qID in range(1,101):
                if str(qID) in self.total_word_count.keys():
                    k=int(0.1*self.total_word_count[str(qID)]) #from the entrie list take 10 percent, and from that take 5 percent
                    currResult=[(key,val) for key,val in self.words_count.items() if int(key[0]) == qID][0:k]
                    for (key,val) in currResult:
                        self.kTop_freq_words[key]= val              
                    self.kTop_freq_words[(key[0],"#k#")]=int(0.05*self.total_word_count[str(qID)])
                    
            self.kTop_freq_words = collections.OrderedDict(sorted(self.kTop_freq_words.items(), key= lambda x: (-int(x[0][0]),float(x[1])),reverse=True))             
            with open("kTop_freq_words_NoSW_"+model+"_"+wiki_RT, 'wb') as handle:
                        pickle.dump(self.kTop_freq_words, handle)
            self.write_top_words_res_file(model,wiki_RT)
        except Exception as err: 
                    sys.stderr.write('problem in calculate_top_freq_word:')     
                    print err.args      
                    print err 
   
   
    def expand_claims(self,file,sen_or_doc,model,wiki_RT,q_orig_weight):
        try:
        #read claims file, and for each claim write an expanded claims file with top words in the rel model
            with open('kTop_freq_words_NoSW_'+model+'_'+wiki_RT, 'rb') as handle:
                self.kTop_freq_words = pickle.loads(handle.read()) 
            w_expnaded_claims=csv.writer(open("expnaded_claims_"+sen_or_doc+"_"+model+"_"+wiki_RT+".csv", "wb"))
            claims_dict={}
            expanding_terms={}
            curr_terms=[]
#              w_top_freq_words.writerow([item,self.kTop_freq_words[item]])
            claims_file = open(file,'r').read().strip()
            for i, line in enumerate(claims_file.split('\n')):          
                claims_dict[i+1]=(line.split("|")[0],[word.lower() for word in line.split("|")[1].split()],line.split("|")[1]) #first is the movie title. second is the claim itself
            for qID in range(1,101):
                term_cnt=0;
                for qIDword in self.kTop_freq_words.keys(): #key is qID and Term
                      
                        if int(qIDword[0]) is qID and not (qIDword[1] =="#k#") and qIDword[1] not in claims_dict[qID][1] and  len(qIDword[1])> 1 : #if the word is not in the claim
                            curr_terms.append(qIDword[1])   
                            term_cnt=term_cnt+1
                            if term_cnt > int(self.kTop_freq_words[qIDword[0],"#k#"]) or term_cnt > 20 : #the maximum is 20
                                expanding_terms[qID]=curr_terms[:-1] #take all but the last which break the above condition
                                break
                            
                curr_terms=[]
            w_expnaded_claims.writerow(["<parameters>"]) 
            for qID in range(1,101):
                    if sen_or_doc is "doc":
                        if model is not "OKAPI":
                            if qID in expanding_terms.keys() and len(expanding_terms[qID])> 0 :
                                w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#weight( 2.0 #1("+str(claims_dict[qID][0])+".title) 1.0 #weight ( "+str(1-q_orig_weight)+" #uw("+ str(claims_dict[qID][2])+") "+str(q_orig_weight)+" #combine("+' '.join(expanding_terms[qID])+")))</text></query>"])
                            else:
                                w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#weight( 2.0 #1("+str(claims_dict[qID][0])+".title) 1.0 #uw("+ str(claims_dict[qID][2])+" ))</text></query>"])
                        elif model is "OKAPI":
                            if qID in expanding_terms.keys() and len(expanding_terms[qID])> 0 :
                                w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>"+ str(claims_dict[qID][2])+' '.join(expanding_terms[qID])+"</text></query>"])
                            else:
                                w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>"+ str(claims_dict[qID][2])+"</text></query>"])
                                    
    #                         w_expnaded_claims.writerow(["<query><number>]"+str([qID])+["</number><text>#weight( 2.0 #1)"]+str([claims_dict[qID][0]])+[".title) 1.0 #weight ( "]+str([1-q_orig_weight])+[" #uw("]+ str([claims_dict[qID][2]])+[") "]+str([q_orig_weight])+r" #combine("+' '.join(expanding_terms[qID])+r")))</text></query>")
                    if sen_or_doc is "sen":
                        if qID in expanding_terms.keys() and len(expanding_terms[qID])> 0:
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#combine[s]( #weight("+str(q_orig_weight)+" #combine("+str(claims_dict[qID][2])+")"+str(1-q_orig_weight) +" #combine(" +' '.join(expanding_terms[qID])+")))</text>"])
                        else:
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#combine[s]( "+str(claims_dict[qID][2])+")</text>"])
            w_expnaded_claims.writerow(["</parameters>"])
                    
        except Exception as err: 
                sys.stderr.write('problem in expand_claims:')     
                print err.args      
                print err 
                
            #for MRF model -  for the anna.pl script, i need a file of the expanding terms for each q.
            #for every such exp. terms,i will apply the perl script of Don. Metzler, and then combine it 
            #with the original query (also processed by the MRF script with a 0.5 weight          
    def expanding_temrs_for_MRF(self,wiki_RT,file):
        try:
            w_expnading_terms=csv.writer(open("expnading_terms_MRF"+wiki_RT+".csv", "wb"))
            with open('kTop_freq_words_NoSW_MRF_'+wiki_RT, 'rb') as handle:
                self.kTop_freq_words = pickle.loads(handle.read()) 
            
            claims_dict={}
            expanding_terms={}
            curr_terms=[]
#              w_top_freq_words.writerow([item,self.kTop_freq_words[item]])
            claims_file = open(file,'r').read().strip()
            for i, line in enumerate(claims_file.split('\n')):          
                claims_dict[i+1]=(line.split("|")[0],[word.lower() for word in line.split("|")[1].split()],line.split("|")[1]) #first is the movie title. second is the claim itself
            for qID in range(1,101):
                term_cnt=0;
                for qIDword in self.kTop_freq_words.keys(): #key is qID and Term
                      
                        if int(qIDword[0]) is qID and not (qIDword[1] =="#k#") and qIDword[1] not in claims_dict[qID][1] and  len(qIDword[1])> 1 : #if the word is not in the claim
                            curr_terms.append(qIDword[1])   
                            term_cnt=term_cnt+1
                            if term_cnt > int(self.kTop_freq_words[qIDword[0],"#k#"]) or term_cnt > 20 : #the maximum is 20
                                expanding_terms[qID]=curr_terms[:-1] #take all but the last which break the above condition
                                break
                            
                curr_terms=[]
            for qID in range(1,101):
                if qID in expanding_terms.keys() and len(expanding_terms[qID])> 0 :
#                         w_expnading_terms.writerows([' '.join(expanding_terms[qID])])
                        w_expnading_terms.writerow([("  "+str(qID)+" " +' '.join(expanding_terms[qID]))])

        except Exception as err: 
                sys.stderr.write('problem in expanding_temrs_for_MRF:')     
                print err.args      
                print err      
    
    
    def merge_claim_expandeing_terms_files_MRF(self,sen_doc,wiki_RT):
        try:
            claims_doc=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\buidingRM\MRF\MRFclaims"
            exp_terms_doc=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\buidingRM\MRF\expnading_terms_MRF_RT_mrf"
            claims_file = open(claims_doc,'r').read().strip()
            exp_terms_file = open(exp_terms_doc,'r').read().strip()
            w_expnaded_claims=csv.writer(open("expnaded_claims_"+sen_doc+"_MRF_"+wiki_RT+".csv", "wb"))
            claims_dict={}
            exp_terms_MRFed_dict={}
            q_orig_weight=0.5
            for line in claims_file.split('\n'): #key is qID, value is the  the movie title and MRF'ed claim 
                claims_dict[int(line.split("|")[1])]=[line.split("|")[0],line.split("|")[2]]
            for line in exp_terms_file.split('\n'): #key is qID value is the MRF'ed expending terms
                exp_terms_MRFed_dict[int(line.split("|")[0])]=line.split("|")[1]
            
            w_expnaded_claims.writerow(["<parameters>"])
            for qID in range(1,101):
                    if sen_doc is "doc":
                        if qID in exp_terms_MRFed_dict.keys() and len(exp_terms_MRFed_dict[qID])> 0 : #instead of #uw before + str(claims_dict[qID][1])+") i put #combine because it doesn't work otherwise
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#weight( 2.0 #1("+str(claims_dict[qID][0])+".title) 1.0 #weight ( "+str(1-q_orig_weight)+" #combine("+ str(claims_dict[qID][1])+") "+str(q_orig_weight)+" #combine("+exp_terms_MRFed_dict[qID]+")))</text></query>"])
                        else:
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#weight( 2.0 #1("+str(claims_dict[qID][0])+".title) 1.0 #combine("+ str(claims_dict[qID][1])+" ))</text></query>"])
                    elif sen_doc is "sen":
                        if qID in exp_terms_MRFed_dict.keys() and len(exp_terms_MRFed_dict[qID])> 0:
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#combine[s]( #weight("+str(q_orig_weight)+" #combine("+str(claims_dict[qID][1])+")"+str(1-q_orig_weight) +" #combine(" +exp_terms_MRFed_dict[qID]+")))</text>"])
                        else:
                            w_expnaded_claims.writerow(["<query><number>"+str(qID)+"</number><text>#combine[s]( "+str(claims_dict[qID][1])+")</text>"])
            w_expnaded_claims.writerow(["</parameters>"])
                                   
        except Exception as err: 
                sys.stderr.write('problem in merge_claim_expandeing_terms_files_MRF:')     
                print err.args      
                print err
              
def main():
        try:  
            rel_m=rel_model()
            model="MANUAL"
            wiki_RT="RT"
            inpt_file=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\buidingRM\MANUAL\stemmedSentences_"+wiki_RT+"_"+model
#             rel_m.tokenize_sentences(inpt_file,model,wiki_RT)
#             rel_m.word_count(model,wiki_RT)
#             rel_m.calculate_tot_freq_word(model,wiki_RT)
#             rel_m.calculate_top_freq_word(model,wiki_RT)
#        ##for the MRF
#             rel_m.expanding_temrs_for_MRF(wiki_RT,r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\rawClaim_SW.txt")
#             rel_m.merge_claim_expandeing_terms_files_MRF("sen",wiki_RT)
#        ##for the MRF
            rel_m.expand_claims(r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\rawClaim_SW.txt","sen",model,wiki_RT,0.5) 
        except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err


if __name__ == '__main__':
    main() 