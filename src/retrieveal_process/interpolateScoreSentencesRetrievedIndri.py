#!/usr/bin/python
from __future__ import division
import math
import csv
import collections
import sys, traceback
import re
import pickle
from compiler.symbols import Scope


"""
this script takes a retrieval result file of documents, and a retrieval result of sentences , and for matching sentences and docs 
computes the weighted ranking.
"""
# LMPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\qLike\twoIndex_firstTrial"
LMPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_second_wikipedia_all_sections_only\LM"
LMPathExp=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\qLike\twoIndex_firstTrial\standard_rel_model"
# MRFPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MRF"
MRFPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_second_wikipedia_all_sections_only\MRF"
# RMPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\RM"
RMPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_second_wikipedia_all_sections_only\RM"
MRFPathExp=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MRF\expandedWithRelModel"
# OKAPIPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\OKAPI"
OKAPIPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_second_wikipedia_all_sections_only\OKAPI"
OKAPIPathExp=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\OKAPI\expandedWithRelModel"
# MANUALPath=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MANUAL"
MANUALPath = r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_second_wikipedia_all_sections_only\MANUAL"
MANUALPathExp=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MANUAL\standard_rel_model"
# no_sentiment_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\no_senti_LM"
no_sentiment_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_second_wikipedia_all_sections_only\LM_no_senti"


class scoreSentencesIndri:

    doc_file_path=""
    sen_file_path=""
    output_path=""
    output_name=""
    doc_file_path_2=""
    sen_file_path2=""
    total_docs_score={}
    total_sens_score={}
    
    LMdict={}
    MRFdict={}
    RMdict={}
    OKAPIdict={}
    MANUALdict={}
    no_senti_dict={}
    LM_exp_dict={}
    MANUAL_exp_dict={}
    
    tempDict={}
    
    topKLMdict={}
    topKMRFdict={}
    topKRMdict={}
    topKOKAPIdict={}
    topKMANUALdict={}
    topK_no_senti_dict={}
   
    topKLMdictExp={}
    topKMRFdictExp={}
    topKOKAPIdictExp={}
    topKMANUALdictExp={}
    

    validatedSen={}
    sentencesAffiliationDict={}
    
    def __init__(self):
#        self.doc_file_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MRF\MRFDoc_Wiki_res"  
#        self.sen_file_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MRF\MRFSen_wiki_res"
       
        self.output_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MRF"
       
        self.RMdict={}
        self.MRFdict={}
        self.LMdict={}
        self.OKAPIdict={}
        self.MANUALdict={}     

# def main():
    def createRankedSenFile(self,doc_file,sen_file,num):
# extract doc score from doc file
        try:
            doc_file = open(doc_file,'r')
            doc = doc_file.read().strip() # score
            doc_dict = {} # id: score
            for i, line in enumerate(doc.split('\n')):
                data = line.split(' ')
                qId = data[0]
                doc_id = data[2]
                score = data[4]
                if doc_id in doc_dict:
                    raise Exception("DOC ID %s already in dict" % doc_id)
                doc_dict[qId,doc_id] = score
            
            # Extract Sentence file, and create "d"
            sen_file = open(sen_file,'r')
            sen = sen_file.read().strip() # score, sentence
            sen_dict = {} # id: sentence, score1, score2
            try:
                for i, line in enumerate(sen.split('\n')):
                    try:
                        if i%2==0: # a metadata line
                            data = line.split(' ')
                            qId = data[0]
                            doc_id = data[2]
                            sen_score = data[4]       
                        else:
                            if (qId,doc_id) in doc_dict.keys():
                                weighted_score = 0.8*(math.exp(float(doc_dict[qId,doc_id])/self.total_docs_score[int(qId)])) + 0.2*(math.exp(float(sen_score))/self.total_sens_score[int(qId)])
                                sen_dict[qId, line] =(weighted_score,doc_id)
                            else:
                                weighted_score=1*math.exp(float(sen_score))
                                sen_dict[qId, line] =(weighted_score,doc_id)
                    except IOError: 
                        sys.stderr.write('problem in:' , data)          
                    #not working - sen_dict[line].append((weighted_score , doc_id))
            except IOError: 
                sys.stderr.write('problem in:' , i, line)        
            
            #sort the sen_dict by qID first ascending  and then by score descending  
#           order_sen_qID_score = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
            if num==1:
                self.LMdict = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
            elif num==2 :
                self.RMdict = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
            elif num==3:
                self.MRFdict = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
            elif num==4:
                self.OKAPIdict = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
            elif num==5:
                self.MANUALdict = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
            elif num==6:
                self.tempDict = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
            elif num==7:
                self.no_senti_dict = collections.OrderedDict(sorted(sen_dict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True))
                
#             self.saveToCSVFile(self.LMdict,"LM_wiki")
#             self.saveToCSVFile(self.RMdict,"RM_wiki") 
#             self.saveToCSVFile(self.MRFdict,"MRF_wiki") 
#             self.saveToCSVFile(self.OKAPIdict, "OKAPI_RT")
#             self.saveToCSVFile(self.MANUALdict, "MANUAL_wiki")
#             self.saveToCSVFile(self.LMdict, "LM50topSen_wiki")
                           
        #to print just the query ID and the sentence:
        # for k, v in self.orderedSen1:
#                 w.writerow([k,v])                 
        except IOError:
            sys.stderr.write('problem')
    
    
    def topKSentence(self,dict,modelType,wiki_RT):
        k=50
        tempDict={}
        for qID in range(1,101):
            currResult=[(key,val) for key,val in dict.items() if int(key[0]) == qID][0:k]
            for (key,val) in currResult:
                tempDict[key]= val
        if modelType=="LM":
            self.topKLMdict=collections.OrderedDict((sorted(tempDict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True)))      
            self.saveToCSVFile(self.topKLMdict,"top_sen_new"+wiki_RT+"stan_exp_fin_"+modelType)                        
        elif modelType=="MRF":
            self.topKMRFdict=collections.OrderedDict((sorted(tempDict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True)))      
            self.saveToCSVFile(self.topKMRFdict,"top_sen_new"+wiki_RT+"_"+modelType)
        elif modelType=="RM":
            self.topKRMdict=collections.OrderedDict((sorted(tempDict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True)))      
            self.saveToCSVFile(self.topKRMdict,"top_sen_new"+modelType)
        elif modelType=="OKAPI":
            self.topKOKAPIdict=collections.OrderedDict((sorted(tempDict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True)))      
            self.saveToCSVFile(self.topKOKAPIdict,"top_sen_new"+wiki_RT+"_exp_fin_"+modelType)
        elif modelType=="MANUAL":
            self.topKMANUALdict=collections.OrderedDict((sorted(tempDict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True)))      
            self.saveToCSVFile(self.topKMANUALdict,"top_sen_new"+wiki_RT+"_fin_"+modelType)
        elif modelType == "LM_no_senti":
            self.topK_no_senti_dict = collections.OrderedDict((sorted(tempDict.items(), key= lambda x: (-int(x[0][0]),x[1][0]),reverse=True)))
            self.saveToCSVFile(self.topK_no_senti_dict,"top_sen_new"+wiki_RT+"_fin_"+modelType)  

    def calcTotalDocsSensScore(self,doc_file,sen_file):
            self.total_docs_score={}
            self.total_sens_score={}
            try:
                for j in range(1,101):
                    self.total_docs_score[j]=0
                    self.total_sens_score[j]=0
                doc_file = open(doc_file,'r')
                doc = doc_file.read().strip()
                try:
                    for i, line in enumerate(doc.split('\n')):
                        data = line.split(' ')
                        qID=data[0]
                        self.total_docs_score[int(qID)]+=math.exp(float(data[4]))
                except IOError:
                    sys.stderr.write('problem in calcTotalDocsScore for doc in ', i)
           
                sen_file = open(sen_file,'r')
                sen = sen_file.read().strip()
                try:  
                    for i, line in enumerate(sen.split('\n')):
                        if i%2==0: # a metadata line
                                data = line.split(' ')
                                qId = data[0]
                                self.total_sens_score[int(qId)]+=math.exp(float(data[4]))
                except IOError:
                    sys.stderr.write('problem in calcTotalDocsScore for sen in ', i)
                
            except IOError:
                sys.stderr.write('problem in calcTotalDocsScore in ' , doc_file, sen_file)
        

    def saveToCSVFile(self,sen_dict,file_name):
      
        w = csv.writer(open(file_name+"_fin.csv" , "wb"))
#             for k, v in order_sen_qID_score.items():
        for k, v in sen_dict.items():
                w.writerow([k,v])         
    """
    function to compare to files of output from 2 models, and tell the differece between them
    and the intersection =eq between.
    For the eq sentences- check if they are already validated sentences. 
    """  
    def savePickle(self,wikiOrRT,model):
        with open('topSen_'+wikiOrRT+'_new_'+model+'_fin_pickle.txt', 'wb') as handle:
            if model is "LM":
                pickle.dump(self.topKLMdict, handle)
            if model is "MRF":
                pickle.dump(self.topKMRFdict, handle)    
            if model is "RM":
                pickle.dump(self.topKRMdict, handle)     
            if model is "OKAPI":
                pickle.dump(self.topKOKAPIdict, handle)
            if model is "OKAPI":
                pickle.dump(self.topKOKAPIdict, handle)
            if model is "MANUAL":
                pickle.dump(self.topKMANUALdict, handle)
            if model is "LM_no_senti":
                pickle.dump(self.topK_no_senti_dict, handle)
        
   
    def read_pickle(self,wikiOrRT,model):
        with open('topSen_'+wikiOrRT+'_'+model+'_exp_fin_pickle.txt', 'rb') as handle:
            if model is "LM":
                self.topKLMdictExp = pickle.loads(handle.read())
#             if model is "LM":
#                 self.topKLMdictExp = pickle.loads(handle.read())    
            if model is "MRF":
#                 self.topKMRFdict = pickle.loads(handle.read())
#             if model is "MRF_exp":
                self.topKMRFdictExp = pickle.loads(handle.read())   
            if model is "RM":
                self.topKRMdict = pickle.loads(handle.read())
            if model is "OKAPI":
                self.topKOKAPIdictExp = pickle.loads(handle.read())
#             if model is "OKAPI_exp":
#                 self.topKOKAPIdictExp = pickle.loads(handle.read())       
            if model is "MANUAL":
#                 self.topKMANUALdict = pickle.loads(handle.read())
#             if model is "MANUAL_exp":
                self.topKMANUALdictExp = pickle.loads(handle.read())
                #for the original sens
        with open('topSen_'+wikiOrRT+'_'+model+'_fin_pickle.txt', 'rb') as handle:
                if model is "MRF":
                    self.topKMRFdict = pickle.loads(handle.read())
                if model is "OKAPI":
                    self.topKOKAPIdict = pickle.loads(handle.read())
                if model is "LM":
                    self.topKLMdict = pickle.loads(handle.read())
                if model is "MANUAL":
                    self.topKMANUALdict=pickle.loads(handle.read())
#     def read_pickle(self,wikiOrRT,model):
#         with open('topSen_'+wikiOrRT+'_'+model+'_exp_fin_pickle.txt', 'rb') as handle:
#             if model is "LM":
#                 self.topKLMdict = pickle.loads(handle.read())
# #             if model is "LM":
# #                 self.topKLMdictExp = pickle.loads(handle.read())    
#             if model is "MRF":
#                 self.topKMRFdict = pickle.loads(handle.read())
# #             if model is "MRF_exp":
#                 self.topKMRFdictExp = pickle.loads(handle.read())   
#             if model is "RM":
#                 self.topKRMdict = pickle.loads(handle.read())
#             if model is "OKAPI":
#                 self.topKOKAPIdict = pickle.loads(handle.read())
# #             if model is "OKAPI_exp":
#                 self.topKOKAPIdictExp = pickle.loads(handle.read())       
#             if model is "MANUAL":
#                 self.topKMANUALdict = pickle.loads(handle.read())
# #             if model is "MANUAL_exp":
#                 self.topKMANUALdictExp = pickle.loads(handle.read())
                        
    def  compareFiles(self,sourceDict,destDict1,destDict2,destDict3,destDict4,wikiOrRT): 
        #read in validated sentences as dict
        i=1
        try:
            with open('validatedSentences_'+wikiOrRT+'.txt', 'r') as validFile:
                for line in validFile:
                    self.insertIntoDict(re.sub('[\W_]+', '',line.split("|")[0]),re.sub('[\W_]+', '',line.split("|")[1]),self.validatedSen)
                    i+=1
#             with open('eqValid_MANUAL_RM_LM_MRF_OKAPI_wiki.log', 'a+') as fEq, open('diff_MANUAL_RM_LM_MRF_OKAPI_wiki.log', 'a+') as fDiff :
            with open('eqValid_topMAN_all_'+wikiOrRT+'.log', 'a+') as fEq, open('diff_topMAN_all_'+wikiOrRT+'.log', 'a+') as fDiff :
                for i in range (0,len(sourceDict)):
                    wholeLine = sourceDict.keys()[i] #the new model
                    sen = sourceDict.keys()[i][1]
                    qID = sourceDict.keys()[i][0]
                   
    #                     if (qID=='1' and sen=="Kramer vs. Kramer wouldn't be half as good as it is -- half as intriguing and absorbing -- if the movie had taken sides"):
#                     if (wholeLine in self.LMdict.keys() or wholeLine in self.MRFdict.keys() or wholeLine in self.RMdict.keys() or wholeLine in self.OKAPIdict.keys() ): #the old model
                    if (wholeLine in destDict1.keys() or wholeLine in destDict2.keys() or wholeLine in destDict3.keys() or wholeLine in destDict4.keys()):
                        if qID in self.validatedSen.keys() and re.sub('[\W_]+', '',sen) in self.validatedSen[qID]:     
                            fEq.write(str(wholeLine+ sourceDict.values()[i]))
                            fEq.write('\n')
                    else:
                        cleanSen=re.sub('[\W_]+', '',sen)
                        if qID in self.validatedSen.keys() and cleanSen not in self.validatedSen[qID]:
                            fDiff.write(str(wholeLine+ sourceDict.values()[i]))
                            fDiff.write('\n')   
                        
            """ with sets and inetrsection and diff but does not return an sorted by qID list...annoying :/
            keys_new_model = set(self.orderedSen2.keys())
            keys_old_model = set(self.orderedSen1.keys())
            intersection = keys_new_model & keys_old_model
            with open('eqIntersectionSet_MRF_LM_wiki.log', 'a+') as fEq:
                for line in intersection:
                    fEq.write(str(line))
                    fEq.write('\n')
            diff = set(keys_new_model) - set(keys_old_model)
            with open('diffSet_MRF_LM_wiki.log', 'a+') as fDiff:
                for line in diff:
                    fDiff.write(str(line))
                    fDiff.write('\n')
            """
        except Exception:
            sys.stderr.write('problem in:' , i)   
            traceback.print_exc(file=sys.stdout)
#         fDiff.close()
        fEq.close()
        fDiff.close()
        validFile.close()
#

    def insertIntoDict(self,key,value,Dict):
        if not key in Dict:
            Dict[key] = [value]
        else:
            Dict[key].append(value)
     
    
    def sentencesInDictButNotValid(self,sourceDict,wikiOrRT,modelType):
        try:
            i=1
            w = csv.writer(open("forCheckIfValid"+wikiOrRT+".csv", "wb"))
            with open('validatedSentences_'+wikiOrRT+'.txt', 'r') as validFile:
                for line in validFile:
                    self.insertIntoDict(re.sub('[\W_]+', '',line.split("|")[0]),re.sub('[\W_]+', '',line.split("|")[1]),self.validatedSen)
                    i+=1
                for i in range (0,len(sourceDict)):
                    wholeLine = sourceDict.keys()[i] #the new model
                    sen = sourceDict.keys()[i][1]
                    qID = sourceDict.keys()[i][0]
                    if not (qID in self.validatedSen.keys() and re.sub('[\W_]+', '',sen)  in self.validatedSen[qID]):
                        w.writerow([wholeLine,sourceDict[wholeLine]])
     
        except Exception:
            sys.stderr.write('problem in: sentencesInDictButNotValid' )   
            traceback.print_exc(file=sys.stdout)             
        
    def checkDuplicatesInValidFile(self,wikiOrRT):
        validSentences={}
        try:
            w = csv.writer(open("duplicates_"+wikiOrRT+".csv", "wb"))
            with open('validatedSentences_'+wikiOrRT+'.txt', 'r') as validFile:
                for line in validFile:     
                    if (line.split("|")[0],line.split("|")[1]) in validSentences.keys():
                        w.writerow([line.split("|")[0]+"|", line.split("|")[1]])
                    else:
                        validSentences[line.split("|")[0],line.split("|")[1]]=1                    
        except Exception:
            sys.stderr.write('problem in: checkDuplicates' )   
            traceback.print_exc(file=sys.stdout)             
        
        #check the overlap in doc files between models     
            
        except Exception:
            sys.stderr.write('problem in: compareDocFile' )   
            traceback.print_exc(file=sys.stdout)      
    
    def validatedSentnecesModelsAffiliation(self,wikiOrRT,model):
        
        validatedSen_sorted={}
        try:
#             w = csv.writer(open("sentenceModelAffiliation_"+wikiOrRT+".csv", "wb"))
            w = csv.writer(open("sentenceModelAffiliation_"+model+"_"+wikiOrRT+".csv", "wb"))
            try:
                with open('validatedSentences_'+wikiOrRT+'.txt', 'r') as validFile: #open('sentenceModelAffiliation_RT.txt', 'a+') as affilFile:
                    for line in validFile:    
                        self.insertIntoDict((line.split("|")[0],line.split("|")[1]),1,self.validatedSen)    #key is qID,sen value is '1'
#                     with open('validSentences'+wikiOrRT+'_pickle.txt', 'wb') as handle:
#                         pickle.dump(self.validatedSen, handle)
                        
            except Exception:
                sys.stderr.write('problem in: validatedSentnecesModelsAffiliation' , line)  
                                           
            try:
                validatedSen_sorted=collections.OrderedDict(sorted(self.validatedSen.items(), key= lambda x: (-int(x[0][0])),reverse=True))
                for key in validatedSen_sorted.keys():
#                         currModelsAffiliation="" 
                    
                        for keyLM in self.topKMANUALdict.iterkeys(): #(qID,sentence)
                            if keyLM[0]==key[0]:
                                if re.sub('[\W_]+', '',keyLM[1])==re.sub('[\W_]+', '',key[1]):
#                                     currModelsAffiliation+="LM"+","
                                    w.writerow([str(keyLM[0])+'|'+str(keyLM[1])])
#                                 
#                         for keyMRF in self.topKMRFdict.iterkeys():
#                             if keyMRF[0]==key[0]: #same qID
#                                 if re.sub('[\W_]+', '',keyMRF[1])==re.sub('[\W_]+', '',key[1]): #same sentences
#                                     currModelsAffiliation+="MRF"+","
#                                      
#                         for keyRM in self.topKRMdict.keys():
#                             if keyRM[0]==key[0]:
#                                 if re.sub('[\W_]+', '',keyRM[1])==re.sub('[\W_]+', '',key[1]):
#                                     currModelsAffiliation+="RM"+","
#                          
#                         for keyOKAPI in self.topKOKAPIdict.keys():
#                             if keyOKAPI[0]==key[0]:
#                                 if re.sub('[\W_]+', '',keyOKAPI[1])==re.sub('[\W_]+', '',key[1]):
#                                     currModelsAffiliation+="OKAPI"+","
#                                     
#                         for keyMAN in self.topKMANUALdict.keys():
#                             if keyMAN[0]==key[0]:
#                                 if re.sub('[\W_]+', '',keyMAN[1])==re.sub('[\W_]+', '',key[1]):
#                                     currModelsAffiliation+="MANUAL"
                                    
#                         if currModelsAffiliation!="":
#                                     w.writerow([key[0]+"|",key[1].replace('\n', '')+"|",currModelsAffiliation])
#                         else:
#                                     w.writerow([key[0]+"|",key[1].replace('\n', '')+"|","none"])
                        
            except Exception:
                sys.stderr.write('problem in affiliation :' ) 
                traceback.print_exc(file=sys.stdout) 
        except Exception:
            sys.stderr.write('problem in: affiliation' )   
            traceback.print_exc(file=sys.stdout) 
            
    def compareDocSenOverlap(self,wikiOrRT,senOrDocs):
            
            try:
                wAffil = csv.writer(open(senOrDocs+"ModelAffiliation_exp_"+senOrDocs+"_"+wikiOrRT+".csv", "wb"))
                wStats= csv.writer(open(senOrDocs+"ModelAffiliationStats_exp_"+senOrDocs+"_"+wikiOrRT+".csv", "wb"))
                modelsStatisticsDict={}
                aggregatedModelsStatisticsDict={}
                
                if senOrDocs is "doc":
                    #this is the regular files, for the base methods.
#                     files=[(LMPath+"\LM_doc_"+wikiOrRT+"_res",1),(RMPath+"\RM_doc_"+wikiOrRT+"_res",2),(MRFPath+"\MRF_doc_"+wikiOrRT+"_res",3),(OKAPIPath+"\okapi_doc_"+wikiOrRT+"_res",4),(MANUALPath+"\manual_doc_"+wikiOrRT+"_res",5)]
                    #These are the exapnsion results from 4 models (no RM as it doesn't make sense to have real RM on psuedo RM) 
                    files=[(LMPathExp+"\LM_doc_"+wikiOrRT+"_res_exp_fin",1),(MRFPathExp+"\MRF_doc_"+wikiOrRT+"_res_exp_fin",3),(OKAPIPathExp+"\okapi_doc_"+wikiOrRT+"_res_exp_fin",4),(MANUALPathExp+"\manual_doc_"+wikiOrRT+"_res_exp_fin",5)]

#                 files=[(MRFPath+"\MRF_doc_"+wikiOrRT+"_res",3),(RMPath+"\RM_doc_"+wikiOrRT+"_res",2)]
                    LM_doc_dict={}
                    RM_doc_dict={}
                    MRF_doc_dict={}
                    OKAPI_doc_dict={}
                    MAN_doc_dict={}        
                    for f in files:
                        doc_file = open(f[0],'r')
                        doc = doc_file.read().strip() # score
    #                     doc_dict = {} # id: score
                        for i, line in enumerate(doc.split('\n')):
                            data = line.split(' ')
                            qId = data[0]
                            doc_id = data[2]
                            if f[1]==1:
                                self.insertIntoDict((qId,doc_id),1,LM_doc_dict)
                            elif f[1]==2:
                                self.insertIntoDict((qId,doc_id),1,RM_doc_dict)
                            elif f[1]==3:
                                self.insertIntoDict((qId,doc_id),1,MRF_doc_dict)
                            elif f[1]==4:
                                self.insertIntoDict((qId,doc_id),1,OKAPI_doc_dict)
                            elif f[1]==5:
                                self.insertIntoDict((qId,doc_id),1,MAN_doc_dict)
                               
                    set_LM = set(LM_doc_dict)
                    set_RM = set(RM_doc_dict)
                    set_MRF = set(MRF_doc_dict)
                    set_OKAPI = set(OKAPI_doc_dict)
                    set_MAN = set(MAN_doc_dict)
                else: #sens overlap
                    set_LM = set(self.topKLMdict.keys())
                    set_RM =  set(self.topKRMdict.keys())
                    set_MRF = set(self.topKMRFdict.keys())
                    set_OKAPI = set(self.topKOKAPIdict.keys())
                    set_MAN = set(self.topKMANUALdict.keys())
                    
                unionAll= set_LM | set_RM | set_MRF | set_OKAPI | set_MAN
                unionDict= dict.fromkeys(unionAll,0)
                union_sorted = collections.OrderedDict(sorted(unionDict.items(), key=lambda item: (-int(item[0][0])), reverse=True)) #sort by qID
                
                for (q,d) in union_sorted.keys(): #for sentences its q,s
                    currModels=""
                    if (q,d) in set_LM:
                        currModels+="LM"+","
                    if (q,d) in set_RM:
                        currModels+="RM"+","
                    if (q,d) in set_MRF:
                        currModels+="MRF"+","
                    if (q,d) in set_OKAPI:
                        currModels+="OKAPI"+","
                    if (q,d) in set_MAN:
                        currModels+="MAN"+","
                    wAffil.writerow([str((q,d)),currModels])
                    
                    if (q,currModels) not in modelsStatisticsDict.keys():
                        modelsStatisticsDict[(q,currModels)]=[d] #will be s for sentences...
                    else:
                        modelsStatisticsDict[(q,currModels)].append(d)
                
                modelsStatisticsDict = collections.OrderedDict(sorted(modelsStatisticsDict.items(), key=lambda item: (-int(item[0][0])), reverse=True))    
                totalDocsNum=[0]*(101) #list of count of docs or sentences for every query- and we had 100 queries
                for (k,v) in modelsStatisticsDict.items(): # k = (q,currModels), v= doc or sen
                    totalDocsNum[int(k[0])]+=len(v) #k[0] is query, v is list of documents or sentences
                for (k,v) in modelsStatisticsDict.items(): # key is (qID,Models) and value is docnums
                    wStats.writerow([k,float(len(v)/totalDocsNum[int(k[0])])]) 
                    wAffil.writerow([k,v])
                    if k[1] in aggregatedModelsStatisticsDict.keys():#key is models =LM,RM,MAN for instance
                        aggregatedModelsStatisticsDict[k[1]]+=(float(len(v)/totalDocsNum[int(k[0])]))
                    else:
                        aggregatedModelsStatisticsDict[k[1]]=(float(len(v)/totalDocsNum[int(k[0])]))
                wStats.writerow("==============")
                for (currModels,finalCount) in aggregatedModelsStatisticsDict.items():
                    wStats.writerow([currModels,finalCount])
                    
                    
            except Exception:
                sys.stderr.write('problem in: compareDocumentOverlap' )   
                traceback.print_exc(file=sys.stdout) 
 ##calculate total sentences retrieved from each model for each query
    def calculate_total_num_sen_stats(self):
        try: 
            wiki_RT="wiki"
            self.read_pickle(wiki_RT,"LM")
            self.read_pickle(wiki_RT,"MRF")
            self.read_pickle(wiki_RT,"OKAPI")
            self.read_pickle(wiki_RT,"MANUAL")
            
            
            
        except Exception:
                sys.stderr.write('problem in: calculate_total_num_sen_stats' )   
                traceback.print_exc(file=sys.stdout)  
    
    
    #to test the overlap between the expansion method and the base method
    def compare_overlap(self,wikiOrRT,senOrDocs):
            
            try:
                model="MANUAL"
                wAffil = csv.writer(open(senOrDocs+"Compare_orig_exp_"+model+"_"+senOrDocs+"_"+wikiOrRT+".csv", "wb"))
                wStats= csv.writer(open(senOrDocs+"Compare_orig_exp_stats_"+model+"_"+senOrDocs+"_"+wikiOrRT+".csv", "wb"))

                modelsStatisticsDict={}
                aggregatedModelsStatisticsDict={}
                
#                 if senOrDocs is "Docs":
#                 self.read_pickle(wikiOrRT,"MRF_exp")
                self.read_pickle(wikiOrRT,model)
                               
                set_orig = set(self.topKMANUALdict)
                set_exp=set(self.topKMANUALdictExp)
#                 else: #sens overlap
#                     set_LM = set(self.topKLMdict.keys())
#                     set_RM =  set(self.topKRMdict.keys())
#                     set_MRF = set(self.topKMRFdict.keys())
#                     set_OKAPI = set(self.topKOKAPIdict.keys())
#                     set_MAN = set(self.topKMANUALdict.keys())
                    
                unionAll= set_orig | set_exp
                unionDict= dict.fromkeys(unionAll,0)
                union_sorted = collections.OrderedDict(sorted(unionDict.items(), key=lambda item: (-int(item[0][0])), reverse=True)) #sort by qID
                
                for (q,s) in union_sorted.keys(): #for sentences its q,s
                    currModels=""
                    if (q,s) in set_orig:
                        currModels+="orig"+","
                    if (q,s) in set_exp:
                        currModels+="exp"+","
                    
                    wAffil.writerow([str((q,s)),currModels])
                    
                    if (q,currModels) not in modelsStatisticsDict.keys():
                        modelsStatisticsDict[(q,currModels)]=[s] #will be s for sentences...
                    else:
                        modelsStatisticsDict[(q,currModels)].append(s)
                
                modelsStatisticsDict = collections.OrderedDict(sorted(modelsStatisticsDict.items(), key=lambda item: (-int(item[0][0])), reverse=True))    
                totalDocsNum=[0]*(101) #list of count of docs or sentences for every query- and we had 100 queries
                for (k,v) in modelsStatisticsDict.items(): # k = (q,currModels), v= doc or sen
                    totalDocsNum[int(k[0])]+=len(v) #k[0] is query, v is list of documents or sentences
                for (k,v) in modelsStatisticsDict.items(): # key is (qID,Models) and value is docnums
                    wStats.writerow([k,float(len(v)/totalDocsNum[int(k[0])])]) 
                    wAffil.writerow([k,v])
                    if k[1] in aggregatedModelsStatisticsDict.keys():#key is models =LM,RM,MAN for instance
                        aggregatedModelsStatisticsDict[k[1]]+=(float(len(v)/totalDocsNum[int(k[0])]))
                    else:
                        aggregatedModelsStatisticsDict[k[1]]=(float(len(v)/totalDocsNum[int(k[0])]))
                wStats.writerow("==============")
                for (currModels,finalCount) in aggregatedModelsStatisticsDict.items():
                    wStats.writerow([currModels,finalCount])
                    
                    
            except Exception:
                sys.stderr.write('problem in: compareDocumentOverlap' )   
                traceback.print_exc(file=sys.stdout)  
    
    #to compare if the valid sen are in the expanded sentences through all models
    def compare_valid_exp(self,wiki_RT,model):       
             
            with open('validSentences_'+wiki_RT+'_pickle.txt', 'rb') as handle:
                self.validatedSen = pickle.loads(handle.read())
            set_valid = set(self.validatedSen.keys())
            with open('topSen_'+wiki_RT+'_'+model+'_exp_fin_pickle.txt', 'rb') as handle:
                self.topKLMdictExp = pickle.loads(handle.read())
#             set_LM_exp=set( self.topKMRFdictExp.keys())
                set_exp=set( self.topKLMdictExp.keys())
            
            set_diff=set_valid-set_exp
            diff_dict= dict.fromkeys(set_diff,0)
            diff_sorted = collections.OrderedDict(sorted(diff_dict.items(), key=lambda item: (-int(item[0][0])), reverse=True))
            w_compre_exp_valid = csv.writer(open("check_valid_in_exp_"+model+"_"+wiki_RT+".csv", "wb"))
            for item in diff_sorted.items():
                w_compre_exp_valid.writerow([item])
                
            
def main():
    try:
        """
        input: args = wikiOrRT,model,
        """
        scoreIns=scoreSentencesIndri()
        wikiOrRT="wiki"
        model="RM"
    #     scoreIns.compareDocumentOverlap("RT")
            #linux version
        scoreIns.calcTotalDocsSensScore("./RM_doc_"+wikiOrRT+"_new_res","./RM_sen_"+wikiOrRT+"_new_res")
        scoreIns.createRankedSenFile("./RM_doc_"+wikiOrRT+"_new_res","./RM_sen_"+wikiOrRT+"_new_res",2)
        if model is "RM":
            scoreIns.topKSentence(scoreIns.RMdict, model,wikiOrRT)
        elif model is "LM":
            scoreIns.topKSentence(scoreIns.LMdict, model,wikiOrRT)
        elif model is "MRF":
            scoreIns.topKSentence(scoreIns.MRFdict, model,wikiOrRT)
        elif model is "OKAPI":
            scoreIns.topKSentence(scoreIns.OKAPIdict, model,wikiOrRT)
        elif model is "MANUAL":
            scoreIns.topKSentence(scoreIns.MANUALdict, model,wikiOrRT)
        elif model is "LM_Exp":
            scoreIns.topKSentence(scoreIns.LM_exp_dict, model,wikiOrRT)
        elif model is "MAN_exp":
            scoreIns.topKSentence(scoreIns.MANUAL_exp_dict,model,wikiOrRT)
        elif model is "no_sentiment_word":
            scoreIns.topKSentence(scoreIns.no_senti_dict,model,wikiOrRT)
        
        scoreIns.savePickle(wikiOrRT,model)
        """ windows version
        scoreIns.calcTotalDocsSensScore(RMPath+"\RM_doc_"+wikiOrRT+"_new_res",RMPath+"\RM_sen_"+wikiOrRT+"_new_res")
        scoreIns.createRankedSenFile(RMPath+"\RM_doc_"+wikiOrRT+"_new_res",RMPath+"\RM_sen_"+wikiOrRT+"_new_res",2)
        scoreIns.topKSentence(scoreIns.RMdict, model,wikiOrRT)
        scoreIns.savePickle(wikiOrRT,model)
        """
    except IOError:
        sys.stderr.write('problem interpolate main:') 
    ##########for the query expansion - to get the sentences from each model
#     scoreIns.read_pickle(wikiOrRT, model)
#     scoreIns.validatedSentnecesModelsAffiliation(wikiOrRT,model)
    #######
    
#     scoreIns.compare_overlap(wikiOrRT,"sen")
#     scoreIns.compare_overlap(wikiOrRT,"sen")
#     scoreIns.compare_valid_exp(wikiOrRT,"MRF")
    """
    scoreIns.calcTotalDocsSensScore(RMPath+"\RM_doc_"+wikiOrRT+"_res",RMPath+"\RM_sen_"+wikiOrRT+"_res")
    scoreIns.createRankedSenFile(RMPath+"\RM_doc_"+wikiOrRT+"_res",RMPath+"\RM_sen_"+wikiOrRT+"_res",2)
    scoreIns.topKSentence(scoreIns.RMdict, "RM")
    scoreIns.savePickle(wikiOrRT,"RM")
       
    scoreIns.calcTotalDocsSensScore(MRFPath+"\MRF_doc_"+wikiOrRT+"_res",MRFPath+"\MRF_sen_"+wikiOrRT+"_res")
    scoreIns.createRankedSenFile(MRFPath+"\MRF_doc_"+wikiOrRT+"_res",MRFPath+"\MRF_sen_"+wikiOrRT+"_res",3)
    scoreIns.topKSentence(scoreIns.MRFdict, "MRF")
    scoreIns.savePickle(wikiOrRT,"MRF") 
       
    scoreIns.calcTotalDocsSensScore(OKAPIPath+"\okapi_doc_"+wikiOrRT+"_res",OKAPIPath+"\okapi_sen_"+wikiOrRT+"_res")
    scoreIns.createRankedSenFile(OKAPIPath+"\okapi_doc_"+wikiOrRT+"_res",OKAPIPath+"\okapi_sen_"+wikiOrRT+"_res",4)
    scoreIns.topKSentence(scoreIns.OKAPIdict, "OKAPI")
    scoreIns.savePickle(wikiOrRT,"OKAPI") 
    
    scoreIns.calcTotalDocsSensScore(MANUALPath+"\manual_doc_"+wikiOrRT+"_res",MANUALPath+"\manual_sen_"+wikiOrRT+"_res")
    scoreIns.createRankedSenFile(MANUALPath+"\manual_doc_"+wikiOrRT+"_res",MANUALPath+"\manual_sen_"+wikiOrRT+"_res",5)
    scoreIns.topKSentence(scoreIns.MANUALdict, "MANUAL")
    scoreIns.savePickle(wikiOrRT,"MANUAL") 
      """
#     scoreIns.compareFiles(scoreIns.topKMANUALdict,scoreIns.topKLMdict,scoreIns.topKMRFdict,scoreIns.topKRMdict,scoreIns.topKOKAPIdict,"RT")
     
    #for sentences and docs statistics from various models 
#     scoreIns.read_pickle(wikiOrRT,"LM")
# #     scoreIns.read_pickle(wikiOrRT,"RM")
#     scoreIns.read_pickle(wikiOrRT,"MRF")
#     scoreIns.read_pickle(wikiOrRT,"OKAPI")
#     scoreIns.read_pickle(wikiOrRT,"MANUAL")
#     scoreIns.validatedSentnecesModelsAffiliation(wikiOrRT,"LM")
#     scoreIns.compareDocSenOverlap(wikiOrRT,"doc")
    """
    #temp dict for comparing without normalization
#     scoreIns.calcTotalDocsSensScore(r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MANUAL\noNorm\manual_wiki_doc_res",r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MANUAL\noNorm\manual_wiki_sen_res")
#     scoreIns.createRankedSenFile(r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MANUAL\noNorm\manual_wiki_doc_res",r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\MANUAL\noNorm\manual_wiki_sen_res",6)
#     scoreIns.topKSentence(scoreIns.MANUALdict, "MANUAL")
#     
#     scoreIns.compareFiles(scoreIns.topKMANUALdict,scoreIns.tempDict,"RT")

#     scoreIns.sentencesInDictButNotValid(scoreIns.topKLMdict,"wiki","LM")
#     scoreIns.checkDuplicatesInValidFile("RT")
#     scoreIns.calcTotalDocsSensScore(M20RFPath+"\MRFDocSW_RT_res",MRFPath+"\MRF_sen_RT_res")
#     scoreIns.createRankedSenFile(MRFPath+"\MRFDocSW_RT_res",MRFPath+"\MRF_sen_RT_res",3)
#        
#     scoreIns.calcTotalDocsSensScore(RMPath+"\RM_doc_RT_res",RMPath+"\RM_sen_RT_res")
#     scoreIns.createRankedSenFile(RMPath+"\RM_doc_RT_res",RMPath+"\RM_sen_RT_res",2) 
#  
#     scoreIns.calcTotalDocsSensScore(OKAPIPath+"\okapi_doc_RT_res",OKAPIPath+"\okapi_sen_RT_res")
#     scoreIns.createRankedSenFile(OKAPIPath+"\okapi_doc_RT_res",OKAPIPath+"\okapi_sen_RT_res",4)
#     
#     scoreIns.calcTotalDocsSensScore(MANUALPath+"\manual_wiki_doc_res",MANUALPath+"\manual_wiki_sen_res")
#     scoreIns.createRankedSenFile(MANUALPath+"\manual_wiki_doc_res",MANUALPath+"\manual_wiki_sen_res",5)
    
#     scoreIns.compareFiles() #compare btween the new- order 2 and the old -1,3
   
 """   
if __name__ == '__main__':
    import sys
#     main()   
    main(sys.argv[1:])
