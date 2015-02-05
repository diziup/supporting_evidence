'''
Created on Jul 26, 2014
process support crowdsourcing results
@author: liorab
'''


import sys
import csv
import pickle
import collections
import os
import numpy as np
import pandas as pd
import scipy.stats
from __builtin__ import set, len
from my_utils import utils_linux

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
     
    def process_support_results(self,site):
        
        if site == "M-turk":
            batch="batch4"
            output_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\output_files\batch4"
    #             answers_possibilities=["Support","Neutral","Contradict","Not relevant"]
            answers_col=['Answer.Q1','Answer.Q2','Answer.Q3','Answer.Q4','Answer.Q5','Answer.Q6','Answer.Q7','Answer.Q8',
                    'Answer.Q9','Answer.Q10','Answer.Q11']
            claim_num_batches=[['36'],['39','40','41'],['42','45','46','47','49','50','53','54','55','58','59','60','61','62','66','69','70']]
            answers_possibilities=["Not relevant","Contradict","Neutral","Support"]
            
            total_mat=np.zeros((1, len(answers_possibilities)))
            relevance_mat= np.zeros((1,2)) # 14/09/14 update -  measure the agreement only between non relevant and relevant sentences
    #             for claim_num in ['17','18','20','21','23','26','27','29','36','39','40','41']: #batch 2,
    #         for claim_num in ['36','39','40','41','42','45','46','47','49','50','53','54','55','58','59','60','61','62','66','69','70']:#,'39']:#,]:#,'6','7','8','14','15']: #batch 1#             for claim_num in ['14','15']:
            for claim_batch in claim_num_batches:
                answers_col=['Answer.Q1','Answer.Q2','Answer.Q3','Answer.Q4','Answer.Q5','Answer.Q6','Answer.Q7','Answer.Q8',
                    'Answer.Q9','Answer.Q10','Answer.Q11']
                if claim_num_batches.index(claim_batch) == 0:
                    answers_col.remove('Answer.Q5')
                elif claim_num_batches.index(claim_batch) == 1:
                    answers_col.remove('Answer.Q4')
                elif claim_num_batches.index(claim_batch) == 2:
                    answers_col.remove('Answer.Q10')
                    
                for claim_num in claim_batch:   
                    for filename in os.listdir(output_path):
                        try:
                                
                                curr_cl_num = filename.split("_")[2]
                                if claim_num is curr_cl_num or claim_num == curr_cl_num :
                                    curr_support_mat = np.zeros((10, len(answers_possibilities))) # 10 sentences, 4 rating, was 3 rating for batch 1 and 2
                                    with open(output_path+"\\"+filename, "rb") as f:
                                        data = pd.read_csv(f)
                                        for coll_item in answers_col:
                                            curr_answers = data[coll_item]
                                            for ans in curr_answers:#all 5 subjects
                                                for i  in range(0,len(answers_possibilities)):#21.06
                                                    if ans == answers_possibilities[i]:
                                                        curr_support_mat[answers_col.index(coll_item)][i]+=1
                                                        break
            
                                    total_mat = np.vstack((total_mat,curr_support_mat))
                                    curr_claim_10_sen_kappa=utils_linux.computeKappa(curr_support_mat,filename)                       
                                    if claim_num in self.claim_kappa_dict.keys():
                                            self.claim_kappa_dict[claim_num]+= curr_claim_10_sen_kappa
                                    else:
                                            self.claim_kappa_dict[claim_num]=curr_claim_10_sen_kappa
        
                        except Exception as err: 
                            sys.stderr.write('problem in process_support_results M-turk in file:' +filename )   
                            print err.args      
                            print err
        
        #30.06 add the crowdflower results processing
        elif site is 'crowdflower':
            #need to go over the lines one by one, skip test questions, 
            output_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\output_crowdflower_second_trial"
            full_answers_possibilities=["Not relevant","Strong  Contradiction","Moderate Contradiction","Neutral","Moderate  Support","Strong Support"]
            reduced_answers_possibilities=["Not relevant","Contradiction","Neutral","Support"]
            claims_avg_var_score={}
            
            check="reduced"
            perform_smoothing="False"
            if check is "full":
                answers_possibilities=full_answers_possibilities
            else:
                answers_possibilities=reduced_answers_possibilities
            total_mat=np.zeros((1, len(answers_possibilities))) 
            relevance_mat= np.zeros((1,2)) # 14/09/14 update -  measure the agreement only between non relevant and relevant sentences
            for filename in os.listdir(output_path):
                curr_support_mat=np.zeros((60, len(answers_possibilities))) # 10 sentences, 4 rating, was 3 rating for batch 1 and 2 
                curr_relevance_mat = np.zeros((60, 2)) #first entry is for not relevant sen, second if for relevant (support, contra, neut...)
                
                if filename.split("_")[0]=="f":
                    claim_num=filename.split("_")[1]
                    with open(output_path+"\\"+filename, 'r') as f:
                        data = pd.read_csv(f)
                        answer=data['to_what_extent_does_the_sentence_support_the_claim_or_contradict_it']
                        sentence=data['sen']
                        is_gold=data['orig__golden']      
                        sentence_lines_dict={}# a dict key is sentences, value is list lines it is in
    #                     trick_sen_file=open(self.trick_question_singel_path,'r').read().strip() 
                        for line_num in range(0,len(data)):
                            if is_gold[line_num] != 1 :
                                if sentence[line_num] in sentence_lines_dict.keys():
                                    sentence_lines_dict[sentence[line_num]].append(line_num)
                                else:
                                    sentence_lines_dict[sentence[line_num]]=[line_num]
                        
                        for sen in range(0,len(sentence_lines_dict)):  #go over all the sentences we have and their corresponding line nums- keys
                            for line_num in sentence_lines_dict.values()[sen][0:5]:#limit to 5 annotators
                                if check is "reduced":
                                    ans=answer[line_num]
                                    if ans == "Not relevant":
                                        curr_support_mat[sen][0]+=1
                                        curr_relevance_mat[sen][0]+=1   
                                    elif ans == "Strong  Contradiction" or ans == "Moderate Contradiction":
                                        curr_support_mat[sen][1]+=1
                                        curr_relevance_mat[sen][1]+=1   
                                    elif ans == "Neutral":
                                        curr_support_mat[sen][2]+=1  
                                        curr_relevance_mat[sen][1]+=1   
                                    elif ans == "Strong Support" or ans == "Moderate  Support":
                                        curr_support_mat[sen][3]+=1  
                                        curr_relevance_mat[sen][1]+=1   
                                   
                                elif check is "full":
                                    ans=answer[line_num]
                                    for ans_poss in range(0,len(answers_possibilities)):#21.06
                                        if ans == answers_possibilities[ans_poss]:
                                            curr_support_mat[sen][ans_poss]+=1
                                            break
                        try:
                            self.claim_batch_dict[claim_num]=utils_linux.computeKappa(curr_support_mat,filename)
#                             print ("avg and var in claim_num :"+ claim_num+ ":"+ str(compute_average_variance_rating(curr_support_mat)))
                        except Exception as err: 
                            sys.stderr.write('problem in process_support_results compute kappa in file: '+filename+ " sen: "+sentence[sen])     
                            print err.args      
                            print err 
                    
                    if perform_smoothing is "True":
                        for row in curr_support_mat:
                            majority = np.where(row==4)
                            singleton= np.where(row==1)
                            if len(majority[0]) is not 0:
                                row[majority[0]]+=1
                                row[singleton[0]]-=1
                    
                    claims_avg_var_score[claim_num]=(utils_linux.compute_average_variance_rating(curr_support_mat))
                    total_mat = np.vstack((total_mat,curr_support_mat))
                    relevance_mat =  np.vstack((relevance_mat,curr_relevance_mat))
                            
        total_mat=np.delete(total_mat,(0),axis=0)
        relevance_mat = np.delete(relevance_mat,(0),axis=0)
        np.save("total_ratings_mat.npy", total_mat)
        total_kappa=utils_linux.computeKappa(total_mat,"finish")
        total_relevance_kappa = utils_linux.computeKappa(relevance_mat, "finish")
        [average_rating,var_rating]=utils_linux.compute_average_variance_rating(total_mat)
        kappa_claim_avg=float(float(sum(self.claim_batch_dict.values()))/float(len(self.claim_batch_dict.keys())))
        kappa_claim_var=0
        for kappa in self.claim_batch_dict.values():
            kappa_claim_var+=(kappa-kappa_claim_avg)**2
        kappa_claim_var=float(kappa_claim_var/len(self.claim_batch_dict))
        print ("support total kappa:"+ str(total_kappa)+ ", in check: "+check +", perform smoothing: "+perform_smoothing)
        print (" relevance kappa "+ str(total_relevance_kappa)+ ", in check: "+check +", perform smoothing: "+perform_smoothing)
        print "support average rating: "+ str(average_rating) +"support var rating: "+ str(var_rating)
        print ("kappa_claim_avg: "+str(kappa_claim_avg)+ ", kappa_claim_var:"+str(kappa_claim_var) )
        for (claim_num,kappa) in self.claim_batch_dict.items():
            print ("claim_num: "+str(claim_num)+" kappa: "+str(kappa))
        #kappa avg and variance
        for (claim_num,avg_var) in claims_avg_var_score.items():
            print("claim_num- "+str(claim_num)+": avg: "+str(avg_var[0])+", var: "+str(avg_var[1]))
    
    def categorize_sentence_according_to_annotation(self,site):
                if site is "M-turk":
                    output_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\output_files\batch4"
    #             for claim_num in ['14','15']:   
                    Support_annotated_sen_dict={} #key - a claim, sentencce and the document title. the value -  the times the triple was rated as Contraidct
                    Contradict_annotated_sen_dict={}
                    Neutral_annotated_sen_dict={}
                    Not_relevant_annotated_sen_dict={}
                    answer_col=[('Answer.Q1',1),('Answer.Q2',2),('Answer.Q3',3),('Answer.Q4',4),('Answer.Q5',5),('Answer.Q6',6),('Answer.Q7',7),('Answer.Q8',8)
                                ,('Answer.Q9',9),('Answer.Q10',10),('Answer.Q11',11)]
                            
                    for filename in os.listdir(output_path):
                        try:
                            with open(output_path+"\\"+filename, "rb") as f:
    #                                 reader = csv.reader(f, delimiter=",")
                                    data = pd.read_csv(f)
                                    claim_text=data['Input.claim1'][0]
                                    for coll_item in answer_col:
                                        curr_answers=data[coll_item[0]]
                                        triple=(claim_text,data['Input.sen'+str(coll_item[1])][0],data['Input.tit'+str(coll_item[1])][0])  
                                        for answer in curr_answers:
                                            if answer =='Contradict':
                                                if triple in Contradict_annotated_sen_dict.keys():
                                                    Contradict_annotated_sen_dict[triple]+=1
                                                else:
                                                    Contradict_annotated_sen_dict[triple]=1   
                                            elif answer == "Neutral":
                                                if triple in Neutral_annotated_sen_dict.keys():
                                                    Neutral_annotated_sen_dict[triple]+=1
                                                else:
                                                    Neutral_annotated_sen_dict[triple]=1   
                                            elif answer == "Support":
                                                if triple in Support_annotated_sen_dict.keys():
                                                        Support_annotated_sen_dict[triple]+=1
                                                else:
                                                    Support_annotated_sen_dict[triple]=1 
                                            elif answer == "Not relevant":
                                                if triple in Not_relevant_annotated_sen_dict.keys():
                                                        Not_relevant_annotated_sen_dict[triple]+=1
                                                else:
                                                    Not_relevant_annotated_sen_dict[triple]=1 
    
                        except Exception as err: 
                            sys.stderr.write('problem in categorize_sentence_according_to_annotation')   
                            print err.args      
                            print err
                            
                elif site is "crowdflower":
                    Strong_Support_annotated_sen_dict={} #key - a claim, sentence and the document title. the value -  the number of annotators that rated it as Contradict etc
                    Moderate_support_annotated_sen_dict={}
                    Strong_Contradict_annotated_sen_dict={}
                    Moderate_Contradict_annotated_sen_dict={}
                    Neutral_annotated_sen_dict={}
                    Not_relevant_annotated_sen_dict={}
                    annotation_dict_list=[Not_relevant_annotated_sen_dict,Strong_Contradict_annotated_sen_dict,
                                          Moderate_Contradict_annotated_sen_dict, Neutral_annotated_sen_dict,
                                          Moderate_support_annotated_sen_dict,Strong_Support_annotated_sen_dict]
                    output_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\output_crowdflower_second_trial"
                    answers_possibilities=["Not relevant","Strong  Contradiction","Moderate Contradiction","Neutral","Moderate  Support","Strong Support"]
                            
                    for filename in os.listdir(output_path):
                        if filename.split("_")[0]=="f":
                            claim_num = filename.split("_")[1]
                            with open(output_path+"\\"+filename, 'r') as f:
                                data = pd.read_csv(f)
                                answer=data['to_what_extent_does_the_sentence_support_the_claim_or_contradict_it']
                                sentence=data['sen']
                                doc_title=data['tit']
                                is_gold=data['orig__golden']
                                claim_text=data['claim'][0]      
                                sentence_lines_dict={}# a dict key is sentences, value is list lines it is in
            #                     trick_sen_file=open(self.trick_question_singel_path,'r').read().strip() 
                                for line_num in range(0,len(data)):
                                    if is_gold[line_num] != 1 :
                                        if sentence[line_num] in sentence_lines_dict.keys():
                                            sentence_lines_dict[sentence[line_num]].append(line_num)
                                        else:
                                            sentence_lines_dict[sentence[line_num]]=[line_num]
                                #go over the sentence and answers
                                for sen in range(0,len(sentence_lines_dict)):  #go over all the sentences we have and their corresponding line nums- keys
                                    for line_num in sentence_lines_dict.values()[sen]:
                                        ans = answer[line_num]
                                        for poss_ans in answers_possibilities:
                                            if ans == poss_ans:
                                                triple = (claim_text,sentence[line_num],doc_title[line_num])
                                                if triple in annotation_dict_list[answers_possibilities.index(ans)]:
                                                    annotation_dict_list[answers_possibilities.index(ans)][triple]+=1
                                                else:
                                                    annotation_dict_list[answers_possibilities.index(ans)][triple]=1
                        #group by claim and then order by num of raters                                 
#                         Strong_Support_annotated_sen_sorted=collections.OrderedDict(sorted(Strong_Support_annotated_sen_dict.items(),key=lambda x: (str(x[0][0]),x[1]), reverse=True))                                                                                        
#                         Moderate_support_annotated_sen_dict_sorted=collections.OrderedDict(sorted(Moderate_support_annotated_sen_dict.items(),key=lambda x: (str(x[0][0]),x[1]), reverse=True))
#                         Strong_Contradict_annotated_sen_dict_sorted=collections.OrderedDict(sorted(Strong_Contradict_annotated_sen_dict.items(),key=lambda x: (str(x[0][0]),x[1]), reverse=True))
#                         Moderate_Contradict_annotated_sen_dict_sorted=collections.OrderedDict(sorted(Moderate_Contradict_annotated_sen_dict.items(),key=lambda x: (str(x[0][0]),x[1]), reverse=True))
#                         Neutral_annotated_sen_sorted=collections.OrderedDict(sorted(Neutral_annotated_sen_dict.items(),key=lambda x: (str(x[0][0]),x[1]), reverse=True))         
#                         Not_relevant_annotated_sen_sorted=collections.OrderedDict(sorted(Not_relevant_annotated_sen_dict.items(),key=lambda x: (str(x[0][0]),x[1]), reverse=True))
                        
                        #order by num of raters
                        Strong_Support_annotated_sen_sorted=collections.OrderedDict(sorted(Strong_Support_annotated_sen_dict.items(),key=lambda x: (x[1]), reverse=True))                                                                                        
                        Moderate_support_annotated_sen_dict_sorted=collections.OrderedDict(sorted(Moderate_support_annotated_sen_dict.items(),key=lambda x: (x[1]), reverse=True))
                        Strong_Contradict_annotated_sen_dict_sorted=collections.OrderedDict(sorted(Strong_Contradict_annotated_sen_dict.items(),key=lambda x: (x[1]), reverse=True))
                        Moderate_Contradict_annotated_sen_dict_sorted=collections.OrderedDict(sorted(Moderate_Contradict_annotated_sen_dict.items(),key=lambda x: (x[1]), reverse=True))
                        Neutral_annotated_sen_sorted=collections.OrderedDict(sorted(Neutral_annotated_sen_dict.items(),key=lambda x: (x[1]), reverse=True))         
                        Not_relevant_annotated_sen_sorted=collections.OrderedDict(sorted(Not_relevant_annotated_sen_dict.items(),key=lambda x: (x[1]), reverse=True))
                        
                        
                        with open("Strong_Support_annotated_sen_sorted.csv", "wb") as csvfile:
                                csvfile_support= csv.writer(csvfile)      
                                for ((claim,sen,doc_title),num) in Strong_Support_annotated_sen_sorted.items():
                                        l = []
                                        l.append('%s|%s|%s|%d' %(claim, sen,doc_title,num))
                                        csvfile_support.writerow(l)
                        utils_linux.save_pickle("Strong_Support_annotated_sen_sorted", Strong_Support_annotated_sen_sorted)
                        
                        with open("Moderate_Support_annotated_sen_sorted.csv", "wb") as csvfile:
                                csvfile_support= csv.writer(csvfile)      
                                for ((claim,sen,doc_title),num) in Moderate_support_annotated_sen_dict_sorted.items():
                                        l = []
                                        l.append('%s|%s|%s|%d' %(claim, sen,doc_title,num))
                                        csvfile_support.writerow(l)
                        utils_linux.save_pickle("Moderate_Support_annotated_sen_sorted", Moderate_support_annotated_sen_dict_sorted)
                        
                        with open("Strong_Contradict_annotated_sen_sorted.csv", "wb") as csvfile:
                                csvfile_contradict= csv.writer(csvfile)      
                                for ((claim,sen,doc_title),num) in Strong_Contradict_annotated_sen_dict_sorted.items():
                                        l = []
                                        l.append('%s|%s|%s|%d' %(claim, sen,doc_title,num))
                                        csvfile_contradict.writerow(l)
                        utils_linux.save_pickle("Strong_Contradict_annotated_sen_sorted", Strong_Contradict_annotated_sen_dict_sorted)
                        
                        with open("Moderate_Contradict_annotated_sen_sorted.csv", "wb") as csvfile:
                                csvfile_contradict= csv.writer(csvfile)      
                                for ((claim,sen,doc_title),num) in Moderate_Contradict_annotated_sen_dict_sorted.items():
                                        l = []
                                        l.append('%s|%s|%s|%d' %(claim, sen,doc_title,num))
                                        csvfile_contradict.writerow(l)
                        utils_linux.save_pickle("Moderate_Contradict_annotated_sen_sorted", Moderate_Contradict_annotated_sen_dict_sorted)
                        
                        with open("Neutral_annotated_sen_sorted.csv", "wb") as csvfile:
                                csvfile_neutral= csv.writer(csvfile)      
                                for ((claim,sen,doc_title),num) in Neutral_annotated_sen_sorted.items():
                                        l = []
                                        l.append('%s|%s|%s|%d' %(claim, sen,doc_title,num))
                                        csvfile_neutral.writerow(l)
                        utils_linux.save_pickle("Neutral_annotated_sen_sorted", Neutral_annotated_sen_sorted)
                        
                        with open("Non_relevant_annotated_sen_sorted.csv", "wb") as csvfile:
                                csvfile_not_relevant= csv.writer(csvfile)      
                                for ((claim,sen,doc_title),num) in Not_relevant_annotated_sen_sorted.items():
                                        l = []
                                        l.append('%s|%s|%s|%d' %(claim, sen,doc_title,num))
                                        csvfile_not_relevant.writerow(l)      
                        utils_linux.save_pickle("Non_relevant_annotated_sen_sorted", Not_relevant_annotated_sen_sorted)
     
    def compute_spearman_crowdflower(self): #did not finish ask Idan on what is best
        files_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\support_test\output_crowdflower_second_trial"
        label_possibilities=["Not relevant","Strong  Contradiction","Moderate Contradiction","Neutral","Moderate  Support","Strong Support"]
        spearman_avg_across_files=[]
        total_spearman_avg=0
        
        for filename in os.listdir(files_path):
                with open(files_path+"\\"+filename, 'r') as f:
                    data = pd.read_csv(f)
                    answers=data['to_what_extent_does_the_sentence_support_the_claim_or_contradict_it']
                    sentence=data['sen']
                    is_gold=data['orig__golden']
                    tainted=data['_tainted']
                    workers=data['_worker_id']  
                    workers_lines_dict={}# a dict key is worker_id, value is list of num lines it has
                    workers_annotation_list_dict={} #
    
                    for line_num in range(0,len(data)):
                        if is_gold[line_num] != 1 and tainted[line_num] ==0:
                            if workers[line_num] in workers_lines_dict.keys():
                                workers_lines_dict[workers[line_num]].append(line_num)
                            else:
                                workers_lines_dict[workers[line_num]]=[line_num]
                    
                    for (worker,list_of_line_num) in workers_lines_dict.items():
                        curr_wroker_ans_num=[]
                        for line_num in list_of_line_num:
                            for label in label_possibilities:
                                if answers[line_num] == label:
                                    curr_wroker_ans_num.append(label_possibilities.index(label)+1)  
                        workers_annotation_list_dict[worker]=curr_wroker_ans_num
                    #spearman matrix of the workers
                    #create a martrix in the of size num of working annotators
                    spearman_between_single_annotator_mat = np.zeros((len(workers_annotation_list_dict), len(workers_annotation_list_dict)))
                    for i_worker in range(0,len(workers_annotation_list_dict)):
                        curr_worker_answer_list = workers_annotation_list_dict.values()[i_worker]
                        curr_worker_id_x = workers_annotation_list_dict.keys()[i_worker][0]
                        for j_worker in range (0,len(workers_annotation_list_dict)):
                            if i_worker !=j_worker:
                                curr_worker_id_y = workers_annotation_list_dict.keys()[j_worker][0]
                                curr_spearman_res = scipy.stats.spearmanr(curr_worker_answer_list, workers_annotation_list_dict.values()[j_worker])
                                spearman_between_single_annotator_mat[i_worker,j_worker]=curr_spearman_res[0]
                        else:
                            continue
                    #sum the matrix rows for the mean
                    np.save("support_spearman_between_single_annotator_mat_crowdflower.npy", spearman_between_single_annotator_mat)
                
                    spearman_between_single_annotator_mat= np.load("support_spearman_between_single_annotator_mat_M_turk.npy")
                    sum_of_each_row =np.sum(spearman_between_single_annotator_mat, axis=0)
                    avg_spearman_annotator=sum_of_each_row/(spearman_between_single_annotator_mat.shape[0]-1) #because of the alachson   
                    avg_spearman_per_file = float(sum(avg_spearman_annotator)/(spearman_between_single_annotator_mat.shape[0]))
                    spearman_avg_across_files.append(avg_spearman_per_file)
          
    #total spearman avg across of files:
        total_spearman_avg=float(float(sum(spearman_avg_across_files))/float(len(spearman_avg_across_files)))
        print ("support total_spearman_avg: ",total_spearman_avg)
            

def main():
        try:
            site="crowdflower"
            claim_sen_file=support_sen_files()
            
            claim_sen_file.process_support_results(site)
            claim_sen_file.categorize_sentence_according_to_annotation(site) 
#             claim_sen_file.compute_spearman_crowdflower()
        except Exception as err: 
                sys.stderr.write('problem in main:')     
                print err.args      
                print err             
            
            
if __name__ == '__main__':
  main()          
            
            
            
            
               