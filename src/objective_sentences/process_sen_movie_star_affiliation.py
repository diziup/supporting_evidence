'''
Created on Sep 14, 2014
create a dict- key is a sentence, value is the movie star affiliation according to the LM derived for each category
@author: liorab
'''
import csv
from my_utils import utils_linux

def process_objective_LM_res():
    
    LM_res_file=r"C:\study\technion\MSc\Thesis\Y!\support_test\objective_LM\LM_res\sen_LM_scores.txt"
    LM_norm_res_file = r"C:\study\technion\MSc\Thesis\Y!\support_test\objective_LM\LM_res\sen_LM_norm_scores.txt"
    sen_movie_star_classification_dict = {} #key is a sentence, value is a number -1,2...5  for the movie star classification according to the highest prob from the LM
    sen_obj_LM_dist_dict = {} # key is a sentence, value is a list of the dist on the LM
     
    with open(LM_res_file,'r') as f_label:
        LM_res_doc = f_label.read().strip()
        for i,line in enumerate(LM_res_doc.splitlines()):# first split of a line is the sentence, the second is an array of 5 numbers -separated with ,
            sen=line.split("|")[0]
            scores_list =[float(score) for score in line.split("|")[1].split("[")[1].split("]")[0].split(",")]
            sen_movie_star_classification_dict[sen] = scores_list.index(max(scores_list)) +1
    
    with open(LM_norm_res_file,'r') as f_dist:
        LM_dist_res_doc = f_dist.read().strip()
        for i,line in enumerate(LM_dist_res_doc.splitlines()):# first split of a line is the sentence, the second is an array of 5 numbers -separated with ,
            sen=line.split("|")[0]
            scores_list = [float(score) for score in line.split("|")[1].split("[")[1].split("]")[0].split(",")]
            sen_obj_LM_dist_dict[sen] = scores_list
    
    #save pickle
    utils_linux.save_pickle("sen_obj_LM_label_dict", sen_movie_star_classification_dict)        
    utils_linux.save_pickle("sen_obj_LM_dist_dict", sen_obj_LM_dist_dict)        
    

                
            

def main():
    process_objective_LM_res()

if __name__ == '__main__':
    main()