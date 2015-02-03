'''
Created on Sep 16, 2014
have a feature that will be True =1 if the entity name in the claim (movie name for instance),
for each claim and sentence pair, if the entity name 
appears in the sentence or in the document title from which the sentence is from 
@author: liorab
'''

import os
import pandas as pd
from my_utils import utils_linux
import csv

def entity_name_presence():
    """
    1. From the file in the input files for Crowdflower, and the rawClaims.txt file,
    create a dict - key is clm and sen, value is (target entity name and the doc title from which the sen came from
    """
    input_file_crowdflower = r"C:\study\technion\MSc\Thesis\Y!\support_test\input_crowdflower_second_trial" 
    raw_claims_path =  r"C:\study\technion\MSc\Thesis\Y!\rawClaim_SW.txt"
    clm_text_movie_name_dict ={}
    clm_sen_doc_title_entity_presence_flag = {} #key is a clm and sen, value is a tuple of the movie entity name and the name of the movie of the doc title of the sen
    
    raw_claims_file = open(raw_claims_path,'r').read().strip() 
    for i, line in enumerate(raw_claims_file.split('\n')):
        movie_name = line.split("|")[0]
        claim = line.split("|")[1]
        clm_text_movie_name_dict[claim] = movie_name
    
    for filename in os.listdir(input_file_crowdflower):
#         if filename == "supp_claim_55.csv":
            with open(input_file_crowdflower+"\\"+filename, "rb") as f:
                data = pd.read_csv(f)
                claim = data['claim']
                sens =  data['sen']
                doc_title = data['tit']
                is_golden = data['_golden']
                
                for i in range(0,len(claim)):
                    entity_name_presence_flag = 0 
                    remove_chars = ["(",")","'"]
                    replace_with_space_chars = ["-","_"]
                    if is_golden[i] != 1:
                        lowered_edited_sen =""
                        for word in sens[i].split():
                            lowered_edited_sen += " "+word.lower()
                        for ch in remove_chars:
                            if ch in doc_title[i]:
                                doc_title[i] = doc_title[i].replace(ch, "")
                            if ch in lowered_edited_sen:
                                lowered_edited_sen = lowered_edited_sen.replace(ch, "")
                        for ch_rep in replace_with_space_chars:
                            if ch_rep in doc_title[i]:
                                doc_title[i] = doc_title[i].replace(ch, " ")
                            if ch_rep in lowered_edited_sen:
                                lowered_edited_sen = lowered_edited_sen.replace(ch, " ")
                        #lower the sen as well
                        
                        
                        
                        if clm_text_movie_name_dict[claim[i]].lower() in sens[i] or clm_text_movie_name_dict[claim[i]] in sens[i] or clm_text_movie_name_dict[claim[i]].lower() in lowered_edited_sen or clm_text_movie_name_dict[claim[i]].upper() in sens[i] or clm_text_movie_name_dict[claim[i]].lower() in doc_title[i].lower()or clm_text_movie_name_dict[claim[i]].upper() in doc_title[i].upper():
                            entity_name_presence_flag = 1
                        clm_sen_doc_title_entity_presence_flag[claim[i],sens[i]] = (doc_title[i],entity_name_presence_flag)
                
    utils_linux.save_pickle("clm_text_movie_name_dict", clm_text_movie_name_dict)
    utils_linux.save_pickle("clm_sen_doc_title_entity_presence_flag", clm_sen_doc_title_entity_presence_flag)   
    with open("clm_sen_doc_title_entity_presence_flag.csv", 'wb') as csvfile:
            entity_flag_file = csv.writer(csvfile)
            for ((clm,sen),(doc_tit,flag)) in clm_sen_doc_title_entity_presence_flag.items():
                entity_flag_file.writerow([clm,sen,doc_tit,str(flag)])
    
           
def main():
    entity_name_presence()

if __name__ == '__main__':
    main() 
