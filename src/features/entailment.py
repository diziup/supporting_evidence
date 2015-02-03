'''
Created on 11.09.14
@author: liorab
'''
"""
Entailment feature using BIUTEE

Functions:
1. Generate the RTE xml file for BIUTEE 
2. Process BIUTEE test results
"""
import sys
import csv
from my_utils import utils_linux
import os
import collections


def create_RTE_xml_file():
    #read the clm_text_sen_text_dict files for each claim -  key is the claim text and value is the sens text
    claim_num_and_text=utils_linux.read_pickle('claim_dict_pickle') #key is just a number, values are the claim (first item) and afterwards the sentences
    with open("claim_sentence_entailment_test_input.xml", 'wb') as entail_f:
#         entail_f = csv.writer(csvfile)
        entail_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        entail_f.write('<entailment-corpus>\n')
        pair_cnt=1
        
        for (clm_num,clm_text) in claim_num_and_text.items():
            curr_clm_and_sens=utils_linux.read_pickle('clm_'+clm_num+'_clm_text_sen_text_dict')
            del curr_clm_and_sens[0] 
            for sen in curr_clm_and_sens.values():                
                entail_f.write('<pair id="'+str(pair_cnt)+'" value="ENTAILMENT" task="CD">\n')
                entail_f.write('    <t>'+clm_text+'</t>\n')
                entail_f.write('    <h>'+sen+'</h>\n')
                entail_f.write('</pair>\n')
                pair_cnt+=1
        entail_f.write('</entailment-corpus>')

def process_entailment_results():
    """
    1. read the claim and sentence pickle
    2. read the log files from BIUTEE,
    look for lines with text = ,
    and this line will be the claim,
    the next line is the hypothesis = sentence,
    and the next line is the classification they gave
    3. Calc the accuracy of this feature - how many pairs entail and are support 
    """
    BIUTEE_log_path=r"C:\study\technion\MSc\Thesis\Y!\support_test\entailment\BIUTEE_log"
    entail_clm_sen_pair={} #key is claim and sen, value is the classification score = confidence of the EOP -  if its positive than the classification is true = entailment.
    #else the score is negative
#     clm_sen_support_rank = utils_linux.read_pickle("clm_sen_support_ranking_sorted_full") 
    
    for filename in os.listdir(BIUTEE_log_path):
        claim=""
        sen=""
        classification=""
        score=""
        with open(BIUTEE_log_path+"\\"+filename, 'r') as f:
            for line in f:
                if "Text =" in line:
                    claim = line.split(" = ")[1].split("\n")[0]
                elif "Hypothesis =" in line:
                    sen = line.split(" = ")[1].split("\n")[0]
                elif "Real annotation =" in line:
                    classification = line.split("Classification = ")[1].split(" Score =")[0]
                    score = float(line.split(" Score =")[1])
                    if classification == "false":
                        score = -1*score  
                if claim != None and sen!= None and "true" in classification and score!=None  :
                    entail_clm_sen_pair[(claim,sen)]=(score)
                    claim=""
                    sen=""
                    classification=""
                    score=""
    #sort the clm and sen that are entailment according to the score = the confidence
    entail_clm_sen_pair_sorted = collections.OrderedDict(sorted(entail_clm_sen_pair.items(),key=lambda x: (str (x[0][0]),float(x[1])), reverse=True))
    utils_linux.save_pickle("entailment_clm_sen_pair_sorted", entail_clm_sen_pair_sorted)

def main():
    process_entailment_results()

if __name__ == '__main__':
    main() 