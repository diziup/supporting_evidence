'''
Created on Dec 14, 2014

@author: liorab
'''
import time
import os

def apply_IndriRunQuery_simClaimEntity_titleBody():
    
    print "started:"+time.strftime("%Y-%m-%d %H:%M")
#     claim_num_list = [7]#@,17,21,36,37,39]#[40,41,42,45,46,47,50]#51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]"
    claim_num_list =  [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
    for claim_num in claim_num_list:
        claim_file = "./sentence_ret_input/sen_ret_top_k_docs_50_clm_"+str(claim_num)
#                 command = "./IndriRunQuerySmoothingFields_liora docRetBaselineClaimLMdocLM_index_param " +claim_file +" "+str(alpha_f)+" " +str(beta_f)
        command = "./IndriRunQuery_simClaimEntity_titleBody docRetBaselineClaimLMdocLM_index_param "+claim_file
        print "command:"+command
        os.system(command)
    print "finished:"+time.strftime("%Y-%m-%d %H:%M")
    
def apply_IndriRunQuery_simClaimEntitySen():
    
    print "started:"+time.strftime("%Y-%m-%d %H:%M")
    claim_num_list =  [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
    for claim_num in claim_num_list:
        claim_file = "./sentence_ret_input/sen_ret_top_k_docs_50_clm_"+str(claim_num)
#                 command = "./IndriRunQuerySmoothingFields_liora docRetBaselineClaimLMdocLM_index_param " +claim_file +" "+str(alpha_f)+" " +str(beta_f)
        command = "./IndriRunQuery_simClaimEntitySen docRetBaselineClaimLMdocLM_index_param "+claim_file
        print "command:"+command
        os.system(command)
    print "finished:"+time.strftime("%Y-%m-%d %H:%M")