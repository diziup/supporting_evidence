'''
for alpha range, beta range and for every query - apply the IndriRunQUery script
command format :
IndriRunQuery_exe retrieval_param_file query file alpha beta
'''
import sys
import os
import shlex
import subprocess
import time

corpus_beta_int = 5
def apply_IndriRunQuery_claimLMdocLM_doc_retrieval():
    
    print "started:"+time.strftime("%Y-%m-%d %H:%M")
#     claim_num_list = [7]#@,17,21,36,37,39]#[40,41,42,45,46,47,50]#51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]"
    claim_num_list =  [51]
    for claim_num in claim_num_list:
        for alpha in range(7,11,1):
            for beta in range(9,10,1):
#                 alpha = 0.0
#                 beta = 0.0 
                alpha_f = float(float(alpha)/float(10))
                beta_f = float(float(beta)/float(10))
                claim_file = "./claimLM_docLM_doc_ret_input/LMdocLM_doc_ret_query_file_clm_"+str(claim_num)
#                 command = "./IndriRunQuerySmoothingFields_liora docRetBaselineClaimLMdocLM_index_param " +claim_file +" "+str(alpha_f)+" " +str(beta_f)
                command = "./IndriRunQuery_claimLMdocLM_liora docRetBaselineClaimLMdocLM_index_param "+claim_file+" "+str(alpha_f)+" " +str(beta_f)
                print "command:"+command
                os.system(command)
    print "finished:"+time.strftime("%Y-%m-%d %H:%M")

    
def apply_IndriRunQuery_claimLMAdjacent_senLM_sen_retrieval():
    
    print "started:"+time.strftime("%Y-%m-%d %H:%M")
#     claim_num_list = [59]#@,17,21,36,37,39]#[40,41,42,45,46,47,50]#51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]"
    claim_num_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
    top_k_docs_vals = [50]
    for top_k_docs in top_k_docs_vals:
        for claim_num in claim_num_list:
            for alpha in range(0,11,1):
                for beta in range(0,10,1):
                    for delta_1 in range(0,11-corpus_beta_int,1): #was until 10
                        for delta_2 in range(0,11-corpus_beta_int,1):#was until 10
                            if not delta_1+delta_2 > (10-corpus_beta_int):
                                (alpha_f,beta_f,delta_1_f,delta_2_f) = turn_to_float([alpha,beta,delta_1,delta_2])            
                                claim_file = "./claimLM_senLM_sen_ret_input/claimLM_senLM_sen_ret_docno_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(top_k_docs)+"_clm_"+str(claim_num)
                #                 command = "./IndriRunQuerySmoothingFields_liora docRetBaselineClaimLMdocLM_index_param " +claim_file +" "+str(alpha_f)+" " +str(beta_f)
                                command = "./IndriRunQuery_sentence_QL_neighboringSentenceLM_liora docRetBaselineClaimLMdocLM_index_param "+claim_file+" "+str(alpha_f)+" "+str(beta_f)+" "+str(top_k_docs) +" "+str(delta_1_f) +" "+str(delta_2_f) 
            #                     print "command:"+c
                                os.system(command)
    print "finished:"+time.strftime("%Y-%m-%d %H:%M")
    
def apply_IndriRunQuery_claimLMsenLM_sen_retrieval():
    
    print "started:"+time.strftime("%Y-%m-%d %H:%M")
#     claim_num_list = [59]#@,17,21,36,37,39]#[40,41,42,45,46,47,50]#51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]"
    claim_num_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
    top_k_docs_vals = [50,100]
    for top_k_docs in top_k_docs_vals:
        for claim_num in claim_num_list:
            for alpha in range(0,11,1):
                for beta in range(0,11-corpus_beta_int,1):
                    (alpha_f,beta_f) = turn_to_float([alpha,beta])
                    claim_file = "./claimLM_senLM_sen_ret_input/claimLM_senLM_sen_ret_docno_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(top_k_docs)+"_clm_"+str(claim_num)
    #                 command = "./IndriRunQuerySmoothingFields_liora docRetBaselineClaimLMdocLM_index_param " +claim_file +" "+str(alpha_f)+" " +str(beta_f)
                    command = "./IndriRunQuery_sentence_QL_corpus_smoothing_liora docRetBaselineClaimLMdocLM_index_param "+claim_file+" "+str(alpha_f)+" "+str(beta_f)+" "+str(top_k_docs)
#                     print "command:"+command
                    os.system(command)
    print "finished:"+time.strftime("%Y-%m-%d %H:%M")

def turn_to_float(my_input):
    output = []
    if len(my_input)>1:
        for number in my_input:
            if number == 0:
                number_f = number
            elif number == 10:
                number_f = 1
            else:
                number_f = float(float(number)/float(10))
            output.append(number_f)
        return output
    else:
        my_input_f = 0
        if my_input[0] == 0:
                my_input_f = my_input[0]
        elif my_input[0] == 10:
                my_input_f = 1
        else:
            my_input_f = float(float(my_input[0])/float(10))
        return my_input_f
