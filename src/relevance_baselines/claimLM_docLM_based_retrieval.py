'''
18.10.14
based on the retrieveal of a claim as represented by the claim and the entity,
and representend by the document body and title,
with alpha and beta weighting.
@author: liorab
'''
import sys
import math
try:
    import cPickle as pickle
except:
    import pickle
import collections
import os.path
from my_utils import utils_linux
import csv
import string
# from collections import namedtuple
from my_utils import rcdtype
from collections import namedtuple

# base_path = r"/home/liorab/softwares/indri-5.5/retrieval_baselines/"
base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\baseline_clmLMdocLM"
linux_base_path = r"/home/liorab/softwares/indri-5.5/retrieval_baselines"
curr_source = "wiki"
claim_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
setup = "_adjacent_sen" #or "_corpus_smoothing
corpus_beta = 0.1
corpus_beta_int = int(10*corpus_beta)


def save_pickle(file_name,d):
        with open(file_name, 'wb') as handle:
            pickle.dump(d, handle)

def read_pickle(file_name):
    d={}
    with open(file_name, 'rb') as handle:
            d = pickle.loads(handle.read()) 
    return d

def create_entity_claim_input_file_doc_ret():
    """
    for the input for indri document retrieval
    when all the claim and entity have been lower cased
    """
    claim_doc = open(r"C:\study\technion\MSc\Thesis\Y!\rawClaim_SW.txt").read().strip()
    "remove the stop words from the claims"
    SW_doc = r"C:\study\technion\MSc\Thesis\Y!\stopWords.xml"
    stopWords_list = []
    claims_no_SW_dict = {}
    with open(SW_doc, 'r') as f:
        line = f.readline()
        while line !="":
            if "<word>" in line:
                stopWords_list.append(line.split("<word>")[1].split("</word>")[0])
            line = f.readline()
    
    for i,line in enumerate(claim_doc.split("\n")):
        clmLMdocLM_doc_ret_query_file = open("LMdocLM_doc_ret_query_file_clm_"+str(i+1),"wb")
        clmLMdocLM_doc_ret_query_file.write("<parameters>\n")
        curr_claim_words = line.split("|")[1].lower().split()
        curr_entity_words = line.split("|")[0].lower().split()
        noSW_claim = ""
        noSW_entity = ""
        for word in curr_claim_words:
            if word not in stopWords_list:      
                noSW_claim += word+" "
        for word in curr_entity_words:
            if word not in stopWords_list:      
                noSW_entity += word+" "
#         clmLMdocLM_doc_ret_query_file.write("<query><number>"+str(i+1)+"</number><text>"+noSW_entity+"|"+noSW_claim+"</text></query>\n")
#         clmLMdocLM_doc_ret_query_file.write("</parameters>")
#         clmLMdocLM_doc_ret_query_file.close()
        claims_no_SW_dict[str(i+1)] = (noSW_entity,noSW_claim)
    save_pickle("claims_no_SW_dict", claims_no_SW_dict)
       
def normalize_doc_scores():
    """
    from the documents retrieved,
    score= exp(score)
    and then normalize according to sum normalization
    file is an input file - claim num, alpha, beta
    for each alpha beta, create and than add to it the different claims' doc scores
    """
#     doc_res_files_path = base_path+r"claimLM_docLM_doc_ret_output"
    claims_file_counters_dict = {} #for each claim numas key, have the val a counter - if not 110 per claim -> problem!
    doc_res_files_path = linux_base_path+"/claimLM_docLM_doc_ret_output"
#     doc_res_files_path = base_path +"\\claimLM_docLM_doc_ret_output"
    for filename in os.listdir(doc_res_files_path):
#     filename = r"C:\study\technion\MSc\Thesis\Y!\support_test\baseline_clmLMdocLM\claimLM_docLM_doc_ret_output\doc_res_alpha_0_beta_0.2_clm_47"
        print "filename:"+filename
        doc_score_dict = {} # key is docno, val is the exp(score)
        curr_claim = filename.split("_clm_")[1]
        curr_alpha = filename.split("_alpha_")[1].split("_beta_")[0]
        curr_beta = filename.split("_beta_")[1].split("_clm_")[0]
        curr_dict_name = "docs_scores_norm_alpha_"+curr_alpha+"_beta_"+curr_beta+"_clm_"+curr_claim+"_dict"
        try:
#             if os.path.exists(base_path+"\\docs_norm_scores_dicts\\"+curr_dict_name+"_sorted"):
#                 print curr_dict_name +" already there"
#                 continue
#             else:
#                 print "applying on "+curr_dict_name
            # check if the curr alpha beta dict exists already
                doc_file = open(doc_res_files_path+"/"+filename,'r')
                doc = doc_file.read().strip() # score
                scores_sum = 0.0
                if curr_claim in claims_file_counters_dict.keys():
                    claims_file_counters_dict[curr_claim] += 1 
                else:
                    claims_file_counters_dict[curr_claim] = 1
                for i, line in enumerate(doc.split('\n')):
                    data = line.split(' ')
                    query_Id = data[0]
                    doc_id = data[2]
                    norm_score = math.exp(float(data[4]))
                    scores_sum += norm_score
                    if os.path.exists(curr_dict_name) == True:
                        doc_score_dict = read_pickle(curr_dict_name)
                    if doc_id in doc_score_dict:
                        raise Exception("DOC ID %s already in dict" % doc_id)
                    doc_score_dict[query_Id,doc_id] = norm_score
            # divide by scores_sum
                for ((query_Id,doc_id),score) in doc_score_dict.items():
                    new_score = float(float(score)/float(scores_sum))
                    doc_score_dict[query_Id,doc_id] = new_score
                #rank according to score
                doc_score_dict_sorted = collections.OrderedDict(sorted(doc_score_dict.items(), key= lambda x: (-int(x[0][0]),x[1]),reverse=True))
                save_pickle(linux_base_path+"/"+"docs_norm_scores_dicts/"+curr_dict_name+"_sorted",doc_score_dict_sorted)
#                     save_pickle(base_path+ "\\docs_norm_scores_dicts"+curr_dict_name+"_sorted",doc_score_dict_sorted)
        except Exception as err: 
                            sys.stderr.write('problem in normalize_doc_scores in file:'+ filename)     
                            print err.args      
                            print err 
    for (claim_num,counter) in claims_file_counters_dict.items():
        if counter!=110:
            print claim_num+" not 110 files , but " +str(counter) +" files"

def normalize_sen_scores_corpus_smoothing():
    """
    from the sen retrieved,
    score= exp(score)
    and then normalize according to sum normalization
    file is an output file - claim num, alpha, beta
    for each alpha beta, create and than add to it the different claims' doc scores
    """
    param_len = 10*11#  for alpha*beta =110 
    k_len = 1
#     sen_res_files_path = linux_base_path+r"/claimLM_senLM_sen_ret_output_corpus_smoothing_corpus_beta_"+str(corpus_beta)
    sen_res_files_path = base_path+r"\claimLM_senLM_sen_ret_output_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
    norm_sen_res_path = base_path+r"\sen_norm_scores_dicts_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
    
    claims_file_counters_dict = {} #for each claim numas key, have the val a counter - if not alpha_beta_len*k_lem per claim -> problem!
    claim_list = [4]#,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
    top_k_docs_vals = [50]#    ,100,500]   
    for claim_num in claim_list:
        for k_val in top_k_docs_vals:
            for alpha in range(4,5,1):
                for beta in range(8,9,1):
                    (alpha_f,beta_f) = turn_to_float([alpha,beta])        
                    filename = "sen_res_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_clm_"+str(claim_num)
                    sen_file = open(sen_res_files_path+filename,'r')
                    print "in filename: "+filename
                    sen_score_dict = {} # key is docno, val is the exp(score)
                    curr_dict_name = "sen_scores_norm_clm_"+str(claim_num)+"_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_dict"
                    sen = sen_file.read().strip() # score
                    scores_sum = 0.0
                    if claim_num in claims_file_counters_dict.keys():
                        claims_file_counters_dict[claim_num] += 1 
                    else:
                        claims_file_counters_dict[claim_num] = 1
                    for i, line in enumerate(sen.split('\n')):
                        if i%2==0: # a data line
                            try:
                                data = line.split(' ')
                                query_Id = data[0]
                                doc_id = data[2]
                                if (data[4])!= "nan":
                                    norm_score = math.exp(float(data[4]))
                                    scores_sum += norm_score
                                    if os.path.exists(curr_dict_name) == True:
                                        sen_score_dict = read_pickle(curr_dict_name)
                                    if doc_id in sen_score_dict:
                                        raise Exception("DOC ID %s already in dict" % doc_id)
                            except Exception as err:
                                sys.stderr.write('problem in filename:'+ filename +' line: '+line)     
                                print err.args      
                                print err
                        else:
                            if len(line) >0:   
                                sen_score_dict[query_Id,line] = (doc_id,norm_score)
                # divide by scores_sum
                    for ((query_Id,sen),(doc_id,score)) in sen_score_dict.items():
                        new_score = float(float(score)/float(scores_sum))
                        sen_score_dict[query_Id,sen] = (doc_id,new_score)
                    #rank according to score
                    sen_score_dict_sorted = collections.OrderedDict(sorted(sen_score_dict.items(), key= lambda x: (-int(x[0][0]),x[1][1]),reverse=True))
            #         save_pickle(base_path+r"sen_norm_scores_dicts"+"\\"+curr_dict_name+"_sorted",doc_score_dict_sorted)
                    save_pickle(norm_sen_res_path+curr_dict_name+"_sorted",sen_score_dict_sorted) 
        for (claim_num,counter) in claims_file_counters_dict.items():
            if counter!=(param_len*k_len):
                print str(claim_num)+" not "+str(param_len*k_len)+" files , but " +str(counter) +" files"
        print "finished"
           
def normalize_sen_scores_adj_sen():
    """
    from the sen retrieved,
    score= exp(score)
    and then normalize according to sum normalization
    file is an output file - claim num, alpha, beta
    for each alpha beta, create and than add to it the different claims' doc scores
    """
    param_len = 10*11*55# alpha*beta*delta1*delta2, else just for alpha*beta =110 
    k_len = 1
#     sen_res_files_path = linux_base_path+r"/claimLM_senLM_adjacent_sen_ret_output_corpus_beta_"+str(corpus_beta)+"/"
    #norm_sen_res_path = linux_base_path+r"/sen_norm_scores_dicts_adjacent_sen_corpus_beta_+str(corpus_beta)+"/"
    sen_res_files_path = base_path + r"\claimLM_senLM_adj_sen_ret_output_corpus_beta_"+str(corpus_beta)+"\\"    
    norm_sen_res_path = base_path + r"\sen_norm_scores_dicts_adjacent_sen_corpus_beta_"+str(corpus_beta)+"\\"
#     sen_res_files_path = linux_base_path+r"/claimLM_senLM_doc_ret_output"
    claims_file_counters_dict = {} #for each claim numas key, have the val a counter - if not alpha_beta_len*k_lem per claim -> problem!
    claim_list = [4]#,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
    #claim_list = [36,37,39]
    #claim_list = [40,41,42,45,46,47]
    #claim_list = [50,51,53,54]
    #claim_list = [55,57,58,59]
    #claim_list = [60,61,62,66]
    #claim_list = [70,79,80]
    top_k_docs_vals = [50]#    ,100,500]   
    for claim_num in claim_list:
        for k_val in top_k_docs_vals:
            for alpha in range(4,5,1):
                for beta in range(8,9,1):
                    for delta_1 in range(9,10,1):
                            for delta_2 in range(0,1,1):
                                if not delta_1+delta_2 >9:
                                    (alpha_f,beta_f,delta_1_f,delta_2_f) = turn_to_float([alpha,beta,delta_1,delta_2])        
                                    filename = "sen_res_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_clm_"+str(claim_num)
                                    sen_file = open(sen_res_files_path+filename,'r')
                                    print "in filename: "+filename
                                    sen_score_dict = {} # key is docno, val is the exp(score)
                                    curr_dict_name = "sen_scores_norm_clm_"+str(claim_num)+"_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_dict"
                                    sen = sen_file.read().strip() # score
                                    scores_sum = 0.0
                                    if claim_num in claims_file_counters_dict.keys():
                                        claims_file_counters_dict[claim_num] += 1 
                                    else:
                                        claims_file_counters_dict[claim_num] = 1
                                    for i, line in enumerate(sen.split('\n')):
                                        if i%2==0: # a data line
                                            try:
                                                data = line.split(' ')
                                                query_Id = data[0]
                                                doc_id = data[2]
                                                if (data[4])!= "nan":
                                                    norm_score = math.exp(float(data[4]))
                                                    scores_sum += norm_score
                                                    if os.path.exists(curr_dict_name) == True:
                                                        sen_score_dict = read_pickle(curr_dict_name)
                                                    if doc_id in sen_score_dict:
                                                        raise Exception("DOC ID %s already in dict" % doc_id)
                                            except Exception as err:
                                                sys.stderr.write('problem in filename:'+ filename +' line: '+line)     
                                                print err.args      
                                                print err
                                        else:
                                            if len(line) >0:   
                                                sen_score_dict[query_Id,line] = (doc_id,norm_score)
                                # divide by scores_sum
                                    for ((query_Id,sen),(doc_id,score)) in sen_score_dict.items():
                                        new_score = float(float(score)/float(scores_sum))
                                        sen_score_dict[query_Id,sen] = (doc_id,new_score)
                                    #rank according to score
                                    sen_score_dict_sorted = collections.OrderedDict(sorted(sen_score_dict.items(), key= lambda x: (-int(x[0][0]),x[1][1]),reverse=True))
                            #         save_pickle(base_path+r"sen_norm_scores_dicts"+"\\"+curr_dict_name+"_sorted",doc_score_dict_sorted)
                                    save_pickle(norm_sen_res_path+curr_dict_name+"_sorted",sen_score_dict_sorted) 
        for (claim_num,counter) in claims_file_counters_dict.items():
            if counter!=(param_len*k_len):
                print str(claim_num)+" not "+str(param_len*k_len)+" files , but " +str(counter) +" files"
        print "finished"
                                    
#     for filename in os.listdir(sen_res_files_path):
#         print "filename:"+filename
#         sen_score_dict = {} # key is docno, val is the exp(score)
#         curr_claim = filename.split("_clm_")[1]
#         curr_alpha = filename.split("_alpha_")[1].split("_beta_")[0]
#         curr_beta = filename.split("_beta_")[1].split("_top_k_docs_")[0]
#         curr_k_val = filename.split("_top_k_docs_")[1].split("_clm_")[0]
#         curr_delta1 = filename.split("_delta1_")[1].split("_delta2_")[0]
#         curr_delta2 = filename.split("_delta2_")[1].split("_top_k_docs_")[0]
#         if curr_k_val == "50":
#             curr_dict_name = "sen_scores_norm_clm_"+curr_claim+"_alpha_"+curr_alpha+"_beta_"+curr_beta+"_delta1_"+str(curr_delta1)+"_delta2_"+str(curr_delta2)+"_top_k_docs_"+str(curr_k_val)+"_dict"
#             # check if the curr alpha beta dict exists already        
#     #         sen_file = open(sen_res_files_path+"\\"+filename,'r')
#             sen_file = open(sen_res_files_path+"/"+filename,'r')
#             sen = sen_file.read().strip() # score
#             scores_sum = 0.0
#             if curr_claim in claims_file_counters_dict.keys():
#                 claims_file_counters_dict[curr_claim] += 1 
#             else:
#                 claims_file_counters_dict[curr_claim] = 1
#             for i, line in enumerate(sen.split('\n')):
#                 if i%2==0: # a data line
#                     try:
#                         data = line.split(' ')
#                         query_Id = data[0]
#                         doc_id = data[2]
#                         if (data[4])!= "nan":
#                             norm_score = math.exp(float(data[4]))
#                             scores_sum += norm_score
#                             if os.path.exists(curr_dict_name) == True:
#                                 sen_score_dict = read_pickle(curr_dict_name)
#                             if doc_id in sen_score_dict:
#                                 raise Exception("DOC ID %s already in dict" % doc_id)
#                     except Exception as err:
#                         sys.stderr.write('problem in filename:'+ filename +' line: '+line)     
#                         print err.args      
#                         print err
#                 else:
#                     if len(line) >0:   
#                         sen_score_dict[query_Id,line] = (doc_id,norm_score)
#     #                     if (query_Id,line) in sen_score_dict.keys():
#     #                         sen_score_dict[query_Id,line].append((doc_id,norm_score))
#     #                     else:
#     #                         sen_score_dict[query_Id,line] = [(doc_id,norm_score)]
#     #                 sen_score_dict[query_Id,doc_id,line] = norm_score
#         # divide by scores_sum
#             for ((query_Id,sen),(doc_id,score)) in sen_score_dict.items():
#                 new_score = float(float(score)/float(scores_sum))
#                 sen_score_dict[query_Id,sen] = (doc_id,new_score)
#     #         for ((query_Id,doc_id),sen_scores_list) in sen_score_dict.items():
#     #             new_sen_scores_list = []
#     #             for (sen,score) in sen_scores_list:
#     #                 new_score = float(float(score)/float(scores_sum))
#     #                 new_sen_scores_list.append((sen,new_score))
#     #             sen_score_dict[query_Id,doc_id] = new_sen_scores_list
#             #rank according to score
#             sen_score_dict_sorted = collections.OrderedDict(sorted(sen_score_dict.items(), key= lambda x: (-int(x[0][0]),x[1][1]),reverse=True))
#     #         save_pickle(base_path+r"sen_norm_scores_dicts"+"\\"+curr_dict_name+"_sorted",doc_score_dict_sorted)
#             save_pickle(linux_base_path+r"/sen_norm_scores_dicts"+setup+"/"+curr_dict_name+"_sorted",sen_score_dict_sorted) 
#         
#     for (claim_num,counter) in claims_file_counters_dict.items():
#         if counter!=(param_len*k_len):
#             print claim_num+" not "+str(param_len*k_len)+" files , but " +str(counter) +" files"
#     print "finished"


# def remove_double_delta_from_filename():
#     claim_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
#     top_k_docs_vals = [50]#    ,100,500]   
#     for claim_num in claim_list:
#         for k_val in top_k_docs_vals:
#             for alpha in range(0,11,1):
#                 for beta in range(0,10,1):
#                     for delta_1 in range(0,10,1):
#                             for delta_2 in range(0,10,1):
#                                 if not delta_1+delta_2 >9:
#                                     (alpha_f,beta_f,delta_1_f,delta_2_f) = turn_to_float([alpha,beta,delta_1,delta_2])         
#                                     filename = "sen_res_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_clm_"+str(claim_num)
#                                     if file                                 
                                    
def get_top_k_docs():
    """for each alpha and beta values, 
    Get the top k =[50,100,500] docs from which sentences will be retrieved
    As one of the setups is a according to the LM of the sen and the corpus alone (as done with usual sentence retieval in Indri) then I will create a file
    for Indri as done in pool creation  n ret' process -  add the top docs as docWorkingSet
     read each dict of a norm_scores 
     get the top K docs and create a file for each claim for sentence retrieval.
    """
#     doc_res_dicts_path = base_path+"\\docs_norm_scores_dicts"
#     sen_ret_input_path = base_path+"\\claimLM_senLM_sen_ret_input"
    doc_res_dicts_path = linux_base_path+"/docs_norm_scores_dicts"
    sen_ret_input_path = linux_base_path+"/claimLM_senLM_sen_ret_input"
    claims_no_SW_dict = read_pickle("claims_no_SW_dict")
    k_values = [100]
    for filename in os.listdir(doc_res_dicts_path):
        if not "clm_key_ranked_list_of_docs_" in filename:
            curr_dict = read_pickle(doc_res_dicts_path+"/"+filename)
            curr_claim_num = filename.split("_clm_")[1].split("_dict_sorted")[0]
            curr_alpha = filename.split("_alpha_")[1].split("_beta_")[0]
            curr_beta = filename.split("_beta_")[1].split("_clm_")[0]
            for k_val in k_values:
                top_k_docs = [key[1] for key in curr_dict.keys()][0:k_val]
                # write to sen query file with workingSetDocno
                sen_ret_docno_file = open(sen_ret_input_path+"/claimLM_senLM_sen_ret_docno_alpha_"+curr_alpha+"_beta_"+curr_beta+"_top_k_docs_"+str(k_val)+"_clm_"+curr_claim_num,"wb")
                sen_ret_docno_file.write("<parameters>\n")
                sen_ret_docno_file.write("<query><number>"+curr_claim_num+"</number><text>"+claims_no_SW_dict[curr_claim_num][0].strip()+"|"+claims_no_SW_dict[curr_claim_num][1].strip()+"</text>")
                for workingDoc in top_k_docs:
                    sen_ret_docno_file.write("<workingSetDocno>"+workingDoc+"</workingSetDocno>")
                sen_ret_docno_file.write("</query>\n")
                sen_ret_docno_file.write("</parameters>")
                sen_ret_docno_file.close()
 
def merge_all_claims_norm_dicts_for_docs():
    """
    for each alpha, beta value turn the docs norms dicts to a single dict - for all claims
     sen_norm_scores_dicts_path =  base_path+"\\sen_norm_scores_dicts"
    for each alpha,beta, k  value turn the sens norms dicts to a single dict - for all claims
     """     
#     docs_norm_scores_dicts_path = base_path+"\\docs_norm_scores_dicts"
    docs_norm_scores_dicts_path = linux_base_path+"/docs_norm_scores_dicts"
#     all_claims_norms_scores_merged_dict = base_path +"\\all_claims_norms_scores_merged_dict"
    all_claims_norms_scores_merged_dict = linux_base_path +"/all_claims_norms_scores_merged_dict"
    for alpha in range(0,11,1):
            for beta in range(0,10,1):
                docs_scores_all_claims = {}
                for filename in os.listdir(docs_norm_scores_dicts_path):
                    (alpha_f,beta_f)=turn_to_float([alpha,beta])
                    if "_alpha_"+str(alpha_f)+"_" in filename and "_beta_"+str(beta_f)+"_" in filename:
                        curr_dict = read_pickle(docs_norm_scores_dicts_path+"/"+filename)
                        docs_scores_all_claims = dict(docs_scores_all_claims.items() + curr_dict.items()) #merge dicts
                save_pickle(all_claims_norms_scores_merged_dict+"/docs_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f),docs_scores_all_claims)

def merge_all_claims_norm_dicts_for_sen_corpus_smoothing():
#     sen_norm_scores_dicts_path = linux_base_path+"/sen_norm_scores_dicts_corpus_smoothing_corpus_beta_"+str(corpus_beta))+"/"
#     all_claims_norms_scores_merged_dict = linux_base_path +"/all_claims_norms_scores_merged_dict_corpus_smoothing_corpus_beta_"+str(corpus_beta))+"/"
    all_claims_norms_scores_merged_dict = base_path +"\\all_claims_norms_scores_merged_dict_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
    sen_norm_scores_dicts_path = base_path +r"\sen_norm_scores_dicts_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
#     claim_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80] 
    claim_list = [4]
    #alpha_f_vals = [0,0.1,0.2]#0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#     alpha_f_vals = [0.2,0.5,0.9,1]
    #alpha_f_vals = [0.3,0.4,0.5]
    #alpha_f_vals = [0.6,0.7]
    #alpha_f_vals = [0.8,0.9,1]
    top_k_docs_vals = [50]#    ,100,500]
    for k_val in top_k_docs_vals:
        for alpha in range(4,5,1):
            for beta in range(8,9,1):
                sen_norm_scores_all_claims = {}
                for claim_num in claim_list:   
                    (alpha_f,beta_f) = turn_to_float([alpha,beta])  
                    print "in alpha "+str(alpha_f)
                    filename = "sen_scores_norm_clm_"+str(claim_num) +"_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_dict_sorted"
                    curr_dict = read_pickle(sen_norm_scores_dicts_path+filename)
                    sen_norm_scores_all_claims = dict(sen_norm_scores_all_claims.items() + curr_dict.items())
                save_pickle(all_claims_norms_scores_merged_dict+"sen_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val),sen_norm_scores_all_claims)            
    print "finished merge"

def merge_all_claims_norm_dicts_for_sen_adj_sen():
#     sen_norm_scores_dicts_path = linux_base_path+"/sen_norm_scores_dicts_adjacent_sen_corpus_beta_"+str(corpus_beta)
#     all_claims_norms_scores_merged_dict = linux_base_path +"/all_claims_norms_scores_merged_dict_adjacent_sen_corpus_beta_"+str(corpus_beta)
    sen_norm_scores_dicts_path = base_path +r"\sen_norm_scores_dicts_adjacent_sen_corpus_beta_"+str(corpus_beta)+"\\"
    all_claims_norms_scores_merged_dict = base_path +r"\all_claims_norms_scores_merged_dict_adjacent_sen_corpus_beta_"+str(corpus_beta)+"\\"
    claim_list = [4] 
    alpha_f_vals = [0.4]#0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #alpha_f_vals = [0.2,0.3]
    #alpha_f_vals = [0.4,0.5]
    #alpha_f_vals = [0.3,0.4,0.5]
    #alpha_f_vals = [0.6,0.7]
    #alpha_f_vals = [0.8] 
    #alpha_f_vals = [0.9]
    #alpha_f_vals = [1]
    #claim_list = [40,41,42,45,46,47]   
    #claim_list = [50,51,53,54,55,57,58,59]
    #claim_list = [60,61,62,66,69,70,79,80]
    top_k_docs_vals = [50]#    ,100,500]
    for k_val in top_k_docs_vals:
        for alpha_f in alpha_f_vals:
            print "in alpha "+str(alpha_f)
            for beta in range(8,9,1):
                for delta_1 in range(9,10,1):
                        for delta_2 in range(0,1,1):
                            if not delta_1+delta_2 >9: 
                                sen_norm_scores_all_claims = {}
                                for claim_num in claim_list:   
                                    (beta_f,delta_1_f,delta_2_f) = turn_to_float([beta,delta_1,delta_2])        
    #                                     filename = "sen_res_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_clm_"+str(claim_num)
                                    filename = "sen_scores_norm_clm_"+str(claim_num) +"_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_dict_sorted"
    #                                     for filename in os.listdir(sen_norm_scores_dicts_path):
    #                                     if "_alpha_"+str(alpha_f)+"_" in filename and "_beta_"+str(beta_f)+"_" in filename and "_top_k_docs_"+str(k_val)+"_" in filename:
                                    curr_dict = read_pickle(sen_norm_scores_dicts_path+filename)
                                    sen_norm_scores_all_claims = dict(sen_norm_scores_all_claims.items() + curr_dict.items())
                                save_pickle(all_claims_norms_scores_merged_dict+"sen_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val),sen_norm_scores_all_claims)            
    print "finished merge"
    
def interpolate_doc_sen_score_corpus_smoothing():
    print "started interpolate_doc_sen_score"
#     all_claims_norms_scores_merged_dict_docs = linux_base_path +"/all_claims_norms_scores_merged_dict_docs"
#     all_claims_norms_scores_merged_dict_sen = linux_base_path +"/all_claims_norms_scores_merged_dict_corpus_smoothing_corpus_beta_"+str(corpus_beta)
#     final_sen_scores = linux_base_path + "/final_sen_scores_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"/"
    all_claims_norms_scores_merged_dict_docs = base_path +r"\all_claims_norms_scores_merged_dict_docs"+"\\"
    all_claims_norms_scores_merged_dict_sen = base_path +r"\all_claims_norms_scores_merged_dict_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
    final_sen_scores = base_path + r"\final_sen_scores_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
    
    top_k_docs_vals = [50]#,100,500]   
#     alpha_range = [0,0.1,0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#     #alpha_range = [0.3,0.4,0.5]
#     #alpha_range = [0.6,0.7,0.8]
#     #alpha_range = [0.9,1]
    for k_val in top_k_docs_vals:
        for alpha in range(4,5,1): #change just for test!
#         for alpha_f in alpha_range:
            for beta in range(8,9,1):
                (alpha_f,beta_f)=turn_to_float([alpha, beta])
                docs_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict_docs+"docs_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f))
                sen_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict_sen+"sen_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val))
#                 docs_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict+"docs_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f))
#                 sen_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict+"sen_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val))
                for lambda_int in range(2,3,1):
                    lambda_f = turn_to_float([lambda_int])
                    curr_final_sen_score_dict = {}
#                     if os.path.exists(final_sen_scores+"clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted") == True:
#                         continue
#                     else:
                    try:
                        for ((claim_num,sen),(doc_id,sen_score)) in sen_scores_all_claims.items():
                            new_sen_score = lambda_f*sen_score + (1-lambda_f)*docs_scores_all_claims[claim_num,doc_id]
                            curr_final_sen_score_dict[claim_num,sen] = ((doc_id,new_sen_score))
                    except Exception as err: 
                        sys.stderr.write('problem in claim_num:' +claim_num +" sen:" +sen)     
                        print err.args      
                        print err
                        #rank per claim  according to score
                    curr_final_sen_score_dict_sorted = collections.OrderedDict(sorted(curr_final_sen_score_dict.items(), key= lambda x: (-int(x[0][0]),x[1][1]),reverse=True)) 
                    #move directly to a ranked list:
                    clm_num_key_final_ranked_list_sen = {} #key is a claim num, value is a list 
                    for ((clm_num,sen),(doc_id,score)) in curr_final_sen_score_dict_sorted.items():
                        if (clm_num) in clm_num_key_final_ranked_list_sen.keys():
                            clm_num_key_final_ranked_list_sen[clm_num].append((sen,score))
                        else:
                            clm_num_key_final_ranked_list_sen[clm_num]=[(sen,score)]
#                     save_pickle(final_sen_scores+"all_claims_final_sen_score_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted",curr_final_sen_score_dict_sorted)
                    save_pickle(final_sen_scores+"clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted",clm_num_key_final_ranked_list_sen)
    print "finished interpolate_doc_sen_score"


def interpolate_doc_sen_score_adj_sen():
    """
    after sen retrieval for each alpha and beta,
    interpolate with the weight between as free param
    and rank according to lambda value
    """
    print "started interpolate_doc_sen_score"
#     all_claims_norms_scores_merged_dict_docs = linux_base_path +"/all_claims_norms_scores_merged_dict_docs"
#     all_claims_norms_scores_merged_dict_sen = linux_base_path +"/all_claims_norms_scores_merged_dict_adjacent_sen_corpus_beta_"+str(corpus_beta)
#     final_sen_scores = linux_base_path + "/final_sen_scores_adjacent_sen_corpus_beta_"+str(corpus_beta)+"/"

    all_claims_norms_scores_merged_dict_docs = base_path +"\\all_claims_norms_scores_merged_dict_docs\\"
    all_claims_norms_scores_merged_dict_sen = base_path +"\\all_claims_norms_scores_merged_dict_adjacent_sen_corpus_beta_"+str(corpus_beta)+"\\"
    final_sen_scores = base_path + "\\final_sen_scores_adjacent_sen_corpus_beta_"+str(corpus_beta)+"\\"

#     all_claims_norms_scores_merged_dict_sen = base_path + 
    top_k_docs_vals = [50]#,100,500]   
    alpha_range = [0.4]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #alpha_range = [0.2,0.3]
    #alpha_range = [0.4,0.5]
    #alpha_range = [0.6,0.7]
    #alpha_range = [0.8]
    #alpha_range = [0.9]
    #alpha_range = [1]
    for k_val in top_k_docs_vals:
#         for alpha in range(0,11,1): #change just for test!
        for alpha_f in alpha_range:
            for beta in range(8,9,1):
                for delta_1 in range(9,10,1):
                    for delta_2 in range(0,1,1):
                        if not delta_1+delta_2 >9: 
                            (beta_f, delta_1_f,delta_2_f)=turn_to_float([beta,delta_1,delta_2])
                            docs_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict_docs+"/docs_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f))
                            sen_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict_sen+"/sen_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val))
#                             docs_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict_docs+"docs_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f))
#                             sen_scores_all_claims = read_pickle(all_claims_norms_scores_merged_dict_sen+"sen_norm_scores_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val))
                            for lambda_int in range(2,3,1):
                                lambda_f = turn_to_float([lambda_int])
                                curr_final_sen_score_dict = {}
#                                 if os.path.exists(final_sen_scores+"clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted") == True:
#                                     continue
#                                 else:
                                try:
                                    for ((claim_num,sen),(doc_id,sen_score)) in sen_scores_all_claims.items():
                                        new_sen_score = lambda_f*sen_score + (1-lambda_f)*docs_scores_all_claims[claim_num,doc_id]
                                        curr_final_sen_score_dict[claim_num,sen] = ((doc_id,new_sen_score))
                                except Exception as err: 
                                        sys.stderr.write('problem in claim_num:' +claim_num +" sen:" +sen)     
                                        print err.args      
                                        print err
                                    #rank per claim  according to score
                                curr_final_sen_score_dict_sorted = collections.OrderedDict(sorted(curr_final_sen_score_dict.items(), key= lambda x: (-int(x[0][0]),x[1][1]),reverse=True)) 
                                #move directly to a ranked list:
                                clm_num_key_final_ranked_list_sen = {} #key is a claim num, value is a list 
                                for ((clm_num,sen),(doc_id,score)) in curr_final_sen_score_dict_sorted.items():
                                    if (clm_num) in clm_num_key_final_ranked_list_sen.keys():
                                        clm_num_key_final_ranked_list_sen[clm_num].append((sen,score))
                                    else:
                                        clm_num_key_final_ranked_list_sen[clm_num]=[(sen,score)]
            #                     save_pickle(final_sen_scores+"all_claims_final_sen_score_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted",curr_final_sen_score_dict_sorted)
                                save_pickle(final_sen_scores+"clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted",clm_num_key_final_ranked_list_sen)
    print "finished interpolate_doc_sen_score"

# def convert_ranked_sen_keys_to_list_of_sen():
#     """
#     from a (clm_num, sen) keys as ranked, 
#     turn to (clm) -> list of the ranked sentences
#     for the nDCG and MAP calculations
#     """
#     final_sen_scores = linux_base_path + "/final_sen_scores/"
#     top_k_docs_vals = [50]#,100,500]   
#     for k_val in top_k_docs_vals:
#         for alpha in range(0,11,1): #change just for test!
#             for beta in range(0,10,1):
#                 for lambda_int in range(0,11,1):
#                     (alpha_f,beta_f)=turn_to_float([alpha,beta])
#                     lambda_f = turn_to_float([lambda_int])
#                     clm_num_key_final_ranked_list_sen = {} #key is a claim num, value is a list
#                     final_sen_scores_all_claims = read_pickle(final_sen_scores+"/all_claims_final_sen_score_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted")
#                     for ((clm_num,sen),(doc_id,score)) in final_sen_scores_all_claims.items():
#                         if (clm_num) in clm_num_key_final_ranked_list_sen.keys():
#                             clm_num_key_final_ranked_list_sen[clm_num].append((sen,score))
#                         else:
#                             clm_num_key_final_ranked_list_sen[clm_num]=[(sen,score)]
#                     os.remove(final_sen_scores+"all_claims_final_sen_score_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted")
#                     print "removed" +"/all_claims_final_sen_score_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted"
#                     save_pickle(final_sen_scores+"/clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f),clm_num_key_final_ranked_list_sen)                        
#     print "finished conversion"
#                      

def convert_true_support_to_relevance():
    """
    compute the nDCG for all claims, based on the ranking of each baseline
    """
    clm_as_key_sen_support_score_val_wiki = read_pickle("clm_as_key_sen_support_score_val_wiki") #key is a claim num, value is a list of tuple - sen and their 0/1/2 support score)
    claim_sen_true_relevance_dict = {}
    claim_dict = read_pickle("claim_dict")
    #t Dead Poets Society , Peter Weir , Steven HaftPaurn clm_as_key_sen_support_score_val_wiki to a relevance dict
    for (claim_num,list_sen_support_score) in clm_as_key_sen_support_score_val_wiki.items():
        new_relevance_list = []
        for (sen,support_score) in list_sen_support_score:
            if support_score !=0:
                new_relevance_list.append((sen,1))
            else:
                new_relevance_list.append((sen,0))
        claim_sen_true_relevance_dict[claim_dict[str(claim_num)]] = new_relevance_list
    save_pickle("claim_sen_relevance_dict_"+curr_source, claim_sen_true_relevance_dict)

def create_rel_doctitle_dict():
    """
    A doc is relevant if it has a relevant sentence, according to the true rel annotation.
    """
    claim_rel_docno_dict = {} #key is claim text, value is a set of doc_title that are relevant
    clm_sen_doc_title_dict = read_pickle("sen_doc_title_dict")
    claim_sen_true_relevance_dict = read_pickle("claim_sen_relevance_dict_"+curr_source)
    exclude = set(string.punctuation)
    docID_title_mapping_wiki_pickle = read_pickle("dicID_title_mapping_wiki_pickle")
    
    title_docID_mapping_wiki_pickle = {}
    for (docID,doc_title) in docID_title_mapping_wiki_pickle.iteritems():
        non_asci_char = [c for c in doc_title if not 0 < ord(c) < 127]
        new_doc_title = doc_title
        for c in non_asci_char:
            new_doc_title = new_doc_title.replace(c,"")
        doc_title_no_punc = ''.join(ch for ch in new_doc_title if ch not in exclude)
        doc_title_no_space = doc_title_no_punc.replace(" ","")
        title_docID_mapping_wiki_pickle[doc_title_no_space] = docID
#     title_docID_mapping_wiki_pickle = dict((y,x) for x,y in docID_title_mapping_wiki_pickle.iteritems())   
    for (clm) in claim_sen_true_relevance_dict.keys():
        rel_docno_set = set()
        for (sen,rel_score) in claim_sen_true_relevance_dict[clm]:
            try:    
                    if rel_score == 1:
                        sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
                        sen_no_space = sen_no_punc.replace(" ","")
                        curr_rel_doc_title = clm_sen_doc_title_dict[sen_no_space]
                        non_asci_char = [c for c in curr_rel_doc_title if not 0 < ord(c) < 127]
                        new_curr_doc_title = curr_rel_doc_title
                        for c in non_asci_char:
                            new_curr_doc_title = new_curr_doc_title.replace(c,"")
                        curr_doc_title_no_punc = ''.join(ch for ch in new_curr_doc_title if ch not in exclude)
                        curr_doc_title_no_space = curr_doc_title_no_punc.replace(" ","")
                        rel_docno_set.add((title_docID_mapping_wiki_pickle[curr_doc_title_no_space],1))
                
            except Exception as err: 
                sys.stderr.write('problem in sen:'+sen)     
                print err.args
        
        rel_docno_list = [(docid,rel_score) for (docid,rel_score) in rel_docno_set]
        claim_rel_docno_dict[clm] = rel_docno_list
    save_pickle("claim_rel_docno_dict", claim_rel_docno_dict)      

def create_list_of_retrieved_docs():
    """
    for the document retrirval baseline performance:
    turn the clm, docid dict to a dict of key is a claim and value is a ranked list of sentences
    """
#     docs_norms_path = base_path+"\\docs_norm_scores_dicts\\"
    docs_norms_path = linux_base_path+"/docs_norm_scores_dicts/"
    clm_key_ranked_list_of_docs = {} #key is a claim num, value is a ranked list of the docs retrieved
    for curr_clm in claim_list:
        for alpha in range(0,11,1): #change just for test!
                for beta in range(0,10,1):
                    (alpha_f,beta_f) = turn_to_float([alpha,beta])
                    curr_filename = docs_norms_path+"docs_scores_norm_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_clm_"+str(curr_clm)+"_dict_sorted"
                    curr_docs_baseline_dict = read_pickle(curr_filename) #key is qID and docid
                    curr_docs_list = []
                    for (clm_num,docid) in curr_docs_baseline_dict.keys():
                        if clm_num == str(curr_clm):
                            curr_docs_list.append(docid)
                    clm_key_ranked_list_of_docs[curr_clm] = curr_docs_list
                save_pickle(docs_norms_path+"clm_key_ranked_list_of_docs_alpha_"+str(alpha_f)+"_beta_"+str(beta_f),clm_key_ranked_list_of_docs)             
                    
def calc_doc_ret_MAP():
    """
    on the document retrieval results, 
    calc the MAP
    """
    
#     docs_norm_scores_dicts_path = linux_base_path+"/docs_norm_scores_dicts"
    claim_rel_docno_dict = read_pickle("claim_rel_docno_dict") #key is clm, value is a set of the relevant docno
#     nDCG_MAP_res = base_path +"\\nDCG_MAP_res\\"
#     docs_norms_path = base_path+"\\docs_norm_scores_dicts\\"
    docs_norms_path = linux_base_path+"/docs_norm_scores_dicts/"
    nDCG_MAP_res = linux_base_path +"/nDCG_MAP_res/"
    
    AP_cut_off = 1000
    k_val = 100
    p = 10
    log = open("calc_doc_avg_nDCG_MAP_log_k_top_docs_"+str(k_val)+"_at_"+str(p),"wb")
    res_file = open(nDCG_MAP_res+"doc_ret_nDCG_MAP_res_k_top_docs_"+str(k_val)+"_at_"+str(p),"wb")
#     each_params_AVGnDCG_MAP_dict = {} #key is alpha,beta,k_docs,lambda and val is the avg nDCG and MAP across all claims together
    each_params_MAP_dict = {}
#     NDCG_AP_all_claims_all_param_values = {}
    AP_all_claims_all_param_values = {}
    best_avg_nDCG = 0
    best_MAP = 0 #across all claims in a given configuration, find the max measures
    
#     docs_norms_path = base_path+"\\docs_norm_scores_dicts\\"
    claims_dict = read_pickle("claim_dict")
    #count the number of sentences that were retrived that are in the true data....sum for each claim, then average.
    
    
    for alpha in range(0,11,1): #change just for test!
        for beta in range(0,10,1):
            (alpha_f,beta_f) = turn_to_float([alpha,beta])
            NDCG_all_claims= {} #key is a claim, value is the nDCG
            AP_all_claims= {} 
            AP_cut_off = 1000
            curr_filename = docs_norms_path+"clm_key_ranked_list_of_docs_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)
            clm_key_ranked_list_of_docs_baseline = read_pickle(curr_filename) #key is qID and docid
            #need to turn it to a list of docs
            for clm in claim_list:
                try:
#                     nDCG_score = utils_linux.calc_doc_emp_NDCG(curr_source,str(clm),clm_key_ranked_list_of_docs_baseline[clm],claim_rel_docno_dict[claims_dict[str(clm)]],p)
#                     NDCG_all_claims[clm] = nDCG_score
                    AP_score = utils_linux.calc_doc_AP_relevance(AP_cut_off,curr_source,clm,clm_key_ranked_list_of_docs_baseline[clm],claim_rel_docno_dict[claims_dict[str(clm)]])
                    AP_all_claims[clm] = AP_score
                    AP_all_claims_all_param_values[clm,alpha_f,beta_f,k_val] = AP_score
                except Exception as err: 
                    log.write('problem in calculations: in source: '+ curr_source+' in clm '+ claims_dict[str(clm)]+" alpha:"+str(alpha_f)+ "beta:"+str(beta_f)+" \n" )     
                    for arg in err.args:
                        log.write(arg+" ")      
                    log.write("\n")  
#             average_NDCG = float(float(sum(NDCG_all_claims.values()))/float(len(NDCG_all_claims))) #across all claims...
#             if average_NDCG > best_avg_nDCG:
#                 best_avg_nDCG = average_NDCG
#                 best_avg_nDCG_configuration = (alpha_f,beta_f,k_val)
            MAP = float(float(sum(AP_all_claims.values()))/float(len(AP_all_claims)))
            if MAP > best_MAP:
                best_MAP = MAP
                best_MAP_configuration = (alpha_f,beta_f,k_val)
            each_params_MAP_dict[alpha_f,beta_f,k_val] = MAP
            utils_linux.save_pickle(nDCG_MAP_res+"doc_ret_NDCG_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_at_"+str(p),NDCG_all_claims)
            utils_linux.save_pickle(nDCG_MAP_res+"doc_ret_AP_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val),AP_all_claims)
#             res_file.write("alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"|"+"AnDCG_"+str(average_NDCG)+"_MAP_"+str(MAP)+"\n")
            res_file.write("alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"|_MAP_"+str(MAP)+"\n")
    save_pickle(nDCG_MAP_res+"doc_ret_NDCG_AP_all_claims_all_param_values_k_top_docs_"+str(k_val)+"_at_"+str(p),AP_all_claims_all_param_values)
    save_pickle(nDCG_MAP_res+"doc_ret_each_params_AVGnDCG_MAP_dict_k_top_docs_"+str(k_val)+"_at_"+str(p),each_params_MAP_dict)
#     best_row = "best_avg_nDCG|"+str(best_avg_nDCG)+"|best_avg_nDCG_configuration|"+str(best_avg_nDCG_configuration[0])+","+str(best_avg_nDCG_configuration[1])+","+str(best_avg_nDCG_configuration[3])+","+str(best_avg_nDCG_configuration[2])+"|"
    best_row = "best_MAP|" +str(best_MAP)+"|best_MAP_configuration|"+str(best_MAP_configuration[0])+","+str(best_MAP_configuration[1])+","+str(best_MAP_configuration[2])
    res_file.write(best_row)
    res_file.close()
    log.close()
    
def process_baseline_sen_ret_result():  
    p = 10
    param_range = namedtuple('param_range','start, end')
    alpha_range = param_range(0,11)
    beta_range = param_range(0,11-corpus_beta_int)
    lambda_range = param_range(0,11)
    calc_sen_ret_avg_nDCG_MAP_prec_at_k_corpus_smoothing(p)
#     find_best_free_param_configuration_LOO_corpus_smoothing(p,alpha_range,beta_range,lambda_range)

#     calc_sen_ret_avg_nDCG_MAP_prec_at_k_adj_sen(p)
#     find_best_free_param_configuration_LOO_adj_sen(p)
#     calc_std_nDCG_AP_adj_sen(p)
#     calc_std_nDCG_AP(p)
    #compare alpha, beta, k_val, lambda
#     final_sen_scores = linux_base_path + "/final_sen_scores/"
    print "finished process_baseline_sen_ret_result"
    
def calc_sen_ret_avg_nDCG_MAP_prec_at_k_corpus_smoothing(p):
    log = open("calc_avg_nDCG_MAP_log_at_"+str(p),"wb")
    claim_sen_true_relevance_dict = read_pickle("claim_sen_relevance_dict_"+curr_source) 
    claims_dict = read_pickle("claim_dict")
    final_sen_scores = base_path + "\\final_sen_scores_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
    measures_res = base_path +"\\measures_res_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"\\"
#     final_sen_scores = linux_base_path + "/final_sen_scores_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"/"
#     final_sen_scores = "/IBM_STORAGE/USERS_DATA/liorab/baseline_ret/sen_ret_corpus_smoothing_res/final_sen_scores_corpus_smoothing/"
#     measures_res = linux_base_path+ "/measures_res_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"/"
#     p = 100
    top_k_docs_values = [50]
    
    each_params_AVGnDCG_MAP_prec_at_k_dict = {} #key is alpha,beta,k_docs,lambda and val is the avg nDCG and MAP across all claims together
    NDCG_AP_prec_at_k_all_claims_all_param_values = {}
    best_avg_nDCG = 0
    best_MAP = 0 #across all claims in a given configuration, find the max measures
    prec_at_k = rcdtype.recordtype('prec_at_k', 'at_5 at_10')
    prec_at_k_avg = rcdtype.recordtype('prec_at_k_avg', 'at_5 at_10')
    best_prec_at_k = rcdtype.recordtype('best_prec_at_k_avg', 'value config')
#     p_at_k_avg = prec_at_k_avg(0,0)
    best_p_at_5 = best_prec_at_k(0,"")
    best_p_at_10 = best_prec_at_k(0,"")
    #count the number of sentences that were retrived that are in the true data....sum for each claim, then average.
    claim_list = ['4']
    for k_val in top_k_docs_values:
        res_file = open(measures_res+"nDCG_MAP_prec_at_k_res_k_top_docs_"+str(k_val)+"_at_"+str(p),"wb")
        for alpha in range(4,5,1): #change just for test!
#             for beta in range(0,11-corpus_beta_int,1):
            for beta in range(8,9,1):
                for lambda_int in range(2,3,1):
                    lambda_f = turn_to_float([lambda_int])
                    (alpha_f,beta_f) = turn_to_float([alpha,beta])
                    curr_baseline_final_sen_score_dict = read_pickle(final_sen_scores+"clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted")
                    NDCG_all_claims= {} #key is a claim, value is the nDCG
                    AP_all_claims= {}
                    prec_at_k_all_claims = {} #precision@5,10 as average 
                    AP_cut_off = 1000
                    p_at_k_avg = prec_at_k_avg(0,0)
                    
#                     for clm in curr_baseline_final_sen_score_dict.keys():
                    for clm in claim_list:
                        try:
                            NDCG_all_claims[clm] = utils_linux.calc_sen_emp_NDCG(curr_source,clm,curr_baseline_final_sen_score_dict[clm],claim_sen_true_relevance_dict[claims_dict[clm]],p)
                            AP_all_claims[clm] = utils_linux.calc_sen_AP_relevance(AP_cut_off,curr_source,clm,curr_baseline_final_sen_score_dict[clm],claim_sen_true_relevance_dict[claims_dict[clm]])
                            p_at_k = prec_at_k(utils_linux.calc_sen_precision_at_k(5, curr_baseline_final_sen_score_dict[clm], claim_sen_true_relevance_dict[claims_dict[clm]]),utils_linux.calc_sen_precision_at_k(10, curr_baseline_final_sen_score_dict[clm], claim_sen_true_relevance_dict[claims_dict[clm]]))
                            
                            p_at_k_avg.at_5 += p_at_k.at_5
                            p_at_k_avg.at_10 += p_at_k.at_10
                            prec_at_k_all_claims[clm] = (p_at_k.at_5,p_at_k.at_10)
                            NDCG_AP_prec_at_k_all_claims_all_param_values[clm,alpha_f,beta_f,k_val,lambda_f] = (NDCG_all_claims[clm],AP_all_claims[clm],p_at_k.at_5,p_at_k.at_10)
                        except Exception as err: 
                            log.write('problem in calc_NDCG: in source: '+ curr_source+' in clm '+ claims_dict[clm]+" alpha:"+str(alpha_f)+ "beta:"+str(beta_f)+ " lambda:"+str(lambda_f)+" \n" )     
                            for arg in err.args:
                                log.write(arg+" ")      
                            log.write("\n")  
                    average_NDCG = float(float(sum(NDCG_all_claims.values()))/float(len(NDCG_all_claims))) #across all claims...
                    if average_NDCG > best_avg_nDCG:
                        best_avg_nDCG = average_NDCG
                        best_avg_nDCG_configuration = (alpha_f,beta_f,k_val,lambda_f)
                    MAP = float(float(sum(AP_all_claims.values()))/float(len(AP_all_claims)))                   
                    if MAP > best_MAP:
                        best_MAP = MAP
                        best_MAP_configuration = (alpha_f,beta_f,k_val,lambda_f)
                    p_at_k_avg.at_5 = float(float(p_at_k_avg.at_5)/float(len(AP_all_claims)))
                    p_at_k_avg.at_10 = float(float(p_at_k_avg.at_10)/float(len(AP_all_claims)))
                    if p_at_k_avg.at_5 > best_p_at_5.value:
                        best_p_at_5.value = p_at_k_avg.at_5
                        best_p_at_5.config = (alpha_f,beta_f,lambda_f)
                    if p_at_k_avg.at_10 > best_p_at_10.value:
                        best_p_at_10.value = p_at_k_avg.at_10
                        best_p_at_10.config = (alpha_f,beta_f,lambda_f)
                    each_params_AVGnDCG_MAP_prec_at_k_dict[alpha_f,beta_f,k_val,lambda_f] = (average_NDCG,MAP,p_at_k_avg.at_5,p_at_k_avg.at_10)
                    utils_linux.save_pickle(measures_res+"NDCG_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_at_"+str(p),NDCG_all_claims)
                    utils_linux.save_pickle(measures_res+"AP_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f),AP_all_claims)
                    utils_linux.save_pickle(measures_res+"prec_at_k_all_claims_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f),prec_at_k_all_claims)
                    res_file.write("alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"|"+"AnDCG_"+str(average_NDCG)+"_MAP_"+str(MAP)+"_prec_at_5_"+str(p_at_k_avg.at_5)+"_prec_at_10_"+str(p_at_k_avg.at_10)+"\n")
        save_pickle(measures_res+"NDCG_AP_prec_at_k_all_claims_all_param_values_top_k_docs_"+str(k_val)+"_at_"+str(p),NDCG_AP_prec_at_k_all_claims_all_param_values)
        save_pickle(measures_res+"each_params_AVGnDCG_MAP_prec_at_k_dict_top_k_docs_"+str(k_val)+"_at_"+str(p),each_params_AVGnDCG_MAP_prec_at_k_dict)
        best_row = "measure & value & alpha,beta,k,lambda "
        best_row += "best_avg_nDCG&"+'%.3f'%best_avg_nDCG+"&"+str(best_avg_nDCG_configuration[0])+","+str(best_avg_nDCG_configuration[1])+","+str(best_avg_nDCG_configuration[3])+","+str(best_avg_nDCG_configuration[2])
        best_row += "best_MAP&" +'%.3f'%best_MAP+"&"+str(best_MAP_configuration[0])+","+str(best_MAP_configuration[1])+","+str(best_MAP_configuration[3])+","+str(best_MAP_configuration[2])
        best_row += "best_prec_at_5&"+'%.3f'% +"&"+str(best_p_at_5.config[0])+","+str(best_p_at_5.config[1])+","+str(best_p_at_5.config[2])
        best_row += "best_prec_at_10&"+'%.3f'%best_p_at_10.value+"&"+str(best_p_at_10.config[0])+","+str(best_p_at_10.config[1])+","+str(best_p_at_10.config[2])
        res_file.write(best_row)
        res_file.close()
        log.close()

def calc_sen_ret_avg_nDCG_MAP_prec_at_k_adj_sen(p):
    claim_sen_true_relevance_dict = read_pickle("claim_sen_relevance_dict_"+curr_source) 
    claims_dict = read_pickle("claim_dict")
    final_sen_scores = base_path + "\\final_sen_scores_adjacent_sen_corpus_beta_"+str(corpus_beta)+"/"
    measures_res = base_path +"\\measures_res_adjacent_sen_corpus_beta_"+str(corpus_beta)+"/"
#     final_sen_scores = linux_base_path + "/final_sen_scores_adjacent_sen_corpus_beta_"+str(corpus_beta)+"/"
#     measures_res = linux_base_path+ "/measures_res_adjacent_sen_corpus_beta_"+str(corpus_beta)+"/"
#     p = 100
    top_k_docs_values = [50]
    
    each_params_measures_dict = {} #key is alpha,beta,k_docs,lambda and val is the avg nDCG and MAP across all claims together
    measures_val_all_claims_all_param_values = {}
    best_avg_nDCG = 0
    best_MAP = 0 #across all claims in a given configuration, find the max measures
    prec_at_k = rcdtype.recordtype('prec_at_k', 'at_5 at_10')
    prec_at_k_avg = rcdtype.recordtype('prec_at_k_avg', 'at_5 at_10')
    best_prec_at_k = rcdtype.recordtype('best_prec_at_k_avg', 'value config')
    measures_tuple = rcdtype.recordtype('measures_tuple', 'nDCG AP p_at_5 p_at_10')
    measures_tuple_sum_for_clms = rcdtype.recordtype('measures_tuple_avg', 'nDCG AP p_at_5 p_at_10')
#     p_at_k_avg = prec_at_k_avg(0,0)
    best_p_at_5 = best_prec_at_k(0,"")
    best_p_at_10 = best_prec_at_k(0,"")
    claim_list = ['4']
    #count the number of sentences that were retrived that are in the true data....sum for each claim, then average.
    #run in paraller for the alpha values
    alpha_range = [0.4]#,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for k_val in top_k_docs_values:
        for alpha_f in alpha_range:
            log = open("calc_measures_log_at_"+str(p)+"_alpha_"+str(alpha_f),"wb")
            res_file = open(measures_res+"adj_sen_measures_res_k_top_docs_"+str(k_val)+"_at_"+str(p)+"_alpha_"+str(alpha_f),"wb")
            for beta in range(8,9,1):
                for lambda_int in range(2,3,1):
                    for delta_1 in range(9,10,1):
                        for delta_2 in range(0,1,1):
                            if not delta_1+delta_2 >9: 
                                try:
                                    lambda_f = turn_to_float([lambda_int])
                                    (beta_f,delta_1_f,delta_2_f) = turn_to_float([beta,delta_1,delta_2])
                                    delta_3_f = 1-corpus_beta-delta_1_f-delta_2_f
                                    curr_baseline_final_sen_score_dict = read_pickle(final_sen_scores+"clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted")
                                    num_of_claims = len(curr_baseline_final_sen_score_dict.keys())
                                    measures_all_claims = {} #key is a claim, value is the nDCG,MAP, prec@5, prec@10
                                    
                                    AP_cut_off = 1000
                                    p_at_k_avg = prec_at_k_avg(0,0)
                                    measures_tuple_curr_conf_sum_clm = measures_tuple_sum_for_clms(0,0,0,0)
                                    for clm in claim_list:
#                                     for clm in curr_baseline_final_sen_score_dict.keys():
                                        try:
                                            measures_tuple_in_curr_conf = measures_tuple(0,0,0,0)
                                            measures_tuple_in_curr_conf.nDCG = utils_linux.calc_sen_emp_NDCG(curr_source,clm,curr_baseline_final_sen_score_dict[clm],claim_sen_true_relevance_dict[claims_dict[clm]],p)
                                            measures_tuple_curr_conf_sum_clm.nDCG += measures_tuple_in_curr_conf.nDCG
                                            measures_tuple_in_curr_conf.AP = utils_linux.calc_sen_AP_relevance(AP_cut_off,curr_source,clm,curr_baseline_final_sen_score_dict[clm],claim_sen_true_relevance_dict[claims_dict[clm]])
                                            measures_tuple_curr_conf_sum_clm.AP += measures_tuple_in_curr_conf.AP
                                            p_at_k = prec_at_k(utils_linux.calc_sen_precision_at_k(5, curr_baseline_final_sen_score_dict[clm], claim_sen_true_relevance_dict[claims_dict[clm]]),utils_linux.calc_sen_precision_at_k(10, curr_baseline_final_sen_score_dict[clm], claim_sen_true_relevance_dict[claims_dict[clm]]))
                                            measures_tuple_in_curr_conf.p_at_5 = p_at_k.at_5
                                            measures_tuple_in_curr_conf.p_at_10 = p_at_k.at_10
                                            p_at_k_avg.at_5 += p_at_k.at_5
                                            p_at_k_avg.at_10 += p_at_k.at_10
                                            measures_all_claims[clm] = (measures_tuple_in_curr_conf.nDCG,measures_tuple_in_curr_conf.AP,measures_tuple_in_curr_conf.p_at_5,measures_tuple_in_curr_conf.p_at_10)
                                            measures_val_all_claims_all_param_values[clm,alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f] = (measures_all_claims[clm][0],measures_all_claims[clm][1],p_at_k.at_5,p_at_k.at_10)

        #                                     NDCG_all_claims[clm] = utils_linux.calc_sen_emp_NDCG(curr_source,clm,curr_baseline_final_sen_score_dict[clm],claim_sen_true_relevance_dict[claims_dict[clm]],p)
        #                                     AP_all_claims[clm] = utils_linux.calc_sen_AP_relevance(AP_cut_off,curr_source,clm,curr_baseline_final_sen_score_dict[clm],claim_sen_true_relevance_dict[claims_dict[clm]])
                                            
        #                                     prec_at_k_all_claims[clm] = (p_at_k.at_5,p_at_k.at_10)
        #                                     measures_val_all_claims_all_param_values[clm,alpha_f,beta_f,delta_1_f,delta_2_f,k_val,lambda_f] = (NDCG_all_claims[clm],AP_all_claims[clm],p_at_k.at_5,p_at_k.at_10)
                                        except Exception as err: 
                                            log.write('problem in calc_NDCG: in source: '+ curr_source+' in clm '+ claims_dict[clm]+" alpha:"+str(alpha_f)+ "beta:"+str(beta_f)+ " lambda:"+str(lambda_f)+" \n" )     
                                            for arg in err.args:
                                                log.write(arg+" ")      
                                            log.write("\n")  
        #                             average_NDCG = float(float(sum(NDCG_all_claims.values()))/float(len(NDCG_all_claims))) #across all claims...
                                    average_NDCG = float(float(measures_tuple_curr_conf_sum_clm.nDCG)/float(num_of_claims)) #across all claims...
                                    if average_NDCG > best_avg_nDCG:
                                        best_avg_nDCG = average_NDCG
                                        best_avg_nDCG_configuration = (alpha_f,beta_f,delta_1_f,delta_2_f,k_val,lambda_f)
                                    MAP = float(float(measures_tuple_curr_conf_sum_clm.AP)/float(num_of_claims))                   
                                    if MAP > best_MAP:
                                        best_MAP = MAP
                                        best_MAP_configuration = (alpha_f,beta_f,delta_1_f,delta_2_f,k_val,lambda_f)
                                    p_at_k_avg.at_5 = float(float(p_at_k_avg.at_5)/float(num_of_claims))
                                    p_at_k_avg.at_10 = float(float(p_at_k_avg.at_10)/float(num_of_claims))
                                    if p_at_k_avg.at_5 > best_p_at_5.value:
                                        best_p_at_5.value = p_at_k_avg.at_5
                                        best_p_at_5.config = (alpha_f,beta_f,delta_1_f,delta_2_f,k_val,lambda_f)
                                    if p_at_k_avg.at_10 > best_p_at_10.value:
                                        best_p_at_10.value = p_at_k_avg.at_10
                                        best_p_at_10.config = (alpha_f,beta_f,delta_1_f,delta_2_f,k_val,lambda_f)
                                                
                                    res_file.write("alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"|"+"AnDCG_"+str(average_NDCG)+"_MAP_"+str(MAP)+"_prec_at_5_"+str(p_at_k_avg.at_5)+"_prec_at_10_"+str(p_at_k_avg.at_10)+"\n")
                                    utils_linux.save_pickle(measures_res+"measures_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f), measures_all_claims)
                                    each_params_measures_dict[alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f] = (average_NDCG,MAP,p_at_k_avg.at_5,p_at_k_avg.at_10)
                                except Exception as err: 
                                    sys.stderr.write('problem in alpha '+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f))
                                    log.write('problem in alpha '+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f))
                                    print err      
#     save_pickle(measures_res+"NDCG_AP_prec_at_k_all_claims_all_param_values_top_k_docs_"+str(k_val)+"_at_"+str(p),NDCG_AP_prec_at_k_all_claims_all_param_values)
#     save_pickle(measures_res+"each_params_AVGnDCG_MAP_prec_at_k_dict_top_k_docs_"+str(k_val)+"_at_"+str(p),each_params_AVGnDCG_MAP_prec_at_k_dict)   
    save_pickle(measures_res+"each_params_measures_avg_dict_top_k_docs_"+str(k_val)+"_at_"+str(p),each_params_measures_dict)
    save_pickle(measures_res+"each_params_per_claim_measure_val_top_k_docs_"+str(k_val)+"_at_"+str(p),measures_val_all_claims_all_param_values)
    best_row = " measure & value & alpha,beta,delta1,delta2,k,lambda"
    best_row += "best_avg_nDCG&"+'%.3f'%best_avg_nDCG+"&"+str(best_avg_nDCG_configuration[0])+","+str(best_avg_nDCG_configuration[1])+","+str(best_avg_nDCG_configuration[2])+","+str(best_avg_nDCG_configuration[3])+","+str(best_avg_nDCG_configuration[4])+","+str(best_avg_nDCG_configuration[5])
    best_row += "best_MAP&" +'%.3f'%best_MAP+"&"+str(best_MAP_configuration[0])+","+str(best_MAP_configuration[1])+","+str(best_MAP_configuration[2])+","+str(best_MAP_configuration[3])+","+str(best_MAP_configuration[4])+","+str(best_MAP_configuration[5])
    best_row += "best_prec_at_5&"+'%.3f'%best_p_at_5.value+"&"+str(best_p_at_5.config[0])+","+str(best_p_at_5.config[1])+","+str(best_p_at_5.config[2])+","+str(best_p_at_5.config[3])+","+str(best_p_at_5.config[4])+","+str(best_p_at_5.config[5])
    best_row += "best_prec_at_10&"+'%.3f'%best_p_at_10.value+"&"+str(best_p_at_10.config[0])+","+str(best_p_at_10.config[1])+","+str(best_p_at_10.config[2])+","+str(best_p_at_10.config[3])+","+str(best_p_at_10.config[4])+","+str(best_p_at_10.config[5])
    res_file.write(best_row)
    res_file.close()
    log.close()
      
def calc_intersection_true_data_baselines_data_cnt():
    """
    for each ranked list of sentences for alpha, beta (the lambda doesnt matter),
    calculate the numebr of sentences that in the true_pridiction list for the claim 
    """
#     final_sen_scores = linux_base_path + "/final_sen_scores/"
    final_sen_scores = base_path + "\\final_sen_scores\\"
    top_k_docs_values = [50]
    lambda_f = "1"
    claim_sen_true_relevance_dict = read_pickle("claim_sen_relevance_dict_"+curr_source)
    claims_dict =  read_pickle("claim_dict")
    intersection_count_avg_all_claims =0
    exclude = set(string.punctuation)
    
    for k_val in top_k_docs_values:
        intersection_count = 0
        for alpha in range(10,11,1): #change just for test!
            for beta in range(0,1,1):
                (alpha_f,beta_f) = turn_to_float([alpha,beta])
                curr_baseline_final_sen_score_dict = read_pickle(final_sen_scores+"clm_num_key_final_ranked_list_sen_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_sorted")
                for clm in curr_baseline_final_sen_score_dict.keys():
                    curr_clm_baseline_set = set()
                    curr_true_relevance_set = set()
                    for (sen,score) in curr_baseline_final_sen_score_dict[clm]:
                        sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
                        sen_no_space = sen_no_punc.replace(" ","")
                        curr_clm_baseline_set.add(sen_no_space)
                    for (sen,score) in claim_sen_true_relevance_dict[claims_dict[clm]]:
                        sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
                        sen_no_space = sen_no_punc.replace(" ","")
                        curr_true_relevance_set.add(sen_no_space) 
                    intersection_count += len(curr_clm_baseline_set.intersection(curr_true_relevance_set))
        intersection_count_avg_all_claims += float(float(intersection_count)/float(110))
    intersection_count_avg_all_claims = float(float(intersection_count_avg_all_claims)/float(len(curr_baseline_final_sen_score_dict.keys())))
    print "intersection_count_avg_all_claims:"+str(intersection_count_avg_all_claims)               

def calc_std_nDCG_AP_adj_sen(p):
#     measures_res = linux_base_path+ "/measures_res"+setup+"/"
    measures_res = base_path +"\\measures_res"+setup+"\\"
    k_val = 50
    measures_val_all_claims_all_param_values = read_pickle(measures_res+"each_params_per_claim_measure_val_top_k_docs_"+str(k_val)+"_at_"+str(p)) #key:clm,alpha_f,beta_f,k_val,lambda_f val nDCG_score,AP_score
    each_params_measures_dict = read_pickle(measures_res+"each_params_measures_avg_dict_top_k_docs_"+str(k_val)+"_at_"+str(p)) #key:alpha_f,beta_f,k_val,lambda_f
    measures_std = {} #key is a configuration quadruplet, value is the std of the measures
       
#     for k_val in top_k_docs_values:
    for alpha in range(0,11,1): #change just for test!
        for beta in range(0,10,1):
            for lambda_int in range(0,11,1):
                for delta_1 in range(0,10,1):
                        for delta_2 in range(0,10,1):
                            if not delta_1+delta_2 >9: 
                                lambda_f = turn_to_float([lambda_int])
                                (alpha_f,beta_f,delta_1_f,delta_2_f) = turn_to_float([alpha,beta,delta_1,delta_2])
                                delta_3_f = 0.9-delta_1_f-delta_2_f
                                curr_AP_var = 0
                                curr_nDCG_var = 0
                                curr_prec_at_5_var = 0
                                curr_prec_at_10_var = 0
                                for clm in claim_list:
                                    curr_nDCG_var += (measures_val_all_claims_all_param_values[str(clm),alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][0] - each_params_measures_dict[alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][0])**2
                                    curr_AP_var += (measures_val_all_claims_all_param_values[str(clm),alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][1] - each_params_measures_dict[alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][1])**2
                                    curr_prec_at_5_var += (measures_val_all_claims_all_param_values[str(clm),alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][2] - each_params_measures_dict[alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][2])**2
                                    curr_prec_at_10_var +=(measures_val_all_claims_all_param_values[str(clm),alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][3] - each_params_measures_dict[alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f][3])**2
                                curr_nDCG_std = float(float(math.sqrt(curr_nDCG_var))/float(len(claim_list)))
                                curr_AP_std = float(float(math.sqrt(curr_AP_var))/float(len(claim_list)))
                                curr_prec_at_5_std = float(float(math.sqrt(curr_prec_at_5_var))/float(len(claim_list)))
                                curr_prec_at_10_std =float(float(math.sqrt(curr_prec_at_10_var))/float(len(claim_list)))
                                measures_std[alpha_f,beta_f,delta_1_f,delta_2_f,delta_3_f,k_val,lambda_f] = (curr_nDCG_std,curr_AP_std,curr_prec_at_5_std,curr_prec_at_10_std)
    save_pickle(measures_res+"measures_std_for_each_configuration_k_top_docs_"+str(k_val)+"_at_"+str(p), measures_std)

def calc_std_nDCG_AP_corpus_smoothing(p):
    """
    for every configuration, calc the std of the measures
    """
    
#     nDCG_MAP_res = base_path +"\\nDCG_MAP_res\\"
    measures_res = linux_base_path+ "/measures_res"+setup+"/"
    k_val = 50
    NDCG_AP_all_claims_all_param_values = read_pickle(measures_res+"NDCG_AP_prec_at_k_all_claims_all_param_values_top_k_docs_"+str(k_val)+"_at_"+str(p)) #key:clm,alpha_f,beta_f,k_val,lambda_f val nDCG_score,AP_score
    each_params_AVGnDCG_MAP_dict = read_pickle(measures_res+"each_params_AVGnDCG_MAP_prec_at_k_dict_top_k_docs_"+str(k_val)+"_at_"+str(p)) #key:alpha_f,beta_f,k_val,lambda_f
    nDCG_MAP_std = {} #key is a configuration quadruplet, value is the std of the measures
    
    
    
#     for k_val in top_k_docs_values:
    for alpha in range(0,11,1): #change just for test!
        for beta in range(0,10,1):
            for lambda_int in range(0,11,1):
                lambda_f = turn_to_float([lambda_int])
                (alpha_f,beta_f) = turn_to_float([alpha,beta])
                curr_AP_var = 0
                curr_nDCG_var = 0
                curr_prec_at_5_var = 0
                curr_prec_at_10_var = 0
                for clm in claim_list:
                    curr_nDCG_var += (NDCG_AP_all_claims_all_param_values[str(clm),alpha_f,beta_f,k_val,lambda_f][0] - each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][0])**2
                    curr_AP_var += (NDCG_AP_all_claims_all_param_values[str(clm),alpha_f,beta_f,k_val,lambda_f][1] - each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][1])**2
                    curr_prec_at_5_var += (NDCG_AP_all_claims_all_param_values[str(clm),alpha_f,beta_f,k_val,lambda_f][2] - each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][2])**2
                    curr_prec_at_10_var +=(NDCG_AP_all_claims_all_param_values[str(clm),alpha_f,beta_f,k_val,lambda_f][3] - each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][3])**2
                curr_nDCG_std = float(float(math.sqrt(curr_nDCG_var))/float(len(claim_list)))
                curr_AP_std = float(float(math.sqrt(curr_AP_var))/float(len(claim_list)))
                curr_prec_at_5_std = float(float(math.sqrt(curr_prec_at_5_var))/float(len(claim_list)))
                curr_prec_at_10_std =float(float(math.sqrt(curr_prec_at_10_var))/float(len(claim_list)))
                nDCG_MAP_std[alpha_f,beta_f,k_val,lambda_f] = (curr_nDCG_std,curr_AP_std,curr_prec_at_5_std,curr_prec_at_10_std)
    save_pickle(measures_res+"nDCG_MAP_prec_at_k_std_for_each_configuration_k_top_docs_"+str(k_val)+"_at_"+str(p), nDCG_MAP_std)

def find_best_free_param_configuration_LOO_adj_sen(p):
    """
    1. Leave one out each claim,
    2. For every possible value of alpha,beta,lambda -  calc the nDCG,MAP on the train claims (without the left out)
    3. Find the configuration that maximises the measures
    4. report the meatures of th e left out claim with this configuration
    """

    measures_res = linux_base_path+ "/measures_res"+setup+"/"
#     measures_res = base_path +"\\measures_res"+setup+"\\"
#     nDCG_MAP_res = base_path +"\\nDCG_MAP_res\\"
    claim_dict = read_pickle("claim_dict")
    claim_num_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
#     claim_num_list = [4,47,53,58,7,54]
    best_configuration_for_nDCG_AP_prec_at_k_left_out_res = {} #key is left out claim and  and value is the alpha,beta,lambda configuration that led to best measures - avg nDCG and AP across the train claims
    measures_res_of_left_out_in_its_best_conf = {} #key - left out claim num, and value is the measures of it, in the best configuration without it.
    
    k_val = 50
    prec_at_k_train = rcdtype.recordtype('prec_at_k_train', 'at_5 at_10')
    max_prec_at_k = rcdtype.recordtype('max_prec_at_k', 'max_val max_conf')
    try:
        for left_out_claim_indx in range(len(claim_num_list)):
            max_nDCG = 0
            max_MAP = 0
            max_nDCG_conf = []
            max_MAP_conf = []
            max_prec_at_5 = max_prec_at_k(0,"")
            max_prec_at_10 = max_prec_at_k(0,"")
            
            left_out_claim_num = claim_num_list[left_out_claim_indx]
            temp_claim_num_list = claim_num_list[:]
            temp_claim_num_list.remove(left_out_claim_num)
            for alpha in range(0,7,1): #change just for test!
                for beta in range(0,10,1):
                    for lambda_int in range(0,11,1):
                        for delta_1 in range(0,10,1):
                            for delta_2 in range(0,10,1):
                                if not delta_1+delta_2 >9: 
                                    lambda_f = turn_to_float([lambda_int])
                                    (alpha_f,beta_f,delta_1_f,delta_2_f) = turn_to_float([alpha,beta,delta_1,delta_2])
                                    measures_all_claims = utils_linux.read_pickle(measures_res+"measures_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_delta1_"+str(delta_1_f)+"_delta2_"+str(delta_2_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f))
    
    #                                 AP_all_claims_curr_param_values = read_pickle(nDCG_MAP_res+"AP_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f))
    #                                 nDCG_all_claims_curr_param_values = read_pickle(nDCG_MAP_res+"NDCG_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_at_"+str(p))
    #                                 prec_at_k_all_claims_params_values = read_pickle(nDCG_MAP_res+"prec_at_k_all_claims_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f))
                                    avg_nDCG_on_train = 0
                                    MAP_on_train = 0
                                    p_at_k_train_avg = prec_at_k_train(0,0)
                                    for clm_num_train in temp_claim_num_list:
                                        avg_nDCG_on_train += measures_all_claims[str(clm_num_train)][0]
                                        MAP_on_train += measures_all_claims[str(clm_num_train)][1] #in this config' -> get the measures
                                        p_at_k_train_avg.at_5 += measures_all_claims[str(clm_num_train)][2]
                                        p_at_k_train_avg.at_10 += measures_all_claims[str(clm_num_train)][3]
                                    avg_nDCG_on_train = float(float(avg_nDCG_on_train)/float(len(temp_claim_num_list)))
                                    MAP_on_train = float(float(MAP_on_train)/float(len(temp_claim_num_list)))
                                    p_at_k_train_avg.at_5 = float(float(p_at_k_train_avg.at_5)/float(len(temp_claim_num_list)))
                                    p_at_k_train_avg.at_10 = float(float(p_at_k_train_avg.at_10)/float(len(temp_claim_num_list)))
                                    
                                    if avg_nDCG_on_train > max_nDCG:
                                        max_nDCG = avg_nDCG_on_train
                                        max_nDCG_conf = (alpha_f,beta_f,lambda_f,delta_1_f,delta_2_f)
                                    if MAP_on_train > max_MAP:
                                        max_MAP = MAP_on_train
                                        max_MAP_conf = (alpha_f,beta_f,lambda_f,delta_1_f,delta_2_f)
                                    if p_at_k_train_avg.at_5 > max_prec_at_5.max_val:
                                        max_prec_at_5.max_val = p_at_k_train_avg.at_5
                                        max_prec_at_5.max_conf = (alpha_f,beta_f,lambda_f,delta_1_f,delta_2_f)
                                    if p_at_k_train_avg.at_10 > max_prec_at_10.max_val:
                                        max_prec_at_10.max_val = p_at_k_train_avg.at_10
                                        max_prec_at_10.max_conf = (alpha_f,beta_f,lambda_f,delta_1_f,delta_2_f)
            best_configuration_for_nDCG_AP_prec_at_k_left_out_res[left_out_claim_num] = [(max_nDCG,max_nDCG_conf),(max_MAP,max_MAP_conf),(max_prec_at_5.max_val,max_prec_at_5.max_conf),(max_prec_at_10.max_val,max_prec_at_10.max_conf)]
        #finished leaving out,
        #now calculate the nDCG and MAP of the left out claims with its best configuration results
        avg_nDCG_on_left_out = 0
        MAP_on_left_out = 0
        avg_prec_at_5_on_left_out = 0
        avg_prec_at_10_on_left_out = 0
        for clm_num in claim_num_list:
            (best_alpha_nDCG,best_beta_nDCG,best_lambda_nDCG,best_delta1_nDCG,best_delta2_nDCG) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1]
            (best_alpha_MAP,best_beta_MAP,best_lambda_MAP,best_delta1_MAP,best_delta2_MAP) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1]
            (best_alpha_prec_at_5,best_beta_prec_at_5,best_lambda_prec_at_5,best_delta1_prec_at_5,best_delta2_prec_at_5) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1]
            (best_alpha_prec_at_10,best_beta_prec_at_10,best_lambda_prec_at_10,best_delta1_prec_at_10,best_delta2_prec_at_10) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1]
            #read the best config' dict
            best_config_of_nDCG_dict = read_pickle(measures_res+"measures_all_claims_alpha_"+str(best_alpha_nDCG)+"_beta_"+str(best_beta_nDCG)+"_delta1_"+str(best_delta1_nDCG)+"_delta2_"+str(best_delta2_nDCG)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_nDCG))
            best_config_of_AP_dict = read_pickle(measures_res+"measures_all_claims_alpha_"+str(best_alpha_MAP)+"_beta_"+str(best_beta_MAP)+"_delta1_"+str(best_delta1_MAP)+"_delta2_"+str(best_delta2_MAP)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_MAP))
            best_config_of_prec_at_5_dict = read_pickle(measures_res+"measures_all_claims_alpha_"+str(best_alpha_prec_at_5)+"_beta_"+str(best_beta_prec_at_5)+"_delta1_"+str(best_delta1_prec_at_5)+"_delta2_"+str(best_delta2_prec_at_5)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_prec_at_5)) #take only the first item in the tuple
            best_config_prec_of_at_10_dict = read_pickle(measures_res+"measures_all_claims_alpha_"+str(best_alpha_prec_at_10)+"_beta_"+str(best_beta_prec_at_10)+"_delta1_"+str(best_delta1_prec_at_10)+"_delta2_"+str(best_delta2_prec_at_10)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_prec_at_10)) #take only the second item in the tuple
            measures_res_of_left_out_in_its_best_conf[clm_num] = (best_config_of_nDCG_dict[str(clm_num)][0],best_config_of_AP_dict[str(clm_num)][1],best_config_of_prec_at_5_dict[str(clm_num)][2],best_config_prec_of_at_10_dict[str(clm_num)][3])
            avg_nDCG_on_left_out += best_config_of_nDCG_dict[str(clm_num)][0]
            MAP_on_left_out += best_config_of_AP_dict[str(clm_num)][1]
            avg_prec_at_5_on_left_out += best_config_of_prec_at_5_dict[str(clm_num)][2]
            avg_prec_at_10_on_left_out += best_config_prec_of_at_10_dict[str(clm_num)][3]
            
        save_pickle(measures_res+"measures_res_of_left_out_in_its_best_conf_k_top_docs_"+str(k_val)+"_at_"+str(p), measures_res_of_left_out_in_its_best_conf)
        #report the avg
        avg_nDCG_on_left_out = float(float(avg_nDCG_on_left_out)/float(len(claim_num_list)))          
        MAP_on_left_out = float(float(MAP_on_left_out)/float(len(claim_num_list)))          
        avg_prec_at_5_on_left_out = float(float(avg_prec_at_5_on_left_out)/float(len(claim_num_list)))
        avg_prec_at_10_on_left_out = float(float(avg_prec_at_10_on_left_out)/float(len(claim_num_list)))
        #write res to file:
        # claim text, the best nDCG conf and result on train, the nDCG it really has, and the same for AP
        with open(measures_res+"nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf_k_top_docs_"+str(k_val)+"_at_"+str(p)+".csv", 'wb') as csvfile:
                w = csv.writer(csvfile)
                row = "claim|best_nDCG|alpha,beta,lambda,delta_1,delta_2,delta_3|best_AP|alpha,beta,lambda,delta_1,delta_2,delta_3|best_prec_at_5|alpha,beta,lambda,delta_1,delta_2,delta_3|best_prec_at_10|alpha,beta,lambda,delta_1,delta_2,delta_3"
                w.writerow([row])
                for (clm_num,(nDCG,AP,prec_at_5,prec_at_10)) in measures_res_of_left_out_in_its_best_conf.items():
                    row = claim_dict[str(clm_num)]+"&"+'%.3f'%nDCG+"&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][2])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][3])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][4])+","+str(0.9-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][4]-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][3])
                    row += "&"+'%.3f'%AP+"&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][2])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][3])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][4])+","+str(0.9-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][4]-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][3])
                    row += "&"+'%.3f'%prec_at_5+ "&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][2])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][3])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][4])+","+str(0.9-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][4]-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][3])
                    row += "&"+'%.3f'%prec_at_10+ "&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][2])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][3])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][4])+","+str(0.9-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][4]-best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][3])
                    w.writerow([row])
                w.writerow(["avg_nDCG_on_left_out: "+ '%.4f'%avg_nDCG_on_left_out ])
                w.writerow(["MAP_on_left_out: "+ '%.4f'%MAP_on_left_out])
                w.writerow(["avg_prec_at_5_on_left_out: "+ '%.4f'%avg_prec_at_5_on_left_out])
                w.writerow(["avg_prec_at_10_on_left_out: "+ '%.4f'%avg_prec_at_10_on_left_out])
    except Exception as err: 
                sys.stderr.write('problem in LOO')     
                print err                
                
def find_best_free_param_configuration_LOO_corpus_smoothing(p,alpha_range,beta_range,lambda_range):
    """
    1. Leave one out each claim,
    2. For every possible value of alpha,beta,lambda -  calc the nDCG,MAP on the train claims (without the left out)
    3. Find the configuration that maximises the measures
    4. report the meatures of th e left out claim with this configuration
    """
    measures_res = linux_base_path+ "/measures_res_corpus_smoothing_corpus_beta_"+str(corpus_beta)+"/"
#     measures_res = "/IBM_STORAGE/USERS_DATA/liorab/baseline_ret/sen_ret_corpus_smoothing_res/nDCG_MAP_res_corpus_smoothing/"
#     measures_res = base_path +"\\measures_res\\"
    claim_dict = read_pickle("claim_dict")
    claim_num_list = [4,7,17,21,36,37,39,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
#     claim_num_list = [4,47,53,58,7,54]
    best_configuration_for_nDCG_AP_prec_at_k_left_out_res = {} #key is left out claim and  and value is the alpha,beta,lambda configuration that led to best measures - avg nDCG and AP across the train claims
    nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf = {} #key - left out claim num, and value is the nDCG and AP of it, in the best configuration without it.
    
    k_val = 50
    prec_at_k_train = rcdtype.recordtype('prec_at_k_train', 'at_5 at_10')
    max_prec_at_k = rcdtype.recordtype('max_prec_at_k', 'max_val max_conf')
    
    for left_out_claim_indx in range(len(claim_num_list)):
        max_nDCG = 0
        max_MAP = 0
        max_nDCG_conf = []
        max_MAP_conf = []
        max_prec_at_5 = max_prec_at_k(0,"")
        max_prec_at_10 = max_prec_at_k(0,"")
        
        left_out_claim_num = claim_num_list[left_out_claim_indx]
        temp_claim_num_list = claim_num_list[:]
        temp_claim_num_list.remove(left_out_claim_num)
        for alpha in range(alpha_range.start,alpha_range.end,1): #change just for test!
            for beta in range(beta_range.start,beta_range.end,1):
                for lambda_int in range(lambda_range.start,lambda_range.end,1):
                    lambda_f = turn_to_float([lambda_int])
                    (alpha_f,beta_f) = turn_to_float([alpha,beta])
                    AP_all_claims_curr_param_values = read_pickle(measures_res+"AP_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f))
                    nDCG_all_claims_curr_param_values = read_pickle(measures_res+"NDCG_all_claims_alpha_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f)+"_at_"+str(p))
                    prec_at_k_all_claims_params_values = read_pickle(measures_res+"prec_at_k_all_claims_"+str(alpha_f)+"_beta_"+str(beta_f)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(lambda_f))
                    avg_nDCG_on_train = 0
                    MAP_on_train = 0
                    p_at_k_train_avg = prec_at_k_train(0,0)
                    for clm_num_train in temp_claim_num_list:
#                         avg_nDCG_on_train += nDCG_AP_all_claims_all_param_values[str(clm_num_train),alpha_f,beta_f,lambda_f][0]
#                         MAP_on_train += nDCG_AP_all_claims_all_param_values[str(clm_num_train),alpha_f,beta_f,lambda_f][1]
                        avg_nDCG_on_train += nDCG_all_claims_curr_param_values[str(clm_num_train)]
                        MAP_on_train += AP_all_claims_curr_param_values[str(clm_num_train)] #in this config' -> get the measures
                        p_at_k_train_avg.at_5 += prec_at_k_all_claims_params_values[str(clm_num_train)][0]
                        p_at_k_train_avg.at_10 += prec_at_k_all_claims_params_values[str(clm_num_train)][1]
                    avg_nDCG_on_train = float(float(avg_nDCG_on_train)/float(len(temp_claim_num_list)))
                    MAP_on_train = float(float(MAP_on_train)/float(len(temp_claim_num_list)))
                    p_at_k_train_avg.at_5 = float(float(p_at_k_train_avg.at_5)/float(len(temp_claim_num_list)))
                    p_at_k_train_avg.at_10 = float(float(p_at_k_train_avg.at_10)/float(len(temp_claim_num_list)))
                    
                    if avg_nDCG_on_train > max_nDCG:
                        max_nDCG = avg_nDCG_on_train
                        max_nDCG_conf = (alpha_f,beta_f,lambda_f)
                    if MAP_on_train > max_MAP:
                        max_MAP = MAP_on_train
                        max_MAP_conf = (alpha_f,beta_f,lambda_f)
                    if p_at_k_train_avg.at_5 > max_prec_at_5.max_val:
                        max_prec_at_5.max_val = p_at_k_train_avg.at_5
                        max_prec_at_5.max_conf = (alpha_f,beta_f,lambda_f)
                    if p_at_k_train_avg.at_10 > max_prec_at_10.max_val:
                        max_prec_at_10.max_val = p_at_k_train_avg.at_10
                        max_prec_at_10.max_conf = (alpha_f,beta_f,lambda_f)
        best_configuration_for_nDCG_AP_prec_at_k_left_out_res[left_out_claim_num] = [(max_nDCG,max_nDCG_conf),(max_MAP,max_MAP_conf),(max_prec_at_5.max_val,max_prec_at_5.max_conf),(max_prec_at_10.max_val,max_prec_at_10.max_conf)]
    #finished leaving out,
    #now calculate the nDCG and MAP of the left out claims with its best configuration results
    avg_nDCG_on_left_out = 0
    MAP_on_left_out = 0
    avg_prec_at_5_on_left_out = 0
    avg_prec_at_10_on_left_out = 0
    #17/11/14 update
    avg_nDCG_on_left_out_based_on_best_AP_conf = 0
    avg_prec_at_5_on_left_out_based_on_best_AP_conf = 0
    avg_prec_at_10_on_left_out_based_on_best_AP_conf = 0
    #end update
    for clm_num in claim_num_list:
        (best_alpha_nDCG,best_beta_nDCG,best_lambda_nDCG) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1]
        (best_alpha_MAP,best_beta_MAP,best_lambda_MAP) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1]
        (best_alpha_prec_at_5,best_beta_prec_at_5,best_lambda_prec_at_5) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1]
        (best_alpha_prec_at_10,best_beta_prec_at_10,best_lambda_prec_at_10) = best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1]
        
#         nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf[clm_num] = (nDCG_AP_all_claims_all_param_values[str(clm_num),best_alpha_nDCG,best_beta_nDCG,best_lambda_nDCG][0],nDCG_AP_all_claims_all_param_values[str(clm_num),best_alpha_MAP,best_beta_MAP,best_lambda_MAP][1])
#         avg_nDCG_on_left_out += nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf[clm_num][0]
#         MAP_on_left_out += nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf[clm_num][1]
        #read the best config' dict
        best_config_nDCG_dict = read_pickle(measures_res+"NDCG_all_claims_alpha_"+str(best_alpha_nDCG)+"_beta_"+str(best_beta_nDCG)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_nDCG)+"_at_"+str(p))
        best_config_AP_dict = read_pickle(measures_res+"AP_all_claims_alpha_"+str(best_alpha_MAP)+"_beta_"+str(best_beta_MAP)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_MAP))
        best_config_prec_at_5_dict = read_pickle(measures_res+"prec_at_k_all_claims_"+str(best_alpha_prec_at_5)+"_beta_"+str(best_beta_prec_at_5)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_prec_at_5)) #take only the first item in the tuple
        best_config_prec_at_10_dict = read_pickle(measures_res+"prec_at_k_all_claims_"+str(best_alpha_prec_at_10)+"_beta_"+str(best_beta_prec_at_10)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_prec_at_10)) #take only the second item in the tuple
        # 17/11/14 update - report the p@5,p@10, and nDCG in the configuration that is the best for AP
        nDCG_from_best_AP_conf_dict = read_pickle(measures_res+"NDCG_all_claims_alpha_"+str(best_alpha_MAP)+"_beta_"+str(best_beta_MAP)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_MAP)+"_at_"+str(p))
        prec_at_k_from_best_AP_conf_dict = read_pickle(measures_res+"prec_at_k_all_claims_"+str(best_alpha_MAP)+"_beta_"+str(best_beta_MAP)+"_top_k_docs_"+str(k_val)+"_lambda_"+str(best_lambda_MAP))
        # 17/11/14 update - add the last three values ->the nDCG and prec@k from the conf best for AP
        nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf[clm_num] = (best_config_nDCG_dict[str(clm_num)],best_config_AP_dict[str(clm_num)],best_config_prec_at_5_dict[str(clm_num)][0],best_config_prec_at_10_dict[str(clm_num)][1],nDCG_from_best_AP_conf_dict[str(clm_num)],prec_at_k_from_best_AP_conf_dict[str(clm_num)][0],prec_at_k_from_best_AP_conf_dict[str(clm_num)][1])
        avg_nDCG_on_left_out += best_config_nDCG_dict[str(clm_num)]
        MAP_on_left_out += best_config_AP_dict[str(clm_num)]
        avg_prec_at_5_on_left_out += best_config_prec_at_5_dict[str(clm_num)][0]
        avg_prec_at_10_on_left_out += best_config_prec_at_10_dict[str(clm_num)][1]
        #17/11/14 update
        avg_nDCG_on_left_out_based_on_best_AP_conf += nDCG_from_best_AP_conf_dict[str(clm_num)]
        avg_prec_at_5_on_left_out_based_on_best_AP_conf += prec_at_k_from_best_AP_conf_dict[str(clm_num)][0]
        avg_prec_at_10_on_left_out_based_on_best_AP_conf += prec_at_k_from_best_AP_conf_dict[str(clm_num)][1]
        #end update
    save_pickle(measures_res+"nDCG_AP_res_of_left_out_in_its_best_conf_k_top_docs_"+str(k_val)+"_at_"+str(p), nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf)
    save_pickle(measures_res+"best_configuration_for_nDCG_AP_prec_at_k_left_out_res_k_top_docs_"+str(k_val)+"_at_"+str(p),best_configuration_for_nDCG_AP_prec_at_k_left_out_res)
    #report the avg
    avg_nDCG_on_left_out = float(float(avg_nDCG_on_left_out)/float(len(claim_num_list)))          
    MAP_on_left_out = float(float(MAP_on_left_out)/float(len(claim_num_list)))          
    avg_prec_at_5_on_left_out = float(float(avg_prec_at_5_on_left_out)/float(len(claim_num_list)))
    avg_prec_at_10_on_left_out = float(float(avg_prec_at_10_on_left_out)/float(len(claim_num_list)))
    #17/11/14 update
    avg_nDCG_on_left_out_based_on_best_AP_conf = float(float(avg_nDCG_on_left_out_based_on_best_AP_conf)/float(len(claim_num_list)))
    avg_prec_at_5_on_left_out_based_on_best_AP_conf = float(float(avg_prec_at_5_on_left_out_based_on_best_AP_conf)/float(len(claim_num_list)))
    avg_prec_at_10_on_left_out_based_on_best_AP_conf = float(float(avg_prec_at_10_on_left_out_based_on_best_AP_conf)/float(len(claim_num_list)))
    #end update
    
    #write res to file:
    # claim text, the best nDCG conf and result on train, the nDCG it really has, and the same for AP
    with open(measures_res+"nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf_k_top_docs_"+str(k_val)+"_at_"+str(p)+".csv", 'wb') as csvfile:
        w = csv.writer(csvfile, delimiter='&', dialect='excel')
        row = "claim&best_conf_nDCG&best_conf_for_nDCG&best_conf_AP&best_conf_for_AP&best_prec_at_5&best_prec_at_5&best_prec_at_10&best_conf_for_prec_at_10&nDCG_in_best_AP_conf&prec_at_5_in_best_AP_conf&prec_at_10_in_best_AP_conf"
        w.writerow([row])
        for (clm_num,(nDCG,AP,prec_at_5,prec_at_10,nDCG_based_on_best_AP_conf,p_at_5_based_on_best_AP_conf,p_at_10_based_on_best_AP_conf)) in nDCG_AP_prec_at_k_res_of_left_out_in_its_best_conf.items():
            row = claim_dict[str(clm_num)]+"&"+'%.3f'%nDCG+"&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][0][1][2])
            row += "&"+'%.3f'%AP+"&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][1][1][2])
            row += "&"+'%.3f'%prec_at_5+ "&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][2][1][2])
            row += "&"+'%.3f'%prec_at_10+ "&"+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][0])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][1])+","+str(best_configuration_for_nDCG_AP_prec_at_k_left_out_res[clm_num][3][1][2])
            row += "&"+'%.3f'%nDCG_based_on_best_AP_conf+"&"+'%.3f'%p_at_5_based_on_best_AP_conf+"&"+'%.3f'%p_at_10_based_on_best_AP_conf
            w.writerow([row])
        w.writerow(["avg_nDCG_on_left_out: "+ '%.4f'%avg_nDCG_on_left_out ])
        w.writerow(["MAP_on_left_out: "+ '%.4f'%MAP_on_left_out])
        w.writerow(["avg_prec_at_5_on_left_out: "+ '%.4f'%avg_prec_at_5_on_left_out])
        w.writerow(["avg_prec_at_10_on_left_out: "+ '%.4f'%avg_prec_at_10_on_left_out])
        w.writerow(["avg_nDCG_on_left_out_based_on_best_AP_conf: "+ '%.4f'%avg_nDCG_on_left_out_based_on_best_AP_conf])
        w.writerow(["avg_prec_at_5_on_left_out_based_on_best_AP_conf: "+ '%.4f'%avg_prec_at_5_on_left_out_based_on_best_AP_conf])
        w.writerow(["avg_prec_at_10_on_left_out_based_on_best_AP_conf: "+ '%.4f'%avg_prec_at_10_on_left_out_based_on_best_AP_conf])
            
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

def claim_document_baseline():
    """
    After results, compare to different setups with the alpha,beta weighting settings.
    For each setting:
    1. Perform an overfit evaluation (go over all the other params value and find the highest measures value)
    2.LOO process (set the value as constant, and for all the other params value do  leave-one-out process 
    the settings:
    1. beta=0 
    2. beta=alpha=0
    3.beta=0,alpha=1
    4. beta=0.9,alpha=0
    5. alpha=1, beta=0.9
    6. alpha=1
    7. alpha=0
    """
    param_range = namedtuple('param_range','start, end')
    alpha_settings = param_range(0,1)#,param_range(10,11)
    beta_settings = param_range(0,1)#,param_range(9,10)] #as it finished at beta=0.9
    lambda_settings = param_range(0,11)
    
    claim_document_baseline_LOO(alpha_settings,beta_settings,lambda_settings)
    
    
def claim_document_baseline_LOO(alpha_setting, beta_setting, lambda_settings):
    """
    call the loo function
    """
    find_best_free_param_configuration_LOO_corpus_smoothing(10, alpha_setting, beta_setting, lambda_settings) 
# def claim_document_baseline_overfit():
      
def main():
    try:
#         create_entity_claim_input_file_doc_ret()
#         normalize_doc_scores()
#         get_top_k_docs()
#         normalize_sen_scores()
#         merge_all_claims_norm_dicts_for_docs_and_sen()
#         interpolate_doc_sen_score()
#         convert_ranked_sen_keys_to_list_of_sen()
#         process_baseline_sen_ret_result()
#         convert_true_support_to_relevance()
#         process_baseline_sen_ret_result()
#         calc_std_nDCG_AP()
#         test()
#         calc_intersection_true_data_baselines_data_cnt()
#         create_rel_doctitle_dict()
#             create_list_of_retrieved_docs()
#         calc_doc_ret_MAP()
#         interpolate_doc_sen_score_adj_sen()
#         interpolate_doc_sen_score_corpus_smoothing()
#         normalize_sen_scores_adj_sen()
#         normalize_sen_scores_corpus_smoothing()
#         merge_all_claims_norm_dicts_for_sen_corpus_smoothing()
#         merge_all_claims_norm_dicts_for_sen_adj_sen()
        claim_document_baseline
    except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err     
        


if __name__ == '__main__':
    main()
    
    