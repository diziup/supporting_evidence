'''
Created on Sep 9, 2014
A one stage model for ranking supporting evidence per claim

Using LTR (Learning to rank) such as SRM-rank,
Features: Sentiment Similarity (a cosine score)
Semantic similarity (a cosine score)
Objective sentences LM - an affiliation for any of the 5 options (so this feature can take value 1,2,...5
Entailment- to add. an score of entailemt (positive means entailment, negative means no entailment)

@author: Liora Braunstein
'''

from my_utils import utils
import sys
import os
import csv
import collections
import random
import copy
import numpy as np
import string

features_str = ""
CV_method ="LOOCV"
learner ="SVM_rank"
similarity_function = "JSD"
sentiment_model = "orig"
obj_LM = "dist"
semantic_sim = "additive"
semantic_sim_entity_removal= "entity_presence" 
if semantic_sim_entity_removal == "entity_remove" :
    features_list = ["sen_sim","sem_sim_"+semantic_sim+"_"+semantic_sim_entity_removal,"objective_LM_"+obj_LM,"entity_presence"]
else:
    features_list = ["sen_sim","sem_sim_"+semantic_sim,"objective_LM_"+obj_LM,"entity_presence"]

# features_list = ["sen_sim","sem_sim","entity_presence"]

features_str = "_".join(features_list)

# features_str ="sen_sim_sem_sim_objective_LM_"+obj_LM+"_entity_presence" 
kernel= "linear"
supp_scale = "zero_to_two" #22.09 update  -  scores are 0 - not support , 1 is support, 2 is strong support
setup ="separate" #unified collection, or separate -  RT or wiki

# curr_features_path = learner+"_"+features_str+"_"+sentiment_model+"_"+similarity_function+"_"+kernel
curr_features_path = learner+"_"+features_str+"_"+kernel
# curr_features_path = r'SVM_rank_sen_sim_orig_JSD_sem_sim_objective_LM_entity_presence_linear_kernel'
train_path = r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\"+supp_scale+r"_scale_SVM_res\\"+curr_features_path+r"\\train\\"
test_path = r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\"+supp_scale+r"_scale_SVM_res\\"+curr_features_path+r"\\test\\"
model_path =  r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\"+supp_scale+r"_scale_SVM_res\\"+curr_features_path+r"\\model\\"
prediction_path = r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\"+supp_scale+r"_scale_SVM_res\\"+curr_features_path+r"\\prediction\\"
# prediction_path = r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\"+supp_scale+r"_scale_SVM_res\\"+curr_features_path+r"\\prediction\\"

exclude = set(string.punctuation)



"""
For SVM-rank create an input file 
1. For every claim and sentence pair create a feature file for SVM rank: 
<line> .support score by majority. <target> clm_num:<qid> <sen_sim>:<value> <semnatic_sim>:<value>, <movie_star_affiliation>:<value> # <clm_text sen text>
2. Use LOOCV -  so create a dict - key is clm and sen, value is a list of features
3. For every clm and sen pair, create a different training file - remove a pair i, train on all but it, and test on it.
"""

def convert_support_dict():
    #21.09 update -convert the scores to a 0-2 scale-0 zero in everyhing that is not support, and 1 is support,2 is strong support
     
    clm_sen_support_ranking= utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\clm_sen_support_ranking_sorted_full")#for each clm and sen, the support score, sorted.
    clm_sen_support_ranking_zero_to_five_scores_scale_converted = {}
    clm_sen_support_ranking_zero_to_two_scores_scale_converted = {} 
    
    for (clm,sen,supp_score) in clm_sen_support_ranking.keys():
        clm_sen_support_ranking_zero_to_five_scores_scale_converted[(clm,sen)]=supp_score
        if supp_score == 0 or supp_score == 1 or supp_score == 2 or supp_score == 3:
            clm_sen_support_ranking_zero_to_two_scores_scale_converted[(clm,sen)] = 0
        elif supp_score == 4:
            clm_sen_support_ranking_zero_to_two_scores_scale_converted[(clm,sen)] = 1
        elif supp_score == 5:
            clm_sen_support_ranking_zero_to_two_scores_scale_converted[(clm,sen)] = 2
    
            
#     utils.save_pickle("clm_sen_support_ranking_"+supp_scale+"_clm_sen_key_supp_score_value", clm_sen_support_ranking_zero_to_five_scores_scale_converted)
    utils.save_pickle("clm_sen_support_ranking_"+supp_scale+"_clm_sen_key_supp_score_value", clm_sen_support_ranking_zero_to_two_scores_scale_converted)
    
    save_to_csv_file(clm_sen_support_ranking_zero_to_five_scores_scale_converted, "clm_sen_support_ranking_clm_sen_key_supp_score_value.csv")
    save_to_csv_file(clm_sen_support_ranking_zero_to_two_scores_scale_converted, "clm_sen_support_ranking_zero_to_two_scores_scale_converted.csv")
        
def save_to_csv_file(d,file_name):      #save to file
        with open(file_name, 'wb') as csvfile:
            w = csv.writer(csvfile)
            for sen in d.items():
                w.writerow([sen])    

def features_max_min_normalization():
    """
    from each collection  (separated =wiki and RT or unified), for each claim and its sentences, and for each feature,
    normalize each pair of clm and sen in the max-min value of that feature.
    Since I am performing a LOOCV, I will find the max and min for each LOO setup, and normalize the test with the corresponding value.
    This is opposed to finding the max-min across all the data and norm with it since I am not supposed to know anything about the test data after training...
    stages:
    1. Perform LOO process
    2. save for each left_out_clm, the max-min.
    """
    
    claim_num_and_text=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict_pickle')
    if setup == "separate":
        clm_and_sen_feature_vector_RT = {}
        clm_and_sen_feature_vector_wiki = {}
        clm_and_sen_feature_vector_RT = utils.read_pickle("clm_and_sen_feature_vector_"+features_str+"_RT")
        clm_and_sen_feature_vector_wiki = utils.read_pickle("clm_and_sen_feature_vector_"+features_str+"_wiki")
        left_out_max_min_val_dict_wiki = {} # key is claim number that is now left out, value is a list: for each feature, the max-min value in the training data -  without this claim
        left_out_max_min_val_dict_RT = {}  
        separated_dicts_list = [(clm_and_sen_feature_vector_RT,left_out_max_min_val_dict_RT), (clm_and_sen_feature_vector_wiki,left_out_max_min_val_dict_wiki)]    
        feature_values_list = d = [[] for x in xrange(len(clm_and_sen_feature_vector_RT.values()[0][1]))]  #the first value in the features is the true supp score, which is not relevant for the norm
#         min_feature_list = d = [[] for x in xrange(len(clm_and_sen_feature_vector_RT.values()[0][1]))]
                
        for (curr_source_features_dict, curr_source_max_min_val) in separated_dicts_list:
            for (out_clm_num, out_clm_text) in claim_num_and_text.items():
                for (curr_clm_num,curr_clm,curr_sen) in curr_source_features_dict.keys():   
                    if int(out_clm_num) != curr_clm_num: #training data
                        features_data = curr_source_features_dict[(curr_clm_num,curr_clm,curr_sen)][1] #the features list
                        for feature_idx in range(0,len(features_data)):
#                             if features_data[feature_idx] > max_feature_list[feature_idx]:
                                feature_values_list[feature_idx].append(features_data[feature_idx])
#                             if features_data[feature_idx] < min_feature_list[feature_idx]:
#                                 min_feature_list[feature_idx].append(features_data[feature_idx]) 
            #find the max and min from each feature list 
                max_feature_list = [max(curr_feature_list) for curr_feature_list in feature_values_list ]
                min_feature_list = [min(curr_feature_list) for curr_feature_list in feature_values_list ]
                curr_source_max_min_val[out_clm_num] =[float(max_i - min_i) for max_i, min_i in zip(max_feature_list, min_feature_list)]             
        
        utils.save_pickle("left_out_max_min_val_dict_"+features_str+"_RT", left_out_max_min_val_dict_RT)     
        utils.save_pickle("left_out_max_min_val_dict_"+features_str+"_wiki",left_out_max_min_val_dict_wiki)
        
                
def  create_features_dict_for_input_files():
    """
    read the different features dict, according to current setup, and create a dict of clm_num, clm_text, sen and its correspinding features list for the SVM.
    
    """
    clm_and_sen_feature_vector = {}#input_for_SVM_rank
    if setup == "separate":
        clm_and_sen_feature_vector_wiki = {}
        clm_and_sen_feature_vector_RT = {}
    clm_sen_sentiment_sim_score = {}
    clm_sen_support_ranking = utils.read_pickle("clm_sen_support_ranking_"+supp_scale+"_clm_sen_key_supp_score_value")
    if "sen_sim" in features_list:
            clm_sen_sentiment_sim_score = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\claim_sen_sentiment_"+similarity_function+"_simialrity_socher_"+sentiment_model+"_sorted")
    if "sem_sim_"+semantic_sim in features_list:
        clm_sen_semantic_sim_score_cosine = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\claim_sen_VSM_similarity_sorted_word2vec_"+semantic_sim+"_300")
    elif "sem_sim_"+semantic_sim+"_"+ semantic_sim_entity_removal in features_list:
            clm_sen_semantic_sim_score_cosine = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\claim_sen_VSM_similarity_sorted_word2vec_"+semantic_sim+"_300_"+semantic_sim_entity_removal)           
    if "objective_LM_"+obj_LM in features_list: 
        sen_obj_LM_dict = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\objective_sentences\sen_obj_LM_"+obj_LM+"_dict")
    if "entity_presence" in features_list: 
        entity_name_presence_dict = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\clm_sen_doc_title_entity_presence_flag")   
    
    claim_num_and_text=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict')
        
    for (clm_num,clm_text) in claim_num_and_text.items():
        try:
            curr_clm_and_sens = utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\clm_'+clm_num+'_clm_text_sen_text_dict')
        except Exception as err: 
            sys.stderr.write('problem in curr_clm_and_sens:'+clm_text )     
            print err.args      
            print err
            continue
        del curr_clm_and_sens[0] 
        for sen in curr_clm_and_sens.values():
            try:
                features_vector = []
                if "sen_sim" in features_list:
                    features_vector.append(clm_sen_sentiment_sim_score[(clm_text,sen)])
                if "sem_sim_"+semantic_sim in features_list:
                    features_vector.append(clm_sen_semantic_sim_score_cosine[(clm_text,sen)])
                elif "sem_sim_"+semantic_sim+"_"+ semantic_sim_entity_removal in features_list:
                    features_vector.append(clm_sen_semantic_sim_score_cosine[(clm_text,sen)][2])
                if "objective_LM_"+obj_LM in features_list:
                    if obj_LM == "label":
                        features_vector.append(sen_obj_LM_dict[sen])
                    else:
                        one,two,three,four,five  = sen_obj_LM_dict[sen]
                        for val in [ one,two,three,four,five]:
                            features_vector.append(val)
                if "entity_presence" in features_list:
                    features_vector.append(entity_name_presence_dict[clm_text,sen][1])
                                                                       
                clm_and_sen_feature_vector[(int(clm_num),clm_text,sen)] = (clm_sen_support_ranking[(clm_text,sen)],
                                                                           features_vector)
                if setup == "separate":
                    if  curr_clm_and_sens.values().index(sen) %2 == 0: #odd sen is from wiki:
                        clm_and_sen_feature_vector_wiki[(int(clm_num),clm_text,sen)] = (clm_sen_support_ranking[(clm_text,sen)],
                                                                           features_vector)
                    else: #RT sen
                        clm_and_sen_feature_vector_RT[(int(clm_num),clm_text,sen)] = (clm_sen_support_ranking[(clm_text,sen)],
                                                                           features_vector)
            except Exception as err: 
                    sys.stderr.write('problem in create_input_files_SVM_Rank with:'+ clm_text+ " and sen"+ sen)     
                    print err.args      
                    print err    
    utils.save_pickle("clm_and_sen_feature_vector_"+features_str+"_RT", clm_and_sen_feature_vector_RT)
    utils.save_pickle("clm_and_sen_feature_vector_"+features_str+"_wiki",clm_and_sen_feature_vector_wiki)
    
def create_input_files_SVM_Rank():
    """
    1. Create a dict: key is clm and sen pair, value is a tuple -first value is the support score, second is a list of the features values
    """
    clm_and_sen_feature_vector = {}#input_for_SVM_rank
        #23.09 update -if setup is separated collections, add two more dicts -for wiki sen and RT sen. 
    if setup == "separate":
        clm_and_sen_feature_vector_wiki = {}
        clm_and_sen_feature_vector_RT = {}
    claim_num_and_text=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict_pickle')
                         
    if setup is "separate":
        clm_and_sen_feature_vector_RT = utils.read_pickle("clm_and_sen_feature_vector_"+features_str+"_RT")
        clm_and_sen_feature_vector_wiki = utils.read_pickle("clm_and_sen_feature_vector_"+features_str+"_wiki")
        #need to sort according to increasing claim num, for SVM train:
        clm_and_sen_feature_vector_sorted_by_clm_num_wiki = collections.OrderedDict(sorted(clm_and_sen_feature_vector_wiki.items(),key=lambda x: (-int(x[0][0])), reverse=True))
        clm_and_sen_feature_vector_sorted_by_clm_num_RT = collections.OrderedDict(sorted(clm_and_sen_feature_vector_RT.items(),key=lambda x: (-int(x[0][0])), reverse=True))
        left_out_max_min_val_dict_RT = utils.read_pickle("left_out_max_min_val_dict_"+features_str+"_RT")     
        left_out_max_min_val_dict_wiki = utils.read_pickle("left_out_max_min_val_dict_"+features_str+"_wiki")
    else:
        clm_and_sen_feature_vector = utils.read_pickle("clm_and_sen_feature_vector_"+features_str+"unified")
        clm_and_sen_feature_vector_sorted_by_clm_num = collections.OrderedDict(sorted(clm_and_sen_feature_vector.items(),key=lambda x: (-int(x[0][0])), reverse=True))
    
    #create a dict -  key is claim text, value is claim number, for the train data files for SVM rank
    #all the clm and sen pairs are with their feature values, now write files for LOOCV
    #for each claim - leave it out, create a train file with all but it, and a test file with it.
    if setup == "separate": #For a specific feature setup, have two different folders - for the wiki sen and RT sen
        separated_dict_list = [ (clm_and_sen_feature_vector_sorted_by_clm_num_RT,left_out_max_min_val_dict_RT,"RT"),(clm_and_sen_feature_vector_sorted_by_clm_num_wiki,left_out_max_min_val_dict_wiki,"wiki")]
        for (sep_features_dict,sep_max_min_val_dict,source) in separated_dict_list:
            for (out_clm_num,out_clm_text) in claim_num_and_text.items():
                curr_test_LOOCV = open (test_path+"test_clm_num_"+str(out_clm_num)+"_CV_"+source, 'wb')
                curr_train_LOOCV = open (train_path+"train_left_out_"+str(out_clm_num)+"_CV_"+source, 'wb')                 
                for (curr_clm_num,curr_clm,curr_sen) in sep_features_dict.keys():
                    try:  
                        if out_clm_text == curr_clm:
                            data = sep_features_dict[(curr_clm_num,curr_clm,curr_sen)]
                            line_to_write = str(data[0])+" "+"qid:"+str(out_clm_num) +" "
                            normalized_features_data = [float(data[1][i]/sep_max_min_val_dict[(str(curr_clm_num))][i]) for i in range(0,len(data[1]))]
                            for feature_idx in range(0,len(normalized_features_data)):
                                line_to_write += str(feature_idx+1)+":"+str(normalized_features_data[feature_idx])+" "
                            line_to_write += "#"+curr_clm+"|"+curr_sen+"\n"
                            curr_test_LOOCV.write(line_to_write)
                        else:   
                            data = sep_features_dict[(curr_clm_num,curr_clm,curr_sen)]
                            normalized_features_data = [float(data[1][i]/sep_max_min_val_dict[(str(curr_clm_num))][i]) for i in range(0,len(data[1]))]
                            line_to_write = str(data[0])+" "+"qid:"+str(curr_clm_num) +" "
                            for feature_idx in range(0,len(normalized_features_data)):
                                line_to_write += str(feature_idx+1)+":"+str(normalized_features_data[feature_idx])+" "
                            line_to_write += "#"+curr_clm+"|"+curr_sen+"\n"
                            curr_train_LOOCV.write(line_to_write)
                    except Exception as err: 
                            sys.stderr.write('problem in create_input_files_SVM_Rank with:'+ curr_clm_num+ " and sen"+ curr_sen)     
                            print err.args      
                            print err 
                curr_test_LOOCV.close()
                curr_train_LOOCV.close()     
    else: #unified collection setup
        for (out_clm_num,out_clm_text) in claim_num_and_text.items():
            test_LOOCV = open (test_path+"test_clm_num_"+str(out_clm_num)+"_CV", 'wb')
            train_LOOCV = open (train_path+"train_left_out_"+str(out_clm_num)+"_CV", 'wb') 
            try:
                for (curr_clm_num,curr_clm,curr_sen) in clm_and_sen_feature_vector_sorted_by_clm_num.keys():
                    if out_clm_text == curr_clm:  
                        data = clm_and_sen_feature_vector_sorted_by_clm_num[(curr_clm_num,curr_clm,curr_sen)]
                        line_to_write = str(data[0])+" "+"qid:"+str(out_clm_num) +" "
                        for feature_idx in range(0,len(data[1])):
                            line_to_write += str(feature_idx+1)+":"+str(data[1][feature_idx])+" "
                        line_to_write += "#"+curr_clm+"|"+curr_sen+"\n"
                        test_LOOCV.write(line_to_write)
                    else:   
                        data = clm_and_sen_feature_vector_sorted_by_clm_num[(curr_clm_num,curr_clm,curr_sen)]
                        line_to_write = str(data[0])+" "+"qid:"+str(curr_clm_num) +" "
                        for feature_idx in range(0,len(data[1])):
                            line_to_write += str(feature_idx+1)+":"+str(data[1][feature_idx])+" "
                        line_to_write += "#"+curr_clm+"|"+curr_sen+"\n"
                        train_LOOCV.write(line_to_write)
    #                     train_LOOCV.write(str(data[0])+" "+"qid:"+str(curr_clm_num)+" 1:"+str(data[1][0])+" 2:"+str(data[1][1])+" 3:"+ str(data[1][2])+"#"+curr_clm+"|"+curr_sen+"\n")
                test_LOOCV.close()
                train_LOOCV.close()
            except Exception as err: 
                sys.stderr.write('problem in create_input_files_SVM_Rank with:'+ curr_clm+ " and sen"+ curr_sen)     
                print err.args      
                print err       

def  train_SVM_rank(c):
    """
    train_SVM_rank with LOOCV
    """
    print "train_SVM_rank"
    if kernel is "polynomial":
        kernel_opt_str = "-t 1 -d 2 " #the first is the kernel number, the second is the param
    elif kernel is "linear":
        kernel_opt_str = "-t 0 "
    elif kernel is "radial":
        kernel_opt_str = "-t 2 -g 3.5275 "  ## the g is the variance of the data, calc from analyze_sentence_support, in calc_variance_on_all_data func. 
    
    for filename in os.listdir(train_path):
        print "filename: "+filename
        command = 'svm_rank_learn -c '+c +' '+kernel_opt_str
        claim_num = filename.split("out_")[1].split("_CV")[0]
        if setup == "separate":
            curr_source = filename.split("out_")[1].split("_CV_")[1]
            command += train_path+filename + " "+model_path+"left_out_"+claim_num+"_model_"+curr_source
        else:
            command += train_path+filename + " "+model_path+"left_out_"+claim_num+"_model"
        utils.apply_command_line(train_path,filename,claim_num, command, r"C:\softwares\SVMRank", model_path, "_model")

def predict_with_SVM_rank():
    """
    train_SVM_rank with LOOCV
    """
    print "predict_with_SVM_rank "
    
    for filename in os.listdir(train_path):
        claim_num = filename.split("out_")[1].split("_CV")[0]
        if setup == "separate":
            curr_source = filename.split("out_")[1].split("_CV_")[1]
            command = 'svm_rank_classify '+test_path+'test_clm_num_'+claim_num+'_CV_'+curr_source+" " +model_path+'left_out_'+claim_num+'_model_'+curr_source+" " +prediction_path+claim_num+r'_prediction_'+curr_source
        else:
            command = 'svm_rank_classify '+test_path+'test_clm_num_'+claim_num+'_CV ' +model_path+'left_out_'+claim_num+'_model '+prediction_path+claim_num+r'_prediction '
        utils.apply_command_line(train_path,filename,claim_num, command, r"C:\softwares\SVMRank", prediction_path, "_prediction")
    
def read_predicted_support_score():
#     prediction_path = r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\"+curr_SVM_with_features_path+r"\\prediction\\"
#     test_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\SVM_rank\test"
    clm_sen_predicition_score_dict_sorted = {}
    claim_num_and_text=utils.read_pickle(r'C:\Users\liorab\workspace\supporting_evidence\src\features\claim_dict_pickle')
    entity_name_presence_dict = utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\clm_sen_doc_title_entity_presence_flag") #key is clm and sen, val is the doc tit of the sen and a 
                                                                                                                                                    #flag of whether the entity is the doc title or in the sen itself
    #prediction score
    if setup == "separate":
        prediction_score_dict_wiki = {}
        prediction_score_dict_RT = {}
        separated_dict_list = [ (prediction_score_dict_RT,"RT"),(prediction_score_dict_wiki,"wiki")]
        for (d,curr_source) in separated_dict_list:
            for clm_num in claim_num_and_text.keys():
                curr_pred_file = open(prediction_path+"\\"+clm_num+"_prediction_"+curr_source, 'r').read().strip()
                curr_test_file = open(test_path+"\\"+"test_clm_num_"+clm_num+"_CV_"+curr_source, 'r').read().strip()
#                 prediction_score_dict = {}
                sen_dict = {} #key is a line number from the file, val is a sen
                for i, line in enumerate(curr_pred_file.split('\n')):
#                     prediction_score_dict[i] = float(line)
                    d[i] = float(line)
                for i, line in enumerate(curr_test_file.split('\n')):
                    #need to check if # is also in the sen itself, meaning if there are m
                    if line.count("#") >1:
                        sen = line.split("#",1)[1].split("|")[1]
                    else:
                        sen = line.split("#")[1].split("|")[1]
                    sen_dict[i] = sen
                clm = line.split("#")[1].split("|")[0]
#                 sen_predicted_score_sorted = sorted(zip(sen_dict.values(), prediction_score_dict.values()),key=lambda x: (float(x[1])), reverse=True)
                sen_predicted_score_sorted = sorted(zip(sen_dict.values(), d.values()),key=lambda x: (float(x[1])), reverse=True)
                clm_sen_predicition_score_dict_sorted[clm] = (sen_predicted_score_sorted) #key is clm , value is a list of sen and the predicted score
                
            with open("sort_sen_per_clm_pred_"+curr_source+"_"+CV_method+"_"+kernel+"_"+features_str+"_"+sentiment_model+"_"+similarity_function+".csv", 'wb') as csvfile:
                    w = csv.writer(csvfile)
                    for (clm,sen_predicted_score_list) in clm_sen_predicition_score_dict_sorted.items():
                        for (sen, score) in sen_predicted_score_list:
                            w.writerow([clm,sen,str(score),entity_name_presence_dict[clm,sen][0]])
            utils.save_pickle("clm_as_key_sen_predicted_support_score_val_"+curr_source+"_"+CV_method+"_"+kernel+"_"+features_str+"_"+sentiment_model+"_"+similarity_function, clm_sen_predicition_score_dict_sorted)               

def read_true_support_score():
    clm_sen_support_ranking=utils.read_pickle(r"C:\Users\liorab\workspace\supporting_evidence\src\features\clm_sen_support_ranking_sorted_full")
    clm_as_key_sen_support_score_val={} #the same only with supp score
    clm_as_key_sen_support_score_val_zero_to_two = {}  #22.09 update - for the 0-2 support scale
    new_score = 0
     
    for (clm,sen,score) in clm_sen_support_ranking.keys():
        if score == 0 or score == 1 or score == 2 or score == 3:
            new_score = 0
        elif new_score == 4 :
            score =1
        elif score ==5:
            new_score = 2
        if clm in clm_as_key_sen_support_score_val.keys():
            clm_as_key_sen_support_score_val[clm].append((sen,score))
            clm_as_key_sen_support_score_val_zero_to_two[clm].append((sen,new_score))
        else:
            clm_as_key_sen_support_score_val[clm]=[(sen,score)]
            clm_as_key_sen_support_score_val_zero_to_two[clm] = [(sen,new_score)]
            
    utils.save_pickle("clm_as_key_sen_true_support_score_val_zero_to_five",clm_as_key_sen_support_score_val) 
    utils.save_pickle("clm_as_key_sen_true_support_score_val_"+supp_scale,clm_as_key_sen_support_score_val_zero_to_two) 
       
def process_SVM_rank_prediction(p):
    """
    the SVM rank output are scores for each claim,
    This function creates a ranking according to these scores.
    For each line in the test file, for which there is a claim and a sentence:
    """
    claim_dict = utils.read_pickle("claim_dict")
    #SVM rank support score -read each predicition file, sort the 60 sentences, and calc ndcg
    if setup == "separate":
#         clm_sen_predicition_score_dict_RT = utils.read_pickle("clm_as_key_sen_predicted_support_score_val_RT_"+CV_method+"_"+kernel+"_"+features_str+"_"+sentiment_model+"_"+similarity_function)
        clm_sen_predicition_score_dict_wiki = utils.read_pickle("clm_as_key_sen_predicted_support_score_val_wiki_"+CV_method+"_"+kernel+"_"+features_str+"_"+sentiment_model+"_"+similarity_function)
#         clm_as_key_sen_support_score_val_RT = utils.read_pickle("clm_as_key_sen_support_score_val_RT")
        clm_as_key_sen_support_score_val_wiki = utils.read_pickle("clm_as_key_sen_support_score_val_wiki")
    else:
            #the true supportiveness data
            clm_as_key_sen_support_score_val = utils.read_pickle("clm_as_key_sen_true_support_score_val_"+supp_scale)  #key is a claim, val is a sorted list of sen according to true supp score

#     separated_list = [(clm_sen_predicition_score_dict_RT,clm_as_key_sen_support_score_val_RT,"RT"),(clm_sen_predicition_score_dict_wiki,clm_as_key_sen_support_score_val_wiki,"wiki")]
    separated_list = [(clm_sen_predicition_score_dict_wiki,clm_as_key_sen_support_score_val_wiki,"wiki")]
    for (curr_pred_supp_dict,curr_true_supp_dict,curr_source) in separated_list:
        NDCG_all_claims= {} #key is a claim, value is the nDCG
        AP_all_claims= {} 
        prec_at_5_all_claims = {}
        prec_at_10_all_claims = {}
        
        for clm in curr_true_supp_dict.keys():
            try:
                if claim_dict[str(clm)]  in curr_pred_supp_dict.keys():
                    NDCG_all_claims[clm] = utils.calc_emp_NDCG(curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm],p)
                    AP_all_claims[clm] = utils.calc_AP_support(curr_source,clm,curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm],p)
                    prec_at_5_all_claims[clm] = utils.calc_precision_at_k(5, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                    prec_at_10_all_claims[clm] = utils.calc_precision_at_k(10, curr_pred_supp_dict[claim_dict[str(clm)]],curr_true_supp_dict[clm])
                
            except Exception as err: 
                sys.stderr.write('problem in calc measures: in source: '+ curr_source+' in clm '+ claim_dict[str(clm)])     
                print err.args      
                print err
                
        average_NDCG = float(float(sum(NDCG_all_claims.values()))/float(len(NDCG_all_claims)))
        MAP = float(float(sum(AP_all_claims.values()))/float(len(AP_all_claims)))
        average_prec_at_5 = float(float(sum(prec_at_5_all_claims.values()))/float(len(prec_at_5_all_claims)))
        average_prec_at_10 = float(float(sum(prec_at_10_all_claims.values()))/float(len(prec_at_10_all_claims)))
        std_NDCG = np.std(NDCG_all_claims.values())
        std_MAP = np.std(AP_all_claims.values())
        std_prec_at_5 = np.std(prec_at_5_all_claims.values())
        std_prec_at_10 = np.std(prec_at_10_all_claims  .values())
        
        print curr_source+ ": in "+ curr_features_path+" average_NDCG: " +str(average_NDCG) +" std: "+str(std_NDCG) +" MAP :" +str(MAP) + " std:"+str(std_MAP)+ " average_prec_at_5:"+ str(average_prec_at_5) +" std:"+ str(std_prec_at_5)+" average_prec_at_10:"+str(average_prec_at_10) + " std:"+ str(std_prec_at_10)
        all_claims_sorted_by_NDCG_feature = collections.OrderedDict(sorted(NDCG_all_claims.items(),key=lambda x: (float(x[1])), reverse=True))
         
        with open('SVM_rank_nDCG@'+str(p)+"_"+curr_source+"_"+CV_method+"_"+kernel+"_"+features_str+"_"+sentiment_model+"_"+similarity_function+".csv", 'wb') as csvfile:
            w = csv.writer(csvfile)
            for (clm,ndcg_score) in all_claims_sorted_by_NDCG_feature.items():
                w.writerow([clm,ndcg_score])
            w.writerow(['average NDCG:'+str(average_NDCG)])
        
        all_claims_sorted_by_prec_at_5_feature = collections.OrderedDict(sorted(prec_at_5_all_claims.items(),key=lambda x: (float(x[1])), reverse=True))
         
        with open('SVM_rank_nDCG@'+str(p)+"_"+curr_source+"_"+CV_method+"_"+kernel+"_"+features_str+"_"+sentiment_model+"_"+similarity_function+".csv", 'wb') as csvfile:
            w = csv.writer(csvfile)
            for (clm,prec_at_5) in all_claims_sorted_by_prec_at_5_feature.items():
                w.writerow([clm,prec_at_5])
            w.writerow(['average NDCG:'+str(average_prec_at_5)])
            
def calc_random_rank_measure_with_respect_to_features_ranking(p):
    """
    Check the nDCG/MRR on a randomized list with respect to the ranking received by the features and the ranker,
    """
    clm_sen_predicition_score_dict = utils.read_pickle("clm_as_key_sen_predicted_support_score_val_"+CV_method+"_"+kernel+"_"+features_str+"_"+sentiment_model+"_"+similarity_function)
    clm_sen_predicition_score_dict_copy = copy.deepcopy(clm_sen_predicition_score_dict)
    NDCG_all_claims_randomized_rank = {}
    
    num_iter = 100
    nDCG_per_iter =[]
    for iter in range(0,num_iter):
        for clm in clm_sen_predicition_score_dict.keys():
            random.shuffle(clm_sen_predicition_score_dict_copy[clm])
            NDCG_all_claims_randomized_rank[clm]=utils.calc_emp_NDCG(clm_sen_predicition_score_dict_copy[clm],clm_sen_predicition_score_dict[clm],p)
        average_NDCG_random=float(float(sum(NDCG_all_claims_randomized_rank.values()))/float(len(NDCG_all_claims_randomized_rank)))
        nDCG_per_iter.append(average_NDCG_random)
    
    average_random_nDCG_across_iter = float(float(sum(nDCG_per_iter))/float(num_iter))
    std_nDCG_across_iter = 0
    for avg_NDCG in nDCG_per_iter:
        std_nDCG_across_iter += (avg_NDCG-average_random_nDCG_across_iter)**2
    std_nDCG_across_iter = float(std_nDCG_across_iter/num_iter)      
    
    print ("average random nDCG@"+str(p)+" across "+str(num_iter)+" iterations with features: "+curr_features_path +" : "
           +str(average_random_nDCG_across_iter)+", std: "+str(std_nDCG_across_iter))      

def create_separate_true_suppport_score_dicts_key_clm_val_list_of_sen():
    """
    Separate the true support dict to wiki and RT, to create a dict where key is a claim, and value is a sorted list of sen -from most supp to least. 0-2 scale/0-5
    """
    if setup == "separate":
#         clm_and_sen_feature_vector_RT = utils.read_pickle("clm_and_sen_feature_vector_"+features_str+"_RT")
        clm_and_sen_feature_vector_wiki = utils.read_pickle("clm_and_sen_feature_vector_"+features_str+"_wiki")#key is clm_num,clm,sen
        clm_sen_support_ranking = utils.read_pickle("clm_sen_support_ranking_"+supp_scale+"_clm_sen_key_supp_score_value")
        claim_dict = utils.read_pickle("claim_dict")
#         clm_sen_support_ranking_RT = {}
        clm_sen_support_ranking_wiki = {}
        clm_as_key_sen_support_score_val_wiki = {} #key is a clm, va;ue is a list of sentences and their supp score
#         clm_as_key_sen_support_score_val_RT = {}
        #divide the orig true support score to wiki and RT
        try:
            for (clm_num, clm,sen) in clm_and_sen_feature_vector_wiki.keys():
                clm_sen_support_ranking_wiki[(clm_num,sen)] = clm_sen_support_ranking[(claim_dict[str(clm_num)],sen)]
#             for (clm,sen) in clm_sen_support_ranking.keys():
#                 clm_sen_support_ranking_wiki[(clm_num,sen)] = clm_sen_support_ranking[(clm_num,sen)]
 
        except Exception as err: 
                sys.stderr.write('problem in create_separate_true_suppport_score_dicts_key_clm_val_list_of_sen: in clm '+ clm +" in sen"+ sen)     
                print err  
#         try:
#             for (clm_num, clm,sen) in clm_and_sen_feature_vector_RT.keys():
#                 clm_sen_support_ranking_RT[(clm,sen)] = clm_sen_support_ranking[(clm,sen)] 
#         except Exception as err: 
#                 sys.stderr.write('problem in create_separate_true_suppport_score_dicts_key_clm_val_list_of_sen: in clm '+ clm +" in sen"+ sen)     
#                 print err  
                
        clm_sen_support_ranking_wiki_sorted = collections.OrderedDict(sorted(clm_sen_support_ranking_wiki.items(),key=lambda x: (str(x[0][0]),int(x[1])), reverse=True))
#         clm_sen_support_ranking_RT_sorted = collections.OrderedDict(sorted(clm_sen_support_ranking_RT.items(),key=lambda x: (str(x[0][0]),int(x[1])), reverse=True))
        
#         utils.save_pickle("clm_sen_support_ranking_wiki_sorted",clm_sen_support_ranking_wiki_sorted)
#         utils.save_pickle("clm_sen_support_ranking_RT_sorted",clm_sen_support_ranking_RT_sorted)
        
        for ((clm,sen),score) in clm_sen_support_ranking_wiki_sorted.items():
            if clm in clm_as_key_sen_support_score_val_wiki.keys():
                clm_as_key_sen_support_score_val_wiki[clm].append((sen,score))
            else:
                clm_as_key_sen_support_score_val_wiki[clm]=[(sen,score)]
        
#         for ((clm,sen),score) in clm_sen_support_ranking_RT_sorted.items():
#             if clm in clm_as_key_sen_support_score_val_RT.keys():
#                  clm_as_key_sen_support_score_val_RT[clm].append((sen,score))
#             else:
#                 clm_as_key_sen_support_score_val_RT[clm]=[(sen,score)]
          
#         utils.save_pickle("clm_as_key_sen_support_score_val_RT", clm_as_key_sen_support_score_val_RT)
        utils.save_pickle("clm_as_key_sen_support_score_val_wiki",clm_as_key_sen_support_score_val_wiki)
        
#         with open("clm_as_key_sorted_sen_support_score_val_RT.csv", 'wb') as csvfile:
#             w = csv.writer(csvfile)
#             for (clm,list_of_sen) in clm_as_key_sen_support_score_val_RT.items():
#                 w.writerow([clm])
#                 for sen in list_of_sen:
#                     w.writerow([sen])
        
        with open("clm_as_key_sorted_sen_support_score_val_wiki.csv", 'wb') as csvfile:
            w = csv.writer(csvfile)
            for (clm,list_of_sen) in clm_as_key_sen_support_score_val_wiki.items():
                w.writerow([clm])
                for sen in list_of_sen:
                    w.writerow([sen])
             
        
        
def calc_random_ranking_with_respect_to_true_suppport_scores(p):

        if setup == "separate":
            clm_as_key_sen_support_score_val_RT = utils.read_pickle("clm_as_key_sen_support_score_val_RT")
            clm_as_key_sen_support_score_val_wiki = utils.read_pickle("clm_as_key_sen_support_score_val_wiki")
            
            clm_sen_support_ranking_RT_copy = copy.deepcopy(clm_as_key_sen_support_score_val_RT)
            clm_sen_support_ranking_wiki_copy = copy.deepcopy(clm_as_key_sen_support_score_val_wiki)
        
            NDCG_all_claims_randomized_rank_RT = {}
            NDCG_all_claims_randomized_rank_wiki = {}
    
            num_iter = 100
            nDCG_per_iter_wiki = []
            nDCG_per_iter_RT = []
            for iter in range(0,num_iter):
                for (clm) in clm_as_key_sen_support_score_val_wiki.keys():
                    random.shuffle(clm_sen_support_ranking_wiki_copy[clm])
                    NDCG_all_claims_randomized_rank_wiki[clm] = utils.calc_emp_NDCG("wiki",clm,clm_sen_support_ranking_wiki_copy[clm],clm_as_key_sen_support_score_val_wiki[clm],p)
                average_NDCG_random_wiki = float(float(sum(NDCG_all_claims_randomized_rank_wiki.values()))/float(len(NDCG_all_claims_randomized_rank_wiki)))
                nDCG_per_iter_wiki.append(average_NDCG_random_wiki)
                
                for (clm) in clm_as_key_sen_support_score_val_RT.keys():
                    random.shuffle(clm_sen_support_ranking_RT_copy[clm])
                    NDCG_all_claims_randomized_rank_RT[clm] = utils.calc_emp_NDCG("RT",clm,clm_sen_support_ranking_RT_copy[clm],clm_as_key_sen_support_score_val_RT[clm],p)
                average_NDCG_random_RT = float(float(sum(NDCG_all_claims_randomized_rank_RT.values()))/float(len(NDCG_all_claims_randomized_rank_RT)))
                nDCG_per_iter_RT.append(average_NDCG_random_RT)
            
            average_random_nDCG_across_iter_wiki = float(float(sum(nDCG_per_iter_wiki))/float(num_iter))
            average_random_nDCG_across_iter_RT = float(float(sum(nDCG_per_iter_RT))/float(num_iter))
            std_nDCG_across_iter_wiki = 0
            std_nDCG_across_iter_RT = 0
            
            for avg_NDCG in nDCG_per_iter_wiki:
                std_nDCG_across_iter_wiki += (avg_NDCG-average_random_nDCG_across_iter_wiki)**2
            std_nDCG_across_iter_wiki = float(std_nDCG_across_iter_wiki/num_iter)
            
            for avg_NDCG in nDCG_per_iter_RT:
                std_nDCG_across_iter_RT += (avg_NDCG-average_random_nDCG_across_iter_RT)**2
            std_nDCG_across_iter_RT = float(std_nDCG_across_iter_RT/num_iter)     
    
            print ("wiki average random nDCG@"+str(p)+" across "+str(num_iter)+" iterations: "
                   +str(average_random_nDCG_across_iter_wiki)+", std: "+str(std_nDCG_across_iter_wiki))  
            
            print ("RT average random nDCG@"+str(p)+" across "+str(num_iter)+" iterations: "
                   +str(average_random_nDCG_across_iter_RT)+", std: "+str(std_nDCG_across_iter_RT))   
                
#             print "wiki: random_nDCG@p:"+str(p)+": "+str( average_NDCG_random_wiki)
#             print "RT: random_nDCG@p:"+str(p)+": "+str( average_NDCG_random_RT)
        
def main():
    p = 10
#     calc_random_ranking_with_respect_to_true_suppport_scores(p)
#     convert_support_dict()
#     read_true_support_score()
#     create_separate_true_suppport_score_dicts_key_clm_val_list_of_sen()

#     create_features_dict_for_input_files()
#     features_max_min_normalization()
#     create_input_files_SVM_Rank()
# # # 
#     train_SVM_rank(c="0.1")
#     predict_with_SVM_rank()
#     read_predicted_support_score()
    process_SVM_rank_prediction(p)


#     calc_random_rank_measure_with_respect_to_features_ranking(p)

if __name__ == '__main__':
    main() 

