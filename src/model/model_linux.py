'''
Created on Sep 9, 2014
A one stage model for ranking supporting evidence per claim

Using LTR (Learning to rank) such as SVM-rank,
Features: Sentiment Similarity (a JSD score)
Semantic similarity (a cosine score)
Objective sentences LM - an affiliation for any of the 5 options (so this feature can take value 1,2,...5
entailment- to add. an score of entailment (positive means entailment, negative means no entailment)

@author: Liora Braunstein
'''

import sys
import os
import shlex
import subprocess

CV_method ="LOOCV"
learner ="SVM_rank"
similarity_function = "JSD"
sentiment_model = "orig"
obj_LM = "label"
features_list = ["sen_sim","sem_sim","objective_LM_"+obj_LM,"entity_presence"]
features_str ="sen_sim_sem_sim_objective_LM_"+obj_LM+"_entity_presence" 
kernel= "radial"

my_path = "/lv_local/home/liorab/support/"
curr_features_path = learner+"_"+features_str+"_"+sentiment_model+"_"+similarity_function+"_"+kernel
# curr_features_path = r'SVM_rank_sen_sim_orig_JSD_sem_sim_objective_LM_entity_presence_linear_kernel'
train_path = my_path + curr_features_path+r"/train/"
test_path = my_path + curr_features_path+r"/test/"
model_path = my_path + curr_features_path+r"/model/"
prediction_path = my_path + curr_features_path+r"/prediction/"


def train_SVM_rank():
    """
    train_SVM_rank with LOOCV
    """
    print "####enter train_SVM_rank"
    if kernel is "polynomial":
        kernel_opt_str = "-t 1 -d 2 " #the first is the kernel number, the second is the param
    elif kernel is "linear":
        kernel_opt_str = "-t 0 "
    elif kernel is "radial":
        kernel_opt_str = "-t 2 -g 3.5275 "## the g is the variance of the data, calc from analyze_sentence_support, in calc_variance_on_all_data func. 
    for filename in os.listdir(train_path):
        print "filename: "+filename
        command = './svm_rank_learn -c 0.01 '+kernel_opt_str
        claim_num = filename.split("out_")[1].split("_CV")[0]
        command += train_path+filename + " "+model_path+"left_out_"+claim_num+"_model"
        print "command is:"+ command 
        os.system(command)
#         apply_command_line(train_path,filename,claim_num, command, "/home/liorab/softwares/svm_rank", model_path, "_model")

def  apply_command_line(input_file_path,input_file_name,claim_num,command_input,cwd_input,output_files_path,file_suffix):
    print "####enter apply_command_line"
    command = command_input #+" %s"  % input_file_path+"\\"+input_file_name
    proc = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,shell=True,stderr=subprocess.PIPE)
#      cwd=cwd_input)
#     proc = subprocess.Popen(command.split(),stdout=subprocess.PIPE,shell=True,stderr=subprocess.PIPE)
#                        
    print "before communicate"
    stdout, stderr = proc.communicate()
    print "after communicate"
    print "err:" +stderr
    print "stdout: " +stdout
    retcode = proc.wait()
    if retcode < 0:
        print >>sys.stderr, "error in command line", retcode
    else:
        print  "finished "+ str(claim_num)


def predict_with_SVM_rank():
    """
    train_SVM_rank with LOOCV
    """
    print "predict_with_SVM_rank "
    
    for filename in os.listdir(train_path):
        claim_num = filename.split("out_")[1].split("_CV")[0]
        command = './svm_rank_classify '+test_path+'test_clm_num_'+claim_num+'_CV ' +model_path+'left_out_'+claim_num+'_model ' +prediction_path+claim_num+'_prediction'
#                 command += train_path+filename + " "+model_path+"left_out_"+claim_num+"_model"
        apply_command_line(train_path,filename,claim_num, command, r"/home/liorab/softwares/svm_rank/", prediction_path, "_prediction")
    

    
# def main():
# #     convert_support_dict()
#     create_input_files_SVM_Rank()
#     train_SVM_rank()
#     predict_with_SVM_rank()
# #     read_true_support_score()
#     read_predicted_support_score()
#     process_SVM_rank_prediction(10)


# if __name__ == '__main__':
#     main() 

