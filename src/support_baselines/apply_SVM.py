import os
import subprocess
import shlex
import sys


kernel= "linear"
support_linux_base_path = r"/lv_local/home/liorab/softwares/indri-5.5/retrieval_baselines_support"
SVM_path = support_linux_base_path+"/SVM_zero_two_scale"
train_path = SVM_path+"/train/"
test_path = SVM_path +r"/test/"
model_path = SVM_path +r"/model/"
prediction_path = SVM_path +r"/prediction/"

def train_SVM_rank():
    """
    train_SVM_rank with LOOCV
    """
    claim_list = [4,7,17,21,36,37,39]#,40,41,42,45,46,47,50,51,53,54,55,57,58,59,60,61,62,66,69,70,79,80]
    print "####enter train_SVM_rank"
    if kernel is "polynomial":
        kernel_opt_str = "-t 1 -d 2 " #the first is the kernel number, the second is the param
    elif kernel is "linear":
        kernel_opt_str = "-t 0 "
    elif kernel is "radial":
        kernel_opt_str = "-t 2 -g 3.5275 "## the g is the variance of the data, calc from analyze_sentence_support, in calc_variance_on_all_data func. 
    for filename in os.listdir(train_path):
        claim_num = filename.split("out_")[1].split("_CV")[0]
        if int(claim_num) in claim_list:
            print "filename: "+filename
            command = './svm_rank_learn -c 0.01 '+kernel_opt_str
            command += train_path+filename + " "+model_path+"left_out_"+claim_num+"_model"
            print "command is:"+ command 
            os.system(command)

def predict_with_SVM_rank():
    """
    train_SVM_rank with LOOCV
    """
    print "predict_with_SVM_rank "
    
    for filename in os.listdir(train_path):
        claim_num = filename.split("out_")[1].split("_CV")[0]
        command = './svm_rank_classify '+test_path+'test_clm_num_'+claim_num+'_CV ' +model_path+'left_out_'+claim_num+'_model ' +prediction_path+claim_num+'_prediction'
        os.system(command)
#                 command += train_path+filename + " "+model_path+"left_out_"+claim_num+"_model"
        print "command: " +command
        
#         apply_command_line(train_path,filename,claim_num, command, r"/home/liorab/softwares/svm_rank/", prediction_path, "_prediction")
 
def apply_command_line(input_file_path,input_file_name,claim_num,command_input,cwd_input,output_files_path,file_suffix):
    print "####enter apply_command_line"
    command = command_input #+" %s"  % input_file_path+"\\"+input_file_name
    proc = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,shell = True,stderr=subprocess.PIPE)
#   cwd=cwd_input)
#   proc = subprocess.Popen(command.split(),stdout=subprocess.PIPE,shell=True,stderr=subprocess.PIPE)
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
   
