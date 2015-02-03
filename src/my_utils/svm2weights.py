# Compute the weight vector of linear SVM based on the model file
# Original Perl Author: Thorsten Joachims (thorsten@joachims.org)
# Python Version: Ori Cohen (orioric@gmail.com)
# Call: python svm2weights.py svm_model

import sys
from operator import itemgetter
import  os


try:
    import psyco
    psyco.full()
except ImportError:
    print 'Psyco not installed, the program will just run slower'

def sortbyvalue(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.iteritems(), key=itemgetter(1), reverse=True)

def sortbykey(d,reverse=True):
    ''' proposed in PEP 265, using  the itemgetter this function sorts a dictionary'''
    return sorted(d.iteritems(), key=itemgetter(0), reverse=False)

def get_file():
    """
    Tries to extract a filename from the command line.  If none is present, it
    assumes file to be svm_model (default svmLight output).  If the file 
    exists, it returns it, otherwise it prints an error message and ends
    execution. 
    """
    # Get the name of the data file and load it into 
    if len(sys.argv) < 2:
        # assume file to be svm_model (default svmLight output)
        print "Assuming file as svm_model"
        filename = 'svm_model' 
        #filename = sys.stdin.readline().strip()
    else:
        filename = sys.argv[1]

    
    try:
        f = open(filename, "r")
    except IOError:
        print "Error: The file '%s' was not found on this system." % filename
        sys.exit(0)

    return f




if __name__ == "__main__":
#     f = get_file()
    CV_method ="LOOCV"
    learner ="SVM_rank"
    similarity_function = "JSD"
    sentiment_model = "orig"
    obj_LM = "dist"
    semantic_sim = "additive"
    semantic_sim_entity_removal= "no_entity_remove" 
    if semantic_sim_entity_removal == "entity_remove" :
        features_list = ["sen_sim","sem_sim_"+semantic_sim+"_"+semantic_sim_entity_removal,"entity_presence"]
    else:
        features_list = ["sen_sim","sem_sim_"+semantic_sim,"objective_LM_"+obj_LM,"entity_presence"]

#     features_list = ["sen_sim","sem_sim","objective_LM_"+obj_LM,"entity_presence"]
#     features_list = ["sen_sim","sem_sim_"+semantic_sim,"entity_presence"]
    features_str = "_".join(features_list)
    kernel= "linear"
    supp_scale = "zero_to_two" #22.09 update  -  scores are 0 - not support , 1 is support, 2 is strong support
    setup ="separate" #unified collection, or separate -  RT or wiki
#     curr_features_path = learner+"_"+features_str+"_"+sentiment_model+"_"+similarity_function+"_"+kernel
    curr_features_path = learner+"_"+features_str+"_"+kernel
    model_path = r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\"+supp_scale+r"_scale_SVM_res\\"+curr_features_path+r"\\model\\"
    setup = "separate"
    
    w_RT = {}
    w_wiki = {}
    
    features_dict_numbers = {1:'sentiment',2:'semantic',3:'1_star',4:'2_star',5:'3_star',6:'4_star',7:'5_star',8:'entity_presence'}
    
    if setup is "separate":
        sources = ["RT","wiki"]
    models_cnt = 0 
    for filename in os.listdir(model_path):
        curr_source = filename.split("out_")[1].split("_model_")[1]
        f =  open(model_path+filename, 'r')
        models_cnt += 1
        i = 0
        lines = f.readlines()
        printOutput = True
       
        for line in lines:
            if i>10:
                features = line[:line.find('#')-1]
                comments = line[line.find('#'):]
                alpha = features[:features.find(' ')]
                feat = features[features.find(' ')+1:]
                for p in feat.split(' '): # Changed the code here. 
                    a,v = p.split(':')
                    if curr_source == "RT":
                        if not (int(a) in w_RT):
                            w_RT[int(a)] = 0
                    elif curr_source == "wiki":
                        if not (int(a) in w_wiki):
                            w_wiki[int(a)] = 0
                for p in feat.split(' '): 
                    a,v = p.split(':')
                    if curr_source == "RT":
                        w_RT[int(a)] +=float(alpha)*float(v)
                    elif curr_source == "wiki":
                        w_wiki[int(a)] +=float(alpha)*float(v) 
            elif i==1:
                if line.find('0')==-1:
                    print 'Not linear Kernel!\n'
                    printOutput = False
                    break
            elif i==10:
                if line.find('threshold b')==-1:
                    print "Parsing error!\n"
                    printOutput = False
                    break
            i+=1    
        f.close()

    #if you need to sort the features by value and not by feature ID then use this line intead:
#         ws = sortbyvalue(w) 
    ws_RT = sortbyvalue(w_RT)
    ws_wiki = sortbyvalue(w_wiki)
    if printOutput == True:
        log_RT = "RT:"
        log_wiki = "wiki:"
        for (i,j) in ws_RT:
            log_RT += " "+ str(features_dict_numbers[i])+':'+str(float(j/models_cnt/2))
            i+=1
        for (i,j) in ws_wiki:
            log_wiki +=" "+  str(features_dict_numbers[i])+':'+str(float(j/models_cnt))
            i+=1
        print log_RT
        print log_wiki
