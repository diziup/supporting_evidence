'''
26/07/14
@author: Liora
'''
import pickle
import math
from itertools import izip
import os
import shlex
import subprocess
import sys  
import string

def dot_product(v1, v2):
        return sum(map(lambda x: float(x[0]) * float(x[1]), izip(v1, v2)))
    
def cosine_measure(v1, v2):
        prod = dot_product(v1, v2)
        len1 = math.sqrt(dot_product(v1, v1))
        len2 = math.sqrt(dot_product(v2, v2))
        return prod / (len1 * len2) 
    
def computeKappa(mat,file_name):
        
        """
  Computes the Fleiss' Kappa value as described in (Fleiss, 1971)
  Copied from http://en.wikibooks.org/wiki/Algorithm_Implementation/Statistics/Fleiss'_kappa#Python
  More info about Fleiss' Kappa: http://en.wikipedia.org/wiki/Fleiss%27_kappa

  Computes the Kappa value
      @param n Number of rating per subjects (number of human raters)
      @param mat Matrix[subjects][categories]
      @return The Kappa value
  """
    
        DEBUG = False
        n = checkEachLineCount(mat,file_name)   # PRE : every line count must be equal to n
        N = len(mat)
        k = len(mat[0])
        
        if DEBUG:
            print n, "raters."
            print N, "subjects."
            print k, "categories."
        
        # Computing p[]
        p = [0.0] * k
        for j in xrange(k):
            p[j] = 0.0
            for i in xrange(N):
                p[j] += mat[i][j]
            p[j] /= N*n
        if DEBUG: print "p =", p
        
        # Computing P[]
        P = [0.0] * N
        for i in xrange(N):
            P[i] = 0.0
            for j in xrange(k):
                P[i] += mat[i][j] * mat[i][j]
            P[i] = (P[i] - n) / (n * (n - 1))
        if DEBUG: print "P =", P
        
        # Computing Pbar
        Pbar = sum(P) / N
        if DEBUG: print "Pbar =", Pbar
        
        # Computing PbarE
        PbarE = 0.0
        for pj in p:
            PbarE += pj * pj
        if DEBUG: print "PbarE =", PbarE
        
        kappa = (Pbar - PbarE) / (1 - PbarE)
        if DEBUG: print "kappa =", kappa
        
        return kappa

def checkEachLineCount(mat,file_name):
        """
        Assert that each line has a constant number of ratings
          @param mat The matrix checked
          @return The number of ratings
          @throws AssertionError If lines contain different number of ratings
        """
        n = sum(mat[0])
#         print "file:" , file_name
        for line in mat[1:]:
            if sum(line) !=n :
                print "Line count !=%d (n value) in line number %d " , n, line
#         assert all(sum(line) == n for line in mat[1:]), "Line= count  != %d (n value)."  % n
        return n
      
def compute_average_variance_rating(data_mat):
    
    average_rating=0
    var_rating=0
    num_of_sentences=data_mat.shape[0]
    for row in data_mat: #now the matrix is rating*sentences - for each sentence calculate the average rating (divide by 
        row_avg_rating=0                        #number of annotators,sum across all sentences, and in the end divide by total number of sentences.
        for j in range(0,len(row)):
            row_avg_rating += row[j]*(j+1)
        average_rating += float(row_avg_rating/5) 
    
    average_rating=float(average_rating/num_of_sentences)         
    #compute variance
    for row in data_mat:
        row_var_rating=0
        for j in range(0,len(row)):
            row_var_rating += row[j]*(((j+1)-average_rating)**2)
    row_var_rating=float(row_var_rating/num_of_sentences)   
    res=[average_rating,row_var_rating]   
    return res

def save_pickle(file_name,d):
        with open(file_name, 'wb') as handle:
            pickle.dump(d, handle)
        handle.close()    
        
def read_pickle(file_name):
    d={}
    with open(file_name, 'rb') as handle:
            d = pickle.loads(handle.read()) 
    return d

def read_pickle_set(file_name):
    s=set()
    with open(file_name, 'rb') as handle:
            s = pickle.loads(handle.read()) 
    return s


def apply_command_line(input_file_path,input_file_name,claim_num,command_input,cwd_input,output_files_path,file_suffix):
    """
    to apply programatically a command line - for sentiment analysis, for stanford parser,etc
    """
#     input_files_path=r"C:\\study\\technion\\MSc\\Thesis\\Y!\\support_test\\stanford_contituent_parser\\input\\"
#     output_files_path=r"C:\study\technion\MSc\Thesis\Y!\support_test\stanford_contituent_parser\output"
    
    try:
#         for filename in os.listdir(input_files_path):
#             print "command line in filename:"+ filename             
#             input_f=input_files_path+filename
#             command='java -mx1g -cp \"*\" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -outputFormat oneline edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz %s' % input_f 
            command = command_input #+" %s"  % input_file_path+"\\"+input_file_name
            proc = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,shell=True,stderr=subprocess.PIPE,
                                cwd=cwd_input)
            stdout, stderr = proc.communicate()
            retcode = proc.wait()
            if retcode < 0 or stderr != "":
                print >>sys.stderr, "error in command line", retcode
            else:
                print  "finished "+ str(claim_num)
#             filename_to_save=filename.split(".")[0]
#             curr_stdout_file = open(output_files_path+"\\"+claim_num+file_suffix, "w")
#             curr_stdout_file.write(stdout)
#             curr_stdout_file.close()
            
            
#         for filename in os.listdir(input_files_path):
#             print "command line in filename:"+ filename             
#             input_f=input_files_path+filename
#             command='java -mx1g -cp \"*\" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -outputFormat oneline edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz %s' % input_f 
#             proc = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE,shell=True,stderr=subprocess.PIPE,
#                                   cwd=r'C:\\softwares\stanford-parser-full-2014-06-16\stanford-parser-full-2014-06-16')
#             stdout, stderr = proc.communicate()
#             retcode = proc.wait()
#             if retcode < 0:
#                 print >>sys.stderr, "error in command line", retcode
#             filename_to_save=filename.split(".")[0]
#             curr_stdout_file = open(output_files_path+"\\"+filename_to_save+"_parser_res.txt", "w")
#             curr_stdout_file.write(stdout)
#             curr_stdout_file.close()

    except Exception as err: 
        sys.stderr.write('problem in apply_command_line:')     
        print err.args      
        print err
    
def calc_NDCG(nominator_list,denominator_list,p):
    """
    calc the DCG for a ranking.
    l=list of clm,sen, and the rating- as support, sentiment similarity etc.
    the clm in all the pairs is the same - this is calculated per claim.
    We compare the order between the two lists, when the rel=supp, is the only score that is taken into account
    1. Create a dict of key as sentence and value as the support score - the denominator_list 
    2. Calc DCG and IDCG
    """
    try:
        #create dict of key as sentence and value as the support score 
        sen_supp_score={}
        for (sen,supp_score) in denominator_list:
            sen_supp_score[sen]=supp_score
            
        DCG=0
        IDCG=0
        for i in range(0,p):
            nominator_score = sen_supp_score[nominator_list[i][0]]
            DCG += float(nominator_score)/float(math.log(i+1+1,2))
    #         DCG+=float((2**(nominator_score))-1)/float(math.log(i+1+1,2))
        for j in range(0,p):
            denominator_score=denominator_list[j][1]
            IDCG+=float(denominator_score)/float(math.log(j+1+1,2))
    #         IDCG+=float((2**(denominator_score))-1)/float(math.log(j+1+1,2))
        return float(DCG/IDCG) 
    except Exception as err: 
            sys.stderr.write('problem in calc_NDCG:')     
            print err.args      
            print err

def calc_doc_emp_NDCG(source,clm,nominator_list,denominator_list,p):
    """
    for the document retrieval avaluation
    """
    try:
        #create dict of key as sentence and value as the support score 
        doc_rel_score={}
        for (doc,rel_score) in denominator_list:
            doc_rel_score[doc] = rel_score  

        DCG=0
        IDCG=0

        for i in range(0,p):
            if nominator_list[i] in doc_rel_score.keys():
                nominator_score = doc_rel_score[doc]
            else:
                nominator_score = 0
            DCG += float(2**(nominator_score)-1)/float(math.log(i+1+1,2))
                
    #         DCG+=float((2**(nominator_score))-1)/float(math.log(i+1+1,2))
        for j in range(0,p):
            denominator_score=denominator_list[j][1]
            IDCG+=float(2**(denominator_score)-1)/float(math.log(j+1+1,2))
    #         IDCG+=float((2**(denominator_score))-1)/float(math.log(j+1+1,2))
        if IDCG == 0:
            print "IDCG zero in source " +source +' in claim ' +clm
            return 0
        return float(DCG/IDCG) 
    except Exception as err: 
            sys.stderr.write('problem in doc_calc_emp_NDCG: from source '+ source+ 'in clm' + clm)     
            print err.args      
            print err  
            
def calc_doc_AP_relevance(AP_cut_off,source,clm,predicted_list,true_data_list):
    """
    correct version:
    p=1000
    1. find Rj = number of relevant docs/sen per claim j.
    2. find the ranks (=i) in which there are relevant items.
    3. calc precision at those i (until i=1000)
    4. divide by Rj
    """
    num_of_relevant = 0
    doc_true_rel_score = {}
    #find Rj
    
    for (doc,true_rel_score) in true_data_list:
        if true_rel_score == 1:
            num_of_relevant += 1 
    
    for (doc,rel_score) in true_data_list:
        doc_true_rel_score[doc]=rel_score
    #find the i's in which there are rel items -> i =[1,...1000]
    i_relevant = []
    index_cnt = 0
    
    for doc in predicted_list:
        index_cnt += 1
        if index_cnt <= AP_cut_off:
            if doc in doc_true_rel_score.keys():
                if doc_true_rel_score[doc] == 1:
                    i_relevant.append(index_cnt)        
#     cut_off = min(len(predicted_list),1000)
    cut_off = 1000
    
    average_precision = 0
    for rel_index in i_relevant:
        if rel_index <=cut_off:
            precision = calc_doc_precision_at_k(rel_index,predicted_list,true_data_list) 
            average_precision =  average_precision + precision
            
    return float(float(average_precision)/float(num_of_relevant))        
    
def calc_sen_emp_NDCG(source,clm,nominator_list,denominator_list,p):
    """
   formula that emphasises the ret of relevant (supp) items
    """
    try:
        #create dict of key as sentence and value as the support score 
        sen_supp_score={}
        exclude = set(string.punctuation)
        for (sen,supp_score) in denominator_list:
#             if "Few of the original movie's political and philosophical " in sen:
#                 print 'here it is'
            sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
            sen_no_space = sen_no_punc.replace(" ","")
            sen_supp_score[sen_no_space]=supp_score  

        DCG=0
        IDCG=0

        for i in range(0,p):
            sen = ''.join(ch for ch in nominator_list[i][0] if ch not in exclude)
            sen_no_space = sen.replace(" ","")
            if sen_no_space in sen_supp_score.keys():
                nominator_score = sen_supp_score[sen_no_space]
            else:
                nominator_score = 0
            DCG += float(2**(nominator_score)-1)/float(math.log(i+1+1,2))
                
    #         DCG+=float((2**(nominator_score))-1)/float(math.log(i+1+1,2))
        for j in range(0,p):
            denominator_score=denominator_list[j][1]
            IDCG+=float(2**(denominator_score)-1)/float(math.log(j+1+1,2))
    #         IDCG+=float((2**(denominator_score))-1)/float(math.log(j+1+1,2))
        if IDCG == 0:
            print "IDCG zero in source " +source +' in claim ' +clm
            return 0
        return float(DCG/IDCG) 
    except Exception as err: 
            sys.stderr.write('problem in calc_emp_NDCG: from source '+ source+ 'in clm' + clm)     
            print err.args      
            print err      
               
def calc_sen_AP_relevance(AP_cut_off,source,clm,predicted_list,true_data_list):
    """
    correct version:
    p=1000
    1. find Rj = number of relevant docs/sen per claim j.
    2. find the ranks (=i) in which there are relevant items.
    3. calc precision at those i (until i=1000)
    4. divide by Rj
    """
    exclude = set(string.punctuation)
    sen_true_rel_score={}
    num_of_relevant = 0
    #find Rj
    
    for (sen,true_rel_score) in true_data_list:
        if true_rel_score == 1:
            num_of_relevant += 1 
    
    for (sen,rel_score) in true_data_list:
        sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
        sen_no_space = sen_no_punc.replace(" ","")
        sen_true_rel_score[sen_no_space] = rel_score
    #find the i's in which there are rel items -> i =[1,...1000]
    i_relevant = []
    index_cnt = 0
    
    for (sen,rel_score) in predicted_list:
        index_cnt += 1
        if index_cnt <= AP_cut_off:
            sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
            sen_no_space = sen_no_punc.replace(" ","")
            if sen_no_space in sen_true_rel_score.keys():
                if sen_true_rel_score[sen_no_space] == 1:
                    i_relevant.append(index_cnt)
            
#     cut_off = min(len(predicted_list),1000)
    cut_off = 1000
    
    average_precision = 0
    for rel_index in i_relevant:
        if rel_index <=cut_off:
            precision = calc_sen_precision_at_k(rel_index,predicted_list,true_data_list) 
            average_precision =  average_precision + precision
            
    return float(float(average_precision)/float(num_of_relevant))    
# 
def calc_doc_precision_at_k(k,predicted_list,true_data_list):
    doc_true_rel_score = {}
    for (doc,rel_score) in true_data_list:
        doc_true_rel_score[doc]=rel_score
     
    precision = 0
    for rank in range(0,k):
        if predicted_list[rank] in doc_true_rel_score.keys():
            if doc_true_rel_score[predicted_list[rank]] > 0:
                precision += 1
    precision = float(float(precision)/float(k))  
    return precision
                        
def calc_sen_precision_at_k(k,predicted_list,true_data_list): 
     
    sen_score={}
    exclude = set(string.punctuation)
    for (sen,supp_score) in true_data_list:
        sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
        sen_no_space = sen_no_punc.replace(" ","")
        sen_score[sen_no_space]=supp_score
     
    precision = 0
    for rank in range(0,k):
        sen_no_punc = ''.join(ch for ch in predicted_list[rank][0] if ch not in exclude)
        sen_no_space = sen_no_punc.replace(" ","")
        if sen_no_space in sen_score.keys():
            if sen_score[sen_no_space] > 0:
                precision += 1
    precision = float(float(precision)/float(k))  
    return precision

def calc_sen_AP_support(source,clm,predicted_list,true_data_list,p):
    """
    "1. find Rj = number of relevant docs/sen per claim j.
    2. find the ranks (=i) in which there are relevant items.
    3. calc precision at those i (until i=1000)
    4. divide by Rj
    """
    AP_cut_off = 28 #SHOULD CHANGE TO 30!!
    exclude = set(string.punctuation)
    sen_true_supp_score={}
    num_of_support = 0
    #find Rj
    
    for (sen,true_rel_score) in true_data_list:
        if true_rel_score == 1 or true_rel_score == 2:
            num_of_support += 1 
    
    for (sen,supp_score) in true_data_list:
        sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
        sen_no_space = sen_no_punc.replace(" ","")
        sen_true_supp_score[sen_no_space]=supp_score
    #find the i's in which there are rel items -> i =[1,...1000]
    i_support = []
    index_cnt = 0
    
    for (sen,supp_score) in predicted_list:
        index_cnt += 1
        if index_cnt <= AP_cut_off:
            sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
            sen_no_space = sen_no_punc.replace(" ","")
            if sen_no_space in sen_true_supp_score.keys():
                if sen_true_supp_score[sen_no_space] == 1 or sen_true_supp_score[sen_no_space] == 2:
                    i_support.append(index_cnt)
            
#     cut_off = min(len(predicted_list),1000)
#     cut_off = 1000
    
    average_precision = 0
    for supp_index in i_support:
        if supp_index <=AP_cut_off:
            precision = calc_sen_precision_at_k(supp_index,predicted_list,true_data_list) 
            average_precision =  average_precision + precision
            
    return float(float(average_precision)/float(num_of_support)) 
    
    
    sen_supp_score={}
    exclude = set(string.punctuation)
     
    for (sen,supp_score) in true_data_list:
        sen_no_punc = ''.join(ch for ch in sen if ch not in exclude)
        sen_no_space = sen_no_punc.replace(" ","")
        sen_supp_score[sen_no_space]=supp_score
        
    AP = 0
    precision = 0
    for i in range(0,p):
        sen_no_punc = ''.join(ch for ch in predicted_list[i][0] if ch not in exclude)
        sen_no_space = sen_no_punc.replace(" ","")
        if sen_no_space in sen_supp_score.keys():
            if sen_supp_score[sen_no_space] > 0:
                precision += 1
    AP = float(float(precision)/float(p))
    return  float(float(AP)/float(len(predicted_list))) #divide by the total relevant
                                   
def calc_CE(p,q):
    """
    calc the cross entropy between two vectors p,q
    sum over the vectors : p*log(q), not minus since we are looking for similarity, and not distance
    """
    if len(p) != len(q):
        print "p and q not equal length!"
    else:
        CE_res = 0
        for vector_idx in range(0,len(p)):
            CE_res +=p[vector_idx]*math.log(q[vector_idx])
        
        return CE_res        

def main():
    calc_NDCG()
    
if __name__ == '__main__':
    main() 