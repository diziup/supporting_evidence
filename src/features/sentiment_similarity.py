"""
Refactoring of the analyze_sentiment module -  06/01/15
"""
import os
from my_utils import utils
import sys
import numpy as np
import csv
import collections

class sentiment_similarity():
    claim_sentences_dict = {}
    setup = ""
    claims_dict = {}
    
    def __init__(self,setup,sentiment_files_path, input_sentiment_files_path,output_sentiment_files_path,sen_sim_res_path):
        self.claim_sentences_dict = {}
        self.setup = setup
        self.claims_dict = {}
        self.sentiment_path = sentiment_files_path
        self.input_sentiment_files_path = input_sentiment_files_path
        self.output_sentiment_files_path = output_sentiment_files_path
        self.stanford_path = r"/lv_local/home/liorab/softwares/stanford-corenlp-full-2014-06-16"
#         self.stanford_path = r"C:\softwares\stanford-corenlp-full-2014-06-16\stanford-corenlp-full-2014-06-16"
        self.sen_sim_res_path = sen_sim_res_path
        self.claim_sen_sentiment_vector_and_label_dict = {} #key is claim num and sen num, value is the sen sentiment vector and label 
        self.claim_sentiment_vector_and_label_dict = {}
        self.claim_sen_sentiment_cos_simialrity_socher={} #key is claim text and sen text, and val is the similairty based on the cosine between the vectors on the 
                                            #label itself given
        self.claim_sen_sentiment_JSD_simialrity_socher={} #key is claim text and sen text, and val is the similairty based on the Jansen-Shannen div between the vectors                            
        self.claim_sen_similarty_dict = {}
        # update 2.2.15 -keep the entropy of the sentiment vector
        self.claim_sentiment_vector_entropy = {} #key is a claim_num, value is the entropy of the sentiment vector
        self.claim_sen_sentiment_vector_entropy = {} #key is a claim_num and sen_num value is the entropy of the sentiment vector
        
    def create_claim_and_sentences_dict_from_retrieval_files(self,input_files_path):
        print "creating claim and sens dict..."
        self.claims_dict = utils.read_pickle("claim_dict")
        for f in os.listdir(input_files_path): 
            curr_file = open(input_files_path+"\\"+f)
            sen = curr_file.read().strip() # score, sentence
            for i, line in enumerate(sen.split('\n')):                   
                if i%2 == 0: # a metadata line
                    data = line.split(' ')
                    curr_claim =int(data[0])
                else:
                    if self.claim_sentences_dict.has_key(curr_claim):
                        self.claim_sentences_dict[curr_claim].append(line)
                    else:
                        self.claim_sentences_dict[curr_claim] = [line]
            
        utils.save_pickle(self.setup+"_claim_sentences", self.claim_sentences_dict)
        
    def create_sentiment_socher_input_files(self,sentiment_files_path):
        print "creating sentiment input files..."
        self.claim_sentences_dict = utils.read_pickle(self.setup+"_claim_sentences")
        self.claims_dict = utils.read_pickle("claim_dict")
        
        for (clm_num, sentences_list) in self.claim_sentences_dict.items():
            print "in claim", clm_num
            with open (sentiment_files_path+"\clm_"+str(clm_num),'wb') as clm_file:
                clm_file.write(self.claims_dict[str(clm_num)])
                for sen_num in range(0,len(sentences_list)):
                    with open (sentiment_files_path+"\clm_"+str(clm_num)+"_sen_" + str(sen_num),'wb') as sen_file:
                        sen_file.write(sentences_list[sen_num])
                        clm_file.close()
                        sen_file.close()
    
    def apply_socher_sentiment_analysis_tool_on_missing_files(self,empty_files_dict,sentiment_files_path):
        print "enter apply_socher_sentiment_analysis_tool in model: "
        try:
            for clm_num in empty_files_dict.keys(): 
#                     print "filename:" + filename
                for missing_f in empty_files_dict[clm_num]:
#                     input_file = open(sentiment_files_path+"/input_socher_"+str(clm_num)+"/missing_f","rb")
#                     if missing_f == filename:
#                         print "    found!"
                ##first check with the original model from the corenlp##
                    output_f = self.output_sentiment_files_path + missing_f + "_model_res.txt"
                    curr_stdout_file = open(output_f, "w")
                    
                    input_f = sentiment_files_path+"/input_socher_"+str(clm_num) +"/"+ missing_f
                    command = 'java -cp \"*\" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -output probabilities,root -file %s' % \
                    input_f +" >> %s " % output_f  
                    os.chdir(self.stanford_path)
                    stdout = os.system(command)
                    print stdout
                    curr_stdout_file.close()
        except Exception as err: 
            sys.stderr.write('problem in call_socher_sentiment_analysis_tool:')     
            print err.args      
            print err   
                       
    def apply_socher_sentiment_analysis_tool(self):                    
        print "enter apply_socher_sentiment_analysis_tool in model: "
        try:
            for filename in os.listdir(self.input_sentiment_files_path):
                print "filename:" + filename
                ##first check with the original model from the corenlp##
                output_f = self.output_sentiment_files_path + filename + "_model_res.txt"
                curr_stdout_file = open(output_f, "w")
                
                input_f = self.input_sentiment_files_path + filename
                command = 'java -cp \"*\" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -output probabilities,root -file %s' % \
                input_f +" >> %s " % output_f  
                os.chdir(self.stanford_path)
                stdout = os.system(command)
                print stdout
                curr_stdout_file.close()
#                 
#                 curr_stdout_file.write(', '.join(stdout))
#                 curr_stdout_file.close()
#                 proc = subprocess.Popen(shlex.split(command), stdout = subprocess.PIPE, shell = True, stderr = subprocess.PIPE,
#                                       cwd = self.stanford_path)
#                 stdout, stderr = proc.communicate()
#                 print stderr
#                 retcode = proc.wait()
#                 if retcode < 0:
#                     print >>sys.stderr, "error in command line", retcode
#                 curr_stdout_file = open(self.output_sentiment_files_path + filename + "_model_res.txt", "w")
#                 curr_stdout_file.write(stdout)
#                 curr_stdout_file.close()

        except Exception as err: 
            sys.stderr.write('problem in call_socher_sentiment_analysis_tool:')     
            print err.args      
            print err
   
    def process_sentiment_tool_result(self):
        print "enter process_sentiment_tool_result in model"
        try:
            self.claim_dict = utils.read_pickle("claim_dict")
#             self.claim_sen_dict = utils.read_pickle("claim_sen_dict")
           
            sentiment_options = ["Very negative","Negative","Neutral","Positive","Very positive"]
            for filename in os.listdir(self.output_sentiment_files_path):  # a curr_file for each claim and for each sentence
#             path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\sentiment_similarity\test"
#             for filename in os.listdir(path):
                claim_num = ""
                sen_num = ""
                sentiment_label = []
                sentiment_label_index = []
                sentiment_matrix = np.zeros(shape=((1,len(sentiment_options))))
                curr_file = open(self.output_sentiment_files_path+"\\"+filename,'r').read().strip()
                statinfo = os.stat(self.output_sentiment_files_path+filename)
#                 curr_file = open(path +"\\"+filename,'r').read().strip()
#                 statinfo = os.stat(path +"\\"++filename)
                if statinfo.st_size > 0:
                    #need to go over the lines to see if there is another sentence that got analyzed
                    for i, line in enumerate((curr_file).split('\n')):
                        if line.startswith("  0:  "):
                            sentiment_vector_str = line.split("  0:  ")[1].split()
                            num = map(float,sentiment_vector_str)
                            sentiment_matrix = np.vstack((sentiment_matrix,num))
                            continue
                        if len(line.split()) == 1 or len(line.split()) == 2: #a label line:
                            sentiment_label.append(line.strip())
                            continue
    #                     print line  
                    
                    #convert the sentiment label to numbers
                    for l in sentiment_label:
                        for i in range(0,len(sentiment_options)):
                            if l == sentiment_options[i]:
                                sentiment_label_index.append(i+1)
                    #determine for each claim and each sentence the overall sentiment vector and label             
                    claim_num = filename.split("_")[1]
                    if "sen" in filename:
                        sen_num =  filename.split("_")[3]
                    
                    #if it is more than one sentence, create a single label as a mean of the labels
                    sentiment_label_index = float(sum(sentiment_label_index)/float(len(sentiment_label_index)))
                    sentiment_matrix = np.delete(sentiment_matrix, 0, 0)
                    sentiment_matrix = np.mean(sentiment_matrix, axis=0) #the mean of the rows in case there is more than 1 sentence
    #                     
                    if sen_num is "": #only a claim    
                        self.claim_sentiment_vector_and_label_dict[claim_num] = (sentiment_matrix,sentiment_label_index)
                    else: #sentence of a claim
                        self.claim_sen_sentiment_vector_and_label_dict[(claim_num,sen_num)] = (sentiment_matrix,sentiment_label_index)         
            #save to file
#             with open ("claim_sentiment_socher.csv","wb") as csvfile: orig version of the model - from corenlp
            with open (self.sen_sim_res_path+ self.setup+ "_claim_sentiment_socher.csv","wb") as csvfile:
                clm_sen_socher = csv.writer(csvfile)
                for (clm,senti) in self.claim_sentiment_vector_and_label_dict.items():
                    clm_sen_socher.writerow([self.claim_dict[str(clm)]+" | "+','.join(map(str, senti[0]))+" | "+str(senti[1])])
#             with open ("sen_sentiment_socher_"+".csv","wb") as csvfile:
#                 sen_sen_socher = csv.writer(csvfile)
#                 for ((clm,sen),senti) in self.claim_sen_sentiment_vector_and_label_dict.items():
#                     sen_sen_socher.writerow([self.claim_dict[str(clm)]+" | "+self.claim_sen_dict[clm,int(sen)]+"|"+','.join(map(str, senti[0]))+" | "+str(senti[1])])
            #pickle it
            utils.save_pickle(self.sen_sim_res_path+ self.setup+"_claim_sentiment_vector_and_label_dict",self.claim_sentiment_vector_and_label_dict)
            utils.save_pickle(self.sen_sim_res_path+ self.setup+"_claim_sen_sentiment_vector_and_label_dict",self.claim_sen_sentiment_vector_and_label_dict)
            
        except Exception as err: 
            sys.stderr.write('problem in process_sentiment_tool_result:')     
            print err.args      
            print err
                
    def apply_sentiment_sim_tool(self):
   
#         self.create_claim_and_sentences_dict_from_retrieval_files(sentiment_files_path_input)
#         self.create_sentiment_socher_input_files(sentiment_files_path)
        self.apply_socher_sentiment_analysis_tool()
    
    def process_sentiment_result(self):
        
        self.process_sentiment_tool_result()
    
    def convert_claim_sen_sentiment_vector_and_label_dict(self):
        self.claim_sen_sentiment_vector_and_label_dict = utils.read_pickle(self.sen_sim_res_path+ self.setup+"_claim_sen_sentiment_vector_and_label_dict")
        claim_sen_list_sentiment_vector_and_label_dict = {}
        for ((clm,sen),sentiment_score) in self.claim_sen_sentiment_vector_and_label_dict.items():
            if claim_sen_list_sentiment_vector_and_label_dict.has_key(clm):
                claim_sen_list_sentiment_vector_and_label_dict[clm].append((sen,sentiment_score))
            else:
                claim_sen_list_sentiment_vector_and_label_dict[clm] = [(sen,sentiment_score)]
        utils.save_pickle(self.setup+"_claim_sen_list_sentiment_vector_and_label_dict", claim_sen_list_sentiment_vector_and_label_dict)
        
    def calc_sentiment_similarity_socher_tool(self):
        """
        given the claim sentiment dict- key is claim num and val is sentiment vector and label as given
        by Socher's tool,
        calculate the sentiment similarity between the claim and its sentences
        """
        print "enter calc_sentiment_similarity_socher_tool model:"
        
        self.claim_sentiment_vector_and_label_dict = utils.read_pickle(self.sen_sim_res_path+self.setup+"_claim_sentiment_vector_and_label_dict")
        print "num of claims:" +str(len(self.claim_sentiment_vector_and_label_dict.keys()))
        print self.claim_sentiment_vector_and_label_dict.keys()
        self.claim_sen_sentiment_vector_and_label_dict = utils.read_pickle(self.setup+"_claim_sen_list_sentiment_vector_and_label_dict")
                
        self.claim_dict = utils.read_pickle("claim_dict")
        self.claim_sentences_dict = utils.read_pickle(self.setup+"_claim_sentences") #the sen_num index in the index of the sentence in the list
        # of sentence in this dict
#         self.claim_sen_dict = utils.read_pickle("claim_sen_dict")
        
        #compute the similarity based on the label- a binary similarity
        for claim_num in self.claim_sentiment_vector_and_label_dict.keys():
            sentences_sentiment_score = self.claim_sen_sentiment_vector_and_label_dict[claim_num]
            print "in claim: "+ claim_num +" with "+str(len(sentences_sentiment_score)) +" sentences"
            for (sen,sentiment_vector_and_score) in sentences_sentiment_score:
#             for (clm,sen) in self.claim_sen_sentiment_vector_and_label_dict.keys():
#                 if claim_num == clm:
                    #17.09.14 update - removed the label sim, not interesting for now!
#                     if not self.claim_sen_sentiment_vector_and_label_dict[clm,sen][1] == 3.0: 
#                         sen_sim_based_on_label = math.fabs(self.claim_sentiment_vector_and_label_dict[claim_num][1]-self.claim_sen_sentiment_vector_and_label_dict[clm,sen][1])#e.g Very Posirive- Positive = 5-4=1
#                     else:
#                         sen_sim_based_on_label=10
                sen_sim_based_on_cosine = utils.cosine_measure(self.claim_sentiment_vector_and_label_dict[claim_num][0], sentiment_vector_and_score[0])
                #17.09.2014 edit  -  add similarity based on Jensen-Shannon div
                sen_sim_based_on_JSD = utils.jsd(self.claim_sentiment_vector_and_label_dict[claim_num][0], sentiment_vector_and_score[0])
                claim_sentiment_vector_entropy  = utils.calc_entropy(self.claim_sentiment_vector_and_label_dict[claim_num][0])
                sentence_sentiment_vector_entropy  = utils.calc_entropy(sentiment_vector_and_score[0])
#                     if sen_sim == 1 or sen_sim == 0:
                self.claim_sen_similarty_dict[claim_num,sen]=[sen_sim_based_on_JSD,sen_sim_based_on_cosine] #key is claim num and sen num, val is the
                self.claim_sentiment_vector_entropy[claim_num] =  claim_sentiment_vector_entropy
                self.claim_sen_sentiment_vector_entropy[claim_num,sen] = sentence_sentiment_vector_entropy
            print "current dict len" ,len(self.claim_sen_similarty_dict.keys())                                                                   #difference in the labels of the claim and sen sentiment - only cases of 1/0 matters 
                                                                                #(on 1-5 scale as Socher's output ands so 5-4, 4-4, 2 
                       
        #sort the claim sentence similarity dict by claim, and then by the sen_sim, in increarsing order
#         claim_sen_similarty_dict_based_on_label_sorted = collections.OrderedDict(sorted(self.claim_sen_similarty_dict.items(),key=lambda x: (-int(x[0][0]),-int(x[1][0])), reverse=True))
        claim_sen_similarty_dict_based_on_JSD_sorted = collections.OrderedDict(sorted(self.claim_sen_similarty_dict.items(),key=lambda x: (-int(x[0][0]),-float(x[1][0])), reverse=True)) #- float cus the smaller the JSD is, the more similar the clm and sen 
        claim_sen_similarty_dict_based_on_cosine_sorted = collections.OrderedDict(sorted(self.claim_sen_similarty_dict.items(),key=lambda x: (-int(x[0][0]),float(x[1][1])), reverse=True))           
        print "claim_sen_similarty_dict_based_on_cosine_sorted len" ,len(claim_sen_similarty_dict_based_on_cosine_sorted.keys())
        #save to file:
#         with open ("claim_sen_sentiment_similarity_based_on_label.csv","wb") as csvfile:
#             clm_sen_sim = csv.writer(csvfile)
#             for ((clm,sen),sim) in claim_sen_similarty_dict_based_on_label_sorted.items():
#                 clm_sen_sim.writerow([self.claim_dict[clm]+" | "+self.claim_sen_dict[clm,int(sen)]+" | "+str(sim[0])])
#                 self.claim_sen_sentiment_cos_simialrity_socher[(self.claim_dict[clm],self.claim_sen_dict[clm,int(sen)])]=[sim[0]]
        with open (self.setup+"_claim_sen_sentiment_similarity_based_on_cosine.csv","wb") as csvfile:
            clm_sen_sim = csv.writer(csvfile)
            cnt = 0
            for ((clm,sen),sim) in claim_sen_similarty_dict_based_on_cosine_sorted.items():
#                 clm_sen_sim.writerow([self.claim_dict[clm]+" | "+self.claim_sen_dict[clm,int(sen)]+" | "+str(sim[1])])
                clm_sen_sim.writerow([self.claim_dict[clm]+" | "+self.claim_sentences_dict[int(clm)][int(sen)]+" | "+str(sim[1])])
#                 self.claim_sen_sentiment_cos_simialrity_socher[(self.claim_dict[clm],self.claim_sen_dict[clm,int(sen)])].append(sim[1])
#                 self.claim_sen_sentiment_cos_simialrity_socher[(self.claim_dict[clm],self.claim_sen_dict[clm,int(sen)])]=sim[1]
                if self.claim_sen_sentiment_cos_simialrity_socher.has_key((self.claim_dict[clm],self.claim_sentences_dict[int(clm)][int(sen)])):
                    cnt += 1 
                else:
                    self.claim_sen_sentiment_cos_simialrity_socher[(self.claim_dict[clm],self.claim_sentences_dict[int(clm)][int(sen)])]=sim[1]
            print "existing items" ,cnt
        print "claim_sen_sentiment_cos_simialrity_socher len" , len(self.claim_sen_sentiment_cos_simialrity_socher.keys())
        
        with open ("claim_sen_sentiment_similarity_based_on_JSD.csv","wb") as csvfile:
            clm_sen_sim = csv.writer(csvfile)
            for ((clm,sen),sim) in claim_sen_similarty_dict_based_on_JSD_sorted.items():
                clm_sen_sim.writerow([self.claim_dict[clm]+" | "+self.claim_sentences_dict[int(clm)][int(sen)]+" | "+str(sim[0])])     
                self.claim_sen_sentiment_JSD_simialrity_socher[(self.claim_dict[clm],self.claim_sentences_dict[int(clm)][int(sen)])] = sim[0]
        #save to pickle
#         utils_linux.save_pickle("claim_sen_sentiment_cos_simialrity_socher_"+orig_retrinaed_model, self.claim_sen_sentiment_cos_simialrity_socher)
#         utils_linux.save_pickle("claim_sen_sentiment_JSD_simialrity_socher_"+orig_retrinaed_model, self.claim_sen_sentiment_JSD_simialrity_socher)
#         self.save_pickle("claim_sen_sentiment_cos_simialrity_socher", "claim_sen_sentiment_cos_simialrity_socher")
        #sort the results according to the cosine/JSD sim, from the most similar to the least similar -for the ranking
        claim_sen_sentiment_cos_simialrity_socher_sorted = collections.OrderedDict(sorted(self.claim_sen_sentiment_cos_simialrity_socher.items(),key=lambda x: (x[0][0],float(x[1])), reverse=True))
        claim_sen_sentiment_JSD_simialrity_socher_sorted = collections.OrderedDict(sorted(self.claim_sen_sentiment_JSD_simialrity_socher.items(),key=lambda x: (x[0][0],-float(x[1])), reverse=True))
        utils.save_pickle(self.sen_sim_res_path + self.setup+"_claim_sen_sentiment_cos_similarity_socher_sorted",claim_sen_sentiment_cos_simialrity_socher_sorted)
        utils.save_pickle(self.sen_sim_res_path + self.setup+"_claim_sen_sentiment_JSD_similarity_socher_sorted",claim_sen_sentiment_JSD_simialrity_socher_sorted)
        utils.save_pickle(self.sen_sim_res_path + self.setup +"_claim_sentiment_vector_entropy",self.claim_sentiment_vector_entropy)
        utils.save_pickle(self.sen_sim_res_path + self.setup +"_claim_sen_sentiment_vector_entropy",self.claim_sen_sentiment_vector_entropy)
        print "num of items in final dict: "+str(len(claim_sen_sentiment_cos_simialrity_socher_sorted.keys())) 
        
    def find_zero_files_and_apply_tool(self,sentiment_files_path):
        """
        find zero byte files, and activate the socher tool on them again
        """
        empty_files = {} #key is clm_num, value is a list of the files
        for filename in os.listdir(self.output_sentiment_files_path):
            statinfo = os.stat(self.output_sentiment_files_path+"/"+filename)    
            if "_sen_" in filename:
                clm_num = filename.split("clm_")[1].split("_sen_")
            else:
                clm_num = filename.split("clm_")[1].split("_model_res")[0]
            filename_saving = filename.split("_model_res.txt")[0]
            if statinfo.st_size == 0:
                if empty_files.has_key(clm_num):
                    empty_files[clm_num].append(filename_saving)
                else:
                    empty_files[clm_num] = [filename_saving]
        print "len empty_files: " +str(len(empty_files.keys()))        
        self.apply_socher_sentiment_analysis_tool_on_missing_files(empty_files,sentiment_files_path)
            
            
def main():
    setup = "support_baseline"
    sentences_files_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\claimEntity_sen_output"
    sentiment_files_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\support_baselines\sentiment_similarity"
    sentiment_files_path_input = sentiment_files_path+r"\\input_socher\\"
    sentiment_files_path_output = sentiment_files_path+r"\\output_socher\\"
#     sentiment_files_path_input = sentiment_files_path + r"/input_socher/"
#     sentiment_files_path_output = sentiment_files_path + r"/output_socher/"
    sen_sim_res_path = sentiment_files_path+"\sen_sim_res\\"
    sen_sim = sentiment_similarity(setup, sentiment_files_path, sentiment_files_path_input, sentiment_files_path_output,sen_sim_res_path)
#     sen_sim.process_sentiment_result()
#     d = {}
#     d = utils.read_pickle(sen_sim.sen_sim_res_path+ sen_sim.setup+"_claim_sen_sentiment_JSD_similarity_socher_sorted")
    sen_sim.calc_sentiment_similarity_socher_tool()
#     sen_sim.convert_claim_sen_sentiment_vector_and_label_dict()
#     sen_sim.find_zero_files_and_apply_tool(sentiment_files_path)
#     sen_sim.apply_sentiment_sim_tool()


    
if __name__ == '__main__':
    main()