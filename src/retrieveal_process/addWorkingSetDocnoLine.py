"""
Feb 2014
This module will take the docno that are the result of
document retrieval,
 and add them as a line for each query element in a query file for sentence retrieval
 
 arg 1: doc file
 arg 2: query for sentence file
  
"""
import sys
doc_file_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\seqDocRes"
# query_sen_path=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0214_second_moviesonly\claims_s_.txt_mrf"
claim_file_path = r"" 

def main(argv):
    doc_file = open(doc_file_path,'r')
    doc = doc_file.read().strip() 
    querySen_file = open(claim_file_path,'r')
    sentence = querySen_file.read().strip() 
    for i, line in enumerate(sentence.split('\n')):
        if line.contains("<number>"):
            line.split("<number>")[1]
    

if __name__ == '__main__':
    main(sys.argv[1:])  