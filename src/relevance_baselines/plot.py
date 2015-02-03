'''
Created on Oct 27, 2014

@author: liorab
'''

import matplotlib.pyplot as plt
import numpy as np
from my_utils import utils

def plot_graph():
    """
    for a given alpha,beta value - plot the nDCG and MAP across lambda vals
    read the res file, and plot the values
    """ 
    p = 10
    base_path = r"C:\study\technion\MSc\Thesis\Y!\support_test\baseline_clmLMdocLM"
    nDCG_MAP_res = base_path +"\\measures_res_corpus_smoothing\\"
    k_val = 50
    each_params_AVGnDCG_MAP_dict = utils.read_pickle(nDCG_MAP_res+"each_params_AVGnDCG_MAP_prec_at_k_dict_top_k_docs_"+str(k_val)+"_at_"+str(p))
    measures_std = utils.read_pickle(nDCG_MAP_res+"nDCG_MAP_prec_at_k_std_for_each_configuration_k_top_docs_"+str(k_val)+"_at_"+str(p)) #key is configuration, value is the nDCG std and AP std
#     NDCG_AP_all_claims_all_param_values = utils.read_pickle(nDCG_MAP_res+"NDCG_AP_all_claims_all_param_values")   #key:clm,alpha_f,beta_f,k_val,lambda_f 
#     alpha_f = 0.5
    lambda_f = 0.5
#     alpha_f = 0.8
    beta_f = 0.0
    variable = "alpha"
    num_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    N = len(num_range)
    Avg_nDCG = []
    MAP = []
    avg_p_at_5 = []
    std_p_at_5 = []
    avg_p_at_10 = []
    std_p_at_10 = []
    std_nDCG = []
    std_AP = []
    
    for alpha_f in num_range: #a list of 11 values - for each lambda 
        Avg_nDCG.append(each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][0])
        MAP.append(each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][1])
        avg_p_at_5.append(each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][2])
        avg_p_at_10.append(each_params_AVGnDCG_MAP_dict[alpha_f,beta_f,k_val,lambda_f][3])
        std_nDCG.append(measures_std[alpha_f,beta_f,k_val,lambda_f][0])
        std_AP.append(measures_std[alpha_f,beta_f,k_val,lambda_f][1])  
        std_p_at_5.append(measures_std[alpha_f,beta_f,k_val,lambda_f][2])  
        std_p_at_10.append(measures_std[alpha_f,beta_f,k_val,lambda_f][3])  
         
#     ind = np.arange(N)  # the x locations for the groups
#     width = 0.15       # the width of the bars
#     
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(ind, avg_p_at_5, width, color='y', yerr = std_p_at_5)
#     rects2 = ax.bar(ind+width, avg_p_at_10, width, color='m', yerr = std_p_at_10)
#     rects3 = ax.bar(ind+2*width, MAP, width, color='c',yerr = std_AP )
#     rects4 = ax.bar(ind+3*width, Avg_nDCG, width, color='0.75',yerr = std_nDCG )
#      
#    
#     # add some
#     plt.xlabel(variable)
# #     plt.ylabel('values')
# #     ax.set_ylabel('values')
# #     ax.set_title('lambda')
#      
#     ax.set_xticks(ind+width)
#     ax.set_xticklabels( ('0', '0.1', '0.2', '0.3', '0.4','0.5','0.6','0.7','0.8','0.9','1') )
#       
#     ax.legend( (rects1[0], rects2[0],rects3[0],rects4[0]), ('avg_p@5', 'avg_p@10','MAP','avg_nDCG') )
#      
#     def autolabel(rects):
#         # attach some text labels
#         for rect in rects:
#             height = rect.get_height()
#             ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.3f'%float(height),
#                     ha='center', va='bottom',fontsize=10)
#      
#     autolabel(rects1)
#     autolabel(rects2)
#     autolabel(rects3)
#     autolabel(rects4)
#      
#     plt.show()
#     plt.savefig("beta0lambda05_k_50.jpg",dpi=100)
    
    #http://matplotlib.org/users/pyplot_tutorial.html
    #http://stackoverflow.com/questions/8409095/matplotlib-set-markers-for-individual-points-on-a-line
    #https://bespokeblog.wordpress.com/2011/07/07/basic-data-plotting-with-matplotlib-part-2-lines-points-formatting/
    x_values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    plt.plot(x_values,avg_p_at_5 ,marker='o', linestyle='--', color='b', label='p@5')
    plt.plot(x_values, avg_p_at_10, marker='v', linestyle='--', color='b', label='p@10')
    plt.plot(x_values, MAP, marker='*', linestyle='--', color='b', label='MAP')
    plt.plot(x_values, Avg_nDCG, marker='D', linestyle='--', color='b', label='avg_nDCG')
    plt.xlabel(variable)
    plt.ylabel('values')
    plt.legend()
    plt.show()
    print "kui"
def main():
    plot_graph()
    
    
if __name__ == '__main__':
    main()
