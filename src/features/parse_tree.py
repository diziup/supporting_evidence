'''
19.06.14
@author: Liora
Input: parse tree from Stanford parser, in a one line format.
output - is a 'tree" structure. nodes are words/pieces of text and their POS -modifiers.
For use in the sentiment similarity module
'''
import sys
from compiler.ast import Node
from logging import root

class node:
        
    
    def __init__(self,text):
        self.working_text = text
        self.work_text=""
        self.word=""
        self.label = ""
        self.children = []
    
    def add_child_node(self,substring):
        child_node = node(substring)
        child_node.build_node()
        self.children.append(child_node)
        

    def build_node(self):
            open_count = 0;
            modifier = "";
            final_modifier = "";
            substring = "";
            tempIndex = 0;
            temp_str = "";
            
            for j in range(0,len(self.working_text)):
                if self.working_text[j] is ' ':
                    if modifier != "":
                        final_modifier = modifier
                    continue
                if self.working_text[j] is '(':
                    if final_modifier != "":
                        tempIndex = j
                        break
                else:
                    if final_modifier == "":
                        modifier+=self.working_text[j];
                    else:
                        if not self.working_text[j] is ')':
                            self.word += self.working_text[j]
                    
            self.label = final_modifier
            
            for i in range(j,len(self.working_text)):
                if self.working_text[i] == "(":
                    if open_count == 0:
                        substring = "("
                    else:
                        substring += self.working_text[i]
                    open_count += 1
                elif self.working_text[i] == ")":
                    open_count -= 1
                    if open_count == 0:
                        substring += self.working_text[i]
                        self.add_child_node(substring)
                    else:
                        substring += self.working_text[i]
                else:  
                    substring += self.working_text[i]

def main(args):
    try: 
#         input_parse=r"C:\study\technion\MSc\Thesis\Y!Test\Mturk\0614_support_sentiment_moviesonly\sentiment_sim_test\single_sen\parser_res"
        input_parse=args
        parsed = open(input_parse,'r').read().strip()
        root_nodes=[]
        
        for line in (parsed.split('\n')):
            root_node=node(line)
            root_node.build_node()
            root_nodes.append(root_node)
           
        print "finished parse_tree"
#         for r_node in root_nodes:      
#             for i in range(0,len(r_node.children)):
#                 print root_node.children[i].word +"|"+root_node.children[i].label
        
        return root_nodes
    
    except Exception as err: 
                    sys.stderr.write('problem in main:')     
                    print err.args      
                    print err

if __name__ == '__main__':
    main(sys.argv[1:]) 