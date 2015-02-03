'''
For the objective sentences Language model building, 
Using the IMDB ratings dump, create a list for each 1-5 "stars" category.
With the name of the movies in each list
With which afterward using the wikipedia API get these movies articles 
'''
import pandas as pd
import csv
import sys
from my_utils import utils_linux
import os


def read_IMDB_ratings():
    all_movies_rating_file=r"C:\supporting_evidence\external resources\IMDB\all_movies_IMDB_rating.csv"
    top_250_rating_file=r"C:\supporting_evidence\external resources\IMDB\250_top_movies.txt"
    bottom_10_movies=r"C:\supporting_evidence\external resources\IMDB\bottom_10_movies.txt"
    one_star_movie_name=[] 
    two_star_movie_name=[] 
    three_star_movie_name=[] 
    four_star_movie_name=[] 
    five_star_movie_name=[] 
    doc_list=[top_250_rating_file,bottom_10_movies,all_movies_rating_file]
    log_movies=[]
    not_movies=[]
    seen_movies=[]
    #go over the file a0nd save each film according to the rating it got
    for doc in doc_list:
        rating_f = open(doc,'r')
        rating_doc = rating_f.read().strip()
        for i,line in enumerate(rating_doc.splitlines()):
            try:
                rating=float(line.split('|')[3])
                movie_name=line.split('|')[4].strip()
                movie_year=line.split('|')[5].strip().split("(")[1].split(")")[0]
                if not movie_name in seen_movies:
                    if utils_linux.query_DBpedia(movie_name) ==1:
                        seen_movies.append(movie_name)
                        print "found " +movie_name
                        if rating >=1 and rating <4:
                            if not movie_name+"_"+movie_year in one_star_movie_name:
                                one_star_movie_name.append(movie_name+"_"+movie_year)
                        elif rating >=4 and rating <5:
                            if not movie_name+"_"+movie_year in two_star_movie_name:
                                two_star_movie_name.append(movie_name+"_"+movie_year)
                        elif rating >=5 and rating < 7:
                            if not movie_name+"_"+movie_year in three_star_movie_name:
                                three_star_movie_name.append(movie_name+"_"+movie_year)
                        elif rating >=7 and rating <8.5:
                            if not movie_name+"_"+movie_year in four_star_movie_name:
                                four_star_movie_name.append(movie_name+"_"+movie_year)
                        elif rating >=8.5 and rating <=10:
                            if not movie_name+"_"+movie_year in five_star_movie_name: 
                                five_star_movie_name.append(movie_name+"_"+movie_year)
                    else:
                        continue
                else:
                    not_movies.append(movie_name)    
            except Exception as err: 
#                 sys.stderr.write('problem in read_IMDB_ratings: in doc '+doc+" in line: " +line)     
                print err.args      
                print err
                log_movies.append(movie_name)
    #create a file for each list
    print "finishe, now save"
    with open("one_star_movies.csv", 'wb') as csvfile:
        w_one_star = csv.writer(csvfile)
        for movie in one_star_movie_name:
                w_one_star.writerow([movie])
    
    with open("two_star_movies.csv", 'wb') as csvfile:
        w_two_star = csv.writer(csvfile)
        for movie in two_star_movie_name:
                w_two_star.writerow([movie])
                
    with open("three_star_movies.csv", 'wb') as csvfile:
        w_three_star = csv.writer(csvfile)
        for movie in three_star_movie_name:
                w_three_star.writerow([movie])
    
    with open("four_star_movies.csv", 'wb') as csvfile:
        w_four_star = csv.writer(csvfile)
        for movie in four_star_movie_name:
                w_four_star.writerow([movie])
    
    with open("five_star_movies.csv", 'wb') as csvfile:
        w_five_star = csv.writer(csvfile)
        for movie in five_star_movie_name:
                w_five_star.writerow([movie])
    with open("IMDB_log.csv", 'wb') as csvfile:  
        log=csv.writer(csvfile)
        for movie in log_movies:
            log.writerow([movie])
    with open("not_movies.csv", 'wb') as csvfile:  
        log=csv.writer(csvfile)
        for movie in not_movies:
            log.writerow([movie])
    
def concatinate_documents_to_single_doc():
    """for the objective language model,from
    for each folder, open all the files and concatinate to a single file 
    """ 
    stars_list=["one_star","two_star","three_star","four_star","five_star"]
    docs_path="C:\supporting_evidence\external resources\IMDB\movie_articles"
    for star in stars_list:
        curr_star_docs_sentence=""
        for filename in os.listdir(docs_path+"\\"+ star):
            with open(docs_path+"\\"+ star+"\\"+filename, 'r') as f:
                doc_lines=f.read()
                curr_star_docs_sentence+=doc_lines
        with open(star+"_single_doc.txt",'wb') as csvfile:
            f=csv.writer(csvfile)
            f.writerow([curr_star_docs_sentence])
           
        
def main():
    concatinate_documents_to_single_doc()
    
if __name__ == '__main__':
    main() 