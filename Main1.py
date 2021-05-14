import sys
#from pyspark.sql import SparkSession, functions, types, Row
import re
import pandas as pd
import numpy as np
import math
from scipy import stats
from pyspark.sql import SparkSession, functions, types, Row
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dateT

import build_wikidata_movies as BWM # import make_wikidata_movies
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs





spark = SparkSession.builder.appName('correlate logs').getOrCreate()
assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+


#otten_tomatoes.show()
#wikidata_movies=wikidata_movies.drop(wikidata_movies['wiki_id'])
"""wiki_data=wikidata_movies.groupBy('imdb_id')

wiki_data=wiki_data.filter(wiki_data['enwiki_title']!= 'null' ).filter(wiki_data['rotten_tomatoes_id']!= 'null' )
wiki_data=wiki_data.filter(wiki_data['main_subject']!= 'null' )
"""




#wiki_with_tomatoes['filming_location'],wiki_with_tomatoes['main_subject'],

#wiki_with_tomatoes.show()


"""
	I need to exclude the null enwiki title b/c if there isn't build them I cannot get enought message
	Exclude non-label abject, that I don't know the name
	I want to analyze the director and cast member, therefore there cannot be null
"""


#.filter(wiki_data_with_genre['genre_label']!="null" )\ <--- there is only 3 genre_label on it, it might not help 


#wiki_with_tomatoes.show()
#wiki_with_tomatoes=wiki_with_tomatoes['publication_date'].apply(types.TimestampType())


def string_to_date(string):
	return int(string[0:4])

def split_years(data): # according to the publication_date return to data ( I use 2016)
	#the year<2016
	A=data[data['publication_date']<2016]
	#the year>=2016
	B=data[data['publication_date']>=2016]
	return A,B


#wliki_director_and_most_genre.show()
#wiki_data_with_tomato.show()
"""
def Check_location(filming_location):
	if filming_location==null:
		return None
	else
		fl=filming_location.astype(types.StringType())
		return fl
""" 
# defualt already use String

def GDM(data): #input as a seriesGet_Directors_Movies_(Rate
	D_list = []
	Movie_list = []
	Aavg_list = []
	CRate_list = []
	Aper_list = []
	#Arate_list = []
	Cper_list = []
	award_list = []
	Date_list=[]
	size = data['director'].size
	
	for count in range (size) : 
	#the column can interpreted as an integer 
		len_of_series = len(data['director'][count]) # how many directors (int)
		if (len_of_series>1):
			
			for i in range (len_of_series):
				D_list.append(str(data['director'][count][i]))
				Movie_list.append(str(data['label'][count]))
				Aavg_list.append(data['audience_average'][count])
				CRate_list.append(data['critic_average'][count])
				award_list.append(str(data['awards'][count]))
				Aper_list.append(str(data['audience_percent'][count]))
				Date_list.append(int(data['publication_date'][count]))
				Cper_list.append(str(data['critic_percent'][count]))
				
				#if data['omdb_awards'][count]=="N/A":
				#	award_list.append(np.nan)
				#else:
				#	award_list.append(str(data['omdb_awards'][count]))"""
		elif len_of_series==1:
			D_list.append(str(data['director'][count][0]))
			Movie_list.append(str(data['label'][count]))
			Aavg_list.append(data['audience_average'][count])
			CRate_list.append(data['critic_average'][count])
			award_list.append(str(data['awards'][count]))
			Aper_list.append(str(data['audience_percent'][count]))
			Date_list.append(int(data['publication_date'][count]))
			Cper_list.append(str(data['critic_percent'][count]))
			#if data['omdb_awards'][count]=="N/A":
				#award_list.append(np.nan)
			#else:
				
		else:
			pass

	DM_list=pd.DataFrame({'directorID':D_list,'Movie_names':Movie_list,\
		'audience_average':Aavg_list,'critic_average':CRate_list,\
		'audience_percent':Aper_list,'critic_percent':Cper_list,\
		'awards':award_list,'publication_date':Date_list})
	#,'audience_ratings':Arate_list
	DM_list=DM_list.dropna()
	
	# clean up all the NaN
	return DM_list

def get_Oscars(text):
	p = re.compile(r"Won (\d+) Oscars")
	win_result=p.search(text)
	if win_result!=None:
		return int(win_result.group(1))
	else: 
		return 0

def get_Win(text):
	p = re.compile(r"(\d+) win")
	win_result=p.search(text)
	if win_result!=None:
		return int(win_result.group(1))
	else: 
		return 0

def get_nomation(text):
	p = re.compile(r"(\d+) nomination")
	win_result=p.search(text)
	if win_result!=None:
		return int(win_result.group(1))
	else: 
		return 0

def get_Onomi(text):
	p = re.compile(r"Nominated for (\d+) Oscars")
	win_result=p.search(text)
	if win_result!=None:
		return int(win_result.group(1))
	else: 
		return 0
#wiki_with_tomatoes['director'].show()

#DM_list=DM_list.dropna()

"""DMCA_group=DM_list.groupby(['directorID'])['critic_average'].mean()
DMCP_group=DM_list.groupby(['directorID'])['critic_percent'].mean()
"""
def get_award_analyze(DM_list):
	
	DM_list['nomi_times']=DM_list['awards'].apply(get_nomation)
	DM_list['win_times']=DM_list['awards'].apply(get_Win)
	
	DM_list['Oscars_nomi']=DM_list['awards'].apply(get_Onomi)
	DM_list['Oscars']=DM_list['awards'].apply(get_Oscars)
	return DM_list


def To_dataframe(DM_list):
	DMAA_group=DM_list.groupby(['directorID'])['audience_average'].mean().reset_index()
	DMAP_group=DM_list.groupby(['directorID'])['critic_average'].mean().reset_index()
	DMWT_group=DM_list.groupby(['directorID'])['win_times'].mean().reset_index()
	DMNT_group=DM_list.groupby(['directorID'])['nomi_times'].mean().reset_index()
	DMO_group=DM_list.groupby(['directorID'])['Oscars'].mean().reset_index()
	DMON_group=DM_list.groupby(['directorID'])['Oscars_nomi'].mean().reset_index()
	result=pd.concat([DMAA_group,DMAP_group,DMWT_group,DMNT_group,DMO_group,DMON_group],axis=1,keys='directorID')
		# Now each one is the value of pandas.core.series.Series and resultcan equal to .values
	res=pd.DataFrame({'directorID':result.d.directorID.values,\
		'audience_average':result.d.audience_average.values,\
		'critic_average':result.i.critic_average.values,\
		'nomi_times':result.e.nomi_times.values,\
		'win_times':result.r.win_times.values,\
		'Oscars_time':result.c.Oscars.values,\
		'Oscars_nomination':result.t.Oscars_nomi.values,\
		})
	
	return res

def run_the_MLs(X,y):
	
	X_train, X_test, y_train, y_test=train_test_split(X,y)

	modelKNC=KNeighborsClassifier(n_neighbors=40)#96
	modelGNB=GaussianNB()#82
	modelSVC=SVC(10)#96
	modelKNC.fit(X_train,y_train)
	modelGNB.fit(X_train,y_train)
	modelSVC.fit(X_train,y_train)
	tested=pd.DataFrame()
	#tested['directorID']=dataGet['directorID']
	tested['truth']=y_test
	tested['KNC']=modelKNC.predict(X_test)
	tested['GNB']=modelGNB.predict(X_test)
	tested['SVC']=modelSVC.predict(X_test)

	print(tested)
	print('KNC =' , modelKNC.score(X_test,y_test))
	print('GNB =' , modelGNB.score(X_test,y_test))
	print('SVC =' , modelSVC.score(X_test,y_test))



	#print(TT)#['audience_percent'],['critic_average'],['critic_percent']].values)
	#X=DMA_group.drop('director', axis=1).values
	#y=DMA_group['director'].values
	#X_train, X_test, y_train, y_test=train_test_split(X,y)
def run_Two_data_predicts(Train_data,Test_data):
	# the column is ['audience_percent'],['critic_average'],['critic_percent']

	X_train=Train_data.loc[:,'nomi_times':'win_times']
	#X_train=X_train.drop('win_times','nomi_times')
	y_train=Train_data['Oscars']
	X_test=Test_data.loc[:, 'nomi_times':'win_times']
	#X_test=X_test.drop('win_times','nomi_times')
	y_test=Test_data.loc[:,'Oscars']


	modelKNC=KNeighborsClassifier(n_neighbors=40)#96
	modelGNB=GaussianNB()#82
	modelSVC=SVC(10)#96
	modelKNC.fit(X_train,y_train)
	modelGNB.fit(X_train,y_train)
	modelSVC.fit(X_train,y_train)
	tested=pd.DataFrame()
		#tested['directorID']=dataGet['directorID']
	tested['truth']=y_test
	tested['truth']=y_test
	tested['KNC']=modelKNC.predict(X_test)
	tested['GNB']=modelGNB.predict(X_test)
	tested['SVC']=modelSVC.predict(X_test)


	print(tested)
	print('KNC =' , modelKNC.score(X_test,y_test))
	print('GNB =' , modelGNB.score(X_test,y_test))
	print('SVC =' , modelSVC.score(X_test,y_test))
#result=DMAA_group.join(DMAP_group,on='directorID')
#'critic_average':DMAP_group,'win_times':DMWT_group,'nomi_times':DMNT_group,'Oscars':DMO_group

"""result.to_csv('avgs.csv', index=False)
#spark-submit Main.py wikidata-movies.json.gz omdb-data.json.gz rotten-tomatoes.json.gz genres.json.gz
DM_list.to_csv('DM_list.csv', index=False)"""
def main(in_directory1,in_directory2,in_directory3,in_directory4):
	#include all the data with all the connect coolumn
	wikidata_movies = spark.read.json(in_directory1).cache()
	#with omdb data canbe connect with imdb
	omdb_data=spark.read.json(in_directory2).cache()
	#with rating data canbe connect with imdb
	rotten_tomatoes=spark.read.json(in_directory3).cache()
	#with genres data canbe connect with wikidata id
	genresa=spark.read.json(in_directory4).cache()
	
	#Oscar=spark.read.csv('Oscar_2003_2018.csv').cache()
	#academy_awards=spark.read.csv('academy_awards_07-17-13_１.csv').cache()
	#omdb_data.show()
	rotten_tomatoes=rotten_tomatoes.select(rotten_tomatoes['imdb_id'],
		rotten_tomatoes['audience_average'],
		rotten_tomatoes['audience_percent'],
		rotten_tomatoes['audience_ratings'],
		rotten_tomatoes['critic_average'],
		rotten_tomatoes['critic_percent']
		).cache()
	rotten_tomatoes=functions.broadcast(rotten_tomatoes)
	wiki_with_toma=wikidata_movies.join(rotten_tomatoes, on ='imdb_id',how='left')
	wiki_with_toma=functions.broadcast(wiki_with_toma)
	wiki_with_tomatoes=wiki_with_toma.join(omdb_data, on ='imdb_id',how='inner')
	wiki_with_tomatoes=wiki_with_tomatoes.select(wiki_with_tomatoes['based_on'],
		wiki_with_tomatoes['publication_date'],
		wiki_with_tomatoes['cast_member'],
		wiki_with_tomatoes['country_of_origin'],
		wiki_with_tomatoes['director'],#.astype(types.ArrayType(types.StringType())),
		wiki_with_tomatoes['enwiki_title'],
		wiki_with_tomatoes['genre'],
		wiki_with_tomatoes['imdb_id'],
		wiki_with_tomatoes['label'],
		(wiki_with_tomatoes['omdb_awards']).alias('awards'),
		(wiki_with_tomatoes['audience_average']/10).alias('audience_average'),
		(wiki_with_tomatoes['audience_percent']/100).alias('audience_percent'),
		wiki_with_tomatoes['audience_ratings'],
		(wiki_with_tomatoes['critic_average']/10).alias('critic_average'),
		(wiki_with_tomatoes['critic_percent']/100).alias('critic_percent')).cache()

	wiki_with_tomatoes=wiki_with_tomatoes.filter(wiki_with_tomatoes['director'][0]!="null" )\
		.filter(wiki_with_tomatoes['publication_date']!="null" )\
		.filter(wiki_with_tomatoes['cast_member'][0]!="null" )\
		.filter(wiki_with_tomatoes['enwiki_title']!="null" )\
		.filter(wiki_with_tomatoes['critic_average']<10 )\
		.filter(wiki_with_tomatoes['critic_average']>=0 )\
		.filter(wiki_with_tomatoes['audience_average']<10 )\
		.filter(wiki_with_tomatoes['audience_average']>=0 )\
		.filter(wiki_with_tomatoes['audience_percent']<100 )\
		.filter(wiki_with_tomatoes['audience_percent']>=0 )\
		.filter(wiki_with_tomatoes['critic_percent']<100 )\
		.filter(wiki_with_tomatoes['critic_percent']>=0 )\
		.filter(wiki_with_tomatoes['label']!="null" )

#-----------------------------------------------------------End of the filter and inputs



	Pd1_datas=wiki_with_tomatoes.toPandas()
	Pd1_datas['publication_date']=Pd1_datas['publication_date'].apply(string_to_date)
	#Pd_datas.to_csv('Pd_datas.csv', index=False)

	Pd_datas,Test_data=split_years(Pd1_datas)
	DM_list=GDM(Pd1_datas)
	DM_list=get_award_analyze(DM_list)#Get the awards details
	Pd_datas=get_award_analyze(Pd1_datas)#Get the awards details
	
	
	result=To_dataframe(DM_list)

#------------------------------------------------Print the linear regression

	AT=Pd_datas[Pd_datas['win_times']==0]
	ATT=Pd_datas[Pd_datas['win_times']>0]
	ATTO=ATT[ATT['Oscars']>0]

	print(stats.linregress(Pd_datas['audience_average'],Pd_datas['audience_percent']))
	print(stats.linregress(Pd_datas['critic_average'],Pd_datas['critic_percent']))
	X = Pd_datas['audience_average'].values[:, np.newaxis]
	model = LinearRegression(fit_intercept=True)
	model.fit(X , Pd_datas['audience_percent'].values)
	print(model.coef_[0], model.intercept_)
	plt.figure(figsize=(10, 5))
	plt.plot(AT['audience_average'], AT['audience_percent'], 'y.',alpha=0.5)
	plt.plot(ATT['audience_average'], ATT['audience_percent'], 'b.',alpha=0.5)
	plt.plot(ATTO['audience_average'], ATTO['audience_percent'], 'g.',alpha=1.0)
	plt.plot(Pd_datas['audience_average'].values, model.predict(X), 'r-')
	plt.legend(['training data', 'predicted line'])
	plt.savefig('Test3-1.png')

	X = Pd_datas['critic_average'].values[:, np.newaxis]
	model = LinearRegression(fit_intercept=True)
	model.fit(X , Pd_datas['critic_percent'].values)
	print(model.coef_[0], model.intercept_)
	plt.figure(figsize=(10, 5))
	plt.plot(AT['critic_average'], AT['critic_percent'], 'y.',alpha=0.5)
	plt.plot(ATT['critic_average'], ATT['critic_percent'], 'b.',alpha=0.5)
	plt.plot(ATTO['critic_average'], ATTO['critic_percent'], 'g.',alpha=1.0)
	plt.plot(Pd_datas['critic_average'].values, model.predict(X), 'r-')
	plt.legend(['training data', 'predicted line'])
	plt.savefig('Test4-1.png') 

	Oscars_nomination=Pd_datas[Pd_datas['Oscars_nomi']>0]
	Oscars_winner=Pd_datas[Pd_datas['Oscars']>0]
	k=Pd_datas[Pd_datas['Oscars_nomi']==0]
	k=k[k['Oscars']==0]


	X = Oscars_nomination['audience_average'].values[:, np.newaxis]
	model = LinearRegression(fit_intercept=True)
	model.fit(X , Oscars_nomination['critic_average'].values)
	print(model.coef_[0], model.intercept_)
	plt.figure(figsize=(10, 5))
	plt.plot(k['audience_average'], k['critic_average'], 'y.',alpha=0.5)
	plt.plot(Oscars_nomination['audience_average'], Oscars_nomination['critic_average'], 'b.',alpha=0.5)
	plt.plot(Oscars_winner['audience_average'], Oscars_winner['critic_average'], 'g.',alpha=1.0)
	#plt.plot(Oscars_nomination['critic_average'].values, model.predict(X), 'r-')
	plt.legend(['No nomination', 'get nomination','get Oscar'])
	plt.savefig('Test_Oscar.png') 

	model = LinearRegression(fit_intercept=True)
	model.fit(X , Oscars_nomination['critic_average'].values)
	print(model.coef_[0], model.intercept_)
	plt.figure(figsize=(10, 5))
	plt.plot(k['audience_percent'], k['critic_percent'], 'y.',alpha=0.5)
	plt.plot(Oscars_nomination['audience_percent'], Oscars_nomination['critic_percent'], 'b.',alpha=0.5)
	plt.plot(Oscars_winner['audience_percent'], Oscars_winner['critic_percent'], 'g.',alpha=1.0)
	#plt.plot(Oscars_nomination['critic_average'].values, model.predict(X), 'r-')
	plt.legend(['No nomination', 'get nomination','get Oscar'])
	plt.savefig('Test_Oscar2.png') 


#------------------------------------------------Print the boxplot
	plt.figure()
	plt.boxplot(Oscars_winner['critic_average'], notch=True, vert=True)
	plt.legend(['win Oscar critic score'])
	plt.savefig('Oscar1.png')

	plt.figure()
	plt.boxplot(Oscars_winner['audience_average'], notch=True, vert=True)
	plt.legend(['win Oscar audience score'])
	plt.savefig('Oscar2.png')

	plt.figure()
	plt.boxplot(Oscars_nomination['critic_average'], notch=True, vert=True)
	plt.legend(['Oscar nomination critic score'])
	plt.savefig('Oscar3.png')

	plt.figure()
	plt.boxplot(Oscars_nomination['audience_average'], notch=True, vert=True)
	plt.legend(['Oscar nomination audience score'])
	plt.savefig('Oscar4.png')

	 
	k=Pd_datas[Pd_datas['Oscars']==0]
#------------------------------------------------Print the hist
	plt.figure()
	plt.hist([Oscars_winner['critic_average'],Pd_datas['critic_average']])
	plt.savefig('OscarHis1.png')

	plt.figure()
	plt.hist([Oscars_winner['audience_average'],Pd_datas['audience_average']])
	plt.savefig('OscarHis2.png')

	plt.figure()
	plt.hist([Pd_datas['audience_average'],k['audience_average']])
	plt.savefig('OscarHis3.png')

	plt.figure()
	plt.hist([Pd_datas['critic_average'],k['critic_average']])
	plt.savefig('OscarHis4.png')

	print(stats.normaltest(Oscars_winner['critic_average']).pvalue)
	print(stats.normaltest(Oscars_nomination['critic_average']).pvalue)
	print(stats.normaltest(Oscars_winner['audience_average']).pvalue)
	print(stats.normaltest(Oscars_nomination['audience_average']).pvalue)
	print('A=',stats.ttest_ind(Oscars_winner['audience_average'],k['audience_average']))
	print('C=',stats.ttest_ind(Oscars_winner['critic_average'],k['critic_average']))
	print('A=',Oscars_winner['audience_average'].mean(),k['audience_average'].mean())
	print('C=',Oscars_winner['critic_average'].mean(),k['critic_average'].mean())
#------------------------------------------------Print the hist
	Train_data,Test_data=split_years(Pd_datas)
	#Train_data.to_csv('Train_data.csv', index=False)
	#Test_data.to_csv('Test_data.csv', index=False)
	run_Two_data_predicts(Train_data,Test_data)



if __name__ == '__main__':
	in_directory1=sys.argv[1]
	in_directory2=sys.argv[2]
	in_directory3=sys.argv[3]
	in_directory4=sys.argv[4]
	main(in_directory1,in_directory2,in_directory3,in_directory4)

plt.figure(figsize=(10, 5))
plt.plot(k['audience_average'],k['critic_average'],'y.',alpha=0.4)
plt.plot(nomi_some['audience_average'],nomi_some['critic_average'],'r.',alpha=0.7)
plt.plot(win_some['audience_average'],win_some['critic_average'],'b.',alpha=1.0)
plt.savefig('normal win and nomi.png')
plt.title('normal win and nomi')
plt.ylabel('Audience average')
plt.xlabel('Critic average')

k=Pd_datas[Pd_datas['Oscars_nomi']==0]
k=k[k['Oscars']==0]

Os_nomi=Pd_datas[Pd_datas['Oscars_nomi']>0]
Os_win=Os_nomi[Os_nomi['Oscars']>0]



plt.figure(figsize=(10, 5))
plt.plot(k['audience_average'],k['critic_average'],'y.',alpha=0.4)
plt.plot(Os_nomi['audience_average'],Os_nomi['critic_average'],'r.',alpha=0.7)
plt.plot(Os_win['audience_average'],Os_win['critic_average'],'b.',alpha=1.0)
plt.savefig('Oscars win and nomi.png')
plt.title('Oscars win and nomi')
plt.ylabel('Audience average')
plt.xlabel('Critic average')


############################################################################
#Topic 3 directors
"""
Oscars_nomination=result[result['Oscars_nomination']>0]
Oscars_winner=result[result['Oscars_time']>0]
k=result[result['Oscars_time']==0]
k=k[k['Oscars_nomination']==0]

plt.figure(figsize=(10, 5))
plt.plot(k['audience_average'],k['critic_average'],'.',alpha=0.4)
plt.plot(Oscars_nomination['audience_average'],Oscars_nomination['critic_average'],'.',alpha=0.7)
plt.plot(Oscars_winner['audience_average'],Oscars_winner['critic_average'],'.',alpha=1.0)
plt.savefig('wikipedia.png')
plt.title('Director win and nomi Oscars ')
plt.ylabel('Audience average')
plt.xlabel('Critic average')


Audience_Ttest_report=stats.ttest_ind((Pd_datas['audience_average'].values)/10,\
	(Pd_datas['audience_percent'].values)/100)
Critic_Ttest_report=stats.ttest_ind((Pd_datas['critic_average'].values)/10,\
	(Pd_datas['critic_percent'].values)/100)

print('Audience p-values=',Audience_Ttest_report)
print('Critic p-values=',Critic_Ttest_report)
plt.figure(figsize=(10, 5))
plt.plot(Pd_datas['critic_average'],Pd_datas['critic_average'])
plt.savefig('Audience.png')

plt.figure(figsize=(10, 5))
plt.plot(Pd_datas['critic_average'],Pd_datas['critic_average'])
plt.savefig('Critic.png')
"""


"""plt.figure(figsize=(10, 5)) # change the size to something sensible
#plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
plt.plot(Oscars_winner['audience_average'],Oscars_winner['critic_average'],style='r')
plt.plot(Oscars_nomination['audience_average'],Oscars_nomination['critic_average'],style='r')
plt.plot(k['audience_average'],k['critic_average'],style='r')

plt.title('Popularity Distribution ')
plt.legend() 
plt.savefig('wikipedia.png')	"""


#errorlist=tested[tested['prediction']!=tested['truth']]


"""f = {'Movie_names':list,'audience_average':[list,'mean'],\
'critic_average':[list,'mean']}
DMA_group=DM_list.groupby('director').agg(f)
DMA_group=DMA_group.add_suffix('_Count').reset_index()

X=DMA_group.drop('director', axis=1).values
y=DMA_group['director'].values
#Data get (without the years)

X_train, X_test, y_train, y_test=train_test_split(X,y)
model=make_pipeline(StandardScaler(),SVC(kernel='linear',C=1e-3))
model.fit(X_train,y_train)
tested= pd.DataFrame()
tested['truth']=y_test
tested['prediction']=model.predict(X_test)
errorlist=tested[tested['prediction']!=tested['truth']]
errorlist.show()
"""


#re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")
"""def get_wins(data):
#re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")
	Num_win=re.compile(r"\d+\'win'")
	Num_won=re.compile(r"'won'\d+")
	win_result=Num_win.match(data)
	won_result=Num_won.match(data)
	if win_result[0]==True:
		if won_result[0]==True:
			return w1n_result
		else:
			return win_result
	if won_result[0]==True:
		return won_result
	return result
re.match()"""

#DA_group['avg']=DA_group['audience_average'].apply(get_mean)
#DA_group=DA_group.mean()
#DM_list=DM_list.groupby('director')[['Movie_names'],['audience_average'],['critic_average']].apply(list)

#print(DMA_group['awards'])


#def start_ML(df):

"""	if directors==null:
		return "we don't know"
	else :
		pass
		#check the location rate on web"""











"""