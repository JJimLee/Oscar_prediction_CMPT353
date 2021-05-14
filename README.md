import sys  

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
#pyspark submit Main1.py wikidata-movies.json.gz omdb-data.json.gz rotten-tomatoes.json.gz genres.json.gz  