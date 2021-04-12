import os
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
# https://gist.github.com/pankajti/e631e8f6ce067fc76dfacedd9e4923ca#file-surprise_knn_recommendation-ipynb
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import KNNBasic,  KNNWithMeans, KNNBaseline
from surprise.model_selection import KFold
from surprise.model_selection.split import ShuffleSplit
from surprise.trainset import Trainset
from surprise import Reader
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise import similarities
# import seaborn as sns
from surprise.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from surprise import NMF
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from caching import cached
from sklearn.decomposition import NMF as NMF_matrix
from surprise.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
# from sklearn.decomposition import NMF




def get_ml_filenames():
	mldir = "input/ml-latest-small"
	return [f"{mldir}/{fname}" for fname in os.listdir(mldir) if fname.endswith('.csv')]

# the 'movieId' column does not consist of consecutive integers starting at 0. This makes it so.
# the 'userId' column does not consist of consecutive integers starting at 0. This makes it so.
# a movieId and userId now corresponds to the appropriate column/row in the ratings matrix
def fix_indices(data):
	movies = data['movies'].copy()
	ratings = data['ratings'].copy()
	ratings.drop_duplicates(subset = 'movieId',inplace=True)
	movies = movies.merge(ratings,on='movieId',how='left',indicator=True)
	movies = movies.loc[movies['_merge'] == 'both']
	movies['new_index']= range(len(movies))
	for k,df in data.items():
		if 'movieId' in df:
			# df = df.loc[df.movieId.isin(movies.movieId)]
			df.drop(df[~df.movieId.isin(movies.movieId)].index,inplace=True)
			df['movieId'] = pd.merge(df,movies,how='left',on='movieId')['new_index'].values
		if 'userId' in df:
			df['userId'] -= 1
		data[k] = df
	return data

# returns {"links" : <links dataframe>, ..., "tags" : <tags dataframe>}
def get_ml_data():
	return fix_indices({ fname.split('.')[0].rsplit("/",1)[-1] : pd.read_csv(fname) for fname in get_ml_filenames()})

def get_movies_ratings():
	data = get_ml_data()
	return data['movies'],data['ratings']

def get_ratings_df():
	data = get_ml_data()
	return data['ratings']

#requires that all the movies considered are in the ratings df and have labels in 0,....,nMovies
def get_ratings_matrix(ratings=None):
	if ratings is None:
		ratings = get_ratings_df()
	nMovies = len(set(ratings.movieId))
	nUsers = len(set(ratings.userId))
	ratings_mat = dok_matrix((nUsers,nMovies))
	for userId, movieId, rating in ratings[['userId','movieId','rating']].values:
		ratings_mat[userId,movieId] = rating #subtract 1 because the indices start at 1
	return ratings_mat


def get_accuracies(predictions):
	return accuracy.rmse(predictions,verbose=False) , accuracy.mae(predictions,verbose=False)


@cached
def get_knn_performance_vs_k_over_k_folds():
	movies, ratings = get_movies_ratings()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	# anti_set = data.build_full_trainset().build_anti_testset()
	kvals = list(range(2,101,2))
	kf = KFold(n_splits=10)
	folds = list(kf.split(data))
	performance = np.array([
		np.mean([
			get_accuracies(KNNBasic(k=k_nn,sim_options={'name': 'pearson'},verbose=False).fit(trainset).test(testset)) for trainset,testset in folds
		],axis=0)
		for k_nn in kvals])
	rmse,mae = performance[:,0],performance[:,1]
	return kvals,rmse,mae

@cached
def get_NNMF_performance_vs_k_over_10_folds():
	_, ratings = get_movies_ratings()
	reader = Reader(rating_scale = (0.5, 5))
	data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']],reader)
	kvals = list(range(2,52,2))
	kf = KFold(n_splits=10)
	folds = list(kf.split(data))
	performance = np.array([
		np.mean([
			get_accuracies(NMF(n_factors=k).fit(trainset).test(testset)) for trainset,testset in folds
		],axis=0)
		for k in kvals])
	rmse,mae = performance[:,0],performance[:,1]
	return kvals,rmse,mae

@cached
def get_MF_performance_vs_k_over_10_folds():
	_, ratings = get_movies_ratings()
	reader = Reader(rating_scale = (0.5, 5))
	data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']],reader)
	kvals = list(range(2,52,2))
	kf = KFold(n_splits=10)
	folds = list(kf.split(data))
	performance = np.array([
		np.mean([
			get_accuracies(SVD(n_factors=k).fit(trainset).test(testset)) for trainset,testset in folds
		],axis=0)
		for k in kvals])
	rmse,mae = performance[:,0],performance[:,1]
	return kvals,rmse,mae



def apply_threshold(value,threshold):
	return 0 if value < threshold else 1


@cached
def get_knn_performance_fixed_k_different_thresholds(best_k=20,thresholds=(2.5,3,3.5,4)):
	ratings_src = get_ratings_df()
	reader = Reader(rating_scale = (0.5,5))
	ground_truth_for_thresholds = {threshold : None for threshold in thresholds}
	predictions_for_thresholds = {threshold : None for threshold in thresholds}
	# threshold=thresholds[0]
	for threshold in thresholds:
		ratings = ratings_src.copy()
		# ratings.rating = (ratings.rating > threshold).astype(int)
		data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
		shuf = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
		trainset, testset = next(shuf.split(data))
		classifier = KNNBasic(k=best_k,sim_options={'name': 'pearson'},verbose=False).fit(trainset)
		predictions = classifier.test(testset)
		predictions_for_thresholds[threshold]  = [prediction.est for prediction in predictions]
		ground_truth_for_thresholds[threshold] = [apply_threshold(prediction.r_ui,threshold) for prediction in predictions]
	return ground_truth_for_thresholds, predictions_for_thresholds

@cached
def get_NNMF_performance_fixed_k_different_thresholds(best_k=20,thresholds=(2.5,3,3.5,4)):
	ratings_src = get_ratings_df()
	reader = Reader(rating_scale = (0.5,5))
	ground_truth_for_thresholds = {threshold : None for threshold in thresholds}
	predictions_for_thresholds = {threshold : None for threshold in thresholds}
	for threshold in thresholds:
		ratings = ratings_src.copy()
		# ratings.rating = (ratings.rating > threshold).astype(int)
		data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
		shuf = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
		trainset, testset = next(shuf.split(data))
		classifier = NMF(n_factors=best_k).fit(trainset)
		predictions = classifier.test(testset)
		predictions_for_thresholds[threshold]  = [prediction.est for prediction in predictions]
		ground_truth_for_thresholds[threshold] = [apply_threshold(prediction.r_ui,threshold) for prediction in predictions]
	return ground_truth_for_thresholds, predictions_for_thresholds


def apply_threshold_to_values(values, threshold):
	return [apply_threshold(value,threshold) for value in values]


@cached
def get_MF_performance_fixed_k_different_thresholds(best_k=20,thresholds=(2.5,3,3.5,4)):
	ratings_src = get_ratings_df()
	reader = Reader(rating_scale = (0.5,5))
	ground_truth_for_thresholds = {threshold : None for threshold in thresholds}
	predictions_for_thresholds = {threshold : None for threshold in thresholds}
	for threshold in thresholds:
		ratings = ratings_src.copy()
		# ratings.rating = (ratings.rating > threshold).astype(int)
		data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
		shuf = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
		trainset, testset = next(shuf.split(data))
		classifier = SVD(n_factors=best_k).fit(trainset)
		predictions = classifier.test(testset)
		predictions_for_thresholds[threshold]  = [prediction.est for prediction in predictions]
		ground_truth_for_thresholds[threshold] = [apply_threshold(prediction.r_ui,threshold) for prediction in predictions]
	return ground_truth_for_thresholds, predictions_for_thresholds


@cached
def get_knn_performance_vs_k_over_k_folds_trimmed():
	trimmers = (get_popular_movieIds, get_unpopular_movieIds, get_high_variance_movieIds)
	ratings = get_ratings_df()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	# anti_set = data.build_full_trainset().build_anti_testset()
	kvals = list(range(2,101,2))
	kf = KFold(n_splits=10)
	folds = list(kf.split(data))
	trimmed_folds = [ (trainset, {trimmer: trim_testset(trimmer,testset) for trimmer in trimmers}) for (trainset,testset) in folds ]
	print("Got trimmed folds.")
	avg_rmses = {trimmer : [] for trimmer in trimmers}
	avg_maes = {trimmer : [] for trimmer in trimmers}
	for k_nn in kvals:
		print(f"k={k_nn}")
		rmses = {trimmer : [] for trimmer in trimmers}
		maes = {trimmer : [] for trimmer in trimmers}
		for trainset,trimmed_testsets in trimmed_folds:
			classifier = KNNBasic(k=k_nn,sim_options={'name': 'pearson'},verbose=False).fit(trainset)
			for trimmer,testset in trimmed_testsets.items():
				rmse, mae = get_accuracies(classifier.test(testset))
				rmses[trimmer].append(rmse)
				maes[trimmer].append(mae)
		for trimmer in trimmers:
			avg_rmses[trimmer].append(np.mean(rmses[trimmer]))
			avg_maes[trimmer].append(np.mean(maes[trimmer]))
	return kvals,avg_rmses,avg_maes

@cached
def get_NNMF_performance_vs_k_over_k_folds_trimmed():
	trimmers = (get_popular_movieIds, get_unpopular_movieIds, get_high_variance_movieIds)
	ratings = get_ratings_df()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	kvals = list(range(2,52,2))
	kf = KFold(n_splits=10)
	folds = list(kf.split(data))
	trimmed_folds = [ (trainset, {trimmer: trim_testset(trimmer,testset) for trimmer in trimmers}) for (trainset,testset) in folds ]
	print("Got trimmed folds.")
	avg_rmses = {trimmer : [] for trimmer in trimmers}
	avg_maes = {trimmer : [] for trimmer in trimmers}
	for k in kvals:
		print(f"k={k}")
		rmses = {trimmer : [] for trimmer in trimmers}
		maes = {trimmer : [] for trimmer in trimmers}
		for trainset,trimmed_testsets in trimmed_folds:
			classifier = NMF(n_factors=k).fit(trainset)
			for trimmer, testset in trimmed_testsets.items():
				rmse, mae = get_accuracies(classifier.test(testset))
				rmses[trimmer].append(rmse)
				maes[trimmer].append(mae)
		for trimmer in trimmers:
			avg_rmses[trimmer].append(np.mean(rmses[trimmer]))
			avg_maes[trimmer].append(np.mean(maes[trimmer]))
	return kvals,avg_rmses,avg_maes
		
@cached
def get_MF_performance_vs_k_over_k_folds_trimmed():
	trimmers = (get_popular_movieIds, get_unpopular_movieIds, get_high_variance_movieIds)
	ratings = get_ratings_df()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	kvals = list(range(2,52,2))
	kf = KFold(n_splits=10)
	folds = list(kf.split(data))
	trimmed_folds = [ (trainset, {trimmer: trim_testset(trimmer,testset) for trimmer in trimmers}) for (trainset,testset) in folds ]
	print("Got trimmed folds.")
	avg_rmses = {trimmer : [] for trimmer in trimmers}
	avg_maes = {trimmer : [] for trimmer in trimmers}
	for k in kvals:
		print(f"k={k}")
		rmses = {trimmer : [] for trimmer in trimmers}
		maes = {trimmer : [] for trimmer in trimmers}
		for trainset,trimmed_testsets in trimmed_folds:
			classifier = SVD(n_factors=k).fit(trainset)
			for trimmer, testset in trimmed_testsets.items():
				rmse, mae = get_accuracies(classifier.test(testset))
				rmses[trimmer].append(rmse)
				maes[trimmer].append(mae)
		for trimmer in trimmers:
			avg_rmses[trimmer].append(np.mean(rmses[trimmer]))
			avg_maes[trimmer].append(np.mean(maes[trimmer]))
	return kvals,avg_rmses,avg_maes


def trim_testset(trimmer,testset):
	desired_ids = trimmer()
	return [tup for tup in testset if tup[1] in desired_ids]

def get_popular_movieIds():
	ratings_mat = get_ratings_matrix()
	return (np.sum(ratings_mat,axis=0) > 2).nonzero()[1]

def get_unpopular_movieIds():
	ratings_mat = get_ratings_matrix()
	return (np.sum(ratings_mat,axis=0) <= 2).nonzero()[1]

def get_high_variance_movieIds():
	ratings_mat = get_ratings_matrix()
	at_least_5_ratings = (np.sum(ratings_mat,axis=0) >= 5).nonzero()[1]
	# at_least_one_5 = (ratings_mat.tocsc().max(axis=0).toarray()==5).nonzero()[1].tolist()
	# col_means = np.sum(ratings_mat,axis=0)/(np.sum(ratings_mat!=0,axis=0))
	col_variances = np.array([np.var(list(ratings_mat[:,j].values())) for j in range(ratings_mat.shape[1])])
	high_variance = (col_variances >= 2).nonzero()[0].tolist()
	return sorted(list(set(at_least_5_ratings).intersection(set(high_variance))))


def get_Naive_performance_10_folds():
    _, ratings = get_movies_ratings()
    reader = Reader(rating_scale = (0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']],reader)
    kf = KFold(n_splits=10)
    folds = list(kf.split(data))
    bsl_options = {'reg_u': 0, 'reg_i': 0}
    performance = np.array([
        np.mean([
            accuracy.rmse(BaselineOnly(bsl_options=bsl_options).fit(trainset).test(testset)) for trainset,testset in folds])])
    rmse = performance
    return rmse

def get_Naive_performance_10_folds_trimmed():
    trimmers = (get_popular_movieIds, get_unpopular_movieIds, get_high_variance_movieIds)
    ratings = get_ratings_df()
    reader = Reader(rating_scale = (0.5,5))
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
    kf = KFold(n_splits=10)
    folds = list(kf.split(data))
    trimmed_folds = [ (trainset, {trimmer: trim_testset(trimmer,testset) for trimmer in trimmers}) for (trainset,testset) in folds ]
    print("Got trimmed folds.")
    bsl_options = {'reg_u': 0, 'reg_i': 0}
    avg_rmses = {trimmer : [] for trimmer in trimmers}

    rmses = {trimmer : [] for trimmer in trimmers}
    for trainset,trimmed_testsets in trimmed_folds:
        classifier = BaselineOnly(bsl_options=bsl_options).fit(trainset)
        for trimmer, testset in trimmed_testsets.items():
            rmse = accuracy.rmse(classifier.test(testset))
            rmses[trimmer].append(rmse)
    for trimmer in trimmers:
        avg_rmses[trimmer].append(np.mean(rmses[trimmer]))
    return avg_rmses


def q23_helper():
	reader = Reader(rating_scale = (0.5, 5))
	ratings = get_ratings_df()
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	train_data = data.build_full_trainset()
	model = NMF(n_factors=20)
	model.fit(train_data)
	V = model.qi
	movies_df = pd.read_csv('input/ml-latest-small/movies.csv', names=['movieid','title','genres'])
	for i in [0,2,4,6,12,14,16,18]:
		print(f"Column {i}")
		top_10 = np.argsort(V[:,i])[-11:-1]
		for j in top_10:
			print(movies_df['genres'][j])

	
# def shuffled_inplace(arr):
# 	np.random.shuffle(arr)
# 	return arr

# def get_knn_predictions(k=5):
# 	movies, ratings = get_movies_ratings()
# 	ratings_mat = get_ratings_matrix()
# 	nUsers, nMovies = ratings_mat.shape
# 	pearson = get_pearson_matrix(ratings_mat)
# 	mu = np.sum(ratings_mat,axis=1)/(np.sum(ratings_mat!=0,axis=1))
# 	row_threshold = np.sort(pearson,axis=1)[:,-k]
# 	knn_mask = np.array([[el >= row_threshold[i] for el in row] for i,row in enumerate(pearson)]) #note - this may have MORE THAN k nearest neighbors if there are ties, but that shouldn't affect the prediction function AT ALL.
# 	for row in knn_mask:
# 		row[shuffled_inplace(row.nonzero()[0])[k:]]=False #zero out all but k nearest neighbors

# 	# knn_mask = pearson >= row_threshold
# 	pearson_knn_only = np.multiply(pearson,knn_mask)

	# predictions = mu +
# def get_pearson_matrix(ratings_mat=None):
# 	if ratings_mat is None:
# 		ratings_mat = get_ratings_matrix()
# 	mu = np.sum(ratings_mat,axis=1)/(np.sum(ratings_mat!=0,axis=1))
# 	centered = (ratings_mat !=0).multiply(ratings_mat - mu ).tocsc() #subtract mean of each row's nonzero entries, store as csc for matrix multiplication
# 	scales = np.sqrt((centered !=0) @ centered.T.power(2))
# 	scales = scales.multiply(scales.T)
# 	scales.data = np.reciprocal(scales.data)
# 	pearson = ((centered @ centered.T).multiply( scales  )).toarray()
# 	return pearson

# This function is extremely slow. It checks that the pearson
# matrix from get_pearson_matrix is correct. Calculating with
# Python loops is too slow and the matrix operations in
# get_pearson_matrix (which are executed by Numpy in C) should
# be used.
# def check_pearson_matrix(tol=0.05):
# 	movies, ratings = get_movies_ratings()
# 	ratings_mat = get_ratings_matrix()
# 	pearson_fast = get_pearson_matrix(ratings_mat)
# 	print("Got fast pearson matrix.")
# 	nUsers, nMovies = ratings_mat.shape
# 	userMeans = np.sum(ratings_mat,axis=1)/(np.sum(ratings_mat!=0,axis=1))
# 	pearson = np.zeros((nUsers,nUsers))
# 	for u in range(nUsers):
# 		if u %10 == 0:
# 			print(f"{u=}")
# 		uvSums = np.zeros(nUsers)
# 		uSums = np.zeros(nUsers)
# 		vSums = np.zeros(nUsers)
# 		for (_,k), ruk in ratings_mat[u,:].items():
# 			for (v,_), rvk in ratings_mat[:,k].items():
# 				uvSums[v] += (ruk - userMeans[u])*(rvk-userMeans[v])
# 				uSums[v] += (ruk-userMeans[u])**2
# 				vSums[v] += (rvk-userMeans[v])**2
# 		pearson[u,:] = uvSums / np.multiply(np.sqrt(uSums),np.sqrt(vSums))
# 		if any(abs(pearson[u,:] - pearson_fast[u,:]) > tol):
# 			wrong = np.array(abs(pearson[u,:] - pearson_fast[u,:]) > tol).nonzero()[0].tolist()
# 			print(f"{len(wrong)} elements of pearson matrix in row {u} are incorrect: {wrong}")
# 			for ind in wrong:
# 				print(f"pearson fast : {pearson_fast[u,ind]} ≠ {pearson[u,ind]} : pearson")
# 	return pearson

##
## FROM ranking.py
##
#return a list contain sorted rating and corresponding
#movie ID(incluidng movie without groundtruth rating) for each User
def sorted_rating():
    kf = KFold(n_splits=10)
    algo = KNNBasic(sim_options={'name': 'pearson'},verbose=False)
    movies, ratings = get_movies_ratings()
    reader = Reader(rating_scale = (0.5,5))
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
    total_prediction=[]
    for trainset, testset in kf.split(data):
        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)
        total_prediction.append(predictions)
    total_prediction=pd.DataFrame(np.concatenate(total_prediction, axis=0))
    #set of those user and item pairs for which a rating doesn't exist in original dataset.
    anti_set = data.build_full_trainset().build_anti_testset()
    q=algo.test(anti_set)
    t=np.asarray(total_prediction)
    new=np.concatenate((t,q))
    movies, ratings = get_movies_ratings()
    arrayUserID = new[:,0]
    arrayMovieID = new[:,1]
    arrayRating = new[:,3]
    sortedU = arrayUserID[np.argsort(arrayUserID)]
    sortedM = arrayMovieID[np.argsort(arrayUserID)]
    sortedR = arrayRating[np.argsort(arrayUserID)]
    u, count = np.unique(sortedU, return_counts=True)
    index = 0
    user_rating = []
    user_movie = []
    for i in range(len(u)):
        user_rating.append(sortedR[index:index+count[i]])
        index += count[i]
    index = 0
    for i in range(len(u)):
        user_movie.append(sortedM[index:index+count[i]])
        index += count[i]
    sorted_user_rating=[]
    sorted_user_movie=[]
    for i in range(len(user_rating)):
        sorted_user_rating.append(user_rating[i][np.argsort(-user_rating[i])])
        sorted_user_movie.append(user_movie[i][np.argsort(-user_rating[i])])
    return sorted_user_rating, sorted_user_movie

#return a list contain sorted rating and corresponding
#movie ID(with groundtruth rating) for each User
def sorted_groundtruth():
	movies, ratings = get_movies_ratings()
	ratings[['userId','movieId','rating']]
	arrayUserID = np.asarray(ratings[['userId']])[:,0]
	arrayMovieID =  np.asarray(ratings[['movieId']])[:,0]
	arrayRating =  np.asarray(ratings[['rating']])[:,0]
	sortedU = arrayUserID[np.argsort(arrayUserID)]
	sortedM = arrayMovieID[np.argsort(arrayUserID)]
	sortedR = arrayRating[np.argsort(arrayUserID)]
	u, count = np.unique(sortedU, return_counts=True)
	index = 0
	user_rating = []
	user_movie = []
	for i in range(len(u)):
		user_rating.append(sortedR[index:index+count[i]])
		index += count[i]
	index = 0
	for i in range(len(u)):
		user_movie.append(sortedM[index:index+count[i]])
		index += count[i]
	truth_sorted_user_rating=[]
	truth_sorted_user_movie=[]
	for i in range(len(user_rating)):
		u=user_rating[i][np.argsort(-user_rating[i])]
		t=user_movie[i][np.argsort(-user_rating[i])]
		truth_sorted_user_rating.append(u[u>3])
		truth_sorted_user_movie.append(t[u>3])
	return truth_sorted_user_rating, truth_sorted_user_movie

def num_intersection(lst1, lst2):
    return len(list(set(lst1) & set(lst2)))

# def user_precision_recall():
#     sorted_user_rating, sorted_user_movie = sorted_rating()
#     truth_sorted_user_rating, truth__sorted_user_movie = sorted_groundtruth()
#     empties=[]
#     for s in truth__sorted_user_movie:
#         if len(s) == 0:
#             empties.append(truth__sorted_user_movie.index(s))
#     truth__sorted_user_movie.pop(empties[0])
#     sorted_user_movie.pop(empties[0])
#     precision=[]
#     recall=[]
#     t_list=[]
#     t=20
#     for i in range(len (truth__sorted_user_movie)):
#         tru=truth__sorted_user_movie[i]
#         s=sorted_user_movie[i][:t]
#         num=num_intersection(tru,s)
#         precision.append(num/t)
#         recall.append(num/len(truth__sorted_user_movie[i]))
#     return precision, recall


def user_precision_recall(predictions, t_value):
	record = defaultdict(list)
	for uid, _, r_actual, r_est, _ in predictions:
		record[uid].append((r_actual, r_est))
	
	precision_list = []
	recall_list = []

	for uid, ratings in record.items():
		ratings.sort(key=lambda x: x[1], reverse=True)
		G_size = 0
		S_t_size = 0
		G_and_S_t = 0
		for i, _ in ratings:
			if i >= 3:
				G_size += 1
		if G_size != 0:
			if len(ratings) >= t_value:
				for i, j in ratings[:t_value]:
					
					if j >= 3:
						S_t_size += 1
					if i >= 3 and j >=3:
						G_and_S_t += 1
				if S_t_size != 0:
					precision_list.append(G_and_S_t / S_t_size)
				else:
					precision_list.append(1)
				recall_list.append(G_and_S_t / G_size)
	return sum(precision_list)/len(precision_list), sum(recall_list)/len(recall_list)


	
def q36_helper():
	_, ratings = get_movies_ratings()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	kf = KFold(n_splits=10)
	precision = []
	recall = [] 
	for t_value in range(1, 26):
		precision_t = []
		recall_t = []
		for trainset, testset in kf.split(data):
			model = KNNBasic(k=20,sim_options={'name': 'pearson'},verbose=False)
			model.fit(trainset)
			predictions = model.test(testset)
			user_precision_recall(predictions, t_value)
			precision_iter, recall_iter = user_precision_recall(predictions, t_value)
			precision_t.append(precision_iter)
			recall_t.append(recall_iter)
		precision.append(sum(precision_t)/len(precision_t))
		recall.append(sum(recall_t)/len(recall_t))
	
	plt.plot(list(range(1, 26)), precision)
	plt.xlabel('t')
	plt.ylabel('Average Precision')
	plt.title('Average Precision vs. T for KNN')
	plt.show()
	
	plt.plot(list(range(1, 26)), recall)
	plt.xlabel('t')
	plt.ylabel('Average Recall')
	plt.title('Average Recall vs. T for KNN')
	plt.show()

	plt.plot(recall, precision)
	plt.xlabel('Average Recall')
	plt.ylabel('Average Precision')
	plt.title('Average Recall vs. Average Recall for KNN')
	plt.show()

def q37_helper():
	_, ratings = get_movies_ratings()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	kf = KFold(n_splits=10)
	precision = []
	recall = [] 
	for t_value in range(1, 26):
		precision_t = []
		recall_t = []
		for trainset, testset in kf.split(data):
			model = NMF(n_factors=20)
			model.fit(trainset)
			predictions = model.test(testset)
			user_precision_recall(predictions, t_value)
			precision_iter, recall_iter = user_precision_recall(predictions, t_value)
			precision_t.append(precision_iter)
			recall_t.append(recall_iter)
		precision.append(sum(precision_t)/len(precision_t))
		recall.append(sum(recall_t)/len(recall_t))
	
	plt.plot(list(range(1, 26)), precision)
	plt.xlabel('t')
	plt.ylabel('Average Precision')
	plt.title('Average Precision vs. T for NMF')
	plt.show()
	
	plt.plot(list(range(1, 26)), recall)
	plt.xlabel('t')
	plt.ylabel('Average Recall')
	plt.title('Average Recall vs. T for NMF')
	plt.show()

	plt.plot(recall, precision)
	plt.xlabel('Average Recall')
	plt.ylabel('Average Precision')
	plt.title('Average Recall vs. Average Recall for NMF')
	plt.show()

def q38_helper():
	_, ratings = get_movies_ratings()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	kf = KFold(n_splits=10)
	precision = []
	recall = [] 
	for t_value in range(1, 26):
		precision_t = []
		recall_t = []
		for trainset, testset in kf.split(data):
			model = SVD(n_factors=20)
			model.fit(trainset)
			predictions = model.test(testset)
			user_precision_recall(predictions, t_value)
			precision_iter, recall_iter = user_precision_recall(predictions, t_value)
			precision_t.append(precision_iter)
			recall_t.append(recall_iter)
		precision.append(sum(precision_t)/len(precision_t))
		recall.append(sum(recall_t)/len(recall_t))
	
	plt.plot(list(range(1, 26)), precision)
	plt.xlabel('t')
	plt.ylabel('Average Precision')
	plt.title('Average Precision vs. T for MF')
	plt.show()
	
	plt.plot(list(range(1, 26)), recall)
	plt.xlabel('t')
	plt.ylabel('Average Recall')
	plt.title('Average Recall vs. T for MF')
	plt.show()

	plt.plot(recall, precision)
	plt.xlabel('Average Recall')
	plt.ylabel('Average Precision')
	plt.title('Average Recall vs. Average Recall for MF')
	plt.show()

def q39_helper():
	_, ratings = get_movies_ratings()
	reader = Reader(rating_scale = (0.5,5))
	data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
	kf = KFold(n_splits=10)
	precision_Knn = []
	recall_Knn = []
	precision_NNMF = []
	recall_NNMF = [] 
	precision_MF= []
	recall_MF = []  
	for t_value in range(1, 26):
		precision_t_Knn = []
		recall_t_Knn= []
		precision_t_NNMF = []
		recall_t_NNMF = []
		precision_t_MF = []
		recall_t_MF = []
		for trainset, testset in kf.split(data):
			MF = SVD(n_factors=20)
			NNMF = NMF(n_factors=20)
			Knn  = KNNBasic(k=20,sim_options={'name': 'pearson'},verbose=False)
			MF.fit(trainset)
			NNMF.fit(trainset)
			Knn.fit(trainset)
			predictions_Knn = Knn.test(testset)
			predictions_MF = MF.test(testset)
			predictions_NNMF = NNMF.test(testset)
			precision_iter_Knn, recall_iter_Knn = user_precision_recall(predictions_Knn, t_value)
			precision_iter_NNMF, recall_iter_NNMF = user_precision_recall(predictions_NNMF, t_value)
			precision_iter_MF, recall_iter_MF = user_precision_recall(predictions_MF, t_value)
			precision_t_Knn.append(precision_iter_Knn)
			recall_t_Knn.append(recall_iter_Knn)
			precision_t_NNMF.append(precision_iter_NNMF)
			recall_t_NNMF.append(recall_iter_NNMF)
			precision_t_MF.append(precision_iter_MF)
			recall_t_MF.append(recall_iter_MF)
		precision_Knn.append(sum(precision_t_Knn)/len(precision_t_Knn))
		recall_Knn.append(sum(recall_t_Knn)/len(recall_t_Knn))
		precision_NNMF.append(sum(precision_t_NNMF)/len(precision_t_NNMF))
		recall_NNMF.append(sum(recall_t_NNMF)/len(recall_t_NNMF))
		precision_MF.append(sum(precision_t_MF)/len(precision_t_MF))
		recall_MF.append(sum(recall_t_MF)/len(recall_t_MF))


	plt.plot(recall_Knn, precision_Knn,label='Knn')
	plt.plot(recall_NNMF, precision_NNMF,label='NNMF')
	plt.plot(recall_MF, precision_MF,label='MF')
	plt.xlabel('Average Recall')
	plt.ylabel('Average Precision')
	plt.title('Average precision vs. Average Recall for Knn, NNMF,MF')
	plt.legend()
	plt.show()
	

def q34_helper():
    
    best_k = 20 # value found in question 11, where the RMSE/MAE curves level out
    thresholds = 3
    threshold = 3
    ratings_src = get_ratings_df()
    reader = Reader(rating_scale = (0.5,5))
    ground_truth_for_thresholds1 = {thresholds}
    predictions_for_thresholds1 = {thresholds}
    ratings = ratings_src.copy()
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
    shuf = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
    trainset, testset = next(shuf.split(data))
    classifier = KNNBasic(k=best_k,sim_options={'name': 'pearson'},verbose=False).fit(trainset)
    predictions = classifier.test(testset)
    predictions_for_thresholds1  = [prediction.est for prediction in predictions]
    ground_truth_for_thresholds1 = [apply_threshold(prediction.r_ui,threshold) for prediction in predictions]

    ratings_src = get_ratings_df()
    reader = Reader(rating_scale = (0.5,5))
    ground_truth_for_thresholds2 = {thresholds}
    predictions_for_thresholds2 = {thresholds}
    ratings = ratings_src.copy()
    # ratings.rating = (ratings.rating > threshold).astype(int)
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
    shuf = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
    trainset, testset = next(shuf.split(data))
    classifier = NMF(n_factors=best_k).fit(trainset)
    predictions = classifier.test(testset)
    predictions_for_thresholds2  = [prediction.est for prediction in predictions]
    ground_truth_for_thresholds2 = [apply_threshold(prediction.r_ui,threshold) for prediction in predictions]

    ratings_src = get_ratings_df()
    reader = Reader(rating_scale = (0.5,5))
    ground_truth_for_thresholds3 = {thresholds}
    predictions_for_thresholds3 = {thresholds}
    ratings = ratings_src.copy()
    # ratings.rating = (ratings.rating > threshold).astype(int)
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
    shuf = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
    trainset, testset = next(shuf.split(data))
    classifier = SVD(n_factors=best_k).fit(trainset)
    predictions = classifier.test(testset)
    predictions_for_thresholds3 = [prediction.est for prediction in predictions] 
    ground_truth_for_thresholds3 = [apply_threshold(prediction.r_ui,threshold) for prediction in predictions]

    ratings_src = get_ratings_df()
    reader = Reader(rating_scale = (0.5,5))
    ground_truth_for_thresholds4 = {thresholds}
    predictions_for_thresholds4 = {thresholds}
    bsl_options = {'reg_u': 0, 'reg_i': 0}
    ratings = ratings_src.copy()
    # ratings.rating = (ratings.rating > threshold).astype(int)
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
    shuf = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
    trainset, testset = next(shuf.split(data))
    classifier = BaselineOnly(bsl_options=bsl_options).fit(trainset)
    predictions = classifier.test(testset)
    predictions_for_thresholds4 = [prediction.est for prediction in predictions]
    ground_truth_for_thresholds4 = [apply_threshold(prediction.r_ui,threshold) for prediction in predictions]

    return predictions_for_thresholds1,predictions_for_thresholds2,predictions_for_thresholds3,predictions_for_thresholds4,ground_truth_for_thresholds1,ground_truth_for_thresholds2,ground_truth_for_thresholds3,ground_truth_for_thresholds4

