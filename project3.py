import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pprint import pprint
# from textwrap import dedent
from itertools import product
import json


# from sklearn.metrics import confusion_matrix
# from scipy.optimize import linear_sum_assignment
from plotmat import plot_mat
# import umap

from helpers import get_movies_ratings, get_ratings_matrix,get_knn_performance_vs_k_over_k_folds,get_knn_performance_vs_k_over_k_folds_trimmed,get_knn_performance_fixed_k_different_thresholds,get_NNMF_performance_vs_k_over_10_folds
from helpers import get_MF_performance_vs_k_over_10_folds, get_MF_performance_vs_k_over_k_folds_trimmed, get_MF_performance_fixed_k_different_thresholds
from helpers import get_NNMF_performance_fixed_k_different_thresholds, get_NNMF_performance_vs_k_over_k_folds_trimmed, get_ratings_df,get_Naive_performance_10_folds_trimmed, q23_helper
from helpers import get_Naive_performance_10_folds, get_Naive_performance_10_folds_trimmed, q34_helper
from logger import logged
from sklearn.metrics import roc_curve, auc
from helpers import sorted_rating, sorted_groundtruth, num_intersection, user_precision_recall
from helpers import q36_helper, q37_helper, q38_helper, q39_helper
np.random.seed(0)
random.seed(0)

@logged
def q1():
	movies, ratings = get_movies_ratings()
	ratings_mat = get_ratings_matrix()
	# nMovies = len(set(movies.movieId))
	# nUsers = len(set(ratings.userId))
	nRatings = len(ratings)
	nPosRatings = np.prod(ratings_mat.shape)
	print(
	f"""
	Total possible ratings: {nPosRatings}
	Number Ratings: {nRatings}
	Sparsity: {nRatings/nPosRatings}
	""")

def q2():
    movies, ratings = get_movies_ratings()
    his = np.histogram(ratings.rating,bins=[0.01, 0.5001, 1.0001, 1.5001, 2.0001, 2.5001, 3.0001, 3.5001, 4.0001, 4.5001, 5.0001])
    fig, ax = plt.subplots() 
    plt.bar(his[1][1:],his[0],width=0.5,edgecolor='k')
    ax.set_xticks(his[1][1:])
    ax.set_xticklabels( ('0.5', '1', '1.5', '2', '2.5','3','3.5','4','4.5','5') )
    plt.xlabel("Movie Rating")
    plt.ylabel("Number of Movies with Rating")
    plt.title("Movie Ratings Histogram")
    plt.savefig(f"figures/q2_ratings_histogram.pdf")
    plt.show()

def q3():
	movies, ratings = get_movies_ratings()
	arrayMovieId = ratings.movieId
	u, count = np.unique(arrayMovieId, return_counts=True)
	sortedU = u[np.argsort(-count)]
	sortedCount = count[np.argsort(-count)]
	xIdx = np.arange(sortedU.shape[0])
	plt.title("Movie rating frequency")
	plt.xlabel("Movie Id ranking")
	plt.ylabel("Rating frequency")
	plt.plot(xIdx , sortedCount)
	# plt.show()
	plt.savefig(f"figures/q3_movie_ratings.pdf")

def q4():
	movies, ratings = get_movies_ratings()
	arrayUserId = ratings.userId
	u, count = np.unique(arrayUserId, return_counts=True)
	sortedU = u[np.argsort(-count)]
	sortedCount = count[np.argsort(-count)]
	xIndex = np.arange(sortedU.shape[0])
	plt.title("Distribution of ratings among users")
	plt.xlabel("User index ordered by decreasing frequency")
	plt.ylabel("Rating frequency")
	plt.plot(xIndex , sortedCount)
	# plt.show()
	plt.savefig(f"figures/q4_user_ratings.pdf")


def q6():
	movies, ratings = get_movies_ratings()
	arrayMovieId = ratings.movieId
	arrayRating = ratings.rating
	sortedM = arrayMovieId[np.argsort(arrayMovieId)]
	sortedR = arrayRating[np.argsort(arrayMovieId)]
	u, count = np.unique(sortedM, return_counts=True)
	index = 0
	ratingVar = np.zeros(u.shape)
	for i in range(len(u)):
		ratingVar[i] = np.var(sortedR[index:index+count[i]])
		index += count[i]
	print("Histogram with 0 variance")
	bins = np.arange(0, 6, 0.1)
	plt.hist(ratingVar, bins=bins)
	plt.title("Variance of the rating values received by each movie")
	plt.xlabel("Variance of the ratings")
	plt.ylabel("Number of movies")
	plt.show()
	plt.savefig(f"figures/q6_movie_rating_variance_histogram.pdf")
	plt.clf()
	print("Histogram without 0 variance")
	bins = np.arange(0.0001, 6, 0.1)
	plt.hist(ratingVar, bins=bins)
	plt.title("Variance of the rating values received by each movie without 0 variance")
	plt.xlabel("Variance of the ratings")
	plt.ylabel("Number of movies")
	plt.show()
	plt.savefig(f"figures/q6_movie_rating_variance_histogram_0_variance_removed.pdf")
	print("+ The number of variance 0 is",len(ratingVar)-np.count_nonzero(ratingVar))



def q10():
	kvals,rmse,mae = get_knn_performance_vs_k_over_k_folds()
	plt.plot(kvals , rmse,label="RMSE")
	plt.plot(kvals , mae,label="MAE")
	plt.xlabel("$K$ Nearest Neighbors")
	plt.ylabel("Accuracy (RMSE or MAE)")
	plt.title("Accuracy vs. $K$ in Movielens KNN")
	plt.legend()
	plt.savefig(f"figures/q10_error_vs_k_knn_kfold.pdf")



def q12_13_14():
	kvals,avg_rmses,avg_maes = get_knn_performance_vs_k_over_k_folds_trimmed()
	trimmers = list(avg_rmses.keys())
	nTrimmers = len(trimmers)
	fig,axs = plt.subplots(1,nTrimmers,figsize=(15,5),sharey=True)
	for i, (ax,trimmer) in enumerate(zip(axs,trimmers)):
		min_rmse = min(avg_rmses[trimmer])
		best_k_rmse = kvals[np.argmin(avg_rmses[trimmer])]
		min_mae = min(avg_maes[trimmer])
		best_k_mae = kvals[np.argmin(avg_maes[trimmer])]
		ax.plot(kvals,avg_rmses[trimmer],label=f"RMSE (minimum=${min_rmse :.3}$ at $k={best_k_rmse}$)")
		ax.plot(kvals,avg_maes[trimmer],label=f"MAE (minimum=${min_mae :.3}$ at $k={best_k_mae}$)")
		ax.set_title(f"Accuracy vs. $K$ in Movielens KNN\nWith Trimmer: {trimmer.__name__}")
		ax.set_xlabel("$K$ Nearest Neighbors")
		ax.legend()
		if i == 0:
			ax.set_ylabel("Accuracy (RMSE or MAE)")
	plt.tight_layout()
	plt.savefig(f"figures/q12_error_vs_k_knn_kfold_trimmed.pdf")

def plot_roc(targets, scores,ax,xlabel='False Positive Rate',ylabel='True Positive Rate',title="ROC Curve",nlabels=5):
	# title= f"ROC Curve with Threshold ${threshold}$"
	fprs,tprs,thresh = roc_curve(targets,scores)
	label_inds = [i for i in range(0,len(thresh),len(thresh)//nlabels)]
	if len(thresh)-1 not in label_inds:
		label_inds.pop()
		label_inds.append(len(thresh)-1)
	labelx = fprs[label_inds]
	labely = tprs[label_inds]
	labeltext = thresh[label_inds]
	roc_auc = auc(fprs,tprs)
	ax.plot(fprs, tprs, lw=2, label= f'AUC= {roc_auc : 0.04f}')
	ax.scatter(labelx,labely,marker="x")
	for x,y,lab in zip(labelx,labely,labeltext):
		ax.text(x+0.02,y-0.04,f"{lab:.02}",fontsize=12)
	ax.grid(color='0.7', linestyle='--', linewidth=1)
	# ax.set_xlim([-0.1, 1.1])
	# ax.set_ylim([0.0, 1.05])
	ax.set_xlabel(xlabel,fontsize=15)
	ax.set_ylabel(ylabel,fontsize=15)
	ax.set_title(title)
	ax.legend(loc="lower right",fontsize="x-small")
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(15)

def q15():
	best_k = 20 # value found in question 11, where the RMSE/MAE curves level out
	thresholds = (2.5,3,3.5,4)
	ground_truth_for_thresholds,predictions_for_thresholds = get_knn_performance_fixed_k_different_thresholds(best_k = best_k,thresholds=thresholds)
	fig,axs = plt.subplots(1,len(thresholds),figsize=(15,5),sharey=True)
	# i, (ax,threshold) = next(enumerate(zip(axs,thresholds)))
	for i, (ax,threshold) in enumerate(zip(axs,thresholds)):
		targets,scores = ground_truth_for_thresholds[threshold],predictions_for_thresholds[threshold]
		plot_roc(targets, scores,ax,title= f"ROC Curve with Threshold ${threshold}$")
	plt.savefig(f"figures/q15_thresholded_roc.pdf")

def q17():
	kvals,rmse,mae = get_NNMF_performance_vs_k_over_10_folds()
	plt.plot(kvals , rmse,label="RMSE")
	plt.plot(kvals , mae,label="MAE")
	print(rmse)
	print(mae)
	plt.xlabel("Number of Latent Factors")
	plt.ylabel("Accuracy (RMSE or MAE)")
	plt.title("Accuracy vs. Number of Latent Factors in Movielens NNMF")
	plt.legend()
	plt.savefig(f"figures/q17_error_vs_k_NNMF_kfold.pdf")

def q19_20_21():
	kvals,avg_rmses,avg_maes = get_NNMF_performance_vs_k_over_k_folds_trimmed()
	trimmers = list(avg_rmses.keys())
	nTrimmers = len(trimmers)
	fig,axs = plt.subplots(1,nTrimmers,figsize=(15,5),sharey=True)
	for i, (ax,trimmer) in enumerate(zip(axs,trimmers)):
		min_rmse = min(avg_rmses[trimmer])
		best_k_rmse = kvals[np.argmin(avg_rmses[trimmer])]
		min_mae = min(avg_maes[trimmer])
		best_k_mae = kvals[np.argmin(avg_maes[trimmer])]
		ax.plot(kvals,avg_rmses[trimmer],label=f"RMSE (minimum=${min_rmse :.3}$ at $k={best_k_rmse}$)")
		ax.plot(kvals,avg_maes[trimmer],label=f"MAE (minimum=${min_mae :.3}$ at $k={best_k_mae}$)")
		ax.set_title(f"Accuracy vs. Number of Latent Factors in Movielens NNMF\nWith Trimmer: {trimmer.__name__}")
		ax.set_xlabel("Number of Latent Factors")
		ax.legend()
		if i == 0:
			ax.set_ylabel("Accuracy (RMSE or MAE)")
	plt.tight_layout()
	plt.savefig(f"figures/q19_error_vs_k_knn_kfold_trimmed_1.pdf")

def q22():
	best_k = 20 # value found in question 11, where the RMSE/MAE curves level out
	thresholds = (2.5,3,3.5,4)
	ground_truth_for_thresholds,predictions_for_thresholds = get_NNMF_performance_fixed_k_different_thresholds(best_k = best_k,thresholds=thresholds)
	fig,axs = plt.subplots(1,len(thresholds),figsize=(15,5),sharey=True)
	# i, (ax,threshold) = next(enumerate(zip(axs,thresholds)))
	for i, (ax,threshold) in enumerate(zip(axs,thresholds)):
		targets,scores = ground_truth_for_thresholds[threshold],predictions_for_thresholds[threshold]
		# fprs,tprs,thresh = roc_curve(targets,scores)
		plot_roc(targets, scores,ax,title= f"ROC Curve with Threshold ${threshold}$")
	plt.savefig(f"figures/q22_thresholded_roc.pdf")

@logged
def q23():
	q23_helper()


def q24():
	kvals,rmse,mae = get_MF_performance_vs_k_over_10_folds()
	plt.plot(kvals , rmse,label="RMSE")
	plt.plot(kvals , mae,label="MAE")
	plt.xlabel("Number of Latent Factors")
	plt.ylabel("Accuracy (RMSE or MAE)")
	plt.title("Accuracy vs. Number of Latent Factors in Movielens MF")
	plt.legend()
	plt.savefig(f"figures/q24_error_vs_k_MF_kfold.pdf")

def q26_27_28():
	kvals,avg_rmses,avg_maes = get_MF_performance_vs_k_over_k_folds_trimmed()
	trimmers = list(avg_rmses.keys())
	nTrimmers = len(trimmers)
	fig,axs = plt.subplots(1,nTrimmers,figsize=(15,5),sharey=True)
	for i, (ax,trimmer) in enumerate(zip(axs,trimmers)):
		min_rmse = min(avg_rmses[trimmer])
		best_k_rmse = kvals[np.argmin(avg_rmses[trimmer])]
		min_mae = min(avg_maes[trimmer])
		best_k_mae = kvals[np.argmin(avg_maes[trimmer])]
		ax.plot(kvals,avg_rmses[trimmer],label=f"RMSE (minimum=${min_rmse :.3}$ at $k={best_k_rmse}$)")
		ax.plot(kvals,avg_maes[trimmer],label=f"MAE (minimum=${min_mae :.3}$ at $k={best_k_mae}$)")
		ax.set_title(f"Accuracy vs. $K$ in Movielens MF\nWith Trimmer: {trimmer.__name__}")
		ax.set_xlabel("$K$ Nearest Neighbors")
		ax.legend()
		if i == 0:
			ax.set_ylabel("Accuracy (RMSE or MAE)")
	plt.tight_layout()
	plt.savefig(f"figures/q26_error_vs_k_MF_kfold_trimmed.pdf")

def q29():
	best_k = 20 # value found in question 11, where the RMSE/MAE curves level out
	thresholds = (2.5,3,3.5,4)
	ground_truth_for_thresholds,predictions_for_thresholds = get_MF_performance_fixed_k_different_thresholds(best_k = best_k,thresholds=thresholds)
	fig,axs = plt.subplots(1,len(thresholds),figsize=(15,5),sharey=True)
	# i, (ax,threshold) = next(enumerate(zip(axs,thresholds)))
	for i, (ax,threshold) in enumerate(zip(axs,thresholds)):
		targets,scores = ground_truth_for_thresholds[threshold],predictions_for_thresholds[threshold]
		# fprs,tprs,thresh = roc_curve(targets,scores)
		plot_roc(targets, scores,ax,title= f"ROC Curve with Threshold ${threshold}$")
	plt.savefig(f"figures/q29_thresholded_roc.pdf")

def q30():
	rmse = get_Naive_performance_10_folds()
	print(rmse)

def q31_32_33():
	avg_rmses = get_Naive_performance_10_folds_trimmed()
	trimmers = list(avg_rmses.keys())
	for trimmer in trimmers:
		print(avg_rmses[trimmer])


def q34():
    p1,p2,p3,p4,g1,g2,g3,g4 = q34_helper()
    fig,axs = plt.subplots(1,4,figsize=(15,5),sharey=True)

    targets1,scores1 = g1,p1
    plot_roc(targets1, scores1,axs[0],title= f"KNN ROC Curve")
    targets2,scores2 = g2,p2
    plot_roc(targets2, scores2,axs[1],title= f"NNMF ROC Curve")
    targets3,scores3 = g3,p3
    plot_roc(targets3, scores3,axs[2],title= f"MF ROC Curve")
    targets4,scores4 = g4,p4
    plot_roc(targets4, scores4,axs[3],title= f"Basefilter ROC Curve")

    plt.savefig(f"figures/q34_thresholded_roc_all.pdf")

    

def q36():
	q36_helper()


def q37():
	q37_helper()


def q38():
	q38_helper()


def q39():
	q39_helper()
	pass



# In [4]: ratings_mat
# Out[4]:
# <610x9724 sparse matrix of type '<class 'numpy.float64'>'
# 	with 100836 stored elements in Dictionary Of Keys format>
if __name__=="__main__":
	q1()
	q2()
	q3()
	q4()
	q6()
	q10()
	q12_13_14()
	q15()
	q17()
	q19_20_21()
	q22()
	q23()
	q24()
	q26_27_28()
	q29()
	q30()
	q31_32_33()
	q34()
	q36()
	q37()
	q38() 
	q39()
	pass

