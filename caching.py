import os
from os.path import isfile,isdir
from glob import glob
import pickle
from functools import wraps
# from hashlib import sha1
from deepdiff import DeepHash
import time
# from copy import copy, deepcopy
# from mmh3 import hash128



printing = False
def log(*args,**kwargs):
	if printing:
		print(*args,**kwargs)


def timed(f):
	@wraps(f)
	def wrapped(*args,**kwargs):
		start=time.time()
		output = f(*args,**kwargs)
		end=time.time()
		log(f"{f.__name__} took {int(end-start)}s")
		return output
	return wrapped

def get_cachedir():
	return './cache'

def ensure_cache():
	cachedir = get_cachedir()
	if not isdir(cachedir):
		os.mkdir(cachedir)

@timed
def load(fname):
	log(f"loading {fname} (already executed)")
	with open(f"{fname}",'rb') as f:
		return pickle.load(f)

@timed
def dump(obj,fname):
	with open(f"{fname}",'wb') as f:
		pickle.dump(obj,f)

def pre_hash(obj):
	# numpy sparse array
	if hasattr(obj,'todok'):
		# return dict(obj.todok())
		canonical = obj.sorted_indices() #copies
		canonical.sum_duplicates()
		return (canonical.shape,tuple(canonical.indices),tuple(canonical.data))
	else:
		return obj

@timed
def get_hash(args,kwargs):
	hashable_args = [pre_hash(arg) for arg in args] #
	hashable_kwargs = {k : pre_hash(v) for k,v in kwargs.items()}
	obj = (hashable_args,hashable_kwargs)
	return DeepHash(obj,ignore_string_case=True,significant_digits=1)[obj]

@timed
def cache_execute(func,file_out_base,args,kwargs,hashed_args):
	log(f"EXECUTING (and caching) {func.__name__}")
	output = func(*args,**kwargs)
	dump(output,f"{file_out_base}{hashed_args}")
	return output

def cached(func):
	file_base = f"{get_cachedir()}/{func.__name__}"
	@wraps(func)
	def decorated(*args,**kwargs):
		ensure_cache()
		hashvals = [filename.split(file_base,1)[-1] for filename in glob(f"{file_base}*")]
		hashed_args = get_hash(args,kwargs)
		if hashed_args in hashvals:
			file_out = f"{file_base}{hashed_args}"
			return load(file_out)
		return cache_execute(func,file_base,args,kwargs,hashed_args)
	return decorated
