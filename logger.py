import sys
import os
from os.path import isfile,isdir
from functools import wraps
from textwrap import dedent

def get_logdir():
	return './logs'

def ensure_directory(directory):
	if not isdir(directory):
		os.mkdir(directory)

class Tee(object):
	def __init__(self, name, mode,dedent=False):
		self.file = open(name, mode)
		self.stdout = sys.stdout
		self.dedent=dedent
	def __enter__(self):
		sys.stdout = self
	def __exit__(self,*args):
		sys.stdout = self.stdout
		self.file.close()
	def write(self, data):
		if self.dedent:
			self.file.write(dedent(data))
		else:
			self.file.write(data)
		self.stdout.write(data)
	def flush(self):
		self.file.flush()


def logged(func):
	logfilename = f"{get_logdir()}/{func.__name__}_output.txt"
	@wraps(func)
	def decorated(*args,**kwargs):
		ensure_directory(get_logdir())
		with Tee(logfilename,'w',dedent=True):
			output = func(*args,**kwargs)
		print(f"Logged to {logfilename}.")
		return output
	return decorated
