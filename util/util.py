import os

def goToFileDir(it):
	print "Current directory:"
	print(os.getcwd())
	os.chdir(os.path.dirname(os.path.realpath(it)))
	print "Change to:"
	print(os.getcwd())