import os 
from shutil import copyfile, rmtree
import ntpath
import random
import shutil
import warnings
from glob import glob

def find(pathname):
    glob(pathname, recursive=True) # accept "**" expression

# list all files inside a directory and return as a list.
def laf(root_dir):
    l=list()
    for path, _, files in os.walk(root_dir):
        for name in files:
            l.append(os.path.abspath(os.path.join(path, name)))
    return l

def mv(src,dst):
    if os.path.exists(src):
        # cp(src,dst)
        # rm(src)
        shutil.move(src,dst)
    else:
        warnings.warn('file or folder "%s" does not exist.' % src)

def rm(file_or_dir):
    if os.path.exists(file_or_dir) == False: return
    if os.path.isfile(file_or_dir) == False:
        rmtree(file_or_dir)
    else:
        os.remove(file_or_dir)

def cd(path):
    os.chdir(path)

def cwd():
    return os.getcwd()

def cp(src,dst):
    copyfile(src,dst)

def mkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    return os.path.abspath(path)

def abs_path(path):
    return os.path.abspath(path)

def join_path(*args):
    path = os.path.join(*args)
    return os.path.abspath(path)

def file_exist(path):
    if os.path.exists(path) and os.path.isfile(path): return True
    else: return False

def files_exist(path_list):
    for path in path_list:
        if os.path.exists(path) and os.path.isfile(path):
            continue
        else: return False
    return True

def file_empty(path):
    if os.stat(path).st_size == 0: return True
    else: return False

def dir_exist(path):
    if os.path.exists(path) and os.path.isdir(path): return True
    else: return False

def ls(root_dir, full_path = False):
    if full_path == False:
        return os.listdir(root_dir)
    else:
        l = []
        for item in os.listdir(root_dir):
            l.append(join_path(root_dir, item))
        return l

def lsdir(root_dir, full_path=False):
    dirs = [ item for item in os.listdir(root_dir) if dir_exist(join_path(root_dir, item))]
    if full_path == False:
        return dirs
    else:
        l=[]
        for folder in dirs:
            l.append( join_path(root_dir, folder) )
        return l

def lsfile(root_dir, full_path=False):
    dirs = [ item for item in os.listdir(root_dir) if file_exist(join_path(root_dir, item))]
    if full_path == False:
        return dirs
    else:
        l=[]
        for folder in dirs:
            l.append( join_path(root_dir, folder) )
        return l

# get filename from path
def gfn(path, no_extension = False):
    name = ntpath.basename(os.path.abspath(path))
    if no_extension:
        index = name.find('.')
        name = name[:index]
    return name

# get file directory from path
def gfd(path):
    return os.path.abspath(os.path.dirname(os.path.abspath(path)))

def make_unique_dir(basedir=None):
	while True:
		randstr = ''.join(random.choice('0123456789abcdef') for _ in range(8))
		randstr = '__' + randstr + '__'
		if basedir is not None:
			dirpath = join_path(basedir,randstr)
		else:
			dirpath = abs_path(randstr)
		if dir_exist(dirpath):
			continue
		else:
			mkdir(dirpath)
			return dirpath

