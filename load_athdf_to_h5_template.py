import os

cdir_root = os.getcwd().split('/')

if cdir_root[1] in ['Users', 'Volumes']:
    python_masters_path = '/Users/johza22p/python_masters/'
elif cdir_root[1] == 'home':
    python_masters_path = '~/python_masters/'
else:
    python_masters_path = '/home/johza721/masters_2021/python_masters/'

import sys
sys.path.insert(0, python_masters_path)
genics = __import__('generate_ics')

folder = ''  # folder to save h5 file
athinput_in_folder = ''  # path to folder containing original athinput
athinput_in = ''  # name of athinput file
h5name = ''  # name of h5 file to be saved
athdf_input = ''  # full path to athdf input to be read

genics.create_athena_fromh5(folder, athinput_in_folder, athinput_in, h5name, athdf_input)