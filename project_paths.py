import os

cdir_root = os.getcwd().split('/')
# push test

if cdir_root[1] == 'Users':
    COMPUTER = 'mac'
elif cdir_root[1] == 'home':
    COMPUTER = 'linux'
elif cdir_root[1] == 'nesi':
    COMPUTER = 'nesi'
else:
    COMPUTER = 'linux'

if COMPUTER == 'linux':
    PATH = '/home/zade/masters_2021/'
    # PATH = '/home/zade/GoogleDrive/masters_2021/'
    SCRATCH = PATH
    python_masters_path = '/home/zade/python_masters/'
    from_array_path = PATH + 'templates_athinput/athinput.from_array'
    athena_path = '/home/zade/athena/athena_masters/bin/from_array/athena'
elif COMPUTER == 'mac':
    PATH = '/Users/johza22p/masters_2021/'
    # PATH = '/Users/johza22p/GoogleDrive/masters_2021/'
    SCRATCH = PATH
    python_masters_path = '/Users/johza22p/python_masters/'
    from_array_path = PATH + 'templates_athinput/athinput.from_array'
    athena_path = '/Users/johza22p/athena/athena_masters/bin/from_array/athena'
elif COMPUTER == 'nesi':
    PATH = '/nesi/project/uoo02637/zade/masters_2021/' # project directory
    SCRATCH = '/nesi/nobackup/uoo02637/zade/masters_2021/'
    python_masters_path = '/home/johza721/masters_2021/python_masters/'
    from_array_path = '/home/johza721/masters_2021/templates_athinput/athinput.from_array'
    athena_path = '/home/johza721/masters_2021/athena/bin/athena_maui/athena_nompi'