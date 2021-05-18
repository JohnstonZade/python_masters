COMPUTER = 'linux'
#  COMPUTER = 'mac'
# COMPUTER = 'nesi'

if COMPUTER == 'linux':
    PATH = '/home/zade/masters_2021/'
    python_masters_path = '/home/zade/python_masters/'
    from_array_path = '/home/zade/masters_2021/templates_athinput/athinput.from_array'
    from_array_reinterp_path = '/home/zade/masters_2021/templates_athinput/athinput.from_array_reinterp'
    athena_path = '/home/zade/masters_2021/athena/bin/from_array/athena'
elif COMPUTER == 'mac':
    PATH = '/Users/johza22p/masters_2021/'
    python_masters_path = '/Users/johza22p/python_masters/'
    from_array_path = '/Users/johza22p/masters_2021/templates_athinput/athinput.from_array'
    from_array_reinterp_path = '/Users/johza22p/masters_2021/templates_athinput/athinput.from_array_reinterp'
    athena_path = '/Users/johza22p/masters_2021/athena/bin/from_array/athena'
elif COMPUTER == 'nesi':
    PATH = '/nesi/project/uoo02637/zade/masters_2021/' # project directory
    python_masters_path = '/home/johza721/masters_2021/python_masters/'
    from_array_path = '/home/johza721/masters_2021/templates_athinput/athinput.from_array'
    from_array_reinterp_path = '/home/johza721/masters_2021/templates_athinput/athinput.from_array_reinterp'
    athena_path = '/home/johza721/masters_2021/athena/bin/athena_mauivis/athena'