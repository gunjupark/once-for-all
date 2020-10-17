import json
import os
import numpy as np



file_list = os.listdir("sm_tables")
file_list.remove('file_subnet_dict.json')


file_subnet_dict = json.load(open("sm_tables/file_subnet_dict.json"))


for file_name in file_list :
    sm_data = np.load("sm_tables/"+file_name)
    print(sm_data)

    file_key = os.path.splitext(file_name)[0]
    print(file_key)

    subnet_dict = file_subnet_dict[file_key]
    print(subnet_dict)

    subnet_ks = subnet_dict['ks']
    subnet_d = subnet_dict['d']
    subnet_e = subnet_dict['e']

    print(subnet_ks)
    print(subnet_d)
    print(subnet_e)



