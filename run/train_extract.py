import subprocess
import os
import time

for i in range(10):
    dir1 = "../src/"
    t1=time.time()
    os.chdir(dir1)
    subprocess.run(["python", "train.py", "--src_dir", "../data/Synthetic", "--num_bands", "224", "--end_members", "4", "--batch_size", "256", "--learning_rate", "1e-4", "--epochs", "300", "--patch_size", "5"], check=True)
    t2=time.time()
    t=t2-t1
    print('time:',t)
    os.chdir(dir1)
    subprocess.run(["python", "extract.py",  '-src_dir','../data/Synthetic',  '-num_bands', '224',  '-end_members', '4',  '-batch_size','256', '-patch_size','5', '-image_size','256','-time',str(t)], check=True)
    