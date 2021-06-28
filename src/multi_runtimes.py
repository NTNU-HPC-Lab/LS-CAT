import pandas as pd
from mainfilev3 import main_skeleton
import re
from pathlib import Path
import glob
import os
import subprocess
import numpy as np
import time
from os import path
import ast
import sys
import datetime

def remake_main(path:str,variables:str,function_name:str):
    variables =ast.literal_eval(variables)
    main_body = main_skeleton()
    main_body.add_variables(function_name,variables)
    main_body.change_function(function_name)
    main_body.add_includes(['#include "'+function_name+'.cu"'])
    main_body.save_main(path)

def find_value_for_list(in_var:str,checklist,exact):
    if exact:
        if in_var.lower() in checklist:
            return True
        else:
            return False

    else:
        for l in checklist:
            if in_var.lower().find(l)!=-1:
                return True

        return False

def find_value(in_var:str):
    in_var=in_var.split()[1]
    sizes_c=["size"]
    sizes_e=["n"]
    width_c=["width"]
    width_e=["w","rows","n_a","ixsize"]
    height_c=["height","iysize"]
    height_e=["h","cols","n_b"]
    irrelevant_c=["pitch","batch","stride","spatial","filter","scale","alpha","beta"]
    irrelevant_e=["a","b","c","d3","d2","d1","m","dim","seed","value","offset","pad"]
    
    
    if find_value_for_list(in_var,width_c,False):
        return "XSIZE"
    elif find_value_for_list(in_var,width_e,True):
        return "XSIZE"
    elif find_value_for_list(in_var,height_c,False):
        return "YSIZE"
    elif find_value_for_list(in_var,height_e,True):
        return "YSIZE"
    elif find_value_for_list(in_var,sizes_e,True):
        return "XSIZE*YSIZE"
    elif find_value_for_list(in_var,sizes_c,False):
        return "XSIZE*YSIZE"
    elif find_value_for_list(in_var,irrelevant_c,False):
        return "2"
    elif find_value_for_list(in_var,irrelevant_e,True):
        return "2"

    return "1"

def edit_values(path:str,device_id):
    a_file=None
    out = []
    try:
        a_file = open(path, "r")
        for line in a_file:
            stripped_line = line.strip()
            out.append(stripped_line)
        a_file.close()
    except UnicodeDecodeError:
        out=[]
        a_file = open(path, "r",encoding="latin1")
        for line in a_file:
            stripped_line = line.strip()
            out.append(stripped_line)
        a_file.close()
    for c in range(len(out)):

        
        if out[c].find("=")!=-1 and  out[c].find("for")==-1 and  out[c].find("while")==-1 and  out[c].find("auto")==-1 and out[c].find("blocks_")==-1 and out[c].find("matrices_")==-1:
            if out[c].split("=")[0].find("*")==-1:
                left=out[c].split("=")[0]
                out[c]=left+"= "+find_value(left)+";"
            if out[c]=="cudaSetDevice(0);":
                out[c]="cudaSetDevice("+str(device_id)+");"
    fs=open(path,'w')
    s1='\n'.join(out)
    fs.write(s1)
    fs.close()



def run_file(path:str, function, run_times, device_id, flags, timeout, matrix_len):
    bin_path = "./bin/"
    compile_cmd = "nvcc " + str(flags) + " " +  str(path)+ " -o="+bin_path+str(device_id)+".out"
    #print(compile_cmd)
    proc = subprocess.Popen([compile_cmd], stderr=subprocess.PIPE, shell=True,universal_newlines=True)
    (out, err) = proc.communicate()
    try:
            
            run_cmd = "timeout " + timeout + " " + bin_path + str(device_id) + ".out" + " "+str(matrix_len)
            #print(run_cmd)
            proc = subprocess.Popen([run_cmd], stdout=subprocess.PIPE, shell=True,universal_newlines=True)
            (out, err) = proc.communicate()
            results = out.split("\n")[:-1]
            for r in results:
                res = ast.literal_eval(r)
                run_times=run_times.append({'path':path,'function':function , 'time' : res[0], 'blocks': res[1], 'matrix':res[2]} , ignore_index=True)
    except KeyboardInterrupt:
        print("\nQuitting ...")
        sys.exit(0)
    except:
            run_times=run_times.append({'path':path,'function':function , 'time' : -1} , ignore_index=True)
        
    return run_times

def add_variables(df,variables,function):
    variables =ast.literal_eval(variables)
    for v in variables:
        if v[0].find("*")==-1:
            assumed_type = None
            for e in excacts:
                if v[1].lower().find(e)!=-1:
                    assumed_type=e
                    break
            for a in avoids:
                if v[1].lower().find(a)!=-1:
                    assumed_type=a
                    break
            df=df.append({'function':function , 'type' : v[0],"name":v[1],"assumed_type":assumed_type} , ignore_index=True)
    return df

def main():
    device_id=int(sys.argv[1])
    timeout=str(sys.argv[2])
    matrix_len=str(sys.argv[3])
    flags=str(sys.argv[4])
    data_path = "./data/"
    results_path = "./results/"
    runs = pd.read_csv(data_path+'kernel_list.csv')
    runtimes_path = results_path+"runtimes_temp_"+str(device_id)+".csv"
    run_times = pd.DataFrame(columns=['path',"function","time",'blocks','matrix'])#,'blocksizex',"blocksizey"])
    if path.exists(runtimes_path): 
        run_times = pd.read_csv(runtimes_path,low_memory=False)

    counter=0
    total_compute_time=time.time()
    prev_output_str_len = 0
    print("Started measurement process ...")
    for index, row in runs.iterrows():
        kernel_path = data_path+"kernels/"+str(row["Repo"])+"/"+str(row["underdirectory"])
        mainfile = kernel_path+"/"+"main.cu"
        if not any(run_times["path"].isin([mainfile])):
            remake_main(kernel_path,row['variables'],row['function'])
            edit_values(mainfile,device_id)
            compute_time=time.time()
            run_times=run_file(mainfile,row["function"],run_times,device_id,flags,timeout,matrix_len)
            run_times.to_csv(results_path+"runtimes_temp"+str(device_id)+".csv",index=False)
            
        counter+=1
        time_left = ((time.time()-total_compute_time) / counter)*(len(runs)-counter)
        print(" "*prev_output_str_len, end="\r")
        output_str = "Completion: " + str(counter) + "/" + str(len(runs)) + " (" + str(round(100*counter/len(runs), 4)) + "%), " + "Time left: " + str(datetime.timedelta(seconds=int(time_left)))
        prev_output_str = len(output_str)
        print(output_str, end="\r")

    run_times.dropna().to_csv(results_path+"runtimes_done_"+str(device_id)+".csv",index=False)
    print("Done")

if __name__ == "__main__":
    main()

