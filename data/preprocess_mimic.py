import sys
sys.path.append("../")

import numpy as np 
import pandas as pd 

# Data dir
data_dir = "mimic/"

# Read csv files
pad = pd.read_csv(data_dir + "patients.csv")
adm = pd.read_csv(data_dir + "admissions.csv")
proc = pd.read_csv(data_dir + "procedures_icd.csv")
diag = pd.read_csv(data_dir + "diagnoses_icd.csv")
icustay = pd.read_csv(data_dir + "icustays.csv")

# ICU details
# This includes demographics, diagnoses and procedures 
adm["admittime"] = pd.to_datetime(adm["admittime"])
adm["dischtime"] = pd.to_datetime(adm["dischtime"])
diag['icd_version'] = diag['icd_version'].astype(int)

# Inner join to get demographics
demo = pd.merge(pd.merge(pat,adm,on='subject_id'),ie,on=['subject_id', 'hadm_id'])

# Assign age, all patients are older than 18 years old
demo = demo.assign(admission_age = demo['admittime'].dt.year - demo['anchor_year'] + demo['anchor_age'])

# Merge with procedure and diagnosis
demo = pd.merge(demo, diag, on=['subject_id', 'hadm_id'])
demo = demo.rename(columns={"icd_code": "diag_icd_code", "icd_version": "diag_icd_version", "seq_num": "diag_seq_num"})
demo = pd.merge(demo, proc, on=['subject_id', 'hadm_id'])
demo = demo.rename(columns={"icd_code": "proc_icd_code", "icd_version": "proc_icd_version", "seq_num": "proc_seq_num"})

# Group into higher order categories
def group_and_prefix_diag(code,version):
    return str(version) + "_" + code[:3]

def group_and_prefix_proc(code,version):
    return "IP_" + str(version) + "_" + code[:3]

demo['diag_icd_code'] = demo.apply(lambda x: group_and_prefix_diag(x.diag_icd_code, x.diag_icd_version), axis = 1)
demo['proc_icd_code'] = demo.apply(lambda x: group_and_prefix_proc(x.proc_icd_code, x.proc_icd_version), axis = 1)

# Group by visit id
group_diag = demo.groupby(['subject_id','hadm_id'])['diag_icd_code'].apply(list)
group_proc = demo.groupby(['subject_id','hadm_id'])['proc_icd_code'].apply(list)
group_age = demo.groupby(['subject_id','hadm_id'])['admission_age'].apply(lambda x: list(x)[0])
group_gender = demo.groupby(['subject_id','hadm_id'])['gender'].apply(lambda x: list(x)[0])
group_ethnicity = demo.groupby(['subject_id','hadm_id'])['ethnicity'].apply(lambda x: list(x)[0])
group_los = demo.groupby(['subject_id','hadm_id'])['los'].apply(lambda x: list(x)[0])

temp_final = pd.concat([group_age, group_gender,group_ethnicity, group_los, group_diag, group_proc], axis=1).reset_index()

# Group by patient id
group_diag_0 = temp_final.groupby(['subject_id'])['diag_icd_code'].apply(list)
group_proc_0 = temp_final.groupby(['subject_id'])['proc_icd_code'].apply(list)
group_age_0 = temp_final.groupby(['subject_id'])['admission_age'].apply(list)
group_gender_0 = temp_final.groupby(['subject_id'])['gender'].apply(lambda x: list(x)[0])
group_ethnicity_0 = temp_final.groupby(['subject_id'])['ethnicity'].apply(lambda x: list(x)[0])
group_los_0 = temp_final.groupby(['subject_id'])['los'].apply(list)
group_hadm_id_0 = temp_final.groupby(['subject_id'])['hadm_id'].apply(list)

final = pd.concat([group_hadm_id_0, group_age_0, group_gender_0 ,group_ethnicity_0 , group_los_0, group_diag_0, group_proc_0], axis=1).reset_index()

# Sequentialize codes
def _sequentialize_codes(diag, proc):

    lst_len = len(diag)
    seqs = []
        
    for i in range(lst_len):
        
        # Filter unique values
        u_diag = list(set(diag[i]))
        u_proc = list(set(proc[i]))
        concat = u_diag + u_proc
        inner_seq = "<SEP>".join(concat)
        seqs.append(inner_seq)
    seq = ";".join(seqs)
    return seq

final["seqential_code"] = final.apply(lambda x: _sequentialize_codes(x.diag_icd_code, x.proc_icd_code), axis = 1)

final.to_csv(data_dir + "sequential_mimic.csv")