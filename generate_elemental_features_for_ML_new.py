#!/usr/bin/env python
# coding: utf-8
# Written by Arun, (year 2019)


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os 
import sys
import glob
import warnings
warnings.simplefilter("error")
np.seterr(divide='ignore', invalid='ignore')

#----------------------------------------------------------------------------------------------
data      =  open("elemental_features_reduced.dat", "r")
null      =  data.readline()
tags      =  data.readline()
features  =  data.readlines()

symbol,   Z,            period, group,     mass,   kai_P, kai_A    = [], [], [], [], [], [], []
EA,       IE1,          IE2,    Rps_s,     Rps_p,  Rps_d, Rvdw     = [], [], [], [], [], [], []
Rcov,     MP,           BP,     Cp_g,      Cp_mol, rho,   E_fusion = [], [], [], [], [], [], []
E_vapor,  Thermal_Cond, Ratom,  Mol_Volume                         = [], [], [], []

for _features in features:
        _features  =  _features.split()
        symbol.append(_features[0])
        Z.append(_features[1])
        period.append(_features[2])  
        group.append(_features[3])
        mass.append(_features[4])         
        kai_P.append(_features[5])             #Pauling EN
        kai_A.append(_features[6])             #Allen EN
        EA.append(_features[7])     
        IE1.append(_features[8])     
        Rvdw.append(_features[9])  
        Rcov.append(_features[10])  
        MP.append(_features[11])       
        BP.append(_features[12])
        Cp_g.append(_features[13])
        Cp_mol.append(_features[14])
        rho.append(_features[15])
        E_fusion.append(_features[16])
        E_vapor.append(_features[17])
        Thermal_Cond.append(_features[18])
        Ratom.append(_features[19])
        Mol_Volume.append(_features[20])
        
        
# for _features in features:
#         _features  =  _features.split()
#         symbol.append(_features[0])
#         Z.append(_features[1])
#         period.append(_features[2])  
#         group.append(_features[3])
#         mass.append(_features[4])         
#         kai_P.append(_features[5])             #Pauling EN
#         kai_A.append(_features[6])             #Allen EN
#         EA.append(_features[7])     
#         IE1.append(_features[8])     
#         IE2.append(_features[9])     
#         Rps_s.append(_features[10])  
#         Rps_p.append(_features[11])  
#         Rps_d.append(_features[12])  
#         Rvdw.append(_features[13])  
#         Rcov.append(_features[14])  
#         MP.append(_features[15])       
#         BP.append(_features[16])
#         Cp_g.append(_features[17])
#         Cp_mol.append(_features[18])
#         rho.append(_features[19])
#         E_fusion.append(_features[20])
#         E_vapor.append(_features[21])
#         Thermal_Cond.append(_features[22])
#         Ratom.append(_features[23])
#         Mol_Volume.append(_features[24])
#----------------------------------------------------------------------------------------------
elemental_features = []

rootdir = 'materials'
list2=[]
inp1 = []
#mx_name, output = np.loadtxt("file-ml", unpack=True)
file1=open('file-ml', 'r')
#file2=open('mxene.out','w')
data = file1.readlines()
for ctr, line in enumerate(data):
        list1 = line.split()
#        list2.append(list1)
        print(list1[0])
      #  file2.write(ctr, list1)
        newdir = "{}/{}".format(rootdir,list1[0])
        print (newdir)
        os.chdir(newdir)
#        print(glob.glob("POSCAR"))       
#        coord= open(fname, "r")
#        print (coord)
#        print(coord.readlines())
#        inp1 = coord.readlines()
#        print (inp1)
#        inp1 = coord.read()
#        inp=[inp1[i].split for i in range(len(inp1))]
#        print(inp)
                
#print (list1)
#print(mx_name)
#sys.exit(0)   

# =============================================================================
# for subdir, dirs, files in os.walk(rootdir):
#     for dir in dirs:
#         print(dir)
#         newdir = "{}/{}".format(rootdir,dir)
#         print(newdir)
#         os.chdir(newdir) 
#         print(glob.glob("POSCAR"))
# 
# =============================================================================

        #----------------------------------------------------------------------------------------------
        inp     = open("POSCAR", "r")
        tag     = inp.readline()
        factor  = inp.readline()
        lv_x    = inp.readline()
        lv_y    = inp.readline()
        lv_z    = inp.readline()
        atoms   = inp.readline()
 #       print (atoms)
        natm    = inp.readline()
#        print(natm)
        natm    = natm.split()
#        print(natm)
        # print(natm)
        ntotal = 0
        for _natm in natm:
            ntotal = ntotal + int(_natm)

#        atoms   = list(atoms.split())
        atoms   = atoms.split()
#        print (atoms)
        total_Z, total_period, total_group,total_mass, total_kai_P, total_kai_A            = 0, 0, 0, 0, 0, 0
        total_EA, total_IE1, total_Rvdw  = 0, 0, 0
        total_Rcov, total_MP, total_BP, total_Cp_g, total_Cp_mol,total_rho, total_E_fusion = 0, 0, 0, 0, 0, 0, 0
        total_E_vapor, total_Thermal_Cond, total_Ratom,  total_Mol_Volume                  = 0, 0, 0, 0
        # print("ntotal", ntotal)

        ctr = 0
        for element in atoms:
            element_id         = symbol.index(str(element))
            atomic_no          = int(element_id)    + 1
        #     ctr                = natm.index(str(element_id))
        #     print("ctr", ctr, " natm ", int(natm[ctr]))
        #     print(str(element), element_id, atomic_no, float(kai_P[element_id]))
            total_Z            = total_Z            + int(natm[ctr]) * float(Z[element_id])
            total_period       = total_period       + int(natm[ctr]) * float(period[element_id])
            total_group        = total_group        + int(natm[ctr]) * float(group[element_id])
            total_mass         = total_mass         + int(natm[ctr]) * float(mass[element_id])
            total_kai_P        = total_kai_P        + int(natm[ctr]) * float(kai_P[element_id])
            total_kai_A        = total_kai_A        + int(natm[ctr]) * float(kai_A[element_id])
            total_EA           = total_EA           + int(natm[ctr]) * float(EA[element_id])
            total_IE1          = total_IE1          + int(natm[ctr]) * float(IE1[element_id])
#             total_IE2          = total_IE2          + int(natm[ctr]) * float(IE2[element_id])
#             total_Rps_s        = total_Rps_s        + int(natm[ctr]) * float(Rps_s[element_id])
#             total_Rps_p        = total_Rps_p        + int(natm[ctr]) * float(Rps_p[element_id])
#             total_Rps_d        = total_Rps_d        + int(natm[ctr]) * float(Rps_d[element_id])
            total_Rvdw         = total_Rvdw         + int(natm[ctr]) * float(Rvdw[element_id])
            total_Rcov         = total_Rcov         + int(natm[ctr]) * float(Rcov[element_id])
            total_MP           = total_MP           + int(natm[ctr]) * float(MP[element_id])
            total_BP           = total_BP           + int(natm[ctr]) * float(BP[element_id])
            total_Cp_g         = total_Cp_g         + int(natm[ctr]) * float(Cp_g[element_id])
            total_Cp_mol       = total_Cp_mol       + int(natm[ctr]) * float(Cp_mol[element_id])
            total_rho          = total_rho          + int(natm[ctr]) * float(rho[element_id])
            total_E_fusion     = total_E_fusion     + int(natm[ctr]) * float(E_fusion[element_id])
            total_E_vapor      = total_E_vapor      + int(natm[ctr]) * float(E_vapor[element_id])
            total_Thermal_Cond = total_Thermal_Cond + int(natm[ctr]) * float(Thermal_Cond[element_id])
            total_Ratom        = total_Ratom        + int(natm[ctr]) * float(Ratom[element_id])
            total_Mol_Volume   = total_Mol_Volume   + int(natm[ctr]) * float(Mol_Volume[element_id])    
        #     ntotal             = ntotal             + int(natm[ctr])
            ctr += 1
        # print(total_kai_P, ntotal)  


        mean_Z            = total_Z            / int(ntotal) 
        mean_period       = total_period       / int(ntotal)  
        mean_group        = total_group        / int(ntotal)  
        mean_mass         = total_mass         / int(ntotal)  
        mean_kai_P        = total_kai_P        / int(ntotal)  
        mean_kai_A        = total_kai_A        / int(ntotal)  
        mean_EA           = total_EA           / int(ntotal)  
        mean_IE1          = total_IE1          / int(ntotal)  
#         mean_IE2          = total_IE2          / int(ntotal)  
#         mean_Rps_s        = total_Rps_s        / int(ntotal)  
#         mean_Rps_p        = total_Rps_p        / int(ntotal)  
#         mean_Rps_d        = total_Rps_d        / int(ntotal)  
        mean_Rvdw         = total_Rvdw         / int(ntotal)  
        mean_Rcov         = total_Rcov         / int(ntotal)  
        mean_MP           = total_MP           / int(ntotal)  
        mean_BP           = total_BP           / int(ntotal)  
        mean_Cp_g         = total_Cp_g         / int(ntotal)  
        mean_Cp_mol       = total_Cp_mol       / int(ntotal)  
        mean_rho          = total_rho          / int(ntotal)  
        mean_E_fusion     = total_E_fusion     / int(ntotal)  
        mean_E_vapor      = total_E_vapor      / int(ntotal)  
        mean_Thermal_Cond = total_Thermal_Cond / int(ntotal)  
        mean_Ratom        = total_Ratom        / int(ntotal)  
        mean_Mol_Volume   = total_Mol_Volume   / int(ntotal)  

        mean_values       = [mean_Z, mean_period, mean_group, mean_mass, mean_kai_P,mean_kai_A,  mean_EA, mean_IE1,  
                             mean_Rvdw, mean_Rcov, mean_MP, mean_BP, mean_Cp_g, 
                             mean_Cp_mol, mean_rho, mean_E_fusion, mean_E_vapor, mean_Thermal_Cond, mean_Ratom, mean_Mol_Volume]
#         mean_values       = [mean_Z, mean_period, mean_group, mean_mass]
#         mean_values       = [mean_Z, mean_period, mean_group, mean_mass, mean_kai_P,mean_kai_A,  mean_EA, mean_IE1]

        # print(mean_values)

        #----------------------------------------------------------------------------------------------------------
        # print()
        sd_Z, sd_period, sd_group, sd_mass, sd_kai_P, sd_kai_A        = 0, 0, 0, 0, 0, 0
        sd_EA, sd_IE1, sd_Rvdw  = 0, 0, 0
        sd_Rcov, sd_MP, sd_BP, sd_Cp_g, sd_Cp_mol,sd_rho, sd_E_fusion = 0, 0, 0, 0, 0, 0, 0
        sd_E_vapor, sd_Thermal_Cond, sd_Ratom,  sd_Mol_Volume         = 0, 0, 0, 0

        ctr = 0
        for element in atoms:
            element_id      = symbol.index(str(element))
            atomic_no       = int(element_id)    + 1
        #     print(str(element), element_id, atomic_no, float(kai_P[element_id]), mean_kai_P)    
            sd_Z            = sd_Z            + int(natm[ctr]) * (float(Z[element_id])     - mean_Z)**2
            sd_period       = sd_period       + int(natm[ctr]) * (float(period[element_id])- mean_period)**2
            sd_group        = sd_group        + int(natm[ctr]) * (float(group[element_id]) - mean_group)**2
            sd_mass         = sd_mass         + int(natm[ctr]) * (float(mass[element_id])  - mean_mass)**2
            sd_kai_P        = sd_kai_P        + int(natm[ctr]) * (float(kai_P[element_id]) - mean_kai_P)**2
            sd_kai_A        = sd_kai_A        + int(natm[ctr]) * (float(kai_A[element_id]) - mean_kai_A)**2
            sd_EA           = sd_EA           + int(natm[ctr]) * (float(EA[element_id])    - mean_EA)**2
            sd_IE1          = sd_IE1          + int(natm[ctr]) * (float(IE1[element_id])   - mean_IE1)**2
#             sd_IE2          = sd_IE2          + int(natm[ctr]) * (float(IE2[element_id])   - mean_IE2)**2
#             sd_Rps_s        = sd_Rps_s        + int(natm[ctr]) * (float(Rps_s[element_id]) - mean_Rps_s)**2
#             sd_Rps_p        = sd_Rps_p        + int(natm[ctr]) * (float(Rps_p[element_id]) - mean_Rps_p)**2
#             sd_Rps_d        = sd_Rps_d        + int(natm[ctr]) * (float(Rps_d[element_id]) - mean_Rps_d)**2
            sd_Rvdw         = sd_Rvdw         + int(natm[ctr]) * (float(Rvdw[element_id])  - mean_Rvdw)**2
            sd_Rcov         = sd_Rcov         + int(natm[ctr]) * (float(Rcov[element_id])  - mean_Rcov)**2
            sd_MP           = sd_MP           + int(natm[ctr]) * (float(MP[element_id])    - mean_MP)**2
            sd_BP           = sd_BP           + int(natm[ctr]) * (float(BP[element_id])    - mean_BP)**2
            sd_Cp_g         = sd_Cp_g         + int(natm[ctr]) * (float(Cp_g[element_id])  - mean_Cp_g)**2
            sd_Cp_mol       = sd_Cp_mol       + int(natm[ctr]) * (float(Cp_mol[element_id])- mean_Cp_mol)**2
            sd_rho          = sd_rho          + int(natm[ctr]) * (float(rho[element_id])   - mean_rho)**2
            sd_E_fusion     = sd_E_fusion     + int(natm[ctr]) * (float(E_fusion[element_id])     - mean_E_fusion)**2
            sd_E_vapor      = sd_E_vapor      + int(natm[ctr]) * (float(E_vapor[element_id])      - mean_E_vapor)**2
            sd_Thermal_Cond = sd_Thermal_Cond + int(natm[ctr]) * (float(Thermal_Cond[element_id]) - mean_Thermal_Cond)**2
            sd_Ratom        = sd_Ratom        + int(natm[ctr]) * (float(Ratom[element_id])        - mean_Ratom)**2
            sd_Mol_Volume   = sd_Mol_Volume   + int(natm[ctr]) * (float(Mol_Volume[element_id])   - mean_Mol_Volume)**2
            ctr += 1
        #     ntotal          = ntotal          + int(natm[ctr])
        # print(np.sqrt(sd_kai_P/(ntotal-1)))  

        sd_Z            = np.sqrt(sd_Z            / (ntotal-1))
        sd_period       = np.sqrt(sd_period       / (ntotal-1)) 
        sd_group        = np.sqrt(sd_group        / (ntotal-1))
        sd_mass         = np.sqrt(sd_mass         / (ntotal-1))
        sd_kai_P        = np.sqrt(sd_kai_P        / (ntotal-1))
        sd_kai_A        = np.sqrt(sd_kai_A        / (ntotal-1)) 
        sd_EA           = np.sqrt(sd_EA           / (ntotal-1)) 
        sd_IE1          = np.sqrt(sd_IE1          / (ntotal-1))
#         sd_IE2          = np.sqrt(sd_IE2          / (ntotal-1)) 
#         sd_Rps_s        = np.sqrt(sd_Rps_s        / (ntotal-1))
#         sd_Rps_p        = np.sqrt(sd_Rps_p        / (ntotal-1))
#         sd_Rps_d        = np.sqrt(sd_Rps_d        / (ntotal-1))
        sd_Rvdw         = np.sqrt(sd_Rvdw         / (ntotal-1)) 
        sd_Rcov         = np.sqrt(sd_Rcov         / (ntotal-1)) 
        sd_MP           = np.sqrt(sd_MP           / (ntotal-1))
        sd_BP           = np.sqrt(sd_BP           / (ntotal-1)) 
        sd_Cp_g         = np.sqrt(sd_Cp_g         / (ntotal-1))
        sd_Cp_mol       = np.sqrt(sd_Cp_mol       / (ntotal-1)) 
        sd_rho          = np.sqrt(sd_rho          / (ntotal-1))
        sd_E_fusion     = np.sqrt(sd_E_fusion     / (ntotal-1))
        sd_E_vapor      = np.sqrt(sd_E_vapor      / (ntotal-1)) 
        sd_Thermal_Cond = np.sqrt(sd_Thermal_Cond / (ntotal-1))
        sd_Ratom        = np.sqrt(sd_Ratom        / (ntotal-1)) 
        sd_Mol_Volume   = np.sqrt(sd_Mol_Volume   / (ntotal-1))

        sd_values       = [sd_Z, sd_period, sd_group, sd_mass, sd_kai_P,sd_kai_A,  sd_EA, sd_IE1,   
                             sd_Rvdw, sd_Rcov, sd_MP, sd_BP, sd_Cp_g, 
                             sd_Cp_mol, sd_rho, sd_E_fusion, sd_E_vapor, sd_Thermal_Cond, sd_Ratom, sd_Mol_Volume]
#         sd_values       = [sd_Z, sd_period, sd_group, sd_mass]
#         sd_values       = [sd_Z, sd_period, sd_group, sd_mass, sd_kai_P,sd_kai_A,  sd_EA, sd_IE1]

        #         print(mean_values,sd_values)
#         print()
#         print(mean_values+sd_values)
        mean_sd = np.concatenate([[list1[0]], mean_values,sd_values, [list1[1]]])
        print(mean_sd)
        elemental_features.append(mean_sd)
        os.chdir(rootdir)
        print("======================================================")

#print("Mx", len(elemental_features))
# print(elemental_features[0:2])
elemental_features = np.asarray(elemental_features)
# np.savetxt("MXenes_features.csv", elemental_features, delimiter=",")


import pandas as pd

# X=[['Sc-Sc', 1,2],['Mo-Mo',2,3],['Y-Y',3,4]]
# print(X)
#print(elemental_features.shape)
columns=["MXene",
    "mean_Z", "mean_period", "mean_group", "mean_mass", "mean_kai_P","mean_kai_A",  "mean_EA", "mean_IE1", "mean_Rvdw", "mean_Rcov", "mean_MP", "mean_BP", "mean_Cp_g", 
    "mean_Cp_mol", "mean_rho", "mean_E_fusion", "mean_E_vapor", "mean_Thermal_Cond", "mean_Ratom", "mean_Mol_Volume",
    "sd_Z", "sd_period", "sd_group", "sd_mass", "sd_kai_P","sd_kai_A",  "sd_EA", "sd_IE1", "sd_Rvdw", "sd_Rcov", "sd_MP", "sd_BP", "sd_Cp_g", 
    "sd_Cp_mol", "sd_rho", "sd_E_fusion", "sd_E_vapor", "sd_Thermal_Cond", "sd_Ratom", "sd_Mol_Volume", "y"]
# columns=[
#     "mean_Z", "mean_period", "mean_group", "mean_kai_P",  "sd_Z", "sd_period", "sd_group", "sd_mass"]
# columns=[
#     "mean_Z", "mean_period", "mean_group", "mean_kai_P","mean_kai_A",  "mean_EA", "mean_IE1", "sd_Z", "sd_period", "sd_group", "sd_mass", "sd_kai_P","sd_kai_A",  "sd_EA", "sd_IE1"]

# df = pd.DataFrame(elemental_features)
df = pd.DataFrame(elemental_features, columns=columns)

#print(np.array(columns).shape)
# print(df)
df.to_csv('example1.csv', index=False)
print (df) 
