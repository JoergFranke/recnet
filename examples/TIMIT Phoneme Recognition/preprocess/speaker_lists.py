__author__ = 'joerg'

import os
import numpy as np


def get_speaker_lists(rootDir):

    np.random.seed(100)

    dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']

    train_speaker = []
    valid_speaker = []

    for i in dirlist:

        region_speakers = []

        for dirName, subdirList, fileList in os.walk(rootDir + i + "/"):
            #print(dirName)
            path,folder_name = os.path.split(dirName)
            if folder_name.__len__() >= 1:
                region_speakers.append(folder_name)

        len = region_speakers.__len__()
        print(len)
        valid_len = int(round(len * 0.1))
        random_valid = np.random.random_integers(0,region_speakers.__len__()-1,valid_len)
        random_train = np.delete(np.arange(0,region_speakers.__len__()),random_valid)
        region_speakers = np.asarray(region_speakers)

        train_speaker = train_speaker + list(region_speakers[random_train])
        valid_speaker = valid_speaker + list(region_speakers[random_valid])

    return train_speaker, valid_speaker


# valid_speaker = ['MEGJ0', 'FBJL0', 'MJWG0', 'MBGT0', 'MTDP0', 'FEAR0', 'MSRR0', 'MRMB0', 'MSVS0', 'MSVS0', 'MMDB0',
#                  'FKDE0', 'MBTH0', 'FKSR0', 'MSDB0', 'MPAR0', 'MGAK0', 'FCJS0', 'MBOM0', 'MCDD0', 'MRWA0', 'MDLH0',
#                  'FSKC0', 'MDEF0', 'FPAZ0', 'FSKC0', 'MILB0', 'MRFK0', 'MHRM0', 'FEAC0', 'MKAJ0', 'MKAH0', 'FAEM0',
#                  'MTBC0', 'FDAS1', 'MRCG0', 'MJEB1', 'MTJS0', 'MWAD0', 'MSMS0', 'MJXL0', 'FCAG0', 'MTRC0', 'FEEH0',
#                  'FBMJ0', 'FPAF0', 'FCLT0', 'FCEG0']
#
#
# train_speaker = ['MDWH0', 'MDAS0', 'MPMB0', 'FDMY0', 'MCLM0', 'FLJA0', 'MSDH0', 'FSKP0', 'MRML0', 'MEWM0', 'MRAV0',
#                  'FPMY0', 'FSAG0', 'FLOD0', 'MTAT0', 'MRKM0', 'MCHL0', 'FTLG0', 'FMPG0', 'FKKH0', 'MJPG0', 'MMAB1',
#                  'MVLO0', 'FGDP0', 'MJRG0', 'MWCH0', 'MHMG0', 'MGES0', 'FLJG0', 'FLMK0', 'FDTD0', 'MDHL0', 'MDSJ0',
#                  'MMWB0', 'FBMH0', 'MRLD0', 'MJFH0', 'MRAM0', 'MJXA0', 'FTBW0', 'FGMB0', 'MMCC0', 'FSDC0', 'MWSH0',
#                  'MWAC0', 'FSMM0', 'FCDR1', 'MFER0', 'MGSH0', 'MRVG0', 'MMVP0', 'MTMT0', 'MJDM0', 'FSMS1', 'MREW1',
#                  'MSEM1', 'FJXM0', 'MHIT0', 'MWEM0', 'FSJG0', 'MMDM1', 'FEXM0', 'MSAS0', 'MSJK0', 'MDRD0', 'MTJU0',
#                  'FSBK0', 'MPGR1', 'FTAJ0', 'FBCH0', 'MABC0', 'FRJB0', 'MTXS0', 'MCAE0', 'MSDS0', 'MRXB0', 'FSGF0',
#                  'FAPB0', 'MSAT1', 'MKES0', 'MEAL0', 'FJDM2', 'FSDJ0', 'MKLN0', 'FHXS0', 'MESJ0', 'MEJL0', 'FMJU0',
#                  'FKLC1', 'MBMA1', 'MJRK0', 'FPAD0', 'FLAG0', 'MAJP0', 'MSMR0', 'MTLC0', 'MRPC1', 'MDLR1', 'MGAR0',
#                  'MMDG0', 'MJRA0', 'MTMN0', 'MBBR0', 'FVKB0', 'MCRE0', 'MAFM0', 'MJAI0', 'FJEN0', 'MSAH1', 'FMAH1',
#                  'MTPR0', 'FSPM0', 'MDPB0', 'FLEH0', 'MRMG0', 'MGAW0', 'MDCM0', 'MWRE0', 'MTML0', 'MJJM0', 'MRLJ1',
#                  'FBLV0', 'FMKC0', 'MCLK0', 'MDLM0', 'MTKD0', 'MDLR0', 'MTER0', 'FPAC0', 'MADD0', 'MFXS0', 'MSES0',
#                  'MNTW0', 'MBAR0', 'MJDG0', 'MFXV0', 'MWRP0', 'MDED0', 'FLET0', 'MTAB0', 'FPAB1', 'FJRP1', 'MVRW0',
#                  'FREH0', 'MDLC1', 'MHBS0', 'FJHK0', 'MBML0', 'MMWS1', 'MKLR0', 'MREM0', 'MTWH1', 'FCRZ0', 'MKAG0',
#                  'MGSL0', 'FJSK0', 'MRMH0', 'MCTH0', 'MDKS0', 'MKDB0', 'MJFR0', 'MAEO0', 'MPFU0', 'MHXL0', 'MRTC0',
#                  'FJLR0', 'FEME0', 'FGCS0', 'MPRD0', 'MDLC0', 'FJLG0', 'MKLS1', 'MWGR0', 'MAKR0', 'MMAM0', 'FDFB0',
#                  'MJJB0', 'MCEF0', 'FCKE0', 'MTLB0', 'MJDA0', 'MBEF0', 'MGAF0', 'MCAL0', 'MHMR0', 'FSJS0', 'MJLG1',
#                  'MDHS0', 'FLAC0', 'FALK0', 'MFMC0', 'MRBC0', 'MJKR0', 'MDBB1', 'MTKP0', 'FGRW0', 'MAPV0', 'MMEB0',
#                  'MKXL0', 'MMAR0', 'MAKB0', 'MDSS1', 'MTJM0', 'FLJD0', 'MCDC0', 'MVJH0', 'FSLS0', 'MMSM0', 'FMJF0',
#                  'MMJB1', 'FNTB0', 'MLNS0', 'MDNS0', 'MSFV0', 'MDTB0', 'MREH1', 'MDWM0', 'FCMG0', 'MDDC0', 'MADC0',
#                  'MWDK0', 'MTPG0', 'MHJB0', 'FDJH0', 'MTPP0', 'MREE0', 'MRJB1', 'FSJW0', 'FLTM0', 'MJRH1', 'MRTJ0',
#                  'MRDS0', 'MDJM0', 'MARC0', 'MWSB0', 'MDLC2', 'MTAT1', 'MJBG0', 'MPPC0', 'MRJT0', 'MDSS0', 'MJPM0',
#                  'FMMH0', 'MMGK0', 'FHLM0', 'MZMB0', 'MJHI0', 'FCYL0', 'MRGS0', 'MBJV0', 'MSAT0', 'FJKL0', 'FLMC0',
#                  'MCTM0', 'MPRB0', 'FSCN0', 'MDLB0', 'FLMA0', 'MDEM0', 'MJEB0', 'FRLL0', 'MMAA0', 'FMKF0', 'MRAB0',
#                  'MTDB0', 'FTMG0', 'MDMT0', 'FAJW0', 'MCEW0', 'MMXS0', 'FPJF0', 'MEFG0', 'MMAG0', 'MRLJ0', 'MDPS0',
#                  'MKDT0', 'MJDE0', 'FCAJ0', 'MRJM1', 'FMJB0', 'MTJG0', 'FDNC0', 'FSRH0', 'MRLR0', 'MRMS0', 'FDXW0',
#                  'MRHL0', 'MJRP0', 'MDBP0', 'MDWD0', 'MKJO0', 'FCMM0', 'MJMA0', 'MRJM0', 'FKAA0', 'MJMD0', 'MJAE0',
#                  'MMDS0', 'MRCW0', 'MRJH0', 'FSKL0', 'MEDR0', 'FSMA0', 'FVFB0', 'MKLW0', 'MGRL0', 'MPGH0', 'MMRP0',
#                  'MJWT0', 'MRDD0', 'FSJK1', 'MKLS0', 'FSAH0', 'FDML0', 'FKFB0', 'FTBR0', 'FECD0', 'MWAR0', 'MPGR0',
#                  'MTPF0', 'MPSW0', 'FMEM0', 'FVMH0', 'MCPM0', 'MRAI0', 'MRSO0', 'FDAW0', 'FETB0', 'MDPK0', 'MRWS0',
#                  'FJSP0', 'MDAC0', 'FCJF0', 'MTRR0', 'MMGG0', 'MJWS0', 'MRSP0', 'MGRP0', 'MJAC0', 'FKDW0', 'MFWK0',
#                  'MJLS0', 'MMGC0', 'MAEB0', 'MPRT0', 'FJXP0', 'MGAG0', 'MBMA0', 'MJPM1', 'MJJJ0', 'FBAS0', 'FJWB1',
#                  'MRAB1', 'MTRT0', 'MJMM0', 'MRFL0', 'MTAS0', 'MDMA0', 'MPEB0', 'MLJH0', 'MJSR0', 'MMDM0', 'FDKN0',
#                  'MMBS0', 'MSFH0', 'MBWP0', 'MKAM0', 'MCSS0', 'MLSH0', 'MNET0', 'MJEE0', 'MJRH0', 'FALR0', 'MDCD0',
#                  'FSSB0', 'MJDC0', 'MCDR0', 'MLJC0', 'MJLB0', 'MRGM0', 'MPRK0', 'MLBC0', 'MLEL0', 'MSMC0', 'FLKM0',
#                  'MSRG0', 'MFRM0', 'MESG0', 'MSTF0', 'FLHD0', 'MGXP0', 'FKLC0', 'FSAK0', 'MTQC0', 'MGJC0', 'MARW0',
#                  'MMWS0', 'FMBG0', 'FPLS0', 'MMLM0', 'MRLK0', 'FBCG1', 'MKRG0', 'MKDD0', 'MRRE0', 'MRDM0', 'FKLH0',
#                  'MCXM0', 'MTCS0', 'MMPM0', 'MEJS0', 'FNKL0', 'MMEA0', 'MBCG0', 'MBSB0', 'FJRB0']
#
#
#




