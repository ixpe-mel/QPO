#Packages
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import os 
import glob
import pandas as pd
from scipy import stats
from scipy import optimize
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.io import ascii
from astropy.table import Table
import csv
import math
import stingray
import lightcurve
from stingray import Lightcurve
from stingray import Powerspectrum
from stingray import AveragedPowerspectrum
from stingray import Crossspectrum
from stingray.exceptions import StingrayError
from stingray import AveragedCrossspectrum
from more_itertools import locate
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

#Constants
x_0= 291.41708
y_0=286.77622
r_0=30.769326
E_lo=2
E_hi=8
Pmin=((375/15)*E_lo)+1
Pmax=(375/15)*E_hi


def cleaner(filename,Pmin,Pmax,source_name_num):
     with fits.open(str(filename)) as hdu:
        data=hdu[1].data #loading in main data
        # r cut 
        r_sqrd=abs((data.field('X')-x_0)**2-(data.field('Y')-y_0)**2)    #r cut
        index_r=[j<r_0 for j in r_sqrd]
        data=data[index_r] #indexing photon dataset

        #PI channel cut
        index_energy=list(locate(data.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut 
        data=data[index_energy]
        
        fits.writeto(str(source_name_num)+'_clean_.fits',data,overwrite=True)


# make sure to use in virtual environment (type ve into terminal)
def concatenate_fits_files(file1, file2, output_file):
    # Read the first table FITS file
    table1 = Table.read(file1)
    
    # Read the second table FITS file
    table2 = Table.read(file2)

    # Concatenate the tables
    concatenated_table =np.hstack([table1, table2])
    
    t = fits.BinTableHDU.from_columns(concatenated_table)
    # Write the concatenated table to a new FITS file
    with fits.open(file1) as hdu1:
        prihdr = hdu1[1].header
        prihdu = fits.PrimaryHDU(header=prihdr)
        thdulist = fits.HDUList([prihdu, t])
        thdulist.writeto(output_file, overwrite=True)



def averaged_cross_spectrum_err(det12,det3,gti,Pmin,Pmax,bin_length,seg_length):
    
    mega_real=[]
    mega_im=[]
    
    GTI=list(np.loadtxt(str(gti)))
    
    with fits.open(str(det12)) as hdu:

        data_main=hdu[1].data  #reading in the file we will use the main data of (ie stokes, PI channel etc)
        #print('Initial number of data {}'.format(len(data_main)))
    with fits.open(str(det3)) as hdu1:
        data_ref=hdu1[1].data  #reading in the file we will use the main
        data_header=hdu1[1].header #reading in the file we will use the header of (the header data we need here is the same for all raw fits so any will do)
        
        
       
        

#indexing on energy and r
    
    
    
        r_sqrd_1=abs((data_main.field('X')-x_0)**2-(data_main.field('Y')-y_0)**2)    #r cut just like before
        r_sqrd_3=abs((data_ref.field('X')-x_0)**2-(data_ref.field('Y')-y_0)**2)
        r_0=30.769326
        index_r_1=[u<r_0 for u in r_sqrd_1]
        index_r_3=[r3<r_0 for r3 in r_sqrd_3]
        
        data_main=data_main[index_r_1]
        data_ref=data_ref[index_r_3]
       


        index_energy_1=list(locate(data_main['PI'], lambda x: Pmin < x < Pmax))  #energy cut just like before
        index_energy_3=list(locate(data_ref['PI'], lambda x: Pmin < x < Pmax))  #energy cut just like before
        
        
        
        
        
        data_main_total_obs=data_main[index_energy_1]
        data_ref_total_obs=data_ref[index_energy_3]    #useable photons (before gtis) to index over 
        #-----------------------------------------------------------------------------cleaning complete

        
        #Defining fits header info
        
        TSTART=data_header['TSTART']
        TSTOP=data_header['TSTOP']
        MJDREFF=data_header['MJDREFF']   
        MJDREFI=data_header['MJDREFI']
        MJD_ref_day=MJDREFF+MJDREFI
        curve_duration=TSTOP-TSTART
#------------------------------------------------------------------------------------- This bit will always stay the same!

        
        #making time list based on length of bin to index on

        time_bin_number=int((TSTOP-TSTART)//seg_length)  #needs to be an integer so round down to 'remove' the short timebin before its made
        #print(time_bin_number)
        #time_bin_number=2**18
        time_space=np.linspace(TSTART,TSTOP,time_bin_number)
        seg_time_list=[(time_space[w-1],time_space[w]) for w in range(len(time_space))] 
        seg_time_list.pop(0)  
        #print((seg_time_list))
        
        #testing the time intervals on GTI values
        
        #gti_segs=[]
        
        
        
        gti_segs=[]
        for time_value in seg_time_list:
            
            time_min_val=time_value[0]
            time_max_val=time_value[1]
            #print(time_min_val)
            #print(time_max_val)
            for j in GTI:
                #print(j[0])
                #print(j[1])
                if time_min_val>j[0] and time_max_val<j[1]:
                    gti_segs.append(time_value)

            
        real=[]
        im=[]

        #print(len(gti_segs))   
        for good_seg in gti_segs:
             
            time_min=good_seg[0]
            time_max=good_seg[1]

            time_index_1=[time_min<photon_time_1<time_max for photon_time_1 in data_main_total_obs['TIME']]
            time_index_3=[time_min<photon_time_3<time_max for photon_time_3 in data_ref_total_obs['TIME']]
            #print(len(data_main))

            data_main_time_bin=data_main_total_obs[time_index_1]
            data_ref_time_bin=data_ref_total_obs[time_index_3]



            #making lcs over all mod angles before we index over the
            
            
       
           #making lightcurves binned over mod angle (also time)
        
        
            lc_12=Lightcurve.make_lightcurve(data_main_time_bin['TIME'],dt=bin_length,tseg=time_max-time_min,tstart=time_min)#,gti=GTI) 
            lc_3=Lightcurve.make_lightcurve(data_ref_time_bin['TIME'],dt=bin_length,tseg=time_max-time_min,tstart=time_min)#),gti=GTI)
            
            
            
        

            
            cs=Crossspectrum.from_lightcurve(lc_12,lc_3,norm='frac') #making averagd cross spec
         

            
            real.append(cs.power.real)
            im.append(cs.power.imag)
            
            
        
            
        sem_real=[]
        sem_im=[]
        
        mean_real=[]
        mean_im=[]
        
        for i in range(len(real[0])):
        
            mean_r = np.mean([arr[i] for arr in real])
            mean_i=np.mean([ip[i] for ip in im])
        
            mean_real.append(mean_r)
            mean_im.append(mean_i)
        
        #Standard error on the average imaginary power
            sem_i=np.std([ip[i] for ip in im]) / np.sqrt((np.size(im)))
            sem_im.append(sem_i)
            
            sem_r=np.std([r[i] for r in real]) / np.sqrt((np.size(real)))
            sem_real.append(sem_r)
            
            
            
            
            
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.plot(cs.freq, mean_real*cs.freq)
        plt.errorbar(cs.freq,mean_real*cs.freq,yerr=sem_real)
        #plt.xlim(0.001, max(cs.freq))
        #plt.ylim(-0.5,0.5)
        #plt.show()
        lc12all=Lightcurve.make_lightcurve(data_main_total_obs.field('TIME'),bin_length,gti=GTI)  
        lc3all=Lightcurve.make_lightcurve(data_ref_total_obs.field('TIME'),bin_length,gti=GTI)
        cs=AveragedCrossspectrum.from_lightcurve(lc12all,lc3all,seg_length,norm='frac')
        
        plt.plot(cs.freq,cs.power.real*cs.freq,color='red')
        plt.show()
        
        
    

def master_GTI(data1,data2,data3,source_name):
    
    with fits.open(str(data1)) as hdu1:
        GTI1=list(hdu1[2].data)
        gtistart1=[item[0] for item in GTI1]
        gtiend1=[item[1] for item in GTI1]
        EVENTS_header1=hdu1[1].header
        TSTOP1=EVENTS_header1['TSTOP']
    
    
    with fits.open(str(data2)) as hdu2:
        GTI2=list(hdu2[2].data)
        gtistart2=[item[0] for item in GTI2]
        gtiend2=[item[1] for item in GTI2]

    with fits.open(str(data3)) as hdu3:
        GTI3=list(hdu3[2].data)
        gtistart3=[item[0] for item in GTI3]
        gtiend3=[item[1] for item in GTI3]


    
    
    
    
    beep=0
    counter = 0
    i1 = 0 
    i2 = 0
    i3 = 0
    gtistart=[]
    gtiend=[]
    while TSTOP1 - beep > 1e-6:  #while the beep indexer isnt the end of the observation:
        
        gtistart.append(max(gtistart1[i1], gtistart2[i2], gtistart3[i3]) ) #defining start of current master GTI
        
        gtiend.append(0)
        
        gtiend[counter]=TSTOP1 #making the current index of gtiend equal to the end of obs (broken: 
        
        if( gtiend1[i1] > gtistart[counter]):
            gtiend[counter]=( min( gtiend1[i1] , gtiend[counter] ))
        else:
            None
        if( gtiend2[i2] > gtistart[counter] ):
            gtiend[counter]=( min( gtiend2[i2] , gtiend[counter] ))
        else:
            None
        if( gtiend3[i3] > gtistart[counter] ):
            gtiend[counter]=( min( gtiend3[i3] , gtiend[counter] ))
        else:
            None
    
        
        
        
        
        
        if gtiend1[i1] - gtiend[counter] < 1e-6:
            i1 = i1 + 1
        else:
            None
        if gtiend2[i2] - gtiend[counter] < 1e-6:
            i2 = i2 + 1
        else:
            None
        if gtiend3[i3] - gtiend[counter] < 1e-6:
            i3 = i3 + 1
        else:
            None
        
        beep=gtiend[counter]   
        counter=counter+1
        
        
    #return [gtistart,gtiend]

    master_gti=[list(x) for x in zip(gtistart,gtiend)]
    np.savetxt('Results/'+str(source_name)+'_gti.txt',master_gti)
    print(master_gti)
    

            
        


#calculate normalised modulation angle

def cal_eff_mod_angle(filename,source_name_datanumber):
    with fits.open(str(filename)) as hdu:
        data=hdu[1].data
        q=data.field('q')
        u=data.field('u')
        q_renormalised=q/np.sqrt(q**2+u**2)
        u_renormalised=u/np.sqrt(q**2+u**2)
        atan2_vec=np.vectorize(math.atan2)
        eff_mod_angle=0.5 *atan2_vec(q_renormalised,u_renormalised)
        np.savetxt('Results/eff_mod_angle'+str(source_name_datanumber)+'.txt',eff_mod_angle)

        
#calculate unnormalised modulation angle        
def cal_eff_mod_angle_unnormalised(filename,source_name_datanumber):
    with fits.open(str(filename)) as hdu:
        data=hdu[1].data
        q=data.field('q')
        u=data.field('u')
        #q_renormalised=q/np.sqrt(q**2+u**2)
        #u_renormalised=u/np.sqrt(q**2+u**2)
        atan2_vec=np.vectorize(math.atan2)
        eff_mod_angle=0.5 *atan2_vec(q,u)
        np.savetxt('Results/eff_mod_angle'+str(source_name_datanumber)+'.txt',eff_mod_angle)
        
        


#make a power spectrum

def powerspectrum(data_file,gti,bin_length,seg_length):
   #data
   
    
    GTI=list(np.loadtxt(str(gti)))
    #print(GTI)
    
    
    with fits.open(str(data_file)) as hdu:
        data=hdu[1].data
      # if data_header is not None:
         
        # r cut 
        r_sqrd=abs((data.field('X')-x_0)**2-(data.field('Y')-y_0)**2)    #r cut just like before now i is each mod angle select fits file 
        index_r=[j<r_0 for j in r_sqrd]
        data=data[index_r] #indexing photon dataset
       # eff_mod_angle=eff_mod_angle[index_r] #indexing mod angle dataset
  

        #PI channel cut
        index_energy=list(locate(data.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
        data=data[index_energy]
       # eff_mod_angle=eff_mod_angle[index_energy]

  
        TIME=data.field('TIME')
       

      #Lightcurve

        lightcurve_12=Lightcurve.make_lightcurve(TIME,dt=bin_length,gti=GTI)
        lightcurve_12.apply_gtis()

        ps=Powerspectrum.from_lightcurve(lightcurve_12,seg_length)

        fig, ax1 = plt.subplots(1,1,figsize=(9,6))
                #ax1.plot(cs.freq, cs.power, color='blue',label='no log rebin')
        ax1.plot(ps.freq, ps.power.real, color='green')
                #ax1.plot(avg_ps_log.freq, avg_ps_log.power, color='red',label='log rebin')
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_title('Power spec')
        ax1.set_ylabel("Real Power")
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.tick_params(which='major', width=1.5, length=7)
        ax1.tick_params(which='minor', width=1.5, length=4)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(1.5)
            plt.show()



#making data ready to be lightcurved
def cleaner_and_mod_angle_selector(filename,filename_eff,source_name_datanumber,Pmin,Pmax,mod_bin_num):
    # Data
    with fits.open(str(filename)) as hdu:
        data=hdu[1].data #loading in main data
       # print('Initial number of data {}'.format(len(data)))
    eff_mod_angle=np.loadtxt(str(filename_eff)) #loading in mod angles of combined 
    
    # Header Data
    #with fits.open(str(filename_fits)) as hdu2:
    #    data_header=hdu2[1].header 
    #    TSTART=data_header['TSTART']
    #    TSTOP=data_header['TSTOP']
    #    MJDREFF=data_header['MJDREFF']   #defining the fits header data 
    #    MJDREFI=data_header['MJDREFI']
    #    MJD_ref_day=MJDREFF+MJDREFI
    #    curve_duration=TSTOP-TSTART
        
        # r cut 
        r_sqrd=abs((data.field('X')-x_0)**2-(data.field('Y')-y_0)**2)    #r cut just like before now i is each mod angle select fits file 
        index_r=[j<r_0 for j in r_sqrd]
        data=data[index_r] #indexing photon dataset
        eff_mod_angle=eff_mod_angle[index_r] #indexing mod angle dataset
  

        #PI channel cut
        index_energy=list(locate(data.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
        data=data[index_energy]
        eff_mod_angle=eff_mod_angle[index_energy]


        #Modulation Angle List
        mod_min_global=np.radians(-90)
        mod_max_global=np.radians(90)

        a=np.linspace(mod_min_global,mod_max_global,mod_bin_num+1) #defining even space between min and max for (mod_bin_num) list
        mod_angle_list=[(a[i-1],a[i]) for i in range(len(a))]  #making a list of mod angle bins to select over
        mod_angle_list.pop(0) #removing the dodger first one
             
        
        #for each [min,max] in the mod angle list, only the the angles between these vals are selected and the same index is applied to the data
        
        for i in mod_angle_list:
            mod_min=i[0] #defining lhs bin edge
            mod_max=i[1] #defining rhs bin edge
            index_mod_angle=[mod_min<=k<=mod_max for k in eff_mod_angle] #define the index over mod angle
            data_bin=data[index_mod_angle] #selecting/indexing the photons that meet the criteria of this mod angle range             
            #save cleaned and mod selected fits file
            fits.writeto('Lightcurves/lc_'+str(source_name_datanumber)+'_'+str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_.fits',data_bin,overwrite=True)
            
            

#make a cross spectrum            
def crossspectrum(file12,file3,gti,bin_length,seg_length,Pmin,Pmax,clean):
    
    GTI=list(np.loadtxt(str(gti)))
    
    with fits.open(str(file12)) as hdu:
            data_12=hdu[1].data  #reading in DU1+DU2
            print(data_12[0])
    
    with fits.open(str(file3)) as hdu2:
            #data_header=hdu2[1].header #reading in header 
            data_3=hdu2[1].data #reading in DU3
            print(data_3[0])
            

            #TSTART=data_header['TSTART']
            #TSTOP=data_header['TSTOP']
            #MJDREFF=data_header['MJDREFF']   #defining the fits header data from the raw fits file (since our combined cyg12 data doesnt have a header i dont think
            #MJDREFI=data_header['MJDREFI']
            #MJD_ref_day=MJDREFF+MJDREFI
            #curve_duration=TSTOP-TSTART
            
            if clean is not None:
                print('oh no')
                # r cut index
                r_sqrd=abs((data_12.field('X')-x_0)**2-(data_12.field('Y')-y_0)**2)  #r cut just like before now i is each mod angle select fits file
                r_sqrd_3=abs((data_3.field('X')-x_0)**2-(data_3.field('Y')-y_0)**2) 
                index_r=[j<r_0 for j in r_sqrd]
                index_r_3=[m<r_0 for m in r_sqrd_3]
                data_12=data_12[index_r] #indexing photon dataset
                data_3=data_3[index_r_3]
                #eff_mod_angle_12=eff_mod_angle_12[index_r] #indexing mod angle dataset


                #PI channel/energy index
                index_energy=list(locate(data_12.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
                data_12=data_12[index_energy]
                index_energy_3=list(locate(data_3.field('PI'), lambda x: Pmin < x < Pmax))
                data_3=data_3[index_energy_3]
                #eff_mod_angle_12=eff_mod_angle_12[index_energy]

            else:
                pass

            TIME=data_12.field('TIME')
            TIME_3=data_3.field('TIME')

            #Lightcurve

            lightcurve_12=Lightcurve.make_lightcurve(TIME,dt=bin_length,gti=GTI)
            lightcurve_12.apply_gtis()
            print('lc')

            lightcurve_3=Lightcurve.make_lightcurve(TIME_3,dt=bin_length,gti=GTI)
            lightcurve_3.apply_gtis()
            print('lc_3')
            
            

            #Cross spec 

            avg_cs = AveragedCrossspectrum.from_lightcurve(lightcurve_12,lightcurve_3,seg_length,norm='frac')
            avg_cs=avg_cs.rebin_log(f=0.1)

           #Plotting the real part against fourier frequency

            fig, ax1 = plt.subplots(1,1,figsize=(9,6))
                #ax1.plot(cs.freq, cs.power, color='blue',label='no log rebin')
            ax1.plot(avg_cs.freq, avg_cs.power.real*avg_cs.freq,'.', color='green')
                #ax1.plot(avg_ps_log.freq, avg_ps_log.power, color='red',label='log rebin')
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_title('Cross spec')
            ax1.set_ylabel("Power x Fourier frequency")
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax1.tick_params(axis='x', labelsize=16)
            ax1.tick_params(axis='y', labelsize=16)
            ax1.tick_params(which='major', width=1.5, length=7)
            ax1.tick_params(which='minor', width=1.5, length=4)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(1.5)
                plt.show()

        



            
def G_norm(source_name,bin_length,seg_length,Pmin,Pmax,fmin,fmax,mod_bin_number,norm12,norm3,gti):
    
    av_power_array=[]
    av_power_array_real=[]
    av_power_array_im=[]
    err_array=[]
    err_array_real=[]
    err_array_im=[]
    norm_factor_array=[]
    err_array_norm_real=[]
    err_array_norm_im=[]
    
    GTI=list(np.loadtxt(str(gti)))
  
    with fits.open(str(norm12)) as hdu:
            data_12=hdu[1].data  #reading in DU1+DU2
    
    with fits.open(str(norm3)) as hdu2:
            data_header=hdu2[1].header #reading in header 
            data_3=hdu2[1].data #reading in DU3
            
            TSTART=data_header['TSTART']
            TSTOP=data_header['TSTOP']
            MJDREFF=data_header['MJDREFF']   #defining the fits header data from the raw fits file (since our combined cyg12 data doesnt have a header i dont think
            MJDREFI=data_header['MJDREFI']
            MJD_ref_day=MJDREFF+MJDREFI
            curve_duration=TSTOP-TSTART                  

            # r cut index
            r_sqrd=abs((data_12.field('X')-x_0)**2-(data_12.field('Y')-y_0)**2)  #r cut just like before now i is each mod angle select fits file
            r_sqrd_3=abs((data_3.field('X')-x_0)**2-(data_3.field('Y')-y_0)**2) 
            index_r=[j<r_0 for j in r_sqrd]
            index_r_3=[m<r_0 for m in r_sqrd_3]
            data_12=data_12[index_r] #indexing photon dataset
            data_3=data_3[index_r_3]
            #eff_mod_angle_12=eff_mod_angle_12[index_r] #indexing mod angle dataset


            #PI channel/energy index
            index_energy=list(locate(data_12.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
            data_12=data_12[index_energy]
            index_energy_3=list(locate(data_3.field('PI'), lambda x: Pmin < x < Pmax))
            data_3=data_3[index_energy_3]
            #eff_mod_angle_12=eff_mod_angle_12[index_energy]


            TIME=data_12.field('TIME')
            TIME_3=data_3.field('TIME')
           
            #Lightcurves for norm

            lightcurve_12=Lightcurve.make_lightcurve(TIME,dt=bin_length,tseg=curve_duration,tstart=TSTART,gti=GTI)
            lightcurve_12.apply_gtis()

            lightcurve_3=Lightcurve.make_lightcurve(TIME_3,dt=bin_length,tseg=curve_duration,tstart=TSTART,gti=GTI)
            lightcurve_3.apply_gtis()

            #Cross spec for norm

            avg_cs = AveragedCrossspectrum.from_lightcurve(lightcurve_12,lightcurve_3,seg_length,norm='frac')
       
     
            norm_power_real=avg_cs.power.real  #cross spec properties
            #print(norm_power_real)
            norm_power_im=avg_cs.power.imag
            norm_freq=avg_cs.freq
          
            av_power_norm_array=[]    
            av_power_im_norm_array=[]
            #make a pandas dataframe for  power and frequency
            #real
            norm_d = {'all_power': np.array(norm_power_real), 'all_fourier_freq': np.array(norm_freq)} #total pwr and freq in dataset
            df_norm = pd.DataFrame(data=norm_d)
            selected_rows_norm = df_norm[(df_norm['all_fourier_freq'] >= fmin) & (df_norm['all_fourier_freq'] <= fmax)] #selecting freq range
      
            av_power_norm=selected_rows_norm['all_power'].mean() #calculating mean pwr
            av_power_norm_array.append(av_power_norm)
            np.savetxt('Results/allmod_av_real_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_norm_array)
            
            #im
            norm_d_im = {'all_power_im': np.array(norm_power_im), 'all_fourier_freq': np.array(norm_freq)} #total pwr and freq in dataset
            df_norm_im = pd.DataFrame(data=norm_d_im)
            selected_rows_norm_im = df_norm_im[(df_norm_im['all_fourier_freq'] >= fmin) & (df_norm_im['all_fourier_freq'] <= fmax)] #selecting freq range
      
            av_power_norm_im=selected_rows_norm_im['all_power_im'].mean() #calculating mean pwr
            av_power_im_norm_array.append(av_power_norm_im)
            np.savetxt('Results/allmod_av_im_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_im_norm_array)



            #calculating standard error on the mean
            sem_real_norm=np.std(selected_rows_norm['all_power'], ddof=1) / np.sqrt((np.size(selected_rows_norm['all_power'])))
            err_array_norm_real.append(sem_real_norm)
            np.savetxt('Results/allmod_av_real_err_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_norm_real) #one of these comes out for every run
             

            sem_im_norm=np.std(selected_rows_norm_im['all_power_im'], ddof=1) / np.sqrt((np.size(selected_rows_norm_im['all_power_im'])))
            err_array_norm_im.append(sem_im_norm)
            np.savetxt('Results/allmod_av_im_err_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_norm_im) #one of these comes out for every run
             







            #calculating normalisation constant
            norm_factor=(np.sqrt((fmax-fmin))/np.sqrt(av_power_norm))
            norm_factor_array.append(norm_factor) 
            np.savetxt('Results/norm_cs_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',norm_factor_array)

            #make reference lightcurve
            TIME_ref=data_3['TIME']   #defining the ref lightcurve data
            lc_ref=Lightcurve.make_lightcurve(TIME_ref,dt=bin_length,tseg=curve_duration,tstart=TSTART,mjdref=MJD_ref_day,gti=GTI)
            lc_ref.apply_gtis()
           
            #making a list of mod angle bins to select over
            mod_minimum=np.radians(-90)
            mod_maximum=np.radians(90)
            aspace=np.linspace(mod_minimum,mod_maximum,mod_bin_number+1)
            mod_angle_list=[(aspace[i-1],aspace[i]) for i in range(len(aspace))]  #making a list of mod angle bins to select over
            mod_angle_list.pop(0) #removing the dodger first one
    
    
    
    for i in mod_angle_list:
        mod_min=i[0]
        mod_max=i[1]
     
        for file_12 in glob.iglob('Lightcurves/lc_12_'+str(source_name)+'_'+str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_'+'.fits'): #lc file name  

            with fits.open(file_12) as hdu1:
                data_cut=hdu1[1].data  #reading in each mod angle selected file
                TIME=data_cut['TIME']
                
                #making subject lightcurve
                lc=Lightcurve.make_lightcurve(TIME,dt=bin_length,tseg=curve_duration,tstart=TSTART,mjdref=MJD_ref_day,gti=GTI)
                lc.apply_gtis()
               
                #making averagd cross spectrum
                cs = AveragedCrossspectrum.from_lightcurve(lc,lc_ref,seg_length,norm='frac') #making averagd cross spec
                cs_log=cs.rebin_log()  #logarithmic rebinning xo
     
                fourier_f=cs.freq
                real_power=cs.power.real 
                im_power=cs.power.imag


            #Plotting the real part against fourier frequency
            
         #       fig, ax1 = plt.subplots(1,1,figsize=(9,6))
                #ax1.plot(cs.freq, cs.power, color='blue',label='no log rebin')
         #       ax1.plot(cs.freq, real_power, color='green')
                #ax1.plot(avg_ps_log.freq, avg_ps_log.power, color='red',label='log rebin')
         #       ax1.set_xlabel("Frequency (Hz)")
         #       ax1.set_title('Real Power over fourier frequency')
         #       ax1.set_ylabel("Power")
         #       ax1.set_yscale('log')
         #       ax1.set_xscale('log')
         #       ax1.tick_params(axis='x', labelsize=16)
         #       ax1.tick_params(axis='y', labelsize=16)
         #       ax1.tick_params(which='major', width=1.5, length=7)
         #       ax1.tick_params(which='minor', width=1.5, length=4)
         #       for axis in ['top', 'bottom', 'left', 'right']:
         #           ax1.spines[axis].set_linewidth(1.5)
                #plt.show()



            #Plotting the imaginary part against fourier frequency
      #          fig, ax1 = plt.subplots(1,1,figsize=(9,6))
                #ax1.plot(cs.freq, cs.power, color='blue',label='no log rebin')
      #          ax1.plot(cs.freq, im_power, color='green')
                #ax1.plot(avg_ps_log.freq, avg_ps_log.power, color='red',label='log rebin')
       #         ax1.set_xlabel("Frequency (Hz)")
       #         ax1.set_title('Imaginary Power over fourier frequency')
       #         ax1.set_ylabel("Power")
       #         ax1.set_yscale('log')
       #         ax1.set_xscale('log')
       #         ax1.tick_params(axis='x', labelsize=16)
       #         ax1.tick_params(axis='y', labelsize=16)
       #         ax1.tick_params(which='major', width=1.5, length=7)
       #         ax1.tick_params(which='minor', width=1.5, length=4)
       #         for axis in ['top', 'bottom', 'left', 'right']:
       #             ax1.spines[axis].set_linewidth(1.5)
                #plt.show()


                #Average real power over frequency range
                d_real = {'real_power': np.array(real_power), 'fourier_freq': np.array(cs.freq)}
                df_real = pd.DataFrame(data=d_real)
                selected_rows_real = df_real[(df_real['fourier_freq'] >= fmin) & (df_real['fourier_freq'] <= fmax)]
                av_power_real=selected_rows_real['real_power'].mean()
                av_power_array_real.append(av_power_real)
                np.savetxt('Results/G_av_real_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_real)
               
                #Standard error on the average real power
                sem_real=np.std(selected_rows_real['real_power'], ddof=1) / np.sqrt((np.size(selected_rows_real['real_power'])))
                err_array_real.append(sem_real)
                np.savetxt('Results/G_av_real_err_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_real) #one of these comes out for every run
             

                #Average imaginary power over frequency
                d_im = {'im_power': np.array(im_power), 'fourier_freq': np.array(cs.freq)}
                df_im = pd.DataFrame(data=d_im)
                selected_rows_im = df_im[(df_im['fourier_freq'] >= fmin) & (df_im['fourier_freq'] <= fmax)]
                av_power_im=selected_rows_im['im_power'].mean()
                av_power_array_im.append(av_power_im)
                np.savetxt('Results/G_av_im_'+str(source_name)+'_'str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_im)
            

                #Standard error on the average imaginary power
                sem_im=np.std(selected_rows_im['im_power'], ddof=1) / np.sqrt((np.size(selected_rows_im['im_power'])))
                err_array_im.append(sem_im)
                np.savetxt('Results/G_av_im_err_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_im) #one of these comes out for every run
                
                
                
                
                
        #np.savetxt('norm_cs_'+str(mod_bin_number)+'_bins_'+'freqs_'+str(fmin)+str(fmax)+'.txt',norm_factor)        
                
                

def mod_angle_cross_spec_ith(gti,bin_length,seg_length,Pmin,Pmax,fmin,fmax,mod_bin_number):
   
    #power arrays
    av_power_array_real=[]
    av_power_array_im=[] 
    
    #error arrays
    err_array_im=[]
    err_array_real=[]
    
    GTI=list(np.loadtxt(str(gti)))
    
    #making mod angle list
    mod_minimum=np.radians(-90)
    mod_maximum=np.radians(90)
    aspace=np.linspace(mod_minimum,mod_maximum,mod_bin_number+1)
    mod_angle_list=[(aspace[i-1],aspace[i]) for i in range(len(aspace))]  #making a list of mod angle bins to select over
    mod_angle_list.pop(0) #removing the dodger first one
  #  print('mod list made')    
  #  with fits.open(header_file) as hduh:
  #      data_header=hduh[1].header
  #      TSTART=data_header['TSTART']
  #      TSTOP=data_header['TSTOP']
  #      MJDREFF=data_header['MJDREFF']   #defining the modulation angle selected fits data
  #      MJDREFI=data_header['MJDREFI']
  #      MJD_ref_day=MJDREFF+MJDREFI
  #      curve_duration=TSTOP-TSTART
  #      print('header read in')
    for i in mod_angle_list:
        mod_min=i[0]
        mod_max=i[1]
       
        for file_12 in glob.iglob('Lightcurves/lc_12_'+str(source_name)+'_'+str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_'+'.fits'): #lc file name
            with fits.open(file_12) as hdu1:
                data_12=hdu1[1].data
               # print(data_12)
          
        for ref_curve_data in glob.iglob('Lightcurves/lc_3_'+str(source_name)+'_'str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_'+'.fits'): 
            with fits.open(str(ref_curve_data)) as hdu2:
                data_ref=hdu2[1].data
         

                TIME=data_12['TIME']
                TIME_ref=data_ref['TIME']   #defining the ref lightcurve data
                   
                #12 lightcurve
                lc=Lightcurve.make_lightcurve(TIME,dt=bin_length,gti=GTI)
                lc.apply_gtis()
                print(lc.counts)           

                #3 lightcurve (ref)
                lc_ref=Lightcurve.make_lightcurve(TIME_ref,dt=bin_length,gti=GTI)
                lc_ref.apply_gtis()
                
                #ith to ith cross spectrum
                cs = AveragedCrossspectrum.from_lightcurve(lc,lc_ref,seg_length,norm='frac') #making averagd cross spec
                cs_log=cs.rebin_log()  #logarithmic rebinning xo

                
                #wanna see cs?
                
                #fig, ax1 = plt.subplots(1,1,figsize=(9,6))
                ##ax1.plot(cs.freq, cs.power, color='blue',label='no log rebin')
                #ax1.plot(cs.freq, cs.power, color='red')
                ##ax1.plot(avg_ps_log.freq, avg_ps_log.power, color='red',label='log rebin')
                #ax1.set_xlabel("Frequency (Hz)")
                #ax1.set_title('Power over fourier frequency')
                #ax1.set_ylabel("Power")
                #ax1.set_yscale('log')
                #ax1.set_xscale('log')
                #ax1.tick_params(axis='x', labelsize=16)
                #ax1.tick_params(axis='y', labelsize=16)
                #ax1.tick_params(which='major', width=1.5, length=7)
                #ax1.tick_params(which='minor', width=1.5, length=4)
                #plt.legend()
                #for axis in ['top', 'bottom', 'left', 'right']:
                #    ax1.spines[axis].set_linewidth(1.5)
                #plt.show()

                #Averaging real part
                d_real = {'power_real': np.array(cs.power.real), 'fourier_freq': np.array(cs.freq)}
                df_real = pd.DataFrame(data=d_real)
                selected_rows_real = df_real[(df_real['fourier_freq'] >= fmin) & (df_real['fourier_freq'] <= fmax)]
                av_power_real=selected_rows_real['power_real'].mean()
                av_power_array_real.append(av_power_real)
                np.savetxt('Results/cs_ith_av_real_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_real)
               
                #errors on real part
                sem_real=np.std(selected_rows_real['power_real'], ddof=1) / np.sqrt((np.size(selected_rows_real['power_real'])))
                err_array_real.append(sem_real)
                np.savetxt('Results/cs_ith_av_real_err_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_real)
        
        
        
                #Averaging im part
                d_im = {'power_im': np.array(cs.power.imag), 'fourier_freq': np.array(cs.freq)}
                df_im = pd.DataFrame(data=d_im)
                selected_rows_im = df_im[(df_im['fourier_freq'] >= fmin) & (df_im['fourier_freq'] <= fmax)]
                av_power_im=selected_rows_im['power_im'].mean()
                av_power_array_im.append(av_power_im)
                np.savetxt('Results/cs_ith_av_im_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_im)

                #errors on im part
                sem_im=np.std(selected_rows_im['power_im'], ddof=1) / np.sqrt((np.size(selected_rows_im['power_im'])))
                err_array_im.append(sem_im)
                np.savetxt('Results/cs_ith_av_im_err_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_im)
                

                
def results(mod_bin_number,fmin,fmax,bin_length,seg_length):

    #mod_min=np.radians(-90)
    #print(mod_min)
    #mod_max=np.radians(90)
    #print(mod_max)
    mod_min_global=np.radians(-90)
    mod_max_global=np.radians(90)
    #mod_bin_num=50
    a=np.linspace(mod_min_global,mod_max_global,mod_bin_number+1)
    mod_angle_list=[(a[i-1],a[i]) for i in range(len(a))] #making a list of mod angle bins to select over
    mod_angle_list.pop(0) #removing the dodger first one
    #print(np.degrees(mod_angle_list))
    #print(np.degrees(mod_angle_list))

    #print(np.degrees(mod_angle_list[26]))
    #print(mod_angle_list[24])

    av_mod=[np.mean(i) for i in mod_angle_list]

    #-------------------------------------------------------------------------------------

    normalisation_factor=np.loadtxt('Results/norm_cs_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')

    norm_factor_list=[np.sqrt(1/(normalisation_factor**2))]*len(av_mod)
    #123each
#Results/cs_ith_av_real'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.t
    each_real=np.loadtxt('Results/cs_ith_av_real_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    each_im=np.loadtxt('Results/cs_ith_av_im_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
  
 eacherr_power_real=np.loadtxt('Results/cs_ith_av_real_err_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    eacherr_power_im=np.loadtxt('Results/cs_ith_av_im_err_'+str(source_name)+'_'str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    
    #123all
    all_im=np.loadtxt('Results/allmod_av_im_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    all_real=np.loadtxt('Results/allmod_av_real_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    all_im_err=np.loadtxt('Results/allmod_av_im_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    all_real_err=np.loadtxt('Results/allmod_av_real_err_'+str(source_name)+'_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
   

    #G
    newG_im=np.loadtxt('Results/G_av_im_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real=np.loadtxt('Results/G_av_real_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im=np.loadtxt('Results/G_av_im_err_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real=np.loadtxt('Results/G_av_real_err_'+str(source_name)+'_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
  


   #coherence

    mag_G_sqrd=abs((newG_real)**2+(newG_im)**2)
    coherence=mag_G_sqrd/(all_real*each_real)


    #plt.figure()
    #plt.title('ith to ith bin')
    #plt.legend()
    #plt.plot(np.degrees(av_mod),each_real)
    #plt.plot(np.degrees(av_mod),each_im)
    #plt.errorbar(np.degrees(av_mod),each_im,yerr=eacherr_power_im,label='each_im')
    #plt.errorbar(np.degrees(av_mod),each_real,yerr=eacherr_power_real,label='each_real')
    #plt.legend()
 

    gs=gridspec.GridSpec(2,2)
    pl.figure()
    ax=pl.subplot(gs[0,0])
   # fig,axs=plt.subplots(2,2)
   # fig.tight_layout()
   # axs[0,0].set_title('Fractional rms')
   # axs[0,0].plot(av_mod,newG_real)
   # axs[0,0].errorbar(av_mod,newG_real,yerr=newG_err_real,label='G_real')
    frac_norm=newG_real*normalisation_factor
    pl.plot(av_mod,frac_norm,'.')
    pl.plot(av_mod,norm_factor_list)
    pl.errorbar(av_mod,newG_real*normalisation_factor,yerr=newG_err_real,label='G_real')
    pl.title('Fractional rms')
    pl.xlabel('Modulation angle (radians)')
    pl.ylabel('Fractional rms')                         
    ax=pl.subplot(gs[0,1])
   # axs[0,1].set_title('Phase (radians)')
   # axs[0,1].plot(av_mod,newG_im)
   # axs[0,1].errorbar(av_mod,newG_im,yerr=newG_err_im,label='G_im')
    pl.plot(av_mod,newG_im)
    pl.errorbar(av_mod,newG_im,yerr=newG_err_im,label='G_im')

    pl.title('Phase (radians)')
    pl.xlabel('Modulation angle (radians)')
   # axs[1,0].set_title('Coherence')
   # axs[1,0].plot(av_mod,coherence,'.')
    #plt.errorbar(av_mod,coherence_50,yerr=err_coherence)
    #plt.show()
    ax=pl.subplot(gs[1,:])
    pl.plot(av_mod,coherence,'.')
    pl.title('Coherence')
    pl.ylabel('$\gamma^2$')
    pl.xlabel('Modulation Angle (radians)')
    pl.show()



def combo_results(bins1,bins2,bins3,fmin,fmax,bin_length,seg_length):

    normalisation_factor=np.loadtxt('Results/norm_cs_'+'freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')


    #G1
    newG_im=np.loadtxt('Results/G_av_im_'+str(source_name)+'_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real=np.loadtxt('Results/G_av_real_'+str(source_name)+'_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im=np.loadtxt('Results/G_av_im_err_'+str(source_name)+'_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real=np.loadtxt('Results/G_av_real_err_'+str(source_name)+'_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
  
    #G2
    newG_im_2=np.loadtxt('Results/G_av_im_'+str(source_name)+'_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real_2=np.loadtxt('Results/G_av_real_'+str(source_name)+'_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im_2=np.loadtxt('Results/G_av_im_err_'+str(source_name)+'_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real_2=np.loadtxt('Results/G_av_real_err_'+str(source_name)+'_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    #G3
    newG_im_3=np.loadtxt('Results/G_av_im_'+str(source_name)+'_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real_3=np.loadtxt('Results/G_av_real_'+str(source_name)+'_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im_3=np.loadtxt('Results/G_av_im_err_'+str(source_name)+'_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real_3=np.loadtxt('Results/G_av_real_err_'+str(source_name)+'_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')



    mod_min_global=np.radians(-90)
    mod_max_global=np.radians(90)
  
    a=np.linspace(mod_min_global,mod_max_global,bins1+1)
    mod_angle_list_1=[(a[i-1],a[i]) for i in range(len(a))] #making a list of mod>
    mod_angle_list_1.pop(0) #removing the dodger first one
    av_mod_1=[np.mean(i) for i in mod_angle_list_1]
    
      
    b=np.linspace(mod_min_global,mod_max_global,bins2+1)
    mod_angle_list_2=[(b[i-1],b[i]) for i in range(len(b))] #making a list of m>
    mod_angle_list_2.pop(0) #removing the dodger first one
    av_mod_2=[np.mean(i) for i in mod_angle_list_2]

    
      
    c=np.linspace(mod_min_global,mod_max_global,bins3+1)
    mod_angle_list_3=[(c[i-1],c[i]) for i in range(len(c))] #making a list of m>
    mod_angle_list_3.pop(0) #removing the dodger first one
    av_mod_3=[np.mean(i) for i in mod_angle_list_3]
    
    

    norm_factor_list_1=[np.sqrt(1/(normalisation_factor**2))]*len(av_mod_1)
    norm_factor_list_2=[np.sqrt(1/(normalisation_factor**2))]*len(av_mod_2)
    norm_factor_list_3=[np.sqrt(1/(normalisation_factor**2))]*len(av_mod_3)


    gs=gridspec.GridSpec(2,2)
    pl.figure()
    ax=pl.subplot(gs[0,0])
  
    frac_norm=newG_real*normalisation_factor
    pl.plot(av_mod_1,frac_norm,'.')
    pl.plot(av_mod_1,norm_factor_list_1)
    pl.errorbar(av_mod_1,newG_real*normalisation_factor,yerr=newG_err_real)

    frac_norm_2=newG_real_2*normalisation_factor
    pl.plot(av_mod_2,frac_norm_2,'.')
    pl.plot(av_mod_2,norm_factor_list_2)
    pl.errorbar(av_mod_2,newG_real_2*normalisation_factor,yerr=newG_err_real_2)


    frac_norm_3=newG_real_3*normalisation_factor
    pl.plot(av_mod_3,frac_norm_3,'.')
    pl.plot(av_mod_3,norm_factor_list_3)
    pl.errorbar(av_mod_3,newG_real_3*normalisation_factor,yerr=newG_err_real_3)







    pl.title('Fractional rms')
    pl.xlabel('Modulation angle (radians)')

    pl.show()


def time_lag_high_low(det12fits,det3fits,gti,eff_mod_12,eff_mod_3):
    
    GTI=list(np.loadtxt(str(gti)))
    
    #det 1+2 
    with fits.open(str(det12fits)) as hdu:
        data_12=hdu[1].data  #reading in the file we will use the main data, DU1+DU2
        eff_mod_angle_12=np.loadtxt(str(eff_mod_12)) #loading in mod angles of combined DU1+DU2
    
    #det 3
    with fits.open(str(det3fits)) as hdu2:
        data_3=hdu2[1].data #reading in the file we will use the header of (the header data we need here is the same for all raw fits so any will do)
        data_3_header=hdu2[1].header
        eff_mod_angle_3=np.loadtxt(str(eff_mod_3))
        
       
        
        # r cut 12
        r_sqrd=abs((data_12.field('X')-x_0)**2-(data_12.field('Y')-y_0)**2)    #r cut just like before now i is each mod angle select fits file
        r_0=30.769326 
        index_r=[j<r_0 for j in r_sqrd]
        data_12=data_12[index_r] #indexing photon dataset
        eff_mod_angle_12=eff_mod_angle_12[index_r] #indexing mod angle dataset
  
        #r cut 3
    
        r_sqrd_3=abs((data_3.field('X')-x_0)**2-(data_3.field('Y')-y_0)**2)    #r cut just like before now i is each mod angle select fits file
        r_0=30.769326 
        index_r_3=[k<r_0 for k in r_sqrd_3]
        data_3=data_3[index_r_3] #indexing photon dataset
        eff_mod_angle_3=eff_mod_angle_3[index_r_3] #indexing mod angle dataset
  
    
        #PI channel/energy index 12
    
        index_energy=list(locate(data_12.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
        data_12=data_12[index_energy]
        eff_mod_angle_12=eff_mod_angle_12[index_energy]
        
        #PI cut for detector 3
        
        index_energy_3=list(locate(data_3.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
        data_3=data_3[index_energy_3]
        eff_mod_angle_3=eff_mod_angle_3[index_energy_3]
        
        #plotting the modulation curve
        
       # num_bins = 1000
       # counts, bin_edges = np.histogram(eff_mod_angle_3, bins=num_bins)
       # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
       # plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]))
       # plt.xlabel('Modulation Angle')
       # plt.ylabel('Counts')
       # plt.title('Histogram of Counts over Modulation Angle')
       # plt.show()

        
        
        #plt.figure()
        #plt.plot(eff_mod_angle_3,)
        #plt.show()
        
        
        # indexing 12 photons 
        theta=np.radians(-18)
       
    
 #       mod_bin_index_a=list(locate(eff_mod_angle_12, lambda q: q>theta+(np.pi/4)))
 #      # mod_bin_index_b=list(locate(eff_mod_angle_12, lambda y: y<theta-(np.pi/4)))
 #       #mod_bin_index=[mod_12 >theta+(np.pi)/4 and mod_12<theta-np.pi/4 for mod_12 in eff_mod_angle_12]
 #       selected_mod_angle_12_a=eff_mod_angle_12[mod_bin_index_a]
 #       #selected_mod_angle_a=eff_m0d_angle_12[mod_bin_index_a]
 #       selected_data_12_a=data_12[mod_bin_index_a]
 #       
 #       
 #       mod_bin_index_b=list(locate(selected_mod_angle_12_a, lambda y: y<theta-(np.pi/4)))
 #       
 #       selected_mod_angle_12_b=selected_mod_angle_12_a[mod_bin_index_b]
  #      selected_data_12_b=selected_data_12_a[mod_bin_index_b]
        
        #print(len(selected_data_12))
        
        
        mod_bin_index_12=list(locate(eff_mod_angle_12, lambda q: q>theta+(np.pi/4) or q<theta-(np.pi/4))) 
        #selected_mod_angle_12_or=selected_mod_angle_12[mod_bin_index_12]
        selected_data_12_or=data_12[mod_bin_index_12]
        
        
        
        #indexing 3 photons
        #theta=1
        mod_bin_index_3=[mod_3>theta-(np.pi)/4 and mod_3<theta+np.pi/4 for mod_3 in eff_mod_angle_3]
        selected_mod_angle_3=eff_mod_angle_3[mod_bin_index_3]
        selected_data_3=data_3[mod_bin_index_3]
        
        
        lc_12=Lightcurve.make_lightcurve(selected_data_12_or['TIME'],dt=1/256,gti=GTI)
        lc_12.apply_gtis()
        
        lc_3=Lightcurve.make_lightcurve(selected_data_3['TIME'],dt=1/256,gti=GTI)
        lc_3.apply_gtis()
        
        
        cs=AveragedCrossspectrum.from_lightcurve(lc_12,lc_3,20.48,norm='frac')
        cs=cs.rebin_log(f=1.0)
        
        
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.title('REAL')
        plt.plot(cs.freq,cs.power.real,'.')
       # plt.show()
        
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.title('IMAG')
        plt.plot(cs.freq,cs.power.imag,'.')
       # plt.show()
        
        power=cs.power
        print(power)
        
        t_lag_array=[]
        arg_array=[]
        for complex_val in power:
            arg= np.angle(complex_val)
            arg_array.append(arg)
           # real=complex_val.real
           # print(real)
           # imag=complex_val.imag
           # print(imag)
           # arg=math.atan2(imag,real)
           # t_lag=arg/2*np.pi
           # t_lag_array.append(t_lag)
        t_arg=arg_array/(2*np.pi*cs.freq)
       # time_lag=t_lag_array/cs.freq        
        freq_lags, freq_lags_err = cs.time_lag()        
       # t_lag_auto = cs.time_lag()
        plt.figure()
        plt.plot(cs.freq,t_arg,'.')
        plt.errorbar(cs.freq,freq_lags,yerr=freq_lags_err)
       # plt.plot(cs.freq,time_lag,'.',label='my lag')
       # plt.plot(cs.freq,t_lag_auto[0],'.')
        plt.xscale('log')
       # plt.yscale('log')
        plt.title('t_lag')
        plt.xlabel('Frequency')
        plt.ylabel('Time lag (s)')
        
    
        
       # t_lag_auto = cs.time_lag()
       # plt.plot(cs.freq,t_lag_auto[0],'.',label='auto lag')
       # plt.legend()
        plt.show()



def time_lag_high_low_av(det12fits,det3fits,eff_mod_12,eff_mod_3):
    
    GTI=list(np.loadtxt(str(gti)))
    
    with fits.open(str(det12fits)) as hdu:
        data_12=hdu[1].data  #reading in the file we will use the main data, DU1+DU2
        eff_mod_angle_12=np.loadtxt(str(eff_mod_12)) #loading in mod angles of combined DU1+DU2
    
    #det 3
    with fits.open(str(det3fits)) as hdu2:
        data_3=hdu2[1].data #reading in the file we will use the header of (the header data we need here is the same for all raw fits so any will do)
        data_3_header=hdu2[1].header
        eff_mod_angle_3=np.loadtxt(str(eff_mod_3))

      
        # r cut 12
        r_sqrd=abs((data_12.field('X')-x_0)**2-(data_12.field('Y')-y_0)**2)    #r cut just like before now i is each mod angle select fits file
        r_0=30.769326 
        index_r=[j<r_0 for j in r_sqrd]
        data_12=data_12[index_r] #indexing photon dataset
        eff_mod_angle_12=eff_mod_angle_12[index_r] #indexing mod angle dataset
  
        #r cut 3
    
        r_sqrd_3=abs((data_3.field('X')-x_0)**2-(data_3.field('Y')-y_0)**2)    #r cut just like before now i is each mod angle select fits file
        r_0=30.769326 
        index_r_3=[k<r_0 for k in r_sqrd_3]
        data_3=data_3[index_r_3] #indexing photon dataset
        eff_mod_angle_3=eff_mod_angle_3[index_r_3] #indexing mod angle dataset
  
    
        #PI channel/energy index 12
    
        index_energy=list(locate(data_12.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
        data_12=data_12[index_energy]
        eff_mod_angle_12=eff_mod_angle_12[index_energy]

        #PI cut for detector 3

        index_energy_3=list(locate(data_3.field('PI'), lambda x: Pmin < x < Pmax))  #energy cut just like before
        data_3=data_3[index_energy_3]
        eff_mod_angle_3=eff_mod_angle_3[index_energy_3]


#---------------------------------------------------------------------------------------------
        # indexing 12 photons 
        theta=np.radians(-18)


        mod_bin_index_12=list(locate(eff_mod_angle_12, lambda q: q>theta+(np.pi/4) or q<theta-(np.pi/4))) 
        #selected_mod_angle_12_or=selected_mod_angle_12[mod_bin_index_12]
        selected_data_12_or=data_12[mod_bin_index_12]



        #indexing 3 photons
        #theta=1
        mod_bin_index_3=[mod_3>theta-(np.pi)/4 and mod_3<theta+np.pi/4 for mod_3 in eff_mod_angle_3]
        selected_mod_angle_3=eff_mod_angle_3[mod_bin_index_3]
        selected_data_3=data_3[mod_bin_index_3]


        lc_12=Lightcurve.make_lightcurve(selected_data_12_or['TIME'],dt=1/256,gti=GTI)
        lc_12.apply_gtis()

        lc_3=Lightcurve.make_lightcurve(selected_data_3['TIME'],dt=1/256,gti=GTI)
        lc_3.apply_gtis()


        cs=AveragedCrossspectrum.from_lightcurve(lc_12,lc_3,20.48,norm='frac')
        cs=cs.rebin_log(f=1.0)
        
#----------------------------------------------------------------------------------------------------
#Now the opposite way to average over 


        #indexing 3 photons
        mod_bin_index_3_2=list(locate(eff_mod_angle_3, lambda l: l>theta+(np.pi/4) or l<theta-(np.pi/4))) 
        #selected_mod_angle_12_or=selected_mod_angle_12[mod_bin_index_12]
        selected_data_3_2=data_3[mod_bin_index_3_2]



        #indexing 12 photons
        #theta=1
        mod_bin_index_12_2=[mod_12_2>theta-(np.pi)/4 and mod_12_2<theta+np.pi/4 for mod_12_2 in eff_mod_angle_12]
        selected_data_12_2=data_12[mod_bin_index_12_2]


        lc_12_2=Lightcurve.make_lightcurve(selected_data_12_2['TIME'],dt=1/256,gti=GTI)
        lc_12_2.apply_gtis()

        lc_3_2=Lightcurve.make_lightcurve(selected_data_3_2['TIME'],dt=1/256,gti=GTI)
        lc_3_2.apply_gtis()


        cs_2=AveragedCrossspectrum.from_lightcurve(lc_12_2,lc_3_2,20.48,norm='frac')
        cs_2=cs_2.rebin_log(f=1.0)

        power_2=cs_2.power
        #print(power)

        t_lag_array_2=[]
        arg_array_2=[]
        for complex_val_2 in power_2:
            arg_2= np.angle(complex_val_2)
            arg_array_2.append(arg_2)
        t_arg_2=arg_array_2/(2*np.pi*cs_2.freq)
        freq_lags_2, freq_lags_err_2 = cs_2.time_lag() #to be used for error for now (before i calc the ensemble av err associated with the cs)       
        plt.figure()

        plt.plot(cs_2.freq,t_arg_2,'.')
        plt.title('2')
        plt.errorbar(cs_2.freq,freq_lags_2,yerr=freq_lags_err_2)
        plt.xscale('log')
       # plt.title('t_lag')
        plt.xlabel('Frequency')
        plt.ylabel('Time lag (s)')
        plt.show()



#-----------------------------------------------------------------------------------------------------------

        power=cs.power
        print(power)

        t_lag_array=[]
        arg_array=[]
        for complex_val in power:
            arg= np.angle(complex_val)
            arg_array.append(arg)
        t_arg=arg_array/(2*np.pi*cs.freq)
        freq_lags, freq_lags_err = cs.time_lag()        
        plt.figure()

        plt.plot(cs.freq,t_arg,'.')
        
        plt.errorbar(cs.freq,freq_lags,yerr=freq_lags_err)
        plt.xscale('log')
        plt.title('1')
        plt.xlabel('Frequency')
        plt.ylabel('Time lag (s)')
        plt.show()

#-----------------------------------------------------------------
#------------------------------------------------------------------------------
#Now for averaging

       # t_lag_av=np.average(np.array(t_arg),np.array(t_arg_2))
        t_lag_av=(t_arg+t_arg_2)/2
       # t_arg_err_av=np.average(np.array(freqs_lags_err),np.array(freqs_lags))
        t_arg_err_av=(freq_lags_err+freq_lags_err_2)/2
        plt.figure()
        plt.plot(cs.freq,t_lag_av,'.')
        plt.title('av')
        plt.errorbar(cs.freq,t_lag_av,yerr=t_arg_err_av)
        plt.xscale('log')
       # plt.title('t_lag')
        plt.xlabel('Frequency')
        plt.ylabel('Time lag (s)')
        plt.show()

