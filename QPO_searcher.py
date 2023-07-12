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







#GTI=[(169402263.0, 169405703.512488), (169405708.193735, 169405928.0), (169408049.0, 169408120.332665), (169408120.471237, 169408208.295056), (169408213.000139, 169408318.32), (169408378.323, 169411714.0), (169413834.0, 169414228.7420001), (169414230.04200006, 169414272.531237), (169414272.571104, 169414498.48), (169414558.48, 169417499.0), (169419620.0, 169419958.62), (169420018.62, 169420050.0), (169420052.62, 169420061.527), (169420064.98, 169420597.0), (169420599.983, 169420601.0), (169420603.0, 169420679.637), (169420682.0, 169420690.543), (169420695.0, 169420708.64), (169420768.64, 169421068.65), (169421128.65, 169421533.79699993), (169421535.09699988, 169421541.604726), (169421541.669245, 169421818.667), (169421878.67, 169423285.0), (169425568.763, 169426168.683), (169426228.78, 169426650.0), (169426653.03, 169426654.0), (169426657.0, 169426749.8369999), (169426751.13700008, 169426756.279274), (169426756.313549, 169426824.370576), (169426830.711964, 169426918.797), (169426978.797, 169429071.0), (169431748.92, 169432348.933), (169432408.937, 169433128.953), (169433188.957, 169434857.0), (169436978.0, 169437179.057), (169437959.077, 169440643.0), (169442764.0, 169443389.213), (169444107.233, 169444108.137), (169444169.233, 169444769.25), (169444829.25, 169445323.9749999), (169445325.2750001, 169445519.267), (169445579.27, 169446429.0), (169448550.0, 169449599.277), (169450349.39, 169450894.826919), (169450902.786284, 169452215.0), (169454336.0, 169455809.433), (169456559.547, 169457159.563), (169457219.563, 169457329.473), (169457333.0, 169457334.473), (169457337.253, 169457381.473), (169457385.0, 169457386.473), (169457390.0, 169457851.06500006), (169457852.365, 169457939.583), (169457999.583, 169458001.0), (169460122.0, 169460488.0), (169460491.481757, 169461414.005146), (169461414.072369, 169462019.687), (169462769.703, 169463787.0), (169465908.0, 169468229.843), (169468979.86, 169469573.0), (169471694.0, 169472383.810444), (169472383.950761, 169473247.470367), (169473253.654601, 169473872.834047), (169473873.01679, 169474440.0), (169475160.017, 169475359.0), (169477480.0, 169480323.560135), (169480323.638249, 169480647.157), (169483266.0, 169484839.362372), (169484845.818273, 169486830.313), (169489051.0, 169492716.0), (169494837.0, 169495140.523), (169495200.523, 169495767.945608), (169495773.598179, 169496928.945826), (169497182.409367, 169498502.0), (169500660.663, 169501350.68), (169501410.68, 169504288.0), (169506409.0, 169506561.792343), (169513633.369607, 169513770.997), (169513830.997, 169514900.0), (169514903.685173, 169515860.0), (169518601.12, 169519201.133), (169519261.137, 169519663.52900004), (169519664.829, 169519739.551417), (169519739.682221, 169519951.153), (169520011.153, 169521645.0), (169523767.0, 169524031.257), (169524781.277, 169527431.0), (169529553.0, 169530241.413), (169530991.433, 169531591.45), (169531651.45, 169532159.6229999), (169532160.9230001, 169532371.47), (169532431.47, 169533055.0), (169533057.91383, 169533217.0), (169535339.0, 169536396.76305), (169536396.905443, 169536451.57), (169537140.59, 169537141.497), (169537201.59, 169539003.0), (169541124.0, 169542661.73), (169543381.747, 169544011.763), (169544071.763, 169544131.45129), (169544136.769236, 169544139.67), (169544143.0, 169544197.0), (169544199.767, 169544202.0), (169544286.917, 169544672.71600008), (169544674.01600003, 169544761.783), (169546910.0, 169547630.567274), (169547634.943411, 169548868.652133), (169549591.903, 169550341.7579999), (169550343.0580001, 169550575.0), (169552696.0, 169552918.641118), (169552924.316563, 169554081.932868), (169554082.17333, 169555052.043), (169555802.063, 169556361.0), (169558482.0, 169559652.273943), (169559657.36877, 169561262.2), (169561951.217, 169561952.123), (169562012.22, 169562147.0), (169564268.0, 169564968.315737), (169564968.541417, 169567249.017395), (169567249.274049, 169567472.26), (169570054.0, 169573424.543795), (169573424.804023, 169573652.513), (169575842.57, 169576521.579138), (169576526.659403, 169576983.091639), (169576987.763453, 169577980.403134), (169577986.435245, 169579212.0), (169579216.0, 169579504.0), (169581625.0, 169581991.0), (169582052.727, 169582859.584748), (169595385.962115, 169595511.0), (169595553.07, 169596862.0), (169599213.163, 169599813.18), (169599873.18, 169600154.982351), (169600161.515907, 169600299.0), (169600304.0, 169600379.14499998), (169600380.44499993, 169600593.2), (169600653.2, 169601206.992947), (169601207.111154, 169602412.692608), (169602412.819768, 169602647.0), (169605423.323, 169606023.337), (169606083.34, 169606490.194), (169606491.49399996, 169606590.284267), (169606590.427063, 169606803.357), (169606863.36, 169608433.0), (169610555.0, 169610853.46), (169611633.48, 169614219.0), (169616341.0, 169616761.344285), (169616767.066489, 169617063.62), (169617813.64, 169618438.653), (169618445.0, 169618773.0), (169618778.0, 169618995.2909999), (169618996.59100008, 169619000.340993), (169619000.412711, 169619193.673), (169619253.673, 169619658.957603), (169619663.63883, 169620005.0), (169622127.0, 169623273.777), (169624023.797, 169625791.0), (169641191.03246, 169641904.25), (169642624.267, 169643149.0), (169645270.0, 169648084.407), (169648834.427, 169648934.0), (169651357.719319, 169651379.031241), (169651385.518603, 169651390.265149), (169651428.375901, 169653010.0), (169653012.0, 169653023.0), (169653025.0, 169653060.0), (169653064.0, 169653078.0), (169653080.0, 169653084.0), (169653086.0, 169653261.0), (169653263.0, 169653302.0), (169653304.0, 169653321.0), (169653323.0, 169653364.0), (169653366.0, 169653451.0), (169653453.0, 169653462.0), (169653464.0, 169653470.0), (169653472.0, 169653476.0), (169653480.0, 169653484.0), (169653486.0, 169653491.0), (169653493.0, 169653496.0), (169653498.0, 169653519.0), (169653521.0, 169653530.0), (169653532.0, 169653534.0), (169653536.0, 169653548.0), (169653550.0, 169653555.0), (169653557.0, 169653560.0), (169653562.0, 169653573.0), (169653575.0, 169653598.0), (169653600.0, 169653604.0), (169653606.0, 169653608.0), (169653610.0, 169653624.0), (169653626.0, 169653654.0), (169653656.0, 169653660.0), (169653662.0, 169653754.0), (169653756.0, 169653757.0), (169653761.0, 169653772.0), (169653774.0, 169653813.0), (169653817.0, 169653820.0), (169653822.0, 169653824.0), (169653826.0, 169653828.0), (169653830.0, 169653847.0), (169653849.0, 169653852.0), (169653854.0, 169653869.0), (169653871.0, 169653880.0), (169653882.0, 169653884.0), (169653886.0, 169653888.0), (169653890.0, 169653897.0), (169653899.0, 169653918.0), (169653920.0, 169653940.0), (169653942.0, 169653944.0), (169653946.0, 169653948.0), (169653950.0, 169653955.0), (169653957.0, 169654294.563), (169656842.0, 169659250.5999999), (169659253.797993, 169660504.723), (169662627.0, 169666292.0), (169668413.0, 169668814.933), (169668874.937, 169669754.455107), (169669761.307517, 169671818.069277), (169679985.0, 169680455.23), (169680470.0, 169680994.0), (169680995.97, 169681024.0), (169681025.97, 169681028.0), (169681029.97, 169681082.0), (169681083.97, 169681087.0), (169681088.97, 169681115.0), (169681116.97, 169681160.0), (169681162.97, 169681168.153), (169681171.0, 169681232.25), (169681235.0, 169683649.0), (169686065.373, 169686635.39), (169686695.39, 169687123.0), (169687129.0, 169687194.8269999), (169687196.1270001, 169687415.41), (169687475.41, 169688776.0), (169688780.0, 169689435.0), (169692245.533, 169692845.547), (169692905.55, 169693316.8770001), (169693318.17700005, 169693625.567), (169693685.57, 169695221.0), (169697342.0, 169697705.577), (169698455.69, 169699071.0), (169699073.0, 169699474.92600012), (169699535.72, 169699846.0), (169699849.12, 169699964.137286), (169699964.251522, 169701007.0), (169703128.0, 169703915.78), (169704665.85, 169706756.078681), (169706761.505473, 169706793.0), (169708914.0, 169709001.417009), (169709001.491343, 169710124.937), (169710846.007, 169712579.0), (169714700.0, 169715141.3113), (169715145.232823, 169715193.587144), (169715193.622754, 169715579.721837), (169715583.093153, 169715887.704621), (169715891.878414, 169716333.147), (169717056.167, 169717682.0), (169717689.263, 169717746.0), (169717751.0, 169717818.0), (169717821.263, 169717824.0), (169717827.0, 169717874.0), (169717879.263, 169718364.0), (169720486.0, 169720936.0), (169720941.0, 169722516.303), (169723266.323, 169724150.0), (169726272.0, 169728713.023007), (169728716.576945, 169728726.367), (169729476.48, 169729936.0), (169732057.0, 169733190.98786), (169733474.270775, 169733484.0866), (169733484.123304, 169734936.62), (169735686.64, 169735722.0), (169737843.0, 169741116.777), (169743629.0, 169746965.249019), (169751353.189569, 169751611.8884), (169751611.924595, 169753079.0), (169755200.0, 169755637.147), (169755697.15, 169758175.908106), (169758181.440323, 169758865.0), (169760986.0, 169761097.287), (169761157.287, 169761326.0), (169761329.61, 169761331.0), (169761333.0, 169761861.0), (169761864.613, 169762194.0), (169762196.0, 169762206.0), (169762208.0, 169762696.42799997), (169762697.72799993, 169762972.0), (169767337.447, 169767505.4690001), (169767506.76900005, 169767838.0), (169767841.0, 169767843.0), (169767846.0, 169767888.0), (169767891.667, 169767893.0), (169767895.667, 169767958.0), (169767961.367, 169767963.367), (169767966.667, 169768027.467), (169768087.467, 169770436.0), (169772887.59, 169773487.607), (169773547.607, 169773947.0), (169773952.717, 169773959.975066), (169773964.446328, 169774011.52399993), (169774012.82399988, 169774267.627), (169774327.627, 169774601.421522), (169774605.307099, 169776222.0), (169779035.75, 169779036.653), (169779097.75, 169779697.763), (169779757.767, 169780140.57399988), (169780141.875, 169780477.783), (169780537.787, 169782008.0), (169784129.0, 169784527.89), (169785277.907, 169785677.524745), (169785682.712857, 169787794.0), (169789915.0, 169790031.439212), (169790031.604236, 169790738.047), (169791488.067, 169792088.083), (169792148.083, 169792332.167656), (169792336.372255, 169792423.997), (169792427.0, 169792427.09), (169792430.0, 169792868.1), (169792928.103, 169793580.0), (169795701.0, 169795975.964391), (169795976.02066, 169796948.11), (169797698.223, 169799366.0), (169801487.0, 169803158.363), (169803860.287, 169804516.4), (169804519.0, 169804541.303), (169804546.963, 169804547.303), (169804553.0, 169804597.0), (169804598.963, 169804718.0), (169804719.977, 169805124.77399993), (169805126.07399988, 169805151.0), (169807273.0, 169807330.654286), (169807335.40273, 169809368.43), (169810088.54, 169810937.0), (169813059.0, 169813681.368827), (169813994.479807, 169814004.087464), (169814008.718678, 169815577.63), (169816298.7, 169816723.0), (169818845.0, 169821758.837), (169822508.857, 169822509.0), (169824630.0, 169827887.965849), (169827892.004198, 169827968.903), (169830416.0, 169832598.412697), (169832603.795778, 169834080.0), (169836202.0, 169836286.0), (169836289.22, 169838851.869716), (169838851.928716, 169839430.819732), (169839433.737699, 169839866.0), (169841987.0, 169842151.07599998), (169842152.37599993, 169842486.273), (169842549.37, 169845155.862854), (169853559.0, 169854022.214747), (169854022.27136, 169854129.67), (169854189.67, 169854652.0), (169854655.377, 169854656.587), (169854660.0, 169854698.0), (169854703.587, 169854705.0), (169854707.0, 169854760.59), (169854766.377, 169854826.18400002), (169854827.48399997, 169854879.69), (169854939.69, 169857223.0), (169859709.813, 169860309.827), (169860369.83, 169860832.23600006), (169860833.536, 169861089.847), (169861149.85, 169861279.845278), (169861286.289086, 169863009.0), (169865130.0, 169865169.953), (169865919.973, 169866138.607307), (169866144.473713, 169866422.728976), (169866428.34788, 169866519.99), (169866579.99, 169866966.28699994), (169866967.5869999, 169867300.007), (169867360.01, 169868795.0), (169870916.0, 169871379.113), (169872130.133, 169872274.08328), (169872279.734843, 169872730.147), (169872790.15, 169873127.339), (169873128.63899994, 169873510.167), (169873570.17, 169874581.0), (169876702.0, 169877376.983275), (169877381.508085, 169877588.22), (169878310.29, 169878445.182499), (169878450.45017, 169878939.0), (169879000.307, 169879249.0), (169879254.0, 169879289.3900001), (169879290.69000006, 169879322.298834), (169879327.819881, 169879690.327), (169879750.327, 169880366.0), (169882488.0, 169883770.43), (169884520.45, 169885402.475301), (169888574.576905, 169888737.748539), (169888742.718755, 169889392.497001), (169902081.893794, 169902400.91), (169903120.927, 169903510.0), (169905631.0, 169908581.067), (169911417.0, 169913397.825173), (169913403.39435, 169913778.0), (169913781.0, 169914791.227), (169917202.0, 169920357.21593), (169920362.411701, 169920867.0), (169922988.0, 169923117.347), (169923120.0, 169926311.523), (169926371.523, 169926652.0)]

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
            #print('new time seg')
            #gti_segs=[]
            #gti_segs=[]
           # averaged_G_power_real=[]
           # err_real_array_G=[]
           # averaged_G_power_im=[]
           # err_im_array_G=[]
            
            time_min_val=time_value[0]
            time_max_val=time_value[1]
            #print(time_min_val)
            #print(time_max_val)
            for j in GTI:
                #print(j[0])
                #print(j[1])
                if time_min_val>j[0] and time_max_val<j[1]:
                    gti_segs.append(time_value)
                

        #print('new_seg')
        #print(gti_segs)

      


        
           
            
        real=[]
        im=[]

        #print(len(gti_segs))   
        for good_seg in gti_segs:
            #print(good_seg)
            #coherence_array=[]
            #averaged_G_power_real=[]
            #err_real_array_G=[]
            #averaged_G_power_im=[]
            #err_im_array_G=[]
            
            
            time_min=good_seg[0]
            time_max=good_seg[1]

            time_index_1=[time_min<photon_time_1<time_max for photon_time_1 in data_main_total_obs['TIME']]
            time_index_3=[time_min<photon_time_3<time_max for photon_time_3 in data_ref_total_obs['TIME']]
            #print(len(data_main))

            data_main_time_bin=data_main_total_obs[time_index_1]
            data_ref_time_bin=data_ref_total_obs[time_index_3]



            #making lcs over all mod angles before we index over them


          

      

       


            #av_power_array_real_G=[]
            #err_real_array_G=[]

           # av_power_array_im_G=[]
           # err_im_array_G=[]

            
            
            #TIME_1=data_bin['TIME']
            #TIME_days=TIME_1/(24*60*60)
            #ENERGY=data_bin['PI']


            #TIME_3=data_bin_3['TIME']
            #TIME_days_3=TIME_3/(24*60*60)
            #ENERGY_3=data_bin_3['PI']

           #making lightcurves binned over mod angle (also time)
        
        
            lc_12=Lightcurve.make_lightcurve(data_main_time_bin['TIME'],dt=bin_length,tseg=time_max-time_min,tstart=time_min)
            lc_3=Lightcurve.make_lightcurve(data_ref_time_bin['TIME'],dt=bin_length,tseg=time_max-time_min,tstart=time_min)
            
            
            
            #plt.figure()
            #plt.title('lc_12_all:time interval {} {}'.format(time_min,time_max))
            #plt.plot(lc_12_all.time,lc_12_all.counts)
            #plt.show()

            
            #plt.figure()
            #plt.title('lc_all_3: time interval {} {}'.format(time_min,time_max))
            #plt.plot(lc_3_all.time,lc_3_all.counts)
            #plt.show()
            
            cs=Crossspectrum.from_lightcurve(lc_12,lc_3,norm='frac') #making averagd cross spec
           
           # plt.figure()
           # plt.title('cs_G')
           # plt.plot(cs_G.freq,cs_G.power.real)
           # plt.show()

            
            real.append(cs.power.real)
            im.append(cs.power.imag)
            
            
            
            #np.savetxt(new_prompt+'_'+'allpower.txt',av_power_array_real_G)
            
        #np.savetxt('my_selected_freqs.txt',cs_G.freq)
            
            
        sem_real=[]
        sem_im=[]
        
        mean_real=[]
        mean_im=[]
        
        for i in range(len(real[0])):
        
            mean_r = np.mean([arr[i] for arr in real])
            mean_i=np.mean(ip[i] for ip in im)
        
            mean_real.append(mean_r)
            mean_im.append(mean_i)
        
        #Standard error on the average imaginary power
            sem_i=np.std([ip[i] for ip in im]) / np.sqrt((np.size(im)))
            sem_im.append(sem_i)
            
            sem_r=np.std([r[i] for r in real]) / np.sqrt((np.size(real)))
            sem_real.append(sem_r)
            
            
            
            
            
        plt.figure()
        plt.plot(cs.freq,mean_r)
        plt.errorbars(cs,freq,mean_r,yerr=sem_real)
               
            
            
            
            
            #REAL G


          #  d_real_G = {'power_real': np.array(cs_G.power.real), 'fourier_freq': np.array(cs_G.freq)}
           # df_real_G = pd.DataFrame(data=d_real_G)

           # selected_rows_real_G = df_real_G[(df_real_G['fourier_freq'] >= fmin) & (df_real_G['fourier_freq'] <= fmax)]
           # av_power_real_G=selected_rows_real_G['power_real'].mean()
           # av_power_array_real_G.append(av_power_real_G)
           # np.savetxt(str(new_prompt)+'_'+'av_ith_power_real.txt',av_power_array_real_G)
            #errors on real part
           # sem_real_G=np.std(selected_rows_real_G['power_real'], ddof=1) / np.sqrt((np.size(selected_rows_real_G['power_real'])))
           # err_real_array_G.append(sem_real_G)
           # np.savetxt(str(new_prompt)+'_'+'err_ith_power_real.txt',err_real_array_G)





            #IM G



          #  d_im_G = {'power_im': np.array(cs_G.power.imag), 'fourier_freq': np.array(cs_G.freq)}
          #  df_im_G = pd.DataFrame(data=d_im_G)


           # selected_rows_im_G = df_im_G[(df_im_G['fourier_freq'] >= fmin) & (df_im_G['fourier_freq'] <= fmax)]
           # av_power_im_G=selected_rows_im_G['power_im'].mean()
           # av_power_array_im_G.append(av_power_im_G)
           # np.savetxt(str(new_prompt)+'_'+'av_ith_power_im.txt',av_power_array_im_G)

            #errors on im part
           # sem_im_G=np.std(selected_rows_im_G['power_im'], ddof=1) / np.sqrt((np.size(selected_rows_im_G['power_im'])))
           # err_im_array_G.append(sem_im_G)
           # np.savetxt(str(new_prompt)+'_'+'err_ith_power_im.txt',err_im_array_G)
            
            
           











#make a master gti file
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


    counter = 0
    beep=0
    i1 = 0 
    i2 = 0
    i3 = 0
    gtistart=[]
    gtiend=[]
    while TSTOP1 - beep > 1e-6:
        
        gtistart.append(max(gtistart1[i1], gtistart2[i2], gtistart3[i3]) ) #defining start and end of current master GTI
        gtiend.append(min( gtiend1[i1],   gtiend2[i2],   gtiend3[i3] ))
        
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
        

    master_gti=[list(x) for x in zip(gtistart,gtiend)]
    np.savetxt('Results/'+str(source_name)+'_gti.txt',master_gti)
    print(master_gti)
    

#calculate normalised modulation angle

def cal_eff_mod_angle(filename,datanumber):
    with fits.open(str(filename)) as hdu:
        data=hdu[1].data
        q=data.field('q')
        u=data.field('u')
        q_renormalised=q/np.sqrt(q**2+u**2)
        u_renormalised=u/np.sqrt(q**2+u**2)
        atan2_vec=np.vectorize(math.atan2)
        eff_mod_angle=0.5 *atan2_vec(q_renormalised,u_renormalised)
        np.savetxt('Results/eff_mod_angle'+str(datanumber)+'.txt',eff_mod_angle)

        
#calculate unnormalised modulation angle        
def cal_eff_mod_angle_unnormalised(filename,datanumber):
    with fits.open(str(filename)) as hdu:
        data=hdu[1].data
        q=data.field('q')
        u=data.field('u')
        #q_renormalised=q/np.sqrt(q**2+u**2)
        #u_renormalised=u/np.sqrt(q**2+u**2)
        atan2_vec=np.vectorize(math.atan2)
        eff_mod_angle=0.5 *atan2_vec(q,u)
        np.savetxt('Results/eff_mod_angle'+str(datanumber)+'.txt',eff_mod_angle)
        
        


#make a power spectrum

def powerspectrum(data_file,gti,data_header,bin_length,seg_length):
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
def cleaner_and_mod_angle_selector(filename_fits,filename,filename_eff,datanumber,Pmin,Pmax,mod_bin_num):
    # Data
    with fits.open(str(filename)) as hdu:
        data=hdu[1].data #loading in main data
       # print('Initial number of data {}'.format(len(data)))
    eff_mod_angle=np.loadtxt(str(filename_eff)) #loading in mod angles of combined 
    
    # Header Data
    with fits.open(str(filename_fits)) as hdu2:
        data_header=hdu2[1].header 
        TSTART=data_header['TSTART']
        TSTOP=data_header['TSTOP']
        MJDREFF=data_header['MJDREFF']   #defining the fits header data 
        MJDREFI=data_header['MJDREFI']
        MJD_ref_day=MJDREFF+MJDREFI
        curve_duration=TSTOP-TSTART
        
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
            fits.writeto('Lightcurves/lc_'+str(datanumber)+'_'+str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_.fits',data_bin,overwrite=True)
            
            

#make a cross spectrum            
def crossspectrum(file12,file3,gti,bin_length,seg_length,Pmin,Pmax):
    
    GTI=list(np.loadtxt(str(gti)))
    
    with fits.open(str(file12)) as hdu:
            data_12=hdu[1].data  #reading in DU1+DU2
    
    with fits.open(str(file3)) as hdu2:
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

            #Lightcurve

            lightcurve_12=Lightcurve.make_lightcurve(TIME,dt=bin_length,tseg=curve_duration,tstart=TSTART,gti=GTI)
            lightcurve_12.apply_gtis()

            lightcurve_3=Lightcurve.make_lightcurve(TIME_3,dt=bin_length,tseg=curve_duration,tstart=TSTART,gti=GTI)
            lightcurve_3.apply_gtis()

            #Cross spec 

            avg_cs = AveragedCrossspectrum.from_lightcurve(lightcurve_12,lightcurve_3,seg_length,norm='frac')
            avg_cs=avg_cs.rebin_log(f=0.1)

           #Plotting the real part against fourier frequency

            fig, ax1 = plt.subplots(1,1,figsize=(9,6))
                #ax1.plot(cs.freq, cs.power, color='blue',label='no log rebin')
            ax1.plot(avg_cs.freq, avg_cs.power.real*avg_cs.freq, color='green')
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

        



            
def G_norm(bin_length,seg_length,Pmin,Pmax,fmin,fmax,mod_bin_number,norm12,norm3,gti):
    
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
            print(norm_power_real)
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
            np.savetxt('Results/allmod_av_real_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_norm_array)
            
            #im
            norm_d_im = {'all_power_im': np.array(norm_power_im), 'all_fourier_freq': np.array(norm_freq)} #total pwr and freq in dataset
            df_norm_im = pd.DataFrame(data=norm_d_im)
            selected_rows_norm_im = df_norm_im[(df_norm_im['all_fourier_freq'] >= fmin) & (df_norm_im['all_fourier_freq'] <= fmax)] #selecting freq range
      
            av_power_norm_im=selected_rows_norm_im['all_power_im'].mean() #calculating mean pwr
            av_power_im_norm_array.append(av_power_norm_im)
            np.savetxt('Results/allmod_av_im_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_im_norm_array)



            #calculating standard error on the mean
            sem_real_norm=np.std(selected_rows_norm['all_power'], ddof=1) / np.sqrt((np.size(selected_rows_norm['all_power'])))
            err_array_norm_real.append(sem_real_norm)
            np.savetxt('Results/allmod_av_real_err_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_norm_real) #one of these comes out for every run
             

            sem_im_norm=np.std(selected_rows_norm_im['all_power_im'], ddof=1) / np.sqrt((np.size(selected_rows_norm_im['all_power_im'])))
            err_array_norm_im.append(sem_im_norm)
            np.savetxt('Results/allmod_av_im_err_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_norm_im) #one of these comes out for every run
             







            #calculating normalisation constant
            norm_factor=(np.sqrt((fmax-fmin))/np.sqrt(av_power_norm))
            norm_factor_array.append(norm_factor) 
            np.savetxt('Results/norm_cs_'+'freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',norm_factor_array)

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
     
        for file_12 in glob.iglob('Lightcurves/lc_12_'+str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_'+'.fits'): #lc file name  

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
                np.savetxt('Results/G_av_real_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_real)
               
                #Standard error on the average real power
                sem_real=np.std(selected_rows_real['real_power'], ddof=1) / np.sqrt((np.size(selected_rows_real['real_power'])))
                err_array_real.append(sem_real)
                np.savetxt('Results/G_av_real_err_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_real) #one of these comes out for every run
             

                #Average imaginary power over frequency
                d_im = {'im_power': np.array(im_power), 'fourier_freq': np.array(cs.freq)}
                df_im = pd.DataFrame(data=d_im)
                selected_rows_im = df_im[(df_im['fourier_freq'] >= fmin) & (df_im['fourier_freq'] <= fmax)]
                av_power_im=selected_rows_im['im_power'].mean()
                av_power_array_im.append(av_power_im)
                np.savetxt('Results/G_av_im_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_im)
            

                #Standard error on the average imaginary power
                sem_im=np.std(selected_rows_im['im_power'], ddof=1) / np.sqrt((np.size(selected_rows_im['im_power'])))
                err_array_im.append(sem_im)
                np.savetxt('Results/G_av_im_err_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_im) #one of these comes out for every run
                
                
                
                
                
        #np.savetxt('norm_cs_'+str(mod_bin_number)+'_bins_'+'freqs_'+str(fmin)+str(fmax)+'.txt',norm_factor)        
                
                
  #_norm(lc_dir_prompt,ref_curve_data,bin_length,seg_length,Pmin,Pmax,cs_name,fmin,fmax,mod_minimum,mod_maximum,mod_bin_number,norm,norm12,norm3):              
                
def mod_angle_cross_spec_ith(header_file,gti,bin_length,seg_length,Pmin,Pmax,fmin,fmax,mod_bin_number):
   
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
    print('mod list made')    
    with fits.open(header_file) as hduh:
        data_header=hduh[1].header
        TSTART=data_header['TSTART']
        TSTOP=data_header['TSTOP']
        MJDREFF=data_header['MJDREFF']   #defining the modulation angle selected fits data
        MJDREFI=data_header['MJDREFI']
        MJD_ref_day=MJDREFF+MJDREFI
        curve_duration=TSTOP-TSTART
        print('header read in')
    for i in mod_angle_list:
        mod_min=i[0]
        mod_max=i[1]
       
        for file_12 in glob.iglob('Lightcurves/lc_12_'+str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_'+'.fits'): #lc file name
            with fits.open(file_12) as hdu1:
                data_12=hdu1[1].data
               # print(data_12)
          
        for ref_curve_data in glob.iglob('Lightcurves/lc_3_'+str(Pmin)+'_'+str(Pmax)+'_'+str(mod_min)+'_'+str(mod_max)+'_'+'.fits'): 
            with fits.open(str(ref_curve_data)) as hdu2:
                data_ref=hdu2[1].data
         

                TIME=data_12['TIME']
                TIME_ref=data_ref['TIME']   #defining the ref lightcurve data
                   
                #12 lightcurve
                lc=Lightcurve.make_lightcurve(TIME,dt=bin_length,tseg=curve_duration,tstart=TSTART,mjdref=MJD_ref_day,gti=GTI)
                lc.apply_gtis()
                print(lc.counts)           

                #3 lightcurve (ref)
                lc_ref=Lightcurve.make_lightcurve(TIME_ref,dt=bin_length,tseg=curve_duration,tstart=TSTART,mjdref=MJD_ref_day,gti=GTI)
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
                np.savetxt('Results/cs_ith_av_real'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_real)
               
                #errors on real part
                sem_real=np.std(selected_rows_real['power_real'], ddof=1) / np.sqrt((np.size(selected_rows_real['power_real'])))
                err_array_real.append(sem_real)
                np.savetxt('Results/cs_ith_av_real_err'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_real)
        
        
        
                #Averaging im part
                d_im = {'power_im': np.array(cs.power.imag), 'fourier_freq': np.array(cs.freq)}
                df_im = pd.DataFrame(data=d_im)
                selected_rows_im = df_im[(df_im['fourier_freq'] >= fmin) & (df_im['fourier_freq'] <= fmax)]
                av_power_im=selected_rows_im['power_im'].mean()
                av_power_array_im.append(av_power_im)
                np.savetxt('Results/cs_ith_av_im_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',av_power_array_im)

                #errors on im part
                sem_im=np.std(selected_rows_im['power_im'], ddof=1) / np.sqrt((np.size(selected_rows_im['power_im'])))
                err_array_im.append(sem_im)
                np.savetxt('Results/cs_ith_av_im_err'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt',err_array_im)
                

                
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

    normalisation_factor=np.loadtxt('Results/norm_cs_'+'freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')

    norm_factor_list=[np.sqrt(1/(normalisation_factor**2))]*len(av_mod)
    #123each
#Results/cs_ith_av_real'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.t
    each_real=np.loadtxt('Results/cs_ith_av_real'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    each_im=np.loadtxt('Results/cs_ith_av_im_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
  
    eacherr_power_real=np.loadtxt('Results/cs_ith_av_real_err'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    eacherr_power_im=np.loadtxt('Results/cs_ith_av_im_err'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    
    #123all
    all_im=np.loadtxt('Results/allmod_av_im_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    all_real=np.loadtxt('Results/allmod_av_real_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    all_im_err=np.loadtxt('Results/allmod_av_im_err_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    all_real_err=np.loadtxt('Results/allmod_av_real_err_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
   

    #G
    newG_im=np.loadtxt('Results/G_av_im_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real=np.loadtxt('Results/G_av_real_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im=np.loadtxt('Results/G_av_im_err_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real=np.loadtxt('Results/G_av_real_err_'+str(mod_bin_number)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
  


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
    newG_im=np.loadtxt('Results/G_av_im_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real=np.loadtxt('Results/G_av_real_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im=np.loadtxt('Results/G_av_im_err_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real=np.loadtxt('Results/G_av_real_err_'+str(bins1)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
  
    #G2
    newG_im_2=np.loadtxt('Results/G_av_im_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real_2=np.loadtxt('Results/G_av_real_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im_2=np.loadtxt('Results/G_av_im_err_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real_2=np.loadtxt('Results/G_av_real_err_'+str(bins2)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    #G3
    newG_im_3=np.loadtxt('Results/G_av_im_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_real_3=np.loadtxt('Results/G_av_real_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    
    newG_err_im_3=np.loadtxt('Results/G_av_im_err_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')
    newG_err_real_3=np.loadtxt('Results/G_av_real_err_'+str(bins3)+'_bins_freqs_'+str(fmin)+'_'+str(fmax)+'_'+str(bin_length)+'_'+str(seg_length)+'.txt')



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

