import jwst
import transitspectroscopy as ts

import matplotlib.pyplot as plt
import glob

#ts.jwst.download(pid = 1118, obs_num = '5')
nrs1_filenames = glob.glob('JWSTdata/jw01118005001_04101*nrs1_uncal.fits')
nrs1_dataset = ts.jwst.load(nrs1_filenames, outputfolder = 'JWSTdata')
nrs1_dataset.detector_calibration()
nrs1_dataset.fit_ramps()

input_dictionary_nrs1 = {}
input_dictionary_nrs2 = {}

nsegments = len(nrs1_dataset.rateints_per_segment)

input_dictionary_nrs1['rampstep'] = nrs1_dataset.rateints_per_segment
input_dictionary_nrs1['times'] = nrs1_dataset.times

input_dictionary_nrs1['ints_per_segment'] = []

for i in range(nsegments):

    input_dictionary_nrs1['ints_per_segment'].append( nrs1_dataset.rateints_per_segment[i].data.shape[0] )

all_outputs_nrs1 = {}

aperture = 2 # pixels

all_outputs_nrs1[aperture] = ts.jwst.stage2(input_dictionary_nrs1, 
                                            nthreads = 5, 
                                            suffix = str(aperture)+'pix', 
                                            aperture_radius = aperture)

plt.figure(figsize=(10,3))

tsincestart_nrs1 = (input_dictionary_nrs1['times'] - input_dictionary_nrs1['times'][0])*24
plt.plot(tsincestart_nrs1, all_outputs_nrs1[aperture]['whitelight'], '.', color = 'cornflowerblue')
rms_nrs1 = np.sqrt(np.var(all_outputs_nrs1[aperture]['whitelight'][0:100]))*1e6
plt.text(0.3, 0.998, '$\sigma_{NRS1} = $'+'{0:.0f} ppm'.format(rms_nrs1), fontsize = 16)

plt.xlabel('Time since exposure start (hours)', fontsize = 18)
plt.ylabel('Relative flux', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.xlim(np.min(tsincestart_nrs1), np.max(tsincestart_nrs1))
plt.ylim(0.992, 1.0025)
plt.show()
