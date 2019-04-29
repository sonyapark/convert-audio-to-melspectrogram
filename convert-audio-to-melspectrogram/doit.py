from __future__ import print_function

''' 
Preprocess audio
'''
import numpy as np
import librosa
import librosa.display
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import pylab


def get_class_names(path="Samples/"):  # class names are subdirectory names in Samples/ directory
    class_names = os.listdir(path)
    return class_names

def preprocess_dataset(inpath="Samples/", outpath="Preproc/"):

    if not os.path.exists(outpath):
        os.mkdir( outpath, 0755 );   # make a new directory for preproc'd files

    class_names = get_class_names(path=inpath)   # get the names of the subdirectories
    nb_classes = len(class_names)
    print("class_names = ",class_names)
    for idx, classname in enumerate(class_names):   # go through the subdirs

        if not os.path.exists(outpath+classname):
            os.mkdir( outpath+classname, 0755 );   # make a new subdirectory for preproc class

        class_files = os.listdir(inpath+classname)
        n_files = len(class_files)
        n_load = n_files
        print(' class name = {:14s} - {:3d}'.format(classname,idx),
            ", ",n_files," files in this class",sep="")

        printevery = 20
        for idx2, infilename in enumerate(class_files):
            audio_path = inpath + classname + '/' + infilename

            if (0 == idx2 % printevery):
                print('\r Loading class: {:14s} ({:2d} of {:2d} classes)'.format(classname,idx+1,nb_classes),
                       ", file ",idx2+1," of ",n_load,": ",audio_path,sep="")
            #start = timer()
            sig, fs = librosa.load(audio_path, sr=None)
   
            save_path = outpath + classname + '/' + infilename+'.jpg'
            #save_path = 'test.jpg'

            pylab.axis('off') # no axis
            pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
            S = librosa.feature.melspectrogram(y=sig, sr=fs)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
            pylab.close()

          

if __name__ == '__main__':
    preprocess_dataset()


