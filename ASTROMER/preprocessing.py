from .core.data import load_numpy, pretraining_records

def make_pretraining(input,
               batch_size=1,
               shuffle= False,
               sampling= False,
               max_obs= 100,
               msk_frac=0.,
               rnd_frac=0.,
               same_frac=0.,
               repeat=1,
               **numpy_args):
    
    """
    Load and format data to feed ASTROMER model. It can process either a list of bumpy arrays or tf.records. At the the end of this method, a tensorflow dataset is generated following the preprocessing pipeline explained in Section 5.3 (Donoso-Oliva, et al. 2022) 
   
    :param input: The data set containing the light curves.
    :type input: object

    :param batch_size: This integer determines the number of subsets that we will pass to our model.
    :type batch_size: Integer

    :param shuffle: A boolean indicating whether to rearrange samples randomly
    :type shuffle: Boolean

    :param sampling: A Boolean that when is true, indicates the model to take samples of every light curve instead of all observation samples. 
    :type sampling: Boolean

    :param max_obs: This Integer indicates how big each lightcurve sample will be. e.g. (with max_obs = 100): The length of a light curve is 720 observations so the model will generate 7 blocks of 100 observations, and the sample with 20 cases will be completed using padding with zero values after the last point in order to obtain a sequence of length 100.
    :type max_obs: Integer

    :param msk_frac: The fraction of samples that will be masked by the model
    :type msk_frac: Float32

    :param rnd_frac: The fraction of samples in which their values will be changed by random numbers.
    :type rnd_frac: Float32

    :param same_frac: It is the fraction of the masked observations that you unmask and allow to be processed in the attention layer
    :type same_frac: Float32

    :param repeat: This Integer determines the number of times the same data set is repeated.
    :type repeat: Integer

    """
    if isinstance(input, str):
        print("[INFO] Loading Records")
        return pretraining_records(input,
                                   batch_size = batch_size, 
                                   max_obs= max_obs, 
                                   msk_frac= msk_frac,
                                   rnd_frac= rnd_frac, 
                                   same_frac= same_frac, 
                                   sampling= sampling,
                                   shuffle= shuffle, 
                                   repeat= repeat)

        
    if isinstance(input, list):
        print("[INFO] Loading Numpy")
        return load_numpy(input,
               ids= numpy_args["ids"] if "ids" in numpy_args.keys() else None,
               labels= numpy_args["labels"] if "labels" in numpy_args.keys() else None,
               batch_size= batch_size,
               shuffle= shuffle,
               sampling= sampling,
               max_obs= max_obs,
               msk_frac= msk_frac,
               rnd_frac= rnd_frac,
               same_frac= same_frac,
               repeat= repeat)
         
         
        


