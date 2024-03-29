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
               n_classes=-1,
               **numpy_args):
    
    """
    Load and format data to feed the ASTROMER model. 
    On this version, this function is able to process a list of numpy arrays or tf.records. 
    The output is a tensorflow dataset (Tf.data) that was generated by following 
    the preprocessing strategy explained in Section 5.3 (Donoso-Oliva, et al. 2022) 
   
    :param input: Dataset source. If using records then 'input' is a string pointing to the local directory containing the records files (e.g., ./my_records/train). The other option consists in passing a list of numpy arrays (light curves) 
    :type input: object

    :param batch_size: Determines the number of subsets using during training. Notice that len(subset)<len(dataset).
    :type batch_size: Integer

    :param shuffle: Shuffle dataset before passing batches
    :type shuffle: Boolean

    :param sampling: If True, for each light curve we will sample a single window of length `max_obs`. If False, the light curve will be divided into `max_obs` windows covering all observations.
    :type sampling: Boolean

    :param max_obs: Indicates how long each input sample will be. In general, we use shorter sequences to train the model, avoiding overloading the memory or extremely zero-padding the sequence.
    :type max_obs: Integer

    :param msk_frac: The fraction of observations for each window that will be masked and therefore not considered by the attention layer. This fraction is used to calculate the RMSE on the loss function. 
    :type msk_frac: Float32

    :param rnd_frac: The fraction of masked values that will be changed by random observations from the same window. (This is inspired by BERT et.al., 2018)
    :type rnd_frac: Float32

    :param same_frac: The fraction of the masked values that will be unmask and processed by the attention layer. Since same_frac observations are initially part of the masked fraction we still use them to evaluate the loss function.
    :type same_frac: Float32

    :param repeat: Determines the number of times we repeat each light curve in the dataset.
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
                                   repeat= repeat,
                                   n_classes=n_classes)

        
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
               repeat= repeat,
               num_cls=n_classes)
         
         
        


