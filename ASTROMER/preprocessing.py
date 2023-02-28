from .core.data import load_numpy, pretraining_records
"""
 This method build the ASTROMER input format, that is based on the BERT (Devlin et al., 2018) masking strategy.
"""

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
    agregar explícitamente si es listado es numpy y si es string es pretraining
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
         
         
        


