# ASTROMER Python library

## Install
```
pip install ASTROMER
```

## how to use it
Specify the pre-trained weights following the same name [on this github](https://github.com/astromer-science/weights)
```
model = ASTROMER()
model.from_pretrained('macho')
```
Assuming you have a list of vary-lenght (numpy) light curves, then:
```
att = model.encode(samples_collection,
                   oids_list=oids_list,
                   batch_size=10,
                   concatenate=True)
```
where
- `samples_collection` is a list of numpy array light curves
- `oids_list` is a list with the light curves ids (needed to concatenate 200-len windows)
- `batch_size` specify the number of samples per forward pass
-  when `concatenate=True` ASTROMER concatenates every 200-lenght windows belonging the same object id (remember this version of ASTROMER only works up to 200 observations). The output when `concatenate=True` is a list of vary-length attention vectors.

## How to train
to be continue
