import warnings

from astromer import ASTROMER

from alerce.core import Alerce


def test_model_instance():
    alerce = Alerce()
    model = ASTROMER()
    model.from_pretrained('macho')

    samples_collection = []
    oids_list = ['ZTF18aabkmmz', 'ZTF18abbuksn', 'ZTF17aaaecwh']
    for oid in oids_list:
        sample = alerce.query_detections(oid, format="pandas")
        sample = sample[sample['fid']==1] # getting single band light curve
        sample = sample[['mjd', 'magpsf', 'sigmapsf']] # use only necessary columns
        samples_collection.append(sample.values)

    att = model.encode(samples_collection,
                       oids_list=oids_list,
                       batch_size=3,
                       concatenate=True)

    assert att[0].shape[-1] == 256
