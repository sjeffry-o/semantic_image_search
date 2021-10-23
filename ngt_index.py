import ngtpy
 
def ngt_index(vectors, name="index_name", dim=1024):
    ngtpy.create(name, dim)
    index = ngtpy.Index(name)
    index.batch_insert(vectors)
    index.save()
