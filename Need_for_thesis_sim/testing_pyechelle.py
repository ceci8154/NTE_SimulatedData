from NTEsim.make_schematics import MakeSchematics
from NTEsim.make_fits import MakeFits
from NTEpyechelle.sources import CSV
import h5py




MS = MakeSchematics()
MS.gpu = False
MS.overw = True
MS.uv_make_bias()#exposure_time=0.01)


