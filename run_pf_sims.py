import numpy as np
import dedalus.public as de
import salty_boussinesq_vpf_tangent as sb
from mpi4py import MPI
commw = MPI.COMM_WORLD
comms = MPI.COMM_SELF
rank, size = commw.rank, commw.size

eps = np.sqrt(np.array([1e-4,5e-5,2e-5,1e-5,5e-6]))
dts = np.array([1e-3,5e-4,2.5e-4,2e-4,5e-5])
simnames = [f'salty-boussinesq-melting-vpf-tangent-SBDF2-{ϵ:.0e}-conserved-passive' for ϵ in eps]

import logging
logger = logging.getLogger(__name__)
fhandler = logging.FileHandler(f'salty-boussinesq-vpf-{rank}.log')
shandler = logging.StreamHandler()
logger.addHandler(fhandler)
logger.addHandler(shandler)

for i in range(rank,len(eps),size):
    print(f'rank {rank} sim starting:')
    sb.run_salty_boussinesq_vpf(simnames[i],eps[i],dts[i],comms,logger)

for handler in logger.handlers:
    handler.close()
    logger.removeFilter(handler)
