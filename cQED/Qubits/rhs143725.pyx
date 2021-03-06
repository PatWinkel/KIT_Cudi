# This file is generated automatically by QuTiP.
# (C) 2011 and later, P. D. Nation & J. R. Johansson

import numpy as np
cimport numpy as np
cimport cython
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.interpolate cimport interp, zinterp
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

include 'C:/Program Files/Anaconda2/envs/qutip_env/lib/site-packages/qutip/cy/complex_math.pxi'

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_td_ode_rhs(
        double t,
        np.ndarray[CTYPE_t, ndim=1] vec,
        np.ndarray[CTYPE_t, ndim=1] data0,np.ndarray[int, ndim=1] idx0,np.ndarray[int, ndim=1] ptr0,
        np.ndarray[CTYPE_t, ndim=1] data1,np.ndarray[int, ndim=1] idx1,np.ndarray[int, ndim=1] ptr1,
        np.ndarray[np.float64_t, ndim=1] _td_array_0):
    
    cdef Py_ssize_t row
    cdef int num_rows = len(vec)
    cdef np.ndarray[CTYPE_t, ndim=1] out = np.zeros((num_rows),dtype=np.complex)
     
    spmvpy(data0, idx0, ptr0, vec, (0 if (t > 25.000000) else _td_array_0[int(round(99 * (t/25.000000)))]), out)
    spmvpy(data1, idx1, ptr1, vec, 1.0, out)
    return out
