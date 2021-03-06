# xi = (t,x,z)
# ξi = (τ,ξ,ζ)
eps, J, K, g, G, Kdet = {}, {}, {}, {}, {}, {}
# Jij = J^j_i = d_xi(ξj)
J[1,0,0] = '1'
J[1,0,2] = '-(1-z)*ht/(2-h)'
J[1,1,1] = '1'
J[1,1,2] = '-(1-z)*hx/(2-h)'
J[1,2,2] = '1/(2-h)'
J[2,0,0] = '1'
J[2,0,2] = '-z*ht/h'
J[2,1,1] = '1'
J[2,1,2] = '-z*hx/h'
J[2,2,2] = '1/h'
# Kij = d_ξi(xj)
K[1,0,0] = '1'
K[1,0,2] = '(1-z)*ht'
K[1,1,1] = '1'
K[1,1,2] = '(1-z)*hx'
K[1,2,2] = '2-h'
K[2,0,0] = '1'
K[2,0,2] = 'z*ht'
K[2,1,1] = '1'
K[2,1,2] = 'z*hx'
K[2,2,2] = 'h'
# Kdet
Kdet[1] = '2-h'
Kdet[2] = 'h'
# gij = ωi.ωj
g[1,1,1] = '1'
g[1,1,2] = '-(1-z)*hx/(2-h)'
g[1,2,1] = 'g112'
g[1,2,2] = '(1 + ((1-z)*hx)**2)/(2-h)**2'
g[2,1,1] = '1'
g[2,1,2] = '-z*hx/h'
g[2,2,1] = 'g212'
g[2,2,2] = '(1 + (z*hx)**2)/h**2'
# Gi_jk = ωi.d_ξk(uj)
G[1,2,1,0] = '(1-z)*dx(ht)/(2-h)'
G[1,2,2,0] = '-ht/(2-h)'
G[1,2,1,1] = '(1-z)*dx(hx)/(2-h)'
G[1,2,1,2] = '-hx/(2-h)'
G[1,2,2,1] = 'G12_12'
G[2,2,1,0] = 'z*dx(ht)/h'
G[2,2,2,0] = 'ht/h'
G[2,2,1,1] = 'z*dx(hx)/h'
G[2,2,1,2] = 'hx/h'
G[2,2,2,1] = 'G22_12'

eps[1,2] = '1'
eps[2,1] = '-1'
dims = range(1,3)