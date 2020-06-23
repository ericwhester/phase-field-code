import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
import remapped_maths as rm

d = de.operators.differentiate
integ = de.operators.integrate

import glob
from dedalus.tools import post
import file_tools as flt
import interpolation as ip
import time
import os
import logging
import sys
logger = logging.getLogger(__name__)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank,size = comm.rank,comm.size

J, K, g, G, Kdet, eps = rm.J, rm.K, rm.g, rm.G, rm.Kdet, rm.eps

savedir = './salty-boussinesq-melting'

ν = 1e-2
κ = 1e-2
γ = 1e-2
μ = 1e-2
M = .2  
N = 0
L = 1
nx = 128
nz = 64
dt = 1e-3
timestepper = 'SBDF2'
simname = f'salty-boussinesq-melting-tangent-conserved-passive'
restart = False
if rank == 0: flt.makedir(f'{savedir}/frames/{simname}')
tend = 10
save_step = 1
save_freq = round(save_step/dt)

xbasis = de.Fourier('x',nx,interval=(0,4),dealias=3/2)
zbasis = de.Chebyshev('z',nz,interval=(0,1),dealias=3/2)
domain = de.Domain([xbasis,zbasis],grid_dtype=np.float64)
if rank == 0: flt.save_domain(f'{savedir}/domain-{simname}.h5',domain)
x, z = domain.grids(scales=domain.dealias)
xx,zz = x+0*z, 0*x+z
dims = range(1,3)

problem = de.IVP(domain,variables=['u21','u22','p2','q2','T1','T1_2','T2','T2_2','C2','C2_2','h','ht','E','S'])
problem.meta[:]['z']['dirichlet'] = True
problem.meta['h','ht']['z']['constant'] = True
problem.meta['E','S']['x','z']['constant'] = True
problem.parameters['κ'] = κ
problem.parameters['ν'] = ν
problem.parameters['γ'] = γ
problem.parameters['μ'] = μ
problem.parameters['M'] = M
problem.parameters['N'] = N
problem.parameters['π'] = np.pi
problem.parameters['L'] = L
As = [4]*5
for i in range(len(As)): problem.parameters[f'A{i+1}'] = As[i]

problem.substitutions['d_1(A)'] = 'dx(A)'
problem.substitutions['d_2(A)'] = 'dz(A)'
problem.substitutions['T1_1'] = 'd_1(T1)'
problem.substitutions['T2_1'] = 'd_1(T2)'
problem.substitutions['C2_1'] = 'd_1(C2)'
problem.substitutions['hx'] = 'd_1(h)'
problem.substitutions['angle'] = 'sqrt(1 + hx**2)'
problem.substitutions['omz1'] = 'sqrt(1 + (hx)**2)/(2-h)'
problem.substitutions['omz2'] = 'sqrt(1 + (hx)**2)/h'
problem.substitutions['curvature'] = 'dx(hx)/angle**3'
problem.substitutions['xc'] = 'x'
problem.substitutions['zc1'] = 'h + (2-h)*z'
problem.substitutions['zc2'] = 'h*z'
for l in [1,2]:
    problem.substitutions[f'Kdet{l}'] = Kdet[l]
    for i in range(3):
        for j in range(3):
            problem.substitutions[f'J{l}{i}{j}'] = J.get((l,i,j),'0')
            problem.substitutions[f'K{l}{i}{j}'] = K.get((l,i,j),'0')
            problem.substitutions[f'g{l}{i}{j}'] = g.get((l,i,j),'0')
            for k in range(3):
                problem.substitutions[f'G{l}{i}_{j}{k}'] = G.get((l,i,j,k),'0')   
l = 2
for i in dims:
    for j in dims:
        problem.substitutions[f'cd_{i}_u{l}{j}'] = f'd_{i}(u{l}{j})' + ' + '.join(['']+[f'G{l}{j}_{k}{i}*u{l}{k}' for k in dims if G.get((l,j,k,i),0)])
problem.substitutions[f'div{l}'] = f'cd_1_u{l}1 + cd_2_u{l}2'
problem.substitutions[f'vorticity{l}'] = f'Kdet{l}*'+'({})'.format(' + '.join([f'{eps[i,j]}*g{l}{i}{k}*cd_{k}_u{l}{j}' for k in dims for j in dims for i in dims if eps.get((i,j),0)]))
for j in dims: problem.substitutions[f'curl_vorticity{l}{j}'] = f'(1/Kdet{l})*'+'({})'.format(' + '.join(f'{eps[i,j]}*d_{i}(q{l})' for i in dims if eps.get((i,j),0)))
for k in dims: problem.substitutions[f'q{l}_cross_u{l}{k}'] = f'Kdet{l}*q{l}*'+'({})'.format(' + '.join(f'{eps[i,j]}*g{l}{i}{k}*u{l}{j}' for i in dims for j in dims if eps.get((i,j),0)))
for j in dims: problem.substitutions[f'grad_p{l}{j}'] = ' + '.join([f'g{l}{i}{j}*d_{i}(p{l})' for i in dims])
for j in dims: problem.substitutions[f'f{l}{j}'] = f'(T{l} + N*C{l})*J{l}2{j}'
problem.substitutions['d_0(A)'] = '0'
for j in dims: problem.substitutions[f'adv_u{l}{j}'] = ' + '.join([f'J{l}0{i}*d_{i}(u{l}{j})' for i in range(3) if J.get((l,0,i),0)] + [f'J{l}0{i}*u{l}{k}*G{l}{j}_{k}{i}' for i in range(3) for k in dims if J.get((l,0,i),0) and G.get((l,j,k,i))])
for i in dims: problem.substitutions[f'dτu{l}{i}'] = f'- adv_u{l}{i} + ν*curl_vorticity{l}{i} - grad_p{l}{i} + q{l}_cross_u{l}{i} + f{l}{i}'
for i in dims: problem.substitutions[f'dtu{l}{i}'] = f'dτu{l}{i} + adv_u{l}{i}'
for l in dims: problem.substitutions[f'lapT{l}'] = '{} - ({})'.format(' + '.join([f'g{l}{i}{j}*d_{i}(T{l}_{j})' for i in dims for j in dims]),' + '.join([f'g{l}{i}{j}*T{l}_{k}*G{l}{k}_{i}{j}' for i in dims for j in dims for k in dims if G.get((l,k,i,j))]))
problem.substitutions[f'lapC{l}'] = '{} - ({})'.format(' + '.join([f'g{l}{i}{j}*d_{i}(C{l}_{j})' for i in dims for j in dims]),' + '.join([f'g{l}{i}{j}*C{l}_{k}*G{l}{k}_{i}{j}' for i in dims for j in dims for k in dims if G.get((l,k,i,j))]))
problem.substitutions[f'udotT1'] = '0'
problem.substitutions[f'udotT2'] = 'u21*T2_1 + u22*T2_2'
problem.substitutions[f'udotC2'] = 'u21*C2_1 + u22*C2_2'
problem.substitutions['ndotT1'] = 'left(( g121*T1_1 + g122*T1_2)/omz1)'
problem.substitutions['ndotT2'] = 'right((g221*T2_1 + g222*T2_2)/omz2)'
problem.substitutions['ndotC2'] = 'right((g221*C2_1 + g222*C2_2)/omz2)'
for l in dims: problem.substitutions[f'dτT{l}'] = 'κ*({}) - ({}) - ({})'.format(f'lapT{l}',f'udotT{l}',' + '.join([f'J{l}0{i}*T{l}_{i}' for i in dims if J.get((l,0,i))]))
problem.substitutions[f'dτC{l}'] = 'μ*({}) - ({}) - ({})'.format(f'lapC{l}',f'udotC{l}',' + '.join([f'J{l}0{i}*C{l}_{i}' for i in dims if J.get((l,0,i))]))
                                                     
# Cartesian quantities
l = 2
problem.substitutions[f'ux{l}'] = ' + '.join(f'u{l}{j}*K{l}{j}1' for j in dims if K.get((l,j,1)))
problem.substitutions[f'uz{l}'] = ' + '.join(f'u{l}{j}*K{l}{j}2' for j in dims if K.get((l,j,2)))
problem.substitutions[f'kenergy{l}'] = f'0.5*(ux{l}**2 + uz{l}**2)'
problem.substitutions[f'pr{l}'] = f'p{l} - kenergy{l}'
problem.substitutions[f'dxc{l}(A)'] = ' + '.join(f'J{l}1{j}*d_{j}(A)' for j in dims if J.get((l,1,j)))
problem.substitutions[f'dzc{l}(A)'] = ' + '.join(f'J{l}2{j}*d_{j}(A)' for j in dims if J.get((l,2,j)))
problem.substitutions[f'dtux{l}'] = f'- dxc{l}(p{l}) - ν*dzc{l}(q{l}) + q{l}*uz{l}'
problem.substitutions[f'dtuz{l}'] = f'- dzc{l}(p{l}) + ν*dxc{l}(q{l}) - q{l}*ux{l}'
# problem.substitutions[f'dtT1'] = f'- dzc{l}(p{l}) + ν*dxc{l}(q{l}) - q{l}*ux{l}'
# problem.substitutions[f'dtT2'] = f'- dzc{l}(p{l}) + ν*dxc{l}(q{l}) - q{l}*ux{l}'

problem.add_equation(f'       A1*(d_1(u{l}1) + d_2(u{l}2)) = A1*(d_1(u{l}1) + d_2(u{l}2)) - div{l}')
problem.add_equation(f'q{l} + A2*(d_2(u{l}1) - d_1(u{l}2)) = A2*(d_2(u{l}1) - d_1(u{l}2)) + vorticity{l}')
problem.add_equation(f'dt(u{l}1) + A3*d_1(p{l}) + A4*ν*d_2(q{l})             = A3*d_1(p{l}) + A4*ν*d_2(q{l})             + dτu{l}1')
problem.add_equation(f'dt(u{l}2) + A3*d_2(p{l}) - A4*ν*d_1(q{l}) - (T2+N*C2) = A3*d_2(p{l}) - A4*ν*d_1(q{l}) - (T2+N*C2) + dτu{l}2')    
problem.add_equation('T1_2 - d_2(T1) = 0')
problem.add_equation('T2_2 - d_2(T2) = 0')
problem.add_equation('dt(T1) - κ*A5*(d_1(T1_1) + d_2(T1_2)) = - κ*A5*(d_1(T1_1) + d_2(T1_2)) + dτT1')
problem.add_equation('dt(T2) - κ*A5*(d_1(T2_1) + d_2(T2_2)) = - κ*A5*(d_1(T2_1) + d_2(T2_2)) + dτT2')
problem.add_equation('dt(C2) - μ*A5*(d_1(C2_1) + d_2(C2_2)) = - μ*A5*(d_1(C2_1) + d_2(C2_2)) + dτC2')
problem.add_equation('C2_2 - d_2(C2) = 0')
problem.add_equation('dt(h) - ht = 0')
problem.add_equation('right(T2_2) - left(T1_2) + right(ht)*(L/κ) = right(T2_2) - left(T1_2) + right(ht)*(L/κ) - (ndotT2 - ndotT1 + right(ht/angle)*(L/κ))')
problem.add_equation('E = integ(Kdet1*T1 + Kdet2*(T2+1))')
problem.add_equation('S = integ(Kdet2*C2)')

# problem.add_bc("right(T1) = -1")
problem.add_bc("right(T1_2) = 0") # Conserved energy
problem.add_bc('right(u21) = 0')
problem.add_bc('right(u22) = 0',condition='nx != 0')
problem.add_bc("right(T2 + M*C2 + γ*dx(hx)) = γ*(dx(hx) - curvature)") # Gibbs Thomson condition
problem.add_bc("right(T2) - left(T1) = 0")
problem.add_bc("right(C2_2) = - right((g221*C2_1 + (ht/h)*C2/μ)/g222)")
#problem.add_bc("right(C2_2 + C2_1 + ht) = right(C2_2 + C2_1 + ht) - right(g222*C2_2 + g221*C2_1 + (ht/h)*C2/μ)")
problem.add_bc('left(p2) = 0',condition='nx == 0')
problem.add_bc('left(u21) = 0')
problem.add_bc('left(u22) = 0')
# problem.add_bc("left(T2) = 1")
problem.add_bc("left(T2_2) = 0") # Conserved energy
problem.add_bc("left(C2_2) = 0")

solver = problem.build_solver(eval(f'de.timesteppers.{timestepper}'))

solver.stop_sim_time = tend

u21,u22,p2,q2,T1,T1_2,T2,T2_2,C2,C2_2,h,ht,E,S = [solver.state[fname] for fname in problem.variables]
for field in u21,u22,p2,q2,T1,T1_2,T2,T2_2,C2,C2_2,h,ht,E,S:
    field.set_scales(domain.dealias)

if not restart:
    h['g'] = 1
    xc = x + 0*z
    zc1 = h['g'] + (2-h['g'])*z
    zc2 = h['g']*z
    T1['g'] = 1-zc1
    T2['g'] = 1-zc2 + np.exp(-((xc-2)**2+(zc2-.5)**2)*5**2)#1-z
    T1.differentiate('z',out=T1_2)
    T2.differentiate('z',out=T2_2)
    C2['g'] = .05 + (1-.05)*.5*(1-np.tanh(10*(zc2-.5)))
    E['g'] = (h*(T2+1) + (2-h)*T1).evaluate().integrate()['g']
    S['g'] = (h*C2).evaluate().integrate()['g']

    # v['g'] = (angle*(ndotT1-ndotT2))['g']
if restart:
    save_file = sorted(glob.glob(f'{savedir}/analysis-{simname}/*.h5'))[-1]
    write, dt = solver.load_state(save_file,-1)
    
analysis = solver.evaluator.add_file_handler(f'{savedir}/analysis-{simname}',iter=save_freq, max_writes=100)
for task in problem.variables + ['zc1','zc2']: analysis.add_task(task)
for l in '2':
    for name in ['div','vorticity','ux','uz','pr','kenergy','dtux','dtuz']:
        analysis.add_task(name+l)

interface = solver.evaluator.add_file_handler(f'{savedir}/interface-{simname}',iter=int(round(.01/dt)), max_writes=2001)
for task in ['h','ht','E','S']: interface.add_task(task)

start_time = time.time()
while solver.ok:
    if solver.iteration % 100 == 0: 
        logger.info('{} time {:.1f} sim_time {:.1f} E {}'.format(solver.iteration,(time.time()-start_time)/60,solver.sim_time,E["g"][0,0]))
        if np.any(np.isnan(T2['g'])): 
            print('Broken')
            break
    solver.step(dt)
solver.step(dt)
