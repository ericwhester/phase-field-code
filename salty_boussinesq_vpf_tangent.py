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
import sys
from mpi4py import MPI
commw = MPI.COMM_WORLD
rank,size = commw.rank,commw.size

J, K, g, G, Kdet, eps = rm.J, rm.K, rm.g, rm.G, rm.Kdet, rm.eps

savedir = './salty-boussinesq-melting'

def run_salty_boussinesq_vpf(simname,ϵ,dt,comm,logger):
    ν = 1e-2
    κ = 1e-2
    γ = 1e-2
    μ = 1e-2
    M = .2
    N = 0
    L = 1
    β = 4/2.6482282
    #ϵ = #np.sqrt(1e-4)
    η = (β*ϵ)**2/ν
    α = (5/6)*(L/κ)*ϵ

    nx = 128
    nz = 256
    #dt = float(sys.argv[2])#2e-4
    timestepper = 'SBDF2'
    #simname = f'salty-boussinesq-melting-vpf-tangent-{timestepper}-{ϵ:.0e}-conserved-passive'
    flt.makedir(f'{savedir}/frames/{simname}')
    tend = 10
    save_step = 1
    save_freq = round(save_step/dt)

    xbasis = de.Fourier('x',nx,interval=(0,4),dealias=3/2)
    zb0 = de.Chebyshev('z0',32,interval=(0,10*ϵ),dealias=3/2)
    zb1 = de.Chebyshev('z1',64,interval=(10*ϵ,1-10*ϵ),dealias=3/2)
    zb2 = de.Chebyshev('z2',32,interval=(1-10*ϵ,1),dealias=3/2)
    zbasis = de.Compound('z',[zb0,zb1,zb2])
    domain = de.Domain([xbasis,zbasis],grid_dtype=np.float64,comm=comm)
    flt.save_domain(f'{savedir}/domain-{simname}.h5',domain)
    x, z = domain.grids(scales=domain.dealias)
    xx,zz = x+0*z, 0*x+z
    xc = xx
    dims = range(1,3)

    # Define GeneralFunction subclass to handle parities
    GeneralFunction = de.operators.GeneralFunction
    class HorizontalFunction(GeneralFunction):
        def __init__(self, domain, layout, func=None, args=[], kw={}, out=None,):
            super().__init__(domain, layout, func=func, args=args, kw=kw, out=out,)

        def meta_constant(self, axis):
            if axis == 1 or axis == 'z': return True
            else: return False

    refname = 'salty-boussinesq-melting-tangent-conserved-passive'
    reffile = glob.glob(f'{savedir}/interface-{refname}/*.h5')[0]
    t0s, x0s = flt.load_data(reffile,'sim_time','x/1.0',group='scales')
    h0s, ht0s = flt.load_data(reffile,'h','ht',group='tasks')
    from scipy.interpolate import RectBivariateSpline
    h0s, ht0s = np.squeeze(h0s), np.squeeze(ht0s)
    hspline = RectBivariateSpline(t0s,x0s,h0s)
    htspline = RectBivariateSpline(t0s,x0s,ht0s)

    def h_func(*args): return hspline(args[0].value,args[1].data[:,0]).T
    def ht_func(*args): return htspline(args[0].value,args[1].data[:,0]).T
    def h_op(*args,domain=domain,h_func=h_func): return HorizontalFunction(domain,layout='g',func=h_func,args=args)
    def ht_op(*args,domain=domain,ht_func=ht_func): return HorizontalFunction(domain,layout='g',func=ht_func,args=args)
    de.operators.parseables['h_op'] = h_op
    de.operators.parseables['ht_op'] = ht_op
    problem = de.IVP(domain,variables=['u11','u12','u21','u22','p1','q1','p2','q2','T1','T1_2','T2','T2_2','C1','C1_2','C2','C2_2','f1','f1_2','f2','f2_2','E','S'])

    problem.meta[:]['z']['dirichlet'] = True
    problem.meta['E','S']['x','z']['constant'] = True
    problem.parameters['ν'] = ν
    problem.parameters['κ'] = κ
    problem.parameters['γ'] = γ
    problem.parameters['μ'] = μ
    problem.parameters['M'] = M
    problem.parameters['N'] = N
    problem.parameters['δ'] = 1e-4
    problem.parameters['ε'] = ϵ
    problem.parameters['L'] = L
    problem.parameters['η'] = η
    problem.parameters['α'] = α
    As = [4]*5
    for i in range(len(As)): problem.parameters[f'A{i+1}'] = As[i]

    problem.substitutions['h'] = 'h_op(t,x)'
    problem.substitutions['ht'] = 'ht_op(t,x)'
    problem.substitutions['d_1(A)'] = 'dx(A)'
    problem.substitutions['d_2(A)'] = 'dz(A)'
    problem.substitutions['T1_1'] = 'd_1(T1)'
    problem.substitutions['T2_1'] = 'd_1(T2)'
    problem.substitutions['C1_1'] = 'd_1(C1)'
    problem.substitutions['C2_1'] = 'd_1(C2)'
    problem.substitutions['f1_1'] = 'd_1(f1)'
    problem.substitutions['f2_1'] = 'd_1(f2)'
    problem.substitutions['hx'] = 'd_1(h)'
    problem.substitutions['angle'] = 'sqrt(1 + hx**2)'
    problem.substitutions['omz1'] = 'sqrt(1 + (hx)**2)/(2-h)'
    problem.substitutions['omz2'] = 'sqrt(1 + (hx)**2)/h'
    problem.substitutions['curvature'] = 'dx(hx)/angle**3'
    problem.substitutions['xc'] = 'x'
    problem.substitutions['zc1'] = 'h + (2-h)*z'
    problem.substitutions['zc2'] = 'h*z'
    problem.substitutions['d_0(A)'] = '0'
    for l in [1,2]:
        problem.substitutions[f'Kdet{l}'] = Kdet[l]
        for i in range(3):
            for j in range(3):
                problem.substitutions[f'J{l}{i}{j}'] = J.get((l,i,j),'0')
                problem.substitutions[f'K{l}{i}{j}'] = K.get((l,i,j),'0')
                problem.substitutions[f'g{l}{i}{j}'] = g.get((l,i,j),'0')
                for k in range(3):
                    problem.substitutions[f'G{l}{i}_{j}{k}'] = G.get((l,i,j,k),'0')   

        for i in dims:
            for j in dims:
                problem.substitutions[f'cd_{i}_u{l}{j}'] = f'd_{i}(u{l}{j})' + ' + '.join(['']+[f'G{l}{j}_{k}{i}*u{l}{k}' for k in dims if G.get((l,j,k,i),0)])
        problem.substitutions[f'div{l}'] = f'cd_1_u{l}1 + cd_2_u{l}2'
        problem.substitutions[f'vorticity{l}'] = f'Kdet{l}*'+'({})'.format(' + '.join([f'{eps[i,j]}*g{l}{i}{k}*cd_{k}_u{l}{j}' for k in dims for j in dims for i in dims if eps.get((i,j),0)]))
        for j in dims: problem.substitutions[f'curl_vorticity{l}{j}'] = f'(1/Kdet{l})*'+'({})'.format(' + '.join(f'{eps[i,j]}*d_{i}(q{l})' for i in dims if eps.get((i,j),0)))
        for k in dims: problem.substitutions[f'q_cross_u{l}{k}'] = f'Kdet{l}*q{l}*'+'({})'.format(' + '.join(f'{eps[i,j]}*g{l}{i}{k}*u{l}{j}' for i in dims for j in dims if eps.get((i,j),0)))
        for j in dims: problem.substitutions[f'grad_p{l}{j}'] = ' + '.join([f'g{l}{i}{j}*d_{i}(p{l})' for i in dims])
        for j in dims: problem.substitutions[f'f{l}{j}'] = f'(T{l}+N*C{l})*J{l}2{j}'
        for j in dims: problem.substitutions[f'adv_u{l}{j}'] = ' + '.join([f'J{l}0{i}*d_{i}(u{l}{j})' for i in dims if J.get((l,0,i),0)] + [f'J{l}0{i}*u{l}{k}*G{l}{j}_{k}{i}' for i in range(3) for k in dims if J.get((l,0,i),0) and G.get((l,j,k,i))])
        for i in dims: problem.substitutions[f'dτu{l}{i}'] = f'- adv_u{l}{i} + ν*curl_vorticity{l}{i} - grad_p{l}{i} + q_cross_u{l}{i} - (f{l}/η)*u{l}{i} + f{l}{i}'
        for i in dims: problem.substitutions[f'dtu{l}{i}'] = f'dτu{l}{i} + adv_u{l}{i}'
        problem.substitutions[f'lapT{l}'] = '{} - ({})'.format(' + '.join([f'g{l}{i}{j}*d_{i}(T{l}_{j})' for i in dims for j in dims]),' + '.join([f'g{l}{i}{j}*T{l}_{k}*G{l}{k}_{i}{j}' for i in dims for j in dims for k in dims if G.get((l,k,i,j))]))
        problem.substitutions[f'lapC{l}'] = '{} - ({})'.format(' + '.join([f'g{l}{i}{j}*d_{i}(C{l}_{j})' for i in dims for j in dims]),' + '.join([f'g{l}{i}{j}*C{l}_{k}*G{l}{k}_{i}{j}' for i in dims for j in dims for k in dims if G.get((l,k,i,j))]))
        problem.substitutions[f'lapf{l}'] = '{} - ({})'.format(' + '.join([f'g{l}{i}{j}*d_{i}(f{l}_{j})' for i in dims for j in dims]),' + '.join([f'g{l}{i}{j}*f{l}_{k}*G{l}{k}_{i}{j}' for i in dims for j in dims for k in dims if G.get((l,k,i,j))]))
        problem.substitutions[f'udotT{l}'] = f'u{l}1*T{l}_1 + u{l}2*T{l}_2'
        problem.substitutions[f'udotC{l}'] = f'u{l}1*C{l}_1 + u{l}2*C{l}_2'
        problem.substitutions[f'ndotT{l}'] = f'left(( g{l}21*T{l}_1 + g{l}22*T{l}_2)/omz{l})'
        problem.substitutions[f'ndotf{l}'] = f'left(( g{l}21*f{l}_1 + g{l}22*f{l}_2)/omz{l})'
        problem.substitutions[f'f_flux{l}'] = f'-ε**(-2)*f{l}*(1-f{l})*(γ*(1-2*f{l}) + ε*(T{l}+M*C{l}))'
        problem.substitutions[f'dtf{l}'] = f'(1/α)*(γ*lapf{l} + f_flux{l})'
        problem.substitutions[f'dτf{l}'] =  f'dtf{l} - '+'({})'.format(' + '.join([f'J{l}0{i}*f{l}_{i}' for i in dims if J.get((l,0,i))]))
        problem.substitutions[f'gradfdotgradC{l}'] = ' + '.join([f'g{l}{i}{j}*f{l}_{i}*C{l}_{j}' for i in dims for j in dims])
        problem.substitutions[f'dτT{l}'] = 'κ*({}) - ({}) + ({}) - ({})'.format(f'lapT{l}',f'udotT{l}',f'L*dtf{l}', ' + '.join([f'J{l}0{i}*T{l}_{i}' for i in dims if J.get((l,0,i))]))
        problem.substitutions[f'dτC{l}'] = 'μ*({}) - ({}) + ({}) - ({})'.format(f'lapC{l}',f'udotC{l}',f'(dtf{l}*C{l} - μ*gradfdotgradC{l})/(1-f{l}+δ)', ' + '.join([f'J{l}0{i}*C{l}_{i}' for i in dims if J.get((l,0,i))]))
    # problem.substitutions['dτh'] = ' - (1/2)*(left(dtf1/(J122*f1_2)) + right(dtf2/(J222*f2_2)))'
    # # Cartesian quantities
    # problem.substitutions[f'ux{l}'] = ' + '.join(f'u{l}{j}*K{l}{j}1' for j in dims if K.get((l,j,1)))
    # problem.substitutions[f'uz{l}'] = ' + '.join(f'u{l}{j}*K{l}{j}2' for j in dims if K.get((l,j,2)))
    # problem.substitutions[f'kenergy{l}'] = f'0.5*(ux{l}**2 + uz{l}**2)'
    # problem.substitutions[f'pr{l}'] = f'p{l} - kenergy{l}'
    # problem.substitutions[f'dxc{l}(A)'] = ' + '.join(f'J{l}1{j}*d_{j}(A)' for j in dims if J.get((l,1,j)))
    # problem.substitutions[f'dzc{l}(A)'] = ' + '.join(f'J{l}2{j}*d_{j}(A)' for j in dims if J.get((l,2,j)))
    # problem.substitutions[f'dtux{l}'] = f'- dxc{l}(p{l}) - ν*dzc{l}(q{l}) + q{l}*uz{l}'
    # problem.substitutions[f'dtuz{l}'] = f'- dzc{l}(p{l}) + ν*dxc{l}(q{l}) - q{l}*ux{l}'
    # problem.substitutions[f'dtT1'] = f'- dzc{l}(p{l}) + ν*dxc{l}(q{l}) - q{l}*ux{l}'
    for l in [1,2]:
        problem.add_equation(f'       A1*(d_1(u{l}1) + d_2(u{l}2)) = A1*(d_1(u{l}1) + d_2(u{l}2)) - div{l}')
        problem.add_equation(f'q{l} + A2*(d_2(u{l}1) - d_1(u{l}2)) = A2*(d_2(u{l}1) - d_1(u{l}2)) + vorticity{l}')
        problem.add_equation(f'dt(u{l}1) + A3*d_1(p{l}) + A4*ν*d_2(q{l}) =             A3*d_1(p{l}) + A4*ν*d_2(q{l}) + dτu{l}1')
        problem.add_equation(f'dt(u{l}2) + A3*d_2(p{l}) - A4*ν*d_1(q{l}) - T2 = - T2 + A3*d_2(p{l}) - A4*ν*d_1(q{l}) + dτu{l}2')    
        problem.add_equation(f'T{l}_2 - d_2(T{l}) = 0')
        problem.add_equation(f'dt(T{l}) - κ*A5*(d_1(T{l}_1) + d_2(T{l}_2)) = - κ*A5*(d_1(T{l}_1) + d_2(T{l}_2)) + dτT{l}')
        problem.add_equation(f'C{l}_2 - d_2(C{l}) = 0')
        problem.add_equation(f'dt(C{l}) - μ*A5*(d_1(C{l}_1) + d_2(C{l}_2)) = - μ*A5*(d_1(C{l}_1) + d_2(C{l}_2)) + dτC{l}')
        problem.add_equation(f'f{l}_2 - d_2(f{l}) = 0')
        problem.add_equation(f'α*dt(f{l})-γ*A5*(d_1(f{l}_1) + d_2(f{l}_2)) = - γ*A5*(d_1(f{l}_1) + d_2(f{l}_2)) + α*dτf{l}')
    # problem.add_equation('ht - (κ/L)*(right(T2_2) - right(T1_2)) = - (κ/L)*(right(T2_2) - right(T1_2)) + dτh')
    # problem.add_equation('ht = -100*(left(f1) - .5)')
    # problem.add_equation('dt(h) - ht = 0')
    problem.add_equation('E = integ(Kdet1*(T1 + L*(1-f1)) + Kdet2*(T2 + L*(1-f2)))')
    problem.add_equation('S = integ(Kdet1*(1-f1)*C1 + Kdet2*(1-f2)*C2)')

    problem.add_bc('right(u11) = 0')
    problem.add_bc('right(u12) = 0',condition='nx != 0')
    # problem.add_bc('right(T1) = -1')
    problem.add_bc('right(T1_2) = 0')
    problem.add_bc('right(C1_2) = 0')
    problem.add_bc('right(f1_2) = 0')
    problem.add_bc('left(u11) - right(u21) = 0')
    problem.add_bc('left(u12) - right(u22) = left((h-1)*u12) + right((h-1)*u22)')
    problem.add_bc('left(p1) - right(p2) = 0')
    problem.add_bc('left(q1) - right(q2) = 0')
    problem.add_bc('right(T2) - left(T1) = 0')
    problem.add_bc('right(C2) - left(C1) = 0')
    problem.add_bc('right(f2) - left(f1) = 0')
    # problem.add_bc('right(f2) = 0.5')
    problem.add_bc('left(T1_2) - right(T2_2) = left((1-h)*T1_2) + right((1-h)*T2_2)')
    problem.add_bc('left(C1_2) - right(C2_2) = left((1-h)*C1_2) + right((1-h)*C2_2)')
    problem.add_bc('left(f1_2) - right(f2_2) = left((1-h)*f1_2) + right((1-h)*f2_2)')
    problem.add_bc('left(p2) = 0',condition='nx == 0')
    problem.add_bc('left(u21) = 0')
    problem.add_bc('left(u22) = 0')
    problem.add_bc('left(T2_2) = 0')
    # problem.add_bc('left(T2) = 1')
    problem.add_bc('left(C2_2) = 0')
    problem.add_bc('left(f2_2) = 0')

    solver = problem.build_solver(eval(f'de.timesteppers.{timestepper}'))

    solver.stop_sim_time = tend

    fields = {fname:solver.state[fname] for fname in problem.variables}
    for fname,field in fields.items():
        field.set_scales(domain.dealias)
    hop, htop = solver.evaluator.vars['h'], solver.evaluator.vars['ht']
    h, ht = hop.evaluate(), htop.evaluate()
    T1, T2, C1,C2,f1, f2,E,S = [fields[name] for name in ['T1','T2','C1','C2','f1','f2','E','S']]
    xc = xx
    zc1 = h['g'] + (2-h['g'])*zz
    zc2 = h['g']*zz
    h['g'] = 1
    T1['g'] = 1-zc1
    T2['g'] = 1-zc2 + np.exp(-((xc-2)**2+(zc2-.5)**2)*5**2)
    T1.differentiate('z',out=fields['T1_2'])
    T2.differentiate('z',out=fields['T2_2'])
    C1['g'] = 0.05 + (1-0.05)*.5*(1 - np.tanh(10*(zc1-.5)))
    C2['g'] = 0.05 + (1-0.05)*.5*(1 - np.tanh(10*(zc2-.5)))
    C1.differentiate('z',out=fields['C1_2'])
    C2.differentiate('z',out=fields['C2_2'])
    zc1 = h['g'] + (2-h['g'])*zz
    zc2 = h['g']*zz
    f1['g'] = .5*(1+np.tanh((1/(2*ϵ))*(zc1-h['g'])/np.sqrt(1+d(h,'x')**2)['g']))
    f2['g'] = .5*(1+np.tanh((1/(2*ϵ))*(zc2-h['g'])/np.sqrt(1+d(h,'x')**2)['g']))
    f1.differentiate('z',out=fields['f1_2'])
    f2.differentiate('z',out=fields['f2_2'])
    E['g'] = (h*(T2+L*(1-f2)) + (2-h)*(T1+L*(1-f1))).evaluate().integrate()['g']
    S['g'] = (h*(1-f2)*C2 + (2-h)*(1-f1)*C1).evaluate().integrate()['g']

    analysis = solver.evaluator.add_file_handler(f'{savedir}/analysis-{simname}',iter=save_freq, max_writes=100,mode='overwrite')
    for task in problem.variables + ['h','ht','zc1','zc2']: analysis.add_task(task)

    while solver.ok:
        if solver.iteration % 100 == 0: 
            logger.info(solver.iteration)
            if np.any(np.isnan(T2['g'])): break
        solver.step(dt)
    solver.step(dt)

