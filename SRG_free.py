#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python

#------------------------------------------------------------------------------
# imsrg_pairing.py
#
# author:   H. Hergert 
# version:  1.5.0
# date:     Dec 6, 2016
# 
# tested with Python v2.7
# 
# Solves the pairing model for four particles in a basis of four doubly 
# degenerate states by means of an In-Medium Similarity Renormalization 
# Group (IMSRG) flow.
#
#------------------------------------------------------------------------------

import numpy as np
from numpy import array, dot, diag, reshape, transpose
from scipy.linalg import eigvalsh
from scipy.integrate import odeint, ode
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from matplotlib import cm
from sys import argv
import math
import itertools
from sympy import *
from pylab import *

#-----------------------------------------------------------------------------------
# basis and index functions
#-----------------------------------------------------------------------------------

def construct_basis_2B(holes, particles):
    basis = []
    for i in holes:
        for j in holes:
            basis.append((i, j))

    for i in holes:
        for a in particles:
            basis.append((i, a))

    for a in particles:
        for i in holes:
            basis.append((a, i))

    for a in particles:
        for b in particles:
            basis.append((a, b))

    return basis


def construct_basis_ph2B(holes, particles):
    basis = []
    for i in holes:
        for j in holes:
            basis.append((i, j))

    for i in holes:
        for a in particles:
            basis.append((i, a))

    for a in particles:
        for i in holes:
            basis.append((a, i))

    for a in particles:
        for b in particles:
            basis.append((a, b))

    return basis


#
# We use dictionaries for the reverse lookup of state indices
#
def construct_index_2B(bas2B):
    index = { }
    for i, state in enumerate(bas2B):
        index[state] = i

    return index



#-----------------------------------------------------------------------------------
# transform matrices to particle-hole representation
#-----------------------------------------------------------------------------------
def ph_transform_2B(H2B, bas2B, idx2B, basph2B, idxph2B):
    dim = len(basph2B)
    H2B_ph = np.zeros((dim, dim))

    for i1, (a,b) in enumerate(basph2B):
        for i2, (c, d) in enumerate(basph2B):
            H2B_ph[i1, i2] -= H2B[idx2B[(a,d)], idx2B[(c,b)]]

    return H2B_ph

def inverse_ph_transform_2B(H2B_ph, bas2B, idx2B, basph2B, idxph2B):
    dim = len(bas2B)
    H2B = np.zeros((dim, dim))

    for i1, (a,b) in enumerate(bas2B):
        for i2, (c, d) in enumerate(bas2B):
            H2B[i1, i2] -= H2B_ph[idxph2B[(a,d)], idxph2B[(c,b)]]
    
    return H2B

#-----------------------------------------------------------------------------------
# commutator of matrices
#-----------------------------------------------------------------------------------
def commutator(a,b):
    return dot(a,b) - dot(b,a)

#-----------------------------------------------------------------------------------
# norms of off-diagonal Hamiltonian pieces
#-----------------------------------------------------------------------------------
def calc_fod_norm(H1B, user_data):
    particles = user_data["particles"]
    holes     = user_data["holes"]
    
    norm = 0.0
    for a in particles:
        for i in holes:
            norm += H1B[a,i]**2 + H1B[i,a]**2

    return np.sqrt(norm)

def calc_Gammaod_norm(H2B, user_data):
    particles = user_data["particles"]
    holes     = user_data["holes"]
    idx2B     = user_data["idx2B"]

    norm = 0.0
    for a in particles:    
        for b in particles:
            for i in holes:
                for j in holes:
                    norm += H2B[idx2B[(a,b)],idx2B[(i,j)]]**2 + H2B[idx2B[(i,j)],idx2B[(a,b)]]**2

    return np.sqrt(norm)

#-----------------------------------------------------------------------------------
# occupation number matrices
#-----------------------------------------------------------------------------------
def construct_occupation_1B(bas1B, holes, particles):
    dim = len(bas1B)
    occ = np.zeros(dim)

    for i in holes:
        occ[i] = 1.

    return occ

# diagonal matrix: n_a - n_b
def construct_occupationA_2B(bas2B, occ1B):
    dim = len(bas2B)
    occ = np.zeros((dim,dim))

    for i1, (i,j) in enumerate(bas2B):
        occ[i1, i1] = occ1B[i] - occ1B[j]

    return occ


# diagonal matrix: 1 - n_a - n_b
def construct_occupationB_2B(bas2B, occ1B):
    dim = len(bas2B)
    occ = np.zeros((dim,dim))

    for i1, (i,j) in enumerate(bas2B):
        occ[i1, i1] = 1. - occ1B[i] - occ1B[j]

    return occ

# diagonal matrix: n_a * n_b
def construct_occupationC_2B(bas2B, occ1B):
    dim = len(bas2B)
    occ = np.zeros((dim,dim))

    for i1, (i,j) in enumerate(bas2B):
        occ[i1, i1] = occ1B[i] * occ1B[j]

    return occ

#-----------------------------------------------------------------------------------
# generators
#-----------------------------------------------------------------------------------
def eta_brillouin(H1B, H2B, user_data):
    dim1B     = user_data["dim1B"]
    particles = user_data["particles"]
    holes     = user_data["holes"]
    idx2B     = user_data["idx2B"]

    # one-body part of the generator
    eta1B  = np.zeros_like(H1B)

    for a in particles:
        for i in holes:
            # (1-n_a)n_i - n_a(1-n_i) = n_i - n_a
            eta1B[a, i] =  f[a,i]
            eta1B[i, a] = -f[a,i]

    # two-body part of the generator
    eta2B = np.zeros_like(H2B)

    for a in particles:
        for b in particles:
            for i in holes:
                for j in holes:
                    val = H2B[idx2B[(a,b)], idx2B[(i,j)]]

                    eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
                    eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

    return eta1B, eta2B

def eta_imtime(H1B, H2B, user_data):
    dim1B     = user_data["dim1B"]
    particles = user_data["particles"]
    holes     = user_data["holes"]
    idx2B     = user_data["idx2B"]

    # one-body part of the generator
    eta1B  = np.zeros_like(f)

    for a in particles:
        for i in holes:
            dE = H1B[a,a] - H1B[i,i] + H2B[idx2B[(a,i)], idx2B[(a,i)]]
            val = np.sign(dE)*H1B[a,i]
            eta1B[a, i] =  val
            eta1B[i, a] = -val 

    # two-body part of the generator
    eta2B = np.zeros_like(H2B)

    for a in particles:
        for b in particles:
            for i in holes:
                for j in holes:
                    dE = ( 
                      H1B[a,a] + H1B[b,b] - H1B[i,i] - H1B[j,j]
                      + H2B[idx2B[(a,b)],idx2B[(a,b)]]
                      + H2B[idx2B[(i,j)],idx2B[(i,j)]]
                      - H2B[idx2B[(a,i)],idx2B[(a,i)]]
                      - H2B[idx2B[(a,j)],idx2B[(a,j)]]
                      - H2B[idx2B[(b,i)],idx2B[(b,i)]]
                      - H2B[idx2B[(b,j)],idx2B[(b,j)]]
                    )

                    val = np.sign(dE)*H2B[idx2B[(a,b)], idx2B[(i,j)]]

                    eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
                    eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

    return eta1B, eta2B


def eta_white(H1B, H2B, user_data):
    dim1B     = user_data["dim1B"]
    particles = user_data["particles"]
    holes     = user_data["holes"]
    idx2B     = user_data["idx2B"]

    # one-body part of the generator
    eta1B  = np.zeros_like(H1B)

    for a in particles:
        for i in holes:
            denom = H1B[a,a] - H1B[i,i] + H2B[idx2B[(a,i)], idx2B[(a,i)]]
            val = H1B[a,i]/denom
            eta1B[a, i] =  val
            eta1B[i, a] = -val 

    # two-body part of the generator
    eta2B = np.zeros_like(H2B)

    for a in particles:
        for b in particles:
            for i in holes:
                for j in holes:
                    denom = ( 
                      H1B[a,a] + H1B[b,b] - H1B[i,i] - H1B[j,j]
                      + H2B[idx2B[(a,b)],idx2B[(a,b)]]
                      + H2B[idx2B[(i,j)],idx2B[(i,j)]]
                      - H2B[idx2B[(a,i)],idx2B[(a,i)]]
                      - H2B[idx2B[(a,j)],idx2B[(a,j)]]
                      - H2B[idx2B[(b,i)],idx2B[(b,i)]]
                      - H2B[idx2B[(b,j)],idx2B[(b,j)]]
                    )

#                    if abs(denom) < 1.0e-10: print "%i %i %i %i\n"%(a,b,i,j)
                    val = H2B[idx2B[(a,b)], idx2B[(i,j)]] / denom

                    eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
                    eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

    return eta1B, eta2B


def eta_white_mp(H1B, H2B, user_data):
    dim1B     = user_data["dim1B"]
    particles = user_data["particles"]
    holes     = user_data["holes"]
    idx2B     = user_data["idx2B"]

    # one-body part of the generator
    eta1B  = np.zeros_like(H1B)

    for a in particles:
        for i in holes:
            denom = H1B[a,a] - H1B[i,i]
            val = H1B[a,i]/denom
            eta1B[a, i] =  val
            eta1B[i, a] = -val 

    # two-body part of the generator
    eta2B = np.zeros_like(Gamma)

    for a in particles:
        for b in particles:
            for i in holes:
                for j in holes:
                    denom = ( 
                      H1B[a,a] + H1B[b,b] - H1B[i,i] - H1B[j,j]
                    )

                    val = H2B[idx2B[(a,b)], idx2B[(i,j)]] / denom

                    eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
                    eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

    return eta1B, eta2B

def eta_white_atan(H1B, H2B, user_data):
    dim1B     = user_data["dim1B"]
    particles = user_data["particles"]
    holes     = user_data["holes"]
    idx2B     = user_data["idx2B"]

    # one-body part of the generator
    eta1B  = np.zeros_like(H1B)

    for a in particles:
        for i in holes:
            denom = H1B[a,a] - H1B[i,i] + H2B[idx2B[(a,i)], idx2B[(a,i)]]
            val = 0.5 * np.arctan(2 * H1B[a,i]/denom)
            eta1B[a, i] =  val
            eta1B[i, a] = -val 

    # two-body part of the generator
    eta2B = np.zeros_like(H2B)

    for a in particles:
        for b in particles:
            for i in holes:
                for j in holes:
                    denom = ( 
                      H1B[a,a] + H1B[b,b] - H1B[i,i] - H1B[j,j]
                      + H2B[idx2B[(a,b)],idx2B[(a,b)]]
                      + H2B[idx2B[(i,j)],idx2B[(i,j)]]
                      - H2B[idx2B[(a,i)],idx2B[(a,i)]]
                      - H2B[idx2B[(a,j)],idx2B[(a,j)]]
                      - H2B[idx2B[(b,i)],idx2B[(b,i)]]
                      - H2B[idx2B[(b,j)],idx2B[(b,j)]]
                    )

                    val = 0.5 * np.arctan(2 * H2B[idx2B[(a,b)], idx2B[(i,j)]] / denom)

                    eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
                    eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

    return eta1B, eta2B


def eta_wegner(H1B, H2B, user_data):

    dim1B     = user_data["dim1B"]
    holes     = user_data["holes"]
    particles = user_data["particles"]
    bas2B     = user_data["bas2B"]
    basph2B   = user_data["basph2B"]
    idx2B     = user_data["idx2B"]
    idxph2B   = user_data["idxph2B"]
    occB_2B   = user_data["occB_2B"]
    occC_2B   = user_data["occC_2B"]
    occphA_2B = user_data["occphA_2B"]


    # split Hamiltonian in diagonal and off-diagonal parts
    fd      = np.zeros_like(H1B)
    fod     = np.zeros_like(H1B)
    Gammad  = np.zeros_like(H2B)
    Gammaod = np.zeros_like(H2B)

    for a in particles:
        for i in holes:
            fod[a, i] = H1B[a,i]
            fod[i, a] = H1B[i,a]
    fd = H1B - fod

    for a in particles:
        for b in particles:
            for i in holes:
                for j in holes:
                    Gammaod[idx2B[(a,b)], idx2B[(i,j)]] = H2B[idx2B[(a,b)], idx2B[(i,j)]]
                    Gammaod[idx2B[(i,j)], idx2B[(a,b)]] = H2B[idx2B[(i,j)], idx2B[(a,b)]]
    Gammad = H2B - Gammaod


    #############################        
    # one-body flow equation  
    eta1B  = np.zeros_like(H1B)

    # 1B - 1B
    eta1B += commutator(fd, fod)

    # 1B - 2B
    for p in range(dim1B):
        for q in range(dim1B):
            for i in holes:
                for a in particles:
                    eta1B[p,q] += (
                      fd[i,a]  * Gammaod[idx2B[(a, p)], idx2B[(i, q)]] 
                      - fd[a,i]  * Gammaod[idx2B[(i, p)], idx2B[(a, q)]] 
                      - fod[i,a] * Gammad[idx2B[(a, p)], idx2B[(i, q)]] 
                      + fod[a,i] * Gammad[idx2B[(i, p)], idx2B[(a, q)]]
                    )

    # 2B - 2B
    # n_a n_b nn_c + nn_a nn_b n_c = n_a n_b + (1 - n_a - n_b) * n_c
    GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))
    for p in range(dim1B):
        for q in range(dim1B):
            for i in holes:
                eta1B[p,q] += (
                  0.5*GammaGamma[idx2B[(i,p)], idx2B[(i,q)]] 
                  - transpose(GammaGamma)[idx2B[(i,p)], idx2B[(i,q)]]
                )

    GammaGamma = dot(Gammad, dot(occC_2B, Gammaod))
    for p in range(dim1B):
        for q in range(dim1B):
            for r in range(dim1B):
                eta1B[p,q] += (
                  0.5*GammaGamma[idx2B[(r,p)], idx2B[(r,q)]] 
                  - transpose(GammaGamma)[idx2B[(r,p)], idx2B[(r,q)]]
                )


    #############################        
    # two-body flow equation  
    eta2B = np.zeros_like(H2B)

    # 1B - 2B
    for p in range(dim1B):
        for q in range(dim1B):
            for r in range(dim1B):
                for s in range(dim1B):
                    for t in range(dim1B):
                        eta2B[idx2B[(p,q)],idx2B[(r,s)]] += (
                          fd[p,t] * Gammaod[idx2B[(t,q)],idx2B[(r,s)]] 
                          + fd[q,t] * Gammaod[idx2B[(p,t)],idx2B[(r,s)]] 
                          - fd[t,r] * Gammaod[idx2B[(p,q)],idx2B[(t,s)]] 
                          - fd[t,s] * Gammaod[idx2B[(p,q)],idx2B[(r,t)]]
                          - fod[p,t] * Gammad[idx2B[(t,q)],idx2B[(r,s)]] 
                          - fod[q,t] * Gammad[idx2B[(p,t)],idx2B[(r,s)]] 
                          + fod[t,r] * Gammad[idx2B[(p,q)],idx2B[(t,s)]] 
                          + fod[t,s] * Gammad[idx2B[(p,q)],idx2B[(r,t)]]
                        )

  
    # 2B - 2B - particle and hole ladders
    # Gammad.occB.Gammaod
    GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))

    eta2B += 0.5 * (GammaGamma - transpose(GammaGamma))

    # 2B - 2B - particle-hole chain
    
    # transform matrices to particle-hole representation and calculate 
    # Gammad_ph.occA_ph.Gammaod_ph
    Gammad_ph = ph_transform_2B(Gammad, bas2B, idx2B, basph2B, idxph2B)
    Gammaod_ph = ph_transform_2B(Gammaod, bas2B, idx2B, basph2B, idxph2B)

    GammaGamma_ph = dot(Gammad_ph, dot(occphA_2B, Gammaod_ph))

    # transform back to standard representation
    GammaGamma    = inverse_ph_transform_2B(GammaGamma_ph, bas2B, idx2B, basph2B, idxph2B)

    # commutator / antisymmetrization
    work = np.zeros_like(GammaGamma)
    for i1, (i,j) in enumerate(bas2B):
        for i2, (k,l) in enumerate(bas2B):
            work[i1, i2] -= (
              GammaGamma[i1, i2] 
              - GammaGamma[idx2B[(j,i)], i2] 
              - GammaGamma[i1, idx2B[(l,k)]] 
              + GammaGamma[idx2B[(j,i)], idx2B[(l,k)]]
            )
    GammaGamma = work

    eta2B += GammaGamma


    return eta1B, eta2B


#-----------------------------------------------------------------------------------
# derivatives 
#-----------------------------------------------------------------------------------
def flow_imsrg2(eta1B, eta2B, f, Gamma, user_data):

    dim1B     = user_data["dim1B"]
    holes     = user_data["holes"]
    particles = user_data["particles"]
    bas2B     = user_data["bas2B"]
    idx2B     = user_data["idx2B"]
    basph2B   = user_data["basph2B"]
    idxph2B   = user_data["idxph2B"]
    occB_2B   = user_data["occB_2B"]
    occC_2B   = user_data["occC_2B"]
    occphA_2B = user_data["occphA_2B"]


    #############################        
    # one-body flow equation  
    df  = np.zeros_like(f)

    # 1B - 1B
    df += commutator(eta1B, f)

    # 1B - 2B
    for p in range(dim1B):
        for q in range(dim1B):
            for i in holes:
                for a in particles:
                    df[p,q] += (
                      eta1B[i,a] * Gamma[idx2B[(a, p)], idx2B[(i, q)]] 
                      - eta1B[a,i] * Gamma[idx2B[(i, p)], idx2B[(a, q)]] 
                      - f[i,a] * eta2B[idx2B[(a, p)], idx2B[(i, q)]] 
                      + f[a,i] * eta2B[idx2B[(i, p)], idx2B[(a, q)]]
                    )

    # 2B - 2B
    # n_a n_b nn_c + nn_a nn_b n_c = n_a n_b + (1 - n_a - n_b) * n_c
    etaGamma = dot(eta2B, dot(occB_2B, Gamma))
    for p in range(dim1B):
        for q in range(dim1B):
            for i in holes:
                df[p,q] += 0.5*(
                  etaGamma[idx2B[(i,p)], idx2B[(i,q)]] 
                  + transpose(etaGamma)[idx2B[(i,p)], idx2B[(i,q)]]
                )

    etaGamma = dot(eta2B, dot(occC_2B, Gamma))
    for p in range(dim1B):
        for q in range(dim1B):
            for r in range(dim1B):
                df[p,q] += 0.5*(
                  etaGamma[idx2B[(r,p)], idx2B[(r,q)]] 
                  + transpose(etaGamma)[idx2B[(r,p)], idx2B[(r,q)]] 
                )


    #############################        
    # two-body flow equation  
    dGamma = np.zeros_like(Gamma)

    # 1B - 2B
    for p in range(dim1B):
        for q in range(dim1B):
            for r in range(dim1B):
                for s in range(dim1B):
                    for t in range(dim1B):
                        dGamma[idx2B[(p,q)],idx2B[(r,s)]] += (
                          eta1B[p,t] * Gamma[idx2B[(t,q)],idx2B[(r,s)]] 
                          + eta1B[q,t] * Gamma[idx2B[(p,t)],idx2B[(r,s)]] 
                          - eta1B[t,r] * Gamma[idx2B[(p,q)],idx2B[(t,s)]] 
                          - eta1B[t,s] * Gamma[idx2B[(p,q)],idx2B[(r,t)]]
                          - f[p,t] * eta2B[idx2B[(t,q)],idx2B[(r,s)]] 
                          - f[q,t] * eta2B[idx2B[(p,t)],idx2B[(r,s)]] 
                          + f[t,r] * eta2B[idx2B[(p,q)],idx2B[(t,s)]] 
                          + f[t,s] * eta2B[idx2B[(p,q)],idx2B[(r,t)]]
                        )

    
    # 2B - 2B - particle and hole ladders
    # eta2B.occB.Gamma
    etaGamma = dot(eta2B, dot(occB_2B, Gamma))

    dGamma += 0.5 * (etaGamma + transpose(etaGamma))

    # 2B - 2B - particle-hole chain
    
    # transform matrices to particle-hole representation and calculate 
    # eta2B_ph.occA_ph.Gamma_ph
    eta2B_ph = ph_transform_2B(eta2B, bas2B, idx2B, basph2B, idxph2B)
    Gamma_ph = ph_transform_2B(Gamma, bas2B, idx2B, basph2B, idxph2B)

    etaGamma_ph = dot(eta2B_ph, dot(occphA_2B, Gamma_ph))

    # transform back to standard representation
    etaGamma    = inverse_ph_transform_2B(etaGamma_ph, bas2B, idx2B, basph2B, idxph2B)

    # commutator / antisymmetrization
    work = np.zeros_like(etaGamma)
    for i1, (i,j) in enumerate(bas2B):
        for i2, (k,l) in enumerate(bas2B):
            work[i1, i2] -= (
              etaGamma[i1, i2] 
              - etaGamma[idx2B[(j,i)], i2] 
              - etaGamma[i1, idx2B[(l,k)]] 
              + etaGamma[idx2B[(j,i)], idx2B[(l,k)]]
            )
    etaGamma = work

    dGamma += etaGamma


    return df, dGamma


#-----------------------------------------------------------------------------------
# derivative wrapper
#-----------------------------------------------------------------------------------
def get_operator_from_y(y, dim1B, dim2B):
  
    # reshape the solution vector into 0B, 1B, 2B pieces
    ptr = 0
    one_body = reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))

    ptr += dim1B*dim1B
    two_body = reshape(y[ptr:ptr+dim2B*dim2B], (dim2B, dim2B))

    return one_body,two_body


def derivative_wrapper(t, y, user_data):

    dim1B = user_data["dim1B"]
    dim2B = dim1B*dim1B

    holes     = user_data["holes"]
    particles = user_data["particles"]
    bas1B     = user_data["bas1B"]
    bas2B     = user_data["bas2B"]
    basph2B   = user_data["basph2B"]
    idx2B     = user_data["idx2B"]
    idxph2B   = user_data["idxph2B"]
    occA_2B   = user_data["occA_2B"]
    occB_2B   = user_data["occB_2B"]
    occC_2B   = user_data["occC_2B"]
    occphA_2B = user_data["occphA_2B"]
    calc_eta  = user_data["calc_eta"]
    calc_rhs  = user_data["calc_rhs"]

    # extract operator pieces from solution vector
    f, Gamma = get_operator_from_y(y, dim1B, dim2B)

    # calculate the generator
    eta1B, eta2B = calc_eta(f, Gamma, user_data)

    # calculate the right-hand side
    df, dGamma = calc_rhs(eta1B, eta2B, f, Gamma, user_data)

    # convert derivatives into linear array
    dy   = np.append(reshape(df, -1), reshape(dGamma, -1))

    # share data
    user_data["eta_norm"] = np.linalg.norm(eta1B,ord='fro')+np.linalg.norm(eta2B,ord='fro')
    
    return dy

#-----------------------------------------------------------------------------------
# pairing Hamiltonian
#-----------------------------------------------------------------------------------
def pairing_hamiltonian(delta, g, ff, h, user_data):
    bas1B = user_data["bas1B"]
    bas2B = user_data["bas2B"]
    idx2B = user_data["idx2B"]

    dim = len(bas1B)
    H1B = np.zeros((dim,dim))

    for i in bas1B:
        H1B[i,i] = delta*np.floor_divide(i, 2)

    dim = len(bas2B)
    H2B = np.zeros((dim, dim))

    # pairing interaction
    # spin up states have even indices, spin down the next odd index
    # A^{p+p-}_{q-q+}
    for (i, j) in bas2B:
        if (i % 2 == 0 and j == i+1):
            for (k, l) in bas2B:
                if (k % 2 == 0 and l == k+1):
                    H2B[idx2B[(i,j)],idx2B[(k,l)]] = -0.5*g
                    H2B[idx2B[(j,i)],idx2B[(k,l)]] = 0.5*g
                    H2B[idx2B[(i,j)],idx2B[(l,k)]] = 0.5*g
                    H2B[idx2B[(j,i)],idx2B[(l,k)]] = -0.5*g
    
    # one particle-hole interaction
    # A^{p+p-}_{q-r+} + A^{r+q-}_{p-p+}
    for (i, j) in bas2B:
        if (i % 2 == 0 and j == i+1):
            for (k, l) in bas2B:
                if (k % 2 == 0 and l % 2 == 1):
                    H2B[idx2B[(i,j)],idx2B[(k,l)]] -= 0.5*ff
                    H2B[idx2B[(j,i)],idx2B[(k,l)]] += 0.5*ff
                    H2B[idx2B[(i,j)],idx2B[(l,k)]] += 0.5*ff
                    H2B[idx2B[(j,i)],idx2B[(l,k)]] -= 0.5*ff
                    
                    H2B[idx2B[(l,k)],idx2B[(j,i)]] -= 0.5*ff
                    H2B[idx2B[(k,l)],idx2B[(j,i)]] += 0.5*ff
                    H2B[idx2B[(l,k)],idx2B[(i,j)]] += 0.5*ff
                    H2B[idx2B[(k,l)],idx2B[(i,j)]] -= 0.5*ff

    # two particle-hole interaction
    # A^{p+q-}_{r-s+} + A^{s+r-}_{q-p+}
    for (i, j) in bas2B:
        if (i % 2 == 0 and j % 2 == 1):
            for (k, l) in bas2B:
                if (k % 2 == 0 and l % 2 == 1):
                    H2B[idx2B[(i,j)],idx2B[(k,l)]] -= 0.5*h
                    H2B[idx2B[(j,i)],idx2B[(k,l)]] += 0.5*h
                    H2B[idx2B[(i,j)],idx2B[(l,k)]] += 0.5*h
                    H2B[idx2B[(j,i)],idx2B[(l,k)]] -= 0.5*h

    return H1B, H2B


def bitCount(int_type):
    """ Count bits set in integer """
    count = 0
    while(int_type):
        int_type &= int_type - 1
        count += 1
    return(count)


# testBit() returns a nonzero result, 2**offset, if the bit at 'offset' is one.

def testBit(int_type, offset):
    mask = 1 << offset
    return(int_type & mask) >> offset

# setBit() returns an integer with the bit at 'offset' set to 1.

def setBit(int_type, offset):
    mask = 1 << offset
    return(int_type | mask)

# clearBit() returns an integer with the bit at 'offset' cleared.

def clearBit(int_type, offset):
    mask = ~(1 << offset)
    return(int_type & mask)

# toggleBit() returns an integer with the bit at 'offset' inverted, 0 -> 1 and 1 -> 0.

def toggleBit(int_type, offset):
    mask = 1 << offset
    return(int_type ^ mask)

# binary string made from number

def bin0(s):
    return str(s) if s<=1 else bin0(s>>1) + str(s&1)

def bin(s, L = 0):
    ss = bin0(s)
    if L > 0:
        return '0'*(L-len(ss)) + ss
    else:
        return ss



class Slater:
    """ Class for Slater determinants """
    def __init__(self, x = int(0)):
        self.word = int(x)
    
    def create(self, j):
        s = 0
        if self.word < 0:
            self.word = abs(self.word)
            ss = -1
        else:
            ss = 1
        isset = testBit(self.word, j)
        if isset == 0:
            bits = bitCount(self.word & ((1<<j)-1))
            s = pow(-1, bits)
            self.word = ss*s*setBit(self.word, j)
        
        else:
            self.word = 0
        
        return self.word
    
    def annihilate(self, j):
        s = 0
        if self.word < 0:
            self.word = abs(self.word)
            ss = -1
        else:
            ss = 1
        isset = testBit(self.word, j)
        if isset == 1:
            bits = bitCount(self.word & ((1<<j)-1))
            s = pow(-1, bits)
            self.word = ss*s*clearBit(self.word, j)
        else:
            self.word = 0
        
        return self.word

def basis0(nparticles, Nstates):
    N_sp   = []
    states = []
    temp   = []
    M      = []
    
    for i in range(Nstates):
        N_sp.append(i)
    
    for bas in itertools.combinations(N_sp, nparticles):
        temp.append(bas)
    for i in temp:
        phi = Slater()
        m   = 0
        
        for j in i:
            if j%2 == 0:
                m += 1
            else:
                m -= 1
            a = phi.create(j)
        states.append(a)
        M.append(m)
    
    return states, M


# collect states based on M
def M_scheme(states, M):
    states_0 = []
    states_2 = []
    states_2_= []
    states_4 = []
    states_4_= []
    for i in range(len(M)):
        
        if M[i] == 0:
            states_0.append(states[i])
        if M[i] == 2:
            states_2.append(states[i])
        if M[i] == -2:
            states_2_.append(states[i])
        if M[i] == 4:
            states_4.append(states[i])
        if M[i] == -4:
            states_4_.append(states[i])

    return states_0, states_2, states_2_, states_4, states_4_


# indexing of basis
def construct_index(nparticles, Nstates, basis0, M_scheme):
    
    states, M = basis0(nparticles, Nstates)
    states_0, states_2, states_2_, states_4, states_4_ = M_scheme(states, M)
    index0 = { }
    
    stat_0 = []
    for i in range(len(states_0)):
        for l in range(Nstates):
            if l%2 == 1:
                continue
            a = testBit(states_0[i],l)
            b = testBit(states_0[i],l+1)
            if (a == 1 and b == 1) or (a == 0 and b == 0):
                if l == Nstates-2:
                    stat_0.append(states_0[i])
                    break
                continue
            else:
                break

    stat_1 = []
    for i in range(len(states_0)):
        for l in range(Nstates):
            if l%2 == 1:
                continue
            a = testBit(states_0[i],l)
            b = testBit(states_0[i],l+1)
            if (a == 1 and b == 0) or (a == 0 and b == 1):
                if l == Nstates-2:
                    stat_1.append(states_0[i])
                    break
                continue
            else:
                break


    for i in stat_0:
        states_0.remove(i)
    states_0 = stat_0 + states_0
    for i in stat_1:
         states_0.remove(i)
    states_0 = states_0 + stat_1

    for i, state in enumerate(states_0):
        index0[state] = i
    return index0


def Hamiltonian(H1B, H2B, user_data):
    
    index0     = user_data["index0"]
    nparticles = user_data["nparticles"]
    Nstates    = user_data["Nstates"]
    basis0     = user_data["basis0"]
    Slater     = user_data["Slater"]
    M_scheme   = user_data["M_scheme"]
    bas1B      = user_data["bas1B"]
    bas2B      = user_data["bas2B"]
    idx2B      = user_data["idx2B"]
    dim1B      = user_data["dim1B"]
    
    states, M = basis0(nparticles, Nstates)
    states_0, states_2, states_2_, states_4, states_4_ = M_scheme(states, M)
    H = np.zeros((len(states_0), len(states_0)))
    
    # one-body interaction
    for i in range(len(states_0)):
        for p in range(dim1B):
            for q in range(dim1B):
                phi = Slater(states_0[i])
                temp1 = H1B[p,q]
                if abs(temp1) > 1e-14:
                    phi1 = phi.annihilate(q)
                    if phi1 != 0:
                        ph = Slater(phi1)
                        phi2 = ph.create(p)
                        if phi2 != 0:
                            for m in range(len(states_0)):
                                if phi2 == states_0[m]:
                                    H[index0[states_0[m]],index0[states_0[i]]] += temp1
                                    break
                                if phi2 == -states_0[m]:
                                    H[index0[states_0[m]],index0[states_0[i]]] -= temp1
                                    break
    
    for i in range(len(states_0)):
        for (p,q) in bas2B:
            for (s,r) in bas2B:
                phi = Slater(states_0[i])
                temp2 = H2B[idx2B[(p,q)],idx2B[(r,s)]]
                if abs(temp2) > 1e-14:
                    phi1 = phi.annihilate(r)
                    phi1 = phi.annihilate(s)
                    if phi1 != 0:
                        ph = Slater(phi1)
                        phi2 = ph.create(q)
                        if phi2 != 0:
                            phh = Slater(phi2)
                            phi3 = phh.create(p)
                            if phi3 != 0:
                                for m in range(len(states_0)):
                                    if phi3 == states_0[m]:
                                        H[index0[states_0[m]],index0[states_0[i]]] += 0.25*temp2
                                        break
                                    if phi3 == -states_0[m]:
                                        H[index0[states_0[m]],index0[states_0[i]]] -= 0.25*temp2
                                        break

    return H


def derivative(y, t, dim):
    
    # reshape the solution vector into a dim x dim matrix
    H = reshape(y, (dim, dim))
    
    # extract diagonal Hamiltonian...
    Hd  = diag(diag(H))
    
    # ... and construct off-diagonal the Hamiltonian
    Hod = H-Hd
    
    # calculate the generator
    eta = commutator(Hd, Hod)
    
    # dH is the derivative in matrix form
    dH  = commutator(eta, H)
    
    # convert dH into a linear array for the ODE solver
    dy = reshape(dH, -1)
    
    return dy


def imageshow(object):
    im = plt.imshow(object,cmap=plt.get_cmap('RdBu_r'),interpolation='nearest',vmin = -g, vmax = g)
    plt.colorbar(im)
    plt.show()

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

# grab delta and g from the command line
delta      = 1.
g          = 0.5
ff         = 0.0
h          = 0.0

nparticles  = 4

# setup shared data
dim1B     = 8
Nstates   = 8
    
index0 = construct_index(nparticles, Nstates, basis0, M_scheme)
states, M = basis0(nparticles, Nstates)
states_0, states_2, states_2_, states_4, states_4_ = M_scheme(states, M)

holes     = [0,1,2,3]
particles = [4,5,6,7]
    

# basis definitions
bas1B     = range(dim1B)
bas2B     = construct_basis_2B(holes, particles)
basph2B   = construct_basis_ph2B(holes, particles)

idx2B     = construct_index_2B(bas2B)
idxph2B   = construct_index_2B(basph2B)

# occupation number matrices
occ1B     = construct_occupation_1B(bas1B, holes, particles)
occA_2B   = construct_occupationA_2B(bas2B, occ1B)
occB_2B   = construct_occupationB_2B(bas2B, occ1B)
occC_2B   = construct_occupationC_2B(bas2B, occ1B)

occphA_2B = construct_occupationA_2B(basph2B, occ1B)

# store shared data in a dictionary, so we can avoid passing the basis
# lookups etc. as separate parameters all the time
user_data  = {
    "nparticles":  nparticles,
    "Nstates":    Nstates,
    "index0":     index0,
    "basis0":     basis0,
    "Slater":     Slater,
    "M_scheme":   M_scheme,
    "dim1B":      dim1B,
    "holes":      holes,
    "particles":  particles,
    "bas1B":      bas1B,
    "bas2B":      bas2B,
    "basph2B":    basph2B,
    "idx2B":      idx2B,
    "idxph2B":    idxph2B,
    "occ1B":      occ1B,
    "occA_2B":    occA_2B,
    "occB_2B":    occB_2B,
    "occC_2B":    occC_2B,
    "occphA_2B":  occphA_2B,

    "eta_norm":   0.0,                # variables for sharing data between ODE solver
    "dE":         0.0,                # and main routine

    "calc_eta":   eta_white,          # specify the generator (function object)
    "calc_rhs":   flow_imsrg2         # specify the right-hand side and truncation
}

# set up initial Hamiltonian
H1B, H2B = pairing_hamiltonian(delta, g, ff, h, user_data)
imageshow(H1B)
imageshow(H2B)

Hamilton0 = Hamiltonian(H1B, H2B, user_data)
imageshow(Hamilton0)

# reshape Hamiltonian into a linear array (initial ODE vector)
y0   = np.append(reshape(H1B, -1), reshape(H2B, -1))

# integrate flow equations 
solver = sp.integrate.ode(derivative_wrapper,jac=None)
solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
solver.set_f_params(user_data)
solver.set_initial_value(y0, 0.)

sfinal = 50
ds = 0.1

print("-" * 118)


while solver.successful() and solver.t < sfinal:
    ys = solver.integrate(sfinal, step=True)
        
    dim2B = dim1B*dim1B
    H1B, H2B = get_operator_from_y(ys, dim1B, dim2B)
    

    norm_fod     = calc_fod_norm(H1B, user_data)
    norm_Gammaod = calc_Gammaod_norm(H2B, user_data)
    
    Hamilton = Hamiltonian(H1B, H2B, user_data)
    
    

    print("%8.5f   %10.8f   %10.8f   %10.8f"%(
       solver.t,  user_data["eta_norm"], norm_fod, norm_Gammaod))
    if abs(user_data["eta_norm"])<10e-7: break

imageshow(H1B)
imageshow(H2B)

Hamilton = Hamiltonian(H1B, H2B, user_data)
imageshow(Hamilton)
print (Hamilton[0,0])




