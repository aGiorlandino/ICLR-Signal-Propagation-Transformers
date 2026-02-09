import numpy as np
import scipy.integrate as spi
from scipy.stats import norm
import matplotlib.pyplot as plt
import multiprocessing
import torch

def phi(x):
    return np.tanh(x) 

def u1(z1, q_prev):
    return np.sqrt(q_prev) * z1

def u2(z1, z2, q_prev, c_prev):
    return np.sqrt(q_prev) * (c_prev * z1 + np.sqrt(1 - c_prev**2) * z2)

def p_integrand(z1, z2, q_prev, c_prev):
    u1_val = u1(z1, q_prev)
    u2_val = u2(z1, z2, q_prev, c_prev)
    return phi(u1_val) * phi(u2_val) * norm.pdf(z1) * norm.pdf(z2)

def compute_p(q_prev, p_prev, var_b, var_w):
    c_prev = p_prev /q_prev
    c_prev = np.clip(c_prev, -1, 1)
    integral, _ = spi.dblquad(lambda z1, z2: p_integrand(z1, z2, q_prev, c_prev),
                              -10, 10, -10, 10)
    
    return var_b + var_w * integral

def q_integrand(z, q_prev):
    return phi(np.sqrt(q_prev) * z)**2 * norm.pdf(z)

def compute_q(q_prev, var_b, var_w):
    integral, _ = spi.quad(q_integrand, -10, 10, args=(q_prev,))
    return var_b + var_w * integral

def get_attention_update(p, beta, q=1):
    beta_c = np.sqrt(2/(1-p))
    if beta < beta_c:
        q_att = p
        p_att = p
    else:
        q_att = p + (1 - p)*(1 - beta_c/beta)
        p_att = p
    return q_att, p_att

def get_MLP_update(p, var_w, var_b, q=1):
    q1 = var_w * q + var_b
    p1 = var_w * p + var_b
    p_new = compute_p(q_prev=q1, p_prev=p1, var_w=var_w, var_b=var_b)
    q_new = compute_q(q1, var_b, var_w)
    return q_new, p_new
    

def get_block_update(p, beta, var_w, var_b, q=1):
    q, p = get_attention_update(p, beta, q)
    p = p/q #LN
    q = 1 #LN 
    if np.isclose(p, 1):
        return 1
    q, p = get_MLP_update(p, var_w, var_b, q)
    return p/q

    
def get_block_update_wres(p, beta, var_w, var_v, var_b, q=1, alpha_A=1, alpha_MLP=1):
    p = p/q
    if p == 1:
        return 1
    q_att, p_att = get_attention_update(p=p, beta= beta, q=1)
    p_0 = ( var_v* p_att + var_b + alpha_A**2 * p) / (var_v * q_att + var_b + alpha_A**2)
    q_0 = 1
    if p_0 == 1:
        return 1
    q_MLP, p_MLP = get_MLP_update(p=p_0, var_w=var_w, var_b=var_b, q=q_0)
    p_block = (p_MLP + alpha_MLP**2 * p_0) / (q_MLP + alpha_MLP**2)
    return p_block