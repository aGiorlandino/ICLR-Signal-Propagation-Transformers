import numpy as np

def get_attention_update(p, beta, q=1):
    beta_c = np.sqrt(2/(1-p))
    if beta < beta_c:
        q_att = p
        p_att = p
    else:
        q_att = p + (1 - p)*(1 - beta_c/beta)
        p_att = p
    return q_att, p_att

def rho(c):
    return 1/np.pi * ( np.sqrt(1 - c**2) + c * (np.pi - np.arccos(c)) )

def get_MLP_update(p, var_w, var_b, q=1):
    q1 = var_w * q + var_b
    p1 = var_w * p + var_b
    q_new = var_w * q1/2 + var_b
    p_new = var_w * q1/2 * rho(c=p1/q1) + var_b 
    return q_new, p_new

    
def get_block_update_wres(p, beta, var_w, var_v, var_b=1/768, q=1, alpha_A=1, alpha_MLP=1):
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
