# This for testing the optimality of heuristic. 
import numpy as np
import gurobipy as gp

def generate_deltaG(k: int):
    randarr = np.random.choice(500, size=k)
    total = np.sum(randarr)
    return np.array([r / total for r in randarr])


def test_optimality(P1, R, k):
    '''
    k is the number of intervals to partition.
    '''
    P2 = 1 - P1
    m = gp.Model()
    deltaG = generate_deltaG(k)
    deltaH = generate_deltaG(k)

    # Indicator variables for putting a backp filter
    ind_var = [m.addVar(vtype='B', name='ind_{}'.format(i)) for i in range(k)]

    # f_i, the target fpr for each backup filter. Not used if interval i has ind_i = 0. 
    # inv_var[i] = 1 / f_i
    # log_var = log_2(inv_i)
    fpr_var = [m.addVar(lb=0, ub=1, vtype='C', name='f_{}'.format(i)) for i in range(k)]
    inv_var = [m.addVar(lb=1, vtype='C', name='invvar_{}'.format(i)) for i in range(k)]
    log_var = [m.addVar(vtype='C', name='logvar_{}'.format(i)) for i in range(k)]
    for i in range(k):
        m.addConstr(inv_var[i] * fpr_var[i] == 1, name='inv helper constraint {}'.format(i))
        m.addGenConstrLogA(inv_var[i], log_var[i], 2, name='log helper constraint {}'.format(i))

    # Objective
    m.setObjective(gp.quicksum([deltaG[i] * log_var[i] for i in range(k)]))

    # Error rate constraint
    m.addConstr(P1 * gp.quicksum([(1-ind_var[i]) * deltaG[i] for i in range(k)]) 
                    + P2 * gp.quicksum([ind_var[i] * deltaH[i] * fpr_var[i] for i in range(k)])
                    <= R
            , name="error rate constraint")
    
    m.Params.NonConvex = 2
    m.optimize()
    print('A')
    print(m.objVal)

test_optimality(0.2, 0.05, 10)