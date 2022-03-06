# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:31:35 2022

@author: joaov
"""

import pandapower as pp
#create empty net
net = pp.create_empty_network() 

min_vm_pu = .95
max_vm_pu = 1.05

#create buses
bus1 = pp.create_bus(net, vn_kv=110., min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
bus2 = pp.create_bus(net, vn_kv=110., min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
bus3 = pp.create_bus(net, vn_kv=110., min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)

#create bus elements
l1 = pp.create_line(net, bus1, bus2, length_km=1., std_type='149-AL1/24-ST1A 110.0')
l2 = pp.create_line(net, bus2, bus3, length_km=1., std_type='149-AL1/24-ST1A 110.0')
l3 = pp.create_line(net, bus3, bus1, length_km=1., std_type='149-AL1/24-ST1A 110.0')

#create loads
pp.create_load(net, bus3, p_mw=300)

#create generators
g1 = pp.create_gen(net, bus1, p_mw=200, min_p_mw=0,max_p_mw=300, controllable=True, slack=True)
g2 = pp.create_gen(net, bus2, p_mw=0, min_p_mw=0,max_p_mw=300,controllable=True)

pp.create_poly_cost(net, element=g1, et="gen", cp1_eur_per_mw=30)
pp.create_poly_cost(net, element=g2, et="gen", cp1_eur_per_mw=30)


pp.runopp(net)

print("Resultados geradores \n", net.res_gen, "\n")

print(net.res_line)
print(net.res_load)
print(net.res_bus)
print(net.res_cost)