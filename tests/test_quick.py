from brain.agent import run_agent
import time

cmds=['open notepad and type hello','open calculator','volume down','search youtube for veritasium','press ctrl s','take a screenshot']
for c in cmds:
    t0=time.perf_counter(); r=run_agent(c); dt=time.perf_counter()-t0; print(c,'->',r,'took',round(dt,3));
