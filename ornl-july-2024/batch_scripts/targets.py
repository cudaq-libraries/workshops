import cudaq

targets = cudaq.get_targets()

print("available cudaq targets:")

for t in targets:
         print(t)
