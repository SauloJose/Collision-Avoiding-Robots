import irsim
import numpy as np

env = irsim.make("envs/orca_conv.yaml")   # ajuste o caminho se necessário

print("=" * 60)
print("DIR(env) com 'obs'/'obj'/'land':")
print([a for a in dir(env) if any(k in a.lower() for k in ("obs", "obj", "land", "wall"))])
print()

print("=" * 60)
print("env.obstacle_list :", getattr(env, "obstacle_list", "NÃO EXISTE"))
print("len               :", len(getattr(env, "obstacle_list", [])))
print()

for name in ("obs_list", "obstacles", "object_list", "obstacle_list"):
    val = getattr(env, name, None)
    if val is not None:
        print(f"env.{name} -> len = {len(val)}")
        for i, o in enumerate(val):
            print(f"  [{i}] type       = {type(o).__name__}")
            print(f"      state      = {getattr(o, 'state', None)}")
            print(f"      radius     = {getattr(o, 'radius', None)}")
            print(f"      vertices   = {getattr(o, 'vertices', None)}")
            print(f"      center     = {getattr(o, 'center', None)}")
            print(f"      shape      = {getattr(o, 'shape', None)}")
            print(f"      kinematics = {getattr(o, 'kinematics', None)}")
        print()

print("=" * 60)
print("env.robot_list len:", len(getattr(env, "robot_list", [])))
env.end()