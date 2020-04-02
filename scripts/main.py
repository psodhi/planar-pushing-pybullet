from Environments import EnvKukaBlock
import matplotlib.pyplot as plt
import pdb

def main():
    vis_flag = True

    envkb = EnvKukaBlock(vis_flag)
    num_runs = 1

    for run in range(0, num_runs):
        envkb.simulate()


if __name__ == "__main__":
    main()