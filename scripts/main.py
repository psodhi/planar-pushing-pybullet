from Environments import EnvKukaBlock
import matplotlib.pyplot as plt
import pdb


def main():
    vis_flag = True

    # sim params
    params = {}
    params['tstart'] = 0.0
    params['tend'] = 10.0
    params['dt'] = 1e-3

    envkb = EnvKukaBlock(params, vis_flag)

    num_runs = 1
    for run in range(0, num_runs):
        envkb.simulate()

    logger = envkb.get_logger()
    logger.visualize_contact_data()

if __name__ == "__main__":
    main()
