import os
from args_parser import args_parser
from mainRL import Run_DQL, Run_ExpectedTaskDDQN, Run_DDQL, Run_DuelingDQL, Run_D3QL, Run_Sarsa, runShortestLatencyGreedy, runShortestExtraTimeGreedy
from Actor_critic import runAC
from Reinforce_algorithms import runRF
from testTransfer import run_Transfer_AODAI

if __name__ == "__main__":
    args = args_parser()
    for i in range(args.num_times_to_run):
        if args.algorithm.upper() == 'DQN':
            Run_DQL(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update)
        elif args.algorithm.upper() == 'DDQN':
            Run_DDQL(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update)
        elif args.algorithm.upper() == 'DuelingDQN':
            Run_DuelingDQL(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update)
        elif args.algorithm.upper() == 'D3QN':
            Run_D3QL(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update)
        elif args.algorithm.upper() == 'AODAI':
            Run_ExpectedTaskDDQN(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update, expectation_update_krate = args.expectation_update_krate)
        elif args.algorithm.upper() == 'SARSA':
            Run_Sarsa(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update)
        elif args.algorithm.upper() == 'SL':
            runShortestLatencyGreedy(f'{args.folder_name}/{i}')
        elif args.algorithm.upper()== 'ST':
            runShortestExtraTimeGreedy(f'{args.folder_name}/{i}')
        elif args.algorithm.upper() == 'AC':
            runAC(f'{args.folder_name}/{i}', gamma = args.gamma, dur = args.dur)
        elif args.algorithm.upper() == 'RF':
            runRF(f'{args.folder_name}/{i}', gamma = args.gamma, dur = args.dur)
        elif args.algorithm.upper() == 'TRANSFER_AODAI':
            run_Transfer_AODAI(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update, type = args.typeTransfer.upper(), pre_numVSs = args.pre_numVSs)
        # elif args.algorithm == 'DDQN':
        #     Run_DQL(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update)
        # elif args.algorithm == 'DDQN':
        #     Run_DQL(f'{args.folder_name}/{i}', gamma = args.gamma, target_model_update = args.target_model_update)
        else:
            print(f"There is no algorithm named \"{args.algorithm}\"!")