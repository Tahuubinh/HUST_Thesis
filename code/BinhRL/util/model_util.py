from schedule.value_based_rl.NAFA import NAFA_Agent
from schedule.value_based_rl.DoubleDQNPer import NAFAPER_Agent
from schedule.best_fit import bestFit
from schedule.linUCB import LinUCBAgent
from schedule.worst_fit import worst_fit
from schedule.best_reward import bestReward


def trainModel(args, env, num_series, max_episode, eplong):
    if args.method == "NAFA":
        agent = NAFA_Agent(args, env, num_series=num_series,
                           max_episode=max_episode, ep_long=eplong)
        # agent.loadWeights("data/model_{}_{}.pkl".format(args.lambda_r,args.tradeoff))
        agent.train()
        # print("data/model_{}_{}_{}.pkl".format(args.lambda_r,args.tradeoff,str(args.trial)))
        agent.loadWeights(
            f"{args.link_project}/result/model_{args.lambda_r}_{args.tradeoff}_{args.trial}.pkl")
    elif args.method == "NAFAPER":
        agent = NAFAPER_Agent(args, env, num_series=num_series,
                              max_episode=max_episode, ep_long=eplong)
        # agent.loadWeights("data/model_{}_{}.pkl".format(args.lambda_r,args.tradeoff))
        agent.train()
        # print("data/model_{}_{}_{}.pkl".format(args.lambda_r,args.tradeoff,str(args.trial)))
        agent.loadWeights(
            f"{args.link_project}/result/model_{args.lambda_r}_{args.tradeoff}_{args.trial}.pkl")
    elif args.method == "BF":  # best-fit
        agent = bestFit(env, max_episode=max_episode, ep_long=eplong)
    elif args.method == "WF":  # worst-fit
        agent = worst_fit(env, max_episode=max_episode, ep_long=eplong)
    elif args.method == "BR":  # worst-fit
        agent = bestReward(env, max_episode=max_episode, ep_long=eplong)
    elif args.method == "linUCB":
        agent = LinUCBAgent(args, env)
        agent.train()
        agent.load_model()
    return agent
