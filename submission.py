import numpy as np
from collections import defaultdict
import random

!pip3 install git+https://github.com/slremy/netsapi --user --upgrade

from netsapi.challenge import *
class CustomAgent():
    
    def __init__(self, environment):
        
        #Hyperparameters
        self.env = environment
        self.epsilon = 0.9
        self.gamma = 0.9
        self.action_resolution = 0.1
        self.action_resolution_year1 = 0.3
        self.Q = defaultdict(lambda : 0.) # Q-function
        self.n = defaultdict(lambda : 1.) # number of visits
        self.actions = self.actionSpace(self.action_resolution)
        self.actionspace = range(len(self.actions)-1)
        self.policymax = []
        self.actionyear1 = self.actionSpace(self.action_resolution_year1)
        self.actionspaceyear1 = range(len(self.actionyear1)-1)
        self.memory = []
        
    
    def actionSpace(self,resolution):
        x,y = np.meshgrid(np.arange(0,1.1,resolution), np.arange(0,1.1,resolution))
        xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
        return xy.round(2).tolist()
    def exploitSpace(self,action,resolution):
        cactionspace = []
        final = []
        for i in [resolution,0,-resolution]:
            for j in [resolution,0,-resolution]:
                cactionspace.append([action[0]+j,action[1]+i])
        for a in cactionspace:
            if(a not in self.memory and a[0]<=1 and a[0]>=0 and a[1]<=1 and a[1]>=0):
                final.append(a)
#         print("final: ", final)
        return final
    def train(self):
        rewardmax = -999999
        policymax = []
        currentReward = 0
        Q = self.Q
        n = self.n
        gamma = self.gamma
        actions = self.actions
        actionspace = self.actionspace
        actionyear1 = self.actionyear1
        actionspaceyear1 = self.actionyear1
        currentPolicy = []
        maxactionyear1 = []
        greedy_action = lambda s : max(actionspace, key=lambda a : Q[(s,a)])
        max_q = lambda sp : max([Q[(sp,a)] for a in actionspace])
        rewardmaxyear1 = -9999
        count = 20 # 20 evaluations = 4 policies
        #find action for the first year with 20 evaluations
        for a in actionyear1:
            
            tempa = a
            count-=1
            self.env.reset()
            _,reward,_,_ = self.env.evaluateAction(tempa);
#             print("57: ", reward, " ", tempa)
            self.memory.append(tempa)
            if(reward > rewardmaxyear1):
                rewardmaxyear1 = reward
                maxactionyear1 = tempa
                
        spaceExploit = self.exploitSpace(maxactionyear1,self.action_resolution)
        while(count>0):
            self.env.reset()
            nextaction = []
            direct = 0
            if(direct == 1):
                actionchoice = nextaction
            else:
                actionchoice = random.choice(spaceExploit)
            if(actionchoice not in self.memory):
                self.env.reset()
                _,reward,_,_ = self.env.evaluateAction(actionchoice)
#                 print("74: ",reward, " ", actionchoice)
                count-=1
                self.memory.append(actionchoice)
                direction = [actionchoice[0] - maxactionyear1[0],actionchoice[1] - maxactionyear1[1]]
                if(reward > rewardmaxyear1):
                    rewardmaxyear1 = reward
                    maxactionyear1 = actionchoice
                    nextaction = [actionchoice[0] + direction[0],actionchoice[1] + direction[1]]
                    direct =1
                    if(nextaction[0] >1 or nextaction[0] <0 or nextaction[1] >1 or nextaction[1] <0):
                        nextaction = [actionchoice[0] - direction[0],actionchoice[1] - direction[1]]
                        spaceExploit = self.exploitSpace(nextaction,self.action_resolution)
                        direct = 0
#                 if(spaceExploit.index[actionchoice])
#                 spaceExploit.remove(actionchoice)
        for e in range(16): #16 policies left
            epsilon = 0.8-(e/(16*1.2))
            self.env.reset()
            nextstate = self.env.state
            currentReward = 0
            currentPolicy=[]
#             print(maxactionyear1," ", rewardmaxyear1)
            while True:
                state = nextstate

                # Epsilon-Greedy Action Selection
                if epsilon > random.random() :
                    action = random.choice(actionspace)
                else :
                    action = greedy_action(state)
                n[(state,action)]+=1
                env_action = actions[action]#convert to ITN/IRS
                #print('env_action', env_action)
                if(state == 1 ):
                    env_action = maxactionyear1
                nextstate, reward, done, _ = self.env.evaluateAction(env_action)
                currentReward += reward
                currentPolicy.append(env_action)
                # Q-learning
                if done :
                    Q[(state,action)] = Q[(state,action)] + 1./n[(state,action)] * ( reward - Q[(state,action)] )
                    if(currentReward > rewardmax):
#                         print(rewardmax)
                        rewardmax = currentReward
                        self.policymax = currentPolicy[:]
#                         print(self.policymax)
                    break
                else :
                    Q[(state,action)] = Q[(state,action)] + 1./n[(state,action)] * ( reward + gamma * max_q(nextstate) - Q[(state,action)] )

        return Q


    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        Q_trained = self.train()
#         greedy_eval = lambda s : max(self.actionspace, key=lambda a : Q_trained[(s,a)])
#         print(self.policymax)
        best_policy = {state : (self.policymax[state-1]) for state in range(1,6)}
        best_reward = self.env.evaluatePolicy(best_policy)
        
        print(best_policy, best_reward)
        
        return best_policy, best_reward