import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        noise_i = np.random.randn(1,env.action_space.n)*(1./(i+1))
        a = np.argmax(Q[s,:] + noise_i)
        next_s, reward, terminated, _ = env.step(a)
        Q[s,a] = (1-lr)*Q[s,a] + lr*(reward + y*np.max(Q[next_s,:]))
        rAll += reward
        s = next_s
        if terminated == True:
            break
    
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
