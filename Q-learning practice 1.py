import numpy as np

gamma = 0.75
alpha = 0.9

location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     }
# Defining the actions
actions = [0,1,2,3]

# Defining the rewards
R = np.array([[0,1,0,0],
              [1,0,0,1],
              [0,0,0,1],
              [0,1,0,1],
              ])
state_to_location = {state:location for location, state in location_to_state.items()}

def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state]=1000
    Q = np.array(np.zeros((4,4)))
    
    for i in range(1000):
        current_state = np.random.randint(0,4)
        playable_actions = []
        for j in range(4):
            if R_new[current_state, j]>0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state,next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])]-Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD
        
    route = [starting_location]
    next_location = starting_location
    while (next_location!=ending_location):
        starting_location = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_location,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

print('Route:')
route('A','D')
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
