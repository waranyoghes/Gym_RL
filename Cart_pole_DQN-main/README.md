# Cart-Pole balancing using DQN
#### Implementation of DQN, a deep reinforcement learning algorithm which uses Q learing and DNN as function approximation, for Cart-pole balancing problem from openai gym.
#### The algorithm was able to achive a score 500 in the 63rd episode. 
![image](https://user-images.githubusercontent.com/73269696/160671044-d709dcb1-c0e6-45fc-8c0c-2435cb1d71e4.png)
![image](https://user-images.githubusercontent.com/73269696/160673180-35d23685-42dc-427a-a9fe-8b33de236329.png)

#### Reference Material and Paper
* Sutton and Barto(chapter 6,8 and 10): http://incompleteideas.net/book/the-book.html
* Temporal differnce methods by Sutton: http://incompleteideas.net/papers/sutton-88-with-erratum.pdf
* DQN paper: https://arxiv.org/abs/1312.5602  
Double Q learning: https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf
It has been shown by Dr. Hado van Hasselt that Double Q learning performs better than Q-learning because of the bias in Q-learning algorithm as Q-learning algo gives a biased estimation( over estimation) of the maximization of Expected Q(s,a) return over Actions.
* Deep Double Q-network: https://arxiv.org/abs/1509.06461

