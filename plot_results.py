import matplotlib.pyplot as plt

# results from tangle-learning on full dataset
# acc_tangle = [0.01207, 0.09284, 0.51838, 0.59258, 0.59595, 0.67366, 0.68317, 0.72917, 0.67452, 0.71101, 0.75984]
# acc_tangle_opt = [0.01207, 0.33778, 0.66924, 0.70139, 0.73551, 0.77578, 0.78985, 0.78960, 0.79859, 0.80375, 0.78872]

# results from tangle-learning on 5% of data
acc_tan = [0.016055766494748058, 0.046234556579690236, 0.4411611493574669, 0.5620469109281471, 0.578442872620357, 0.6301531139081142, 0.6952967752729544, 0.7194763422521793, 0.6676825291212701, 0.7633632790915185, 0.7748994312144739]

# results from avalanche-learning on 5% of data
acc_ava = [0.016055766494748058, 0.08031854822998714, 0.317475716767223, 0.41859534873645915, 0.43983036453299285, 0.5791919785353212, 0.4991466303409869, 0.6370845557322359, 0.6733411150189211, 0.6545152853962993, 0.6620970029577332]


rounds = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

plt.title('10 nodes per round, 5 % of data')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.grid()
plt.plot(rounds, acc_tan, '-o', color='C0', label='tangle-learning')
plt.plot(rounds, acc_ava, '-o', color='C1', label='avalanche-learning\nalpha=0.5')
plt.legend()

plt.savefig('./results.png')
