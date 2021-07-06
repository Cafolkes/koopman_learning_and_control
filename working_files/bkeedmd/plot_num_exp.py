import dill
import os
import numpy as np
import matplotlib.pyplot as plt

sys_name = 'planar_quad'
directory = os.path.abspath("working_files/bkeedmd/")

# Load results:
infile = open(directory + '/data/' + sys_name + 'num_exp.pickle', 'rb')
result_lst = dill.load(infile)
infile.close()

n_lift = np.sort(np.unique([r['n_lift'] for r in result_lst]))
n_data = np.sort(np.unique([r['n_data'] for r in result_lst]))

train_loss = np.zeros((n_lift.size, n_data.size))
train_loss_std = np.zeros((n_lift.size, n_data.size))
test_loss = np.zeros((n_lift.size, n_data.size))
test_loss_std = np.zeros((n_lift.size, n_data.size))
mse = np.zeros((n_lift.size, n_data.size))
mse_std = np.zeros((n_lift.size, n_data.size))

for ii, n_l in enumerate(n_lift):
    for jj, n_d in enumerate(n_data):
        train_loss_tmp = np.array([r['train_loss_kdnn'] for r in result_lst if (r['n_lift'] == n_l and r['n_data'] == n_d)])
        train_loss[ii, jj] = np.mean(train_loss_tmp)
        train_loss_std[ii, jj] = np.std(train_loss_tmp)

        test_loss_tmp = np.array([r['test_loss_kdnn'] for r in result_lst if (r['n_lift'] == n_l and r['n_data'] == n_d)])
        test_loss[ii, jj] = np.mean(test_loss_tmp)
        test_loss_std[ii, jj] = np.std(test_loss_tmp)

        mse_tmp = np.array([r['mse_kdnn'] for r in result_lst if (r['n_lift'] == n_l and r['n_data'] == n_d)])
        mse[ii, jj] = np.mean(mse_tmp)
        mse_std[ii, jj] = np.std(mse_tmp)

data = [train_loss, test_loss, mse]
data_std = [train_loss_std, test_loss_std, mse_std]

n_data = 2./1e-2*n_data
rows = len(data)
ylab_lst = ['Train loss', 'Test loss', 'Open loop error']
label_lst = ['$n_{lift}$=' + str(n) for n in n_lift]

plt.figure()
for ii, (d, d_std) in enumerate(zip(data, data_std)):
    plt.subplot(rows,1, ii+1)
    plt.ylabel(ylab_lst[ii])
    for jj, (d_j, d_j_std) in enumerate(zip(d, d_std)):
        plt.plot(n_data, d_j, label=label_lst[jj])
        plt.fill_between(n_data, d_j - d_j_std, d_j + d_j_std, alpha=0.1)

plt.subplot(rows,1, 1)
plt.title('Train, test, and open loop prediction MSE loss')

plt.subplot(rows,1, 3)
plt.xlabel('Number of data points')
plt.legend()

plt.show()




