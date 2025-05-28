import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.append(os.getcwd())
from network.BAST import BAST, BAST_Variant, AngularLossWithCartesianCoordinate
from utils import *
from conf import *

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--backbone", help="mamba/vanilla/swin")
parser.add_argument("-i", "--integ", help="SUB/ADD/CONCAT")
parser.add_argument("-l", "--loss", help="MSE/AD/MIN")
parser.add_argument("-s", "--shareweights", help="Share weights", type=bool, default=False)
parser.add_argument("-e", "--env", help="RI/RI01/RI02")
args = parser.parse_args()



BINAURAL_INTEGRATION = args.integ
LOSS = args.loss  # MSE AD MIX
SHARE_PARAMS = args.shareweights  # True False
MODEL_DATA_ENV = args.env
MODEL_NAME = 'BAST'  # BAST AudLocViT
MODEL_TYPE = args.backbone
model_name = '{}_{}_{}_XY_{}_{}'.format(MODEL_NAME, BINAURAL_INTEGRATION, LOSS, 'SP' if SHARE_PARAMS else 'NSP', MODEL_TYPE)  # 'BAST_SUB_MIX_XY_NSP'  #args.modelname
TEST_DATA_ENV = args.env
# model_name = 'BAST_SUB_MIX_XY_NSP'  # SUB ADD CONCAT   MSE MIX AD   NSP  AudLocViT
# DATA_ENV = 'RI'  # RI RI01 RI02 RI_SNR_40

if MODEL_DATA_ENV != 'RI':
    model_name += '_' + MODEL_DATA_ENV
model_path = MODEL_SAVE + '{}_best.pkl'.format(model_name)
print('load model', model_path)
fig1_path = FIG_PATH + '{}_AD.pdf'.format(model_name)
fig2_path = FIG_PATH + '{}_DISTRIBUTION.pdf'.format(model_name)

eval_path = EVAL_PATH + '{}.npy'.format(model_name)
model_dict = torch.load(model_path, weights_only=False)
conf = model_dict['conf']
conf['BATCH_SIZE'] = 20 if 'CONCAT' not in model_name else 40

# Load validation and testing data based on designated conf (not from configs.py)
te_x, te_y = load_dataset(TEST_DATA_ENV, train=False, raw_path=DATA_DIR, converted_path=CONVERTED_DATA_DIR)
te_x = norm_image(te_x)
te_x = torch.Tensor(te_x)
te_y = torch.Tensor(te_y)

net = BAST_Variant(
    image_size=conf['SPECTROGRAM_SIZE'],
    patch_size=conf['PATCH_SIZE'],
    patch_overlap=conf['PATCH_OVERLAP'],  # conf['PATCH_SIZE'] - conf['PATCH_OVERLAP']
    num_classes=conf['NUM_OUTPUT'],
    dim=conf['EMBEDDING_DIM'],
    depth=conf['TRANSFORMER_DEPTH'],
    heads=conf['TRANSFORMER_HEADS'],
    mlp_dim=conf['TRANSFORMER_MLP_DIM'],
    pool=conf['TRANSFORMER_POOL'],
    channels=conf['INPUT_CHANNEL'],
    dim_head=conf['TRANSFORMER_DIM_HEAD'],
    dropout=conf['DROPOUT'],
    emb_dropout=conf['EMB_DROPOUT'],
    binaural_integration=conf['BINAURAL_INTEGRATION'],
    share_params=conf['SHARE_PARAMS'],
    transformer_variant=MODEL_TYPE,
)


if GPU_LIST:
    net = nn.DataParallel(net, device_ids=GPU_LIST).cuda()
    net.module.load_state_dict(model_dict['state_dict'])
else:
    net.load_state_dict(model_dict['state_dict'])
net.eval()

num_te = te_y.shape[0]
num_batches_val = int(np.ceil(num_te / conf['BATCH_SIZE']))
criterion = nn.MSELoss()
criterion2 = AngularLossWithCartesianCoordinate()

te_pred = np.zeros(te_y.shape)

batch_loss_te = 0
batch_loss2_te = 0
torch.no_grad()
mse_loss_l = []
for batch in range(num_batches_val):
    net.eval()
    idx_s = conf['BATCH_SIZE'] * batch
    idx_e = conf['BATCH_SIZE'] * (batch + 1)
    idx_e = np.min([idx_e, num_te])
    input_x = te_x[idx_s:idx_e, :, :, :].cuda()
    target = te_y[idx_s:idx_e, :].cuda()
    output = net(input_x)

    loss = criterion(output, target)
    te_pred[idx_s:idx_e] = output.detach().cpu().numpy()
    loss_v = loss.item()
    batch_loss_te += loss_v * (idx_e - idx_s)

    loss2 = criterion2(output, target)
    loss2_v = loss2.item()
    batch_loss2_te += loss2_v * (idx_e - idx_s)

    print('\r[{}] [TESTING] Batch: {}, Curr Loss: {:.6f}, Avg Loss: {:.6f}'.format(datetime.now(), batch, loss_v,
                                                                                   batch_loss_te / idx_e))
avg_batch_loss_te = batch_loss_te / num_te
avg_batch_loss2_te = batch_loss2_te / num_te

mse = np.sum(np.power(te_pred - te_y.detach().cpu().numpy(), 2), axis=1) / 2

te_pred_norm = te_pred / np.linalg.norm(te_pred, axis=1)[:, None]
te_y_norm = (te_y / np.linalg.norm(te_y, axis=1)[:, None]).detach().cpu().numpy()
dot_product = np.clip(np.sum(te_pred_norm * te_y_norm, axis=1), -1, 1)
angle_diff = np.arccos(dot_product)
deg_diff = np.rad2deg(angle_diff)

print(model_name)
print('MSE: ', avg_batch_loss_te, np.mean(mse))
print('AD: ', np.mean(deg_diff))


# plot mean angular error

deg_y = np.round(np.rad2deg(np.arctan2(te_y_norm[:, 1], te_y_norm[:, 0])), 0).astype(np.int16)
df_detail = pd.DataFrame({'angle': deg_y, 'value': deg_diff})
df_save = pd.DataFrame({'angle': deg_y, 'ad': deg_diff, 'mse': mse})
np.save(eval_path, df_save, allow_pickle=True)
df = df_detail.groupby(['angle']).agg('mean')
df = df.reset_index()

t_theta = df['angle'].to_numpy() / 180 * np.pi
t_r = df['value'].to_numpy()
theta1 = np.append(t_theta, t_theta[0])
r1 = np.append(t_r, t_r[0])

fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
ax1.plot(theta1, r1)
ax1.set_rmax(10)
ax1.set_rgrids([2, 4, 6, 8, 10], labels=['2°', '4°', '6°', '8°', '10°'])
# ax1.set_rmax(15)
# ax1.set_rgrids([5, 10, 15], labels=['5°', '10°', '15°'])
ax1.set_rlabel_position(22.5)  # Move radial labels away from plotted line
ax1.grid(True)
ax1.set_theta_zero_location('N')
ax1.set_xticks(np.pi / 180. * np.linspace(180, -180, 8, endpoint=False))
ax1.set_thetalim(-np.pi, np.pi)
ax1.set_theta_direction(-1)
if conf['BINAURAL_INTEGRATION'] == 'ADD':
    title = 'Addition'
elif conf['BINAURAL_INTEGRATION'] == 'SUB':
    title = 'Subtraction'
else:
    title = 'Concatenation'
ax1.set_title(title, va='top')
plt.show()
plt.savefig(fig1_path)

# plot prediction distribution
theta2 = np.arctan2(te_pred[:, 1], te_pred[:, 0])
r2 = np.linalg.norm(te_pred, axis=1)
theta_std = np.arange(-180, 180, 10) / 180 * np.pi
r_std = np.ones((36)) * 1.0
fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
# ax2.plot(theta2, r2, 'o')
ax2.scatter(theta2, r2, c=theta2, cmap='hsv', alpha=0.5, s=3)
ax2.scatter(theta_std, r_std, c=theta_std, cmap='hsv', marker='s', linewidths=1, s=20, edgecolor='k')

# ax2.set_rmin(1)
if conf['LOSS'] in ['MSE', 'MIX']:
    ax2.set_rmax(1.5)
    ax2.set_rgrids([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
else:
    ax2.set_rmax(50)
    ax2.set_rgrids([0, 10, 20, 30, 40, 50])
ax2.set_rlabel_position(22.5)  # Move radial labels away from plotted line
ax2.grid(True)
ax2.set_theta_zero_location('N')
ax2.set_xticks(np.pi / 180. * np.linspace(180, -180, 8, endpoint=False))
ax2.set_thetalim(-np.pi, np.pi)
ax2.set_theta_direction(-1)
ax2.set_title(title, va='top')
ax2.set_thetagrids(np.arange(-180, 180, 10))

plt.show()
plt.savefig(fig2_path)

