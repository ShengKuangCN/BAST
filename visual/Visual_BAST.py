import torch
from network.BAST import BAST
import torch.nn as nn
from utils import *
import os
import matplotlib.pyplot as plt
from recorder import Recorder
# from vit_grad_rollout import VITAttentionGradRollout
import cv2
import matplotlib.patches as patches


def attention_rollout_image_visualization(att_row, image, fig_path=None, fig_name='', interpolation=None):
    aspect_ratio = SPECTROGRAM_SIZE[1]/SPECTROGRAM_SIZE[0]

    mask = att_row[1:]  # remove the CLS token
    mask_expanded = (torch.Tensor(np.ones((PATCH_SIZE*PATCH_SIZE, 1))) * mask.reshape((1, -1))).unsqueeze(0)
    fold = nn.Fold(output_size=(130, 64), kernel_size=PATCH_SIZE, stride=PATCH_STRIDE)  # 130 64 136 66
    mask_fold = fold(mask_expanded)[0, 0, :SPECTROGRAM_SIZE[0], :SPECTROGRAM_SIZE[1]].numpy()
    norm_ones = torch.ones(mask_expanded.shape)
    norm_term = fold(norm_ones)[0, 0, :SPECTROGRAM_SIZE[0], :SPECTROGRAM_SIZE[1]].numpy()
    mask_fold = mask_fold / norm_term
    mask_fold_norm = mask_fold / np.max(mask_fold)

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(image, aspect=aspect_ratio)
    ax[0].axis('off')
    ax[0].set_title('Spectrogram')

    ax[1].imshow(mask_fold_norm, alpha=1, cmap='rainbow', aspect=aspect_ratio, interpolation=interpolation)
    ax[1].axis('off')
    ax[1].set_title('Attention Rollout')
    plt.show()
    if fig_path:
        plt.savefig(fig_path + '{}_attention_rollout.pdf'.format(fig_name))
    # plt.close(fig)
    return mask_fold_norm


def raw_attention_image_visualization(att_map, grid_index, grids_size, alpha, image, side='', layer='', fig_path=None):
    # grid_index = 15
    # grid_size = (6, 13)
    # att_map = attns_tf1_l1
    # alpha = 0.6
    aspect_ratio = SPECTROGRAM_SIZE[1]/SPECTROGRAM_SIZE[0]

    mask = att_map[grid_index]
    mask_expanded = torch.Tensor(np.ones((PATCH_SIZE*PATCH_SIZE, 1)) * mask.reshape((1, -1))).unsqueeze(0)
    fold = nn.Fold(output_size=(136, 66), kernel_size=PATCH_SIZE, stride=PATCH_STRIDE)

    mask_fold = fold(mask_expanded)[0, 0, :SPECTROGRAM_SIZE[0], :SPECTROGRAM_SIZE[1]].numpy()
    norm_ones = torch.ones(mask_expanded.shape)
    norm_term = fold(norm_ones)[0, 0, :SPECTROGRAM_SIZE[0], :SPECTROGRAM_SIZE[1]].numpy()
    mask_fold = mask_fold / norm_term
    mask_fold_norm = mask_fold / np.max(mask_fold)

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(image, aspect=aspect_ratio)
    rect1 = patches.Rectangle((grid_index % NUM_PATCHES_WIDTH * PATCH_STRIDE - 0.5, grid_index // NUM_PATCHES_WIDTH * PATCH_STRIDE - 0.5), PATCH_SIZE, PATCH_SIZE, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect1)
    ax[0].axis('off')
    ax[0].set_title('Spectrogram {}-{} (patch:{})'.format(side, layer, grid_index))

    # ax[1].imshow(image, aspect=SPECTROGRAM_SIZE[1]/SPECTROGRAM_SIZE[0])
    ax[1].imshow(mask_fold_norm, alpha=alpha, cmap='rainbow', aspect=aspect_ratio)
    rect2 = patches.Rectangle((grid_index % NUM_PATCHES_WIDTH * PATCH_STRIDE - 0.5, grid_index // NUM_PATCHES_WIDTH * PATCH_STRIDE - 0.5), PATCH_SIZE, PATCH_SIZE, linewidth=1, edgecolor='r', facecolor='none')
    ax[1].add_patch(rect2)
    ax[1].axis('off')
    ax[1].set_title('Attention Map {}-{} (patch:{})'.format(side, layer, grid_index))
    plt.show()
    if fig_path:
        plt.savefig(fig_path)
    plt.close(fig)


def plot_raw_attention_map(attns, attn_layer, fig_path=None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.tight_layout()
    ax.imshow(attns)
    plt.show()
    ax.set_title('Attention Map (Layer:{})'.format(attn_layer))
    if fig_path:
        plt.savefig(fig_path)
    plt.close(fig)


def plot_attention_image_visualization_by_layer_and_index(attns, attn_layer, grid_index, fig_path=None):
    attns = attns[0, attn_layer, 1:, 1:].numpy()
    if fig_path:
        fig_path1 = fig_path + model_name + '_ly{}_g{}_l'.format(attn_layer, grid_index) + '.pdf'
        fig_path2 = fig_path + model_name + '_ly{}_g{}_r'.format(attn_layer, grid_index) + '.pdf'
        fig_path3 = fig_path + model_name + '_ly{}_attn_map'.format(attn_layer) + '.pdf'
    if attn_layer in [0, 1, 2]:
        image = t_data[0, 0, :, :].numpy()
        raw_attention_image_visualization(attns, grid_index=grid_index, grids_size=(NUM_PATCHES_HEIGHT, NUM_PATCHES_WIDTH), alpha=0.6, image=image, side='Left', layer=attn_layer, fig_path=fig_path1)
    elif attn_layer in [3, 4, 5]:
        image = t_data[0, 1, :, :].numpy()
        raw_attention_image_visualization(attns, grid_index=grid_index, grids_size=(NUM_PATCHES_HEIGHT, NUM_PATCHES_WIDTH), alpha=0.6, image=image, side='Right', layer=attn_layer, fig_path=fig_path2)
    elif attn_layer in [6, 7, 8]:
        image1 = t_data[0, 0, :, :].numpy()
        raw_attention_image_visualization(attns, grid_index=grid_index, grids_size=(NUM_PATCHES_HEIGHT, NUM_PATCHES_WIDTH), alpha=0.6, image=image1, side='Left', layer=attn_layer, fig_path=fig_path1)
        image2 = t_data[0, 1, :, :].numpy()
        raw_attention_image_visualization(attns, grid_index=grid_index, grids_size=(NUM_PATCHES_HEIGHT, NUM_PATCHES_WIDTH), alpha=0.6, image=image2, side='Right', layer=attn_layer, fig_path=fig_path2)
    plot_raw_attention_map(attns, attn_layer, fig_path=fig_path3)


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise Exception("Attention head fusion type Not supported")

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def attn_rollout(attns, start_layer, end_layer, init_attn=None, fusion='max'):
    if init_attn is not None:
        trans_rollout = (init_attn + torch.eye(attns.size(-1))) / 2
        trans_rollout = trans_rollout / trans_rollout.sum(dim=-1)
    else:
        trans_rollout = torch.eye(attns.size(-1))
    I = torch.eye(attns.size(-1))
    for i in range(start_layer, end_layer):
        if fusion == 'max':
            i_fusion_attns = attns[0, i, :, :, :].max(axis=0)[0]
        elif fusion == 'min':
            i_fusion_attns = attns[0, i, :, :, :].min(axis=0)[0]
        elif fusion == 'mean':
            i_fusion_attns = attns[0, i, :, :, :].mean(axis=0)[0]
        a = (i_fusion_attns + I) / 2
        a = a / a.sum(dim=-1)
        trans_rollout = torch.matmul(a, trans_rollout)
    return trans_rollout


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


model_name = 'AudLocViT_SUB_MIX_XY_NSP'  # AudLocViT_SUB_MIX_XY_NSP BAST_SUB_MIX_XY_NSP /  SUB ADD CONCAT
DATA_ENV = 'RI'
model_path = 'F:\\Sheng\\00_TmpTest\\network\\BAST\\{}_best.pkl'.format(model_name)  # BAST
fig_path = 'F:\\Sheng\\00_TmpTest\\visual\\BAST\\'
model_dict = torch.load(model_path)
conf = model_dict['conf']
conf['BATCH_SIZE'] = 10
BINAURAL_INTEGRATION = model_name.split('_')[1]

# Load testing data based on designated conf (not from configs.py)
te_x, te_y = load_dataset(DATA_ENV, train=False, raw_path=DATA_DIR, converted_path=CONVERTED_DATA_DIR)

# Reload network parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
GPU_LIST = []
net = BAST(
    image_size=conf['SPECTROGRAM_SIZE'],
    patch_size=conf['PATCH_SIZE'],
    patch_overlap=conf['PATCH_SIZE'] - conf['PATCH_OVERLAP'],
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
    polar_output=conf['POLAR_OUTPUT'],
    share_params=conf['SHARE_PARAMS']
)

SPECTROGRAM_SIZE = conf['SPECTROGRAM_SIZE']
PATCH_SIZE = conf['PATCH_SIZE']
PATCH_STRIDE = conf['PATCH_OVERLAP']
NUM_PATCHES_HEIGHT = int(np.ceil((SPECTROGRAM_SIZE[0] - PATCH_SIZE) / PATCH_STRIDE)) + 1
NUM_PATCHES_WIDTH = int(np.ceil((SPECTROGRAM_SIZE[1] - PATCH_SIZE) / PATCH_STRIDE)) + 1

if GPU_LIST:
    net = nn.DataParallel(net, device_ids=GPU_LIST).cuda()
    net.module.load_state_dict(model_dict['state_dict'])
else:
    net.load_state_dict(model_dict['state_dict'])
net.eval()
net = Recorder(net, share_params=conf['SHARE_PARAMS'])

sound_index = 20 # 8888
t_data = torch.Tensor(te_x[0+sound_index:1+sound_index, :, :, :])


preds, attns = net(t_data)
# fusion_attns = torch.mean(attns, dim=2)

#  attention roll-out
fusion = 'max'  # max min mean
trans1_rollout_01 = attn_rollout(attns, 0, 1, fusion=fusion)
trans1_rollout_02 = attn_rollout(attns, 0, 2, fusion=fusion)
trans1_rollout_03 = attn_rollout(attns, 0, 3, fusion=fusion)
trans2_rollout_01 = attn_rollout(attns, 3, 4, fusion=fusion)
trans2_rollout_02 = attn_rollout(attns, 3, 5, fusion=fusion)
trans2_rollout_03 = attn_rollout(attns, 3, 6, fusion=fusion)
# trans3_init = None
if BINAURAL_INTEGRATION == 'SUB':
    trans3_init = trans1_rollout_03 + trans2_rollout_03
elif BINAURAL_INTEGRATION == 'ADD':
    trans3_init = trans1_rollout_03 + trans2_rollout_03
elif BINAURAL_INTEGRATION == 'CONCAT':
    trans3_init = trans1_rollout_03 + trans2_rollout_03
# trans3_init = None
trans3_rollout_01 = attn_rollout(attns, 6, 7, trans3_init, fusion=fusion)
trans3_rollout_02 = attn_rollout(attns, 6, 8, trans3_init, fusion=fusion)
trans3_rollout_03 = attn_rollout(attns, 6, 9, trans3_init, fusion=fusion)

trans_attn_map = [trans1_rollout_01, trans1_rollout_02, trans1_rollout_03, trans2_rollout_01, trans2_rollout_02, trans2_rollout_03, trans3_rollout_01, trans3_rollout_02, trans3_rollout_03]

plt.imshow(trans3_rollout_03)

attn_visual_map_t1 = attention_rollout_image_visualization(trans1_rollout_03.mean(axis=0), t_data[0, 0, :, :].numpy(), fig_path=None, fig_name=model_name + 'BAST', interpolation=None)
attn_visual_map_t2 = attention_rollout_image_visualization(trans2_rollout_03.mean(axis=0), t_data[0, 0, :, :].numpy(), fig_path=None, fig_name=model_name + 'BAST', interpolation=None)
attn_visual_map_t3 = attention_rollout_image_visualization(trans3_rollout_03.mean(axis=0), t_data[0, 0, :, :].numpy(), fig_path=None, fig_name=model_name + 'BAST', interpolation=None)

# plot_attention_image_visualization_by_layer_and_index(fusion_attns, attn_layer=0, grid_index=2, fig_path=fig_path)


fig = plt.figure(figsize=(15, 5))
ax1 = plt.subplot(2, 6, 1)
ax2 = plt.subplot(2, 6, 2)
ax3 = plt.subplot(2, 6, 3)
ax4 = plt.subplot(2, 6, 7)
ax5 = plt.subplot(2, 6, 8)
ax6 = plt.subplot(2, 6, 9)
ax7 = plt.subplot(1, 6, 4)
ax8 = plt.subplot(1, 6, 5)
ax9 = plt.subplot(1, 6, 6)

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
titles = [
    '1st layer TE-L', '2nd layer TE-L', '3rd layer TE-L',
    '1st layer TE-R', '2nd layer TE-R', '3rd layer TE-R',
    '1st layer TE-C', '2nd layer TE-C', '3rd layer TE-C']
for idx, ax in enumerate(axes):
    ax.imshow(trans_attn_map[idx])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    ax.set_title(titles[idx])
ax1.set_xlabel('Patches')
ax1.set_ylabel('Patches')

plt.savefig(fig_path + '\\{}_attn_flow_{}.pdf'.format(model_name, DATA_ENV))


# ax_input_l = plt.subplot(2, 8, 1)
# ax_input_l.imshow(t_data[0, 0, :, :].numpy(), aspect=SPECTROGRAM_SIZE[1]/SPECTROGRAM_SIZE[0])
# ax_input_l.axis('off')
# ax_input_l.set_title('Left')
#
# ax_input_r = plt.subplot(2, 8, 9)
# ax_input_r.imshow(t_data[0, 1, :, :].numpy(), aspect=SPECTROGRAM_SIZE[1]/SPECTROGRAM_SIZE[0])
# ax_input_r.axis('off')
# ax_input_r.set_title('Right')
#
# ax_output = plt.subplot(1, 8, 8)
# ax_output.imshow(attn_visual_map * t_data[0, 0, :, :].numpy(), aspect=SPECTROGRAM_SIZE[1]/SPECTROGRAM_SIZE[0])
# ax_output.axis('off')
# ax_output.set_title('Attention Map')

img_left = t_data[0, 0, :, :].numpy()
img_right = t_data[0, 1, :, :].numpy()
aspect = SPECTROGRAM_SIZE[1]/SPECTROGRAM_SIZE[0]
vmax = np.max([np.max(img_left), np.max(img_right)])
vmin = np.min([np.min(img_left), np.min(img_right)])

fig2 = plt.figure(figsize=(7.5, 5))

ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
# ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(1, 3, 3)


ax1.imshow(img_left, aspect=aspect, vmax=vmax, vmin=vmin)
ax4.imshow(img_right, aspect=aspect, vmax=vmax, vmin=vmin)
ax2.imshow(img_left, aspect=aspect, vmax=vmax, vmin=vmin)
ax2.imshow(attn_visual_map_t1, aspect=aspect, alpha=0.5, cmap='jet')
ax5.imshow(img_right, aspect=aspect, vmax=vmax, vmin=vmin)
ax5.imshow(attn_visual_map_t2, aspect=aspect, alpha=0.5, cmap='jet')
# ax3.imshow(img_left, aspect=aspect, vmax=vmax, vmin=vmin)
# ax3.imshow(attn_visual_map_t3, aspect=aspect, alpha=0.5, cmap='jet')
ax6.imshow(img_left-img_right, aspect=aspect, vmax=np.max(img_left-img_right), vmin=np.min(img_left-img_right))
ax6.imshow(attn_visual_map_t3, aspect=aspect, alpha=0.5, cmap='jet')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')
ax6.axis('off')
fig2.show()
plt.savefig(fig_path + '\\{}_attn_rollout_{}.png'.format(model_name, DATA_ENV))
