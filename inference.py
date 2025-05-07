
import torch.nn as nn
import torch
import matplotlib
from PIL import Image
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from dataset.dataset import DataSets_inter_p_reg_Nsclc_30_29
from scipy.io import savemat
import os
import argparse
import time
import numpy as np
import nibabel as nib

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import torch.nn.functional as F
from torch.autograd import Variable
matplotlib.use('Agg')
fig = plt.figure()
ax = fig.add_subplot(211)

plot_loss_value=[]
plot_loss_value1=[]
plot_loss_value2=[]


def cal_dice_3D(seg_in,gt_in,label):
    
    seg=np.zeros(seg_in.shape,dtype='uint8')
    gt=np.zeros(gt_in.shape,dtype='uint8')
    seg[seg_in==label]=1
    
    gt[gt_in==label]=1


    gt_flt=gt.flatten()
    seg_flt=seg.flatten()
                        
    intersection = np.sum(seg_flt * gt_flt)  
                        
                        
    
    dsc_3D=(2. * intersection + 0.0001) / (np.sum(seg_flt) + np.sum(gt_flt) + 0.0001)
    return dsc_3D

def tensor2im_jj(image_tensor):
    if 1>0:
        
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy_tep=image_numpy
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

        

        image_numpy_all=image_numpy#np.concatenate((self.test_A_tep,image_numpy,),axis=2)
        #if np.min(image_numpy_all)<0:
        image_numpy_all = (np.transpose(image_numpy_all, (1, 2, 0)) + 1) / 2.0 * 255.0    
        #else:
        #    image_numpy_all = (np.transpose(image_numpy_all, (1, 2, 0)))*255.0                   
        

        return image_numpy_all.astype(np.uint8),image_numpy_tep

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

class DiceLoss_test(nn.Module):
    def __init__(self,num_organ=2):
        super(DiceLoss_test, self).__init__()

        self.num_organ=num_organ
    def forward(self, pred_stage1, target):
        """
        :param pred_stage1: (B, 9,  256, 256)
        :param pred_stage2: (B, 9, 256, 256)
        :param target: (B, 256, 256)
        :return: Dice
        """
        pred_stage1 = F.softmax(pred_stage1, dim=1)
        num_organ=12
        # 
        #[b,12,256,256]
        organ_target = torch.zeros((target.size(0), num_organ,  256, 256,target.size(3)))
        #print (organ_target.size())
        #[0-11] 
        for organ_index in range(0, num_organ ):
            #print (organ_index)
            temp_target = torch.zeros(target.size())
            #print (temp_target.size())
            #print (organ_target.size())
            temp_target[target == organ_index] = 1
            #print (organ_target[:, organ_index, :, :].size())
            #print (temp_target.size())
            organ_target[:, organ_index, :, :,:] = torch.squeeze(temp_target)
            # organ_target: (B, 8,  128, 128)

        organ_target = organ_target.cuda()

        # loss
        dice_stage1 = 0.0

        for organ_index in range(0, num_organ ):
            dice_stage1 += 2 * (pred_stage1[:, organ_index, :, :,:] * organ_target[:, organ_index , :, :,:]).sum(dim=1).sum(
                dim=1) / (pred_stage1[:, organ_index, :, :,:].pow(2).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index, :, :,:].pow(2).sum(dim=1).sum(dim=1) + 1e-5)

        dice_stage1 /= num_organ


        # 
        dice = dice_stage1 

        # 
        return (1 - dice).mean()

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        #print (input.size())
        #print (target.size())
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

# parse the commandline
parser = argparse.ArgumentParser()
fig = plt.figure()
ax = fig.add_subplot(211)
# data organization parameters
parser.add_argument('--datadir', default='./test_data/', help='base data directory')
parser.add_argument('--savedir', default='./dir_results/', help='base data directory')
parser.add_argument('--model_path', default='./weight/Tracer_weight.pt', help='base data directory')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='2', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet',  action='store_true', help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01, help='weight of deformation loss (default: 0.01)')
parser.add_argument('--smooth', type=float, default=30, help='weight of deformation loss (default: 0.01)')
parser.add_argument('--flownum', type=int, default=7, help='flow number (default: 7)')
parser.add_argument('--flowrange', type=int, default=5, help='flow number (default: 7)')

# for output
parser.add_argument('--svdir', type=str, default='test_deformation_result_save', help='weight of deformation loss (default: 0.01)')
parser.add_argument('--fold', type=str, default=1, help='flow downsample factor for integration (default: 2)')

parser.add_argument('--accu_dvf', type=int, default=1, help='whether use accumulate dvf during training')
parser.add_argument('--organ_loss', type=int, default=0, help='whether use accumulate dvf during training')
parser.add_argument('--use_gtv_msk', type=int, default=0, help='whether use accumulate dvf during training')
parser.add_argument('--use_tumor_dsc', type=int, default=0, help='whether use accumulate dvf during training')
parser.add_argument('--use_tumor_jet', type=int, default=1, help='whether use accumulate dvf during training')

args = parser.parse_args()
smooth_w=args.smooth
bidir = args.bidir
bidir= False

range_flow=args.flowrange

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
accumulate_dvf=args.accu_dvf
accumulate_dvf=1
use_organ_loss=args.organ_loss
use_gtv_msk=args.use_gtv_msk
use_tumor_dsc=args.use_tumor_dsc
use_tumor_jet=args.use_tumor_jet
inshape=(128,128,96)  

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert args.batch_size >= nb_gpus, 'Batch size (%d) should be no less than the number of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

model = vxm.networks.VxmDense_3D_LSTM_Step_Reg_All_Enc_Lstm_target_gtv_in(   # 
    #vxm.networks.( 
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    bidir=bidir,
    int_steps=args.int_steps,
    int_downsize=args.int_downsize
)
print(model)

test_data_path=args.datadir
data_list=os.path.join(test_data_path, 'list.txt')
val_data_set=DataSets_inter_p_reg_Nsclc_30_29(test_data_path,data_list,source_aug=False)
val_loader=torch.utils.data.DataLoader(val_data_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)  
model_path=args.model_path

save_flag=True
sv_folder1=args.savedir
if not os.path.exists(sv_folder1):
    os.makedirs(sv_folder1)
print('#########sv_folder1', sv_folder1)
print ('info: weight_ ',model_path)
model=model.load(model_path, device)
model.to(device)

# set optimizer
flow_ini=torch.zeros(1, 3,128, 128, 96).cuda()

grid_template_ini=torch.zeros(1, 1,128,128,96)
grid_w=12
for i in range(0,26):
            
    grid_template_ini[:,:,:,i*5,:]=1

for i in range(0,20):
            
    grid_template_ini[:,:,:,:,i*5]=1

grid_template_ini=grid_template_ini.cuda()


reg_acc=np.zeros([5000,6])

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(x, y, z)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid

grid = generate_grid(inshape)
grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()
grid=grid.permute(0,4,2,3,1)
print(grid.size())

# prepare deformation loss
nn_up=nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
total_steps=0       
plt_iternm=[]
flow_num=args.flownum
iter_count=0
dsc_best=0
dsc_cur_reg_best=0


with torch.no_grad():
    
    val_id=0
    for i_iter_val, (cbct_all_1) in enumerate(val_loader):      

        val_id=val_id+1

        p_name='val_id_'+str(val_id)
        p_name='val'
        #print ('p_name is ',p_name)
        sv_folder=sv_folder1+'pt'+str(val_id)+'/'
        print('##########', sv_folder)
        if not os.path.exists(sv_folder):
            
            os.makedirs(sv_folder)
        gt_sv_name=sv_folder+'gt_'+p_name 
        seg_sv_name=sv_folder+'seg_'+p_name   
        img_sv_name=sv_folder+'img_'+p_name  
        if 1>0:
            plan_ct_img=cbct_all_1[0]
            planct_val_msk=cbct_all_1[1]
            planct_val_gtv=cbct_all_1[2]

            cbct_val_img=cbct_all_1[3]
            cbct_val_msk=cbct_all_1[4]
            cbct_val_gtv=cbct_all_1[5]

            cbct_val_gtv[cbct_val_gtv>1]=1
            planct_val_gtv[planct_val_gtv>1]=1

            planct_val_msk_all=planct_val_msk 
            planct_val_msk_all[planct_val_msk_all>3]=0 # you can assign organs. our example data has 1:right lung, 2:left lung, 3:heart, 4:esopahgus, 5:spinal cord. here we're removing esophagus and spinal cord masks
            gtv_all=cbct_val_gtv+planct_val_gtv 
            gtv_all[gtv_all>0]=1

            cbct_val_img=cbct_val_img.float().cuda()
            cbct_val_msk=cbct_val_msk.float().cuda()
            plan_ct_img=plan_ct_img.float().cuda()
            planct_val_msk=planct_val_msk.float().cuda()

            cbct_val_gtv=cbct_val_gtv.float().cuda()
            planct_val_gtv=planct_val_gtv.float().cuda()

            planct_val_msk_ori=planct_val_msk

            # 'Multi_channel PlanCT'
            PlanCT_val_msk_mt = torch.zeros((planct_val_msk.size(0), 6,  128, 128,96))
            
            for organ_index in range(1,6): # you can change the range here depending on your OAR labels
                temp_target = torch.zeros(planct_val_msk.size())
                temp_target[planct_val_msk == organ_index] = 1
                
                PlanCT_val_msk_mt[:,organ_index,:,:,:]=torch.squeeze(temp_target)

            planct_val_msk= PlanCT_val_msk_mt.cuda()   
            
        # feed the data in
        save_flag=True
        if 1>0:
            cbct_val_img=cbct_val_img.float().cuda()
            cbct_val_gtv=cbct_val_gtv.float().cuda()
            plan_ct_img=plan_ct_img.float().cuda()
            planct_val_msk=planct_val_msk.float().cuda()

            cbct_val_img_show=torch.squeeze(cbct_val_img)

            cbct_val_img_show=cbct_val_img_show.data.cpu().numpy()
            cbct_val_img_show = nib.Nifti1Image(cbct_val_img_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_Fixed_Image.nii.gz'
            if save_flag:
                nib.save(cbct_val_img_show, pred_sv_name)  

            cbct_val_img_show=torch.squeeze(cbct_val_gtv)
            cbct_val_img_show=cbct_val_img_show.data.cpu().numpy()
            cbct_val_img_show = nib.Nifti1Image(cbct_val_img_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_Fixed_gtv.nii.gz'
            if save_flag:
                nib.save(cbct_val_img_show, pred_sv_name)  

            cbct_val_img_show=torch.squeeze(cbct_val_msk)
            cbct_val_img_show=cbct_val_img_show.data.cpu().numpy()
            cbct_val_img_show = nib.Nifti1Image(cbct_val_img_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_Fixed_msk.nii.gz'
            if save_flag:
                nib.save(cbct_val_img_show, pred_sv_name)  

            plan_ct_img_show=torch.squeeze(plan_ct_img)
            plan_ct_img_show=plan_ct_img_show.data.cpu().numpy()
            plan_ct_img_show = nib.Nifti1Image(plan_ct_img_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_Move_Image.nii.gz'
            if save_flag:
                nib.save(plan_ct_img_show, pred_sv_name)  

            plan_ct_img_show=torch.squeeze(planct_val_msk_ori)
            plan_ct_img_show=plan_ct_img_show.data.cpu().numpy()
            plan_ct_img_show = nib.Nifti1Image(plan_ct_img_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_Move_msk.nii.gz'
            if save_flag:
                nib.save(plan_ct_img_show, pred_sv_name)  

            plan_ct_img_show=torch.squeeze(planct_val_gtv)
            plan_ct_img_show=plan_ct_img_show.data.cpu().numpy()
            plan_ct_img_show = nib.Nifti1Image(plan_ct_img_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_Move_gtv.nii.gz'
            if save_flag:
                nib.save(plan_ct_img_show, pred_sv_name)  


            for seg_iter_val in range (0,flow_num+1):

                if seg_iter_val==0:
                    h=None
                    c=None
                    y_pred_val,flow_acc,_,_,_ ,h,c,y_m_pred_val,flow_acc_seg_val,planct_val_gtv_def,grid_template= model.forward_seg_training_flow_acc_correction_range_flow_gtv_in_grid(plan_ct_img,cbct_val_img,planct_val_msk,0,h,c,flow_ini,plan_ct_img,planct_val_msk,range_flow,planct_val_gtv,cbct_val_gtv,grid_template_ini)
                else:
                    y_pred_val,flow_acc,_,_,_,h,c,y_m_pred_val,flow_acc_seg_val,planct_val_gtv_def,grid_template = model.forward_seg_training_flow_acc_correction_range_flow_gtv_in_grid(y_pred_val,cbct_val_img,planct_val_msk,0,h,c,flow_acc_seg_val,plan_ct_img,planct_val_msk,range_flow,planct_val_gtv_def,cbct_val_gtv,grid_template)


                    end_tm=time.time()

                    
                y_pred_val_show=torch.squeeze(y_pred_val)
                y_m_pred_val_show=torch.squeeze(y_m_pred_val)
                y_gtv_pred_val_show=torch.squeeze(planct_val_gtv_def)
                #DVF=torch.squeeze(flow_abs)
                #DVF_step=torch.squeeze(flow_cur)
                grid_template_show=torch.squeeze(grid_template)

            # save the images
            y_pred_val_show=y_pred_val_show.data.cpu().numpy()
            y_pred_val_show = nib.Nifti1Image(y_pred_val_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_deformed_Move_Image.nii.gz'
            print('saved to:', pred_sv_name)
            if seg_iter_val % 7==0 and save_flag:
                nib.save(y_pred_val_show, pred_sv_name)  

            y_m_pred_val_show = torch.argmax(y_m_pred_val_show, dim=0)
            y_m_pred_val_show = y_m_pred_val_show.data.cpu().numpy().astype(np.uint8)  # 👈 cast to uint8
            # y_m_pred_val_show=y_m_pred_val_show.data.cpu().numpy()
            y_m_pred_val_show = nib.Nifti1Image(y_m_pred_val_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_deformed_Move_mask.nii.gz'
            print('saved to:', pred_sv_name)
            if seg_iter_val % 7==0 and save_flag:
                nib.save(y_m_pred_val_show, pred_sv_name)  

            y_gtv_pred_val_show=y_gtv_pred_val_show.data.cpu().numpy()
            y_gtv_pred_val_show = nib.Nifti1Image(y_gtv_pred_val_show,np.eye(4))    
            pred_sv_name=sv_folder+p_name+'_deformed_Move_gtv.nii.gz'
            print('saved to:', pred_sv_name)
            if seg_iter_val % 7==0 and save_flag:
                nib.save(y_gtv_pred_val_show, pred_sv_name)  