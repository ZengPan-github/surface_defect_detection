import os
import numpy as np
import cv2 as cv
from sklearn import metrics
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from main import instantiate_from_config
from random import shuffle, random
from taming.modules.losses.ssim import SSIM


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image"):
        super().__init__()

        self.first_stage_key = first_stage_key

        self.init_first_stage_from_ckpt(first_stage_config)

        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def forward(self, x, train=True):
        # one step to produce the logits
        z_indices, d, z_shape = self.encode_to_z(x)
        device = z_indices.device
        bsz, T = z_indices.shape
        if train:
            new_z = z_indices.clone().detach()
            cand = int(T * 0.25)
            pos = torch.zeros((bsz, cand), device=device, dtype=torch.int64)
            mask_target = torch.zeros((bsz, cand), device=device, dtype=torch.int64)
            mask = torch.ones((bsz, T),device=device,dtype=torch.bool)
            for _ in range(bsz):
                permute_idx = torch.randperm(T, device=device, dtype=torch.int64)
                samples = permute_idx[:cand]
                pos[_, :] = samples
                mask_target[_, :] = z_indices[_, :].gather(0, samples)
                for i in samples:
                    mask[_, i] = False
#                     new_z[_, i] = self.transformer.config.vocab_size
                    if random() <= 0.8:
                        new_z[_, i] = self.transformer.config.vocab_size
                    elif random() <= 0.5:
                        new_z[_, i] = torch.randint(self.transformer.config.vocab_size, (1,), device=device)
            
            logit_score, loss = self.transformer(new_z, pos, mask_target)  
            return logit_score, loss, z_indices,z_shape, mask
        else:
            d = d.reshape(bsz, -1)
            mask = torch.where(d < 0.6, False, True) 
            new_z = torch.where(mask,self.transformer.config.vocab_size, z_indices)
            logit_score, _ =  self.transformer(new_z, None, None)
            
            return logit_score, None,z_indices, z_shape, mask

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, train, temperature=0.8, top_k=16):
        assert not self.transformer.training
        bsz = x.size(0)
        logits, _, z_index,z_shape, pos_mask = self(x, train)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
        
        sample_enc = probs.view(-1, self.transformer.config.vocab_size).multinomial(1).view(x.size(0),-1)
#         greedy_enc = probs.topk(1)[1].squeeze()
        new_z = torch.where(pos_mask, z_index, sample_enc)
            
        return z_index, new_z, z_shape
        
#         shp = probs.size()
#         probs = probs.view(-1,probs.size(-1))
#         # sample from the distribution or take the most likely
#         if sample:
#             ix = torch.multinomial(probs, num_samples=1)
#         else:
#             _, ix = torch.topk(probs, k=1, dim=-1)
        
#         # append to the sequence and continue
#         ix = ix.view(shp[:-1])
#         new_x = (1-mask)*new_x + mask*ix

#         return new_x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info, d = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return indices, d, quant_z.shape

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch,temperature=0.8, top_k=16, train=True, **kwargs):
        log = dict()

        N = 4
        
        x = self.get_input(self.first_stage_key, batch)
        x = x[:N].to(device=self.device)
        
        rec = []
        for _ in range(8):
            z_indices, new_z, z_shape = self.sample(x,train,
                                                    temperature=temperature if temperature is not None else 1.0, top_k=top_k if top_k is not None else 100)

            x_sample = self.decode_to_img(new_z, z_shape)
            rec.append(x_sample)
        
        ensemble = torch.stack(rec).mean(0)
        
            
        # reconstruction
        x_rec = self.decode_to_img(z_indices, z_shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["topk_sample"] = ensemble


        return log

                
    def get_input(self, key, batch):
        x = batch[key]
#         if len(x.shape) == 3:
#             x = x[..., None]
#         if len(x.shape) == 4:
#             x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

#     def get_xc(self, batch, N=None):
#         x = self.get_input(self.first_stage_key, batch)
#         if N is not None:
#             x = x[:N]
#         return x

#     def shared_step(self, batch, batch_idx):
#         x = self.get_xc(batch)
#         logit, loss, mask_pos = self(x) 
#         return logit, loss, mask_pos

    def training_step(self, batch, batch_idx):
        x = self.get_input(self.first_stage_key, batch)
        
        _, loss,_,_,_= self(x,train=True)
        self.log("train/bert_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(self.first_stage_key, batch)
        pass
                
    def test_step(self, batch, batch_idx):
        x = self.get_input(self.first_stage_key, batch)
        device = x.device
        img_label = batch['label']
        name = batch['name']
        mask = batch['mask']
        rec_imgs = []
        for _ in range(8):
            z_index, new_z, z_shape = self.sample(x,train=False, temperature=0.8, top_k=16)
            new_x = self.decode_to_img(new_z, z_shape)
            rec_imgs.append(new_x)
        
        ensemble = torch.stack(rec_imgs).mean(0)
        
        x.add_(1.0).mul_(0.5)
        ensemble.add_(1.0).mul_(0.5)
        
        ssim = SSIM(win_size=5,sigma=0.25).to(device)
        
        def helper(x,y):
            maps = ssim(x,y,data_range=1.0,use_pad=True,return_full=True)
            maps = maps.squeeze().cpu().numpy()
            maps = 255 - np.uint8(maps*255)
            thres, res_map = cv.threshold(maps, 0, 255, cv.THRESH_OTSU)  #cv.THRESH_TRIANGLE, cv.THRESH_OTSU
        
            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(res_map, connectivity=8)
            area_list = [stats[i][-1] for i in range(num_labels)]
            max_area = max(area_list[1:])
            return thres, max_area, res_map, labels, stats
        
        def dice_coef(pred, target, smooth=0.01):
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()

            return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
       
        thres, max_area, res_map, labels, stats = helper(x,ensemble)
            
        image_filtered = np.zeros_like(res_map)
        for (i, label) in enumerate(np.unique(labels)):
            # 如果是背景，忽略
            if label == 0:
                continue
            if stats[i][-1] >= 137:
                image_filtered[labels == i] = 255
                
        
        seg = torch.from_numpy(image_filtered[None,None,:]).div(255.).to(device)
        dice = dice_coef(seg, mask).item()
#         res_map = torch.from_numpy(res_map[None,None,:]).div(255.).to(device)
#         grid = torch.cat((x,ensemble,mask,res_map,seg))
#         rec_path = f'rec_imgs/{self.first_stage_model.classname}_transformer'
#         os.makedirs(rec_path,exist_ok=True)
#         torchvision.utils.save_image(grid, os.path.join(rec_path, name[0]), nrow=5)

        return {'label':img_label,'max_area':max_area,'thres':thres,'dice': dice}
        
        
    def test_epoch_end(self, outputs):
        labels = np.array([x['label'].cpu().numpy() for x in outputs]).ravel()
        thres = np.array([x['thres'] for x in outputs])
        max_area = np.array([x['max_area'] for x in outputs])
        dice = np.array([x['dice'] for x in outputs]).mean()
        print(f'dice: {dice}')

        
        def cal_metrics(y,y_pred):
            auroc = metrics.roc_auc_score(y, y_pred)
            precisions, recalls, thresholds = metrics.precision_recall_curve(y, y_pred)
            F1_scores = np.divide(2 * precisions * recalls,precisions + recalls,
                              out=np.zeros_like(precisions),where=(precisions + recalls) != 0)
            opt_idx = np.argmax(F1_scores)
            opt_thre = thresholds[opt_idx]
            f1 = F1_scores[opt_idx]
            pred = (y_pred >= opt_thre).astype(int)
            acc = np.sum(pred == y)/len(y)
            recall = recalls[opt_idx]
            precision = precisions[opt_idx]
            
            return {'auroc':auroc,
                    'opt_thre':opt_thre,
                    'f1':f1,
                    'acc':acc,
                    'recall':recall,
                    'precision':precision,
                    }
        
        
        
        thre_metrics = cal_metrics(labels, thres)
        area_metrics = cal_metrics(labels, max_area)
        
        print(f'ostu threshold : {thre_metrics}')
        print(f'defect area : {area_metrics}')

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
