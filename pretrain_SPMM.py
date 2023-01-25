import torch
import torch.nn.functional as F
from torch import nn 
from xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer


class SPMM(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = BertTokenizer('./vocab_bpe_300.txt', do_lower_case=False,do_basic_tokenize=False)
        embed_dim = config['embed_dim']

        smilesAndFusion_config = BertConfig.from_json_file('./config_bert_smiles_and_fusion_encoder.json')
        property_config = BertConfig.from_json_file('./config_bert_property_encoder.json')
        self.smilesEncoder = BertForMaskedLM(config = smilesAndFusion_config)
        self.propertyEncoder = BertForMaskedLM(config = property_config)

        smilesWidth = self.smilesEncoder.config.hidden_size
        propertyWidth = config['property_width']

        self.smilesProj = nn.Linear(smilesWidth, embed_dim)
        self.propertyProj = nn.Linear(propertyWidth, embed_dim)

        # special tokens & embedding for property input
        self.propertyEmbed = nn.Linear(1, propertyWidth)
        self.property_CLS = nn.Parameter(torch.zeros([1, 1, propertyWidth]))
        self.property_MASK = nn.Parameter(torch.zeros([1, 1, propertyWidth]))
        
        self.target_reg = nn.Sequential(nn.Linear(propertyWidth, propertyWidth),
                                        nn.GELU(),
                                        nn.LayerNorm(propertyWidth, property_config.layer_norm_eps),
                                        nn.Linear(propertyWidth, 1))

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        
        self.psm_head = nn.Linear(propertyWidth + smilesWidth, 2)

        # Momentum Model

        self.smilesEncoder_m = BertForMaskedLM(config = smilesAndFusion_config)
        self.propertyEncoder_m = BertForMaskedLM(config = property_config)
        self.smilesProj_m = nn.Linear(smilesWidth, embed_dim)
        self.propertyProj_m = nn.Linear(propertyWidth, embed_dim)

        self.model_pairs = [[self.smilesEncoder, self.smilesEncoder_m],
                            [self.smilesProj, self.smilesProj_m],
                            [self.propertyEncoder, self.propertyEncoder_m],
                            [self.propertyProj, self.propertyProj_m]]
        
        self.copy_params()

        # Create the queue
        self.register_buffer("property_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("smiles_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.property_queue = nn.functional.normalize(self.property_queue, dim=0)
        self.smiles_queue = nn.functional.normalize(self.smiles_queue, dim=0)



    def forward(self, property, smilesIds, smilesAttentionMask, alpha=0):
        
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        #1. property tokenizing & embedding
        embedProperty = self.propertyEmbed(property.unsqueeze(2))
        
        property_MASK = self.property_MASK.expand(property.size(0), property.size(1), -1)
        halfMask = torch.bernoulli(torch.full(property.shape, 0.5))
        halfMaskBatch = halfMask.unsqueeze(2).repeat(1,1,property_MASK.size(2))
        maskedProperty = embedProperty* halfMaskBatch 
        inputProperty = torch.cat([self.property_CLS.expand(property.size(0), -1,-1), maskedProperty], dim=1)

        #2. input through encoders

        encProperty = self.propertyEncoder.bert(inputs_embeds=inputProperty, return_dict=True).last_hidden_state
        propertyAtts = torch.ones(encProperty.size()[:-1], dtype=torch.long).to(inputProperty.device)
        propertyFeat = F.normalize(self.propertyProj(encProperty[:,0,:]), dim=-1)

        encSmiles = self.smilesEncoder.bert(smilesIds, attention_mask=smilesAttentionMask, return_dict=True, mode='text').last_hidden_State
        smilesFeat = F.normalize(self.smilesProj(encSmiles[:,0,:]), dims=-1)
        
        #3. Contrastive Loss between the different & within the same modalities
        with torch.no_grad():
            self._momentum_update()
            
            encProperty_m = self.propertyEncoder_m(inputs_embeds=inputProperty, return_dic=True).last_hidden_state
            #propertyAtts_m = torch.ones(encProperty_m.size()[:-1], dtype=torch.long).to(inputProperty.device)
            propertyFeat_m = F.normalize(self.propertyProj_m(encProperty_m[:,0,:]), dim=-1)
            propertyFeatAll = torch.cat([propertyFeat_m.t(), self.property_queue.clone().detach()], dim=1)
            
            encSmiles_m = self.smilesEncoder_m.bert(smilesIds,attention_mask=smilesAttentionMask, return_dict=True, mode='text').last_hidden_state
            smilesFeat_m = F.normalize(self.smilesProj_m(encSmiles_m[:,0,:]), dim=-1)
            smilesFeatAll = torch.cat([smilesFeat_m.t(), self.smiles_queue.clone().detach()], dim=1)

            sim_p2s_m = propertyFeat_m @ smilesFeatAll / self.temp
            sim_s2p_m = smilesFeat_m @ propertyFeatAll / self.temp
            sim_p2p_m = propertyFeat_m @ propertyFeatAll / self.temp
            sim_s2s_m = smilesFeat_m @ smilesFeatAll / self.temp

            ## Make Target ##
            sim_targets_diff = torch.zeros(sim_p2s_m.size()).to(property.device)
            sim_targets_diff.fill_diagonal_(1)
            sim_targets_same = torch.zeros(sim_p2p_m.size()).to(property.device)
            sim_targets_same.fill_diagonal_(1)
          
            sim_p2s_targets = alpha * F.softmax(sim_p2s_m, dim=1) + (1-alpha) * sim_targets_diff
            sim_s2p_targets = alpha * F.softmax(sim_s2p_m, dim=1) + (1-alpha) * sim_targets_diff
            sim_p2p_targets = alpha * F.softmax(sim_p2p_m, dim=1) + (1-alpha) * sim_targets_same
            sim_s2s_targets = alpha * F.softmax(sim_s2s_m, dim=1) + (1-alpha) * sim_targets_same

        sim_p2s = propertyFeat @ smilesFeatAll / self.temp
        sim_s2p = smilesFeat @ propertyFeatAll / self.temp
        sim_p2p = propertyFeat @ propertyFeatAll / self.temps
        sim_s2s = smilesFeat @ smilesFeatAll / self.temp 

        loss_p2s = -torch.sum(F.log_softmax(sim_p2s, dim=1)*sim_p2s_targets, dim=1).mean()
        loss_s2p = -torch.sum(F.log_softmax(sim_s2p, dim=1)*sim_s2p_targets, dim=1).mean()
        loss_p2p = -torch.sum(F.log_softmax(sim_p2p, dim=1)*sim_p2p_targets, dim=1).mean()
        loss_s2s = -torch.sum(F.log_softmax(sim_s2s, dim=1)*sim_s2s_targets, dim=1).mean()

        loss_psc = (loss_p2s + loss_s2p + loss_p2p + loss_s2s)/2 # property - smiles contrastive loss

        self._dequeue_and_enqueue(propertyFeat_m, smilesFeat_m)

        #4. X-attention
        outputProperty_pos = self.smilesEncoder.bert(encoder_embeds = encProperty,
                                                 attention_mask = propertyAtts,
                                                 encoder_hidden_states = encSmiles,
                                                 encoder_attention_mask = smilesAttentionMask,
                                                 return_dict = True,
                                                 mode = 'fusion'
                                                 ).last_hidden_state[:,0,:] 
        outputSmiles_pos = self.smilesEncoder.bert(encoder_embeds = encSmiles,
                                               attention_mask = smilesAttentionMask,
                                               encoder_hidden_states = encProperty,
                                               encoder_attention_mask = propertyAtts,
                                               return_dict = True,
                                               mode = 'fusion'
                                               ).last_hidden_state[:,0,:]

        pos_embeds = torch.cat([outputProperty_pos, outputSmiles_pos], dim=-1)

        #outputProperty_pos = outputProperty.last_hidden_state[:,0,:]
        #outputSmiles_pos = outputSmiles.last_hidden_state[:,0,:]

        #5. SMILES - Property Matching
        with torch.no_grad():
            batch_size = property.size(0)
            weights_p2s = F.softmax(sim_p2s[:,:batch_size], dim=1)
            weights_s2p = F.softmax(sim_s2p[:,:batch_size], dim=1)

            weights_p2s.fill_diagonal_(0)
            weights_s2p.fill_diagonal_(0)
        
        encProperty_neg = []
        
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_p2s[b], 1).item()
            encProperty_neg.append(encProperty[neg_idx])
        encProperty_neg = torch.stack(encProperty_neg, dim=0)
        
        encSmiles_neg = []
        smilesAtts_neg = []

        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_s2p[b], 1).item()
            encSmiles_neg.append(encSmiles[neg_idx])
            smilesAtts_neg.append(smilesAttentionMask[neg_idx])
        encSmiles_neg = torch.stack(encSmiles_neg, dim=0)
        smilesAtts_neg = torch.stack(smilesAtts_neg, dim=0)

        encProperty_all = torch.cat([encProperty, encProperty_neg], dim=0)
        propertyAtts_all = torch.cat([propertyAtts, propertyAtts], dim=0)

        encSmiles_all = torch.cat([encSmiles_neg, encSmiles], dim=0)
        smilesAtts_all = torch.cat([smilesAtts_neg, smilesAttentionMask], dim=0)

        outputProperty_neg = self.smilesEncoder.bert(encoder_embeds = encProperty_all,
                                                     attention_mask = propertyAtts_all,
                                                     encoder_hidden_states = encSmiles_all,
                                                     encoder_attention_mask = smilesAtts_all,
                                                     return_dict = True,
                                                     mode = 'fusion'
                                                     ).last_hidden_state[:,0,:]
        outputSmiles_neg = self.smilesEncoder.bert(encoder_embeds = encSmiles_all,
                                                   attention_mask = smilesAtts_all,
                                                   encoder_hidden_states = encProperty_all,
                                                   encoder_attention_mask = encProperty_all,
                                                   return_dict = True,
                                                   mode = 'fusion'
                                                   ).last_hidden_state[:,0,:]
        
        
        neg_embeds = torch.cat([outputProperty_neg, outputSmiles_neg], dim=-1)

        ps_embeddings = torch.cat([pos_embeds, neg_embeds], dim=0)
        ps_output = self.psm_head(ps_embeddings)

        psm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size*2, dtype=torch.long)], 
                               dim=0
                               ).to(property.device)
        
        loss_psm = F.cross_entropy(ps_output, psm_labels)  #property-smiles matching loss 

        #6. Next smiles prediction
        
        inputs_ids = smilesIds.clone()
        labels = inputs_ids.clone()[:,1:]

        with torch.no_grad():
            logits_m = self.smilesEncoder_m(inputs_ids, 
                                            attention_mask = smilesAttentionMask,
                                            encoder_hidden_states = encProperty_m,
                                            encoder_attention_mask = propertyAtts,
                                            return_dict = True,
                                            is_decoder = True,
                                            return_logits = True
                                            )[:,:-1,:]
        
        nsp_output = self.smilesEncoder(inputs_ids,
                                        attention_mask = smilesAttentionMask,
                                        encoder_hidden_states = encProperty,
                                        encoder_attention_mask = propertyAtts,
                                        return_dict = True,
                                        is_decoder =True,
                                        return_logits = True
                                        )[:,:-1,:]
        
        per_nsp_output= nsp_output.permute((0,2,1))

        loss_CE = nn.CrossEntropyLoss
        loss_onehot = loss_CE(per_nsp_output, labels)
        
        soft_labels = F.softmax(logits_m, dim=1)
        loss_distill = -torch.sum(F.log_softmax(nsp_output, dim=1) * soft_labels, dim=-1)
        loss_distill = (loss_distill * (labels != 0)).mean()

        loss_nsp = (1-alpha)*loss_onehot + alpha*loss_distill


        #7. Next property prediction

        propertyTargets = property.clone()

        with torch.no_grad():
            encProperty_masked_m = self.propertyEncoder_m.bert(encoder_embeds = inputProperty,
                                                               return_dict = True,
                                                               is_decoder = True
                                                               ).last_hidden_state
            
            npp_output_m = self.smilesEncoder_m.bert(encoder_embeds = encProperty_masked_m,
                                                     attention_mask = propertyAtts,
                                                     encoder_hidden_states = encSmiles_m,
                                                     encoder_attention_mask = smilesAttentionMask,
                                                     return_dict = True,
                                                     is_decoder = True,
                                                     mode = 'fusion'
                                                     ).last_hidden_state[:,:-1,:]
            
            pred_m = self.target_reg(npp_output_m).squeeze(2)
        
        encProperty_masked = self.propertyEncoder.bert(encoder = inputProperty,
                                                       return_dict = True,
                                                       is_decoder = True
                                                       ).last_hidden_state
        
        npp_output = self.smilesEncoder.bert(encode_embeds = encProperty_masked,
                                             attention_mask = propertyAtts,
                                             encoder_hidden_states = encSmiles,
                                             encoder_attention_mask = smilesAttentionMask,
                                             return_dict = True,
                                             is_decoder = True,
                                             mode = 'fusion'
                                             ).last_hidden_state[:,:-1,:]
        
        pred = self.target_reg(npp_output).squeeze(2)

        loss_MSE = nn.MSELoss()
        loss_npp = loss_MSE(pred*halfMask, propertyTargets*halfMask)

        return loss_psc, loss_psm, loss_nsp, loss_npp
        
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameter(), model_pair[1].parameter()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, property_feat, smiles_feat):
        property_feats = concat_all_gather(property_feat)
        smiles_feats = concat_all_gather(smiles_feat)

        batch_size = property_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.property_queue[:, ptr:ptr + batch_size] = property_feats.T
        self.smiles_queue[:, ptr:ptr + batch_size] = smiles_feats.T
        ptr = (ptr + batch_size) & self.queue_size 

        self.queue_ptr[0] = ptr

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output