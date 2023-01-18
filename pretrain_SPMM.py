import torch
import torch.nn.functional as F
from torch import nn 
from xbert import BertConfig 
from transformers import BertTokenizer, BertForMaskedLM


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
        

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        
        self.itm_head_smiles = nn.Linear(smilesWidth, 2)
        self.itm_head_properties = nn.Linear(propertyWidth, 2)

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
        self.register_buffer("smiles_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.property_queue = nn.functional.normalize(self.property_queue, dim=0)
        self.smiles_queue = nn.functional.normalize(self.smiles_queue, dim=0)



    def forward(self, property, smilesIds, smilesAttentionMask, alpha=0):
        
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        #1. property tokenizing & embedding
        embedProperty = self.propertyEmbed(property.unsqueeze(2))
        
        property_MASK = self.property_MASK.expand(property.size(0), property.size(1), -1)
        halfMask = torch.bernoulli(torch.ones_like(property)*0.5)
        halfMaskBatch = halfMask.unsqueeze(2).repeat(1,1,property_MASK.size(2))
        maskedProperty = embedProperty* halfMaskBatch 
        inputProperty = torch.cat([self.property_CLS.expand(property.size(0), -1,-1), maskedProperty], dim=1)

        #2. input through encoders
        encProperty = self.propertyEncoder(inputs_embeds=inputProperty, return_dict=True).last_hidden_state
        propertyAtts = torch.ones(encProperty.size()[:-1], dtype=torch.long).to(inputProperty.device)
        propertyFeat = F.normalize(self.propertyProj(encProperty[:,0,:]), dim=-1)

        encSmiles = self.smilesEncoder.bert(smilesIds, attention_mask=smilesAttentionMask, return_dict=True).last_hidden_State
        smilesFeat = F.normalize(self.smilesProj(encSmiles[:,0,:]), dims=-1)
        
        #3. Contrastive Loss between the different & within the same modalities
        with torch.no_grad():
            self._momentum_update()
            
            encProperty_m = self.propertyEnoder_m(inputs_embeds=inputProperty, return_dic=True).last_hidden_state
            propertyAtts_m = torch.ones(encProperty_m.size()[:-1], dtype=torch.long).to(inputProperty.device)
            propertyFeat_m = F.normalize(self.propertyProj_m(encProperty_m[:,0,:]), dim=-1)
            propertyFeatAll = torch.cat([propertyFeat_m.t(), self.property_queue.clone().detach()], dim=1)
            
            encSmiles_m = self.smilesEncoder_m.bert(smilesIds,attention_mask=smilesAttentionMask, return_dict=True).last_hidden_state
            smilesFeat_m = F.normalize(self.smilesProj_m(encSmiles[:,0,:]), dim=-1)
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
        sim_p2p = propertyFeat @ propertyFeatAll / self.temp
        sim_s2s = smilesFeat @ smilesFeatAll / self.temp 

        loss_p2s = -torch.sum(F.log_softmax(sim_p2s, dim=1)*sim_p2s_targets, dim=1).mean()
        loss_s2p = -torch.sum(F.log_softmax(sim_s2p, dim=1)*sim_s2p_targets, dim=1).mean()
        loss_p2p = -torch.sum(F.log_softmax(sim_p2p, dim=1)*sim_p2p_targets, dim=1).mean()
        loss_s2s = -torch.sum(F.log_softmax(sim_p2p, dim=1)*sim_s2s_targets, dim=1).mean()

        loss_psc = (loss_p2s + loss_s2p + loss_p2p + loss_s2s)/2 

        self._dequeue_and_enqueue(propertyFeat_m, smilesFeat_m)

        #4. X-attention
        outputProperty = self.smilesEncoder

        #5. Next property prediction

        #6. Next word prediction

        #7. SMILES-property matching 
    
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