import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from transformers import XLNetPreTrainedModel, XLNetModel

class XLNetForMultiLabelClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.xlnet = XLNetModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()
        

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.xlnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        print("outputs \n\n\n")
        print(outputs)

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(outputs[0])
        
        print("logits")
        print(logits)
        print("logits.shape: ", logits.shape)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        print("outputs")
        print(outputs)
        # print("outputs.shape: ", outputs.shape)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs
        
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        print("logits:", logits)
        print("logits.shape")
        print(logits.shape)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
