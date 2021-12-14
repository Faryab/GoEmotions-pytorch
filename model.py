import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers import XLNetPreTrainedModel, XLNetModel
from transformers import RobertaModel
import torch

class XLNetForMultiLabelClassification(XLNetPreTrainedModel):

    def __init__(self, config):
        super(XLNetForMultiLabelClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.xlnet = XLNetModel(config)
        # self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # hidden size is 786
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()
        nn.init.xavier_normal_(self.classifier.weight)


    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        # add hidden states and attention if they are here
        outputs = (logits,) + last_hidden_state

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels),
                                 labels.view(-1, self.num_labels))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector 
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state


# class XLNetForMultiLabelClassification(XLNetPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.xlnet = XLNetModel(config)
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
#         self.loss_fct = nn.BCEWithLogitsLoss()
#         self.init_weights()

#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             labels=None,
#     ):
#         outputs = self.xlnet(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             # position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )

#         last_hidden_state = self.xlnet(input_ids=input_ids,
#                                        attention_mask=attention_mask,
#                                        token_type_ids=token_type_ids)

#         print("outputs \n\n\n")
#         print(outputs)

#         outputs = outputs[0]
#         # (batch_size, sequence_length, hidden_size)

#         # pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(outputs)

#         print("logits")
#         print(logits)
#         print("logits.shape: ", logits.shape)

#         # add hidden states and attention if they are here
#         outputs = (logits,) + outputs[2:]

#         print("outputs")
#         print(outputs)
#         # print("outputs.shape: ", outputs.shape)

#         if labels is not None:
#             loss = self.loss_fct(logits, labels)
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)

class RobertaForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = RobertaModel(config)
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
        print("outputs:")
        print(outputs)

        pooled_output = outputs[1]
        print("pooled_output")
        print(pooled_output)

        print("pooled_output.shape", pooled_output.shape)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        print("logits:", logits)
        print("logits.shape")
        print(logits.shape)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

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
        print("outputs:")
        print(outputs)

        pooled_output = outputs[1]
        print("pooled_output")
        print(pooled_output)

        print("pooled_output.shape", pooled_output.shape)

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        print("logits:", logits)
        print("logits.shape")
        print(logits.shape)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

