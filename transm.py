from transformers import AutoConfig, AutoTokenizer, TFAutoModel
import numpy as np
import tensorflow as tf

class albert():
      def __init__(self):
        # AutoTokenizer.local_files_only = False
        # TFAutoModel.local_files_only = False
        # AutoConfig.local_files_only = False
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert = TFAutoModel.from_pretrained("bert-base-uncased")
        input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')
        y = bert(input_ids, attention_mask=mask)[1]
        self.model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

      def return_emmbeding(self,text):
          tokens = self.tokenizer.encode_plus(text, max_length=512,
                                        truncation=True, padding='max_length',
                                        add_special_tokens=True, return_token_type_ids=False,
                                        return_tensors='tf')
          # tokenizer returns int32 tensors, we need to return float64, so we use tf.cast
          in_tensor= {'input_ids': tf.cast(tokens['input_ids'], tf.float64),'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
          return  self.model.predict(in_tensor)[0]

