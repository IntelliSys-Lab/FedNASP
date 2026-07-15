import os
from transformers import BertTokenizer, BertConfig
def get_tokenizer(args):
    tokenizer_class = BertTokenizer
    # VilBERT is the only supported backbone; Oscar/Prevalent tokenizers were
    # model-specific alternatives and are intentionally excluded.
    return tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=True)

def get_vlnbert_models(args, config=None):
    
    from vlnbert.vlnbert_CA import VLNBert
    from vlnbert.vlnbert_CA import BertConfig

    model_name_or_path = args.init_bert_file
    vis_config = BertConfig.from_json_file(os.path.join(
        'datasets/vln-bert', 'bert_base_6_layer_6_connect.json'))
    vis_config.img_feature_dim = 2048 + args.angle_feat_size
    vis_config.img_feature_type = args.features
    vis_config.layer_norm_eps = 1e-12
    vis_config.hidden_dropout_prob = 0.3
    vis_config.v_hidden_dropout_prob = 0.3
    if model_name_or_path:
        return VLNBert.from_pretrained(model_name_or_path, config=vis_config)
    return VLNBert(vis_config)
